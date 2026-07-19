import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./config.js";
import {
    insertSummary,
    getSummariesByTier,
    getSummariesByTierAndSession,
    getSummariesByTierExcludingSession,
    getSummaryById,
    updateSummaryContent,
    deleteSummary,
    getSummaryCountByTier,
    getTotalSummaryTokens,
    withDatabaseTransaction,
    type SummaryRow,
} from "./database.js";
import {
    storeFact,
    queryEntity,
    ensureEntity,
    getAllFacts,
    getGraphStats,
    removeEntity,
    serializeFacts,
} from "./knowledge-graph.js";
import {
    prepareVector,
    persistPreparedVector,
    removeVectorsBySource,
    getVectorStoreCount,
    type PreparedVector,
} from "./vector-store.js";
import { countTokens, truncateToTokenBudget } from "./token-counter.js";
import { getCurrentSessionIdOrNull } from "./session.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MemoryType = "fact" | "preference" | "event" | "summary" | "core";
export type CompressScope = "working" | "session" | "epoch";
export type ForgetAction = "deprecate" | "correct" | "delete";

export interface StoreResult {
    memoryId: string;
    memoryType: MemoryType;
    tier: number;
    entitiesCreated: string[];
    factsStored: number;
    vectorId: string | null;
    sessionId: string | null;
}

export interface MemoryStatus {
    tiers: {
        tier0: { count: number; tokenEstimate: number };
        tier1: { count: number; tokenEstimate: number };
        tier2: { count: number; tokenEstimate: number };
        tier3: { count: number; tokenEstimate: number };
    };
    knowledgeGraph: { entities: number; relations: number };
    vectorStore: { count: number };
    totalTokensStored: number;
    currentSessionId: string | null;
}

// ---------------------------------------------------------------------------
// Working memory (Tier 0) — in-memory ring buffer of recent turns
// ---------------------------------------------------------------------------

export interface WorkingMemoryEntry {
    id: string;
    content: string;
    tokens: number;
    timestamp: string;
    sessionId: string | null;
    memoryType: "event";
    confidence: number;
}

const _workingMemory: WorkingMemoryEntry[] = [];

function getWorkingMemoryTokens(): number {
    return _workingMemory.reduce((sum, entry) => sum + entry.tokens, 0);
}

function getSessionWorkingMemoryTokens(sessionId: string | null): number {
    if (!sessionId) return getWorkingMemoryTokens();
    return _workingMemory
        .filter((e) => e.sessionId === sessionId)
        .reduce((sum, entry) => sum + entry.tokens, 0);
}

function addToWorkingMemory(id: string, content: string, confidence: number, sessionId: string | null): void {
    _workingMemory.push({
        id,
        content,
        tokens: countTokens(content),
        timestamp: new Date().toISOString(),
        sessionId,
        memoryType: "event",
        confidence,
    });
}

/** Persist related summary/vector rows in the same durable SQLite checkpoint. */
function persistSummaryAndVector(
    summary: Omit<SummaryRow, "created_at" | "updated_at">,
    vector: PreparedVector
): string {
    withDatabaseTransaction(() => {
        insertSummary(summary);
        persistPreparedVector(vector);
    });
    return vector.id;
}

function parseSummaryMetadata(metadata: string): Record<string, unknown> {
    try {
        const parsed: unknown = JSON.parse(metadata);
        return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    } catch {
        return {};
    }
}

function summaryVectorType(summary: SummaryRow): string {
    if (summary.tier === 0) return "event";
    if (summary.tier === 3) return "core";
    const type = parseSummaryMetadata(summary.metadata).type;
    return type === "fact" || type === "preference" || type === "event" || type === "core"
        ? type
        : "summary";
}

function summaryVectorConfidence(summary: SummaryRow): number {
    const confidence = parseSummaryMetadata(summary.metadata).confidence;
    return typeof confidence === "number" && confidence >= 0 && confidence <= 1 ? confidence : 0.9;
}

function summaryVectorMetadata(summary: SummaryRow, confidence: number): Record<string, unknown> {
    const metadata = parseSummaryMetadata(summary.metadata);
    return {
        memoryType: summaryVectorType(summary),
        entities: Array.isArray(metadata.entities) ? metadata.entities : [],
        sessionId: summary.session_id,
        confidence,
    };
}

/**
 * Get the current session's working memory as text.
 */
export function getWorkingMemory(): string {
    return getCurrentSessionWorkingMemoryEntries().map((entry) => entry.content).join("\n");
}

/**
 * Return individual Tier 0 entries for the active session only. Returning an
 * empty list without a session is deliberate: unscoped working memory must
 * never become retrievable after a session has ended.
 */
export function getCurrentSessionWorkingMemoryEntries(): WorkingMemoryEntry[] {
    const sessionId = getCurrentSessionIdOrNull();
    if (!sessionId) return [];
    return _workingMemory
        .filter((entry) => entry.sessionId === sessionId)
        .map((entry) => ({ ...entry }));
}

/**
 * Get ALL working memory as text (regardless of session).
 */
export function getAllWorkingMemory(): string {
    if (_workingMemory.length === 0) return "";
    return _workingMemory.map((e) => e.content).join("\n");
}

/**
 * Get working memory entries count for the current session.
 */
export function getWorkingMemoryCount(): number {
    return getCurrentSessionWorkingMemoryEntries().length;
}

/**
 * Clear ALL working memory. Called on session start to guarantee
 * complete session isolation — no residual data from previous sessions.
 */
export function clearWorkingMemory(): void {
    _workingMemory.length = 0;
}
/**
 * Archive all current session working memory into a Tier 1 summary.
 * Called during session transitions to preserve data before clearing.
 * Returns the archive summary text, or null if nothing to archive.
 */
export async function archiveWorkingMemory(sessionId: string): Promise<string | null> {
    const config = getConfig();

    // Tier 0 rows are the canonical checkpoint. The in-memory array is only
    // a retrieval cache and may be empty after a process recovery.
    const sessionEntries = getSummariesByTierAndSession(0, sessionId);
    if (sessionEntries.length === 0) return null;

    const combinedContent = sessionEntries.map((entry) => entry.content).join("\n");
    const originalTokens = sessionEntries.reduce((sum, entry) => sum + entry.token_count, 0);

    const { text: compressed } = truncateToTokenBudget(
        combinedContent,
        config.tokenBudgets.tier1Session
    );

    const summaryId = uuidv4();
    const compressedTokens = countTokens(compressed);

    // Embedding is deliberately outside the transaction; all related rows
    // are then committed together, so a failed archive leaves Tier 0 intact.
    const archiveVector = await prepareVector(compressed, summaryId, "summary", 0.9, {
        sessionArchive: true,
        sessionId,
    });
    withDatabaseTransaction(() => {
        insertSummary({
            id: summaryId,
            tier: 1,
            content: compressed,
            token_count: compressedTokens,
            session_id: sessionId,
            source_ids: JSON.stringify(sessionEntries.map((entry) => entry.id)),
            metadata: JSON.stringify({
                type: "session_archive",
                originalCount: sessionEntries.length,
                originalTokens,
                confidence: 0.9,
                sessionId,
            }),
        });
        persistPreparedVector(archiveVector);
        for (const entry of sessionEntries) {
            removeVectorsBySource(entry.id);
            deleteSummary(entry.id);
        }
    });

    const archiveIds = new Set(sessionEntries.map((entry) => entry.id));
    const remaining = _workingMemory.filter((entry) => !archiveIds.has(entry.id));
    _workingMemory.length = 0;
    _workingMemory.push(...remaining);

    return `Archived ${sessionEntries.length} entries (${originalTokens} tokens) → Tier 1 summary (${compressedTokens} tokens)`;
}

/**
 * Clear all working memory entries for the current session.
 */
export function clearSessionWorkingMemory(sessionId: string): void {
    const remaining = _workingMemory.filter((e) => e.sessionId !== sessionId);
    _workingMemory.length = 0;
    _workingMemory.push(...remaining);
}

/**
 * Clear ALL working memory regardless of session.
 */
export function clearAllWorkingMemory(): void {
    _workingMemory.length = 0;
}

// ---------------------------------------------------------------------------
// Core memory operations
// ---------------------------------------------------------------------------

/**
 * Store information into the appropriate memory subsystem.
 * Automatically categorizes, indexes, and embeds content.
 */
export async function storeMemory(
    content: string,
    memoryType: MemoryType,
    confidence: number = 1.0,
    entities: string[] = []
): Promise<StoreResult> {
    const config = getConfig();
    const sessionId = getCurrentSessionIdOrNull();
    const result: StoreResult = {
        memoryId: "",
        memoryType,
        tier: 0,
        entitiesCreated: [],
        factsStored: 0,
        vectorId: null,
        sessionId,
    };

    switch (memoryType) {
        case "core": {
            const summaryId = uuidv4();
            const tokens = countTokens(content);
            const vector = await prepareVector(content, summaryId, "core", confidence, {
                memoryType: "core", entities, sessionId,
            });
            result.vectorId = persistSummaryAndVector({
                id: summaryId,
                tier: 3,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: "[]",
                metadata: JSON.stringify({ type: "core", entities, confidence, sessionId }),
            }, vector);
            result.memoryId = summaryId;
            result.tier = 3;
            break;
        }

        case "fact": {
            const factId = uuidv4();
            const tokens = countTokens(content);
            const vector = await prepareVector(content, factId, "fact", confidence, {
                memoryType: "fact", entities, sessionId,
            });
            withDatabaseTransaction(() => {
                for (const entityLabel of entities) {
                    ensureEntity(entityLabel, "unknown", {}, confidence);
                }
                if (entities.length >= 2) {
                    const predicate = inferPredicate(content, entities);
                    for (let i = 1; i < entities.length; i++) {
                        storeFact(entities[0], predicate, entities[i], "unknown", "unknown", confidence);
                    }
                }
                insertSummary({
                    id: factId,
                    tier: 1,
                    content,
                    token_count: tokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(entities),
                    metadata: JSON.stringify({ type: "fact", entities, confidence, sessionId }),
                });
                result.vectorId = persistPreparedVector(vector);
            });
            result.entitiesCreated = [...entities];
            result.factsStored = Math.max(entities.length - 1, 0);
            result.memoryId = factId;
            result.tier = 1;
            break;
        }

        case "preference": {
            const prefId = uuidv4();
            const tokens = countTokens(content);
            const vector = await prepareVector(content, prefId, "preference", confidence, {
                memoryType: "preference", entities, sessionId,
            });
            withDatabaseTransaction(() => {
                ensureEntity("User", "person", {}, 1.0);
                for (const entityLabel of entities) {
                    ensureEntity(entityLabel, "unknown", {}, confidence);
                    storeFact("User", "prefers", entityLabel, "person", "unknown", confidence);
                }
                insertSummary({
                    id: prefId,
                    tier: 2,
                    content,
                    token_count: tokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(entities),
                    metadata: JSON.stringify({ type: "preference", entities, confidence, sessionId }),
                });
                result.vectorId = persistPreparedVector(vector);
            });
            result.entitiesCreated = ["User", ...entities];
            result.factsStored = entities.length;
            result.memoryId = prefId;
            result.tier = 2;
            break;
        }

        case "event": {
            // Tier 0's in-memory view is updated only after its summary and
            // vector checkpoint commit, so cache and database never diverge.
            const eventId = uuidv4();
            const tokens = countTokens(content);
            const eventVector = await prepareVector(content, eventId, "event", confidence, {
                memoryType: "event", entities, timestamp: new Date().toISOString(), sessionId,
            });
            withDatabaseTransaction(() => {
                insertSummary({
                    id: eventId,
                    tier: 0,
                    content,
                    token_count: tokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(entities),
                    metadata: JSON.stringify({ type: "event", entities, confidence, sessionId }),
                });
                for (const entityLabel of entities) {
                    ensureEntity(entityLabel, "unknown", {}, confidence);
                }
                result.vectorId = persistPreparedVector(eventVector);
            });
            addToWorkingMemory(eventId, content, confidence, sessionId);
            result.entitiesCreated = [...entities];
            result.memoryId = eventId;
            result.tier = 0;

            await checkTier0Overflow();
            break;
        }

        case "summary": {
            const sumId = uuidv4();
            const tokens = countTokens(content);
            const vector = await prepareVector(content, sumId, "summary", confidence, {
                memoryType: "summary", entities, sessionId,
            });
            result.vectorId = persistSummaryAndVector({
                id: sumId,
                tier: 1,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: JSON.stringify(entities),
                metadata: JSON.stringify({ type: "summary", entities, confidence, sessionId }),
            }, vector);
            result.memoryId = sumId;
            result.tier = 1;
            break;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/**
 * Check if Tier 0 working memory has overflowed its token budget.
 * If so, compress the oldest entries into a Tier 1 summary.
 */
async function checkTier0Overflow(): Promise<void> {
    const config = getConfig();
    const sessionId = getCurrentSessionIdOrNull();
    const currentTokens = sessionId
        ? getSessionWorkingMemoryTokens(sessionId)
        : getWorkingMemoryTokens();

    if (currentTokens <= config.compression.tier0OverflowThreshold) return;

    // Get entries for the current session
    const sessionEntries = sessionId
        ? _workingMemory.filter((e) => e.sessionId === sessionId)
        : _workingMemory;

    // Compress the oldest half of working memory into a Tier 1 summary
    const halfIdx = Math.floor(sessionEntries.length / 2);
    const toCompress = sessionEntries.slice(0, halfIdx);

    if (toCompress.length === 0) return;

    const combinedContent = toCompress.map((e) => e.content).join("\n");

    // Create a compressed Tier 1 summary
    const { text: compressed } = truncateToTokenBudget(
        combinedContent,
        config.tokenBudgets.tier1Session
    );

    const summaryId = uuidv4();
    const tokens = countTokens(compressed);

    const compressedVector = await prepareVector(compressed, summaryId, "summary", 0.9, {
        autoCompressed: true,
        sessionId,
    });
    withDatabaseTransaction(() => {
        insertSummary({
            id: summaryId,
            tier: 1,
            content: compressed,
            token_count: tokens,
            session_id: sessionId,
            source_ids: JSON.stringify(toCompress.map((entry) => entry.id)),
            metadata: JSON.stringify({
                type: "auto_compressed",
                originalCount: toCompress.length,
                originalTokens: toCompress.reduce((sum, entry) => sum + entry.tokens, 0),
                confidence: 0.9,
                sessionId,
            }),
        });
        persistPreparedVector(compressedVector);
        for (const entry of toCompress) {
            removeVectorsBySource(entry.id);
            deleteSummary(entry.id);
        }
    });
    const compressIds = new Set(toCompress.map((entry) => entry.id));
    const remaining = _workingMemory.filter((entry) => !compressIds.has(entry.id));
    _workingMemory.length = 0;
    _workingMemory.push(...remaining);
}

/**
 * Manually compress memory at the specified scope.
 * Returns a textual report of what was compressed.
 */
export async function compressMemory(scope: CompressScope): Promise<string> {
    const config = getConfig();
    const sessionId = getCurrentSessionIdOrNull();

    switch (scope) {
        case "working": {
            // Compress all of current session Tier 0 into a Tier 1 summary
            const sessionEntries = sessionId
                ? _workingMemory.filter((e) => e.sessionId === sessionId)
                : _workingMemory;

            if (sessionEntries.length === 0) {
                return "Working memory is empty, nothing to compress.";
            }

            const content = sessionEntries
                .map((e) => e.content)
                .join("\n");
            const originalTokens = sessionEntries.reduce((s, e) => s + e.tokens, 0);
            const originalCount = sessionEntries.length;

            const { text: compressed } = truncateToTokenBudget(
                content,
                config.tokenBudgets.tier1Session
            );

            const summaryId = uuidv4();
            const compressedTokens = countTokens(compressed);

            const compressedVector = await prepareVector(compressed, summaryId, "summary", 0.9, { sessionId });
            withDatabaseTransaction(() => {
                insertSummary({
                    id: summaryId,
                    tier: 1,
                    content: compressed,
                    token_count: compressedTokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(sessionEntries.map((entry) => entry.id)),
                    metadata: JSON.stringify({
                        type: "manual_compressed",
                        scope: "working",
                        originalCount,
                        originalTokens,
                        confidence: 0.9,
                        sessionId,
                    }),
                });
                persistPreparedVector(compressedVector);
                for (const entry of sessionEntries) {
                    removeVectorsBySource(entry.id);
                    deleteSummary(entry.id);
                }
            });
            const compressIds = new Set(sessionEntries.map((entry) => entry.id));
            const remaining = _workingMemory.filter((entry) => !compressIds.has(entry.id));
            _workingMemory.length = 0;
            _workingMemory.push(...remaining);

            return `Compressed ${originalCount} working memory entries (${originalTokens} tokens) into Tier 1 summary (${compressedTokens} tokens). Compression ratio: ${(originalTokens / Math.max(compressedTokens, 1)).toFixed(1)}x`;
        }

        case "session": {
            // Session summaries are strictly scoped: consolidating a new
            // session must never copy an archived predecessor into it.
            const tier1 = sessionId
                ? getSummariesByTierAndSession(1, sessionId)
                : [];
            if (tier1.length < 2) {
                return "Not enough Tier 1 summaries to consolidate.";
            }

            const combinedContent = tier1
                .map((s) => s.content)
                .join("\n\n");
            const originalTokens = tier1.reduce((s, r) => s + r.token_count, 0);

            const { text: compressed } = truncateToTokenBudget(
                combinedContent,
                config.tokenBudgets.tier1Session * 2
            );

            const summaryId = uuidv4();
            const compressedTokens = countTokens(compressed);

            const compressedVector = await prepareVector(compressed, summaryId, "summary", 0.85, { sessionId });
            withDatabaseTransaction(() => {
                insertSummary({
                    id: summaryId,
                    tier: 1,
                    content: compressed,
                    token_count: compressedTokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(tier1.map((s) => s.id)),
                    metadata: JSON.stringify({
                        type: "session_consolidated",
                        originalCount: tier1.length,
                        originalTokens,
                        confidence: 0.85,
                        sessionId,
                    }),
                });
                persistPreparedVector(compressedVector);
                for (const oldSummary of tier1) {
                    removeVectorsBySource(oldSummary.id);
                    deleteSummary(oldSummary.id);
                }
            });

            return `Consolidated ${tier1.length} Tier 1 summaries (${originalTokens} tokens) into 1 summary (${compressedTokens} tokens). Compression ratio: ${(originalTokens / Math.max(compressedTokens, 1)).toFixed(1)}x`;
        }

        case "epoch": {
            // Epoch promotion must consume only the active session's Tier 1
            // records. Tier 2 remains excluded from session retrieval, but
            // mixing inputs here would still violate lifecycle isolation.
            const tier1 = sessionId
                ? getSummariesByTierAndSession(1, sessionId)
                : [];
            if (tier1.length < config.compression.tier1ConsolidationCount) {
                return `Need at least ${config.compression.tier1ConsolidationCount} Tier 1 summaries for epoch consolidation (have ${tier1.length}).`;
            }

            const combinedContent = tier1
                .map((s) => s.content)
                .join("\n\n");
            const originalTokens = tier1.reduce((s, r) => s + r.token_count, 0);

            const { text: compressed } = truncateToTokenBudget(
                combinedContent,
                config.tokenBudgets.tier2Epoch
            );

            const epochId = uuidv4();
            const compressedTokens = countTokens(compressed);

            const epochVector = await prepareVector(compressed, epochId, "epoch", 0.8, { sessionId });
            withDatabaseTransaction(() => {
                insertSummary({
                    id: epochId,
                    tier: 2,
                    content: compressed,
                    token_count: compressedTokens,
                    session_id: sessionId,
                    source_ids: JSON.stringify(tier1.map((s) => s.id)),
                    metadata: JSON.stringify({
                        type: "epoch_summary",
                        originalCount: tier1.length,
                        originalTokens,
                        confidence: 0.8,
                        sessionId,
                    }),
                });
                persistPreparedVector(epochVector);
                for (const oldSummary of tier1) {
                    removeVectorsBySource(oldSummary.id);
                    deleteSummary(oldSummary.id);
                }
            });

            return `Promoted ${tier1.length} Tier 1 summaries (${originalTokens} tokens) into Tier 2 epoch summary (${compressedTokens} tokens). Compression ratio: ${(originalTokens / Math.max(compressedTokens, 1)).toFixed(1)}x`;
        }

        default:
            return `Unknown compression scope: ${scope}`;
    }
}

// ---------------------------------------------------------------------------
// Forget / deprecate
// ---------------------------------------------------------------------------

/**
 * Mark a memory as outdated, correct it, or delete it entirely.
 */
export async function forgetMemory(
    memoryId: string,
    action: ForgetAction,
    correction?: string
): Promise<string> {
    const summary = getSummaryById(memoryId);
    if (!summary) return `Memory ${memoryId} not found.`;

    const isWorkingMemory = summary.tier === 0;
    const wmIdx = _workingMemory.findIndex((entry) => entry.id === memoryId);
    if (action === "delete") {
        withDatabaseTransaction(() => {
            removeVectorsBySource(memoryId);
            deleteSummary(memoryId);
        });
        if (wmIdx !== -1) _workingMemory.splice(wmIdx, 1);
        return isWorkingMemory
            ? `Deleted working memory entry ${memoryId}.`
            : `Deleted memory ${memoryId} (was Tier ${summary.tier} summary).`;
    }

    if (action === "correct" && !correction) {
        return "Correction text required for 'correct' action.";
    }

    const content = action === "correct" ? correction! : `[DEPRECATED] ${summary.content}`;
    const metadata = parseSummaryMetadata(summary.metadata);
    const confidence = action === "deprecate" ? 0.1 : summaryVectorConfidence(summary);
    metadata.confidence = confidence;
    const vector = await prepareVector(content, memoryId, summaryVectorType(summary), confidence, {
        ...summaryVectorMetadata(summary, confidence),
        memoryType: summaryVectorType(summary),
    });

    // Updating text and replacing its search index is one durable mutation.
    // The cache is invalidated inside the transaction and reloads only after
    // the committed database image is visible.
    withDatabaseTransaction(() => {
        updateSummaryContent(memoryId, content, countTokens(content), JSON.stringify(metadata));
        removeVectorsBySource(memoryId);
        persistPreparedVector(vector);
    });

    if (wmIdx !== -1) {
        _workingMemory[wmIdx].content = content;
        _workingMemory[wmIdx].tokens = countTokens(content);
        _workingMemory[wmIdx].confidence = confidence;
    }
    if (action === "deprecate") return `Deprecated memory ${memoryId}.`;
    return isWorkingMemory
        ? `Corrected working memory entry ${memoryId}.`
        : `Corrected memory ${memoryId} with new content.`;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/**
 * Get comprehensive status of all memory subsystems.
 */
export function getMemoryStatus(): MemoryStatus {
    const tierCounts = getSummaryCountByTier();
    const summariesByTier: Record<number, SummaryRow[]> = {};

    for (let tier = 0; tier <= 3; tier++) {
        summariesByTier[tier] = getSummariesByTier(tier);
    }

    const sessionId = getCurrentSessionIdOrNull();
    const sessionEntries = getCurrentSessionWorkingMemoryEntries();
    const tier0Tokens = sessionEntries.reduce((s, e) => s + e.tokens, 0);
    const tier1Tokens = summariesByTier[1]?.reduce((s, r) => s + r.token_count, 0) || 0;
    const tier2Tokens = summariesByTier[2]?.reduce((s, r) => s + r.token_count, 0) || 0;
    const tier3Tokens = summariesByTier[3]?.reduce((s, r) => s + r.token_count, 0) || 0;

    const graphStats = getGraphStats();

    return {
        tiers: {
            tier0: { count: sessionEntries.length, tokenEstimate: tier0Tokens },
            tier1: { count: tierCounts[1] || 0, tokenEstimate: tier1Tokens },
            tier2: { count: tierCounts[2] || 0, tokenEstimate: tier2Tokens },
            tier3: { count: tierCounts[3] || 0, tokenEstimate: tier3Tokens },
        },
        knowledgeGraph: graphStats,
        vectorStore: { count: getVectorStoreCount() },
        totalTokensStored: tier0Tokens + tier1Tokens + tier2Tokens + tier3Tokens,
        currentSessionId: sessionId,
    };
}

/**
 * Get Tier 3 core memory content.
 */
export function getCoreMemory(): string {
    const tier3 = getSummariesByTier(3);
    if (tier3.length === 0) return "No core memories stored yet.";
    return tier3.map((s) => s.content).join("\n");
}

/**
 * Get current session working memory content.
 */
export function getCurrentSessionMemory(): string {
    const entries = getCurrentSessionWorkingMemoryEntries();
    if (entries.length === 0) return "No working memory entries for this session.";
    return entries.map((e) => `[${e.timestamp}] ${e.content}`).join("\n");
}

// ---------------------------------------------------------------------------
// Utility: Predicate inference
// ---------------------------------------------------------------------------

/**
 * Simple heuristic to infer a predicate from content and entity mentions.
 * In production, the LLM would do this via the extract_facts prompt.
 */
function inferPredicate(content: string, entities: string[]): string {
    const lower = content.toLowerCase();

    // Common predicate patterns
    const patterns: [RegExp, string][] = [
        [/\b(lives?\s+in|located\s+in|resides?\s+in|based\s+in)\b/i, "located_in"],
        [/\b(works?\s+(at|for)|employed\s+(at|by))\b/i, "works_at"],
        [/\b(likes?|loves?|enjoys?|prefers?)\b/i, "prefers"],
        [/\b(hates?|dislikes?|avoids?)\b/i, "dislikes"],
        [/\b(is\s+a|is\s+an|is\s+the)\b/i, "is_a"],
        [/\b(has|owns?|possesses?)\b/i, "has"],
        [/\b(knows?|met|friends?\s+with)\b/i, "knows"],
        [/\b(wants?\s+to|plans?\s+to|intends?\s+to|going\s+to)\b/i, "wants_to"],
        [/\b(created?|built|made|wrote|authored)\b/i, "created"],
        [/\b(uses?|utilizes?)\b/i, "uses"],
        [/\b(visited|went\s+to|traveled\s+to)\b/i, "visited"],
        [/\b(learned|studied|knows\s+about)\b/i, "learned"],
        [/\b(born\s+in|from)\b/i, "from"],
        [/\b(married\s+to|spouse|partner)\b/i, "married_to"],
        [/\b(parent\s+of|father\s+of|mother\s+of)\b/i, "parent_of"],
        [/\b(child\s+of|son\s+of|daughter\s+of)\b/i, "child_of"],
        [/\b(member\s+of|part\s+of|belongs?\s+to)\b/i, "member_of"],
        [/\b(manages?|leads?|heads?)\b/i, "manages"],
        [/\b(reports?\s+to|supervised\s+by)\b/i, "reports_to"],
        [/\b(teaches?|mentors?|coaches?)\b/i, "teaches"],
    ];

    for (const [pattern, predicate] of patterns) {
        if (pattern.test(lower)) {
            return predicate;
        }
    }

    return "related_to";
}
