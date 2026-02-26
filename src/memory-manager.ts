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
    addToVectorStore,
    removeVectorsBySource,
    getVectorStoreCount,
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

interface WorkingMemoryEntry {
    id: string;
    content: string;
    tokens: number;
    timestamp: string;
    sessionId: string | null;
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

function addToWorkingMemory(content: string): string {
    const id = uuidv4();
    const tokens = countTokens(content);
    const sessionId = getCurrentSessionIdOrNull();
    _workingMemory.push({
        id,
        content,
        tokens,
        timestamp: new Date().toISOString(),
        sessionId,
    });
    return id;
}

/**
 * Get the current session's working memory as text.
 */
export function getWorkingMemory(): string {
    const sessionId = getCurrentSessionIdOrNull();
    const entries = sessionId
        ? _workingMemory.filter((e) => e.sessionId === sessionId)
        : _workingMemory;
    if (entries.length === 0) return "";
    return entries.map((e) => e.content).join("\n");
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
    const sessionId = getCurrentSessionIdOrNull();
    if (!sessionId) return _workingMemory.length;
    return _workingMemory.filter((e) => e.sessionId === sessionId).length;
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

    // Get entries for this session only
    const sessionEntries = _workingMemory.filter((e) => e.sessionId === sessionId);
    if (sessionEntries.length === 0) return null;

    const combinedContent = sessionEntries.map((e) => e.content).join("\n");
    const originalTokens = sessionEntries.reduce((s, e) => s + e.tokens, 0);

    const { text: compressed } = truncateToTokenBudget(
        combinedContent,
        config.tokenBudgets.tier1Session
    );

    const summaryId = uuidv4();
    const compressedTokens = countTokens(compressed);

    insertSummary({
        id: summaryId,
        tier: 1,
        content: compressed,
        token_count: compressedTokens,
        session_id: sessionId,
        source_ids: JSON.stringify(sessionEntries.map((e) => e.id)),
        metadata: JSON.stringify({
            type: "session_archive",
            originalCount: sessionEntries.length,
            originalTokens,
            sessionId,
        }),
    });

    // Embed the archive for vector search
    try {
        await addToVectorStore(compressed, summaryId, "summary", 0.9, {
            sessionArchive: true,
            sessionId,
        });
    } catch {
        // non-fatal
    }

    // Remove archived entries from working memory
    const archiveIds = new Set(sessionEntries.map((e) => e.id));
    const remaining = _workingMemory.filter((e) => !archiveIds.has(e.id));
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
            // Tier 3 — always persisted, never evicted
            const summaryId = uuidv4();
            const tokens = countTokens(content);
            insertSummary({
                id: summaryId,
                tier: 3,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: "[]",
                metadata: JSON.stringify({ type: "core", entities, sessionId }),
            });
            result.memoryId = summaryId;
            result.tier = 3;

            // Also embed for vector search
            try {
                result.vectorId = await addToVectorStore(
                    content,
                    summaryId,
                    "core",
                    confidence,
                    { memoryType: "core", entities, sessionId }
                );
            } catch {
                // Embedding failure is non-fatal
            }
            break;
        }

        case "fact": {
            // Store as knowledge graph triples + vector embed
            const factId = uuidv4();

            // Ensure entities exist in the graph
            for (const entityLabel of entities) {
                ensureEntity(entityLabel, "unknown", {}, confidence);
                result.entitiesCreated.push(entityLabel);
            }

            // If we have at least 2 entities, create relations between them
            if (entities.length >= 2) {
                // Simple heuristic: first entity is subject, infer predicate from content
                const predicate = inferPredicate(content, entities);
                for (let i = 1; i < entities.length; i++) {
                    storeFact(
                        entities[0],
                        predicate,
                        entities[i],
                        "unknown",
                        "unknown",
                        confidence
                    );
                    result.factsStored++;
                }
            }

            // Also store as Tier 1 summary for retrieval
            const tokens = countTokens(content);
            insertSummary({
                id: factId,
                tier: 1,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: JSON.stringify(entities),
                metadata: JSON.stringify({
                    type: "fact",
                    entities,
                    confidence,
                    sessionId,
                }),
            });

            // Embed for vector search
            try {
                result.vectorId = await addToVectorStore(
                    content,
                    factId,
                    "fact",
                    confidence,
                    { memoryType: "fact", entities, sessionId }
                );
            } catch {
                // Embedding failure is non-fatal
            }

            result.memoryId = factId;
            result.tier = 1;
            break;
        }

        case "preference": {
            // Preferences are important — store at Tier 2 and in the graph
            const prefId = uuidv4();
            const tokens = countTokens(content);

            // Create entity for "User" preferences
            ensureEntity("User", "person", {}, 1.0);
            for (const entityLabel of entities) {
                ensureEntity(entityLabel, "unknown", {}, confidence);
                storeFact("User", "prefers", entityLabel, "person", "unknown", confidence);
                result.factsStored++;
            }
            result.entitiesCreated = ["User", ...entities];

            insertSummary({
                id: prefId,
                tier: 2,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: JSON.stringify(entities),
                metadata: JSON.stringify({ type: "preference", entities, sessionId }),
            });

            try {
                result.vectorId = await addToVectorStore(
                    content,
                    prefId,
                    "preference",
                    confidence,
                    { memoryType: "preference", entities, sessionId }
                );
            } catch {
                // non-fatal
            }

            result.memoryId = prefId;
            result.tier = 2;
            break;
        }

        case "event": {
            // Events go to Tier 0 working memory + vector embed
            const eventId = addToWorkingMemory(content);

            for (const entityLabel of entities) {
                ensureEntity(entityLabel, "unknown", {}, confidence);
                result.entitiesCreated.push(entityLabel);
            }

            try {
                result.vectorId = await addToVectorStore(
                    content,
                    eventId,
                    "event",
                    confidence,
                    { memoryType: "event", entities, timestamp: new Date().toISOString(), sessionId }
                );
            } catch {
                // non-fatal
            }

            result.memoryId = eventId;
            result.tier = 0;

            // Check if Tier 0 needs overflow compression
            await checkTier0Overflow();
            break;
        }

        case "summary": {
            // Explicit summaries go to Tier 1
            const sumId = uuidv4();
            const tokens = countTokens(content);

            insertSummary({
                id: sumId,
                tier: 1,
                content,
                token_count: tokens,
                session_id: sessionId,
                source_ids: JSON.stringify(entities),
                metadata: JSON.stringify({ type: "summary", entities, sessionId }),
            });

            try {
                result.vectorId = await addToVectorStore(
                    content,
                    sumId,
                    "summary",
                    confidence,
                    { memoryType: "summary", entities, sessionId }
                );
            } catch {
                // non-fatal
            }

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

    // Remove compressed entries from the main array
    const compressIds = new Set(toCompress.map((e) => e.id));
    const remaining = _workingMemory.filter((e) => !compressIds.has(e.id));
    _workingMemory.length = 0;
    _workingMemory.push(...remaining);

    const combinedContent = toCompress.map((e) => e.content).join("\n");

    // Create a compressed Tier 1 summary
    const { text: compressed } = truncateToTokenBudget(
        combinedContent,
        config.tokenBudgets.tier1Session
    );

    const summaryId = uuidv4();
    const tokens = countTokens(compressed);

    insertSummary({
        id: summaryId,
        tier: 1,
        content: compressed,
        token_count: tokens,
        session_id: sessionId,
        source_ids: JSON.stringify(toCompress.map((e) => e.id)),
        metadata: JSON.stringify({
            type: "auto_compressed",
            originalCount: toCompress.length,
            originalTokens: toCompress.reduce((s, e) => s + e.tokens, 0),
            sessionId,
        }),
    });

    // Embed the compressed summary
    try {
        await addToVectorStore(compressed, summaryId, "summary", 0.9, {
            autoCompressed: true,
            sessionId,
        });
    } catch {
        // non-fatal
    }
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

            insertSummary({
                id: summaryId,
                tier: 1,
                content: compressed,
                token_count: compressedTokens,
                session_id: sessionId,
                source_ids: JSON.stringify(sessionEntries.map((e) => e.id)),
                metadata: JSON.stringify({
                    type: "manual_compressed",
                    scope: "working",
                    originalCount,
                    originalTokens,
                    sessionId,
                }),
            });

            try {
                await addToVectorStore(compressed, summaryId, "summary", 0.9, { sessionId });
            } catch {
                // non-fatal
            }

            // Clear current session working memory
            const compressIds = new Set(sessionEntries.map((e) => e.id));
            const remaining = _workingMemory.filter((e) => !compressIds.has(e.id));
            _workingMemory.length = 0;
            _workingMemory.push(...remaining);

            return `Compressed ${originalCount} working memory entries (${originalTokens} tokens) into Tier 1 summary (${compressedTokens} tokens). Compression ratio: ${(originalTokens / Math.max(compressedTokens, 1)).toFixed(1)}x`;
        }

        case "session": {
            // Compress multiple Tier 1 summaries into fewer entries
            const tier1 = getSummariesByTier(1);
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
                    sessionId,
                }),
            });

            // Remove old Tier 1 summaries
            for (const oldSummary of tier1) {
                removeVectorsBySource(oldSummary.id);
                deleteSummary(oldSummary.id);
            }

            try {
                await addToVectorStore(compressed, summaryId, "summary", 0.85, { sessionId });
            } catch {
                // non-fatal
            }

            return `Consolidated ${tier1.length} Tier 1 summaries (${originalTokens} tokens) into 1 summary (${compressedTokens} tokens). Compression ratio: ${(originalTokens / Math.max(compressedTokens, 1)).toFixed(1)}x`;
        }

        case "epoch": {
            // Promote Tier 1 summaries into Tier 2 epoch summaries
            const tier1 = getSummariesByTier(1);
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
                    sessionId,
                }),
            });

            // Remove promoted Tier 1 summaries
            for (const oldSummary of tier1) {
                removeVectorsBySource(oldSummary.id);
                deleteSummary(oldSummary.id);
            }

            try {
                await addToVectorStore(compressed, epochId, "epoch", 0.8, { sessionId });
            } catch {
                // non-fatal
            }

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
    // Check if it's a summary
    const summary = getSummaryById(memoryId);
    if (summary) {
        switch (action) {
            case "delete":
                removeVectorsBySource(memoryId);
                deleteSummary(memoryId);
                return `Deleted memory ${memoryId} (was Tier ${summary.tier} summary).`;

            case "deprecate":
                updateSummaryContent(
                    memoryId,
                    `[DEPRECATED] ${summary.content}`,
                    summary.token_count + 15
                );
                return `Deprecated memory ${memoryId}.`;

            case "correct":
                if (!correction) return "Correction text required for 'correct' action.";
                const tokens = countTokens(correction);
                updateSummaryContent(memoryId, correction, tokens);
                // Re-embed with new content
                removeVectorsBySource(memoryId);
                try {
                    await addToVectorStore(correction, memoryId, summary.tier === 3 ? "core" : "summary", 0.9);
                } catch {
                    // non-fatal
                }
                return `Corrected memory ${memoryId} with new content.`;
        }
    }

    // Check if it's a working memory entry
    const wmIdx = _workingMemory.findIndex((e) => e.id === memoryId);
    if (wmIdx !== -1) {
        if (action === "delete") {
            _workingMemory.splice(wmIdx, 1);
            return `Deleted working memory entry ${memoryId}.`;
        }
        if (action === "correct" && correction) {
            _workingMemory[wmIdx].content = correction;
            _workingMemory[wmIdx].tokens = countTokens(correction);
            return `Corrected working memory entry ${memoryId}.`;
        }
    }

    return `Memory ${memoryId} not found.`;
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
    const sessionEntries = sessionId
        ? _workingMemory.filter((e) => e.sessionId === sessionId)
        : _workingMemory;
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
    const sessionId = getCurrentSessionIdOrNull();
    const entries = sessionId
        ? _workingMemory.filter((e) => e.sessionId === sessionId)
        : _workingMemory;
    if (entries.length === 0) return "No working memory entries for this session.";
    return entries
        .map((e) => `[${e.timestamp}] ${e.content}`)
        .join("\n");
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
