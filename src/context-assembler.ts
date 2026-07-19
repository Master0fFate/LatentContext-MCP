import {
    getSummariesByTierAndSession,
    logAccess,
} from "./database.js";
import { countTokens, truncateToTokenBudget } from "./token-counter.js";
import {
    getCurrentSessionWorkingMemoryEntries,
    type WorkingMemoryEntry,
} from "./memory-manager.js";
import { getConfig } from "./config.js";
import { getCurrentSessionIdOrNull } from "./session.js";

// Filters intentionally use the vector-store spelling internally so existing
// callers remain compatible. In retrieval they apply to memory types, not the
// global vector store (which is deliberately excluded for session isolation).
export interface RetrievalFilters {
    sourceTypes?: string[];
    after?: string;
    before?: string;
    minConfidence?: number;
}

const SUPPORTED_MEMORY_TYPES = new Set(["fact", "preference", "event", "summary", "core"]);
type MemoryTypeName = "fact" | "preference" | "event" | "summary" | "core";

interface ContextCandidate {
    id: string;
    content: string;
    source: "working" | "current_session";
    memoryType: MemoryTypeName;
    confidence: number;
    similarity: number;
    priority: number;
    createdAt: string;
    tokenCount: number;
    /** Cached once per candidate: deduplication compares these sets repeatedly. */
    termSet: ReadonlySet<string>;
}

export interface AssembledContext {
    text: string;
    totalTokens: number;
    budgetUsed: number;
    budgetRemaining: number;
    sourceCounts: Record<string, number>;
    candidatesConsidered: number;
    candidatesSelected: number;
    sessionId: string | null;
}

function sourcePriority(source: ContextCandidate["source"]): number {
    return source === "working" ? 0.95 : 0.9;
}

/** Tokenize locally so retrieval remains fast, deterministic, and model-free. */
function terms(text: string): Set<string> {
    return new Set(
        text
            .toLocaleLowerCase()
            .match(/[\p{L}\p{N}_-]+/gu)
            ?.filter((term) => term.length > 1) ?? []
    );
}

/**
 * Query coverage is a deterministic lexical relevance signal. It avoids an
 * embedding request (and therefore a model download) on every retrieval.
 */
function querySimilarity(queryTerms: ReadonlySet<string>, contentTerms: ReadonlySet<string>): number {
    if (queryTerms.size === 0) return 0;

    let matches = 0;
    for (const term of queryTerms) {
        if (contentTerms.has(term)) matches++;
    }
    return matches / queryTerms.size;
}

function indexTerms(index: number, termSet: ReadonlySet<string>, indexesByTerm: Map<string, Set<number>>): void {
    for (const term of termSet) {
        let indexes = indexesByTerm.get(term);
        if (!indexes) {
            indexes = new Set();
            indexesByTerm.set(term, indexes);
        }
        indexes.add(index);
    }
}

function unindexTerms(index: number, termSet: ReadonlySet<string>, indexesByTerm: Map<string, Set<number>>): void {
    for (const term of termSet) {
        const indexes = indexesByTerm.get(term);
        if (!indexes) continue;
        indexes.delete(index);
        if (indexes.size === 0) indexesByTerm.delete(term);
    }
}

/**
 * Preserve first-match deduplication while avoiding comparisons between
 * candidates that share no term. Term sets are built once, rather than once
 * for every candidate pair.
 */
function deduplicate(candidates: ContextCandidate[], threshold: number): ContextCandidate[] {
    const result: ContextCandidate[] = [];
    const indexesByTerm = new Map<string, Set<number>>();

    for (const candidate of candidates) {
        // At a zero threshold even disjoint (or empty) term sets are duplicates.
        // Keep that configurable edge case identical to the original scan.
        if (threshold <= 0 && result.length > 0) {
            if (compareCandidates(candidate, result[0]) < 0) {
                unindexTerms(0, result[0].termSet, indexesByTerm);
                result[0] = candidate;
                indexTerms(0, candidate.termSet, indexesByTerm);
            }
            continue;
        }

        // Count intersections directly from the inverted index. This avoids
        // sorting candidate indexes and re-scanning candidate terms for every
        // possible pair; a pair is evaluated only when it shares a term.
        const intersectionsByIndex = new Map<number, number>();
        for (const term of candidate.termSet) {
            for (const index of indexesByTerm.get(term) ?? []) {
                intersectionsByIndex.set(index, (intersectionsByIndex.get(index) ?? 0) + 1);
            }
        }

        let duplicateIndex: number | undefined;
        for (const [index, intersection] of intersectionsByIndex) {
            const existing = result[index];
            const similarity = intersection / (candidate.termSet.size + existing.termSet.size - intersection);
            if (similarity >= threshold && (duplicateIndex === undefined || index < duplicateIndex)) {
                // The prior linear scan used the first duplicate, not the
                // highest-scoring one, so retain that observable behavior.
                duplicateIndex = index;
            }
        }

        if (duplicateIndex === undefined) {
            const index = result.length;
            result.push(candidate);
            indexTerms(index, candidate.termSet, indexesByTerm);
        } else if (compareCandidates(candidate, result[duplicateIndex]) < 0) {
            unindexTerms(duplicateIndex, result[duplicateIndex].termSet, indexesByTerm);
            result[duplicateIndex] = candidate;
            indexTerms(duplicateIndex, candidate.termSet, indexesByTerm);
        }
    }
    return result;
}

/** Higher query relevance always wins; remaining criteria break ties predictably. */
function compareCandidates(a: ContextCandidate, b: ContextCandidate): number {
    return (
        b.similarity - a.similarity ||
        b.createdAt.localeCompare(a.createdAt) ||
        b.priority - a.priority ||
        a.id.localeCompare(b.id)
    );
}

function parseMetadata(metadata: string): Record<string, unknown> {
    try {
        const parsed = JSON.parse(metadata);
        return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    } catch {
        return {};
    }
}

function summaryMemoryType(metadata: Record<string, unknown>): MemoryTypeName {
    const type = metadata.type;
    return typeof type === "string" && SUPPORTED_MEMORY_TYPES.has(type)
        ? type as MemoryTypeName
        : "summary";
}

function summaryConfidence(metadata: Record<string, unknown>): number {
    const confidence = metadata.confidence;
    return typeof confidence === "number" && confidence >= 0 && confidence <= 1
        ? confidence
        : 1;
}

function validateFilters(filters?: RetrievalFilters): void {
    if (!filters) return;
    if (filters.sourceTypes !== undefined) {
        if (!Array.isArray(filters.sourceTypes) || filters.sourceTypes.length === 0) {
            throw new Error("memory_types filter must be a non-empty array.");
        }
        for (const type of filters.sourceTypes) {
            if (!SUPPORTED_MEMORY_TYPES.has(type)) {
                throw new Error(`Unsupported memory type filter: ${type}`);
            }
        }
    }
    for (const [name, value] of [["after", filters.after], ["before", filters.before]] as const) {
        if (value !== undefined && Number.isNaN(Date.parse(value))) {
            throw new Error(`Invalid ${name} filter; use an ISO datetime.`);
        }
    }
    if (filters.after && filters.before && Date.parse(filters.after) > Date.parse(filters.before)) {
        throw new Error("after filter must not be later than before filter.");
    }
    if (
        filters.minConfidence !== undefined &&
        (!Number.isFinite(filters.minConfidence) || filters.minConfidence < 0 || filters.minConfidence > 1)
    ) {
        throw new Error("min_confidence filter must be a number between 0 and 1.");
    }
}

function isEligible(candidate: ContextCandidate, filters?: RetrievalFilters): boolean {
    if (!filters) return true;
    if (filters.sourceTypes && !filters.sourceTypes.includes(candidate.memoryType)) return false;
    if (filters.after && Date.parse(candidate.createdAt) < Date.parse(filters.after)) return false;
    if (filters.before && Date.parse(candidate.createdAt) > Date.parse(filters.before)) return false;
    if (filters.minConfidence !== undefined && candidate.confidence < filters.minConfidence) return false;
    return true;
}

function fromWorkingEntry(entry: WorkingMemoryEntry, queryTerms: ReadonlySet<string>): ContextCandidate {
    const termSet = terms(entry.content);
    return {
        id: entry.id,
        content: entry.content,
        source: "working",
        memoryType: entry.memoryType,
        confidence: entry.confidence,
        similarity: querySimilarity(queryTerms, termSet),
        priority: sourcePriority("working"),
        createdAt: entry.timestamp,
        tokenCount: entry.tokens,
        termSet,
    };
}

const SOURCE_ORDER = ["working", "current_session"] as const;
const SOURCE_HEADERS = {
    working: "[Current Session] ",
    current_session: "[Current Session Notes] ",
} as const;

function renderFooter(sessionId: string, sourceCounts: Record<string, number>): string {
    const sourceList = Object.entries(sourceCounts)
        .map(([source, count]) => `${source}:${count}`)
        .join(", ");
    return `--- Session: ${sessionId.substring(0, 20)} | Sources: ${sourceList} ---`;
}

function renderContext(
    selected: ContextCandidate[],
    sessionId: string,
    sourceCounts: Record<string, number>
): string {
    const sections: string[] = [];
    for (const source of SOURCE_ORDER) {
        const items = selected.filter((candidate) => candidate.source === source);
        if (items.length === 0) continue;
        sections.push(`${SOURCE_HEADERS[source]}${items.map((item) => item.content).join("\n")}`);
    }
    return `${sections.join("\n\n")}\n\n${renderFooter(sessionId, sourceCounts)}`;
}

/**
 * Fast packing estimate. Exact rendering is checked after packing, so this
 * estimate can never make the returned payload exceed its token budget.
 */
function estimateContextTokens(
    sessionId: string,
    sourceCounts: Record<string, number>,
    sourceTokenTotals: Record<ContextCandidate["source"], number>
): number {
    let total = 0;
    let sections = 0;
    for (const source of SOURCE_ORDER) {
        const count = sourceCounts[source] ?? 0;
        if (count === 0) continue;
        if (sections > 0) total += countTokens("\n\n");
        total += countTokens(SOURCE_HEADERS[source]);
        total += sourceTokenTotals[source];
        total += (count - 1) * countTokens("\n");
        sections++;
    }
    return total + countTokens("\n\n") + countTokens(renderFooter(sessionId, sourceCounts));
}

/**
 * Assemble only active-session context. Tier 0 entries stay separate
 * candidates so ranking and token packing can choose each entry independently.
 */
export async function assembleContext(
    query: string,
    tokenBudget?: number,
    filters?: RetrievalFilters
): Promise<AssembledContext> {
    if (tokenBudget !== undefined && (!Number.isInteger(tokenBudget) || tokenBudget <= 0)) {
        throw new Error("token budget must be a positive integer.");
    }
    validateFilters(filters);

    const budget = tokenBudget ?? getConfig().tokenBudgets.defaultRetrieveBudget;
    const sessionId = getCurrentSessionIdOrNull();
    const emptyText = "No memories stored in this session yet. This is a fresh session — use memory_store to save important information as you go.";
    const emptyContext = () => truncateToTokenBudget(emptyText, budget);

    // Never fall back to unscoped memory. An inactive session is always empty.
    if (!sessionId) {
        const { text, tokens } = emptyContext();
        return {
            text,
            totalTokens: tokens,
            budgetUsed: tokens,
            budgetRemaining: budget - tokens,
            sourceCounts: {},
            candidatesConsidered: 0,
            candidatesSelected: 0,
            sessionId: null,
        };
    }

    const queryTerms = terms(query);
    const candidates: ContextCandidate[] = [];
    for (const entry of getCurrentSessionWorkingMemoryEntries()) {
        const candidate = fromWorkingEntry(entry, queryTerms);
        if (isEligible(candidate, filters)) candidates.push(candidate);
    }

    for (const summary of getSummariesByTierAndSession(1, sessionId)) {
        const metadata = parseMetadata(summary.metadata);
        const termSet = terms(summary.content);
        const candidate: ContextCandidate = {
            id: summary.id,
            content: summary.content,
            source: "current_session",
            memoryType: summaryMemoryType(metadata),
            confidence: summaryConfidence(metadata),
            similarity: querySimilarity(queryTerms, termSet),
            priority: sourcePriority("current_session"),
            createdAt: summary.created_at,
            // Tier 1 content was tokenized when persisted. Reuse that exact
            // value for greedy packing instead of encoding every summary again.
            // The final rendered payload remains the authoritative budget check.
            tokenCount: summary.token_count,
            termSet,
        };
        if (isEligible(candidate, filters)) candidates.push(candidate);
    }

    const ranked = deduplicate(
        candidates,
        getConfig().ranking.dedupSimilarityThreshold
    ).sort(compareCandidates);
    const selected: ContextCandidate[] = [];
    const sourceCounts: Record<string, number> = {};
    const sourceTokenTotals: Record<ContextCandidate["source"], number> = {
        working: 0,
        current_session: 0,
    };

    // Counting a progressively larger rendered string for every candidate is
    // quadratic in its total text size. Pack from cached content token counts,
    // then verify the exact final payload before recording any access.
    for (const candidate of ranked) {
        const nextCounts = {
            ...sourceCounts,
            [candidate.source]: (sourceCounts[candidate.source] || 0) + 1,
        };
        const nextTokenTotals = {
            ...sourceTokenTotals,
            [candidate.source]: sourceTokenTotals[candidate.source] + candidate.tokenCount,
        };
        if (estimateContextTokens(sessionId, nextCounts, nextTokenTotals) <= budget) {
            selected.push(candidate);
            sourceCounts[candidate.source] = nextCounts[candidate.source];
            sourceTokenTotals[candidate.source] = nextTokenTotals[candidate.source];
        }
    }

    let rendered = selected.length > 0
        ? { text: renderContext(selected, sessionId, sourceCounts), tokens: 0 }
        : emptyContext();
    let totalTokens = rendered.tokens || countTokens(rendered.text);
    while (selected.length > 0 && totalTokens > budget) {
        const removed = selected.pop()!;
        sourceCounts[removed.source]--;
        sourceTokenTotals[removed.source] -= removed.tokenCount;
        if (sourceCounts[removed.source] === 0) delete sourceCounts[removed.source];
        rendered = selected.length > 0
            ? { text: renderContext(selected, sessionId, sourceCounts), tokens: 0 }
            : emptyContext();
        totalTokens = rendered.tokens || countTokens(rendered.text);
    }
    for (const candidate of selected) {
        logAccess(candidate.id, candidate.source);
    }
    return {
        text: rendered.text,
        totalTokens,
        budgetUsed: totalTokens,
        budgetRemaining: budget - totalTokens,
        sourceCounts,
        candidatesConsidered: candidates.length,
        candidatesSelected: selected.length,
        sessionId,
    };
}
