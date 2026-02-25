import { getConfig } from "./config.js";
import {
    getSummariesByTier,
    getSummariesByTierAndSession,
    getSummariesByTierExcludingSession,
    logAccess,
    getAccessFrequency,
} from "./database.js";
import { queryEntity, serializeFacts, getAllFacts } from "./knowledge-graph.js";
import { searchVectors, type VectorSearchResult, type VectorSearchFilter } from "./vector-store.js";
import { countTokens, truncateToTokenBudget } from "./token-counter.js";
import { cosineSimilarity } from "./embeddings.js";
import { getCoreMemory, getWorkingMemory } from "./memory-manager.js";
import { getCurrentSessionIdOrNull, getSessionStartTime } from "./session.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ContextCandidate {
    id: string;
    content: string;
    tokens: number;
    score: number;
    source: string; // 'core' | 'working' | 'vector' | 'graph' | 'current_session' | 'past_sessions' | 'long_term'
    similarity: number;
    recency: number;
    priority: number;
    frequency: number;
    createdAt: string;
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

// ---------------------------------------------------------------------------
// Recency scoring
// ---------------------------------------------------------------------------

function recencyScore(createdAt: string): number {
    const now = Date.now();
    const created = new Date(createdAt).getTime();
    const ageHours = (now - created) / (1000 * 60 * 60);

    // Exponential decay: score = e^(-ageHours / 168)
    // Half-life of ~1 week (168 hours)
    return Math.exp(-ageHours / 168);
}

// ---------------------------------------------------------------------------
// Source priority scoring
// ---------------------------------------------------------------------------

function sourcePriority(source: string): number {
    switch (source) {
        case "core":
            return 1.0;
        case "working":
            return 0.95;
        case "current_session":
            return 0.9;
        case "graph":
            return 0.8;
        case "long_term":
            return 0.65;
        case "past_sessions":
            return 0.5;
        case "vector":
            return 0.4;
        default:
            return 0.3;
    }
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

function deduplicate(candidates: ContextCandidate[], threshold: number): ContextCandidate[] {
    const result: ContextCandidate[] = [];

    for (const candidate of candidates) {
        let isDuplicate = false;

        for (const existing of result) {
            // Simple text-based similarity check using character overlap
            const similarity = textSimilarity(candidate.content, existing.content);
            if (similarity >= threshold) {
                // Keep the one with higher score
                if (candidate.score > existing.score) {
                    const idx = result.indexOf(existing);
                    result[idx] = candidate;
                }
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate) {
            result.push(candidate);
        }
    }

    return result;
}

/**
 * Simple Jaccard-like text similarity based on word overlap.
 * Fast approximation — no embeddings needed.
 */
function textSimilarity(a: string, b: string): number {
    const wordsA = new Set(a.toLowerCase().split(/\s+/).filter((w) => w.length > 2));
    const wordsB = new Set(b.toLowerCase().split(/\s+/).filter((w) => w.length > 2));

    if (wordsA.size === 0 || wordsB.size === 0) return 0;

    let intersection = 0;
    for (const word of wordsA) {
        if (wordsB.has(word)) intersection++;
    }

    const union = wordsA.size + wordsB.size - intersection;
    return union === 0 ? 0 : intersection / union;
}

// ---------------------------------------------------------------------------
// Entity extraction (simple heuristic)
// ---------------------------------------------------------------------------

function extractEntityMentions(query: string): string[] {
    // Extract capitalized words/phrases as potential entity mentions
    const matches = query.match(/\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*/g) || [];

    // Also extract quoted strings
    const quoted = query.match(/"([^"]+)"/g) || [];
    const quotedClean = quoted.map((q) => q.replace(/"/g, ""));

    const entities = [...new Set([...matches, ...quotedClean])];

    // Filter out common English words that might be capitalized (sentence starts, etc.)
    const stopWords = new Set([
        "The", "This", "That", "What", "When", "Where", "Who", "Why", "How",
        "I", "You", "We", "They", "He", "She", "It", "My", "Your", "Our",
        "Can", "Could", "Would", "Should", "Do", "Does", "Did", "Is", "Are",
        "Was", "Were", "Has", "Have", "Had", "Will", "And", "But", "Or",
        "Not", "No", "Yes", "If", "Then", "So", "Also", "Just", "Only",
        "About", "After", "Before", "Between", "From", "Into", "For", "With",
    ]);

    return entities.filter((e) => !stopWords.has(e) && e.length > 1);
}

// ---------------------------------------------------------------------------
// Context assembly (the core algorithm)
// ---------------------------------------------------------------------------

/**
 * Assemble the optimal context payload for a given query within a token budget.
 *
 * Algorithm:
 * 1. Always include Tier 3 core memory
 * 2. Include current session working memory (Tier 0)
 * 3. Embed query → vector search (top 20 candidates)
 * 4. Extract entity mentions → knowledge graph queries
 * 5. Gather current session Tier 1 summaries
 * 6. Gather past-session Tier 1 and Tier 2 summaries
 * 7. Score all candidates by composite relevance
 * 8. Deduplicate
 * 9. Greedily fill budget by score
 * 10. Format output with session-aware section headers
 */
export async function assembleContext(
    query: string,
    tokenBudget?: number,
    filters?: VectorSearchFilter
): Promise<AssembledContext> {
    const config = getConfig();
    const budget = tokenBudget || config.tokenBudgets.defaultRetrieveBudget;
    let remainingBudget = budget;

    const sessionId = getCurrentSessionIdOrNull();
    const candidates: ContextCandidate[] = [];
    const sourceCounts: Record<string, number> = {};

    // ── 1. Always include Core Memory (Tier 3) ──
    const coreMemory = getCoreMemory();
    const coreTokens = countTokens(coreMemory);
    let coreSection = "";

    if (coreMemory !== "No core memories stored yet." && coreTokens > 0) {
        const { text: coreTruncated } = truncateToTokenBudget(
            coreMemory,
            Math.min(coreTokens, config.tokenBudgets.tier3Core)
        );
        coreSection = `[Core Memory] ${coreTruncated}`;
        remainingBudget -= countTokens(coreSection);
        sourceCounts["core"] = 1;
    }

    // ── 2. Current session working memory (Tier 0) ──
    const workingMem = getWorkingMemory();
    if (workingMem && workingMem.length > 0) {
        const wmTokens = countTokens(workingMem);
        candidates.push({
            id: "working-memory",
            content: workingMem,
            tokens: wmTokens,
            score: 0,
            source: "working",
            similarity: 0.6,
            recency: 1.0,
            priority: sourcePriority("working"),
            frequency: 1.0,
            createdAt: new Date().toISOString(),
        });
    }

    // ── 3. Vector search ──
    try {
        const vectorResults = await searchVectors(query, 20, filters);
        for (const vr of vectorResults) {
            if (vr.similarity < 0.3) continue; // Skip low-relevance results

            const freq = getAccessFrequency(vr.id);
            candidates.push({
                id: vr.id,
                content: vr.contentPreview,
                tokens: countTokens(vr.contentPreview),
                score: 0,
                source: "vector",
                similarity: vr.similarity,
                recency: recencyScore(vr.createdAt),
                priority: sourcePriority(vr.sourceType === "core" ? "core" : "vector"),
                frequency: Math.min(freq / 10, 1.0),
                createdAt: vr.createdAt,
            });
        }
    } catch {
        // Vector search failure is non-fatal (e.g., embedding model not loaded)
    }

    // ── 4. Knowledge graph queries ──
    const entityMentions = extractEntityMentions(query);
    const graphContents: string[] = [];

    for (const entity of entityMentions.slice(0, 5)) {
        const result = queryEntity(entity, 2);
        if (result) {
            graphContents.push(result.serialized);
            logAccess(result.entity.id, "entity");
        }
    }

    if (graphContents.length > 0) {
        const graphText = graphContents.join("\n\n");
        const graphTokens = countTokens(graphText);

        candidates.push({
            id: "graph-" + entityMentions.join("-"),
            content: graphText,
            tokens: graphTokens,
            score: 0,
            source: "graph",
            similarity: 0.7,
            recency: 1.0,
            priority: sourcePriority("graph"),
            frequency: 0.5,
            createdAt: new Date().toISOString(),
        });
    }

    // ── 5. Current session Tier 1 summaries ──
    if (sessionId) {
        const currentSessionSummaries = getSummariesByTierAndSession(1, sessionId);
        for (const summary of currentSessionSummaries.slice(0, 5)) {
            const freq = getAccessFrequency(summary.id);
            candidates.push({
                id: summary.id,
                content: summary.content,
                tokens: summary.token_count,
                score: 0,
                source: "current_session",
                similarity: 0.6,
                recency: recencyScore(summary.created_at),
                priority: sourcePriority("current_session"),
                frequency: Math.min(freq / 10, 1.0),
                createdAt: summary.created_at,
            });
        }
    }

    // ── 6. Past session Tier 1 and Tier 2 summaries ──
    const pastTier1 = sessionId
        ? getSummariesByTierExcludingSession(1, sessionId)
        : getSummariesByTier(1);
    for (const summary of pastTier1.slice(0, 10)) {
        const freq = getAccessFrequency(summary.id);
        candidates.push({
            id: summary.id,
            content: summary.content,
            tokens: summary.token_count,
            score: 0,
            source: "past_sessions",
            similarity: 0.5,
            recency: recencyScore(summary.created_at),
            priority: sourcePriority("past_sessions"),
            frequency: Math.min(freq / 10, 1.0),
            createdAt: summary.created_at,
        });
    }

    const tier2Summaries = getSummariesByTier(2);
    for (const summary of tier2Summaries.slice(0, 5)) {
        const freq = getAccessFrequency(summary.id);
        candidates.push({
            id: summary.id,
            content: summary.content,
            tokens: summary.token_count,
            score: 0,
            source: "long_term",
            similarity: 0.4,
            recency: recencyScore(summary.created_at),
            priority: sourcePriority("long_term"),
            frequency: Math.min(freq / 10, 1.0),
            createdAt: summary.created_at,
        });
    }

    // ── 7. Compute composite scores ──
    const weights = config.ranking;
    for (const candidate of candidates) {
        candidate.score =
            weights.semanticWeight * candidate.similarity +
            weights.recencyWeight * candidate.recency +
            weights.priorityWeight * candidate.priority +
            weights.frequencyWeight * candidate.frequency;
    }

    // ── 8. Deduplicate ──
    const deduped = deduplicate(candidates, weights.dedupSimilarityThreshold);

    // ── 9. Sort by score and greedily fill budget ──
    deduped.sort((a, b) => b.score - a.score);

    const selected: ContextCandidate[] = [];
    for (const candidate of deduped) {
        if (candidate.tokens <= remainingBudget) {
            selected.push(candidate);
            remainingBudget -= candidate.tokens;
            sourceCounts[candidate.source] = (sourceCounts[candidate.source] || 0) + 1;

            // Log access for frequency tracking
            logAccess(candidate.id, candidate.source);
        }
    }

    // ── 10. Format output with session-aware sections ──
    const sections: string[] = [];

    // Core Memory always first
    if (coreSection) {
        sections.push(coreSection);
    }

    // Group selected candidates by source
    const bySource: Record<string, ContextCandidate[]> = {};
    for (const s of selected) {
        if (!bySource[s.source]) bySource[s.source] = [];
        bySource[s.source].push(s);
    }

    // Define section order and labels — session-scoped then cross-session
    const sourceOrder = ["working", "current_session", "graph", "long_term", "past_sessions", "vector"];
    const sourceLabels: Record<string, string> = {
        working: "Current Session",
        current_session: "Current Session Notes",
        graph: "Known Facts",
        long_term: "Long-term Knowledge",
        past_sessions: "Past Sessions",
        vector: "Semantic Matches",
    };

    for (const source of sourceOrder) {
        const items = bySource[source];
        if (items && items.length > 0) {
            const label = sourceLabels[source] || source;
            const content = items.map((item) => item.content).join("\n");
            sections.push(`[${label}] ${content}`);
        }
    }

    // Build final text
    let finalText: string;

    if (sections.length > 0) {
        finalText = sections.join("\n\n");

        // Add metadata footer
        const sourceList = Object.entries(sourceCounts)
            .map(([source, count]) => `${source}:${count}`)
            .join(", ");
        const budgetUsed = budget - remainingBudget;
        const sessionLabel = sessionId ? sessionId.substring(0, 8) : "none";
        finalText += `\n\n--- Session: ${sessionLabel} | Sources: ${sourceList} | Tokens: ${budgetUsed}/${budget} ---`;
    } else {
        finalText = "No relevant memories found. This appears to be a new conversation — use memory_store to save important information as you go.";
    }

    const totalTokens = countTokens(finalText);

    return {
        text: finalText,
        totalTokens,
        budgetUsed: budget - remainingBudget,
        budgetRemaining: remainingBudget,
        sourceCounts,
        candidatesConsidered: candidates.length,
        candidatesSelected: selected.length + (coreSection ? 1 : 0),
        sessionId,
    };
}
