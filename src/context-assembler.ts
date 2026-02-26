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
 * Assemble the context payload for a given query within a token budget.
 *
 * SESSION ISOLATION: This function ONLY returns data from the CURRENT session.
 * It does NOT pull from past sessions, the global knowledge graph, or the
 * global vector store. Each session starts with zero memory and the AI fills
 * it during the conversation.
 *
 * Algorithm:
 * 1. Include current session working memory (Tier 0)
 * 2. Include current session Tier 1 summaries (compressed working memory)
 * 3. Score all candidates by composite relevance
 * 4. Deduplicate
 * 5. Greedily fill budget by score
 * 6. Format output with section headers
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

    // ── 1. Current session working memory (Tier 0) ──
    // This is the in-memory buffer of entries stored during THIS session
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

    // ── 2. Current session Tier 1 summaries (compressed working memory) ──
    // These are created when working memory overflows and gets compressed
    if (sessionId) {
        const currentSessionSummaries = getSummariesByTierAndSession(1, sessionId);
        for (const summary of currentSessionSummaries) {
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

    // ── NOTE: The following sources are intentionally EXCLUDED ──
    // - Core memory (Tier 3): Global data from past sessions → NOT included
    // - Vector search: Global vectors from all sessions → NOT included
    // - Knowledge graph: Global entities/relations from all sessions → NOT included
    // - Past session summaries (Tier 1 from other sessions): NOT included
    // - Epoch summaries (Tier 2): NOT included
    //
    // Each session starts COMPLETELY FRESH with zero entries. The AI fills
    // memory during the conversation and only retrieves what was stored in
    // THIS specific session.

    // ── 3. Compute composite scores ──
    const weights = config.ranking;
    for (const candidate of candidates) {
        candidate.score =
            weights.semanticWeight * candidate.similarity +
            weights.recencyWeight * candidate.recency +
            weights.priorityWeight * candidate.priority +
            weights.frequencyWeight * candidate.frequency;
    }

    // ── 4. Deduplicate ──
    const deduped = deduplicate(candidates, weights.dedupSimilarityThreshold);

    // ── 5. Sort by score and greedily fill budget ──
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

    // ── 6. Format output with section headers ──
    const sections: string[] = [];

    // Group selected candidates by source
    const bySource: Record<string, ContextCandidate[]> = {};
    for (const s of selected) {
        if (!bySource[s.source]) bySource[s.source] = [];
        bySource[s.source].push(s);
    }

    // Define section order and labels — current session only
    const sourceOrder = ["working", "current_session"];
    const sourceLabels: Record<string, string> = {
        working: "Current Session",
        current_session: "Current Session Notes",
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
        const sessionLabel = sessionId ? sessionId.substring(0, 20) : "none";
        finalText += `\n\n--- Session: ${sessionLabel} | Sources: ${sourceList} | Tokens: ${budgetUsed}/${budget} ---`;
    } else {
        finalText = "No memories stored in this session yet. This is a fresh session — use memory_store to save important information as you go.";
    }

    const totalTokens = countTokens(finalText);

    return {
        text: finalText,
        totalTokens,
        budgetUsed: budget - remainingBudget,
        budgetRemaining: remainingBudget,
        sourceCounts,
        candidatesConsidered: candidates.length,
        candidatesSelected: selected.length,
        sessionId,
    };
}
