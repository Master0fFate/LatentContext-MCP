import { createRequire } from "node:module";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import {
    storeMemory,
    compressMemory,
    forgetMemory,
    getMemoryStatus,
    getCoreMemory,
    getCurrentSessionMemory,
    archiveWorkingMemory,
    clearWorkingMemory,
} from "./memory-manager.js";
import { assembleContext, type RetrievalFilters } from "./context-assembler.js";
import { getGraphSchema } from "./knowledge-graph.js";
import { log } from "./logger.js";
import { startSession, getSessionInfo } from "./session.js";

const packageVersion = createRequire(import.meta.url)("../package.json").version as string;

export const SERVER_NAME = "latentcontext-mcp";
export const SERVER_TITLE = "LatentContext";

export const MEMORY_TYPES = ["fact", "preference", "event", "summary", "core"] as const;
export const COMPRESS_SCOPES = ["working", "session", "epoch"] as const;
export const FORGET_ACTIONS = ["deprecate", "correct", "delete"] as const;

/** Minimum number of words a stored memory must contain to be worth keeping. */
export const MINIMUM_MEMORY_WORDS = 10;

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Guidance sent to the host during initialization. Hosts that support
 * `instructions` add it to the model's context, which is the most portable
 * way to teach every harness how the tools fit together.
 */
export const SERVER_INSTRUCTIONS = [
    "LatentContext keeps session-scoped working notes, compressed summaries, and a small knowledge graph in local SQLite.",
    "Call session_start once when a new piece of work begins; the previous session is archived automatically.",
    "Use memory_store to record decisions, facts, preferences, and events as self-contained notes of at least 10 words (25 or more is better).",
    "Use memory_retrieve before relying on earlier context; results are limited to the active session and the requested token budget.",
    "Use memory_compress when working memory grows large, memory_forget to deprecate, correct, or delete a note, and memory_status to inspect storage.",
].join(" ");

// ---------------------------------------------------------------------------
// Schemas
//
// Input schemas deliberately use only widely supported JSON Schema keywords
// (type, enum, properties, required, additionalProperties, minimum/maximum,
// minLength/minItems, default, description). Conditional keywords such as
// if/then or oneOf and string formats are rejected by some hosts, so those
// rules are enforced in the handlers with explicit error messages instead.
// ---------------------------------------------------------------------------

const memoryTypeSchema = z.enum(MEMORY_TYPES);

const emptyInputSchema = z.object({}).strict();

const storeInputSchema = z.object({
    content: z.string().min(1).describe("Self-contained text (10+ words)."),
    memory_type: memoryTypeSchema.describe("Memory category."),
    confidence: z.number().min(0).max(1).default(1).describe("Confidence from 0 to 1."),
    entities: z.array(z.string().min(1)).optional().describe("Related entity labels."),
}).strict();

const retrievalFiltersSchema = z.object({
    memory_types: z.array(memoryTypeSchema).min(1).optional().describe("Only return these memory types."),
    after: z.string().optional().describe("Only memories created at or after this ISO 8601 datetime."),
    before: z.string().optional().describe("Only memories created at or before this ISO 8601 datetime."),
    min_confidence: z.number().min(0).max(1).optional().describe("Minimum confidence from 0 to 1."),
}).strict();

const retrieveInputSchema = z.object({
    query: z.string().min(1).describe("Search query."),
    token_budget: z.number().int().min(1).optional().describe("Maximum response tokens."),
    filters: retrievalFiltersSchema.optional().describe("Optional retrieval filters."),
}).strict();

const compressInputSchema = z.object({
    scope: z.enum(COMPRESS_SCOPES).describe("Compression scope."),
}).strict();

const forgetInputSchema = z.object({
    memory_id: z.string().min(1).describe("Memory ID (UUID) returned by memory_store."),
    action: z.enum(FORGET_ACTIONS).describe("Update action."),
    correction: z.string().min(1).optional().describe("Replacement text; required when action is 'correct'."),
}).strict();

const sessionStartOutputSchema = z.object({
    sessionId: z.string(),
    startedAt: z.string(),
    previousSessionId: z.string().nullable(),
    previousSessionArchived: z.boolean(),
    archiveSummary: z.string().nullable(),
});

const storeOutputSchema = z.object({
    memoryId: z.string(),
    memoryType: memoryTypeSchema,
    tier: z.number().int(),
    sessionId: z.string().nullable(),
    factsStored: z.number().int(),
    entitiesCreated: z.array(z.string()),
    vectorIndexed: z.boolean(),
});

const retrieveOutputSchema = z.object({
    sessionId: z.string().nullable(),
    totalTokens: z.number().int(),
    budgetUsed: z.number().int(),
    budgetRemaining: z.number().int(),
    candidatesConsidered: z.number().int(),
    candidatesSelected: z.number().int(),
    sourceCounts: z.object({
        working: z.number().int().optional(),
        current_session: z.number().int().optional(),
    }),
});

const tierStatusSchema = z.object({ count: z.number().int(), tokenEstimate: z.number().int() });

const statusOutputSchema = z.object({
    sessionId: z.string().nullable(),
    startedAt: z.string().nullable(),
    tiers: z.object({
        tier0: tierStatusSchema,
        tier1: tierStatusSchema,
        tier2: tierStatusSchema,
        tier3: tierStatusSchema,
    }),
    knowledgeGraph: z.object({ entities: z.number().int(), relations: z.number().int() }),
    vectorStore: z.object({ count: z.number().int() }),
    totalTokensStored: z.number().int(),
});

// ---------------------------------------------------------------------------
// Validation helpers for rules JSON Schema cannot express portably
// ---------------------------------------------------------------------------

function isIsoDateTime(value: string): boolean {
    const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$/.exec(value);
    if (!match) return false;

    const [year, month, day, hour, minute, second] = match.slice(1, 7).map(Number);
    const timezoneHour = match[7] === undefined ? 0 : Number(match[7]);
    const timezoneMinute = match[8] === undefined ? 0 : Number(match[8]);
    const isLeapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
    const daysInMonth = [31, isLeapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    return month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth[month - 1] &&
        hour <= 23 && minute <= 59 && second <= 59 && timezoneHour <= 23 && timezoneMinute <= 59 &&
        !Number.isNaN(Date.parse(value));
}

function countWords(text: string): number {
    return text.trim().split(/\s+/).filter(Boolean).length;
}

function textResult(text: string, structuredContent?: Record<string, unknown>): CallToolResult {
    return {
        content: [{ type: "text", text }],
        ...(structuredContent ? { structuredContent } : {}),
    };
}

function errorResult(text: string): CallToolResult {
    return { content: [{ type: "text", text }], isError: true };
}

/**
 * Wrap a tool handler so failures are recorded in the server log and returned
 * to the model as an `isError` result with a stable "Error: ..." prefix.
 */
function guarded<Schema extends z.ZodType>(
    toolName: string,
    _inputSchema: Schema,
    handler: (args: z.output<Schema>) => Promise<CallToolResult>
): (args: z.output<Schema>) => Promise<CallToolResult> {
    return async (args) => {
        try {
            return await handler(args);
        } catch (error) {
            log("error", `Tool ${toolName} failed`, error);
            return errorResult(`Error: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
}

function parseRetrievalFilters(
    input: z.infer<typeof retrievalFiltersSchema> | undefined
): { filters?: RetrievalFilters; error?: string } {
    if (!input) return {};

    for (const key of ["after", "before"] as const) {
        const value = input[key];
        if (value !== undefined && !isIsoDateTime(value)) {
            return { error: `Error: '${key}' must be an ISO datetime such as 2025-01-31T12:00:00Z.` };
        }
    }
    if (input.after && input.before && Date.parse(input.after) > Date.parse(input.before)) {
        return { error: "Error: 'after' must not be later than 'before'." };
    }

    return {
        filters: {
            sourceTypes: input.memory_types ? [...input.memory_types] : undefined,
            after: input.after,
            before: input.before,
            minConfidence: input.min_confidence,
        },
    };
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

export function createServer(): McpServer {
    const server = new McpServer(
        { name: SERVER_NAME, version: packageVersion, title: SERVER_TITLE },
        {
            instructions: SERVER_INSTRUCTIONS,
            capabilities: { logging: {} },
        }
    );

    // ── Tools ──────────────────────────────────────────────────────────────

    server.registerTool(
        "session_start",
        {
            title: "Start Session",
            description: "Start a fresh isolated session before session memory work.",
            inputSchema: emptyInputSchema,
            outputSchema: sessionStartOutputSchema,
            annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
        },
        guarded("session_start", emptyInputSchema, async () => {
            const result = await startSession(archiveWorkingMemory);
            // Clear ALL working memory to guarantee complete session isolation.
            clearWorkingMemory();
            log("info", `Session started: ${result.sessionId}`);

            return textResult(
                `New session started: ${result.sessionId}\nStarted at: ${result.startedAt}`,
                {
                    sessionId: result.sessionId,
                    startedAt: result.startedAt,
                    previousSessionId: result.previousSessionId,
                    previousSessionArchived: result.previousSessionArchived,
                    archiveSummary: result.archiveSummary,
                }
            );
        })
    );

    server.registerTool(
        "memory_store",
        {
            title: "Store Memory",
            description: "Store session or persistent memory by type.",
            inputSchema: storeInputSchema,
            outputSchema: storeOutputSchema,
            annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
        },
        guarded("memory_store", storeInputSchema, async ({ content, memory_type: memoryType, confidence, entities: rawEntities }) => {
            if (content.trim().length === 0) {
                return errorResult("Error: 'content' must be a non-empty string.");
            }
            const entities = (rawEntities ?? []).map((entity) => entity.trim());
            if (entities.some((entity) => entity.length === 0)) {
                return errorResult("Error: 'entities' must be an array of non-empty strings.");
            }

            // Reject entries too terse to be useful later. A note like "Fixed
            // audio issue" forces the work to be re-derived, which is exactly
            // what memory exists to avoid.
            const wordCount = countWords(content);
            if (wordCount < MINIMUM_MEMORY_WORDS) {
                return errorResult(
                    `Error: memory content must contain at least ${MINIMUM_MEMORY_WORDS} words (received ${wordCount}).`
                );
            }

            const result = await storeMemory(content, memoryType, confidence, entities);
            const lines = [
                `Stored as ${result.memoryType} (Tier ${result.tier})`,
                `ID: ${result.memoryId}`,
                result.sessionId ? `Session: ${result.sessionId.substring(0, 8)}` : null,
                result.factsStored > 0 ? `Facts stored: ${result.factsStored}` : null,
                result.entitiesCreated.length > 0 ? `Entities: ${result.entitiesCreated.join(", ")}` : null,
                result.vectorId ? "Vector indexed" : null,
            ].filter((line): line is string => line !== null);

            return textResult(lines.join("\n"), {
                memoryId: result.memoryId,
                memoryType: result.memoryType,
                tier: result.tier,
                sessionId: result.sessionId,
                factsStored: result.factsStored,
                entitiesCreated: result.entitiesCreated,
                vectorIndexed: result.vectorId !== null,
            });
        })
    );

    server.registerTool(
        "memory_retrieve",
        {
            title: "Retrieve Memory",
            description: "Search working memory and summaries in the active session.",
            inputSchema: retrieveInputSchema,
            outputSchema: retrieveOutputSchema,
            annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
        },
        guarded("memory_retrieve", retrieveInputSchema, async ({ query, token_budget: tokenBudget, filters: rawFilters }) => {
            if (query.trim().length === 0) {
                return errorResult("Error: 'query' must be a non-empty string.");
            }
            const parsedFilters = parseRetrievalFilters(rawFilters);
            if (parsedFilters.error) return errorResult(parsedFilters.error);

            const result = await assembleContext(query, tokenBudget, parsedFilters.filters);
            return textResult(result.text, {
                sessionId: result.sessionId,
                totalTokens: result.totalTokens,
                budgetUsed: result.budgetUsed,
                budgetRemaining: result.budgetRemaining,
                candidatesConsidered: result.candidatesConsidered,
                candidatesSelected: result.candidatesSelected,
                sourceCounts: result.sourceCounts,
            });
        })
    );

    server.registerTool(
        "memory_compress",
        {
            title: "Compress Memory",
            description: "Compress active-session memory at a selected scope.",
            inputSchema: compressInputSchema,
            annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
        },
        guarded("memory_compress", compressInputSchema, async ({ scope }) => textResult(await compressMemory(scope)))
    );

    server.registerTool(
        "memory_forget",
        {
            title: "Forget Memory",
            description: "Deprecate, correct, or delete a memory.",
            inputSchema: forgetInputSchema,
            annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
        },
        guarded("memory_forget", forgetInputSchema, async ({ memory_id: memoryId, action, correction }) => {
            if (!UUID_PATTERN.test(memoryId)) {
                return errorResult("Error: 'memory_id' must be a UUID.");
            }
            if (correction !== undefined && correction.trim().length === 0) {
                return errorResult("Error: 'correction' must be a non-empty string.");
            }
            if (action === "correct" && correction === undefined) {
                return errorResult("Error: 'correction' is required when action is 'correct'.");
            }
            if (action !== "correct" && correction !== undefined) {
                return errorResult("Error: 'correction' is only valid when action is 'correct'.");
            }
            return textResult(await forgetMemory(memoryId, action, correction));
        })
    );

    server.registerTool(
        "memory_status",
        {
            title: "Memory Status",
            description: "Show memory storage and session status.",
            inputSchema: emptyInputSchema,
            outputSchema: statusOutputSchema,
            annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
        },
        guarded("memory_status", emptyInputSchema, async () => {
            const status = getMemoryStatus();
            const sessionInfo = getSessionInfo();
            const lines = [
                "=== Memory Status ===",
                `Session: ${sessionInfo ? `${sessionInfo.sessionId.substring(0, 8)}... (started ${sessionInfo.startedAt})` : "No active session — call session_start first!"}`,
                "",
                `Tier 0 (Working):  ${status.tiers.tier0.count} entries, ~${status.tiers.tier0.tokenEstimate} tokens`,
                `Tier 1 (Session):  ${status.tiers.tier1.count} entries, ~${status.tiers.tier1.tokenEstimate} tokens`,
                `Tier 2 (Epoch):    ${status.tiers.tier2.count} entries, ~${status.tiers.tier2.tokenEstimate} tokens`,
                `Tier 3 (Core):     ${status.tiers.tier3.count} entries, ~${status.tiers.tier3.tokenEstimate} tokens`,
                `Knowledge Graph:   ${status.knowledgeGraph.entities} entities, ${status.knowledgeGraph.relations} relations`,
                `Vector Store:      ${status.vectorStore.count} vectors`,
                `Total Tokens:      ~${status.totalTokensStored}`,
            ];
            return textResult(lines.join("\n"), {
                sessionId: sessionInfo?.sessionId ?? null,
                startedAt: sessionInfo?.startedAt ?? null,
                tiers: status.tiers,
                knowledgeGraph: status.knowledgeGraph,
                vectorStore: status.vectorStore,
                totalTokensStored: status.totalTokensStored,
            });
        })
    );

    // ── Resources ──────────────────────────────────────────────────────────

    server.registerResource(
        "core_memory",
        "memory://core",
        { title: "Core Memory", description: "Persistent core memories.", mimeType: "text/plain" },
        async (uri) => ({ contents: [{ uri: uri.href, text: getCoreMemory(), mimeType: "text/plain" }] })
    );

    server.registerResource(
        "session_memory",
        "memory://session/current",
        { title: "Current Session Memory", description: "Working memory for the active session.", mimeType: "text/plain" },
        async (uri) => ({ contents: [{ uri: uri.href, text: getCurrentSessionMemory(), mimeType: "text/plain" }] })
    );

    server.registerResource(
        "graph_schema",
        "memory://graph/schema",
        { title: "Knowledge Graph Schema", description: "Entity types and relation predicates.", mimeType: "application/json" },
        async (uri) => ({
            contents: [{ uri: uri.href, text: JSON.stringify(getGraphSchema(), null, 2), mimeType: "application/json" }],
        })
    );

    server.registerResource(
        "memory_stats",
        "memory://stats",
        { title: "Memory Statistics", description: "Memory storage statistics.", mimeType: "application/json" },
        async (uri) => ({
            contents: [{ uri: uri.href, text: JSON.stringify(getMemoryStatus(), null, 2), mimeType: "application/json" }],
        })
    );

    // ── Prompts ────────────────────────────────────────────────────────────

    server.registerPrompt(
        "extract_facts",
        {
            title: "Extract Facts",
            description: "Extract fact triples from text.",
            argsSchema: { text: z.string().min(1).describe("Source text.") },
        },
        ({ text }) => {
            if (text.trim().length === 0) {
                throw new Error("Prompt 'extract_facts' requires a non-empty 'text' argument.");
            }
            return {
                description: "Extract structured facts from text",
                messages: [{
                    role: "user",
                    content: {
                        type: "text",
                        text: `Extract all factual statements from the following text as structured triples. Output a JSON array where each element has: "subject", "predicate", "object", "subject_type", "object_type", "confidence" (0.0-1.0).

Use clear, normalized predicates like: located_in, works_at, is_a, has, prefers, knows, wants_to, created, uses, born_in, member_of, etc.

Text:
${text}

Output only the JSON array, no other text.`,
                    },
                }],
            };
        }
    );

    server.registerPrompt(
        "compress_session",
        {
            title: "Compress Session",
            description: "Summarize working memory.",
            argsSchema: { working_memory: z.string().min(1).describe("Working memory text.") },
        },
        ({ working_memory: workingMemory }) => {
            if (workingMemory.trim().length === 0) {
                throw new Error("Prompt 'compress_session' requires a non-empty 'working_memory' argument.");
            }
            return {
                description: "Compress working memory into session summary",
                messages: [{
                    role: "user",
                    content: {
                        type: "text",
                        text: `Compress the following conversation working memory into a concise summary of ~200 tokens. Preserve:
1. Key decisions made
2. Important facts learned about the user
3. Unresolved questions or pending items
4. Action items or next steps
5. Emotional context if notable

Aggressively compress redundant dialogue, greetings, and filler. Keep only high-information-density content.

Working Memory:
${workingMemory}

Output only the compressed summary.`,
                    },
                }],
            };
        }
    );

    server.registerPrompt(
        "consolidate_epoch",
        {
            title: "Consolidate Epoch",
            description: "Consolidate session summaries.",
            argsSchema: { session_summaries: z.string().min(1).describe("Session summaries.") },
        },
        ({ session_summaries: summaries }) => {
            if (summaries.trim().length === 0) {
                throw new Error("Prompt 'consolidate_epoch' requires a non-empty 'session_summaries' argument.");
            }
            return {
                description: "Consolidate session summaries into epoch summary",
                messages: [{
                    role: "user",
                    content: {
                        type: "text",
                        text: `Merge the following session summaries into a single epoch summary of ~100 tokens. Focus on:
1. Recurring themes and patterns
2. Evolving user preferences
3. Long-term goals and progress
4. Significant milestones or decisions

Session Summaries:
${summaries}

Output only the consolidated epoch summary.`,
                    },
                }],
            };
        }
    );

    return server;
}
