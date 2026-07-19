import { createRequire } from "node:module";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
    ListResourcesRequestSchema,
    ReadResourceRequestSchema,
    ListPromptsRequestSchema,
    GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import {
    storeMemory,
    compressMemory,
    forgetMemory,
    getMemoryStatus,
    getCoreMemory,
    getCurrentSessionMemory,
    archiveWorkingMemory,
    clearWorkingMemory,
    type MemoryType,
    type CompressScope,
    type ForgetAction,
} from "./memory-manager.js";
import { assembleContext, type RetrievalFilters } from "./context-assembler.js";
import {
    queryEntity,
    queryByPredicate,
    serializeFacts,
    getGraphSchema,
} from "./knowledge-graph.js";
import {
    startSession,
    getSessionInfo,
    getCurrentSessionIdOrNull,
} from "./session.js";

const packageVersion = createRequire(import.meta.url)("../package.json").version as string;
const MEMORY_TYPES = ["fact", "preference", "event", "summary", "core"] as const;
const COMPRESS_SCOPES = ["working", "session", "epoch"] as const;
const FORGET_ACTIONS = ["deprecate", "correct", "delete"] as const;

function isMemoryType(value: unknown): value is MemoryType {
    return typeof value === "string" && MEMORY_TYPES.includes(value as MemoryType);
}

function isCompressScope(value: unknown): value is CompressScope {
    return typeof value === "string" && COMPRESS_SCOPES.includes(value as CompressScope);
}

function isForgetAction(value: unknown): value is ForgetAction {
    return typeof value === "string" && FORGET_ACTIONS.includes(value as ForgetAction);
}

const UUID_PATTERN = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$";
const ISO_DATETIME_PATTERN = "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|[+-]\\d{2}:\\d{2})$";

const PROMPT_ARGUMENTS: Record<string, readonly string[]> = {
    extract_facts: ["text"],
    compress_session: ["working_memory"],
    consolidate_epoch: ["session_summaries"],
};

const TOOL_ARGUMENTS: Record<string, readonly string[]> = {
    session_start: [],
    memory_store: ["content", "memory_type", "confidence", "entities"],
    memory_retrieve: ["query", "token_budget", "filters"],
    memory_compress: ["scope"],
    memory_forget: ["memory_id", "action", "correction"],
    memory_status: [],
};

function parseArguments(value: unknown, allowed: readonly string[]): {
    value?: Record<string, unknown>;
    error?: string;
} {
    if (value === undefined) return { value: {} };
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return { error: "Error: arguments must be an object." };
    }

    const input = value as Record<string, unknown>;
    for (const key of Object.keys(input)) {
        if (!allowed.includes(key)) return { error: `Error: unsupported argument '${key}'.` };
    }
    return { value: input };
}

function parseEntities(value: unknown): { entities?: string[]; error?: string } {
    if (value === undefined) return {};
    if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
        return { error: "Error: 'entities' must be an array of non-empty strings." };
    }
    return { entities: value.map((item) => (item as string).trim()) };
}

function isIsoDateTime(value: string): boolean {
    const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$/.exec(value);
    if (!match) return false;

    const [year, month, day, hour, minute, second] = match.slice(1, 7).map(Number);
    const timezoneHour = match[7] === undefined ? 0 : Number(match[7]);
    const timezoneMinute = match[8] === undefined ? 0 : Number(match[8]);
    const daysInMonth = [31, year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    return month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth[month - 1] &&
        hour <= 23 && minute <= 59 && second <= 59 && timezoneHour <= 23 && timezoneMinute <= 59 &&
        !Number.isNaN(Date.parse(value));
}

function errorText(text: string) {
    return {
        content: [{ type: "text" as const, text }],
        isError: true,
    };
}

/** Validate every documented retrieval filter instead of silently ignoring it. */
function parseRetrievalFilters(value: unknown): { filters?: RetrievalFilters; error?: string } {
    if (value === undefined) return {};
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return { error: "Error: 'filters' must be an object." };
    }

    const input = value as Record<string, unknown>;
    const supported = new Set(["memory_types", "after", "before", "min_confidence"]);
    for (const key of Object.keys(input)) {
        if (!supported.has(key)) return { error: `Error: unsupported retrieval filter '${key}'.` };
    }

    let sourceTypes: string[] | undefined;
    if (input.memory_types !== undefined) {
        if (!Array.isArray(input.memory_types) || input.memory_types.length === 0 ||
            input.memory_types.some((type) => typeof type !== "string" || !isMemoryType(type))) {
            return { error: `Error: 'memory_types' must be a non-empty array containing only: ${MEMORY_TYPES.join(", ")}.` };
        }
        sourceTypes = input.memory_types as string[];
    }

    for (const key of ["after", "before"] as const) {
        const date = input[key];
        if (date !== undefined && (typeof date !== "string" || !isIsoDateTime(date))) {
            return { error: `Error: '${key}' must be an ISO datetime.` };
        }
    }
    if (input.after && input.before && Date.parse(input.after as string) > Date.parse(input.before as string)) {
        return { error: "Error: 'after' must not be later than 'before'." };
    }

    const minConfidence = input.min_confidence;
    if (minConfidence !== undefined &&
        (typeof minConfidence !== "number" || !Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1)) {
        return { error: "Error: 'min_confidence' must be a number between 0 and 1." };
    }

    return {
        filters: {
            sourceTypes,
            after: input.after as string | undefined,
            before: input.before as string | undefined,
            minConfidence: minConfidence as number | undefined,
        },
    };
}

// ---------------------------------------------------------------------------
// Create the MCP Server
// ---------------------------------------------------------------------------

export function createServer(): Server {
    const server = new Server(
        {
            name: "latentcontext-mcp",
            version: packageVersion,
        },
        {
            capabilities: {
                tools: {},
                resources: {},
                prompts: {},
            },
        }
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // TOOLS
    // ═══════════════════════════════════════════════════════════════════════════

    server.setRequestHandler(ListToolsRequestSchema, async () => ({
        tools: [
            {
                name: "session_start",
                description: "Start a fresh isolated session before session memory work.",
                inputSchema: { type: "object" as const, properties: {}, additionalProperties: false },
            },
            {
                name: "memory_store",
                description: "Store session or persistent memory by type.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        content: { type: "string", minLength: 1, pattern: "^(?:\\s*\\S+\\s+){9}\\S+\\s*$", description: "Self-contained text (10+ words)." },
                        memory_type: { type: "string", enum: [...MEMORY_TYPES], description: "Memory category." },
                        confidence: { type: "number", minimum: 0, maximum: 1, default: 1, description: "Confidence." },
                        entities: { type: "array", items: { type: "string", minLength: 1, pattern: "\\S" }, description: "Related entities." },
                    },
                    required: ["content", "memory_type"],
                    additionalProperties: false,
                },
            },
            {
                name: "memory_retrieve",
                description: "Search working memory and summaries in the active session.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        query: { type: "string", minLength: 1, pattern: "\\S", description: "Search query." },
                        token_budget: { type: "integer", minimum: 1, description: "Maximum response tokens." },
                        filters: {
                            type: "object",
                            description: "Optional retrieval filters.",
                            properties: {
                                memory_types: { type: "array", minItems: 1, items: { type: "string", enum: [...MEMORY_TYPES] } },
                                after: { type: "string", format: "date-time", pattern: ISO_DATETIME_PATTERN },
                                before: { type: "string", format: "date-time", pattern: ISO_DATETIME_PATTERN },
                                min_confidence: { type: "number", minimum: 0, maximum: 1 },
                            },
                            additionalProperties: false,
                        },
                    },
                    required: ["query"],
                    additionalProperties: false,
                },
            },
            {
                name: "memory_compress",
                description: "Compress active-session memory at a selected scope.",
                inputSchema: {
                    type: "object" as const,
                    properties: { scope: { type: "string", enum: [...COMPRESS_SCOPES], description: "Compression scope." } },
                    required: ["scope"],
                    additionalProperties: false,
                },
            },
            {
                name: "memory_forget",
                description: "Deprecate, correct, or delete a memory.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        memory_id: { type: "string", pattern: UUID_PATTERN, description: "Memory ID." },
                        action: { type: "string", enum: [...FORGET_ACTIONS], description: "Update action." },
                        correction: { type: "string", minLength: 1, pattern: "\\S", description: "Replacement text for correction." },
                    },
                    required: ["memory_id", "action"],
                    allOf: [{
                        if: { properties: { action: { const: "correct" } }, required: ["action"] },
                        then: { required: ["correction"] },
                        else: { not: { required: ["correction"] } },
                    }],
                    additionalProperties: false,
                },
            },
            {
                name: "memory_status",
                description: "Show memory storage and session status.",
                inputSchema: { type: "object" as const, properties: {}, additionalProperties: false },
            },
        ],
    }));

    // ── Tool call handler ──
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const { name, arguments: initialArgs } = request.params;
        let args = initialArgs;

        try {
            const allowedArguments = TOOL_ARGUMENTS[name];
            if (allowedArguments) {
                const parsed = parseArguments(args, allowedArguments);
                if (parsed.error) return errorText(parsed.error);
                args = parsed.value!;
            }

            switch (name) {
                case "session_start": {
                    const result = await startSession(async (oldSessionId) => {
                        return archiveWorkingMemory(oldSessionId);
                    });

                    // Clear ALL working memory to guarantee complete session isolation.
                    // No data from previous sessions should leak into the new one.
                    clearWorkingMemory();

                    return {
                        content: [{
                            type: "text" as const,
                            text: `New session started: ${result.sessionId}\nStarted at: ${result.startedAt}`,
                        }],
                    };
                }

                case "memory_store": {
                    const content = args?.content;
                    const rawMemoryType = args?.memory_type;
                    const confidence = args?.confidence ?? 1.0;
                    const parsedEntities = parseEntities(args?.entities);

                    if (typeof content !== "string" || content.trim().length === 0) {
                        return errorText("Error: 'content' must be a non-empty string.");
                    }
                    if (parsedEntities.error) return errorText(parsedEntities.error);
                    const entities = parsedEntities.entities ?? [];

                    if (!isMemoryType(rawMemoryType)) {
                        return errorText(
                            `Error: 'memory_type' must be one of: ${MEMORY_TYPES.join(", ")}.`
                        );
                    }

                    if (
                        typeof confidence !== "number" ||
                        !Number.isFinite(confidence) ||
                        confidence < 0 ||
                        confidence > 1
                    ) {
                        return errorText("Error: 'confidence' must be a number between 0 and 1.");
                    }

                    // ── Content quality enforcement ──
                    // Reject entries that are too terse to be useful in future retrieval.
                    // A single sentence like "Fixed audio issue" wastes storage and forces
                    // re-derivation later — the whole point of memory is to AVOID that.
                    const wordCount = content.trim().split(/\s+/).length;
                    if (wordCount < 10) {
                        return {
                            content: [{
                                type: "text" as const,
                                text: `Error: memory content must contain at least 10 words (received ${wordCount}).`
                            }],
                            isError: true,
                        };
                    }

                    const result = await storeMemory(content, rawMemoryType, confidence, entities);

                    const response = [
                        `Stored as ${result.memoryType} (Tier ${result.tier})`,
                        `ID: ${result.memoryId}`,
                        result.sessionId ? `Session: ${result.sessionId.substring(0, 8)}` : null,
                        result.factsStored > 0 ? `Facts stored: ${result.factsStored}` : null,
                        result.entitiesCreated.length > 0
                            ? `Entities: ${result.entitiesCreated.join(", ")}`
                            : null,
                        result.vectorId ? "Vector indexed" : null,
                    ]
                        .filter(Boolean)
                        .join("\n");

                    return { content: [{ type: "text" as const, text: response }] };
                }

                case "memory_retrieve": {
                    const query = args?.query;
                    const tokenBudget = args?.token_budget;
                    const rawFilters = args?.filters;

                    if (typeof query !== "string" || query.trim().length === 0) {
                        return errorText("Error: 'query' must be a non-empty string.");
                    }

                    if (
                        tokenBudget !== undefined &&
                        (typeof tokenBudget !== "number" || !Number.isInteger(tokenBudget) || tokenBudget <= 0)
                    ) {
                        return errorText("Error: 'token_budget' must be a positive integer.");
                    }

                    const parsedFilters = parseRetrievalFilters(rawFilters);
                    if (parsedFilters.error) {
                        return errorText(parsedFilters.error);
                    }

                    const result = await assembleContext(query, tokenBudget as number | undefined, parsedFilters.filters);

                    return { content: [{ type: "text" as const, text: result.text }] };
                }


                case "memory_compress": {
                    const rawScope = args?.scope;
                    if (!isCompressScope(rawScope)) {
                        return errorText(
                            `Error: 'scope' must be one of: ${COMPRESS_SCOPES.join(", ")}.`
                        );
                    }
                    const scope = rawScope;
                    const result = await compressMemory(scope);
                    return { content: [{ type: "text" as const, text: result }] };
                }

                case "memory_forget": {
                    const memoryId = args?.memory_id;
                    const rawAction = args?.action;
                    const correction = args?.correction;

                    if (typeof memoryId !== "string" || !new RegExp(UUID_PATTERN, "i").test(memoryId)) {
                        return errorText("Error: 'memory_id' must be a UUID.");
                    }

                    if (!isForgetAction(rawAction)) {
                        return errorText(
                            `Error: 'action' must be one of: ${FORGET_ACTIONS.join(", ")}.`
                        );
                    }

                    if (correction !== undefined && (typeof correction !== "string" || correction.trim().length === 0)) {
                        return errorText("Error: 'correction' must be a non-empty string.");
                    }
                    if (rawAction === "correct" && typeof correction !== "string") {
                        return errorText("Error: 'correction' is required when action is 'correct'.");
                    }
                    if (rawAction !== "correct" && correction !== undefined) {
                        return errorText("Error: 'correction' is only valid when action is 'correct'.");
                    }

                    const result = await forgetMemory(memoryId, rawAction, correction);
                    return { content: [{ type: "text" as const, text: result }] };
                }

                case "memory_status": {
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
                    return { content: [{ type: "text" as const, text: lines.join("\n") }] };
                }

                default:
                    return {
                        content: [{ type: "text" as const, text: `Unknown tool: ${name}` }],
                        isError: true,
                    };
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            return {
                content: [{ type: "text" as const, text: `Error: ${message}` }],
                isError: true,
            };
        }
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // RESOURCES
    // ═══════════════════════════════════════════════════════════════════════════

    server.setRequestHandler(ListResourcesRequestSchema, async () => ({
        resources: [
            {
                uri: "memory://core",
                name: "Core Memory",
                description: "Persistent core memories.",
                mimeType: "text/plain",
            },
            {
                uri: "memory://session/current",
                name: "Current Session Memory",
                description: "Working memory for the active session.",
                mimeType: "text/plain",
            },
            {
                uri: "memory://graph/schema",
                name: "Knowledge Graph Schema",
                description: "Entity types and relation predicates.",
                mimeType: "application/json",
            },
            {
                uri: "memory://stats",
                name: "Memory Statistics",
                description: "Memory storage statistics.",
                mimeType: "application/json",
            },
        ],
    }));

    server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
        const { uri } = request.params;

        switch (uri) {
            case "memory://core": {
                const core = getCoreMemory();
                return {
                    contents: [{ uri, text: core, mimeType: "text/plain" }],
                };
            }

            case "memory://session/current": {
                const session = getCurrentSessionMemory();
                return {
                    contents: [{ uri, text: session, mimeType: "text/plain" }],
                };
            }

            case "memory://graph/schema": {
                const schema = getGraphSchema();
                return {
                    contents: [
                        {
                            uri,
                            text: JSON.stringify(schema, null, 2),
                            mimeType: "application/json",
                        },
                    ],
                };
            }

            case "memory://stats": {
                const status = getMemoryStatus();
                return {
                    contents: [
                        {
                            uri,
                            text: JSON.stringify(status, null, 2),
                            mimeType: "application/json",
                        },
                    ],
                };
            }

            default:
                throw new Error(`Unknown resource: ${uri}`);
        }
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // PROMPTS
    // ═══════════════════════════════════════════════════════════════════════════

    server.setRequestHandler(ListPromptsRequestSchema, async () => ({
        prompts: [
            {
                name: "extract_facts",
                description: "Extract fact triples from text.",
                arguments: [
                    {
                        name: "text",
                        description: "Source text.",
                        required: true,
                    },
                ],
            },
            {
                name: "compress_session",
                description: "Summarize working memory.",
                arguments: [
                    {
                        name: "working_memory",
                        description: "Working memory text.",
                        required: true,
                    },
                ],
            },
            {
                name: "consolidate_epoch",
                description: "Consolidate session summaries.",
                arguments: [
                    {
                        name: "session_summaries",
                        description: "Session summaries.",
                        required: true,
                    },
                ],
            },
        ],
    }));

    server.setRequestHandler(GetPromptRequestSchema, async (request) => {
        const { name, arguments: initialArgs } = request.params;
        let args = initialArgs;
        const allowedArguments = PROMPT_ARGUMENTS[name];
        if (allowedArguments) {
            const parsed = parseArguments(args, allowedArguments);
            if (parsed.error) throw new Error(parsed.error);
            const promptArgs = parsed.value!;
            if (Object.values(promptArgs).some((value) => typeof value !== "string")) {
                throw new Error("Prompt arguments must be strings.");
            }
            args = promptArgs as Record<string, string>;
        }

        switch (name) {
            case "extract_facts": {
                const text = args?.text;
                if (typeof text !== "string" || text.trim().length === 0) {
                    throw new Error("Prompt 'extract_facts' requires a non-empty 'text' argument.");
                }
                return {
                    description: "Extract structured facts from text",
                    messages: [
                        {
                            role: "user" as const,
                            content: {
                                type: "text" as const,
                                text: `Extract all factual statements from the following text as structured triples. Output a JSON array where each element has: "subject", "predicate", "object", "subject_type", "object_type", "confidence" (0.0-1.0).

Use clear, normalized predicates like: located_in, works_at, is_a, has, prefers, knows, wants_to, created, uses, born_in, member_of, etc.

Text:
${text}

Output only the JSON array, no other text.`,
                            },
                        },
                    ],
                };
            }

            case "compress_session": {
                const workingMemory = args?.working_memory;
                if (typeof workingMemory !== "string" || workingMemory.trim().length === 0) {
                    throw new Error("Prompt 'compress_session' requires a non-empty 'working_memory' argument.");
                }
                return {
                    description: "Compress working memory into session summary",
                    messages: [
                        {
                            role: "user" as const,
                            content: {
                                type: "text" as const,
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
                        },
                    ],
                };
            }

            case "consolidate_epoch": {
                const summaries = args?.session_summaries;
                if (typeof summaries !== "string" || summaries.trim().length === 0) {
                    throw new Error("Prompt 'consolidate_epoch' requires a non-empty 'session_summaries' argument.");
                }
                return {
                    description: "Consolidate session summaries into epoch summary",
                    messages: [
                        {
                            role: "user" as const,
                            content: {
                                type: "text" as const,
                                text: `Merge the following session summaries into a single epoch summary of ~100 tokens. Focus on:
1. Recurring themes and patterns
2. Evolving user preferences
3. Long-term goals and progress
4. Significant milestones or decisions

Session Summaries:
${summaries}

Output only the consolidated epoch summary.`,
                            },
                        },
                    ],
                };
            }

            default:
                throw new Error(`Unknown prompt: ${name}`);
        }
    });

    return server;
}
