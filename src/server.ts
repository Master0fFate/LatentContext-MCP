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
    type MemoryType,
    type CompressScope,
    type ForgetAction,
} from "./memory-manager.js";
import { assembleContext } from "./context-assembler.js";
import {
    queryEntity,
    queryByPredicate,
    serializeFacts,
    getGraphSchema,
} from "./knowledge-graph.js";

// ---------------------------------------------------------------------------
// Create the MCP Server
// ---------------------------------------------------------------------------

export function createServer(): Server {
    const server = new Server(
        {
            name: "latentcontext-mcp",
            version: "1.0.0",
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
                name: "memory_store",
                description:
                    "Store information in long-term memory. Automatically categorizes, indexes in the knowledge graph, and creates vector embeddings for semantic retrieval.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        content: {
                            type: "string",
                            description: "The information to store.",
                        },
                        memory_type: {
                            type: "string",
                            enum: ["fact", "preference", "event", "summary", "core"],
                            description:
                                "Category: 'fact' for knowledge graph triples, 'preference' for user likes/dislikes, 'event' for timestamped occurrences, 'summary' for session summaries, 'core' for never-evicted critical info.",
                        },
                        confidence: {
                            type: "number",
                            description: "Confidence 0.0-1.0. Lower confidence = evicted first. Default 1.0.",
                        },
                        entities: {
                            type: "array",
                            items: { type: "string" },
                            description:
                                "Key entities this memory relates to, for knowledge graph indexing. First entity is treated as the subject.",
                        },
                    },
                    required: ["content", "memory_type"],
                },
            },
            {
                name: "memory_retrieve",
                description:
                    "Retrieve relevant context for a query. Returns ranked, deduplicated context from all memory subsystems (knowledge graph, vector index, tiered summaries) within a token budget.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        query: {
                            type: "string",
                            description: "The query to retrieve context for.",
                        },
                        token_budget: {
                            type: "integer",
                            description: "Max tokens to return. Default 3000.",
                        },
                        filters: {
                            type: "object",
                            properties: {
                                memory_types: {
                                    type: "array",
                                    items: { type: "string" },
                                    description: "Filter by source types.",
                                },
                                after: {
                                    type: "string",
                                    description: "ISO datetime — only include memories after this time.",
                                },
                                before: {
                                    type: "string",
                                    description: "ISO datetime — only include memories before this time.",
                                },
                                min_confidence: {
                                    type: "number",
                                    description: "Minimum confidence score to include.",
                                },
                            },
                        },
                    },
                    required: ["query"],
                },
            },
            {
                name: "graph_query",
                description:
                    "Query the knowledge graph for facts about entities or relationships. Returns structured triples.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        entity: {
                            type: "string",
                            description: "Entity to look up.",
                        },
                        relation: {
                            type: "string",
                            description: "Optional predicate filter (e.g., 'located_in', 'works_at').",
                        },
                        depth: {
                            type: "integer",
                            description: "Graph traversal hops. Default 1.",
                        },
                    },
                    required: ["entity"],
                },
            },
            {
                name: "memory_compress",
                description:
                    "Compress memory at a given scope. 'working' compresses Tier 0 → Tier 1. 'session' consolidates Tier 1 entries. 'epoch' promotes Tier 1 → Tier 2.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        scope: {
                            type: "string",
                            enum: ["working", "session", "epoch"],
                            description: "Compression scope.",
                        },
                    },
                    required: ["scope"],
                },
            },
            {
                name: "memory_forget",
                description:
                    "Mark a memory as outdated or incorrect. 'deprecate' lowers confidence, 'correct' replaces content, 'delete' removes entirely.",
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        memory_id: {
                            type: "string",
                            description: "ID of the memory to modify.",
                        },
                        action: {
                            type: "string",
                            enum: ["deprecate", "correct", "delete"],
                            description: "Action to take.",
                        },
                        correction: {
                            type: "string",
                            description: "New content if action is 'correct'.",
                        },
                    },
                    required: ["memory_id", "action"],
                },
            },
            {
                name: "memory_status",
                description:
                    "Get storage stats for all memory subsystems: tier counts, token usage, graph size, vector count.",
                inputSchema: {
                    type: "object" as const,
                    properties: {},
                },
            },
        ],
    }));

    // ── Tool call handler ──
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const { name, arguments: args } = request.params;

        try {
            switch (name) {
                case "memory_store": {
                    const content = args?.content as string;
                    const memoryType = (args?.memory_type || "event") as MemoryType;
                    const confidence = (args?.confidence as number) ?? 1.0;
                    const entities = (args?.entities as string[]) || [];

                    if (!content) {
                        return {
                            content: [{ type: "text" as const, text: "Error: 'content' is required." }],
                            isError: true,
                        };
                    }

                    const result = await storeMemory(content, memoryType, confidence, entities);

                    const response = [
                        `Stored as ${result.memoryType} (Tier ${result.tier})`,
                        `ID: ${result.memoryId}`,
                        result.factsStored > 0 ? `Facts stored: ${result.factsStored}` : null,
                        result.entitiesCreated.length > 0
                            ? `Entities: ${result.entitiesCreated.join(", ")}`
                            : null,
                        result.vectorId ? `Vector indexed` : null,
                    ]
                        .filter(Boolean)
                        .join("\n");

                    return { content: [{ type: "text" as const, text: response }] };
                }

                case "memory_retrieve": {
                    const query = args?.query as string;
                    const tokenBudget = args?.token_budget as number | undefined;
                    const filters = args?.filters as
                        | {
                            memory_types?: string[];
                            after?: string;
                            before?: string;
                            min_confidence?: number;
                        }
                        | undefined;

                    if (!query) {
                        return {
                            content: [{ type: "text" as const, text: "Error: 'query' is required." }],
                            isError: true,
                        };
                    }

                    const vectorFilters = filters
                        ? {
                            sourceTypes: filters.memory_types,
                            after: filters.after,
                            before: filters.before,
                            minConfidence: filters.min_confidence,
                        }
                        : undefined;

                    const result = await assembleContext(query, tokenBudget, vectorFilters);

                    return { content: [{ type: "text" as const, text: result.text }] };
                }

                case "graph_query": {
                    const entity = args?.entity as string;
                    const relation = args?.relation as string | undefined;
                    const depth = (args?.depth as number) ?? 1;

                    if (!entity) {
                        return {
                            content: [{ type: "text" as const, text: "Error: 'entity' is required." }],
                            isError: true,
                        };
                    }

                    if (relation) {
                        const facts = queryByPredicate(relation).filter(
                            (f) =>
                                f.subject.toLowerCase().includes(entity.toLowerCase()) ||
                                f.object.toLowerCase().includes(entity.toLowerCase())
                        );
                        return {
                            content: [{ type: "text" as const, text: serializeFacts(facts) }],
                        };
                    }

                    const result = queryEntity(entity, depth);
                    if (!result) {
                        return {
                            content: [
                                {
                                    type: "text" as const,
                                    text: `No knowledge graph entries found for "${entity}".`,
                                },
                            ],
                        };
                    }

                    return { content: [{ type: "text" as const, text: result.serialized }] };
                }

                case "memory_compress": {
                    const scope = (args?.scope || "working") as CompressScope;
                    const result = await compressMemory(scope);
                    return { content: [{ type: "text" as const, text: result }] };
                }

                case "memory_forget": {
                    const memoryId = args?.memory_id as string;
                    const action = (args?.action || "deprecate") as ForgetAction;
                    const correction = args?.correction as string | undefined;

                    if (!memoryId) {
                        return {
                            content: [{ type: "text" as const, text: "Error: 'memory_id' is required." }],
                            isError: true,
                        };
                    }

                    const result = await forgetMemory(memoryId, action, correction);
                    return { content: [{ type: "text" as const, text: result }] };
                }

                case "memory_status": {
                    const status = getMemoryStatus();
                    const lines = [
                        "=== Memory Status ===",
                        `Tier 0 (Working): ${status.tiers.tier0.count} entries, ~${status.tiers.tier0.tokenEstimate} tokens`,
                        `Tier 1 (Session): ${status.tiers.tier1.count} entries, ~${status.tiers.tier1.tokenEstimate} tokens`,
                        `Tier 2 (Epoch):   ${status.tiers.tier2.count} entries, ~${status.tiers.tier2.tokenEstimate} tokens`,
                        `Tier 3 (Core):    ${status.tiers.tier3.count} entries, ~${status.tiers.tier3.tokenEstimate} tokens`,
                        `Knowledge Graph:  ${status.knowledgeGraph.entities} entities, ${status.knowledgeGraph.relations} relations`,
                        `Vector Store:     ${status.vectorStore.count} vectors`,
                        `Total Tokens:     ~${status.totalTokensStored}`,
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
                description: "Tier 3 persistent core memories. Always available for system prompt injection.",
                mimeType: "text/plain",
            },
            {
                uri: "memory://session/current",
                name: "Current Session Memory",
                description: "Tier 0 working memory for the current session.",
                mimeType: "text/plain",
            },
            {
                uri: "memory://graph/schema",
                name: "Knowledge Graph Schema",
                description:
                    "Current entity types and relation predicates in the knowledge graph.",
                mimeType: "application/json",
            },
            {
                uri: "memory://stats",
                name: "Memory Statistics",
                description: "Storage usage statistics across all memory subsystems.",
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
                description:
                    "Extract structured facts from text as subject-predicate-object triples for knowledge graph storage.",
                arguments: [
                    {
                        name: "text",
                        description: "The text to extract facts from.",
                        required: true,
                    },
                ],
            },
            {
                name: "compress_session",
                description:
                    "Compress working memory into a concise session summary preserving key decisions, unresolved items, and action items.",
                arguments: [
                    {
                        name: "working_memory",
                        description: "The current working memory text to compress.",
                        required: true,
                    },
                ],
            },
            {
                name: "consolidate_epoch",
                description:
                    "Merge multiple session summaries into a higher-level epoch summary capturing themes and patterns.",
                arguments: [
                    {
                        name: "session_summaries",
                        description: "The session summaries to consolidate.",
                        required: true,
                    },
                ],
            },
        ],
    }));

    server.setRequestHandler(GetPromptRequestSchema, async (request) => {
        const { name, arguments: args } = request.params;

        switch (name) {
            case "extract_facts": {
                const text = args?.text || "";
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
                const workingMemory = args?.working_memory || "";
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
                const summaries = args?.session_summaries || "";
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
