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
import { assembleContext } from "./context-assembler.js";
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

// ---------------------------------------------------------------------------
// Create the MCP Server
// ---------------------------------------------------------------------------

export function createServer(): Server {
    const server = new Server(
        {
            name: "latentcontext-mcp",
            version: "1.1.0",
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
                description:
                    `Start a new memory session. Call this ONCE at the very beginning of each new conversation.

MANDATORY WORKFLOW — follow these steps IN ORDER every single time:
1. Call session_start (this tool) FIRST.
2. IMMEDIATELY after, call memory_retrieve to check for any context from this session.
3. Throughout the conversation, call memory_store to save DETAILED notes about everything important.

SESSION ISOLATION:
- Each session starts with ZERO entries — completely fresh memory.
- No data from past sessions, knowledge graph, or vector store leaks in.
- Only data stored via memory_store during THIS session will be retrievable.
- Session IDs are timestamp-prefixed UUIDs for guaranteed uniqueness.

WHAT IT DOES:
- Creates a completely fresh, empty memory for the new conversation.
- Returns the new session ID.

WARNING: Always call this first to ensure session isolation.`,
                inputSchema: {
                    type: "object" as const,
                    properties: {},
                },
            },
            {
                name: "memory_store",
                description:
                    `Store information in long-term memory. Call this PROACTIVELY whenever you learn something new — do NOT wait to be asked.

CRITICAL — CONTENT QUALITY RULES:
Your memory entries MUST be DETAILED, SELF-CONTAINED paragraphs that a future AI can read and FULLY understand WITHOUT needing to re-derive anything. Each entry should be 3-5 sentences minimum. Include WHO, WHAT, WHY, HOW, and WHERE relevant.

BAD (useless — too short, forces re-thinking):
  "Fixed audio issue"
  "User likes dark mode"
  "Website uses Vanta.js"

GOOD (detailed — saves tokens by preventing re-derivation):
  "Fixed a CORS audio playback issue on the user's website fate.rf.gd. The problem was that the audio element was trying to load files from a different origin. Solution: added crossorigin='anonymous' attribute to the <audio> element and configured the server to send Access-Control-Allow-Origin headers. The audio visualizer now works correctly with the Vanta.js background."
  "User strongly prefers dark mode designs with deep blue (#0a0e27) and purple (#6c63ff) accent colors. They want glassmorphism effects, smooth animations, and premium-feeling interfaces. They dislike plain/basic designs and have explicitly requested 'wow factor' aesthetics in multiple conversations."
  "The user's personal website at fate.rf.gd uses a multi-section single-page layout with: Vanta.js globe wireframe background animation, a music player with bass-reactive glow visualizer, glassmorphic card components, Google Fonts (Inter/Outfit), and is hosted on InfinityFree. The tech stack is vanilla HTML/CSS/JS with no framework."

REMEMBER: A memory that takes 20 tokens to store but saves a future LLM from spending 500 tokens re-analyzing is EXTREMELY valuable. Write entries as if briefing a colleague who has never seen this project before.

WHEN TO USE:
- After the user tells you their name, preferences, project details, or any personal information.
- After completing a task — store WHAT was done, WHY, HOW it was solved, and the OUTCOME.
- When you discover important facts about the user's codebase, tech stack, architecture, or workflow.
- When you make design decisions — store the decision AND the reasoning.
- Whenever information might be useful in future conversations.

MEMORY TYPES (choose carefully):
- 'core': CRITICAL permanent facts (user identity, key project info, important preferences). Never evicted. Use sparingly — only for the 5-10 most important things.
- 'fact': Concrete knowledge with entities. Include full context, not just the fact. Automatically indexed in the knowledge graph.
- 'preference': User likes/dislikes/habits. Describe the preference with specifics (colors, styles, patterns), not just the category.
- 'event': What just happened in this session. Describe the task, approach, solution, and result in detail.
- 'summary': Compressed notes about a session or topic. Should be a dense briefing paragraph, not bullet points.

ENTITIES: Always provide relevant entity names in the 'entities' array for 'fact' type. The first entity is treated as the subject. Example: entities: ["User", "dark mode"] for "User prefers dark mode".`,
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        content: {
                            type: "string",
                            description: "DETAILED, self-contained paragraph (3-5+ sentences). Must include full context so a future AI can understand without re-deriving. Include specifics: names, paths, colors, versions, decisions, reasoning. NEVER write single sentences — that defeats the purpose of memory.",
                        },
                        memory_type: {
                            type: "string",
                            enum: ["fact", "preference", "event", "summary", "core"],
                            description:
                                "Category: 'core' for critical permanent info, 'fact' for knowledge, 'preference' for user likes/dislikes, 'event' for what just happened, 'summary' for compressed notes.",
                        },
                        confidence: {
                            type: "number",
                            description: "Confidence 0.0-1.0. Lower confidence = evicted first. Default 1.0. Use lower values for uncertain or temporary information.",
                        },
                        entities: {
                            type: "array",
                            items: { type: "string" },
                            description:
                                "Key entities this memory relates to. REQUIRED for 'fact' type — first entity is the subject. Example: ['User', 'JavaScript'] for 'User knows JavaScript'.",
                        },
                    },
                    required: ["content", "memory_type"],
                },
            },
            {
                name: "memory_retrieve",
                description:
                    `Retrieve memories stored in the CURRENT session. Returns only data from this conversation — no cross-session contamination.

MANDATORY — ALWAYS call this immediately after session_start. Also call before ANY task that could benefit from context stored earlier in this conversation.

WHEN TO USE:
- IMMEDIATELY after session_start — to check if there is any context from earlier in this session.
- When you need to recall what was discussed/decided/stored earlier in THIS conversation.
- Before starting a new task step — retrieve context about what was done so far.

HOW TO USE THE RESULTS:
- READ every section of the returned context carefully.
- APPLY the information to your current task — this is why it was stored.
- If you see stored decisions or notes, BUILD ON THEM.
- Do NOT ignore retrieved context — it was stored specifically to help you.

SESSION ISOLATION:
- Each session starts with ZERO entries — completely fresh.
- Only returns data stored via memory_store during THIS session.
- No data from past sessions, global knowledge graph, or vector store is included.

SECTIONS IN OUTPUT:
- [Current Session]: What has been stored so far in this conversation.
- [Current Session Notes]: Compressed notes from this session (if working memory was compressed).

TIP: Use a higher token_budget (5000-8000) for comprehensive context. Use lower (1000-2000) for focused lookups.`,
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        query: {
                            type: "string",
                            description: "What to search for. Be descriptive — include project names, tech stack, topic areas. Example: 'user website fate.rf.gd design preferences audio visualizer' rather than just 'website'.",
                        },
                        token_budget: {
                            type: "integer",
                            description: "Max tokens to return. Default 3000. Use 5000-8000 for broad conversation-start context. Use 1000-2000 for focused lookups.",
                        },
                        filters: {
                            type: "object",
                            properties: {
                                memory_types: {
                                    type: "array",
                                    items: { type: "string" },
                                    description: "Filter by source types (e.g., ['fact', 'preference']).",
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
                                    description: "Minimum confidence score to include (0.0-1.0).",
                                },
                            },
                        },
                    },
                    required: ["query"],
                },

            },
            {
                name: "memory_compress",
                description:
                    `Compress memory at a given scope to reduce token usage and consolidate information.

WHEN TO USE:
- 'working': When working memory is getting large during a long conversation. Compresses current session buffer into a summary.
- 'session': When there are many session summaries. Merges multiple session-level summaries into fewer entries.
- 'epoch': After many sessions have accumulated. Promotes session summaries into high-level long-term knowledge. Requires at least 10 session summaries.

EFFECTS:
- All compression is lossy — details are condensed but key information is preserved.
- Compressed data is re-embedded for semantic search.
- Original entries are removed after compression.`,
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        scope: {
                            type: "string",
                            enum: ["working", "session", "epoch"],
                            description: "Compression scope: 'working' (current session), 'session' (merge sessions), 'epoch' (long-term consolidation).",
                        },
                    },
                    required: ["scope"],
                },
            },
            {
                name: "memory_forget",
                description:
                    `Mark a memory as outdated, incorrect, or to be deleted.

WHEN TO USE:
- When the user corrects previously stored information.
- When stored facts become outdated (e.g., user changed jobs, moved cities).
- When duplicate or incorrect memories need cleanup.

ACTIONS:
- 'deprecate': Lowers confidence score so the memory is deprioritized but not removed. Use when unsure.
- 'correct': Replaces the memory content with new, correct information. Requires the 'correction' parameter.
- 'delete': Permanently removes the memory. Use for clearly wrong or duplicate entries.

You need the memory_id which is returned when you store a memory, or visible in memory_status output.`,
                inputSchema: {
                    type: "object" as const,
                    properties: {
                        memory_id: {
                            type: "string",
                            description: "ID of the memory to modify (UUID format, returned by memory_store).",
                        },
                        action: {
                            type: "string",
                            enum: ["deprecate", "correct", "delete"],
                            description: "Action to take: 'deprecate' (lower priority), 'correct' (replace content), 'delete' (remove permanently).",
                        },
                        correction: {
                            type: "string",
                            description: "New content to replace the memory with. Required when action is 'correct'.",
                        },
                    },
                    required: ["memory_id", "action"],
                },
            },
            {
                name: "memory_status",
                description:
                    `Get storage statistics for all memory subsystems.

WHEN TO USE:
- When debugging memory-related issues.
- When the user asks "what do you remember?" or "how much is stored?".
- To check if session_start was called (shows current session ID).
- To monitor token budgets and plan compression.

SHOWS: Tier counts, token estimates, knowledge graph size, vector store count, and current session ID.`,
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
                case "session_start": {
                    const result = await startSession(async (oldSessionId) => {
                        return archiveWorkingMemory(oldSessionId);
                    });

                    // Clear ALL working memory to guarantee complete session isolation.
                    // No data from previous sessions should leak into the new one.
                    clearWorkingMemory();

                    const lines = [
                        `New session started: ${result.sessionId}`,
                        `Started at: ${result.startedAt}`,
                        `Session is COMPLETELY FRESH — zero entries from past sessions.`,
                    ];

                    // Remind the LLM to follow the mandatory workflow
                    lines.push("");
                    lines.push("Throughout this conversation, call memory_store to save DETAILED notes (3-5+ sentences each) about tasks, decisions, and discoveries.");
                    lines.push("Call memory_retrieve at any time to recall what was stored earlier in THIS session.");

                    return { content: [{ type: "text" as const, text: lines.join("\n") }] };
                }

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

                    // ── Content quality enforcement ──
                    // Reject entries that are too terse to be useful in future retrieval.
                    // A single sentence like "Fixed audio issue" wastes storage and forces
                    // re-derivation later — the whole point of memory is to AVOID that.
                    const wordCount = content.trim().split(/\s+/).length;
                    const sentenceCount = (content.match(/[.!?;]\s|[.!?;]$/g) || []).length;

                    if (wordCount < 10) {
                        return {
                            content: [{
                                type: "text" as const,
                                text: `REJECTED: Memory too short (${wordCount} words). Entries must be detailed, self-contained paragraphs (3-5+ sentences, 10+ words minimum). Your entry: "${content.substring(0, 80)}"\n\nRewrite with full context: WHO/WHAT/WHY/HOW/WHERE. Example:\nBAD:  "Fixed audio issue"\nGOOD: "Fixed a CORS audio playback issue on the user's website fate.rf.gd. The <audio> element was trying to load files cross-origin. Solution: added crossorigin='anonymous' and configured server CORS headers. Audio visualizer now works with Vanta.js background."\n\nPlease resubmit with a detailed version.`
                            }],
                            isError: true,
                        };
                    }

                    const result = await storeMemory(content, memoryType, confidence, entities);

                    // Build response with quality feedback
                    const qualityNote = wordCount < 25
                        ? "\n⚠️ Entry is brief. Consider adding more context (WHY, HOW, specifics) for better future retrieval."
                        : "";

                    const response = [
                        `Stored as ${result.memoryType} (Tier ${result.tier})`,
                        `ID: ${result.memoryId}`,
                        result.sessionId ? `Session: ${result.sessionId.substring(0, 8)}` : null,
                        result.factsStored > 0 ? `Facts stored: ${result.factsStored}` : null,
                        result.entitiesCreated.length > 0
                            ? `Entities: ${result.entitiesCreated.join(", ")}`
                            : null,
                        result.vectorId ? `Vector indexed` : null,
                        qualityNote || null,
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
                description: "Tier 3 persistent core memories. Always available for system prompt injection.",
                mimeType: "text/plain",
            },
            {
                uri: "memory://session/current",
                name: "Current Session Memory",
                description: "Tier 0 working memory for the current session only.",
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
