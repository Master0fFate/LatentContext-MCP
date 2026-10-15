import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { LoggingMessageNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { CONFIG_FILE_NAME, DATA_DIR_ENV } from "../src/config.js";
import { COMPRESS_SCOPES, FORGET_ACTIONS, MEMORY_TYPES, SERVER_NAME, SERVER_TITLE } from "../src/server.js";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const packageVersion = JSON.parse(readFileSync(join(projectRoot, "package.json"), "utf8")).version as string;
const tempDirs: string[] = [];
const CALL_OPTIONS = { timeout: 10_000 };

after(() => {
    for (const dir of tempDirs.splice(0)) {
        rmSync(dir, { recursive: true, force: true });
    }
});

type ToolResult = Awaited<ReturnType<Client["callTool"]>>;

function toolText(result: ToolResult): string {
    assert.ok("content" in result && Array.isArray(result.content), "tool response must use the content array");
    const content = (result.content as Array<{ type: string; text?: string }>).find((item) => item.type === "text");
    assert.ok(content?.text, "tool response must contain text content");
    return content.text;
}

function structured(result: ToolResult): Record<string, unknown> {
    assert.ok("structuredContent" in result && result.structuredContent, "tool response must include structuredContent");
    return result.structuredContent as Record<string, unknown>;
}

/**
 * Keywords every MCP host is known to accept when it translates a tool's
 * JSON Schema into its model provider's function-calling format. Conditional
 * keywords (allOf/if/then, oneOf), string formats and regular-expression
 * patterns are rejected by at least one host, so they must not appear.
 */
const PORTABLE_SCHEMA_KEYWORDS = new Set([
    "$schema", "type", "title", "description", "properties", "required", "additionalProperties",
    "enum", "items", "minimum", "maximum", "minLength", "minItems", "default",
]);

function assertPortableSchema(schema: unknown, path: string): void {
    assert.ok(schema && typeof schema === "object" && !Array.isArray(schema), `${path} must be a schema object`);
    for (const [keyword, value] of Object.entries(schema as Record<string, unknown>)) {
        assert.ok(PORTABLE_SCHEMA_KEYWORDS.has(keyword), `${path} uses non-portable JSON Schema keyword '${keyword}'`);
        if (keyword === "properties") {
            for (const [property, propertySchema] of Object.entries(value as Record<string, unknown>)) {
                assertPortableSchema(propertySchema, `${path}.${property}`);
            }
        } else if (keyword === "items") {
            assertPortableSchema(value, `${path}[]`);
        }
    }
}

test("MCP stdio discovery and session memory workflow remain JSON-RPC-safe", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-mcp-stdio-"));
    tempDirs.push(dataDir);
    writeFileSync(
        join(dataDir, CONFIG_FILE_NAME),
        JSON.stringify({ embedding: { provider: "none" } })
    );

    const transport = new StdioClientTransport({
        command: process.execPath,
        args: ["--import", "tsx", join(projectRoot, "src", "index.ts")],
        cwd: projectRoot,
        env: { [DATA_DIR_ENV]: dataDir },
        stderr: "pipe",
    });
    const protocolErrors: Error[] = [];
    let stderr = "";
    transport.onerror = (error) => protocolErrors.push(error);
    transport.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
    });

    const client = new Client({ name: "latentcontext-stdio-test", version: "1.0.0" });
    const logMessages: Array<{ level: string; logger?: string; data: unknown }> = [];
    client.setNotificationHandler(LoggingMessageNotificationSchema, (notification) => {
        logMessages.push(notification.params);
    });

    try {
        // A successful SDK initialization proves startup emitted only valid MCP
        // JSON-RPC frames; any ordinary stdout logging would corrupt this flow.
        await client.connect(transport, CALL_OPTIONS);
        assert.deepEqual(client.getServerVersion(), {
            name: SERVER_NAME,
            version: packageVersion,
            title: SERVER_TITLE,
        });
        assert.match(client.getInstructions() ?? "", /session_start/, "server must publish usage instructions");
        const capabilities = client.getServerCapabilities() ?? {};
        for (const capability of ["tools", "resources", "prompts", "logging"] as const) {
            assert.ok(capability in capabilities, `server must advertise the ${capability} capability`);
        }

        // ── Discovery ──────────────────────────────────────────────────────
        const discovery = await client.listTools(undefined, CALL_OPTIONS);
        const tools = new Map(discovery.tools.map((tool) => [tool.name, tool]));
        assert.deepEqual([...tools.keys()].sort(), [
            "code_explore",
            "memory_compress",
            "memory_forget",
            "memory_retrieve",
            "memory_status",
            "memory_store",
            "session_start",
        ]);
        assert.deepEqual(
            Object.fromEntries([...tools].map(([name, tool]) => [name, tool.description])),
            {
                session_start: "Start a fresh isolated session before session memory work.",
                memory_store: "Store session or persistent memory by type.",
                memory_retrieve: "Search working memory and summaries in the active session.",
                memory_compress: "Compress active-session memory at a selected scope.",
                memory_forget: "Deprecate, correct, or delete a memory.",
                memory_status: "Show memory storage and session status.",
                code_explore: "Find code by names or question; source plus call graph.",
            }
        );
        for (const tool of tools.values()) {
            assert.ok(tool.description && tool.description.length < 64, `${tool.name} description must remain compact`);
            assert.ok(!tool.description.includes("\n"), `${tool.name} description must be single-line`);
            assert.ok(tool.title, `${tool.name} must declare a human-readable title`);
            assert.equal(typeof tool.annotations?.readOnlyHint, "boolean", `${tool.name} must declare readOnlyHint`);
            assert.equal(tool.annotations?.openWorldHint, false, `${tool.name} only touches local storage`);
            assertPortableSchema(tool.inputSchema, `${tool.name}.inputSchema`);
            assert.equal(tool.inputSchema.type, "object");
            assert.equal(tool.inputSchema.additionalProperties, false, `${tool.name} must reject unknown arguments`);
            if (tool.outputSchema) assertPortableSchema(tool.outputSchema, `${tool.name}.outputSchema`);
        }
        assert.deepEqual(
            Object.fromEntries([...tools].map(([name, tool]) => [name, tool.annotations?.readOnlyHint])),
            {
                session_start: false,
                memory_store: false,
                memory_retrieve: true,
                memory_compress: false,
                memory_forget: false,
                memory_status: true,
                code_explore: true,
            }
        );
        assert.equal(tools.get("memory_compress")?.annotations?.destructiveHint, true);
        assert.equal(tools.get("memory_forget")?.annotations?.destructiveHint, true);
        assert.equal(tools.get("memory_store")?.annotations?.destructiveHint, false);
        for (const name of ["session_start", "memory_store", "memory_retrieve", "memory_status"]) {
            assert.ok(tools.get(name)?.outputSchema, `${name} must publish an output schema`);
        }

        assert.deepEqual(Object.keys(tools.get("session_start")?.inputSchema.properties ?? { unexpected: true }), []);
        assert.deepEqual(Object.keys(tools.get("memory_status")?.inputSchema.properties ?? { unexpected: true }), []);

        const storeSchema = tools.get("memory_store")?.inputSchema as {
            required?: string[];
            properties?: Record<string, { type?: string; enum?: string[]; minimum?: number; maximum?: number; default?: number; description?: string }>;
        };
        assert.deepEqual(storeSchema.required, ["content", "memory_type"]);
        assert.deepEqual(storeSchema.properties?.memory_type?.enum, [...MEMORY_TYPES]);
        assert.equal(storeSchema.properties?.content?.type, "string");
        assert.match(storeSchema.properties?.content?.description ?? "", /10\+ words/);
        assert.deepEqual(
            [storeSchema.properties?.confidence?.minimum, storeSchema.properties?.confidence?.maximum, storeSchema.properties?.confidence?.default],
            [0, 1, 1]
        );

        const retrieveSchema = tools.get("memory_retrieve")?.inputSchema as {
            required?: string[];
            properties?: {
                token_budget?: { type?: string; minimum?: number };
                filters?: { additionalProperties?: boolean; properties?: Record<string, { description?: string }> };
            };
        };
        assert.deepEqual(retrieveSchema.required, ["query"]);
        assert.equal(retrieveSchema.properties?.token_budget?.type, "integer");
        assert.equal(retrieveSchema.properties?.token_budget?.minimum, 1);
        assert.equal(retrieveSchema.properties?.filters?.additionalProperties, false);
        assert.deepEqual(Object.keys(retrieveSchema.properties?.filters?.properties ?? {}).sort(), ["after", "before", "memory_types", "min_confidence"]);
        assert.match(retrieveSchema.properties?.filters?.properties?.after?.description ?? "", /ISO 8601/);

        const compressSchema = tools.get("memory_compress")?.inputSchema as { properties?: { scope?: { enum?: string[] } } };
        assert.deepEqual(compressSchema.properties?.scope?.enum, [...COMPRESS_SCOPES]);
        const forgetSchema = tools.get("memory_forget")?.inputSchema as {
            required?: string[];
            properties?: { action?: { enum?: string[] }; correction?: { description?: string } };
        };
        assert.deepEqual(forgetSchema.required, ["memory_id", "action"]);
        assert.deepEqual(forgetSchema.properties?.action?.enum, [...FORGET_ACTIONS]);
        assert.match(forgetSchema.properties?.correction?.description ?? "", /required when action is 'correct'/);

        // ── Argument validation ────────────────────────────────────────────
        const invalidCalls: Array<{ name: string; arguments: Record<string, unknown>; error: RegExp }> = [
            { name: "session_start", arguments: { unsupported: true }, error: /unsupported/ },
            { name: "memory_store", arguments: { content: "too short to retain as useful future context", memory_type: "event" }, error: /must contain at least 10 words/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "unknown" }, error: /memory_type/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "event", confidence: 1.1 }, error: /confidence/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "event", entities: [" "] }, error: /'entities' must be an array of non-empty strings/ },
            { name: "memory_retrieve", arguments: { query: "release", unsupported: true }, error: /unsupported/ },
            { name: "memory_retrieve", arguments: { query: "release", token_budget: 1.5 }, error: /token_budget/ },
            { name: "memory_retrieve", arguments: { query: "release", token_budget: 0 }, error: /token_budget/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { after: "not-a-date" } }, error: /'after' must be an ISO datetime/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { after: "2024-02-30T12:00:00Z" } }, error: /'after' must be an ISO datetime/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { after: "2024-03-01T00:00:00Z", before: "2024-02-01T00:00:00Z" } }, error: /'after' must not be later than 'before'/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { memory_types: ["unknown"] } }, error: /memory_types/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { memory_types: [] } }, error: /memory_types/ },
            { name: "memory_compress", arguments: { scope: "unknown" }, error: /scope/ },
            { name: "memory_forget", arguments: { memory_id: "not-a-uuid", action: "delete" }, error: /'memory_id' must be a UUID/ },
            { name: "memory_forget", arguments: { memory_id: "123e4567-e89b-42d3-a456-426614174000", action: "correct" }, error: /'correction' is required/ },
            { name: "memory_forget", arguments: { memory_id: "123e4567-e89b-42d3-a456-426614174000", action: "delete", correction: "replacement" }, error: /'correction' is only valid/ },
            { name: "memory_status", arguments: { unsupported: true }, error: /unsupported/ },
        ];
        for (const invalidCall of invalidCalls) {
            const result = await client.callTool(invalidCall, undefined, CALL_OPTIONS);
            assert.equal(result.isError, true, `${invalidCall.name} must reject ${JSON.stringify(invalidCall.arguments)}`);
            assert.match(toolText(result), invalidCall.error);
        }

        // ── Memory workflow ────────────────────────────────────────────────
        const startedResult = await client.callTool({ name: "session_start", arguments: {} }, undefined, CALL_OPTIONS);
        const started = toolText(startedResult);
        const sessionId = started.match(/^New session started: ([^\n]+)$/m)?.[1];
        assert.ok(sessionId, "session_start must return a session ID");
        assert.equal(structured(startedResult).sessionId, sessionId);
        assert.equal(typeof structured(startedResult).previousSessionId, "string", "the auto-started boot session is archived");

        const memory = "The release workflow uses an isolated temporary data directory, disables embeddings, and verifies MCP stdio JSON-RPC calls return correctly.";
        const storedResult = await client.callTool(
            { name: "memory_store", arguments: { content: memory, memory_type: "event" } },
            undefined,
            CALL_OPTIONS
        );
        const stored = toolText(storedResult);
        assert.match(stored, /^Stored as event \(Tier 0\)$/m);
        const memoryId = stored.match(/^ID: ([^\n]+)$/m)?.[1];
        assert.ok(memoryId, "memory_store must return a memory ID");
        assert.equal(structured(storedResult).memoryId, memoryId);
        assert.equal(structured(storedResult).tier, 0);
        assert.equal(structured(storedResult).sessionId, sessionId);

        const retrievedResult = await client.callTool(
            {
                name: "memory_retrieve",
                arguments: {
                    query: "release workflow JSON-RPC",
                    token_budget: 1_000,
                    // A real leap-day UTC timestamp must pass the public
                    // boundary (unlike the invalid February 30 case above).
                    filters: { after: "2024-02-29T00:00:00Z" },
                },
            },
            undefined,
            CALL_OPTIONS
        );
        assert.match(toolText(retrievedResult), new RegExp(memory));
        assert.equal(structured(retrievedResult).candidatesSelected, 1);
        assert.deepEqual(structured(retrievedResult).sourceCounts, { working: 1 });
        assert.ok((structured(retrievedResult).totalTokens as number) <= 1_000);

        const statusResult = await client.callTool({ name: "memory_status", arguments: {} }, undefined, CALL_OPTIONS);
        const status = toolText(statusResult);
        assert.match(status, new RegExp(`Session: ${sessionId.substring(0, 8)}\\.\\.\\.`));
        assert.match(status, /Tier 0 \(Working\):\s+1 entries/);
        assert.equal(structured(statusResult).sessionId, sessionId);
        assert.deepEqual((structured(statusResult).tiers as { tier0: { count: number } }).tier0.count, 1);

        const forgotten = toolText(await client.callTool(
            { name: "memory_forget", arguments: { memory_id: memoryId, action: "delete" } },
            undefined,
            CALL_OPTIONS
        ));
        assert.equal(forgotten, `Deleted working memory entry ${memoryId}.`);

        const compressibleMemory = "The compression workflow stores a second isolated event so the public working-memory compression response can be verified over stdio.";
        assert.match(toolText(await client.callTool(
            { name: "memory_store", arguments: { content: compressibleMemory, memory_type: "event" } },
            undefined,
            CALL_OPTIONS
        )), /^Stored as event \(Tier 0\)$/m);

        const compressed = toolText(await client.callTool(
            { name: "memory_compress", arguments: { scope: "working" } },
            undefined,
            CALL_OPTIONS
        ));
        assert.match(compressed, /^Compressed 1 working memory entries \(\d+ tokens\) into Tier 1 summary \(\d+ tokens\)\. Compression ratio: \d+\.\dx$/);

        // ── Resources and prompts ──────────────────────────────────────────
        const resources = await client.listResources(undefined, CALL_OPTIONS);
        assert.deepEqual(resources.resources.map((resource) => resource.uri).sort(), [
            "memory://core",
            "memory://graph/schema",
            "memory://session/current",
            "memory://stats",
        ]);
        const sessionResource = await client.readResource({ uri: "memory://session/current" }, CALL_OPTIONS);
        assert.equal(sessionResource.contents[0]?.uri, "memory://session/current");
        assert.equal(sessionResource.contents[0]?.mimeType, "text/plain");
        const statsResource = await client.readResource({ uri: "memory://stats" }, CALL_OPTIONS);
        const statsContent = statsResource.contents[0];
        assert.ok(statsContent && "text" in statsContent, "memory://stats must return text content");
        assert.equal(JSON.parse(statsContent.text).currentSessionId, sessionId);

        const prompts = await client.listPrompts(undefined, CALL_OPTIONS);
        assert.deepEqual(
            Object.fromEntries(prompts.prompts.map((prompt) => [prompt.name, prompt.arguments?.map((argument) => argument.name)])),
            {
                extract_facts: ["text"],
                compress_session: ["working_memory"],
                consolidate_epoch: ["session_summaries"],
            }
        );
        assert.ok(prompts.prompts.every((prompt) => prompt.arguments?.every((argument) => argument.required === true)));
        const prompt = await client.getPrompt({ name: "extract_facts", arguments: { text: "Ada works at Acme." } }, CALL_OPTIONS);
        const promptContent = prompt.messages[0]?.content;
        assert.equal(promptContent?.type, "text");
        assert.match(String(promptContent?.text), /Ada works at Acme\./);
        await assert.rejects(
            () => client.getPrompt({ name: "extract_facts", arguments: {} }, CALL_OPTIONS),
            /text/
        );

        // ── Logging ────────────────────────────────────────────────────────
        // Lifecycle events are forwarded as MCP log notifications after the
        // handshake, so hosts that surface server logs can show them.
        await client.setLoggingLevel("info", CALL_OPTIONS);
        await client.callTool({ name: "session_start", arguments: {} }, undefined, CALL_OPTIONS);
        await client.listTools(undefined, CALL_OPTIONS); // round trip flushes pending notifications
        const sessionLog = logMessages.find((message) =>
            message.logger === "latentcontext" && /Session started/.test(String((message.data as { message?: string })?.message)));
        assert.ok(sessionLog, "session_start must emit an MCP logging notification");
        assert.equal(sessionLog.level, "info");

        assert.deepEqual(protocolErrors, [], "stdio output must remain valid JSON-RPC");
        assert.equal(stderr, "", "startup diagnostics must not leak to stderr");
    } finally {
        await client.close().catch(() => undefined);
    }
});
