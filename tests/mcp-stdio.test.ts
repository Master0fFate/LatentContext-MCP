import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { CONFIG_FILE_NAME, DATA_DIR_ENV } from "../src/config.js";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const packageVersion = JSON.parse(readFileSync(join(projectRoot, "package.json"), "utf8")).version as string;
const tempDirs: string[] = [];

after(() => {
    for (const dir of tempDirs.splice(0)) {
        rmSync(dir, { recursive: true, force: true });
    }
});

function toolText(result: { content: Array<{ type: string; text?: string }> }): string {
    const content = result.content.find((item) => item.type === "text");
    assert.ok(content?.text, "tool response must contain text content");
    return content.text;
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
    try {
        // A successful SDK initialization proves startup emitted only valid MCP
        // JSON-RPC frames; any ordinary stdout logging would corrupt this flow.
        await client.connect(transport, { timeout: 10_000 });
        assert.deepEqual(client.getServerVersion(), {
            name: "latentcontext-mcp",
            version: packageVersion,
        });

        const discovery = await client.listTools(undefined, { timeout: 10_000 });
        const tools = new Map(discovery.tools.map((tool) => [tool.name, tool]));
        assert.deepEqual([...tools.keys()].sort(), [
            "memory_compress",
            "memory_forget",
            "memory_retrieve",
            "memory_status",
            "memory_store",
            "session_start",
        ]);
        assert.deepEqual(tools.get("session_start")?.inputSchema, {
            type: "object",
            properties: {},
            additionalProperties: false,
        });
        assert.deepEqual(
            Object.fromEntries([...tools].map(([name, tool]) => [name, tool.description])),
            {
                session_start: "Start a fresh isolated session before session memory work.",
                memory_store: "Store session or persistent memory by type.",
                memory_retrieve: "Search working memory and summaries in the active session.",
                memory_compress: "Compress active-session memory at a selected scope.",
                memory_forget: "Deprecate, correct, or delete a memory.",
                memory_status: "Show memory storage and session status.",
            }
        );
        for (const tool of tools.values()) {
            assert.ok(tool.description.length < 64, `${tool.name} description must remain compact`);
            assert.ok(!tool.description.includes("\n"), `${tool.name} description must be single-line`);
        }
        const storeSchema = tools.get("memory_store")?.inputSchema as {
            required?: string[];
            additionalProperties?: boolean;
            properties?: {
                content?: { minLength?: number; pattern?: string };
                confidence?: { minimum?: number; maximum?: number };
            };
        };
        assert.deepEqual(storeSchema.required, ["content", "memory_type"]);
        assert.equal(storeSchema.additionalProperties, false);
        assert.deepEqual(storeSchema.properties?.content, {
            type: "string",
            minLength: 1,
            pattern: "^(?:\\s*\\S+\\s+){9}\\S+\\s*$",
            description: "Self-contained text (10+ words).",
        });
        assert.deepEqual(storeSchema.properties?.confidence, {
            type: "number",
            minimum: 0,
            maximum: 1,
            default: 1,
            description: "Confidence.",
        });
        const retrieveSchema = tools.get("memory_retrieve")?.inputSchema as {
            required?: string[];
            additionalProperties?: boolean;
            properties?: { filters?: { additionalProperties?: boolean } };
        };
        assert.deepEqual(retrieveSchema.required, ["query"]);
        assert.equal(retrieveSchema.additionalProperties, false);
        assert.equal(retrieveSchema.properties?.filters?.additionalProperties, false);
        assert.deepEqual(tools.get("memory_status")?.inputSchema, {
            type: "object",
            properties: {},
            additionalProperties: false,
        });

        const invalidCalls: Array<{ name: string; arguments: Record<string, unknown>; error: RegExp }> = [
            { name: "session_start", arguments: { unsupported: true }, error: /unsupported argument 'unsupported'/ },
            { name: "memory_store", arguments: { content: "too short to retain as useful future context", memory_type: "event" }, error: /must contain at least 10 words/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "unknown" }, error: /'memory_type' must be one of/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "event", confidence: 1.1 }, error: /'confidence' must be a number between 0 and 1/ },
            { name: "memory_store", arguments: { content: "This stored memory has enough words to pass the length requirement today.", memory_type: "event", entities: [" "] }, error: /'entities' must be an array of non-empty strings/ },
            { name: "memory_retrieve", arguments: { query: "release", unsupported: true }, error: /unsupported argument 'unsupported'/ },
            { name: "memory_retrieve", arguments: { query: "release", token_budget: 1.5 }, error: /'token_budget' must be a positive integer/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { after: "not-a-date" } }, error: /'after' must be an ISO datetime/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { after: "2024-02-30T12:00:00Z" } }, error: /'after' must be an ISO datetime/ },
            { name: "memory_retrieve", arguments: { query: "release", filters: { memory_types: ["unknown"] } }, error: /'memory_types' must be a non-empty array/ },
            { name: "memory_compress", arguments: { scope: "unknown" }, error: /'scope' must be one of/ },
            { name: "memory_forget", arguments: { memory_id: "not-a-uuid", action: "delete" }, error: /'memory_id' must be a UUID/ },
            { name: "memory_forget", arguments: { memory_id: "123e4567-e89b-42d3-a456-426614174000", action: "correct" }, error: /'correction' is required/ },
            { name: "memory_forget", arguments: { memory_id: "123e4567-e89b-42d3-a456-426614174000", action: "delete", correction: "replacement" }, error: /'correction' is only valid/ },
            { name: "memory_status", arguments: { unsupported: true }, error: /unsupported argument 'unsupported'/ },
        ];
        for (const invalidCall of invalidCalls) {
            const result = await client.callTool(invalidCall, undefined, { timeout: 10_000 });
            assert.equal(result.isError, true, `${invalidCall.name} must reject invalid arguments`);
            assert.match(toolText(result), invalidCall.error);
        }

        const started = toolText(await client.callTool(
            { name: "session_start", arguments: {} },
            undefined,
            { timeout: 10_000 }
        ));
        const sessionId = started.match(/^New session started: ([^\n]+)$/m)?.[1];
        assert.ok(sessionId, "session_start must return a session ID");

        const memory = "The release workflow uses an isolated temporary data directory, disables embeddings, and verifies MCP stdio JSON-RPC calls return correctly.";
        const stored = toolText(await client.callTool(
            { name: "memory_store", arguments: { content: memory, memory_type: "event" } },
            undefined,
            { timeout: 10_000 }
        ));
        assert.match(stored, /^Stored as event \(Tier 0\)$/m);
        const memoryId = stored.match(/^ID: ([^\n]+)$/m)?.[1];
        assert.ok(memoryId, "memory_store must return a memory ID");

        const retrieved = toolText(await client.callTool(
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
            { timeout: 10_000 }
        ));
        assert.match(retrieved, new RegExp(memory));

        const status = toolText(await client.callTool(
            { name: "memory_status", arguments: {} },
            undefined,
            { timeout: 10_000 }
        ));
        assert.match(status, new RegExp(`Session: ${sessionId.substring(0, 8)}\\.\\.\\.`));
        assert.match(status, /Tier 0 \(Working\):\s+1 entries/);

        const forgotten = toolText(await client.callTool(
            { name: "memory_forget", arguments: { memory_id: memoryId, action: "delete" } },
            undefined,
            { timeout: 10_000 }
        ));
        assert.equal(forgotten, `Deleted working memory entry ${memoryId}.`);

        const compressibleMemory = "The compression workflow stores a second isolated event so the public working-memory compression response can be verified over stdio.";
        const compressibleStored = toolText(await client.callTool(
            { name: "memory_store", arguments: { content: compressibleMemory, memory_type: "event" } },
            undefined,
            { timeout: 10_000 }
        ));
        assert.match(compressibleStored, /^Stored as event \(Tier 0\)$/m);

        const compressed = toolText(await client.callTool(
            { name: "memory_compress", arguments: { scope: "working" } },
            undefined,
            { timeout: 10_000 }
        ));
        assert.match(compressed, /^Compressed 1 working memory entries \(\d+ tokens\) into Tier 1 summary \(\d+ tokens\)\. Compression ratio: \d+\.\dx$/);

        assert.deepEqual(protocolErrors, [], "stdio output must remain valid JSON-RPC");
        assert.equal(stderr, "", "startup diagnostics must not leak to stderr");
    } finally {
        await client.close().catch(() => undefined);
    }
});
