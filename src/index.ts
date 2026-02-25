#!/usr/bin/env node

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.js";
import { initDatabase, closeDatabase } from "./database.js";
import { createServer } from "./server.js";

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
    // Load configuration
    const configPath = process.argv[2] || undefined;
    loadConfig(configPath);

    // Initialize database (creates schema if needed) — async for sql.js WASM init
    await initDatabase();

    // Create and start the MCP server
    const server = createServer();
    const transport = new StdioServerTransport();

    // Graceful shutdown
    const shutdown = async () => {
        try {
            closeDatabase();
            await server.close();
        } catch {
            // Ignore shutdown errors
        }
        process.exit(0);
    };

    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
    process.on("SIGHUP", shutdown);

    // Handle uncaught errors gracefully
    process.on("uncaughtException", (error: Error) => {
        console.error("Uncaught exception:", error);
    });

    process.on("unhandledRejection", (reason: unknown) => {
        console.error("Unhandled rejection:", reason);
    });

    // Connect to stdio transport
    await server.connect(transport);
}

main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
