#!/usr/bin/env node

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.js";
import { initDatabase, closeDatabase } from "./database.js";
import { createServer } from "./server.js";
import { startSession, endCurrentSessionWithArchive } from "./session.js";
import { archiveWorkingMemory } from "./memory-manager.js";
import { writeFileSync, appendFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { spawn } from "child_process";
import type { Writable } from "stream";
import { getConfigSource } from "./config.js";

// ---------------------------------------------------------------------------
// Stdio-safe logging: NEVER write to stdout (reserved for MCP JSON-RPC).
// All diagnostic output goes to a log file instead.
// ---------------------------------------------------------------------------

let _logPath: string | null = null;

function createMcpStdout(): Writable {
    const originalWrite = process.stdout.write.bind(process.stdout);
    const originalOnce = process.stdout.once.bind(process.stdout);

    return {
        write: originalWrite,
        once: originalOnce,
    } as unknown as Writable;
}

function initLogFile(): void {
    try {
        const config = loadConfig();
        const dataDir = config.storage.dataDir;
        if (!existsSync(dataDir)) {
            mkdirSync(dataDir, { recursive: true });
        }
        _logPath = join(dataDir, "server.log");
        // Truncate log on startup to avoid unbounded growth
        writeFileSync(_logPath, `[${new Date().toISOString()}] LatentContext MCP server starting...\n`);
    } catch {
        // If we can't create a log file, just swallow everything
        _logPath = null;
    }
}

function logToFile(level: string, message: string, error?: unknown): void {
    if (!_logPath) return;
    try {
        const timestamp = new Date().toISOString();
        let line = `[${timestamp}] [${level}] ${message}`;
        if (error instanceof Error) {
            line += ` | ${error.message}\n${error.stack}`;
        } else if (error !== undefined) {
            line += ` | ${String(error)}`;
        }
        appendFileSync(_logPath, line + "\n");
    } catch {
        // Swallow all logging errors — we must NEVER crash over logging
    }
}

// ---------------------------------------------------------------------------
// Suppress ALL console output to prevent stdio corruption.
// The MCP protocol uses stdout for JSON-RPC messages. ANY stray output
// (console.log, console.error, console.warn, or library output from
// @huggingface/transformers, sql.js, etc.) will corrupt the protocol
// and cause "connection closed: EOF" errors.
// ---------------------------------------------------------------------------

function suppressConsole(): void {
    // Override all console methods to route through our file logger
    console.log = (...args: unknown[]) => {
        logToFile("LOG", args.map(String).join(" "));
    };
    console.info = (...args: unknown[]) => {
        logToFile("INFO", args.map(String).join(" "));
    };
    console.warn = (...args: unknown[]) => {
        logToFile("WARN", args.map(String).join(" "));
    };
    console.error = (...args: unknown[]) => {
        logToFile("ERROR", args.map(String).join(" "));
    };
    console.debug = (...args: unknown[]) => {
        logToFile("DEBUG", args.map(String).join(" "));
    };
    console.trace = (...args: unknown[]) => {
        logToFile("TRACE", args.map(String).join(" "));
    };

    // Also suppress process.stdout.write and process.stderr.write for
    // libraries that bypass console (like @huggingface/transformers
    // progress bars and ONNX runtime status messages).
    // The MCP transport receives a preserved stdout writer created before
    // this patch, so anything else that writes to process.stdout is logged.

    // Monkey-patch stderr.write to route to log file
    process.stderr.write = ((
        chunk: string | Uint8Array,
        encodingOrCallback?: BufferEncoding | ((error?: Error | null) => void),
        callback?: (error?: Error | null) => void
    ): boolean => {
        const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf-8");
        logToFile("STDERR", text.trimEnd());
        // Call the callback if provided to prevent hanging
        const cb = typeof encodingOrCallback === "function" ? encodingOrCallback : callback;
        if (cb) cb(null);
        return true;
    }) as typeof process.stderr.write;

    process.stdout.write = ((
        chunk: string | Uint8Array,
        encodingOrCallback?: BufferEncoding | ((error?: Error | null) => void),
        callback?: (error?: Error | null) => void
    ): boolean => {
        const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf-8");
        logToFile("STDOUT", text.trimEnd());
        const cb = typeof encodingOrCallback === "function" ? encodingOrCallback : callback;
        if (cb) cb(null);
        return true;
    }) as typeof process.stdout.write;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
    const mcpStdout = createMcpStdout();

    // Step 1: Initialize log file FIRST (before suppressing console)
    initLogFile();

    // Step 2: Suppress all console/stdout/stderr output
    suppressConsole();

    logToFile("INFO", "Console output suppressed, initializing...");

    // Step 3: Load configuration
    const config = loadConfig();
    logToFile("INFO", `Config loaded. Data dir: ${config.storage.dataDir}`);
    logToFile("INFO", `Config source: ${getConfigSource() ?? "defaults/env"}`);

    // Step 4: Initialize database (creates schema if needed) — async for sql.js WASM init
    try {
        await initDatabase();
        logToFile("INFO", "Database initialized successfully");
    } catch (error) {
        logToFile("FATAL", "Failed to initialize database", error);
        // Database is critical — we cannot operate without it
        process.exit(1);
    }

    // Step 5: Auto-start a fresh session on server startup
    if (config.session.autoStartOnBoot) {
        try {
            await startSession(archiveWorkingMemory);
            logToFile("INFO", "Auto-started initial session");
        } catch (error) {
            logToFile("ERROR", "Failed to auto-start session (non-fatal)", error);
            // Session auto-start failure is non-fatal — the session_start tool
            // can be called manually by the LLM
        }
    }

    // Step 6: Create and start the MCP server
    const server = createServer();
    const transport = new StdioServerTransport(process.stdin, mcpStdout);

    logToFile("INFO", "MCP server created, connecting transport...");

    // Step 7: Graceful shutdown — do NOT call process.exit() here!
    // Calling process.exit() while the transport is active causes the
    // "connection closed: EOF" error. Instead, let the process close
    // naturally after cleanup.
    let isShuttingDown = false;

    const shutdown = async (signal: string) => {
        if (isShuttingDown) return; // Prevent double-shutdown
        isShuttingDown = true;
        let shutdownFailed = false;
        logToFile("INFO", `Shutdown requested (${signal})`);

        try {
            await endCurrentSessionWithArchive(archiveWorkingMemory);
            logToFile("INFO", "Session archived and ended");
        } catch (error) {
            shutdownFailed = true;
            // The database checkpoint is intentionally retained for recovery
            // on the next boot; never claim a failed lifecycle transition ended.
            logToFile("ERROR", "Error archiving and ending session", error);
        }

        try {
            closeDatabase();
            logToFile("INFO", "Database closed");
        } catch (error) {
            shutdownFailed = true;
            logToFile("ERROR", "Error flushing/closing database", error);
        }

        try {
            await server.close();
            logToFile("INFO", "MCP server closed");
        } catch (error) {
            shutdownFailed = true;
            logToFile("ERROR", "Error closing MCP server", error);
        }

        // Exit only after lifecycle/database errors have been recorded. A
        // non-zero status makes shutdown failures observable to the host.
        setTimeout(() => {
            process.exit(shutdownFailed ? 1 : 0);
        }, 200);
    };

    process.on("SIGINT", () => shutdown("SIGINT"));
    process.on("SIGTERM", () => shutdown("SIGTERM"));
    process.on("SIGHUP", () => shutdown("SIGHUP"));
    // MCP stdio hosts commonly signal normal teardown by closing stdin rather
    // than sending a POSIX signal. Treat it as the same lifecycle checkpoint.
    process.stdin.once("end", () => { void shutdown("stdin closed"); });

    // Step 8: Handle uncaught errors — log but do NOT crash
    process.on("uncaughtException", (error: Error) => {
        logToFile("UNCAUGHT_EXCEPTION", error.message, error);
        // Do NOT call process.exit() — the MCP server should remain alive
    });

    process.on("unhandledRejection", (reason: unknown) => {
        logToFile("UNHANDLED_REJECTION", "Promise rejection", reason);
        // Do NOT call process.exit() — the MCP server should remain alive
    });

    // Step 9: Connect to stdio transport
    try {
        await server.connect(transport);
        logToFile("INFO", "MCP transport connected — server is ready");
    } catch (error) {
        logToFile("FATAL", "Failed to connect MCP transport", error);
        process.exit(1);
    }

    // Step 10: Hide console window on Windows (AFTER transport is connected)
    // This runs async so it can't block or crash the startup sequence.
    // The window will be visible for ~1-2 seconds then disappear.
    if (process.platform === "win32") {
        try {
            const ps = spawn("powershell.exe", [
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                `Add-Type -Name W -Namespace HideConsole -MemberDefinition '[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'; [HideConsole.W]::ShowWindow([HideConsole.W]::GetConsoleWindow(), 0)`,
            ], {
                windowsHide: true,
                stdio: "ignore",
            });
            ps.on("error", () => { }); // Swallow errors — window hiding is cosmetic
            ps.unref(); // Don't keep Node alive waiting for PowerShell
            logToFile("INFO", "Console window hide requested");
        } catch {
            logToFile("WARN", "Failed to hide console window (non-fatal)");
        }
    }
}

main().catch((error) => {
    logToFile("FATAL", "Fatal error in main()", error);
    process.exit(1);
});
