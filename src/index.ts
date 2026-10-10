#!/usr/bin/env node

import { spawn } from "node:child_process";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { getConfigSource, loadConfig } from "./config.js";
import { closeDatabase, initDatabase } from "./database.js";
import { captureProcessOutput, initLogFile, log, setLogSink } from "./logger.js";
import { archiveWorkingMemory } from "./memory-manager.js";
import { createServer } from "./server.js";
import { endCurrentSessionWithArchive, startSession } from "./session.js";

/** Set to `0` or `false` to keep the console window visible on Windows. */
export const HIDE_CONSOLE_ENV = "LATENTCONTEXT_HIDE_CONSOLE";

function envFlagDisabled(value: string | undefined): boolean {
    const normalized = value?.trim().toLowerCase();
    return normalized === "0" || normalized === "false" || normalized === "no";
}

/**
 * Hosts on Windows sometimes launch stdio servers with a visible console
 * window. Hiding it is cosmetic, runs detached, and can never block startup.
 */
function hideWindowsConsole(): void {
    if (process.platform !== "win32" || envFlagDisabled(process.env[HIDE_CONSOLE_ENV])) return;
    try {
        const powershell = spawn("powershell.exe", [
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            `Add-Type -Name W -Namespace HideConsole -MemberDefinition '[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'; [HideConsole.W]::ShowWindow([HideConsole.W]::GetConsoleWindow(), 0)`,
        ], { windowsHide: true, stdio: "ignore" });
        powershell.on("error", () => { /* cosmetic */ });
        powershell.unref();
        log("debug", "Console window hide requested");
    } catch {
        log("warning", "Failed to hide console window (non-fatal)");
    }
}

async function main(): Promise<void> {
    // 1. Configuration and log file come first so every later diagnostic has
    //    somewhere safe to go.
    const config = loadConfig();
    initLogFile(config.storage.dataDir);

    // 2. Reserve stdout for the protocol and silence everything else.
    const protocolStdout = captureProcessOutput();
    log("info", `Config source: ${getConfigSource() ?? "defaults/env"}`);
    log("info", `Data directory: ${config.storage.dataDir}`);

    // 3. Storage is mandatory; the server cannot operate without it.
    try {
        await initDatabase();
        log("info", "Database initialized");
    } catch (error) {
        log("critical", "Failed to initialize database", error);
        process.exit(1);
    }

    // 4. Optionally open a session immediately so the first memory_store has
    //    somewhere to go even if the host never calls session_start.
    if (config.session.autoStartOnBoot) {
        try {
            const session = await startSession(archiveWorkingMemory);
            log("info", `Auto-started session ${session.sessionId}`);
        } catch (error) {
            // Non-fatal: the session_start tool can be called by the model.
            log("error", "Failed to auto-start session (non-fatal)", error);
        }
    }

    // 5. Build the MCP server and forward notable log records to the client
    //    once the handshake has completed (notifications before that are a
    //    protocol violation).
    const server = createServer();
    server.server.oninitialized = () => {
        const client = server.server.getClientVersion();
        log("info", `Client initialized: ${client?.name ?? "unknown"} ${client?.version ?? ""}`.trim());
        setLogSink((level, message, data) => {
            server.server
                .sendLoggingMessage({ level, logger: "latentcontext", data: { message, ...data } })
                .catch(() => undefined);
        }, "info");
    };

    // 6. Graceful shutdown. Never call process.exit() while the transport is
    //    still active: hosts report that as "connection closed: EOF".
    let isShuttingDown = false;
    const shutdown = async (reason: string): Promise<void> => {
        if (isShuttingDown) return;
        isShuttingDown = true;
        let shutdownFailed = false;
        log("info", `Shutdown requested (${reason})`);

        try {
            await endCurrentSessionWithArchive(archiveWorkingMemory);
            log("info", "Session archived and ended");
        } catch (error) {
            shutdownFailed = true;
            // The durable checkpoint is retained for recovery on next boot.
            log("error", "Error archiving and ending session", error);
        }

        try {
            closeDatabase();
            log("info", "Database closed");
        } catch (error) {
            shutdownFailed = true;
            log("error", "Error flushing/closing database", error);
        }

        setLogSink(null);
        try {
            await server.close();
            log("info", "MCP server closed");
        } catch (error) {
            shutdownFailed = true;
            log("error", "Error closing MCP server", error);
        }

        // A non-zero status makes shutdown failures observable to the host.
        setTimeout(() => process.exit(shutdownFailed ? 1 : 0), 200);
    };

    process.on("SIGINT", () => void shutdown("SIGINT"));
    process.on("SIGTERM", () => void shutdown("SIGTERM"));
    process.on("SIGHUP", () => void shutdown("SIGHUP"));
    // Stdio hosts commonly signal teardown by closing stdin rather than
    // sending a POSIX signal. Treat it as the same lifecycle checkpoint.
    process.stdin.once("end", () => void shutdown("stdin closed"));

    // 7. Log, but never crash on, unexpected errors: the host owns the
    //    process lifetime.
    process.on("uncaughtException", (error: Error) => {
        log("error", "Uncaught exception", error);
    });
    process.on("unhandledRejection", (reason: unknown) => {
        log("error", "Unhandled promise rejection", reason);
    });

    // 8. Connect the transport.
    try {
        await server.connect(new StdioServerTransport(process.stdin, protocolStdout));
        log("info", "MCP transport connected; server is ready");
    } catch (error) {
        log("critical", "Failed to connect MCP transport", error);
        process.exit(1);
    }

    hideWindowsConsole();
}

main().catch((error: unknown) => {
    log("critical", "Fatal error in main()", error);
    process.exit(1);
});
