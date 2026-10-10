import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Writable } from "node:stream";

// ---------------------------------------------------------------------------
// Stdio-safe diagnostics.
//
// The MCP stdio transport owns stdout: a single stray byte corrupts the
// JSON-RPC stream and hosts report "connection closed". Everything written by
// this process, including third-party libraries that bypass console, is
// therefore routed to a log file. Once a client has completed the MCP
// handshake, notable messages are additionally forwarded as
// `notifications/message` so hosts that surface server logs can show them.
// ---------------------------------------------------------------------------

export const LOG_FILE_NAME = "server.log";

/** MCP logging levels (RFC 5424 names), ordered from least to most severe. */
export type LogLevel =
    | "debug"
    | "info"
    | "notice"
    | "warning"
    | "error"
    | "critical"
    | "alert"
    | "emergency";

const LEVEL_SEVERITY: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    notice: 2,
    warning: 3,
    error: 4,
    critical: 5,
    alert: 6,
    emergency: 7,
};

export type LogSink = (level: LogLevel, message: string, data?: Record<string, unknown>) => void;

let _logPath: string | null = null;
let _sink: LogSink | null = null;
let _sinkMinimumLevel: LogLevel = "info";

/**
 * Create (and truncate) the log file inside the data directory. Logging is
 * best-effort: if the file cannot be created, diagnostics are discarded rather
 * than risking any write to the protocol stream.
 */
export function initLogFile(dataDir: string): string | null {
    try {
        if (!existsSync(dataDir)) {
            mkdirSync(dataDir, { recursive: true });
        }
        _logPath = join(dataDir, LOG_FILE_NAME);
        writeFileSync(_logPath, `[${new Date().toISOString()}] [info] LatentContext MCP server starting\n`);
    } catch {
        _logPath = null;
    }
    return _logPath;
}

export function getLogPath(): string | null {
    return _logPath;
}

/**
 * Forward log records at or above `minimumLevel` to an additional sink, for
 * example the MCP logging notification channel. Sink failures are swallowed
 * and never logged, so a broken sink cannot recurse.
 */
export function setLogSink(sink: LogSink | null, minimumLevel: LogLevel = "info"): void {
    _sink = sink;
    _sinkMinimumLevel = minimumLevel;
}

function describeError(error: unknown): string {
    if (error instanceof Error) {
        return error.stack ? `${error.message}\n${error.stack}` : error.message;
    }
    return String(error);
}

export function log(level: LogLevel, message: string, error?: unknown): void {
    const detail = error === undefined ? "" : ` | ${describeError(error)}`;

    if (_logPath) {
        try {
            appendFileSync(_logPath, `[${new Date().toISOString()}] [${level}] ${message}${detail}\n`);
        } catch {
            // Never fail (or write anywhere else) because logging failed.
        }
    }

    if (_sink && LEVEL_SEVERITY[level] >= LEVEL_SEVERITY[_sinkMinimumLevel]) {
        try {
            _sink(level, message, error === undefined ? undefined : { error: describeError(error) });
        } catch {
            // A failing sink must not take the server down or loop.
        }
    }
}

/**
 * Reserve the real stdout writer for the MCP transport, then redirect every
 * other console/stdout/stderr write into the log file.
 *
 * The returned object exposes exactly the surface `StdioServerTransport`
 * uses (`write` and `once`), bound to the original stream before patching.
 */
export function captureProcessOutput(): Writable {
    const originalWrite = process.stdout.write.bind(process.stdout);
    const originalOnce = process.stdout.once.bind(process.stdout);
    const protocolStdout = { write: originalWrite, once: originalOnce } as unknown as Writable;

    const route = (level: LogLevel) => (...args: unknown[]) => {
        log(level, args.map((arg) => (typeof arg === "string" ? arg : describeError(arg))).join(" "));
    };
    console.log = route("info");
    console.info = route("info");
    console.debug = route("debug");
    console.warn = route("warning");
    console.error = route("error");
    console.trace = route("debug");

    const redirect = (level: LogLevel): typeof process.stdout.write => ((
        chunk: string | Uint8Array,
        encodingOrCallback?: BufferEncoding | ((error?: Error | null) => void),
        callback?: (error?: Error | null) => void
    ): boolean => {
        const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8");
        if (text.trim().length > 0) log(level, text.trimEnd());
        const done = typeof encodingOrCallback === "function" ? encodingOrCallback : callback;
        if (done) done(null);
        return true;
    }) as typeof process.stdout.write;

    process.stdout.write = redirect("info");
    process.stderr.write = redirect("warning");

    return protocolStdout;
}
