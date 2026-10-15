import { availableParallelism } from "node:os";
import { extname } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import type { ParsedFile } from "./code-parser.js";
import type { ScanResult } from "./code-scan.js";
import type { ParseItem, WorkerTask } from "./code-worker.js";

interface Slot {
    worker: Worker;
    busy: boolean;
    alive: boolean;
    pending: { resolve: (value: unknown) => void; reject: (error: Error) => void } | null;
}

const WORKER_URL = new URL(`./code-worker${extname(fileURLToPath(import.meta.url))}`, import.meta.url);
const POOL_SIZE = Math.max(1, Math.min(availableParallelism() - 1, 8));
const PARSE_CHUNK = 32;
const IDLE_MS = 30_000;

const slots: Slot[] = [];
const waiting: Array<(slot: Slot) => void> = [];
let idleTimer: NodeJS.Timeout | null = null;

function retire(slot: Slot, error: Error): void {
    if (!slot.alive) return;
    slot.alive = false;
    slots.splice(slots.indexOf(slot), 1);
    slot.pending?.reject(error);
    slot.pending = null;
}

function spawn(): Slot {
    const slot: Slot = { worker: new Worker(WORKER_URL), busy: true, alive: true, pending: null };
    slot.worker.unref();
    slot.worker.on("message", (message: { result?: unknown; error?: string }) => {
        const pending = slot.pending;
        slot.pending = null;
        if (message.error !== undefined) pending?.reject(new Error(message.error));
        else pending?.resolve(message.result);
    });
    slot.worker.on("error", (error) => retire(slot, error));
    slot.worker.on("exit", (code) => retire(slot, new Error(`Code worker exited with code ${code}`)));
    slots.push(slot);
    return slot;
}

function take(): Promise<Slot> {
    if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
    }
    const idle = slots.find((slot) => !slot.busy);
    if (idle) {
        idle.busy = true;
        return Promise.resolve(idle);
    }
    if (slots.length < POOL_SIZE) return Promise.resolve(spawn());
    return new Promise((resolve) => waiting.push(resolve));
}

function release(slot: Slot): void {
    const next = waiting.shift();
    if (next) {
        next(slot.alive ? slot : spawn());
        return;
    }
    slot.busy = false;
    if (!idleTimer && slots.every((candidate) => !candidate.busy)) {
        idleTimer = setTimeout(shutdownPool, IDLE_MS);
        idleTimer.unref();
    }
}

async function run<T>(task: WorkerTask): Promise<T> {
    const slot = await take();
    try {
        return await new Promise<T>((resolve, reject) => {
            slot.pending = { resolve: resolve as (value: unknown) => void, reject };
            slot.worker.postMessage(task);
        });
    } finally {
        release(slot);
    }
}

export function scanInWorker(root: string, maxFiles: number): Promise<ScanResult> {
    return run<ScanResult>({ kind: "scan", root, maxFiles });
}

export async function parseInWorkers(
    root: string,
    files: readonly ParseItem[],
    onParsed: (item: ParseItem, parsed: ParsedFile | null) => void
): Promise<void> {
    const chunks: ParseItem[][] = [];
    for (let i = 0; i < files.length; i += PARSE_CHUNK) chunks.push(files.slice(i, i + PARSE_CHUNK));
    await Promise.all(chunks.map(async (chunk) => {
        const results = await run<Array<ParsedFile | null>>({ kind: "parse", root, files: chunk });
        for (let i = 0; i < chunk.length; i++) onParsed(chunk[i], results[i]);
    }));
}

export function shutdownPool(): void {
    if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
    }
    for (const slot of [...slots]) {
        retire(slot, new Error("Code worker pool shut down"));
        void slot.worker.terminate();
    }
}
