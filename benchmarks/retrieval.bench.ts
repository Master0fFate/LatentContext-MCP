import { performance } from "node:perf_hooks";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CONFIG_FILE_NAME, DATA_DIR_ENV, resetConfig } from "../src/config.js";
import { assembleContext } from "../src/context-assembler.js";
import { closeDatabase, initDatabase } from "../src/database.js";
import { clearAllWorkingMemory, storeMemory } from "../src/memory-manager.js";
import { endCurrentSession, startSession } from "../src/session.js";

const entryCount = Number.parseInt(process.env.LATENTCONTEXT_BENCH_ENTRIES ?? "512", 10);
const runs = Number.parseInt(process.env.LATENTCONTEXT_BENCH_RUNS ?? "15", 10);

if (!Number.isInteger(entryCount) || entryCount <= 0 || !Number.isInteger(runs) || runs <= 0) {
    throw new Error("LATENTCONTEXT_BENCH_ENTRIES and LATENTCONTEXT_BENCH_RUNS must be positive integers.");
}

const TOPICS = [
    "deployment rollback",
    "token budget",
    "session isolation",
    "MCP transport",
    "retrieval ranking",
    "database migration",
    "incident response",
    "configuration loading",
];
const summaryEntryCount = Math.floor(entryCount / 4);
const workingEntryCount = entryCount - summaryEntryCount;

function entry(index: number): string {
    const topic = TOPICS[index % TOPICS.length];
    // Concise records resemble notes captured during several parallel
    // workstreams. Topic-local overlap exercises the index while distinct
    // checkpoint fields prevent synthetic near-duplicates.
    return `${topic}: checkpoint-${index}; owner-${index}; verification-${index}.`;
}

function terms(text: string): Set<string> {
    return new Set(
        text.toLocaleLowerCase().match(/[\p{L}\p{N}_-]+/gu)?.filter((term) => term.length > 1) ?? []
    );
}

/**
 * Compare the legacy all-pairs candidate scan with the inverted-index path.
 * This is deterministic evidence rather than a noisy wall-clock comparison:
 * the old path constructs two term sets and evaluates similarity per pair;
 * the current path constructs one set per candidate and evaluates only pairs
 * that share a term.
 */
function deduplicationWork(contents: string[]): {
    legacyPairChecks: number;
    legacyTermSetConstructions: number;
    indexedPairChecks: number;
    indexedTermSetConstructions: number;
} {
    const indexesByTerm = new Map<string, Set<number>>();
    let indexedPairChecks = 0;

    for (let index = 0; index < contents.length; index++) {
        const termSet = terms(contents[index]);
        const possiblePairs = new Set<number>();
        for (const term of termSet) {
            for (const previousIndex of indexesByTerm.get(term) ?? []) possiblePairs.add(previousIndex);
        }
        indexedPairChecks += possiblePairs.size;
        for (const term of termSet) {
            let indexes = indexesByTerm.get(term);
            if (!indexes) {
                indexes = new Set();
                indexesByTerm.set(term, indexes);
            }
            indexes.add(index);
        }
    }

    const legacyPairChecks = (contents.length * (contents.length - 1)) / 2;
    return {
        legacyPairChecks,
        legacyTermSetConstructions: legacyPairChecks * 2,
        indexedPairChecks,
        indexedTermSetConstructions: contents.length,
    };
}

function percentile(values: number[], percentileValue: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * percentileValue))];
}

const originalDataDir = process.env[DATA_DIR_ENV];
const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-retrieval-benchmark-"));
const workload = Array.from({ length: entryCount }, (_, index) => entry(index));

try {
    writeFileSync(
        join(dataDir, CONFIG_FILE_NAME),
        JSON.stringify({
            embedding: { provider: "none" },
            compression: { tier0OverflowThreshold: 1_000_000 },
        })
    );
    process.env[DATA_DIR_ENV] = dataDir;
    resetConfig();
    await initDatabase();
    clearAllWorkingMemory();
    await startSession();

    for (let index = 0; index < entryCount; index++) {
        await storeMemory(workload[index], index < workingEntryCount ? "event" : "summary");
    }

    // Warm the tokenizer, database statement paths, and JIT before measurement.
    const warmed = await assembleContext("retrieval ranking token budget", 8_000);
    if (warmed.candidatesConsidered !== entryCount || warmed.totalTokens > 8_000) {
        throw new Error("benchmark setup did not produce the expected bounded candidate workload");
    }
    const durationsMs: number[] = [];
    for (let run = 0; run < runs; run++) {
        const startedAt = performance.now();
        const result = await assembleContext(TOPICS[run % TOPICS.length], 8_000);
        durationsMs.push(performance.now() - startedAt);
        if (result.totalTokens > 8_000) throw new Error("retrieval exceeded its token budget");
    }

    const meanMs = durationsMs.reduce((sum, value) => sum + value, 0) / durationsMs.length;
    const work = deduplicationWork(workload);
    const pairReductionPercent = work.legacyPairChecks === 0
        ? 0
        : (1 - work.indexedPairChecks / work.legacyPairChecks) * 100;
    console.log(JSON.stringify({
        benchmark: "session-isolated-retrieval",
        entries: entryCount,
        workingEntries: workingEntryCount,
        sessionSummaries: summaryEntryCount,
        runs,
        meanMs: Number(meanMs.toFixed(2)),
        p95Ms: Number(percentile(durationsMs, 0.95).toFixed(2)),
        deduplicationWork: {
            ...work,
            pairReductionPercent: Number(pairReductionPercent.toFixed(2)),
        },
    }));
} finally {
    endCurrentSession();
    clearAllWorkingMemory();
    closeDatabase();
    resetConfig();
    if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
    else process.env[DATA_DIR_ENV] = originalDataDir;
    rmSync(dataDir, { recursive: true, force: true });
}
