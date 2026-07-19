import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { CONFIG_FILE_NAME, DATA_DIR_ENV, resetConfig } from "../src/config.js";
import { assembleContext } from "../src/context-assembler.js";
import { closeDatabase, initDatabase } from "../src/database.js";
import { countTokens } from "../src/token-counter.js";
import { clearAllWorkingMemory, storeMemory } from "../src/memory-manager.js";
import { endCurrentSession, startSession } from "../src/session.js";

const originalDataDir = process.env[DATA_DIR_ENV];

test("retrieval ranks and packs isolated Tier 0 entries without embeddings", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-retrieval-"));
    try {
        // The test configuration ensures storing events never initializes or
        // downloads the local embedding model.
        writeFileSync(
            join(dataDir, CONFIG_FILE_NAME),
            JSON.stringify({ embedding: { provider: "none" } })
        );
        process.env[DATA_DIR_ENV] = dataDir;
        resetConfig();
        await initDatabase();
        clearAllWorkingMemory();

        await startSession();
        const relevant = "The payment retry policy uses exponential backoff after gateway timeout failures and records each retry attempt.";
        const unrelated = "The greenhouse irrigation schedule waters tomato plants at sunrise while checking soil moisture and rain forecasts.";
        await storeMemory(relevant, "event", 0.95);
        await storeMemory(unrelated, "event", 0.2);

        const broad = await assembleContext("payment retry exponential backoff", 2_000);
        assert.ok(broad.text.indexOf(relevant) >= 0);
        assert.ok(broad.text.indexOf(unrelated) >= 0);
        assert.ok(
            broad.text.indexOf(relevant) < broad.text.indexOf(unrelated),
            "the targeted entry must rank before an unrelated current-session entry"
        );

        const packed = await assembleContext("payment retry exponential backoff", broad.budgetUsed - 1);
        assert.equal(packed.candidatesSelected, 1, "Tier 0 candidates must be packed one entry at a time");
        assert.ok(packed.text.includes(relevant));
        assert.ok(!packed.text.includes(unrelated));
        assert.ok(packed.totalTokens <= broad.budgetUsed - 1, "rendered context must honor the token budget");

        const filtered = await assembleContext("payment retry", 2_000, {
            sourceTypes: ["event"],
            minConfidence: 0.8,
        });
        assert.ok(filtered.text.includes(relevant));
        assert.ok(!filtered.text.includes(unrelated), "min_confidence must affect eligible candidates");
        await assert.rejects(
            () => assembleContext("payment", 2_000, { sourceTypes: ["unsupported"] }),
            /Unsupported memory type filter/
        );

        // endCurrentSession intentionally does not clear the in-memory array;
        // retrieval still must not expose those entries without an active ID.
        endCurrentSession();
        const inactive = await assembleContext("payment retry", 2_000);
        assert.equal(inactive.candidatesSelected, 0);
        assert.ok(!inactive.text.includes("payment retry policy"));
    } finally {
        endCurrentSession();
        clearAllWorkingMemory();
        closeDatabase();
        resetConfig();
        if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
        else process.env[DATA_DIR_ENV] = originalDataDir;
        rmSync(dataDir, { recursive: true, force: true });
    }
});

test("retrieval remains bounded for a large current-session input", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-retrieval-"));
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
        for (let index = 0; index < 160; index++) {
            const marker = index === 156 ? "priorityretrievalmarker" : `ordinarymarker${index}`;
            const content = Array.from(
                { length: 18 },
                (_, word) => `${marker} detail${word} record${index}`
            ).join(" ");
            // Exercise both candidate sources at large input size. The target
            // is a Tier 1 entry and its exact marker is a query term.
            await storeMemory(content, index % 4 === 0 ? "summary" : "event");
        }

        const result = await assembleContext("priorityretrievalmarker", 256);
        assert.equal(result.candidatesConsidered, 160);
        assert.ok(result.candidatesSelected > 0);
        assert.ok(result.text.includes("priorityretrievalmarker"), "the lexically matched entry must survive bounded packing");
        assert.ok(result.sourceCounts.current_session > 0, "the large workload must include Tier 1 candidates");
        assert.ok(result.totalTokens <= 256);
        assert.equal(result.totalTokens, countTokens(result.text));
        assert.equal(result.budgetUsed, result.totalTokens);
        assert.equal(result.budgetRemaining, 256 - result.totalTokens);
    } finally {
        endCurrentSession();
        clearAllWorkingMemory();
        closeDatabase();
        resetConfig();
        if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
        else process.env[DATA_DIR_ENV] = originalDataDir;
        rmSync(dataDir, { recursive: true, force: true });
    }
});

test("retrieval deduplicates near-duplicates without discarding the more relevant entry", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-retrieval-"));
    try {
        writeFileSync(
            join(dataDir, CONFIG_FILE_NAME),
            JSON.stringify({ embedding: { provider: "none" } })
        );
        process.env[DATA_DIR_ENV] = dataDir;
        resetConfig();
        await initDatabase();
        clearAllWorkingMemory();

        await startSession();
        const sharedTerms = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
            "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
        ].join(" ");
        const lessRelevant = `${sharedTerms} irrelevantmarker`;
        const targeted = `${sharedTerms} priorityretrievalmarker`;
        await storeMemory(lessRelevant, "event");
        await storeMemory(targeted, "event");

        const result = await assembleContext("priorityretrievalmarker", 2_000);
        assert.equal(result.candidatesConsidered, 2);
        assert.equal(result.candidatesSelected, 1);
        assert.ok(result.text.includes(targeted));
        assert.ok(!result.text.includes(lessRelevant));
    } finally {
        endCurrentSession();
        clearAllWorkingMemory();
        closeDatabase();
        resetConfig();
        if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
        else process.env[DATA_DIR_ENV] = originalDataDir;
        rmSync(dataDir, { recursive: true, force: true });
    }
});

test("retrieval persists summary confidence and bounds empty responses", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-retrieval-"));
    try {
        writeFileSync(
            join(dataDir, CONFIG_FILE_NAME),
            JSON.stringify({ embedding: { provider: "none" } })
        );
        process.env[DATA_DIR_ENV] = dataDir;
        resetConfig();
        await initDatabase();
        clearAllWorkingMemory();

        await startSession();
        const lowConfidenceSummary = "The tentative deployment rollback plan may need approval after reviewing the incomplete incident timeline and service health evidence.";
        await storeMemory(lowConfidenceSummary, "summary", 0.2);

        const filtered = await assembleContext("deployment rollback plan", 256, {
            sourceTypes: ["summary"],
            minConfidence: 0.8,
        });
        assert.equal(filtered.candidatesSelected, 0);
        assert.ok(!filtered.text.includes(lowConfidenceSummary));

        const empty = await assembleContext("deployment rollback", 1, { sourceTypes: ["core"] });
        assert.equal(empty.candidatesSelected, 0);
        assert.ok(empty.totalTokens <= 1);
        assert.equal(empty.totalTokens, countTokens(empty.text));
        assert.equal(empty.budgetUsed, empty.totalTokens);

        endCurrentSession();
        const inactive = await assembleContext("deployment rollback", 1);
        assert.equal(inactive.candidatesSelected, 0);
        assert.ok(inactive.totalTokens <= 1);
        assert.equal(inactive.totalTokens, countTokens(inactive.text));
    } finally {
        endCurrentSession();
        clearAllWorkingMemory();
        closeDatabase();
        resetConfig();
        if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
        else process.env[DATA_DIR_ENV] = originalDataDir;
        rmSync(dataDir, { recursive: true, force: true });
    }
});
