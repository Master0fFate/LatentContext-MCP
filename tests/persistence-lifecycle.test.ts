import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { CONFIG_FILE_NAME, DATA_DIR_ENV, resetConfig } from "../src/config.js";
import {
    closeDatabase,
    getSessionById,
    insertSession,
    insertSummary,
    getSummariesByTierAndSession,
    getVectorCount,
    initDatabase,
} from "../src/database.js";
import {
    archiveWorkingMemory,
    clearAllWorkingMemory,
    compressMemory,
    forgetMemory,
    storeMemory,
} from "../src/memory-manager.js";
import {
    endCurrentSession,
    endCurrentSessionWithArchive,
    getCurrentSessionIdOrNull,
    startSession,
} from "../src/session.js";
import { addVectorDirect, searchVectorsByEmbedding } from "../src/vector-store.js";

const originalDataDir = process.env[DATA_DIR_ENV];

function useTemporaryStorage(): string {
    const dataDir = mkdtempSync(join(tmpdir(), "latentcontext-persistence-"));
    writeFileSync(
        join(dataDir, CONFIG_FILE_NAME),
        JSON.stringify({ embedding: { provider: "none" } })
    );
    process.env[DATA_DIR_ENV] = dataDir;
    resetConfig();
    return dataDir;
}

function restoreStorage(dataDir: string): void {
    clearAllWorkingMemory();
    closeDatabase();
    resetConfig();
    if (originalDataDir === undefined) delete process.env[DATA_DIR_ENV];
    else process.env[DATA_DIR_ENV] = originalDataDir;
    rmSync(dataDir, { recursive: true, force: true });
}

test("Tier 0 checkpoints flush, recover after reopen, and forget removes cached vectors", async () => {
    const dataDir = useTemporaryStorage();
    try {
        await initDatabase();
        const started = await startSession();
        const content = "The persisted event checkpoint records shutdown recovery details and keeps database vectors synchronized after a memory is forgotten.";
        const stored = await storeMemory(content, "event", 0.8);

        assert.equal(getSummariesByTierAndSession(0, started.sessionId).length, 1);
        assert.equal(getVectorCount(), 1);
        assert.equal(searchVectorsByEmbedding(new Array(384).fill(0)).length, 1);

        // Simulate the durable portion of a process restart. The event must
        // remain in SQLite independently of its in-memory working cache.
        endCurrentSession();
        closeDatabase();
        clearAllWorkingMemory();
        await initDatabase();
        assert.equal(getSummariesByTierAndSession(0, started.sessionId).length, 1);
        assert.equal(getVectorCount(), 1);
        assert.equal(searchVectorsByEmbedding(new Array(384).fill(0)).length, 1);

        const corrected = "The corrected event checkpoint updates vector content before the lifecycle state can be retrieved again after persistence recovery.";
        assert.match(await forgetMemory(stored.memoryId, "correct", corrected), /^Corrected working memory entry/);
        assert.equal(getSummariesByTierAndSession(0, started.sessionId)[0].content, corrected);
        assert.equal(searchVectorsByEmbedding(new Array(384).fill(0))[0].contentPreview, corrected);

        assert.match(await forgetMemory(stored.memoryId, "delete"), /^Deleted working memory entry/);
        assert.equal(getSummariesByTierAndSession(0, started.sessionId).length, 0);
        assert.equal(getVectorCount(), 0);
        assert.equal(searchVectorsByEmbedding(new Array(384).fill(0)).length, 0);
    } finally {
        restoreStorage(dataDir);
    }
});

test("summary mutations checkpoint content and vector confidence together", async () => {
    const dataDir = useTemporaryStorage();
    try {
        await initDatabase();
        await startSession();
        const stored = await storeMemory(
            "The durable summary records the release recovery policy, including the required database checkpoint and vector cache replacement steps.",
            "summary",
            0.7
        );

        closeDatabase();
        await initDatabase();
        assert.equal(getSummariesByTierAndSession(1, stored.sessionId!).length, 1);
        assert.equal(getVectorCount(), 1);

        assert.match(await forgetMemory(stored.memoryId, "deprecate"), /^Deprecated memory/);
        const deprecated = getSummariesByTierAndSession(1, stored.sessionId!)[0];
        assert.match(deprecated.content, /^\[DEPRECATED\]/);
        const vector = searchVectorsByEmbedding(new Array(384).fill(0))[0];
        assert.match(vector.contentPreview, /^\[DEPRECATED\]/);
        assert.equal(vector.confidence, 0.1);

        assert.match(await forgetMemory(stored.memoryId, "delete"), /^Deleted memory/);
        assert.equal(getSummariesByTierAndSession(1, stored.sessionId!).length, 0);
        assert.equal(getVectorCount(), 0);
        endCurrentSession();
    } finally {
        restoreStorage(dataDir);
    }
});

test("startup recovers an orphaned active session from its durable Tier 0 checkpoint", async () => {
    const dataDir = useTemporaryStorage();
    const orphanId = "orphaned-session";
    const content = "The orphaned lifecycle event was checkpointed before an unexpected process exit and must be archived during startup recovery.";
    try {
        await initDatabase();
        insertSession({
            id: orphanId,
            started_at: new Date().toISOString(),
            ended_at: null,
            metadata: "{}",
        });
        insertSummary({
            id: "orphaned-event",
            tier: 0,
            content,
            token_count: 20,
            session_id: orphanId,
            source_ids: "[]",
            metadata: JSON.stringify({ type: "event", confidence: 1 }),
        });
        addVectorDirect(new Array(384).fill(0), content, "orphaned-event", "event");

        // Simulate a new process: only the SQLite image survives, while the
        // module-level lifecycle and working-memory caches are empty.
        closeDatabase();
        await initDatabase();
        const recovered = await startSession(archiveWorkingMemory);

        assert.equal(recovered.previousSessionId, orphanId);
        assert.equal(recovered.previousSessionArchived, true);
        assert.equal(getSessionById(orphanId)?.ended_at === null, false);
        assert.equal(getSummariesByTierAndSession(0, orphanId).length, 0);
        assert.equal(getSummariesByTierAndSession(1, orphanId).length, 1);
        assert.equal(getVectorCount(), 1, "recovery replaces the old vector instead of duplicating it");
        endCurrentSession();
    } finally {
        restoreStorage(dataDir);
    }
});

test("session transitions archive durable Tier 0 and lifecycle failures remain retryable", async () => {
    const dataDir = useTemporaryStorage();
    try {
        await initDatabase();
        const first = await startSession();
        await storeMemory(
            "The first session contains a durable working event that must be archived before the lifecycle record is marked ended.",
            "event"
        );

        await assert.rejects(
            () => startSession(async () => { throw new Error("archive storage unavailable"); }),
            /archive storage unavailable/
        );
        assert.equal(getCurrentSessionIdOrNull(), first.sessionId);
        assert.equal(getSessionById(first.sessionId)?.ended_at, null);

        const second = await startSession(archiveWorkingMemory);
        assert.equal(getSessionById(first.sessionId)?.ended_at === null, false);
        assert.equal(getSummariesByTierAndSession(0, first.sessionId).length, 0);
        assert.equal(getSummariesByTierAndSession(1, first.sessionId).length, 1);
        assert.equal(getVectorCount(), 1, "the archive replaces, rather than leaks, Tier 0 vectors");
        assert.equal(getCurrentSessionIdOrNull(), second.sessionId);

        await storeMemory(
            "The second session event verifies shutdown archives its acknowledged state before closing the database checkpoint.",
            "event"
        );
        await compressMemory("working");
        // The archived predecessor is durable, but cannot be consolidated
        // into a current-session record and therefore cannot leak on retrieve.
        assert.match(await compressMemory("session"), /^Not enough Tier 1 summaries/);
        assert.equal(getSummariesByTierAndSession(1, first.sessionId).length, 1);
        assert.equal(getSummariesByTierAndSession(1, second.sessionId).length, 1);

        await endCurrentSessionWithArchive(archiveWorkingMemory);
        assert.equal(getSessionById(second.sessionId)?.ended_at === null, false);
        assert.equal(getSummariesByTierAndSession(0, second.sessionId).length, 0);
        assert.equal(getSummariesByTierAndSession(1, second.sessionId).length, 1);
    } finally {
        restoreStorage(dataDir);
    }
});
