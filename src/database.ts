import initSqlJs, { type Database as SqlJsDatabase } from "sql.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";
import { getConfig } from "./config.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EntityRow {
    id: string;
    label: string;
    entity_type: string;
    properties: string; // JSON
    created_at: string;
    updated_at: string;
    confidence: number;
    source_summary_id: string | null;
}

export interface RelationRow {
    id: string;
    subject_id: string;
    predicate: string;
    object_id: string;
    properties: string; // JSON
    temporal_start: string | null;
    temporal_end: string | null;
    confidence: number;
    source_summary_id: string | null;
}

export interface SummaryRow {
    id: string;
    tier: number;
    content: string;
    token_count: number;
    created_at: string;
    updated_at: string;
    session_id: string | null;
    source_ids: string; // JSON array
    metadata: string; // JSON
}

export interface VectorRow {
    id: string;
    source_id: string;
    source_type: string;
    content_preview: string;
    embedding: Uint8Array;
    dimensions: number;
    metadata: string; // JSON
    created_at: string;
    confidence: number;
}

export interface AccessLogRow {
    id: number;
    memory_id: string;
    memory_type: string;
    accessed_at: string;
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_summary_id TEXT
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    temporal_start TEXT,
    temporal_end TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_summary_id TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    tier INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    session_id TEXT,
    source_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'raw',
    content_preview TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL DEFAULT 384,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    accessed_at TEXT NOT NULL
);
`;

const INDEXES_SQL = `
CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);
CREATE INDEX IF NOT EXISTS idx_summaries_tier ON summaries(tier);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_vectors_source ON vectors(source_id);
CREATE INDEX IF NOT EXISTS idx_vectors_source_type ON vectors(source_type);
CREATE INDEX IF NOT EXISTS idx_access_log_memory ON access_log(memory_id);
`;

// ---------------------------------------------------------------------------
// Database singleton
// ---------------------------------------------------------------------------

let _db: SqlJsDatabase | null = null;
let _dbPath: string = "";
let _saveTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Initialize the database. Must be called before any operations.
 */
export async function initDatabase(): Promise<SqlJsDatabase> {
    if (_db) return _db;

    const config = getConfig();
    const dataDir = config.storage.dataDir;

    if (!existsSync(dataDir)) {
        mkdirSync(dataDir, { recursive: true });
    }

    _dbPath = join(dataDir, config.storage.sqliteFile);

    const SQL = await initSqlJs();

    // Load existing database or create new one
    if (existsSync(_dbPath)) {
        const fileBuffer = readFileSync(_dbPath);
        _db = new SQL.Database(fileBuffer);
    } else {
        _db = new SQL.Database();
    }

    // Run schema migration
    _db.run(SCHEMA_SQL);
    _db.run(INDEXES_SQL);

    // Save to disk
    saveDatabase();

    return _db;
}

/**
 * Get the database instance (must call initDatabase first).
 */
export function getDb(): SqlJsDatabase {
    if (!_db) {
        throw new Error("Database not initialized. Call initDatabase() first.");
    }
    return _db;
}

/**
 * Save the database to disk. Debounced to avoid excessive writes.
 */
export function saveDatabase(): void {
    if (!_db || !_dbPath) return;

    // Cancel pending save
    if (_saveTimer) {
        clearTimeout(_saveTimer);
    }

    // Debounce: save after 500ms of no writes
    _saveTimer = setTimeout(() => {
        if (!_db || !_dbPath) return;
        try {
            const data = _db.export();
            const buffer = Buffer.from(data);
            writeFileSync(_dbPath, buffer);
        } catch {
            // Ignore save errors
        }
    }, 500);
}

/**
 * Force immediate save to disk.
 */
export function saveDatabaseSync(): void {
    if (!_db || !_dbPath) return;
    if (_saveTimer) {
        clearTimeout(_saveTimer);
        _saveTimer = null;
    }
    try {
        const data = _db.export();
        const buffer = Buffer.from(data);
        writeFileSync(_dbPath, buffer);
    } catch {
        // Ignore save errors
    }
}

/**
 * Close the database.
 */
export function closeDatabase(): void {
    if (_saveTimer) {
        clearTimeout(_saveTimer);
        _saveTimer = null;
    }
    if (_db) {
        saveDatabaseSync();
        _db.close();
        _db = null;
    }
}

// ---------------------------------------------------------------------------
// Helper: run a query and return typed rows
// ---------------------------------------------------------------------------

function queryAll<T>(sql: string, params: unknown[] = []): T[] {
    const db = getDb();
    const stmt = db.prepare(sql);
    if (params.length > 0) stmt.bind(params);

    const results: T[] = [];
    while (stmt.step()) {
        const row = stmt.getAsObject() as T;
        results.push(row);
    }
    stmt.free();
    return results;
}

function queryOne<T>(sql: string, params: unknown[] = []): T | undefined {
    const results = queryAll<T>(sql, params);
    return results.length > 0 ? results[0] : undefined;
}

function runSql(sql: string, params: unknown[] = []): void {
    const db = getDb();
    db.run(sql, params);
    saveDatabase();
}

function now(): string {
    return new Date().toISOString();
}

// ---------------------------------------------------------------------------
// Entity operations
// ---------------------------------------------------------------------------

export function insertEntity(entity: Omit<EntityRow, "created_at" | "updated_at">): EntityRow {
    const ts = now();
    runSql(
        `INSERT INTO entities (id, label, entity_type, properties, created_at, updated_at, confidence, source_summary_id)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            entity.id,
            entity.label,
            entity.entity_type,
            entity.properties,
            ts,
            ts,
            entity.confidence,
            entity.source_summary_id ?? null,
        ]
    );
    return { ...entity, created_at: ts, updated_at: ts };
}

export function upsertEntity(
    id: string,
    label: string,
    entityType: string,
    properties: Record<string, unknown>,
    confidence: number = 1.0,
    sourceSummaryId: string | null = null
): EntityRow {
    const ts = now();
    const propsJson = JSON.stringify(properties);

    const existing = getEntityById(id);

    if (existing) {
        runSql(
            `UPDATE entities SET label = ?, entity_type = ?, properties = ?, updated_at = ?, confidence = ?, source_summary_id = ?
       WHERE id = ?`,
            [label, entityType, propsJson, ts, confidence, sourceSummaryId, id]
        );
        return {
            ...existing,
            label,
            entity_type: entityType,
            properties: propsJson,
            updated_at: ts,
            confidence,
            source_summary_id: sourceSummaryId,
        };
    }

    return insertEntity({
        id,
        label,
        entity_type: entityType,
        properties: propsJson,
        confidence,
        source_summary_id: sourceSummaryId,
    });
}

export function getEntityById(id: string): EntityRow | undefined {
    return queryOne<EntityRow>("SELECT * FROM entities WHERE id = ?", [id]);
}

export function getEntityByLabel(label: string): EntityRow | undefined {
    return queryOne<EntityRow>(
        "SELECT * FROM entities WHERE label = ? COLLATE NOCASE",
        [label]
    );
}

export function searchEntities(query: string): EntityRow[] {
    return queryAll<EntityRow>(
        "SELECT * FROM entities WHERE label LIKE ? COLLATE NOCASE ORDER BY confidence DESC",
        [`%${query}%`]
    );
}

export function getAllEntities(): EntityRow[] {
    return queryAll<EntityRow>("SELECT * FROM entities ORDER BY updated_at DESC");
}

export function deleteEntity(id: string): void {
    runSql("DELETE FROM entities WHERE id = ?", [id]);
}

export function getEntityCount(): number {
    const row = queryOne<{ "COUNT(*)": number }>("SELECT COUNT(*) FROM entities");
    return row ? row["COUNT(*)"] : 0;
}

// ---------------------------------------------------------------------------
// Relation operations
// ---------------------------------------------------------------------------

export function insertRelation(relation: RelationRow): void {
    runSql(
        `INSERT INTO relations (id, subject_id, predicate, object_id, properties, temporal_start, temporal_end, confidence, source_summary_id)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            relation.id,
            relation.subject_id,
            relation.predicate,
            relation.object_id,
            relation.properties,
            relation.temporal_start ?? null,
            relation.temporal_end ?? null,
            relation.confidence,
            relation.source_summary_id ?? null,
        ]
    );
}

export function upsertRelation(
    id: string,
    subjectId: string,
    predicate: string,
    objectId: string,
    properties: Record<string, unknown> = {},
    confidence: number = 1.0,
    temporalStart: string | null = null,
    temporalEnd: string | null = null,
    sourceSummaryId: string | null = null
): void {
    const propsJson = JSON.stringify(properties);
    const ts = now();

    // Check for existing relation with same subject-predicate
    const existing = queryOne<RelationRow>(
        "SELECT * FROM relations WHERE subject_id = ? AND predicate = ? AND temporal_end IS NULL",
        [subjectId, predicate]
    );

    if (existing && existing.object_id !== objectId) {
        // Mark old relation as ended (temporal update)
        runSql(
            "UPDATE relations SET temporal_end = ?, confidence = confidence * 0.5 WHERE id = ?",
            [ts, existing.id]
        );
    }

    // Insert or replace (using DELETE + INSERT pattern for sql.js)
    runSql("DELETE FROM relations WHERE id = ?", [id]);
    runSql(
        `INSERT INTO relations (id, subject_id, predicate, object_id, properties, temporal_start, temporal_end, confidence, source_summary_id)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            id,
            subjectId,
            predicate,
            objectId,
            propsJson,
            temporalStart || ts,
            temporalEnd ?? null,
            confidence,
            sourceSummaryId ?? null,
        ]
    );
}

export function getRelationsBySubject(subjectId: string): RelationRow[] {
    return queryAll<RelationRow>(
        "SELECT * FROM relations WHERE subject_id = ? AND temporal_end IS NULL ORDER BY confidence DESC",
        [subjectId]
    );
}

export function getRelationsByObject(objectId: string): RelationRow[] {
    return queryAll<RelationRow>(
        "SELECT * FROM relations WHERE object_id = ? AND temporal_end IS NULL ORDER BY confidence DESC",
        [objectId]
    );
}

export function getRelationsByPredicate(predicate: string): RelationRow[] {
    return queryAll<RelationRow>(
        "SELECT * FROM relations WHERE predicate = ? COLLATE NOCASE AND temporal_end IS NULL ORDER BY confidence DESC",
        [predicate]
    );
}

export function getAllRelations(): RelationRow[] {
    return queryAll<RelationRow>(
        "SELECT * FROM relations WHERE temporal_end IS NULL ORDER BY confidence DESC"
    );
}

export function deleteRelation(id: string): void {
    runSql("DELETE FROM relations WHERE id = ?", [id]);
}

export function getRelationCount(): number {
    const row = queryOne<{ "COUNT(*)": number }>(
        "SELECT COUNT(*) FROM relations WHERE temporal_end IS NULL"
    );
    return row ? row["COUNT(*)"] : 0;
}

// ---------------------------------------------------------------------------
// Summary operations
// ---------------------------------------------------------------------------

export function insertSummary(summary: Omit<SummaryRow, "created_at" | "updated_at">): SummaryRow {
    const ts = now();
    runSql(
        `INSERT INTO summaries (id, tier, content, token_count, created_at, updated_at, session_id, source_ids, metadata)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            summary.id,
            summary.tier,
            summary.content,
            summary.token_count,
            ts,
            ts,
            summary.session_id ?? null,
            summary.source_ids,
            summary.metadata,
        ]
    );
    return { ...summary, created_at: ts, updated_at: ts };
}

export function updateSummaryContent(id: string, content: string, tokenCount: number): void {
    const ts = now();
    runSql(
        "UPDATE summaries SET content = ?, token_count = ?, updated_at = ? WHERE id = ?",
        [content, tokenCount, ts, id]
    );
}

export function getSummariesByTier(tier: number): SummaryRow[] {
    return queryAll<SummaryRow>(
        "SELECT * FROM summaries WHERE tier = ? ORDER BY created_at DESC",
        [tier]
    );
}

export function getSummaryById(id: string): SummaryRow | undefined {
    return queryOne<SummaryRow>("SELECT * FROM summaries WHERE id = ?", [id]);
}

export function getSummariesBySession(sessionId: string): SummaryRow[] {
    return queryAll<SummaryRow>(
        "SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC",
        [sessionId]
    );
}

export function deleteSummary(id: string): void {
    runSql("DELETE FROM summaries WHERE id = ?", [id]);
}

export function getSummaryCountByTier(): Record<number, number> {
    const rows = queryAll<{ tier: number; "COUNT(*)": number }>(
        "SELECT tier, COUNT(*) FROM summaries GROUP BY tier"
    );
    const result: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    for (const row of rows) {
        result[row.tier] = row["COUNT(*)"];
    }
    return result;
}

export function getTotalSummaryTokens(): number {
    const row = queryOne<{ total: number }>(
        "SELECT COALESCE(SUM(token_count), 0) as total FROM summaries"
    );
    return row ? row.total : 0;
}

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

export function insertVector(vector: Omit<VectorRow, "created_at">): void {
    const ts = now();
    // Delete existing if same id (upsert semantics)
    runSql("DELETE FROM vectors WHERE id = ?", [vector.id]);
    runSql(
        `INSERT INTO vectors (id, source_id, source_type, content_preview, embedding, dimensions, metadata, created_at, confidence)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            vector.id,
            vector.source_id,
            vector.source_type,
            vector.content_preview,
            vector.embedding,
            vector.dimensions,
            vector.metadata,
            ts,
            vector.confidence,
        ]
    );
}

export function getVectorById(id: string): VectorRow | undefined {
    return queryOne<VectorRow>("SELECT * FROM vectors WHERE id = ?", [id]);
}

export function getAllVectors(): VectorRow[] {
    return queryAll<VectorRow>("SELECT * FROM vectors ORDER BY created_at DESC");
}

export function getVectorsBySourceType(sourceType: string): VectorRow[] {
    return queryAll<VectorRow>(
        "SELECT * FROM vectors WHERE source_type = ? ORDER BY created_at DESC",
        [sourceType]
    );
}

export function deleteVector(id: string): void {
    runSql("DELETE FROM vectors WHERE id = ?", [id]);
}

export function deleteVectorsBySourceId(sourceId: string): void {
    runSql("DELETE FROM vectors WHERE source_id = ?", [sourceId]);
}

export function getVectorCount(): number {
    const row = queryOne<{ "COUNT(*)": number }>("SELECT COUNT(*) FROM vectors");
    return row ? row["COUNT(*)"] : 0;
}

// ---------------------------------------------------------------------------
// Access log operations
// ---------------------------------------------------------------------------

export function logAccess(memoryId: string, memoryType: string): void {
    const ts = now();
    runSql(
        "INSERT INTO access_log (memory_id, memory_type, accessed_at) VALUES (?, ?, ?)",
        [memoryId, memoryType, ts]
    );
}

export function getAccessFrequency(memoryId: string): number {
    const row = queryOne<{ "COUNT(*)": number }>(
        "SELECT COUNT(*) FROM access_log WHERE memory_id = ?",
        [memoryId]
    );
    return row ? row["COUNT(*)"] : 0;
}

export function getTopAccessedIds(limit: number = 20): { memory_id: string; count: number }[] {
    return queryAll<{ memory_id: string; count: number }>(
        "SELECT memory_id, COUNT(*) as count FROM access_log GROUP BY memory_id ORDER BY count DESC LIMIT ?",
        [limit]
    );
}

// ---------------------------------------------------------------------------
// Bulk / utility operations
// ---------------------------------------------------------------------------

export function getDatabaseStats(): {
    entities: number;
    relations: number;
    summaries: Record<number, number>;
    vectors: number;
    totalSummaryTokens: number;
} {
    return {
        entities: getEntityCount(),
        relations: getRelationCount(),
        summaries: getSummaryCountByTier(),
        vectors: getVectorCount(),
        totalSummaryTokens: getTotalSummaryTokens(),
    };
}

export function applyConfidenceDecay(decayRate: number): void {
    runSql(
        "UPDATE entities SET confidence = confidence * (1.0 - ?) WHERE confidence > 0.1",
        [decayRate]
    );
    runSql(
        "UPDATE relations SET confidence = confidence * (1.0 - ?) WHERE confidence > 0.1 AND temporal_end IS NULL",
        [decayRate]
    );
    runSql(
        "UPDATE vectors SET confidence = confidence * (1.0 - ?) WHERE confidence > 0.1",
        [decayRate]
    );
}

export function wipeAllData(): void {
    runSql("DELETE FROM access_log");
    runSql("DELETE FROM vectors");
    runSql("DELETE FROM relations");
    runSql("DELETE FROM summaries");
    runSql("DELETE FROM entities");
}
