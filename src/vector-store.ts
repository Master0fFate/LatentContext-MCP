import { v4 as uuidv4 } from "uuid";
import {
    insertVector,
    getAllVectors,
    deleteVector,
    deleteVectorsBySourceId,
    getVectorCount,
    getDb,
    type VectorRow,
} from "./database.js";
import {
    embed,
    cosineSimilarity,
    vectorToBuffer,
    bufferToVector,
} from "./embeddings.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VectorSearchResult {
    id: string;
    sourceId: string;
    sourceType: string;
    contentPreview: string;
    similarity: number;
    confidence: number;
    metadata: Record<string, unknown>;
    createdAt: string;
}

export interface VectorSearchFilter {
    sourceTypes?: string[];
    after?: string; // ISO datetime
    before?: string; // ISO datetime
    minConfidence?: number;
}

/** An embedded vector ready to persist in a caller-controlled transaction. */
export interface PreparedVector {
    id: string;
    sourceId: string;
    sourceType: string;
    contentPreview: string;
    embedding: number[];
    confidence: number;
    metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// In-memory vector cache for fast search
// ---------------------------------------------------------------------------

interface CachedVector {
    id: string;
    sourceId: string;
    sourceType: string;
    contentPreview: string;
    embedding: number[];
    confidence: number;
    metadata: Record<string, unknown>;
    createdAt: string;
}

let _vectorCache: CachedVector[] | null = null;
let _cacheStale = true;
let _cacheDatabase: ReturnType<typeof getDb> | null = null;

function loadVectorCache(): CachedVector[] {
    const database = getDb();
    // A database can be closed and reopened in-process during lifecycle
    // recovery/tests. Never serve vectors decoded from the former instance.
    if (_vectorCache && !_cacheStale && _cacheDatabase === database) return _vectorCache;

    const rows = getAllVectors();
    _vectorCache = rows.map((row) => ({
        id: row.id,
        sourceId: row.source_id,
        sourceType: row.source_type,
        contentPreview: row.content_preview,
        embedding: bufferToVector(row.embedding),
        confidence: row.confidence,
        metadata: parseMetadata(row.metadata),
        createdAt: row.created_at,
    }));
    _cacheDatabase = database;
    _cacheStale = false;

    return _vectorCache;
}

function invalidateCache(): void {
    _cacheStale = true;
}

function parseMetadata(metadata: string): Record<string, unknown> {
    try {
        const parsed: unknown = JSON.parse(metadata || "{}");
        return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    } catch {
        return {};
    }
}

// ---------------------------------------------------------------------------
// Vector store operations
// ---------------------------------------------------------------------------

/** Embed content without mutating storage, for atomic multi-row mutations. */
export async function prepareVector(
    content: string,
    sourceId: string,
    sourceType: string,
    confidence: number = 1.0,
    metadata: Record<string, unknown> = {}
): Promise<PreparedVector> {
    const embedding = await embed(content);
    return {
        id: uuidv4(),
        sourceId,
        sourceType,
        contentPreview: content.length > 200 ? content.substring(0, 200) + "..." : content,
        embedding,
        confidence,
        metadata,
    };
}

/** Persist a previously prepared vector in the caller's database transaction. */
export function persistPreparedVector(vector: PreparedVector): string {
    insertVector({
        id: vector.id,
        source_id: vector.sourceId,
        source_type: vector.sourceType,
        content_preview: vector.contentPreview,
        embedding: vectorToBuffer(vector.embedding),
        dimensions: vector.embedding.length,
        metadata: JSON.stringify(vector.metadata),
        confidence: vector.confidence,
    });
    invalidateCache();
    return vector.id;
}

/**
 * Add content to the vector store. Embeds the text and stores the vector.
 */
export async function addToVectorStore(
    content: string,
    sourceId: string,
    sourceType: string,
    confidence: number = 1.0,
    metadata: Record<string, unknown> = {}
): Promise<string> {
    return persistPreparedVector(
        await prepareVector(content, sourceId, sourceType, confidence, metadata)
    );
}

/**
 * Add content with a precomputed embedding vector.
 */
export function addVectorDirect(
    embedding: number[],
    content: string,
    sourceId: string,
    sourceType: string,
    confidence: number = 1.0,
    metadata: Record<string, unknown> = {}
): string {
    const id = uuidv4();
    const preview =
        content.length > 200 ? content.substring(0, 200) + "..." : content;

    insertVector({
        id,
        source_id: sourceId,
        source_type: sourceType,
        content_preview: preview,
        embedding: vectorToBuffer(embedding),
        dimensions: embedding.length,
        metadata: JSON.stringify(metadata),
        confidence,
    });

    invalidateCache();
    return id;
}

/**
 * Search the vector store for the most similar content to the query.
 * Uses brute-force cosine similarity over all vectors.
 */
export async function searchVectors(
    query: string,
    topK: number = 20,
    filters?: VectorSearchFilter
): Promise<VectorSearchResult[]> {
    const queryEmbedding = await embed(query);
    return searchVectorsByEmbedding(queryEmbedding, topK, filters);
}

/**
 * Search using a precomputed query embedding.
 */
export function searchVectorsByEmbedding(
    queryEmbedding: number[],
    topK: number = 20,
    filters?: VectorSearchFilter
): VectorSearchResult[] {
    const cache = loadVectorCache();

    // Compute similarities and apply filters
    const scored: VectorSearchResult[] = [];

    for (const vec of cache) {
        // Apply filters
        if (filters) {
            if (
                filters.sourceTypes &&
                filters.sourceTypes.length > 0 &&
                !filters.sourceTypes.includes(vec.sourceType)
            ) {
                continue;
            }

            if (filters.after && vec.createdAt < filters.after) {
                continue;
            }

            if (filters.before && vec.createdAt > filters.before) {
                continue;
            }

            if (
                filters.minConfidence !== undefined &&
                vec.confidence < filters.minConfidence
            ) {
                continue;
            }
        }

        const similarity = cosineSimilarity(queryEmbedding, vec.embedding);

        scored.push({
            id: vec.id,
            sourceId: vec.sourceId,
            sourceType: vec.sourceType,
            contentPreview: vec.contentPreview,
            similarity,
            confidence: vec.confidence,
            metadata: vec.metadata,
            createdAt: vec.createdAt,
        });
    }

    // Sort by similarity descending and return top K
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
}

/**
 * Remove a vector by its ID.
 */
export function removeVector(id: string): void {
    deleteVector(id);
    invalidateCache();
}

/**
 * Remove all vectors associated with a source ID.
 */
export function removeVectorsBySource(sourceId: string): void {
    deleteVectorsBySourceId(sourceId);
    invalidateCache();
}

/**
 * Get the total number of vectors in the store.
 */
export function getVectorStoreCount(): number {
    return getVectorCount();
}

/**
 * Force a cache reload on next search.
 */
export function refreshVectorCache(): void {
    invalidateCache();
}
