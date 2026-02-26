import { getConfig } from "./config.js";

// ---------------------------------------------------------------------------
// Local embedding via @huggingface/transformers (ONNX runtime)
// ---------------------------------------------------------------------------

type Pipeline = (
    input: string | string[],
    options?: { pooling: string; normalize: boolean }
) => Promise<{ tolist: () => number[][] }>;

let _pipeline: Pipeline | null = null;
let _initPromise: Promise<void> | null = null;
let _initFailed: boolean = false;
let _initError: string | null = null;

/**
 * Initialize the embedding pipeline lazily. The model is downloaded on first
 * use and cached locally (~80MB for all-MiniLM-L6-v2).
 *
 * IMPORTANT: We suppress ALL console output during initialization because
 * @huggingface/transformers outputs download progress bars and ONNX runtime
 * status messages that would corrupt the MCP stdio JSON-RPC protocol.
 */
async function initPipeline(): Promise<void> {
    if (_pipeline) return;
    if (_initFailed) return; // Don't retry if init already failed

    if (_initPromise) {
        await _initPromise;
        return;
    }

    _initPromise = (async () => {
        const config = getConfig();
        const modelName = config.embedding.model;

        try {
            // Dynamic import to avoid loading the heavy ONNX runtime at module level
            const { pipeline, env } = await import("@huggingface/transformers");

            // Suppress all HuggingFace progress output — these progress bars
            // and status messages would corrupt the MCP JSON-RPC protocol
            // since they write to stdout/stderr.
            if (env) {
                // Disable remote model fetching progress bars
                (env as Record<string, unknown>).allowRemoteModels = true;
                // Some versions of @huggingface/transformers support log level
                if ("logLevel" in env) {
                    (env as Record<string, unknown>).logLevel = "error";
                }
            }

            _pipeline = (await pipeline("feature-extraction", modelName, {
                dtype: "fp32",
            })) as unknown as Pipeline;
        } catch (error) {
            _initFailed = true;
            _initError = error instanceof Error ? error.message : String(error);
            _pipeline = null;
            // Don't throw — we'll gracefully degrade to zero vectors
        }
    })();

    await _initPromise;
}

/**
 * Generate an embedding vector for a text string.
 * Returns a float array of `config.embedding.dimensions` length (default 384).
 *
 * If the embedding pipeline fails to initialize (e.g., model download fails,
 * ONNX runtime error), returns a zero vector instead of crashing.
 */
export async function embed(text: string): Promise<number[]> {
    const config = getConfig();

    if (config.embedding.provider === "none") {
        // Return zero vector if embeddings are disabled
        return new Array(config.embedding.dimensions).fill(0);
    }

    try {
        await initPipeline();
    } catch {
        // Pipeline init failed — return zero vector
        return new Array(config.embedding.dimensions).fill(0);
    }

    if (!_pipeline) {
        // Pipeline failed to initialize — return zero vector (graceful degradation)
        return new Array(config.embedding.dimensions).fill(0);
    }

    try {
        const result = await _pipeline(text, {
            pooling: "mean",
            normalize: true,
        });

        const vectors = result.tolist();
        return vectors[0];
    } catch (error) {
        // Embedding call failed — return zero vector instead of crashing
        return new Array(config.embedding.dimensions).fill(0);
    }
}

/**
 * Generate embeddings for multiple texts in a batch.
 */
export async function embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const config = getConfig();

    if (config.embedding.provider === "none") {
        return texts.map(() => new Array(config.embedding.dimensions).fill(0));
    }

    try {
        await initPipeline();
    } catch {
        return texts.map(() => new Array(config.embedding.dimensions).fill(0));
    }

    if (!_pipeline) {
        return texts.map(() => new Array(config.embedding.dimensions).fill(0));
    }

    try {
        const result = await _pipeline(texts, {
            pooling: "mean",
            normalize: true,
        });

        return result.tolist();
    } catch {
        return texts.map(() => new Array(config.embedding.dimensions).fill(0));
    }
}

/**
 * Compute cosine similarity between two vectors.
 * Assumes vectors are already normalized (which they are from the pipeline).
 */
export function cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
}

/**
 * Serialize a float array to a Uint8Array for SQLite BLOB storage.
 */
export function vectorToBuffer(vector: number[]): Uint8Array {
    const buf = new ArrayBuffer(vector.length * 4); // Float32 = 4 bytes each
    const view = new DataView(buf);
    for (let i = 0; i < vector.length; i++) {
        view.setFloat32(i * 4, vector[i], true); // little-endian
    }
    return new Uint8Array(buf);
}

/**
 * Deserialize a Uint8Array back to a float array.
 */
export function bufferToVector(buf: Uint8Array): number[] {
    const vector: number[] = [];
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let i = 0; i < buf.byteLength; i += 4) {
        vector.push(view.getFloat32(i, true)); // little-endian
    }
    return vector;
}

/**
 * Check if the embedding pipeline is initialized.
 */
export function isEmbeddingReady(): boolean {
    return _pipeline !== null;
}

/**
 * Get the initialization error message, if any.
 */
export function getEmbeddingError(): string | null {
    return _initError;
}
