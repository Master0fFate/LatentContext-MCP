import { encodingForModel } from "js-tiktoken";

// ---------------------------------------------------------------------------
// Token counter using cl100k_base encoding (GPT-4 / Claude compatible)
// ---------------------------------------------------------------------------

let _encoder: ReturnType<typeof encodingForModel> | null = null;

function getEncoder() {
    if (!_encoder) {
        _encoder = encodingForModel("gpt-4");
    }
    return _encoder;
}

/**
 * Count the number of tokens in a text string.
 */
export function countTokens(text: string): number {
    if (!text || text.length === 0) return 0;
    const encoder = getEncoder();
    return encoder.encode(text).length;
}

/**
 * Truncate text to fit within a token budget.
 * Returns the truncated text and the actual token count.
 */
export function truncateToTokenBudget(
    text: string,
    budget: number
): { text: string; tokens: number } {
    if (!text || text.length === 0) return { text: "", tokens: 0 };

    const encoder = getEncoder();
    const tokens = encoder.encode(text);

    if (tokens.length <= budget) {
        return { text, tokens: tokens.length };
    }

    // Truncate tokens and decode back to text
    const truncatedTokens = tokens.slice(0, budget);
    // js-tiktoken's decode returns a Uint8Array, convert to string
    const decoded = encoder.decode(truncatedTokens);
    const truncatedText = typeof decoded === "string"
        ? decoded
        : new TextDecoder().decode(decoded as unknown as Uint8Array);

    return { text: truncatedText, tokens: budget };
}

/**
 * Estimate token count from character length (fast approximation).
 * Use for quick budget checks before doing precise counting.
 * Roughly 1 token ≈ 4 characters for English text.
 */
export function estimateTokens(text: string): number {
    if (!text) return 0;
    return Math.ceil(text.length / 4);
}

/**
 * Check if text fits within a token budget without counting precisely.
 * Uses character-based estimation for speed, with a safety margin.
 */
export function fitsInBudget(text: string, budget: number): boolean {
    // Quick check: if estimated tokens are well under budget, skip precise count
    const estimate = estimateTokens(text);
    if (estimate < budget * 0.8) return true;
    if (estimate > budget * 1.3) return false;
    // Edge case: do precise count
    return countTokens(text) <= budget;
}

/**
 * Split text into chunks that each fit within the given token budget.
 */
export function splitIntoChunks(text: string, chunkBudget: number): string[] {
    if (!text || text.length === 0) return [];

    const totalTokens = countTokens(text);
    if (totalTokens <= chunkBudget) return [text];

    // Split by sentences first
    const sentences = text.match(/[^.!?\n]+[.!?\n]*/g) || [text];
    const chunks: string[] = [];
    let currentChunk = "";
    let currentTokens = 0;

    for (const sentence of sentences) {
        const sentenceTokens = countTokens(sentence);

        if (currentTokens + sentenceTokens > chunkBudget && currentChunk.length > 0) {
            chunks.push(currentChunk.trim());
            currentChunk = sentence;
            currentTokens = sentenceTokens;
        } else {
            currentChunk += sentence;
            currentTokens += sentenceTokens;
        }
    }

    if (currentChunk.trim().length > 0) {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}
