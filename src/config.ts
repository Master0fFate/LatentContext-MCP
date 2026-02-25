import { readFileSync, existsSync } from "fs";
import { join } from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StorageConfig {
    dataDir: string;
    sqliteFile: string;
}

export interface EmbeddingConfig {
    provider: "local" | "none";
    model: string;
    dimensions: number;
}

export interface TokenBudgetConfig {
    tier0Working: number;
    tier1Session: number;
    tier2Epoch: number;
    tier3Core: number;
    retrieval: number;
    graphFacts: number;
    defaultRetrieveBudget: number;
}

export interface CompressionConfig {
    tier0OverflowThreshold: number;
    tier1ConsolidationCount: number;
    confidenceDecayRate: number;
    confidenceDecayIntervalHours: number;
}

export interface RankingConfig {
    semanticWeight: number;
    recencyWeight: number;
    priorityWeight: number;
    frequencyWeight: number;
    dedupSimilarityThreshold: number;
}

export interface SessionConfig {
    autoStartOnBoot: boolean;
}

export interface LatentContextConfig {
    storage: StorageConfig;
    embedding: EmbeddingConfig;
    tokenBudgets: TokenBudgetConfig;
    compression: CompressionConfig;
    ranking: RankingConfig;
    session: SessionConfig;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: LatentContextConfig = {
    storage: {
        dataDir: "./data",
        sqliteFile: "memory.db",
    },
    embedding: {
        provider: "local",
        model: "Xenova/all-MiniLM-L6-v2",
        dimensions: 384,
    },
    tokenBudgets: {
        tier0Working: 2000,
        tier1Session: 500,
        tier2Epoch: 300,
        tier3Core: 200,
        retrieval: 2000,
        graphFacts: 500,
        defaultRetrieveBudget: 3000,
    },
    compression: {
        tier0OverflowThreshold: 2500,
        tier1ConsolidationCount: 10,
        confidenceDecayRate: 0.01,
        confidenceDecayIntervalHours: 24,
    },
    ranking: {
        semanticWeight: 0.4,
        recencyWeight: 0.3,
        priorityWeight: 0.2,
        frequencyWeight: 0.1,
        dedupSimilarityThreshold: 0.85,
    },
    session: {
        autoStartOnBoot: true,
    },
};

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

function deepMerge(
    base: Record<string, unknown>,
    override: Record<string, unknown>
): Record<string, unknown> {
    const result: Record<string, unknown> = { ...base };
    for (const key of Object.keys(override)) {
        const baseVal = result[key];
        const overrideVal = override[key];
        if (
            baseVal !== null &&
            overrideVal !== null &&
            typeof baseVal === "object" &&
            typeof overrideVal === "object" &&
            !Array.isArray(baseVal) &&
            !Array.isArray(overrideVal)
        ) {
            result[key] = deepMerge(
                baseVal as Record<string, unknown>,
                overrideVal as Record<string, unknown>
            );
        } else if (overrideVal !== undefined) {
            result[key] = overrideVal;
        }
    }
    return result;
}

let _config: LatentContextConfig | null = null;

export function loadConfig(configPath?: string): LatentContextConfig {
    if (_config) return _config;

    let userConfig: Record<string, unknown> = {};

    // Determine config file location
    const searchPaths: string[] = [];
    if (configPath) {
        searchPaths.push(configPath);
    }
    searchPaths.push(join(process.cwd(), "latentcontext.config.json"));

    for (const p of searchPaths) {
        if (existsSync(p)) {
            try {
                const raw = readFileSync(p, "utf-8");
                userConfig = JSON.parse(raw) as Record<string, unknown>;
                break;
            } catch {
                // ignore malformed config, use defaults
            }
        }
    }

    const merged = deepMerge(
        DEFAULT_CONFIG as unknown as Record<string, unknown>,
        userConfig
    ) as unknown as LatentContextConfig;

    _config = merged;

    // Resolve dataDir to absolute path
    if (!_config.storage.dataDir.startsWith("/") && !_config.storage.dataDir.match(/^[A-Za-z]:\\/)) {
        _config.storage.dataDir = join(process.cwd(), _config.storage.dataDir);
    }

    return _config;
}

export function getConfig(): LatentContextConfig {
    if (!_config) return loadConfig();
    return _config;
}

export function resetConfig(): void {
    _config = null;
}
