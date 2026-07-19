import { readFileSync, existsSync } from "fs";
import { basename, dirname, isAbsolute, join, resolve } from "path";
import { fileURLToPath } from "url";

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

export const CONFIG_FILE_NAME = "latentcontext.config.json";
export const DATA_DIR_ENV = "LATENTCONTEXT_DATA_DIR";
export const CONFIG_PATH_ENV = "LATENTCONTEXT_CONFIG";
export const ALLOW_PROJECT_CONFIG_ENV = "LATENTCONTEXT_ALLOW_PROJECT_CONFIG";

const PROJECT_DATA_DIR_NAME = ".latentcontext";

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: LatentContextConfig = {
    storage: {
        dataDir: "",
        sqliteFile: "memory.db",
    },
    embedding: {
        provider: "local",
        model: "Xenova/all-MiniLM-L6-v2",
        dimensions: 384,
    },
    tokenBudgets: {
        tier0Working: 16000,        // Working memory buffer — plenty of room for detailed entries
        tier1Session: 4000,         // Compressed session summaries retain more detail
        tier2Epoch: 2000,           // Epoch-level long-term knowledge
        tier3Core: 1000,            // Permanent core facts about the user
        retrieval: 8000,            // How much context can be retrieved per query
        graphFacts: 2000,           // Knowledge graph facts budget
        defaultRetrieveBudget: 8000, // Default budget when LLM doesn't specify one
    },
    compression: {
        tier0OverflowThreshold: 20000, // Compress when working memory exceeds 20K tokens
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

function getPackageRoot(): string {
    const moduleDir = dirname(fileURLToPath(import.meta.url));
    const leaf = basename(moduleDir);
    if (leaf === "src" || leaf === "dist") {
        return dirname(moduleDir);
    }
    return moduleDir;
}

function resolvePath(pathValue: string, baseDir: string = process.cwd()): string {
    return isAbsolute(pathValue) ? pathValue : resolve(baseDir, pathValue);
}

function envFlagEnabled(value: string | undefined): boolean {
    return value === "1" || value?.toLowerCase() === "true";
}

export function getDefaultDataDir(env: NodeJS.ProcessEnv = process.env): string {
    const explicit = env[DATA_DIR_ENV]?.trim();
    if (explicit) {
        return resolvePath(explicit);
    }

    return join(process.cwd(), PROJECT_DATA_DIR_NAME);
}

function getConfigSearchPaths(configPath: string | undefined, defaultDataDir: string): string[] {
    const paths: string[] = [];

    if (configPath) {
        paths.push(resolvePath(configPath));
    }

    const envConfigPath = process.env[CONFIG_PATH_ENV]?.trim();
    if (envConfigPath) {
        paths.push(resolvePath(envConfigPath));
    }

    paths.push(join(defaultDataDir, CONFIG_FILE_NAME));
    paths.push(join(getPackageRoot(), CONFIG_FILE_NAME));

    // Project-level config is opt-in because MCP hosts often launch servers
    // with the user's current project as CWD.
    if (envFlagEnabled(process.env[ALLOW_PROJECT_CONFIG_ENV])) {
        paths.push(join(process.cwd(), CONFIG_FILE_NAME));
    }

    return [...new Set(paths)];
}

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
let _configSource: string | null = null;

export function loadConfig(configPath?: string): LatentContextConfig {
    if (_config) return _config;

    let userConfig: Record<string, unknown> = {};
    let configBaseDir: string | null = null;
    const defaultDataDir = getDefaultDataDir();

    for (const p of getConfigSearchPaths(configPath, defaultDataDir)) {
        if (existsSync(p)) {
            try {
                const raw = readFileSync(p, "utf-8");
                userConfig = JSON.parse(raw) as Record<string, unknown>;
                _configSource = p;
                configBaseDir = dirname(p);
                break;
            } catch {
                // ignore malformed config, use defaults
            }
        }
    }

    const merged = deepMerge(
        structuredClone(DEFAULT_CONFIG) as unknown as Record<string, unknown>,
        userConfig
    ) as unknown as LatentContextConfig;

    _config = merged;

    const envDataDir = process.env[DATA_DIR_ENV]?.trim();
    if (envDataDir) {
        _config.storage.dataDir = resolvePath(envDataDir);
    } else if (!_config.storage.dataDir) {
        _config.storage.dataDir = defaultDataDir;
    } else {
        _config.storage.dataDir = resolvePath(
            _config.storage.dataDir,
            configBaseDir ?? getPackageRoot()
        );
    }

    return _config;
}

export function getConfig(): LatentContextConfig {
    if (!_config) return loadConfig();
    return _config;
}

export function resetConfig(): void {
    _config = null;
    _configSource = null;
}

export function getConfigSource(): string | null {
    return _configSource;
}
