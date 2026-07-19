import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, test } from "node:test";
import {
    ALLOW_PROJECT_CONFIG_ENV,
    CONFIG_PATH_ENV,
    DATA_DIR_ENV,
    getConfigSource,
    getDefaultDataDir,
    loadConfig,
    resetConfig,
} from "../src/config.js";

const originalCwd = process.cwd();
const originalEnv = {
    [ALLOW_PROJECT_CONFIG_ENV]: process.env[ALLOW_PROJECT_CONFIG_ENV],
    [CONFIG_PATH_ENV]: process.env[CONFIG_PATH_ENV],
    [DATA_DIR_ENV]: process.env[DATA_DIR_ENV],
};
const tempDirs: string[] = [];

function tempDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "latentcontext-test-"));
    tempDirs.push(dir);
    return dir;
}

function clearConfigEnv(): void {
    delete process.env[ALLOW_PROJECT_CONFIG_ENV];
    delete process.env[CONFIG_PATH_ENV];
    delete process.env[DATA_DIR_ENV];
}

afterEach(() => {
    resetConfig();
    process.chdir(originalCwd);
    clearConfigEnv();
    for (const [key, value] of Object.entries(originalEnv)) {
        if (value === undefined) {
            delete process.env[key];
        } else {
            process.env[key] = value;
        }
    }
    while (tempDirs.length > 0) {
        const dir = tempDirs.pop();
        if (dir) rmSync(dir, { recursive: true, force: true });
    }
});

test("default config stores data in the launched project's .latentcontext directory", () => {
    clearConfigEnv();
    const projectDir = tempDir();
    process.chdir(projectDir);

    assert.equal(getDefaultDataDir(), join(projectDir, ".latentcontext"));
    assert.equal(loadConfig().storage.dataDir, join(projectDir, ".latentcontext"));
});

test("different project roots use different default storage directories", () => {
    clearConfigEnv();
    const firstProjectDir = tempDir();
    const secondProjectDir = tempDir();

    process.chdir(firstProjectDir);
    const firstDataDir = loadConfig().storage.dataDir;

    resetConfig();
    process.chdir(secondProjectDir);
    const secondDataDir = loadConfig().storage.dataDir;

    assert.equal(firstDataDir, join(firstProjectDir, ".latentcontext"));
    assert.equal(secondDataDir, join(secondProjectDir, ".latentcontext"));
    assert.notEqual(firstDataDir, secondDataDir);
});

test("LATENTCONTEXT_DATA_DIR overrides configured and project-local storage", () => {
    clearConfigEnv();
    const projectDir = tempDir();
    const configDir = tempDir();
    const dataDir = tempDir();
    const configPath = join(configDir, "latentcontext.config.json");
    writeFileSync(configPath, JSON.stringify({ storage: { dataDir: "./configured-data" } }));
    process.chdir(projectDir);
    process.env[DATA_DIR_ENV] = dataDir;

    const config = loadConfig(configPath);

    assert.equal(config.storage.dataDir, resolve(dataDir));
});

test("environment-selected config loads while LATENTCONTEXT_DATA_DIR takes precedence", () => {
    clearConfigEnv();
    const projectDir = tempDir();
    const configDir = tempDir();
    const dataDir = tempDir();
    const configPath = join(configDir, "latentcontext.config.json");
    writeFileSync(
        configPath,
        JSON.stringify({ storage: { dataDir: "./configured-data", sqliteFile: "from-env-config.db" } })
    );
    process.chdir(projectDir);
    process.env[CONFIG_PATH_ENV] = configPath;
    process.env[DATA_DIR_ENV] = dataDir;

    const config = loadConfig();

    assert.equal(getConfigSource(), resolve(configPath));
    assert.equal(config.storage.sqliteFile, "from-env-config.db");
    assert.equal(config.storage.dataDir, resolve(dataDir));
});

test("relative dataDir in an explicit config resolves relative to that config file", () => {
    clearConfigEnv();
    const configDir = tempDir();
    const launchedProjectDir = tempDir();
    const configPath = join(configDir, "latentcontext.config.json");
    writeFileSync(
        configPath,
        JSON.stringify({ storage: { dataDir: "./state", sqliteFile: "custom.db" } })
    );
    process.chdir(launchedProjectDir);

    const config = loadConfig(configPath);

    assert.equal(config.storage.dataDir, join(configDir, "state"));
    assert.equal(config.storage.sqliteFile, "custom.db");
});

test("project cwd config is ignored unless explicitly enabled", () => {
    clearConfigEnv();
    const projectDir = tempDir();
    writeFileSync(
        join(projectDir, "latentcontext.config.json"),
        JSON.stringify({ storage: { dataDir: "./project-data" } })
    );
    process.chdir(projectDir);

    const defaultConfig = loadConfig();
    assert.notEqual(defaultConfig.storage.dataDir, join(projectDir, "project-data"));

    resetConfig();
    process.env[ALLOW_PROJECT_CONFIG_ENV] = "1";
    const projectConfig = loadConfig();
    assert.equal(projectConfig.storage.dataDir, join(projectDir, "project-data"));
});
