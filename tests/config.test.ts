import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, test } from "node:test";
import {
    ALLOW_PROJECT_CONFIG_ENV,
    CONFIG_PATH_ENV,
    DATA_DIR_ENV,
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

test("default data dir uses platform app storage on Windows", () => {
    const dataDir = getDefaultDataDir(
        { LOCALAPPDATA: "C:\\Users\\Ada\\AppData\\Local" },
        "C:\\Users\\Ada",
        "win32"
    );

    assert.equal(dataDir, "C:\\Users\\Ada\\AppData\\Local\\LatentContext-MCP");
});

test("default config does not resolve storage under the launched project cwd", () => {
    clearConfigEnv();
    const projectDir = tempDir();
    process.chdir(projectDir);

    const config = loadConfig();

    assert.notEqual(config.storage.dataDir, join(projectDir, "data"));
});

test("LATENTCONTEXT_DATA_DIR overrides the default storage directory", () => {
    clearConfigEnv();
    const dataDir = tempDir();
    process.env[DATA_DIR_ENV] = dataDir;

    const config = loadConfig();

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
