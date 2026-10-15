import { existsSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, isAbsolute, join, parse, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { RootsListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { CodeIndex } from "./code-index.js";
import { shutdownPool } from "./code-pool.js";
import { getConfig } from "./config.js";
import { log } from "./logger.js";

export const CODE_INDEX_ENV = "LATENTCONTEXT_CODE_INDEX";
export const CODE_ROOT_ENV = "LATENTCONTEXT_CODE_ROOT";

const PROJECT_MARKERS = [
    ".git", ".hg", "package.json", "Cargo.toml", "go.mod", "pyproject.toml", "setup.py", "requirements.txt", "pom.xml",
    "build.gradle", "build.gradle.kts", "settings.gradle.kts", "composer.json", "Gemfile", "mix.exs", "build.zig",
    "CMakeLists.txt", "Makefile", "deno.json", "tsconfig.json", "global.json", "Directory.Build.props",
];
const ROOTS_TIMEOUT_MS = 3_000;

const indexes = new Map<string, CodeIndex>();
let clientRoots: Promise<string[]> | null = null;

export function codeIndexEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
    const value = env[CODE_INDEX_ENV]?.trim().toLowerCase();
    return value !== "0" && value !== "false" && value !== "off" && value !== "no";
}

function usableRoot(path: string): string | null {
    const root = resolve(path);
    if (root === parse(root).root || root === homedir()) return null;
    return statSync(root, { throwIfNoEntry: false })?.isDirectory() ? root : null;
}

function listClientRoots(server: Server): Promise<string[]> {
    if (!server.getClientCapabilities()?.roots) return Promise.resolve([]);
    clientRoots ??= server.listRoots(undefined, { timeout: ROOTS_TIMEOUT_MS })
        .then(({ roots }) => roots.filter((root) => root.uri.startsWith("file:")).map((root) => fileURLToPath(root.uri)))
        .catch((error: unknown) => {
            log("warning", "Client roots unavailable; using the working directory", error);
            return [];
        });
    return clientRoots;
}

export async function resolveCodeRoots(server: Server, explicit?: string): Promise<string[]> {
    if (explicit !== undefined) {
        const root = isAbsolute(explicit) ? usableRoot(explicit) : null;
        if (!root) throw new Error(`'root' must be an absolute path to a project directory: ${explicit}`);
        return [root];
    }
    const configured = process.env[CODE_ROOT_ENV]?.split(delimiter).map((path) => path.trim()).filter(Boolean) ?? [];
    const candidates = configured.length > 0 ? configured : await listClientRoots(server);
    const roots = [...new Set(candidates.map(usableRoot).filter((root): root is string => root !== null))];
    if (roots.length > 0) return roots;
    const cwd = usableRoot(process.cwd());
    return cwd && PROJECT_MARKERS.some((marker) => existsSync(join(cwd, marker))) ? [cwd] : [];
}

export async function getCodeIndexes(roots: readonly string[]): Promise<CodeIndex[]> {
    const cacheDir = join(getConfig().storage.dataDir, "code-index");
    const selected = roots.map((root) => {
        let index = indexes.get(root);
        if (!index) {
            index = new CodeIndex(root, cacheDir);
            indexes.set(root, index);
        }
        return index;
    });
    await Promise.all(selected.map((index) => index.ready()));
    return selected;
}

export async function warmCodeIndexes(server: Server): Promise<void> {
    const roots = await resolveCodeRoots(server);
    for (const [root, index] of indexes) {
        if (roots.includes(root)) continue;
        index.close();
        indexes.delete(root);
    }
    await getCodeIndexes(roots);
}

export function watchClientRoots(server: Server): void {
    server.setNotificationHandler(RootsListChangedNotificationSchema, async () => {
        clientRoots = null;
        await warmCodeIndexes(server).catch((error: unknown) => log("warning", "Code index refresh after roots change failed", error));
    });
}

export function closeCodeIndexes(): void {
    for (const index of indexes.values()) index.close();
    indexes.clear();
    shutdownPool();
}
