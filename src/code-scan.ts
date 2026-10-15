import { readdirSync, readFileSync, statSync, type Dirent } from "node:fs";
import { join } from "node:path";
import { languageForPath } from "./code-languages.js";

export interface ScanEntry {
    path: string;
    mtimeMs: number;
    size: number;
}

export interface ScanResult {
    files: ScanEntry[];
    dirs: string[];
}

interface Rule {
    regex: RegExp;
    negate: boolean;
    dirOnly: boolean;
}

interface RuleSet {
    base: string;
    rules: Rule[];
}

export const MAX_FILE_BYTES = 1_500_000;

export const IGNORED_DIRS: ReadonlySet<string> = new Set([
    ".git", ".hg", ".svn", ".jj", "node_modules", "bower_components", ".latentcontext", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache",
    ".gradle", ".idea", ".vs", ".vscode", ".cache", ".dart_tool", ".zig-cache", "zig-cache", ".build", "Pods",
]);

const REGEX_SPECIAL = /[.+^${}()|\\]/;

function patternToRegex(pattern: string): RegExp {
    const anchored = pattern.startsWith("/") || pattern.includes("/", 1);
    const body = pattern.startsWith("/") ? pattern.slice(1) : pattern;
    let out = "";
    for (let i = 0; i < body.length; i++) {
        const char = body[i];
        if (char === "*") {
            if (body[i + 1] === "*" && (i === 0 || body[i - 1] === "/")) {
                if (i + 2 === body.length) {
                    out += ".*";
                    i += 1;
                    continue;
                }
                if (body[i + 2] === "/") {
                    out += "(?:.*/)?";
                    i += 2;
                    continue;
                }
            }
            while (body[i + 1] === "*") i++;
            out += "[^/]*";
        } else if (char === "?") {
            out += "[^/]";
        } else if (char === "[") {
            const close = body.indexOf("]", i + 2);
            if (close === -1) {
                out += "\\[";
            } else {
                const set = body.slice(i + 1, close).replace(/\\/g, "\\\\");
                out += `[${set.startsWith("!") ? `^${set.slice(1)}` : set}]`;
                i = close;
            }
        } else if (char === "\\" && i + 1 < body.length) {
            const next = body[++i];
            out += REGEX_SPECIAL.test(next) || next === "*" || next === "?" || next === "[" ? `\\${next}` : next;
        } else {
            out += REGEX_SPECIAL.test(char) ? `\\${char}` : char;
        }
    }
    return new RegExp(`${anchored ? "^" : "(?:^|/)"}${out}$`);
}

export function compileGitignore(text: string): Rule[] {
    const rules: Rule[] = [];
    for (const rawLine of text.split(/\r?\n/)) {
        let line = rawLine.replace(/(?<!\\)\s+$/, "");
        if (line.length === 0 || line.startsWith("#")) continue;
        const negate = line.startsWith("!");
        if (negate) line = line.slice(1);
        else if (line.startsWith("\\!") || line.startsWith("\\#")) line = line.slice(1);
        const dirOnly = line.endsWith("/");
        if (dirOnly) line = line.slice(0, -1);
        if (line.length === 0) continue;
        rules.push({ regex: patternToRegex(line), negate, dirOnly });
    }
    return rules;
}

export function isIgnored(chain: readonly RuleSet[], path: string, isDir: boolean): boolean {
    for (let i = chain.length - 1; i >= 0; i--) {
        const { base, rules } = chain[i];
        const local = base ? path.slice(base.length) : path;
        for (let j = rules.length - 1; j >= 0; j--) {
            const rule = rules[j];
            if (rule.dirOnly && !isDir) continue;
            if (rule.regex.test(local)) return !rule.negate;
        }
    }
    return false;
}

function readRules(root: string, dir: string): RuleSet | null {
    try {
        const rules = compileGitignore(readFileSync(join(root, dir, ".gitignore"), "utf8"));
        return rules.length > 0 ? { base: dir ? `${dir}/` : "", rules } : null;
    } catch {
        return null;
    }
}

export function ruleChainFor(root: string, dir: string): RuleSet[] {
    const chain: RuleSet[] = [];
    const rootRules = readRules(root, "");
    if (rootRules) chain.push(rootRules);
    if (!dir) return chain;
    let prefix = "";
    for (const segment of dir.split("/")) {
        prefix = prefix ? `${prefix}/${segment}` : segment;
        const rules = readRules(root, prefix);
        if (rules) chain.push(rules);
    }
    return chain;
}

export function isPathIgnored(root: string, path: string): boolean {
    const segments = path.split("/");
    if (segments.some((segment) => IGNORED_DIRS.has(segment))) return true;
    const chain: RuleSet[] = [];
    let prefix = "";
    for (let i = 0; i < segments.length; i++) {
        const rules = readRules(root, prefix);
        if (rules) chain.push(rules);
        prefix = prefix ? `${prefix}/${segments[i]}` : segments[i];
        if (isIgnored(chain, prefix, i < segments.length - 1)) return true;
    }
    return false;
}

export function scanTree(root: string, start = "", maxFiles = Number.POSITIVE_INFINITY): ScanResult {
    const files: ScanEntry[] = [];
    const dirs: string[] = [];
    const parent = start.includes("/") ? start.slice(0, start.lastIndexOf("/")) : "";
    const stack: Array<{ dir: string; chain: RuleSet[] }> = [{ dir: start, chain: start ? ruleChainFor(root, parent) : [] }];

    while (stack.length > 0 && files.length < maxFiles) {
        const { dir, chain: parentChain } = stack.pop()!;
        const absolute = dir ? join(root, dir) : root;
        let entries: Dirent[];
        try {
            entries = readdirSync(absolute, { withFileTypes: true });
        } catch {
            continue;
        }
        dirs.push(dir);

        let chain = parentChain;
        if (entries.some((entry) => entry.name === ".gitignore" && entry.isFile())) {
            const rules = readRules(root, dir);
            if (rules) chain = [...parentChain, rules];
        }

        for (const entry of entries) {
            const name = entry.name;
            const path = dir ? `${dir}/${name}` : name;
            if (entry.isDirectory()) {
                if (!IGNORED_DIRS.has(name) && !isIgnored(chain, path, true)) stack.push({ dir: path, chain });
            } else if (entry.isFile() && languageForPath(name) && !isIgnored(chain, path, false)) {
                try {
                    const stats = statSync(join(absolute, name));
                    if (stats.size <= MAX_FILE_BYTES) files.push({ path, mtimeMs: stats.mtimeMs, size: stats.size });
                } catch {
                    continue;
                }
            }
        }
    }

    return { files, dirs };
}
