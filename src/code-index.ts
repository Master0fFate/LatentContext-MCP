import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, statSync, watch, writeFileSync, type FSWatcher } from "node:fs";
import { rename, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join, sep } from "node:path";
import { deserialize, serialize } from "node:v8";
import { languageForPath, REF_KINDS, SYMBOL_KINDS, type LanguageId, type RefKind, type SymbolKind } from "./code-languages.js";
import type { ParsedFile, RefTuple, SymbolTuple } from "./code-parser.js";
import { parseInWorkers, scanInWorker } from "./code-pool.js";
import { IGNORED_DIRS, isPathIgnored, MAX_FILE_BYTES, scanTree, type ScanEntry } from "./code-scan.js";
import { parseFiles, type ParseItem } from "./code-worker.js";
import { log } from "./logger.js";

export interface CodeSymbol {
    name: string;
    kind: SymbolKind;
    start: number;
    end: number;
    owner: string;
    signature: string;
    wordCount: number;
    file: CodeFile;
    calls: CodeRef[];
}

export interface CodeRef {
    name: string;
    kind: RefKind;
    line: number;
    from: CodeSymbol | null;
    file: CodeFile;
}

export interface CodeFile {
    path: string;
    lang: LanguageId;
    mtimeMs: number;
    size: number;
    symbols: CodeSymbol[];
    refs: CodeRef[];
    pathWords: string[];
    index: CodeIndex;
}

type CachedFile = [path: string, lang: LanguageId, mtimeMs: number, size: number, symbols: SymbolTuple[], refs: RefTuple[]];

interface CachePayload {
    version: string;
    root: string;
    files: CachedFile[];
}

const CACHE_VERSION = createRequire(import.meta.url)("../package.json").version as string;
const MAX_FILES = 200_000;
const INLINE_PARSE_LIMIT = 48;
const MAX_DIRTY = 4_000;
const MAX_LINUX_WATCHED_DIRS = 8_000;
const SAVE_DELAY_MS = 1_500;
const RECURSIVE_WATCH = process.platform === "win32" || process.platform === "darwin";

export function splitWords(identifier: string): string[] {
    const words = identifier
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter((word) => word.length > 1);
    const whole = identifier.toLowerCase();
    if (!words.includes(whole)) words.push(whole);
    return [...new Set(words)];
}

function addTo<T>(map: Map<string, Set<T>>, key: string, value: T): void {
    const set = map.get(key);
    if (set) set.add(value);
    else map.set(key, new Set([value]));
}

function removeFrom<T>(map: Map<string, Set<T>>, key: string, value: T): void {
    const set = map.get(key);
    if (set && set.delete(value) && set.size === 0) map.delete(key);
}

function toRelative(path: string): string {
    return sep === "/" ? path : path.split(sep).join("/");
}

export class CodeIndex {
    readonly files = new Map<string, CodeFile>();
    readonly defsByName = new Map<string, Set<CodeSymbol>>();
    readonly defsByWord = new Map<string, Set<CodeSymbol>>();
    readonly refsByName = new Map<string, Set<CodeRef>>();
    symbolCount = 0;

    private readonly cachePath: string;
    private readonly dirty = new Set<string>();
    private readonly dirWatchers = new Map<string, FSWatcher>();
    private rootWatcher: FSWatcher | null = null;
    private loaded = false;
    private watching = false;
    private rescan = false;
    private unsaved = false;
    private refreshing: Promise<void> | null = null;
    private saveTimer: NodeJS.Timeout | null = null;

    readonly root: string;

    constructor(root: string, cacheDir: string) {
        this.root = root;
        this.cachePath = join(cacheDir, `${createHash("sha1").update(root).digest("hex").slice(0, 16)}.bin`);
    }

    ready(): Promise<void> {
        this.refreshing ??= this.refresh().finally(() => {
            this.refreshing = null;
        });
        return this.refreshing;
    }

    close(): void {
        this.stopWatching();
        if (this.saveTimer) clearTimeout(this.saveTimer);
        this.saveTimer = null;
        if (!this.unsaved) return;
        try {
            mkdirSync(dirname(this.cachePath), { recursive: true });
            writeFileSync(this.cachePath, this.snapshot());
            this.unsaved = false;
        } catch (error) {
            log("warning", `Failed to save code index for ${this.root}`, error);
        }
    }

    private async refresh(): Promise<void> {
        if (!this.loaded) {
            this.loaded = true;
            this.loadCache();
            await this.fullSync();
            return;
        }
        if (this.rescan || !this.watching) {
            await this.fullSync();
            return;
        }
        if (this.dirty.size > 0) await this.flushDirty();
    }

    private async fullSync(): Promise<void> {
        this.rescan = false;
        this.dirty.clear();
        const startedAt = performance.now();
        const { files, dirs } = await scanInWorker(this.root, MAX_FILES);
        const seen = new Set<string>();
        const changed: ScanEntry[] = [];
        for (const entry of files) {
            seen.add(entry.path);
            const current = this.files.get(entry.path);
            if (!current || current.mtimeMs !== entry.mtimeMs || current.size !== entry.size) changed.push(entry);
        }
        let removed = 0;
        for (const path of [...this.files.keys()]) {
            if (!seen.has(path)) {
                this.remove(path);
                removed++;
            }
        }
        await this.parse(changed);
        if (changed.length > 0 || removed > 0) this.scheduleSave();
        this.watch(dirs);
        if (files.length >= MAX_FILES) log("warning", `Code index for ${this.root} stopped at ${MAX_FILES} files`);
        log("info", `Code index ${this.root}: ${this.files.size} files, ${this.symbolCount} symbols, ${changed.length} parsed, ${removed} removed in ${Math.round(performance.now() - startedAt)}ms`);
    }

    private async flushDirty(): Promise<void> {
        const paths = [...this.dirty];
        this.dirty.clear();
        const changed: ScanEntry[] = [];
        let removed = 0;
        for (const path of paths) {
            const stats = statSync(join(this.root, path), { throwIfNoEntry: false });
            if (!stats) {
                removed += this.removeTree(path);
            } else if (stats.isDirectory()) {
                if (isPathIgnored(this.root, path)) continue;
                const { files, dirs } = scanTree(this.root, path);
                for (const entry of files) {
                    const current = this.files.get(entry.path);
                    if (!current || current.mtimeMs !== entry.mtimeMs || current.size !== entry.size) changed.push(entry);
                }
                if (!RECURSIVE_WATCH) this.watchDirs(dirs);
            } else if (stats.isFile()) {
                const current = this.files.get(path);
                if (current && current.mtimeMs === stats.mtimeMs && current.size === stats.size) continue;
                if (!languageForPath(path) || stats.size > MAX_FILE_BYTES || isPathIgnored(this.root, path)) {
                    if (current) removed += this.removeTree(path);
                    continue;
                }
                changed.push({ path, mtimeMs: stats.mtimeMs, size: stats.size });
            }
        }
        await this.parse(changed);
        if (changed.length > 0 || removed > 0) this.scheduleSave();
    }

    private async parse(entries: readonly ScanEntry[]): Promise<void> {
        if (entries.length === 0) return;
        const items: ParseItem[] = entries.map((entry) => ({ path: entry.path, lang: languageForPath(entry.path)! }));
        const stats = new Map(entries.map((entry) => [entry.path, entry]));
        const ingest = (item: ParseItem, parsed: ParsedFile | null): void => {
            const entry = stats.get(item.path)!;
            if (parsed) this.ingest(item.path, item.lang, entry.mtimeMs, entry.size, parsed.symbols, parsed.refs);
            else this.remove(item.path);
        };
        if (items.length <= INLINE_PARSE_LIMIT) {
            const results = await parseFiles(this.root, items);
            for (let i = 0; i < items.length; i++) ingest(items[i], results[i]);
        } else {
            await parseInWorkers(this.root, items, ingest);
        }
    }

    private ingest(path: string, lang: LanguageId, mtimeMs: number, size: number, symbolTuples: SymbolTuple[], refTuples: RefTuple[]): void {
        this.remove(path);
        const file: CodeFile = {
            path, lang, mtimeMs, size,
            symbols: new Array(symbolTuples.length),
            refs: new Array(refTuples.length),
            pathWords: splitWords(path.replace(/\.[^./]+$/, "")),
            index: this,
        };
        for (let i = 0; i < symbolTuples.length; i++) {
            const [name, kind, start, end, owner, signature] = symbolTuples[i];
            const words = splitWords(name);
            const symbol: CodeSymbol = {
                name, kind: SYMBOL_KINDS[kind], start, end, owner, signature, wordCount: words.length, file, calls: [],
            };
            file.symbols[i] = symbol;
            addTo(this.defsByName, name, symbol);
            for (const word of words) addTo(this.defsByWord, word, symbol);
        }
        for (let i = 0; i < refTuples.length; i++) {
            const [name, kind, line, from] = refTuples[i];
            const owner = from >= 0 ? file.symbols[from] : null;
            const ref: CodeRef = { name, kind: REF_KINDS[kind], line, from: owner, file };
            file.refs[i] = ref;
            owner?.calls.push(ref);
            addTo(this.refsByName, name, ref);
        }
        this.symbolCount += symbolTuples.length;
        this.files.set(path, file);
    }

    private remove(path: string): boolean {
        const file = this.files.get(path);
        if (!file) return false;
        for (const symbol of file.symbols) {
            removeFrom(this.defsByName, symbol.name, symbol);
            for (const word of splitWords(symbol.name)) removeFrom(this.defsByWord, word, symbol);
        }
        for (const ref of file.refs) removeFrom(this.refsByName, ref.name, ref);
        this.symbolCount -= file.symbols.length;
        this.files.delete(path);
        return true;
    }

    private removeTree(path: string): number {
        if (this.remove(path)) return 1;
        const prefix = `${path}/`;
        let removed = 0;
        for (const candidate of [...this.files.keys()]) {
            if (candidate.startsWith(prefix) && this.remove(candidate)) removed++;
        }
        const watcher = this.dirWatchers.get(path);
        if (watcher) {
            watcher.close();
            this.dirWatchers.delete(path);
        }
        return removed;
    }

    private loadCache(): void {
        try {
            const payload = deserialize(readFileSync(this.cachePath)) as CachePayload;
            if (payload.version !== CACHE_VERSION || payload.root !== this.root) return;
            for (const [path, lang, mtimeMs, size, symbols, refs] of payload.files) this.ingest(path, lang, mtimeMs, size, symbols, refs);
        } catch {
            this.files.clear();
            this.defsByName.clear();
            this.defsByWord.clear();
            this.refsByName.clear();
            this.symbolCount = 0;
        }
    }

    private snapshot(): Buffer {
        const files: CachedFile[] = [];
        for (const file of this.files.values()) {
            const positions = new Map(file.symbols.map((symbol, position) => [symbol, position]));
            files.push([
                file.path, file.lang, file.mtimeMs, file.size,
                file.symbols.map((symbol): SymbolTuple => [symbol.name, SYMBOL_KINDS.indexOf(symbol.kind), symbol.start, symbol.end, symbol.owner, symbol.signature]),
                file.refs.map((ref): RefTuple => [ref.name, REF_KINDS.indexOf(ref.kind), ref.line, ref.from ? positions.get(ref.from)! : -1]),
            ]);
        }
        return serialize({ version: CACHE_VERSION, root: this.root, files } satisfies CachePayload);
    }

    private scheduleSave(): void {
        this.unsaved = true;
        if (this.saveTimer) return;
        this.saveTimer = setTimeout(() => {
            this.saveTimer = null;
            void this.save();
        }, SAVE_DELAY_MS);
        this.saveTimer.unref();
    }

    private async save(): Promise<void> {
        this.unsaved = false;
        const temporary = `${this.cachePath}.${process.pid}.tmp`;
        try {
            mkdirSync(dirname(this.cachePath), { recursive: true });
            await writeFile(temporary, this.snapshot());
            await rename(temporary, this.cachePath);
        } catch (error) {
            this.unsaved = true;
            log("warning", `Failed to save code index for ${this.root}`, error);
        }
    }

    private onEvent(path: string | null): void {
        if (!path) {
            this.rescan = true;
            return;
        }
        const relative = toRelative(path);
        const segments = relative.split("/");
        if (segments.some((segment) => IGNORED_DIRS.has(segment))) return;
        if (segments[segments.length - 1] === ".gitignore" || this.dirty.size >= MAX_DIRTY) {
            this.rescan = true;
            this.dirty.clear();
            return;
        }
        if (!this.rescan) this.dirty.add(relative);
    }

    private onWatchError(error: unknown): void {
        log("warning", `File watching stopped for ${this.root}; falling back to rescans`, error);
        this.stopWatching();
    }

    private watch(dirs: readonly string[]): void {
        if (this.watching) {
            if (!RECURSIVE_WATCH) this.watchDirs(dirs);
            return;
        }
        try {
            if (RECURSIVE_WATCH) {
                this.rootWatcher = watch(this.root, { recursive: true, persistent: false }, (_event, name) => this.onEvent(name));
                this.rootWatcher.on("error", (error) => this.onWatchError(error));
                this.watching = true;
            } else if (dirs.length <= MAX_LINUX_WATCHED_DIRS) {
                this.watching = true;
                this.watchDirs(dirs);
            }
        } catch (error) {
            this.onWatchError(error);
        }
    }

    private watchDirs(dirs: readonly string[]): void {
        for (const dir of dirs) {
            if (this.dirWatchers.has(dir)) continue;
            if (this.dirWatchers.size >= MAX_LINUX_WATCHED_DIRS) {
                this.onWatchError(new Error(`More than ${MAX_LINUX_WATCHED_DIRS} directories`));
                return;
            }
            try {
                const watcher = watch(join(this.root, dir), { persistent: false }, (_event, name) =>
                    this.onEvent(name ? (dir ? `${dir}/${name}` : name) : null)
                );
                watcher.on("error", () => {
                    watcher.close();
                    this.dirWatchers.delete(dir);
                    this.rescan = true;
                });
                this.dirWatchers.set(dir, watcher);
            } catch {
                this.rescan = true;
            }
        }
    }

    private stopWatching(): void {
        this.watching = false;
        this.rootWatcher?.close();
        this.rootWatcher = null;
        for (const watcher of this.dirWatchers.values()) watcher.close();
        this.dirWatchers.clear();
    }
}
