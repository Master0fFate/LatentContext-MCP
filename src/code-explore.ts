import { readFileSync } from "node:fs";
import { basename, join, posix } from "node:path";
import { splitWords, type CodeFile, type CodeIndex, type CodeRef, type CodeSymbol } from "./code-index.js";
import { CONTAINER_KINDS } from "./code-languages.js";

export interface ExploreOptions {
    maxFiles: number;
    tokenBudget: number;
}

export interface ExploreSymbol {
    name: string;
    kind: string;
    file: string;
    startLine: number;
    endLine: number;
}

export interface ExploreResult {
    text: string;
    symbols: ExploreSymbol[];
    indexedFiles: number;
    indexedSymbols: number;
}

interface QueryTerms {
    names: Array<{ name: string; owner: string }>;
    words: string[];
    files: string[];
}

interface Range {
    start: number;
    end: number;
}

interface FileView {
    file: CodeFile;
    lines: string[];
    ranges: Range[];
    labels: string[];
}

const STOP_WORDS = new Set([
    "a", "an", "and", "are", "about", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "if", "in", "into",
    "is", "it", "its", "me", "of", "on", "or", "show", "that", "the", "this", "to", "what", "when", "where", "which", "who",
    "why", "with", "work", "works", "working", "explain", "find", "get", "code", "function", "functions", "method", "methods",
    "class", "classes", "file", "files", "call", "calls", "called", "caller", "callers", "flow", "implementation", "implemented",
    "used", "uses", "using", "between", "through", "should", "would", "could", "there", "their", "they", "them", "we", "our",
]);

const IDENTIFIER = /[A-Za-z_$][\w$]*(?:(?:\.|::|#|->)[A-Za-z_$][\w$]*)*/g;
const FILE_TOKEN = /[\w@.\\/-]*\.[A-Za-z][A-Za-z0-9+]{0,5}(?![\w(])/g;
const CONTAINER_OUTLINE_LINES = 80;
const MAX_SYMBOL_LINES = 160;
const MAX_LINE_CHARS = 320;
const MAX_CANDIDATES = 60;
const MAX_WORD_FANOUT = 5_000;
const PATH_SEARCH_DEPTH = 4;
const PATH_SEARCH_VISITS = 4_000;
const IMPACT_DEPTH = 3;
const IMPACT_VISITS = 3_000;
const LIST_LIMIT = 8;

function parseQuery(query: string): QueryTerms {
    const files = [...query.matchAll(FILE_TOKEN)]
        .map((match) => match[0].replace(/\\/g, "/").replace(/^\.\//, "").toLowerCase())
        .filter((token) => token.length > 2 && !/^\.?\d/.test(token));
    const names: QueryTerms["names"] = [];
    const words = new Set<string>();
    for (const match of query.matchAll(IDENTIFIER)) {
        const parts = match[0].split(/\.|::|#|->/);
        for (let i = 0; i < parts.length; i++) {
            const name = parts[i];
            if (name === name.toLowerCase() && STOP_WORDS.has(name)) continue;
            names.push({ name, owner: i > 0 ? parts[i - 1] : "" });
            for (const word of splitWords(name)) if (!STOP_WORDS.has(word)) words.add(word);
        }
    }
    return { names, words: [...words], files };
}

export function qualified(symbol: CodeSymbol): string {
    return symbol.owner ? `${symbol.owner}.${symbol.name}` : symbol.name;
}

function directoryOf(path: string): string {
    return posix.dirname(path);
}

export function resolveRef(ref: CodeRef): CodeSymbol[] {
    const defs = ref.file.index.defsByName.get(ref.name);
    if (!defs) return [];
    if (defs.size === 1) return [...defs];
    const sameFile: CodeSymbol[] = [];
    const sameDirectory: CodeSymbol[] = [];
    const directory = directoryOf(ref.file.path);
    for (const def of defs) {
        if (def.file === ref.file) sameFile.push(def);
        else if (directoryOf(def.file.path) === directory) sameDirectory.push(def);
    }
    if (sameFile.length > 0) return sameFile;
    if (sameDirectory.length > 0) return sameDirectory;
    return defs.size <= 3 ? [...defs] : [];
}

export function referencesTo(symbol: CodeSymbol): CodeRef[] {
    const refs = symbol.file.index.refsByName.get(symbol.name);
    if (!refs) return [];
    const result: CodeRef[] = [];
    for (const ref of refs) if (ref.from !== symbol && resolveRef(ref).includes(symbol)) result.push(ref);
    return result.sort((a, b) => Number(b.kind === "call") - Number(a.kind === "call"));
}

function callees(symbol: CodeSymbol): Array<{ ref: CodeRef; target: CodeSymbol }> {
    const result: Array<{ ref: CodeRef; target: CodeSymbol }> = [];
    const seen = new Set<CodeSymbol>([symbol]);
    for (const ref of symbol.calls) {
        if (ref.kind !== "call" && ref.kind !== "inherit") continue;
        for (const target of resolveRef(ref)) {
            if (seen.has(target)) continue;
            seen.add(target);
            result.push({ ref, target });
        }
    }
    return result;
}

function score(indexes: readonly CodeIndex[], terms: QueryTerms): Map<CodeSymbol, number> {
    const scores = new Map<CodeSymbol, number>();
    const add = (symbol: CodeSymbol, value: number): void => {
        scores.set(symbol, (scores.get(symbol) ?? 0) + value);
    };

    for (const index of indexes) {
        const total = Math.max(index.symbolCount, 1);
        for (const { name, owner } of terms.names) {
            const exact = index.defsByName.get(name);
            if (exact) {
                for (const symbol of exact) add(symbol, 10 + (owner && symbol.owner === owner ? 8 : 0) + (CONTAINER_KINDS.has(symbol.kind) ? 1 : 0));
            }
            const lower = name.toLowerCase();
            const folded = index.defsByWord.get(lower);
            if (folded) {
                for (const symbol of folded) if (symbol.name !== name && symbol.name.toLowerCase() === lower) add(symbol, 7);
            }
        }

        for (const word of terms.words) {
            const matches = index.defsByWord.get(word);
            if (matches && matches.size <= MAX_WORD_FANOUT) {
                const idf = Math.log(1 + total / matches.size);
                for (const symbol of matches) add(symbol, idf / Math.sqrt(symbol.wordCount));
            }
            if (word.length < 4) continue;
            for (const [vocabulary, symbols] of index.defsByWord) {
                if (vocabulary === word || symbols.size > MAX_WORD_FANOUT || !vocabulary.startsWith(word)) continue;
                const idf = Math.log(1 + total / symbols.size) * 0.4;
                for (const symbol of symbols) add(symbol, idf / Math.sqrt(symbol.wordCount));
            }
        }

        for (const token of terms.files) {
            for (const file of index.files.values()) {
                const path = file.path.toLowerCase();
                if (path !== token && !path.endsWith(`/${token}`)) continue;
                for (const symbol of file.symbols) if (!symbol.owner) add(symbol, 6);
            }
        }
    }

    if (terms.words.length > 0) {
        for (const [symbol, value] of scores) {
            let hits = 0;
            for (const word of terms.words) if (symbol.file.pathWords.includes(word)) hits++;
            const kindWeight = symbol.kind === "constant" ? 0.6 : symbol.kind === "type" || symbol.kind === "impl" ? 0.85 : 1;
            scores.set(symbol, (value + hits * 0.75) * kindWeight);
        }
    }
    return scores;
}

function findPath(from: CodeSymbol, targets: ReadonlySet<CodeSymbol>): CodeSymbol[] | null {
    const parents = new Map<CodeSymbol, CodeSymbol | null>([[from, null]]);
    let frontier = [from];
    for (let depth = 0; depth < PATH_SEARCH_DEPTH && frontier.length > 0 && parents.size < PATH_SEARCH_VISITS; depth++) {
        const next: CodeSymbol[] = [];
        for (const symbol of frontier) {
            for (const { target } of callees(symbol)) {
                if (parents.has(target)) continue;
                parents.set(target, symbol);
                if (targets.has(target)) {
                    const path = [target];
                    for (let step = parents.get(target); step; step = parents.get(step)) path.push(step);
                    return path.reverse();
                }
                next.push(target);
            }
        }
        frontier = next;
    }
    return null;
}

export function impactOf(symbol: CodeSymbol): { symbols: number; files: number } {
    const seen = new Set<CodeSymbol>([symbol]);
    const files = new Set<CodeFile>();
    let frontier = [symbol];
    for (let depth = 0; depth < IMPACT_DEPTH && frontier.length > 0 && seen.size < IMPACT_VISITS; depth++) {
        const next: CodeSymbol[] = [];
        for (const current of frontier) {
            for (const ref of referencesTo(current)) {
                files.add(ref.file);
                if (ref.from && !seen.has(ref.from)) {
                    seen.add(ref.from);
                    next.push(ref.from);
                }
            }
        }
        frontier = next;
    }
    return { symbols: seen.size - 1, files: files.size };
}

function rangeFor(symbol: CodeSymbol): { range: Range; outline: number[] } {
    const length = symbol.end - symbol.start + 1;
    if (CONTAINER_KINDS.has(symbol.kind) && length > CONTAINER_OUTLINE_LINES) {
        const members = symbol.file.symbols
            .filter((member) => member !== symbol && member.start > symbol.start && member.end <= symbol.end && member.owner === symbol.name)
            .map((member) => member.start);
        if (members.length > 0) {
            const headerEnd = Math.min(members[0] - 1, symbol.start + 11);
            return { range: { start: symbol.start, end: Math.max(symbol.start, headerEnd) }, outline: [...members, symbol.end] };
        }
    }
    if (length > MAX_SYMBOL_LINES) return { range: { start: symbol.start, end: symbol.start + MAX_SYMBOL_LINES - 1 }, outline: [] };
    return { range: { start: symbol.start, end: symbol.end }, outline: [] };
}

function covered(ranges: readonly Range[], start: number, end: number): boolean {
    return ranges.some((range) => range.start <= start && range.end >= end);
}

function cost(lines: readonly string[], ranges: readonly Range[], range: Range): number {
    let chars = 0;
    for (let line = range.start; line <= range.end && line <= lines.length; line++) {
        if (!covered(ranges, line, line)) chars += Math.min(lines[line - 1].length, MAX_LINE_CHARS) + 8;
    }
    return chars;
}

function mergeRanges(ranges: readonly Range[]): Range[] {
    const sorted = [...ranges].sort((a, b) => a.start - b.start);
    const merged: Range[] = [];
    for (const range of sorted) {
        const last = merged[merged.length - 1];
        if (last && range.start <= last.end + 2) last.end = Math.max(last.end, range.end);
        else merged.push({ ...range });
    }
    return merged;
}

function renderLines(lines: readonly string[], ranges: readonly Range[]): string {
    const out: string[] = [];
    for (const range of mergeRanges(ranges)) {
        if (out.length > 0) out.push("⋮");
        for (let line = range.start; line <= Math.min(range.end, lines.length); line++) {
            const text = lines[line - 1];
            out.push(`${line}\t${text.length > MAX_LINE_CHARS ? `${text.slice(0, MAX_LINE_CHARS)}…` : text}`);
        }
    }
    return out.join("\n");
}

function readLines(file: CodeFile): string[] | null {
    try {
        return readFileSync(join(file.index.root, file.path), "utf8").split(/\r?\n/);
    } catch {
        return null;
    }
}

function suggestions(indexes: readonly CodeIndex[], terms: QueryTerms): string[] {
    const found = new Set<string>();
    for (const index of indexes) {
        for (const word of terms.words) {
            if (word.length < 3) continue;
            for (const [vocabulary, symbols] of index.defsByWord) {
                if (!vocabulary.includes(word)) continue;
                for (const symbol of symbols) {
                    found.add(symbol.name);
                    if (found.size >= 12) return [...found];
                }
            }
        }
    }
    return [...found];
}

export function explore(indexes: readonly CodeIndex[], query: string, options: ExploreOptions): ExploreResult {
    const indexedFiles = indexes.reduce((sum, index) => sum + index.files.size, 0);
    const indexedSymbols = indexes.reduce((sum, index) => sum + index.symbolCount, 0);
    const display = (file: CodeFile): string => (indexes.length > 1 ? `${basename(file.index.root)}/${file.path}` : file.path);
    const location = (file: CodeFile, line: number): string => `${display(file)}:${line}`;
    const header = `${indexedFiles} files, ${indexedSymbols} symbols indexed${indexes.length > 1 ? ` across ${indexes.length} roots` : ""}.`;

    const terms = parseQuery(query);
    const scores = score(indexes, terms);
    const ranked = [...scores].sort((a, b) => b[1] - a[1]);
    const cutoff = ranked.length > 0 ? ranked[0][1] * 0.2 : 0;
    const seeds = ranked.filter(([, value]) => value >= cutoff).slice(0, MAX_CANDIDATES).map(([symbol]) => symbol);

    if (seeds.length === 0) {
        const similar = suggestions(indexes, terms);
        const text = [header, `No symbols match "${query}".`, similar.length > 0 ? `Similar names: ${similar.join(", ")}` : ""]
            .filter(Boolean).join("\n");
        return { text, symbols: [], indexedFiles, indexedSymbols };
    }

    const seedSet = new Set(seeds);
    const paths: CodeSymbol[][] = [];
    const bridges: CodeSymbol[] = [];
    for (const seed of seeds.slice(0, 6)) {
        const path = findPath(seed, new Set([...seedSet].filter((other) => other !== seed)));
        if (!path || paths.some((existing) => existing.length === path.length && existing.every((step, i) => step === path[i]))) continue;
        paths.push(path);
        for (const step of path.slice(1, -1)) if (!seedSet.has(step) && !bridges.includes(step)) bridges.push(step);
    }

    const budget = options.tokenBudget * 4 * 0.8;
    let spent = 0;
    const views = new Map<CodeFile, FileView>();
    const shown: CodeSymbol[] = [];
    const more: CodeSymbol[] = [];

    for (const symbol of [...seeds, ...bridges]) {
        let view = views.get(symbol.file);
        if (!view) {
            if (views.size >= options.maxFiles) {
                more.push(symbol);
                continue;
            }
            const lines = readLines(symbol.file);
            if (!lines) continue;
            view = { file: symbol.file, lines, ranges: [], labels: [] };
        }
        if (covered(view.ranges, symbol.start, symbol.end)) {
            shown.push(symbol);
            continue;
        }
        const { range, outline } = rangeFor(symbol);
        const extra = cost(view.lines, view.ranges, range) + outline.reduce((sum, line) => sum + cost(view.lines, view.ranges, { start: line, end: line }), 0);
        if (spent + extra > budget && shown.length > 0) {
            more.push(symbol);
            continue;
        }
        spent += extra;
        view.ranges.push(range, ...outline.map((line) => ({ start: line, end: line })));
        const truncated = range.end < symbol.end && outline.length === 0 ? ", truncated" : outline.length > 0 ? ", outline" : "";
        view.labels.push(`${symbol.kind} ${qualified(symbol)} L${symbol.start}-${symbol.end}${truncated}`);
        views.set(symbol.file, view);
        shown.push(symbol);
    }

    const sections: string[] = [header];
    for (const view of views.values()) {
        sections.push(`## ${display(view.file)}\n${view.labels.join(" · ")}\n${renderLines(view.lines, view.ranges)}`);
    }

    if (paths.length > 0) {
        sections.push(`## Call paths\n${paths.map((path) => path.map(qualified).join(" → ")).join("\n")}`);
    }

    const relations: string[] = [];
    for (const symbol of shown.slice(0, 6)) {
        const refs = referencesTo(symbol);
        if (refs.length > 0) {
            const listed = refs.slice(0, LIST_LIMIT).map((ref) => `${ref.from ? qualified(ref.from) : "(top level)"} ${location(ref.file, ref.line)}`);
            relations.push(`${qualified(symbol)} ← ${listed.join(", ")}${refs.length > LIST_LIMIT ? ` +${refs.length - LIST_LIMIT} more` : ""}`);
        }
        const targets = callees(symbol);
        if (targets.length > 0) {
            const listed = targets.slice(0, LIST_LIMIT).map(({ target }) => `${qualified(target)} ${location(target.file, target.start)}`);
            relations.push(`${qualified(symbol)} → ${listed.join(", ")}${targets.length > LIST_LIMIT ? ` +${targets.length - LIST_LIMIT} more` : ""}`);
        }
    }
    if (relations.length > 0) sections.push(`## References and calls\n${relations.join("\n")}`);

    const impact = shown.length > 0 ? impactOf(shown[0]) : { symbols: 0, files: 0 };
    if (impact.symbols > 0) {
        sections.push(`## Impact\n${impact.symbols} symbols in ${impact.files} files reach ${qualified(shown[0])} within ${IMPACT_DEPTH} hops.`);
    }

    if (more.length > 0) {
        sections.push(`## More matches\n${more.slice(0, 15).map((symbol) => `${symbol.kind} ${qualified(symbol)} ${location(symbol.file, symbol.start)}`).join("\n")}`);
    }

    return {
        text: sections.join("\n\n"),
        symbols: shown.map((symbol) => ({
            name: qualified(symbol),
            kind: symbol.kind,
            file: display(symbol.file),
            startLine: symbol.start,
            endLine: symbol.end,
        })),
        indexedFiles,
        indexedSymbols,
    };
}
