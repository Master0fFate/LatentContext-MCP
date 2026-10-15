import treeSitter from "@vscode/tree-sitter-wasm";
import type { Language, Node, Parser as TreeSitterParser, Query } from "@vscode/tree-sitter-wasm";
import { CONTAINER_KINDS, LANGUAGES, REF_KINDS, SYMBOL_KINDS, type LanguageId, type RefKind, type SymbolKind } from "./code-languages.js";

export type SymbolTuple = [name: string, kind: number, start: number, end: number, owner: string, signature: string];
export type RefTuple = [name: string, kind: number, line: number, from: number];

export interface ParsedFile {
    symbols: SymbolTuple[];
    refs: RefTuple[];
}

interface Grammar {
    language: Language;
    query: Query;
}

interface Definition {
    name: string;
    kind: SymbolKind;
    owner: string;
    startIndex: number;
    endIndex: number;
    startRow: number;
    endRow: number;
}

interface Reference {
    name: string;
    kind: RefKind;
    index: number;
    row: number;
}

const MAX_NAME_LENGTH = 120;
const MAX_SIGNATURE_LENGTH = 200;
const EMPTY: ParsedFile = { symbols: [], refs: [] };

const KIND_PRIORITY: Record<SymbolKind, number> = {
    constant: 0, type: 1, function: 2, method: 2, module: 3, impl: 3, macro: 3,
    class: 4, struct: 4, interface: 4, trait: 4, enum: 4,
};
const SYMBOL_KIND_INDEX = new Map<string, number>(SYMBOL_KINDS.map((kind, index) => [kind, index]));
const REF_KIND_INDEX = new Map<string, number>(REF_KINDS.map((kind, index) => [kind, index]));

let parser: Promise<TreeSitterParser> | null = null;
const grammars = new Map<LanguageId, Promise<Grammar>>();

function getParser(): Promise<TreeSitterParser> {
    parser ??= treeSitter.Parser.init().then(() => new treeSitter.Parser());
    return parser;
}

function getGrammar(lang: LanguageId): Promise<Grammar> {
    let grammar = grammars.get(lang);
    if (!grammar) {
        const spec = LANGUAGES[lang];
        grammar = getParser()
            .then(() => treeSitter.Language.load(spec.grammar))
            .then((language) => ({ language, query: new treeSitter.Query(language, spec.query) }));
        grammars.set(lang, grammar);
    }
    return grammar;
}

function endRow(node: Node): number {
    const end = node.endPosition;
    return end.column === 0 && end.row > node.startPosition.row ? end.row - 1 : end.row;
}

function signatureAt(source: string, index: number): string {
    const lineStart = source.lastIndexOf("\n", index - 1) + 1;
    let lineEnd = source.indexOf("\n", index);
    if (lineEnd === -1) lineEnd = source.length;
    const line = source.slice(lineStart, Math.min(lineEnd, lineStart + MAX_SIGNATURE_LENGTH * 2)).trim();
    return line.length > MAX_SIGNATURE_LENGTH ? `${line.slice(0, MAX_SIGNATURE_LENGTH)}…` : line;
}

function collect(query: Query, root: Node): { definitions: Map<number, Definition>; references: Map<number, Reference> } {
    const definitions = new Map<number, Definition>();
    const references = new Map<number, Reference>();

    for (const match of query.matches(root)) {
        let nameNode: Node | null = null;
        let ownerNode: Node | null = null;
        let defNode: Node | null = null;
        let defKind: SymbolKind | null = null;

        for (const capture of match.captures) {
            const captureName = capture.name;
            if (captureName === "name") nameNode = capture.node;
            else if (captureName === "owner") ownerNode = capture.node;
            else if (captureName.startsWith("def.")) {
                defNode = capture.node;
                defKind = captureName.slice(4) as SymbolKind;
            } else if (captureName.startsWith("ref.")) {
                const kind = captureName.slice(4) as RefKind;
                const node = capture.node;
                const existing = references.get(node.startIndex);
                if (existing && REF_KIND_INDEX.get(existing.kind)! >= REF_KIND_INDEX.get(kind)!) continue;
                const name = node.text;
                if (name.length > 0 && name.length <= MAX_NAME_LENGTH) {
                    references.set(node.startIndex, { name, kind, index: node.startIndex, row: node.startPosition.row });
                }
            }
        }

        if (!nameNode || !defNode || !defKind) continue;
        const key = nameNode.startIndex;
        const existing = definitions.get(key);
        const owner = ownerNode?.text ?? "";
        if (existing) {
            if (!existing.owner && owner) existing.owner = owner;
            if (KIND_PRIORITY[defKind] <= KIND_PRIORITY[existing.kind]) continue;
        }
        const name = nameNode.text;
        if (name.length === 0 || name.length > MAX_NAME_LENGTH) continue;
        definitions.set(key, {
            name,
            kind: defKind,
            owner: owner || existing?.owner || "",
            startIndex: defNode.startIndex,
            endIndex: defNode.endIndex,
            startRow: defNode.startPosition.row,
            endRow: endRow(defNode),
        });
    }

    for (const key of definitions.keys()) references.delete(key);
    return { definitions, references };
}

function build(source: string, definitions: Map<number, Definition>, references: Map<number, Reference>): ParsedFile {
    const defs = [...definitions.values()].sort((a, b) => a.startIndex - b.startIndex || b.endIndex - a.endIndex);
    const symbols: SymbolTuple[] = new Array(defs.length);
    const stack: number[] = [];

    for (let i = 0; i < defs.length; i++) {
        const def = defs[i];
        while (stack.length > 0 && defs[stack[stack.length - 1]].endIndex <= def.startIndex) stack.pop();
        const parent = stack.length > 0 ? defs[stack[stack.length - 1]] : undefined;
        const inContainer = parent !== undefined && CONTAINER_KINDS.has(parent.kind);
        if (def.kind === "function" && inContainer) def.kind = "method";
        if (!def.owner && inContainer) def.owner = parent.name;
        symbols[i] = [
            def.name,
            SYMBOL_KIND_INDEX.get(def.kind)!,
            def.startRow + 1,
            def.endRow + 1,
            def.owner,
            signatureAt(source, def.startIndex),
        ];
        stack.push(i);
    }

    const refs = [...references.values()].sort((a, b) => a.index - b.index);
    const refTuples: RefTuple[] = new Array(refs.length);
    stack.length = 0;
    let next = 0;
    for (let i = 0; i < refs.length; i++) {
        const ref = refs[i];
        while (next < defs.length && defs[next].startIndex <= ref.index) {
            while (stack.length > 0 && defs[stack[stack.length - 1]].endIndex <= defs[next].startIndex) stack.pop();
            stack.push(next++);
        }
        while (stack.length > 0 && defs[stack[stack.length - 1]].endIndex <= ref.index) stack.pop();
        refTuples[i] = [ref.name, REF_KIND_INDEX.get(ref.kind)!, ref.row + 1, stack.length > 0 ? stack[stack.length - 1] : -1];
    }

    return { symbols, refs: refTuples };
}

export async function parseSource(lang: LanguageId, source: string): Promise<ParsedFile> {
    if (source.length === 0 || source.includes("\0")) return EMPTY;
    const [instance, grammar] = await Promise.all([getParser(), getGrammar(lang)]);
    instance.setLanguage(grammar.language);
    const tree = instance.parse(source);
    if (!tree) return EMPTY;
    try {
        const { definitions, references } = collect(grammar.query, tree.rootNode);
        return build(source, definitions, references);
    } finally {
        tree.delete();
    }
}
