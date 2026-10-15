import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parentPort } from "node:worker_threads";
import type { LanguageId } from "./code-languages.js";
import { parseSource, type ParsedFile } from "./code-parser.js";
import { scanTree } from "./code-scan.js";

export interface ParseItem {
    path: string;
    lang: LanguageId;
}

export type WorkerTask =
    | { kind: "scan"; root: string; maxFiles: number }
    | { kind: "parse"; root: string; files: ParseItem[] };

export async function parseFiles(root: string, files: readonly ParseItem[]): Promise<Array<ParsedFile | null>> {
    const results: Array<ParsedFile | null> = [];
    for (const file of files) {
        try {
            results.push(await parseSource(file.lang, readFileSync(join(root, file.path), "utf8")));
        } catch {
            results.push(null);
        }
    }
    return results;
}

parentPort?.on("message", (task: WorkerTask) => {
    const work = task.kind === "scan"
        ? Promise.resolve().then(() => scanTree(task.root, "", task.maxFiles))
        : parseFiles(task.root, task.files);
    work.then(
        (result) => parentPort!.postMessage({ result }),
        (error: unknown) => parentPort!.postMessage({ error: error instanceof Error ? error.message : String(error) })
    );
});
