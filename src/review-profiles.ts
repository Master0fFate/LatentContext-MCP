import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { log } from "./logger.js";

export const REVIEW_PROFILES_ENV = "LATENTCONTEXT_REVIEW_PROFILES";

const BUNDLED_PROFILES = fileURLToPath(new URL("../prompts/", import.meta.url));

export interface ReviewProfile {
    id: string;
    title: string;
    body: string;
}

export function loadReviewProfiles(dir: string = process.env[REVIEW_PROFILES_ENV]?.trim() || BUNDLED_PROFILES): ReviewProfile[] {
    let names: string[];
    try {
        names = readdirSync(dir).filter((name) => name.toLowerCase().endsWith(".md")).sort();
    } catch (error) {
        log("warning", `Review profiles unavailable in ${dir}`, error);
        return [];
    }
    const profiles = new Map<string, ReviewProfile>();
    for (const name of names) {
        const id = name.slice(0, -3).toLowerCase();
        if (profiles.has(id)) continue;
        const body = readFileSync(join(dir, name), "utf8");
        profiles.set(id, { id, title: /^#\s+(.+)$/m.exec(body)?.[1].trim() ?? id, body });
    }
    return [...profiles.values()];
}

export function defaultReviewProfile(profiles: readonly ReviewProfile[]): ReviewProfile | undefined {
    const universal = profiles
        .filter((profile) => profile.id.startsWith("universal-auditor"))
        .sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true }));
    return universal[0] ?? profiles[0];
}
