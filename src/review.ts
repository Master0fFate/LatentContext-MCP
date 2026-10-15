import { randomUUID } from "node:crypto";
import { isAbsolute, relative, resolve, sep } from "node:path";
import { impactOf, qualified, referencesTo } from "./code-explore.js";
import type { CodeIndex } from "./code-index.js";
import { languageForPath } from "./code-languages.js";
import {
    deleteReview,
    getReview,
    getReviewSteps,
    getReviewSummaries,
    insertReview,
    recordReviewChunk,
    type ReviewRow,
    type ReviewStepRow,
} from "./database.js";
import { checkChunk, isBlocking, SEVERITIES, type Finding, type Patch } from "./review-checks.js";
import type { ReviewProfile } from "./review-profiles.js";

export const VERDICTS = ["PASS", "NEEDS_WORK"] as const;
export const STEP_STATUSES = ["PENDING", "IN_PROGRESS", "VALIDATED"] as const;
export const REVIEW_STATUSES = ["ACTIVE", "COMPLETED"] as const;
export const MAX_REVIEW_STEPS = 100;

export type Verdict = (typeof VERDICTS)[number];
export type StepStatus = (typeof STEP_STATUSES)[number];

export interface ContextSymbol {
    name: string;
    kind: string;
    startLine: number;
    endLine: number;
    references: number;
    impactSymbols: number;
    impactFiles: number;
    callers: string[];
}

export interface ChunkInput {
    step: number;
    code: string;
    symbol?: string;
    findings?: Finding[];
}

export interface ChunkReview {
    reviewId: string;
    step: number;
    stepDescription: string;
    file: string;
    verdict: Verdict;
    stepStatus: StepStatus;
    validated: number;
    total: number;
    findings: Finding[];
    patches: Patch[];
    context: ContextSymbol[];
}

export interface StepSummary {
    step: number;
    description: string;
    status: string;
    chunks: number;
    lastVerdict: string | null;
    openFindings: Finding[];
}

export interface ReviewSummary {
    reviewId: string;
    root: string;
    profile: string | null;
    status: (typeof REVIEW_STATUSES)[number];
    validated: number;
    total: number;
    updatedAt: string;
    steps?: StepSummary[];
}

const REVIEW_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const LIST_LIMIT = 20;
const MAX_CONTEXT_SYMBOLS = 5;
const MAX_CALLERS_SHOWN = 5;

function expiryCutoff(): string {
    return new Date(Date.now() - REVIEW_TTL_MS).toISOString();
}

export function startReview(root: string, steps: readonly string[], profile: ReviewProfile | undefined): string {
    const id = randomUUID();
    const timestamp = new Date().toISOString();
    insertReview({ id, root, profile: profile?.id ?? null, created_at: timestamp, updated_at: timestamp }, steps, expiryCutoff());
    return id;
}

export function requireReview(id: string): ReviewRow {
    const review = getReview(id, expiryCutoff());
    if (!review) throw new Error(`Review not found or expired: ${id}`);
    return review;
}

export function endReview(id: string): void {
    requireReview(id);
    deleteReview(id);
}

export function reviewFilePath(root: string, file: string): string {
    const path = relative(root, resolve(root, file));
    if (path === "" || path === ".." || path.startsWith(`..${sep}`) || isAbsolute(path)) {
        throw new Error(`'file' must be a file inside the review root ${root}`);
    }
    return path.split(sep).join("/");
}

function localize(index: CodeIndex | undefined, file: string, names: ReadonlySet<string>): ContextSymbol[] {
    const indexed = index?.files.get(file);
    if (!indexed || names.size === 0) return [];
    return indexed.symbols
        .filter((symbol) => names.has(symbol.name))
        .slice(0, MAX_CONTEXT_SYMBOLS)
        .map((symbol) => {
            const refs = referencesTo(symbol);
            const impact = impactOf(symbol);
            return {
                name: qualified(symbol),
                kind: symbol.kind,
                startLine: symbol.start,
                endLine: symbol.end,
                references: refs.length,
                impactSymbols: impact.symbols,
                impactFiles: impact.files,
                callers: refs.slice(0, MAX_CALLERS_SHOWN).map((ref) => `${ref.from ? qualified(ref.from) : "(top level)"} ${ref.file.path}:${ref.line}`),
            };
        });
}

function bySeverity(a: Finding, b: Finding): number {
    return SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) || (a.line ?? 0) - (b.line ?? 0);
}

export async function reviewChunk(review: ReviewRow, file: string, input: ChunkInput, index: CodeIndex | undefined): Promise<ChunkReview> {
    const steps = getReviewSteps(review.id);
    const step = steps.find((candidate) => candidate.step === input.step);
    if (!step) throw new Error(`'step' must be between 1 and ${steps.length}.`);

    const check = await checkChunk(languageForPath(file), input.code);
    const findings = [...check.findings, ...(input.findings ?? [])].sort(bySeverity);
    const verdict: Verdict = findings.some(isBlocking) ? "NEEDS_WORK" : "PASS";
    const stepStatus: StepStatus = verdict === "PASS" ? "VALIDATED" : "IN_PROGRESS";
    const names = new Set(check.symbols);
    if (input.symbol) names.add(input.symbol);
    const context = localize(index, file, names);

    recordReviewChunk({
        reviewId: review.id,
        step: input.step,
        file,
        verdict,
        findings: JSON.stringify(findings.filter(isBlocking)),
        stepStatus,
    });

    return {
        reviewId: review.id,
        step: input.step,
        stepDescription: step.description,
        file,
        verdict,
        stepStatus,
        validated: steps.filter((candidate) => (candidate === step ? stepStatus : candidate.status) === "VALIDATED").length,
        total: steps.length,
        findings,
        patches: check.patches,
        context,
    };
}

function stepSummary(row: ReviewStepRow): StepSummary {
    return {
        step: row.step,
        description: row.description,
        status: row.status,
        chunks: row.chunks,
        lastVerdict: row.last_verdict,
        openFindings: row.last_verdict === "NEEDS_WORK" && row.last_findings ? JSON.parse(row.last_findings) as Finding[] : [],
    };
}

export function reviewSummaries(id?: string): ReviewSummary[] {
    const rows = getReviewSummaries(expiryCutoff(), LIST_LIMIT, id);
    if (id !== undefined && rows.length === 0) throw new Error(`Review not found or expired: ${id}`);
    return rows.map((row) => ({
        reviewId: row.id,
        root: row.root,
        profile: row.profile,
        status: row.validated === row.total ? "COMPLETED" : "ACTIVE",
        validated: row.validated,
        total: row.total,
        updatedAt: row.updated_at,
        ...(id === undefined ? {} : { steps: getReviewSteps(row.id).map(stepSummary) }),
    }));
}

function findingLine(finding: Finding): string {
    const location = finding.line ? ` L${finding.line}` : "";
    const evidence = finding.evidence ? ` \`${finding.evidence}\`` : "";
    return `${finding.severity} ${finding.category}${location} ${finding.message}${evidence}`;
}

export function formatReviewStart(id: string, root: string, steps: readonly string[], profile: ReviewProfile | undefined): string {
    const lines = [
        `Review ${id} started for ${root}.`,
        ...steps.map((step, index) => `${index + 1}. ${step}`),
    ];
    if (profile) lines.push("", `Review lens (${profile.id}):`, "", profile.body.trim());
    return lines.join("\n");
}

export function formatChunkReview(result: ChunkReview): string {
    const sections = [
        `${result.verdict} · step ${result.step} ${result.stepStatus} · ${result.validated}/${result.total} steps validated\nStep ${result.step}: ${result.stepDescription}`,
    ];
    if (result.findings.length > 0) sections.push(`## Findings\n${result.findings.map(findingLine).join("\n")}`);
    if (result.patches.length > 0) {
        sections.push(`## Patches\n${result.patches.map((patch) => `L${patch.line} \`${patch.search}\` → \`${patch.replace}\``).join("\n")}`);
    }
    if (result.context.length > 0) {
        sections.push(`## Context\n${result.context.map((symbol) => {
            const header = `${symbol.kind} ${symbol.name} ${result.file} L${symbol.startLine}-${symbol.endLine} · ${symbol.references} references · impact ${symbol.impactSymbols} symbols in ${symbol.impactFiles} files`;
            return symbol.callers.length > 0 ? `${header}\n  ← ${symbol.callers.join(", ")}` : header;
        }).join("\n")}`);
    }
    return sections.join("\n\n");
}

export function formatReviewSummaries(reviews: readonly ReviewSummary[]): string {
    if (reviews.length === 0) return "No live reviews.";
    return reviews.map((review) => {
        const header = `${review.reviewId} ${review.status} ${review.validated}/${review.total} validated · ${review.profile ?? "no profile"} · ${review.root} · updated ${review.updatedAt}`;
        if (!review.steps) return header;
        const steps = review.steps.map((step) => {
            const line = `${step.step}. [${step.status}] ${step.description} (${step.chunks} chunks${step.lastVerdict ? `, last ${step.lastVerdict}` : ""})`;
            return [line, ...step.openFindings.map((finding) => `   ${findingLine(finding)}`)].join("\n");
        });
        return [header, ...steps].join("\n");
    }).join("\n");
}
