import type { LanguageId } from "./code-languages.js";
import { checkSource } from "./code-parser.js";

export const SEVERITIES = ["critical", "high", "medium", "low", "info"] as const;
export type Severity = (typeof SEVERITIES)[number];

export interface Finding {
    severity: Severity;
    category: string;
    message: string;
    line?: number;
    evidence?: string;
}

export interface Patch {
    line: number;
    search: string;
    replace: string;
    reason: string;
}

export interface ChunkCheck {
    findings: Finding[];
    patches: Patch[];
    symbols: string[];
}

interface Rule {
    pattern: RegExp;
    severity: Severity;
    category: string;
    message: string;
    langs?: ReadonlySet<LanguageId>;
    fix?: (match: RegExpExecArray, lang: LanguageId | undefined) => string | null;
}

const JS = new Set<LanguageId>(["typescript", "tsx", "javascript"]);
const PYTHON = new Set<LanguageId>(["python"]);
const JVM = new Set<LanguageId>(["java", "kotlin", "scala"]);
const NATIVE = new Set<LanguageId>(["c", "cpp"]);
const SCRIPTING = new Set<LanguageId>(["typescript", "tsx", "javascript", "python", "php", "ruby"]);
const SQL_HOSTS = new Set<LanguageId>(["typescript", "tsx", "javascript", "python", "php", "ruby", "java", "kotlin", "csharp"]);
const MAX_MATCHES_PER_RULE = 5;
const MAX_EVIDENCE = 80;

function envName(identifier: string): string {
    return identifier.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/[^A-Za-z0-9]+/g, "_").toUpperCase();
}

const RULES: Rule[] = [
    { pattern: /^(?:<{7}|>{7})(?: |$)/gm, severity: "critical", category: "syntax", message: "Merge conflict marker." },
    { pattern: /\beval\s*\(/g, severity: "critical", category: "security", message: "eval runs arbitrary code.", langs: SCRIPTING },
    { pattern: /\bnew\s+Function\s*\(/g, severity: "critical", category: "security", message: "new Function runs arbitrary code.", langs: JS },
    {
        pattern: /\bchild_process\.exec(?:Sync)?\s*\(|\bexecSync\s*\(/g, severity: "critical", category: "security",
        message: "Shell command execution; use execFile with an argument array.", langs: JS,
    },
    {
        pattern: /\bos\.(?:system|popen)\s*\(|\bsubprocess\.\w+\([^)]*\bshell\s*=\s*True/g, severity: "critical", category: "security",
        message: "Shell command execution; pass an argument list without shell=True.", langs: PYTHON,
    },
    {
        pattern: /\b(?:shell_exec|passthru|proc_open|system)\s*\(/g, severity: "critical", category: "security",
        message: "Shell command execution.", langs: new Set<LanguageId>(["php"]),
    },
    {
        pattern: /\b(?:system|exec|spawn)\s*\(|%x[{(]/g, severity: "critical", category: "security",
        message: "Shell command execution.", langs: new Set<LanguageId>(["ruby"]),
    },
    { pattern: /Runtime\.getRuntime\(\)\.exec\s*\(/g, severity: "critical", category: "security", message: "Shell command execution.", langs: JVM },
    {
        pattern: /\bexec\.Command\s*\(\s*"(?:sh|bash|cmd|powershell)"/g, severity: "critical", category: "security",
        message: "Command runs through a shell.", langs: new Set<LanguageId>(["go"]),
    },
    { pattern: /\b(?:system|popen)\s*\(/g, severity: "critical", category: "security", message: "Shell command execution.", langs: NATIVE },
    {
        pattern: /\b(?:gets|strcpy|strcat|sprintf)\s*\(/g, severity: "high", category: "security",
        message: "Unbounded buffer write; use the bounded variant.", langs: NATIVE,
    },
    { pattern: /\bpickle\.loads?\s*\(/g, severity: "critical", category: "security", message: "pickle runs code from untrusted data.", langs: PYTHON },
    {
        pattern: /\byaml\.load\s*\((?![^)]*SafeLoader)/g, severity: "critical", category: "security",
        message: "yaml.load can build arbitrary objects.", langs: PYTHON, fix: () => "yaml.safe_load(",
    },
    { pattern: /\bunserialize\s*\(/g, severity: "critical", category: "security", message: "unserialize on untrusted data.", langs: new Set<LanguageId>(["php"]) },
    {
        pattern: /\bYAML\.load\s*\(/g, severity: "critical", category: "security",
        message: "YAML.load can build arbitrary objects.", langs: new Set<LanguageId>(["ruby"]), fix: () => "YAML.safe_load(",
    },
    { pattern: /\bMarshal\.load\s*\(/g, severity: "critical", category: "security", message: "Marshal.load on untrusted data.", langs: new Set<LanguageId>(["ruby"]) },
    { pattern: /\bBinaryFormatter\b/g, severity: "critical", category: "security", message: "BinaryFormatter is unsafe.", langs: new Set<LanguageId>(["csharp"]) },
    { pattern: /\bObjectInputStream\b/g, severity: "medium", category: "security", message: "Java deserialization; validate the input.", langs: JVM },
    {
        pattern: /\.innerHTML\s*=(?!=)/g, severity: "high", category: "security",
        message: "innerHTML assignment can inject markup.", langs: JS, fix: () => ".textContent =",
    },
    {
        pattern: /\.outerHTML\s*=(?!=)|\bdocument\.write\s*\(|\bdangerouslySetInnerHTML\b/g, severity: "high", category: "security",
        message: "Raw HTML sink can inject markup.", langs: JS,
    },
    {
        pattern: /\brejectUnauthorized\s*:\s*false\b/g, severity: "critical", category: "security",
        message: "TLS certificate checks are disabled.", langs: JS, fix: () => "rejectUnauthorized: true",
    },
    { pattern: /\bNODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['"]?0/g, severity: "critical", category: "security", message: "TLS certificate checks are disabled." },
    {
        pattern: /\bverify\s*=\s*False\b/g, severity: "critical", category: "security",
        message: "TLS certificate checks are disabled.", langs: PYTHON, fix: () => "verify=True",
    },
    {
        pattern: /\bInsecureSkipVerify\s*:\s*true\b/g, severity: "critical", category: "security",
        message: "TLS certificate checks are disabled.", langs: new Set<LanguageId>(["go"]), fix: () => "InsecureSkipVerify: false",
    },
    {
        pattern: /\b(?:execute|query|raw|exec)\s*\(\s*(?:f["']|["'][^"'\n]*["']\s*(?:\+|%|\.)|`[^`]*\$\{)/g, severity: "high", category: "security",
        message: "SQL built from string interpolation; use bound parameters.", langs: SQL_HOSTS,
    },
    {
        pattern: /\b([A-Za-z_]*(?:api[_-]?key|secret|password|passwd|token)[A-Za-z0-9_]*)(\s*[:=]\s*)(["'])[^"'\s]{4,}\3/gi,
        severity: "critical", category: "security", message: "Hardcoded secret.",
        fix: (match, lang) => (lang && JS.has(lang) ? `${match[1]}${match[2]}process.env.${envName(match[1])}` : null),
    },
    { pattern: /-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----/g, severity: "critical", category: "security", message: "Embedded private key." },
    { pattern: /\bAKIA[0-9A-Z]{16}\b/g, severity: "critical", category: "security", message: "Embedded AWS access key." },
    { pattern: /^[ \t]*debugger[ \t]*;?[ \t]*$/gm, severity: "medium", category: "debug", message: "Leftover debugger statement.", langs: JS },
    { pattern: /\bpdb\.set_trace\s*\(|\bbreakpoint\s*\(\s*\)/g, severity: "medium", category: "debug", message: "Leftover breakpoint.", langs: PYTHON },
    { pattern: /\bdbg!\s*\(/g, severity: "low", category: "debug", message: "Leftover dbg! macro.", langs: new Set<LanguageId>(["rust"]) },
];

function lineStarts(source: string): number[] {
    const starts = [0];
    for (let index = source.indexOf("\n"); index !== -1; index = source.indexOf("\n", index + 1)) starts.push(index + 1);
    return starts;
}

function lineAt(starts: readonly number[], offset: number): number {
    let low = 0;
    let high = starts.length - 1;
    while (low < high) {
        const middle = (low + high + 1) >> 1;
        if (starts[middle] <= offset) low = middle;
        else high = middle - 1;
    }
    return low + 1;
}

function evidence(text: string): string {
    const trimmed = text.trim();
    return trimmed.length > MAX_EVIDENCE ? `${trimmed.slice(0, MAX_EVIDENCE)}…` : trimmed;
}

function scanRules(lang: LanguageId | undefined, code: string, findings: Finding[], patches: Patch[]): void {
    const starts = lineStarts(code);
    for (const rule of RULES) {
        if (rule.langs && (!lang || !rule.langs.has(lang))) continue;
        rule.pattern.lastIndex = 0;
        let count = 0;
        for (let match = rule.pattern.exec(code); match && count < MAX_MATCHES_PER_RULE; match = rule.pattern.exec(code), count++) {
            const line = lineAt(starts, match.index);
            findings.push({ severity: rule.severity, category: rule.category, message: rule.message, line, evidence: evidence(match[0]) });
            const replace = rule.fix?.(match, lang);
            if (replace) patches.push({ line, search: match[0], replace, reason: rule.message });
        }
    }
}

export async function checkChunk(lang: LanguageId | undefined, code: string): Promise<ChunkCheck> {
    const findings: Finding[] = [];
    const patches: Patch[] = [];
    let symbols: string[] = [];
    if (lang) {
        const checked = await checkSource(lang, code);
        symbols = [...new Set(checked.symbols.map(([name]) => name))];
        for (const issue of checked.issues) {
            findings.push(issue.missing
                ? { severity: "medium", category: "syntax", message: `Missing ${issue.missing}.`, line: issue.line }
                : { severity: "medium", category: "syntax", message: "Syntax error.", line: issue.line, evidence: evidence(issue.text) });
        }
    }
    scanRules(lang, code, findings, patches);
    return { findings, patches: patches.sort((a, b) => a.line - b.line), symbols };
}

export function isBlocking(finding: Finding): boolean {
    return finding.severity === "critical" || finding.severity === "high";
}
