# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [3.0.0] - 2026-10-15

LatentContext 3.0 brings session memory, code intelligence, and code review
together in one server. It collects the 2.1 and 2.2 work into a single major
release. Tool names and schemas from 2.x are unchanged.

### Highlights

- `code_explore`: a tree-sitter code index for 19 languages with call
  paths, references, callees, and impact in one call.
- `review_start`, `review_chunk`, `review_status`, and `review_end`:
  plan-driven code review with syntax and security checks, caller impact,
  verdicts, and patch hints.
- Project state in `.latentcontext/` ignores itself in git.

### Changed

- On shutdown, the server releases its file watchers and worker threads
  before it archives the session, so hosts can delete or move the project
  folder right after closing the connection.
- The README opens with a table of all eleven tools.

### Fixed

- The Windows CI job no longer fails when test cleanup deletes a folder the
  server is still releasing.

## [2.2.0] - 2026-10-15

### Added

- Code review tools ported from codeviewer-mcp and merged from eight tools
  into four: `review_start`, `review_chunk`, `review_status`, and
  `review_end`.
- `review_chunk` merges the model's own findings with tree-sitter syntax
  errors for all 19 indexed languages, language-aware security rules, and
  leftover debug statements. It returns a verdict, drop-in patch hints, and
  caller references and impact from the code index, and it tracks each plan
  step's status.
- Four review lenses ship in `prompts/`. `review_start` returns the chosen
  lens text, and `LATENTCONTEXT_REVIEW_PROFILES` points at a custom set.
- Reviews persist in the existing SQLite database and expire 7 days after
  their last update.

## [2.1.1] - 2026-10-15

### Fixed

- The data directory now gets its own `.gitignore` containing `*`, so a
  project-local `.latentcontext/` never shows up in `git status`. An existing
  `.gitignore` in that directory is left untouched.

## [2.1.0] - 2026-10-15

### Added

- `code_explore` tool. It indexes the project with tree-sitter and answers
  symbol names, file names, or plain questions with line-numbered source,
  call paths, references, callees, and an impact count in one call.
- Support for 19 languages: TypeScript, TSX, JavaScript, Python, Rust, Go,
  Java, C#, C, C++, Ruby, PHP, Kotlin, Scala, Lua, Zig, Elixir, Bash, and
  PowerShell.
- Scanning and parsing run on a worker-thread pool. Nested `.gitignore` files
  are honoured, and common dependency and cache directories are skipped.
- The index stays current through native file watchers and re-parses only
  changed files. It persists as a v8 snapshot under `code-index/` in the data
  directory.
- Project roots come from the `root` argument, `LATENTCONTEXT_CODE_ROOT`, the
  client's MCP roots, or a project-like working directory.
- `LATENTCONTEXT_CODE_INDEX=0` disables the code index.

### Changed

- The README was rewritten around memory and code intelligence.

## [2.0.0] - 2026-10-10

### Breaking changes

- Node.js 20 or later is required. Node.js 18 reached end of life and is no
  longer supported.
- The `memory_forget` input schema no longer uses `allOf`/`if`/`then`/`else`
  to require `correction` for the `correct` action, and the `memory_retrieve`
  date filters no longer declare `format`/`pattern`. These rules are still
  enforced, but by the server with explicit error messages, because several
  MCP hosts reject those JSON Schema keywords when translating tool schemas
  for their model provider.
- Validation errors for unknown arguments and out-of-range values now use the
  wording produced by the schema validator (for example
  `Unrecognized key: "x"`). Errors for the server's own rules
  (word count, ISO datetimes, UUIDs, `correction` handling) are unchanged.
- The server advertises `title: "LatentContext"` in its `serverInfo`.

### Added

- Tool `title` and `annotations` (`readOnlyHint`, `destructiveHint`,
  `idempotentHint`, `openWorldHint`) so hosts can label tools and decide which
  calls need confirmation.
- `outputSchema` and `structuredContent` for `session_start`, `memory_store`,
  `memory_retrieve`, and `memory_status`.
- Server `instructions` describing how the tools fit together, delivered to
  hosts during initialization.
- The `logging` capability. Lifecycle events and tool failures are forwarded
  as `notifications/message` after the handshake, in addition to the log
  file. Clients can tune verbosity with `logging/setLevel`.
- `LATENTCONTEXT_HIDE_CONSOLE=0` to keep the console window visible on
  Windows.
- GitHub Actions CI covering Linux, macOS, and Windows on Node.js 20, 22,
  and 24.
- `npm run typecheck` (also covers tests and benchmarks) and `npm run check`.

### Changed

- Migrated from the low-level `Server` request handlers to `McpServer` with
  `registerTool`, `registerResource`, and `registerPrompt`; tool inputs are
  declared with zod and validated by the SDK before handlers run.
- Upgraded `@modelcontextprotocol/sdk` to 1.32, `zod` to 4, and
  `@huggingface/transformers` to 4.
- IDs are generated with `node:crypto` `randomUUID()`; the `uuid` dependency
  was removed.
- `sql.js` types come from `@types/sql.js` instead of a local declaration
  file.
- TypeScript 7 with `NodeNext` module resolution, `verbatimModuleSyntax`,
  and `erasableSyntaxOnly`.
- Diagnostics moved into `src/logger.ts`; the log file uses MCP log level
  names.

### Fixed

- The package smoke test no longer depends on registry metadata being cached,
  so it passes after a clean `npm ci`.
- `@huggingface/transformers` log verbosity is set with the numeric level the
  library expects, rather than a string it ignored.

## [1.2.0]

- Scope runtime state to the launched project.

## [1.1.1]

- npm package distribution.

## [1.1.0]

- Storage isolation and stdio hardening.

[3.0.0]: https://github.com/Master0fFate/LatentContext-MCP/compare/v2.2.0...v3.0.0
[2.2.0]: https://github.com/Master0fFate/LatentContext-MCP/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/Master0fFate/LatentContext-MCP/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/Master0fFate/LatentContext-MCP/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/Master0fFate/LatentContext-MCP/compare/v1.2.0...v2.0.0
