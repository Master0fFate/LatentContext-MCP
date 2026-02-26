# LatentContext MCP Server

A session-scoped memory layer for LLMs, built on the [Model Context Protocol](https://modelcontextprotocol.io/). LatentContext gives AI assistants the ability to explicitly store and retrieve notes, decisions, and context within a single conversation — keeping the AI focused on the current task without cross-session data contamination.

## Why LatentContext?

LLM context windows are large but finite. In long conversations the assistant loses track of earlier decisions, forgets what files it already edited, and repeats work. LatentContext solves this by providing a structured, **strictly session-isolated** scratchpad:

- **Session Isolation** — every conversation starts from a completely clean slate. Data from a previous project or conversation can never leak into the current one.
- **Explicit Memory** — the assistant actively stores detailed notes about what it did, why, and how. This replaces vague "I think we did X" with retrievable, concrete records.
- **Quality Enforcement** — the server rejects overly terse entries (fewer than 10 words) and warns about brief ones (under 25 words), forcing the assistant to write useful context rather than throwaway one-liners.
- **Token-Budgeted Retrieval** — when retrieving context, results are ranked, deduplicated, and trimmed to fit within a configurable token budget so the assistant's context window isn't flooded.
- **Compression** — when working memory grows large during a long session, it can be compressed into a summary, preserving the important details while freeing up space.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Master0fFate/LatentContext-MCP.git
cd LatentContext-MCP
npm install

# Build
npm run build

# Run (usually launched automatically by the MCP host)
npm start
```

**Requirements:** Node.js 18+. No API keys needed — everything runs locally.

## MCP Host Configuration

Add to your MCP host settings (Claude Desktop, Cursor, Antigravity, etc.):

```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "node",
      "args": ["/absolute/path/to/LatentContext-MCP/dist/index.js"]
    }
  }
}
```

Restart the MCP host after saving. The server communicates over stdin/stdout using the MCP JSON-RPC protocol.

## Tools

| Tool | Description |
|------|-------------|
| `session_start` | Start a new session. Clears all in-memory data and generates a unique timestamp-prefixed session ID. **Must be called first.** |
| `memory_store` | Store a detailed note, fact, preference, or event into the current session's working memory. Entries must be 10+ words; 25+ recommended. |
| `memory_retrieve` | Retrieve session-scoped context for a given query. Returns working memory and compressed session notes, ranked and deduplicated within a token budget. |
| `memory_compress` | Compress working memory (`working`), merge session summaries (`session`), or consolidate into long-term knowledge (`epoch`). Lossy but preserves key details. |
| `memory_forget` | Deprecate (lower confidence), correct (replace content), or permanently delete a stored memory by its ID. |
| `memory_status` | Show storage statistics: entry counts and token estimates per tier, knowledge graph size, vector store count, and the current session ID. |

## How It Works

### Session Lifecycle

```
1. session_start        → Fresh session with unique ID, zero entries
2. memory_store (×N)    → AI saves detailed notes as it works
3. memory_retrieve      → AI recalls what it stored earlier
4. memory_compress      → (optional) Compress when memory grows large
```

### Session Isolation Model

Each `session_start` call:
1. Generates a **timestamp-prefixed UUID** (e.g., `1740567150290-ec459108-2bf4-...`) for guaranteed uniqueness.
2. Clears the entire in-memory working buffer.
3. Returns a clean slate — `memory_retrieve` will return nothing until `memory_store` is called.

**What's included in retrieval:**
- Tier 0 (Working Memory) — entries stored via `memory_store` during this session.
- Tier 1 (Session Notes) — compressed summaries created by `memory_compress` during this session.

**What's excluded from retrieval:**
- Data from any other session (past or concurrent).
- Global knowledge graph entities and relations.
- Global vector store search results.
- Core/epoch tier data from previous sessions.

This guarantees that a banking dashboard project won't pull in data from an audio visualizer project, even if they share the same database file.

### Content Quality Enforcement

The server enforces minimum quality on stored memories:

| Entry Length | Behavior |
|---|---|
| < 10 words | **Rejected** with an error and rewrite instructions |
| 10–24 words | **Accepted** with a warning suggesting more detail |
| 25+ words | **Accepted** — this is the target quality level |

This prevents the common failure mode where the AI stores useless entries like `"Fixed the bug"` instead of `"Fixed a null pointer in the UserService.getProfile() method caused by missing null check on the database result. Added Optional<User> return type and a fallback to guest profile when the user record doesn't exist."`.

### Memory Tiers

```
Tier 0 (Working):   In-memory buffer of entries from the current session.
                     Auto-compresses when token count exceeds threshold (~2500 tokens).

Tier 1 (Session):   Compressed summaries of overflowed working memory.
                     Created automatically or via memory_compress scope='working'.
```

## Architecture

```
src/
├── index.ts              # Entry point, stdio transport, console suppression, graceful shutdown
├── server.ts             # MCP tool/resource/prompt definitions and handlers
├── config.ts             # Configuration loader with deep-merge defaults
├── database.ts           # SQLite (via sql.js WASM) schema, CRUD operations
├── session.ts            # Session lifecycle (start, end, ID generation)
├── memory-manager.ts     # Store, compress, forget, and status operations
├── context-assembler.ts  # Retrieval algorithm: ranking, dedup, budget-filling
├── knowledge-graph.ts    # Entity-relation triple store (internal, not exposed as a tool)
├── vector-store.ts       # Embedding-based semantic search (internal)
├── embeddings.ts         # Local embeddings via @huggingface/transformers (ONNX)
├── token-counter.ts      # Token counting and truncation (js-tiktoken)
└── sql.js.d.ts           # Type declarations for sql.js
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `@modelcontextprotocol/sdk` | MCP server framework (stdio transport, JSON-RPC) |
| `sql.js` | SQLite compiled to WASM — no native bindings needed |
| `@huggingface/transformers` | Local sentence embeddings (Xenova/all-MiniLM-L6-v2, 384 dimensions) |
| `js-tiktoken` | Accurate token counting for OpenAI-compatible tokenizers |
| `uuid` | Session and memory ID generation |
| `zod` | Schema validation |

### Platform Notes (Windows)

- The server automatically hides its console window ~1-2 seconds after startup using the Win32 API (`GetConsoleWindow` + `ShowWindow`). Do not manually close the Node.js window — it is the running server process.
- All `console.*` output and `process.stderr.write` are redirected to `data/server.log` to prevent stdio corruption of the MCP JSON-RPC protocol.
- The sql.js WASM binary is explicitly resolved via `createRequire` to handle Windows path resolution edge cases.

## Configuration

Create `latentcontext.config.json` in the project root to override defaults:

```json
{
  "storage": {
    "dataDir": "./data",
    "sqliteFile": "memory.db"
  },
  "embedding": {
    "provider": "local",
    "model": "Xenova/all-MiniLM-L6-v2",
    "dimensions": 384
  },
  "tokenBudgets": {
    "defaultRetrieveBudget": 3000,
    "tier0Working": 2000,
    "tier1Session": 500
  },
  "compression": {
    "tier0OverflowThreshold": 2500
  },
  "session": {
    "autoStartOnBoot": true
  }
}
```

Set `"provider": "none"` to disable embeddings entirely (vector indexing will be skipped).

## Data Storage

All data is stored locally in `./data/memory.db` (SQLite via WASM). No data leaves your machine. The database contains tables for entities, relations, summaries (tiered), vectors (embeddings), access logs, and sessions.

## Resources

| URI | Description |
|-----|-------------|
| `memory://core` | Core memories (Tier 3) |
| `memory://session/current` | Current session working memory |
| `memory://graph/schema` | Entity and relation types in the knowledge graph |
| `memory://stats` | Memory usage statistics |

## Prompts

| Prompt | Description |
|--------|-------------|
| `extract_facts` | Extract structured entity-relation triples from free text |
| `compress_session` | Compress working memory into a session summary |
| `consolidate_epoch` | Merge session summaries into an epoch-level summary |

## Development

```bash
# Build
npm run build

# Dev mode (tsx, auto-recompile)
npm run dev

# Run diagnostic tests
node test-mcp.mjs
```

## License

MIT
