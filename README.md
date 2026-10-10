# LatentContext MCP Server

[![CI](https://github.com/Master0fFate/LatentContext-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Master0fFate/LatentContext-MCP/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/latentcontext-mcp)](https://www.npmjs.com/package/latentcontext-mcp)

LatentContext is a local [Model Context Protocol](https://modelcontextprotocol.io/) server for keeping structured working notes during an assistant session. It stores memories, session summaries, a small knowledge graph, and optional local vector embeddings in SQLite, and exposes them through six tools, four resources, and three prompts over stdio.

It works with any MCP host that supports stdio servers, including Claude Code, Claude Desktop, Cursor, VS Code, Windsurf, Codex CLI, and Gemini CLI. Tool schemas use only widely supported JSON Schema keywords, every tool carries `title` and `annotations`, structured results are published with `outputSchema`, and the server sends `instructions` and MCP log notifications so each host can make the most of what it supports.

## Requirements

- Node.js 20 or later.
- No native build step. SQLite runs through `sql.js` (WebAssembly) and embeddings run through ONNX Runtime, both installed from npm.

## Install

The published package runs directly with `npx`:

```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "npx",
      "args": ["-y", "latentcontext-mcp@latest"]
    }
  }
}
```

Or install it globally and use the executable:

```bash
npm install -g latentcontext-mcp
```

```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "latentcontext-mcp"
    }
  }
}
```

Restart the host after changing its configuration. The server uses stdin/stdout for MCP JSON-RPC, so diagnostics go to a log file (and, once connected, to MCP log notifications) rather than the terminal.

### Host-specific setup

<details>
<summary>Claude Code</summary>

```bash
claude mcp add latentcontext -- npx -y latentcontext-mcp@latest
```

Add `--scope project` to share the configuration through `.mcp.json`, or `--scope user` to make it available in every project.
</details>

<details>
<summary>Claude Desktop</summary>

Add the `mcpServers` block above to `claude_desktop_config.json` (Settings → Developer → Edit Config).
</details>

<details>
<summary>Cursor</summary>

Add the `mcpServers` block above to `.cursor/mcp.json` in the project, or to `~/.cursor/mcp.json` for all projects.
</details>

<details>
<summary>VS Code (GitHub Copilot)</summary>

Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "latentcontext": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "latentcontext-mcp@latest"]
    }
  }
}
```
</details>

<details>
<summary>Windsurf</summary>

Add the `mcpServers` block above to `~/.codeium/windsurf/mcp_config.json`.
</details>

<details>
<summary>Codex CLI</summary>

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.latentcontext]
command = "npx"
args = ["-y", "latentcontext-mcp@latest"]
```
</details>

<details>
<summary>Gemini CLI</summary>

Add the `mcpServers` block above to `~/.gemini/settings.json` (or `.gemini/settings.json` in the project).
</details>

## Storage

By default, runtime state is project-local:

```
<launched-project>/.latentcontext/
├── memory.db
└── server.log
```

`<launched-project>` is the MCP server process's current working directory. Configure the host to launch the server from the project root when project isolation is wanted. The repository ignores only its root `.latentcontext/` directory, so this local state is not committed.

### Share storage deliberately

Storage is only shared when an explicit location is configured. Set `LATENTCONTEXT_DATA_DIR` in the MCP host environment to use one directory across projects:

```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "latentcontext-mcp",
      "env": {
        "LATENTCONTEXT_DATA_DIR": "C:/shared/latentcontext"
      }
    }
  }
}
```

`LATENTCONTEXT_DATA_DIR` takes precedence over all configured storage locations. Alternatively, point `LATENTCONTEXT_CONFIG` at a configuration file:

```json
{
  "storage": {
    "dataDir": "./shared-state",
    "sqliteFile": "memory.db"
  },
  "embedding": {
    "provider": "local"
  }
}
```

A relative `storage.dataDir` is resolved relative to that configuration file. The server also reads `latentcontext.config.json` from the default data directory or beside the installed package. A `latentcontext.config.json` in the launched project is ignored unless `LATENTCONTEXT_ALLOW_PROJECT_CONFIG=1` is set. Use `"provider": "none"` to disable embeddings.

### Environment variables

| Variable | Effect |
| --- | --- |
| `LATENTCONTEXT_DATA_DIR` | Directory for `memory.db` and `server.log`. Overrides every other storage setting. |
| `LATENTCONTEXT_CONFIG` | Path to a `latentcontext.config.json` to load. |
| `LATENTCONTEXT_ALLOW_PROJECT_CONFIG` | Set to `1` to also read `latentcontext.config.json` from the launched project. |
| `LATENTCONTEXT_HIDE_CONSOLE` | Windows only. Set to `0` to keep the console window visible. |

## Tools

| Tool | Use | Annotations |
| --- | --- | --- |
| `session_start` | Start a session; any active session is archived first. | |
| `memory_store` | Record a note, decision, fact, or event for the active session. | |
| `memory_retrieve` | Retrieve relevant session context within a token budget. | read-only |
| `memory_compress` | Compress working memory or summaries (lossy). | destructive |
| `memory_forget` | Deprecate, correct, or remove a memory. | destructive, idempotent |
| `memory_status` | Report current storage and session statistics. | read-only |

Stored notes need at least 10 words; 25 or more are recommended. Session working memory is isolated by session ID, and prior working memory is archived when a session changes or the server shuts down.

`session_start`, `memory_store`, `memory_retrieve`, and `memory_status` return `structuredContent` alongside their text, validated against a published `outputSchema`.

### Resources

| URI | Content |
| --- | --- |
| `memory://core` | Persistent Tier 3 core memories. |
| `memory://session/current` | Working memory for the active session. |
| `memory://graph/schema` | Entity types and relation predicates (JSON). |
| `memory://stats` | Storage statistics (JSON). |

### Prompts

`extract_facts`, `compress_session`, and `consolidate_epoch` return prompt templates for turning text into fact triples, summarizing working memory, and merging session summaries.

## Logging

Diagnostics are written to `server.log` in the data directory. After a client completes the MCP handshake, records at `info` and above are also sent as `notifications/message` with logger `latentcontext`; clients can raise or lower the threshold with `logging/setLevel`. Nothing is ever written to stdout except protocol frames, and nothing is written to stderr.

## Run from source

```bash
git clone https://github.com/Master0fFate/LatentContext-MCP.git
cd LatentContext-MCP
npm install
npm run build
npm start
```

For a local MCP configuration, use the built entry point:

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

## Development

```bash
npm run build             # Compile TypeScript and prepare the executable
npm run typecheck         # Type-check sources, tests, and benchmarks
npm test                  # Run the full test suite
npm run test:smoke        # Build, pack, install, and exercise the packaged server
npm run check             # typecheck + test
npm run audit:production  # Audit production dependencies
npm run benchmark:retrieval
npm run dev               # Run TypeScript source locally
```

The test suite drives the server over real stdio with the MCP SDK client and checks that tool schemas only use portable JSON Schema keywords, that annotations and output schemas are published, and that stdout and stderr stay protocol-clean. CI runs it on Linux, macOS, and Windows with Node.js 20, 22, and 24.

If `npm install` fails while `onnxruntime-node` downloads optional CUDA binaries (Linux x64 only), set `ONNXRUNTIME_NODE_INSTALL_CUDA=skip`; the server does not use them.

## License

[MIT](LICENSE)
