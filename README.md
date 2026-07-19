# LatentContext MCP Server

LatentContext is a local [Model Context Protocol](https://modelcontextprotocol.io/) server for keeping structured working notes during an assistant session. It stores memories, session summaries, a small knowledge graph, and optional local vector embeddings in SQLite.

## Install

Node.js 18 or later is required. Add the published server to an MCP host:

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
      "command": "latentcontext-mcp",
      "args": []
    }
  }
}
```

Restart the MCP host after changing its configuration. The server uses stdin/stdout for MCP JSON-RPC, so diagnostic output is written to its log file rather than the terminal.

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
      "args": [],
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

## Tools

| Tool | Use |
| --- | --- |
| `session_start` | Start a session; any active session is archived first. |
| `memory_store` | Record a note, decision, fact, or event for the active session. |
| `memory_retrieve` | Retrieve relevant session context within a token budget. |
| `memory_compress` | Compress working memory or summaries. |
| `memory_forget` | Deprecate, correct, or remove a memory. |
| `memory_status` | Report current storage and session statistics. |

Stored notes need at least 10 words; 25 or more are recommended. Session working memory is isolated by session ID, and prior working memory is archived when a session changes or the server shuts down.

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
npm test                  # Run the full test suite
npm run test:smoke        # Build and exercise the packaged server
npm run audit:production  # Audit production dependencies
npm run dev               # Run TypeScript source locally
```

## License

[MIT](LICENSE)
