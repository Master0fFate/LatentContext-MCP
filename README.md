# LatentContext MCP Server

An intelligent memory and context management middleware for LLMs via the Model Context Protocol. Reduces token consumption by up to 90% through structured knowledge graphs, semantic vector retrieval, tiered summaries, and relevance-scored context injection.

## Quick Start

```bash
# Install dependencies
npm install

# Build
npm run build

# Run
npm start
```

## MCP Host Configuration

### Claude Desktop / Gemini Code Assist / Cursor

Add to your MCP settings configuration:

```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "node",
      "args": ["c:/Users/fate/Desktop/HybridMCP/dist/index.js"]
    }
  }
}
```

### Development Mode

```bash
npm run dev
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store facts, preferences, events, summaries, or core memories |
| `memory_retrieve` | Retrieve ranked, deduplicated context within a token budget |
| `graph_query` | Query the knowledge graph for entity facts and relationships |
| `memory_compress` | Compress working memory, sessions, or epochs |
| `memory_forget` | Deprecate, correct, or delete memories |
| `memory_status` | Get storage statistics across all subsystems |

## Resources

| URI | Description |
|-----|-------------|
| `memory://core` | Tier 3 core memories (always available) |
| `memory://session/current` | Current session working memory |
| `memory://graph/schema` | Entity and relation types in the graph |
| `memory://stats` | Memory usage statistics |

## Prompts

| Prompt | Description |
|--------|-------------|
| `extract_facts` | Extract structured triples from text |
| `compress_session` | Compress working memory into session summary |
| `consolidate_epoch` | Merge session summaries into epoch summary |

## Memory Architecture

```
Tier 3 (Core):     ~200 tokens  — permanent, never evicted
Tier 2 (Epoch):    ~300 tokens  — weekly/thematic summaries
Tier 1 (Session):  ~500 tokens  — per-session summaries
Tier 0 (Working): ~2000 tokens  — current session buffer
Vector Index:     semantic search over all stored content
Knowledge Graph:  structured entity-relation triples
```

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
    "defaultRetrieveBudget": 3000
  }
}
```

Set `"provider": "none"` to disable embeddings (vector search will return empty results).

## Data Storage

All data is stored locally in `./data/memory.db` (SQLite via WASM). No data leaves your machine when using local embeddings.

>>>
<details>
<summary>For LLMs</summary>

**LLM Installation Prompt for LatentContext-MCP Server:**

```
To install and run this LatentContext-MCP server (Model Context Protocol for latent context handling):

1. Clone: `git clone https://github.com/Master0fFate/LatentContext-MCP.git`
2. Enter dir: `cd LatentContext-MCP`
3. Install deps: `npm install` (or `pip install -r requirements.txt` if Python)
4. Build: `npm run build` (if applicable)
5. Run server: `npm start` (or `python server.py`)

Configure your MCP host (e.g., Claude Desktop settings.json) with:
```json
{
  "mcpServers": {
    "latentcontext": {
      "command": "npm",
      "args": ["start"],
      "cwd": "/path/to/LatentContext-MCP"
    }
  }
}
```

Use: Invoke "use latentcontext-mcp" in prompts for repo context. No further analysis needed.
```

</details>
