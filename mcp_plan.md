

<think>

The user wants me to:

1. Research the feasibility of their Model Context Protocol (MCP) idea - essentially a "reality filter" that implements more efficient context/memory management for LLMs
2. Provide an enhanced, more in-depth `plan-of-concept.md` that they can feed into an LLM to create an MCP Server

Let me analyze what they've described and think about how this maps to the MCP (Model Context Protocol) framework.

**Understanding MCP:**
The Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. It follows a client-server architecture where:


- MCP Hosts are applications like Claude Desktop, IDEs, or AI tools
- MCP Clients maintain 1:1 connections with servers
- MCP Servers provide context, tools, and prompts to the LLM

MCP servers expose three main primitives:
1. **Tools** - Functions the LLM can call
2. **Resources** - Data sources the LLM can read
3. **Prompts** - Templated interactions

**What the user wants:**
They want to build an MCP Server that acts as a "reality filter" - a middleware layer implementing memory efficiency concepts that compresses context, manages token usage, and optimizes what information gets passed to the LLM.

**Feasibility Analysis:**

Within MCP, I can implement knowledge graphs, vector databases, context compression, and smart retrieval systems. However, MCP operates at the application layer, so I can't modify the transformer's internal mechanisms like attention, inject soft prompts into latent states, or directly manipulate the KV cache. The server would need to work within these constraints—handling context optimization at the application level rather than the model architecture level.

Now I'm thinking through how to design an MCP server that maximizes efficiency within these boundaries: storing information in knowledge graphs and vector databases, retrieving only relevant context to reduce token overhead, compressing information before sending it to the LLM, and managing different memory tiers for short-term, long-term, and episodic recall. I should create a detailed plan that's realistic about what MCP can achieve while still providing meaningful memory improvements.

</think>

# Plan of Concept: LatentContext MCP Server

## A Context Efficiency Layer for Large Language Models via the Model Context Protocol

---

## 1. Executive Summary

This document specifies an MCP Server — codenamed **LatentContext** — that acts as an intelligent memory and context management middleware between an LLM host application and its underlying data. The server implements a multi-tiered memory architecture that dramatically reduces token consumption in the LLM's context window by replacing naive text accumulation with structured knowledge graphs, semantic vector retrieval, progressive summarization, and relevance-scored context injection.

### The Core Insight

MCP operates at the **application layer**, not the model architecture layer. This means we cannot inject soft prompts, modify attention mechanisms, or replace the transformer with an SSM. However, we **can** control *what text enters the context window, in what form, and at what priority*. This is where the vast majority of practical efficiency gains are achievable today, because most LLM applications waste 60–80% of their context window on redundant, low-relevance, or poorly structured text.

LatentContext implements the following principle:

> **The most efficient token is the one you never send.**

Rather than sending the LLM a full conversation history or raw documents, LatentContext maintains a structured backend of knowledge graphs, vector indices, and tiered summaries. It exposes MCP tools and resources that allow the LLM to **query** its memory selectively, receiving only the minimal, high-density context needed for the current reasoning step.

---

## 2. Feasibility Analysis

### 2.1 What MCP Can Do

The Model Context Protocol defines three primitives that a server can expose to an LLM host:

**Tools** are functions the LLM can invoke. These are the primary mechanism for LatentContext. The LLM calls tools to store memories, retrieve relevant context, query the knowledge graph, and manage memory tiers. Each tool call returns compressed, high-relevance text rather than raw history.

**Resources** are data sources the LLM (or host application) can read. These serve as passive context sources — the current state of the knowledge graph, session summaries, user profile data. Resources can be static or dynamic (subscribable).

**Prompts** are reusable, templated interaction patterns. These encode best-practice patterns for memory operations — how to summarize a session, how to extract facts into graph triples, how to formulate a retrieval query.

### 2.2 What MCP Cannot Do

The following operations from the original research summary are **outside MCP's scope** because they require model-internal access:

- Injecting continuous latent vectors or soft prompts into the embedding layer
- Modifying the KV cache directly (eviction, quantization)
- Replacing the attention mechanism with a state-space model
- Feeding hidden states from previous inference passes
- Altering the tokenizer or embedding matrix

### 2.3 Feasibility Verdict

The architecture is **highly feasible** within MCP's constraints. The efficiency gains come not from changing how the model processes tokens, but from ensuring only the right tokens reach the model in the first place. This is analogous to the difference between optimizing a database query engine versus optimizing the queries themselves — the latter often yields larger practical gains.

**Quantified estimate:** For a typical multi-turn conversational agent with access to documents, LatentContext can reduce effective context window usage by 70–90% compared to naive full-history injection, while preserving or improving response quality through higher signal-to-noise ratio in the injected context.

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                   LLM Host App                    │
│            (Claude Desktop, IDE, Agent)           │
│                                                   │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ System Prompt│  │ User Msg │  │ Tool Results│ │
│  └──────┬──────┘  └────┬─────┘  └──────┬──────┘ │
│         │              │               │         │
│         └──────────────┼───────────────┘         │
│                        │                          │
│              ┌─────────▼──────────┐               │
│              │   MCP Client       │               │
│              └─────────┬──────────┘               │
└────────────────────────┼──────────────────────────┘
                         │ MCP Protocol (JSON-RPC over stdio/SSE)
┌────────────────────────┼──────────────────────────┐
│              ┌─────────▼──────────┐               │
│              │  LatentContext MCP  │               │
│              │      Server        │               │
│              └─────────┬──────────┘               │
│                        │                          │
│  ┌─────────────────────┼─────────────────────┐   │
│  │                     │                     │   │
│  ▼                     ▼                     ▼   │
│ ┌──────────┐  ┌───────────────┐  ┌────────────┐ │
│ │ Knowledge │  │ Vector Index  │  │  Tiered    │ │
│ │  Graph    │  │ (Embeddings)  │  │  Summary   │ │
│ │  Store    │  │               │  │  Store     │ │
│ └──────────┘  └───────────────┘  └────────────┘ │
│                                                   │
│  ┌──────────────────────────────────────────────┐│
│  │           Context Assembler / Ranker          ││
│  │  (Relevance scoring, dedup, budget allocation)││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### 3.2.1 Knowledge Graph Store

**Purpose:** Store discrete facts extracted from conversations and documents as structured triples, enabling O(1) or O(log n) retrieval of specific facts without scanning text.

**Implementation:** Use an in-process graph database. For a lightweight MCP server, SQLite with a triple-store schema is the most practical choice. For production scale, Neo4j or a dedicated triple store like Apache Jena could be used behind an adapter interface.

**Schema:**

```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    entity_type TEXT,
    properties JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    confidence REAL DEFAULT 1.0,
    source_summary_id TEXT
);

CREATE TABLE relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id TEXT REFERENCES entities(id),
    properties JSON,
    temporal_start TIMESTAMP,
    temporal_end TIMESTAMP,
    confidence REAL DEFAULT 1.0,
    source_summary_id TEXT
);

CREATE INDEX idx_relations_subject ON relations(subject_id);
CREATE INDEX idx_relations_object ON relations(object_id);
CREATE INDEX idx_relations_predicate ON relations(predicate);
CREATE INDEX idx_entities_type ON entities(entity_type);
```

**Fact extraction** is performed by the LLM itself via a dedicated prompt template (exposed as an MCP prompt). When the LLM processes a conversation turn, the host application (or an orchestration layer) can call the `extract_facts` prompt which instructs the LLM to output structured JSON triples. These are then stored via the `store_facts` tool.

#### 3.2.2 Vector Index

**Purpose:** Enable semantic similarity search over all stored memories — summaries, facts, conversation fragments — so the LLM can retrieve contextually relevant information without exact keyword matching.

**Implementation:** Use a lightweight in-process vector search library. The recommended stack is `hnswlib` or `usearch` for the index, paired with a local embedding model (e.g., `all-MiniLM-L6-v2` via `sentence-transformers`, or `nomic-embed-text` via `ollama`). For minimal dependencies, the server can also call an external embedding API.

**Design Decisions:**

The vector index stores embeddings of three types of content: summary chunks (from the tiered summary store), serialized knowledge graph subgraphs (e.g., "John is located in Paris since 2020"), and raw conversation snippets that resist clean summarization (e.g., nuanced user preferences, emotional context). Each vector entry carries metadata including its source type, timestamp, relevance decay weight, and a back-reference to the originating record in either the graph store or summary store. This metadata enables filtered search — for example, "retrieve only facts from the last 24 hours" or "retrieve only user preferences."

#### 3.2.3 Tiered Summary Store

**Purpose:** Maintain progressively compressed summaries of conversation history at multiple granularity levels, implementing a "memory pyramid" where recent interactions are stored in detail and older interactions are stored as increasingly abstract summaries.

**Tier Structure:**

**Tier 0 — Working Memory (current session, last ~5 turns):** Raw conversation text. This is the only tier that stores uncompressed text, and it is bounded by a configurable token budget (default: 2,000 tokens). When the budget is exceeded, the oldest turns are summarized and promoted to Tier 1.

**Tier 1 — Session Summaries (per-session, ~200 tokens each):** Each completed session (or each overflow from Tier 0) is compressed into a concise summary. This summary preserves key decisions, emotional tone, unresolved questions, and action items. Summaries at this tier are generated by the LLM itself via the `compress_session` prompt template.

**Tier 2 — Epoch Summaries (weekly/thematic, ~100 tokens each):** Multiple Tier 1 summaries are periodically merged into higher-level summaries that capture themes, evolving user preferences, and long-term patterns. This tier answers the question: "What has been happening over the past week?"

**Tier 3 — Core Memory (persistent, ~50 tokens each):** The most durable memories — user identity, fundamental preferences, long-running goals, critical decisions. This tier is append-only (entries are only added or updated, never evicted) and is always injected into context.

```
Token Budget Allocation (configurable, example for 8K budget):
──────────────────────────────────────────────────
Tier 3 (Core):        ~200 tokens  (always present)
Tier 2 (Epochs):      ~300 tokens  (most relevant 2-3)
Tier 1 (Sessions):    ~500 tokens  (most relevant 3-5)
Tier 0 (Working):    ~2000 tokens  (current session)
Retrieved Context:   ~2000 tokens  (from vector search)
Graph Facts:          ~500 tokens  (from KG queries)
──────────────────────────────────────────────────
Remaining for LLM:   ~2500 tokens  (system prompt + reasoning)
```

#### 3.2.4 Context Assembler / Ranker

**Purpose:** The critical intelligence layer. Given a token budget and the current query, the Context Assembler selects, ranks, deduplicates, and formats the optimal context payload from all memory subsystems.

**Algorithm:**

```
FUNCTION assemble_context(query, token_budget):
    // 1. Always include Core Memory (Tier 3)
    context_parts = [tier3_core_memory]
    remaining_budget -= token_count(tier3_core_memory)

    // 2. Embed the query and retrieve from vector index
    query_embedding = embed(query)
    candidates = vector_search(query_embedding, top_k=20)

    // 3. Query knowledge graph for entities mentioned in query
    entities = extract_entity_mentions(query)
    graph_facts = knowledge_graph.query(entities, depth=2)
    graph_text = serialize_facts(graph_facts)

    // 4. Score all candidates by composite relevance
    FOR each candidate in candidates + graph_facts:
        candidate.score = (
            0.4 * semantic_similarity(candidate, query) +
            0.3 * recency_weight(candidate.timestamp) +
            0.2 * source_priority(candidate.tier) +
            0.1 * access_frequency(candidate.id)
        )

    // 5. Deduplicate (prefer higher-tier versions of same info)
    candidates = deduplicate(candidates, similarity_threshold=0.85)

    // 6. Greedily fill budget by score
    candidates.sort_by(score, descending)
    FOR each candidate in candidates:
        IF remaining_budget >= token_count(candidate):
            context_parts.append(candidate)
            remaining_budget -= token_count(candidate)

    // 7. Format and return
    RETURN format_context_block(context_parts)
```

---

## 4. MCP Interface Specification

### 4.1 Tools

The server exposes the following tools for the LLM to invoke:

---

**`memory_store`**

Stores a piece of information into the appropriate memory subsystem. The server automatically determines whether the input is best stored as a knowledge graph triple, a summary chunk, or a raw embedding.

```json
{
  "name": "memory_store",
  "description": "Store information in long-term memory. Accepts facts, observations, user preferences, or session summaries. The system automatically categorizes and indexes the information.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The information to store. Can be a natural language statement, a fact, or a summary."
      },
      "memory_type": {
        "type": "string",
        "enum": ["fact", "preference", "event", "summary", "core"],
        "description": "The category of memory. 'core' memories are never evicted."
      },
      "confidence": {
        "type": "number",
        "description": "Confidence level 0.0-1.0. Lower confidence memories are evicted first."
      },
      "entities": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Key entities this memory relates to, for graph indexing."
      }
    },
    "required": ["content", "memory_type"]
  }
}
```

---

**`memory_retrieve`**

The primary retrieval tool. Given a query (typically the user's current message or a reasoning question), returns the most relevant context from all memory subsystems, assembled within a token budget.

```json
{
  "name": "memory_retrieve",
  "description": "Retrieve relevant memories and context for the current query. Returns a ranked, deduplicated context block optimized to fit within the token budget.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The query to retrieve context for. Typically the user's current message."
      },
      "token_budget": {
        "type": "integer",
        "description": "Maximum tokens to return. Default 3000.",
        "default": 3000
      },
      "filters": {
        "type": "object",
        "properties": {
          "memory_types": { "type": "array", "items": { "type": "string" } },
          "after": { "type": "string", "format": "date-time" },
          "before": { "type": "string", "format": "date-time" },
          "min_confidence": { "type": "number" }
        }
      }
    },
    "required": ["query"]
  }
}
```

---

**`graph_query`**

Direct access to the knowledge graph for structured fact retrieval.

```json
{
  "name": "graph_query",
  "description": "Query the knowledge graph for facts about specific entities or relationships. Returns structured triples.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity": {
        "type": "string",
        "description": "The entity to query about."
      },
      "relation": {
        "type": "string",
        "description": "Optional: specific relation type to filter by (e.g., 'located_in', 'works_at')."
      },
      "depth": {
        "type": "integer",
        "description": "How many hops from the entity to traverse. Default 1.",
        "default": 1
      }
    },
    "required": ["entity"]
  }
}
```

---

**`memory_compress`**

Triggers manual compression of the working memory (Tier 0) into a Tier 1 summary. Normally this happens automatically, but the LLM can invoke this proactively when it detects the conversation has shifted topics.

```json
{
  "name": "memory_compress",
  "description": "Compress the current working memory into a summary and promote to long-term storage. Use when the conversation has shifted to a new topic or the working memory is getting large.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "scope": {
        "type": "string",
        "enum": ["working", "session", "epoch"],
        "description": "What level of memory to compress."
      }
    },
    "required": ["scope"]
  }
}
```

---

**`memory_forget`**

Allows the LLM to mark memories as outdated or incorrect, reducing their confidence score or removing them entirely. Critical for maintaining accuracy over time.

```json
{
  "name": "memory_forget",
  "description": "Mark a memory as outdated, incorrect, or no longer relevant. Can reduce confidence or delete entirely.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "The ID of the memory to modify."
      },
      "action": {
        "type": "string",
        "enum": ["deprecate", "correct", "delete"],
        "description": "'deprecate' lowers confidence, 'correct' replaces content, 'delete' removes entirely."
      },
      "correction": {
        "type": "string",
        "description": "If action is 'correct', the new content to store."
      }
    },
    "required": ["memory_id", "action"]
  }
}
```

---

**`memory_status`**

Returns diagnostics about the current state of all memory subsystems — useful for the LLM to reason about what it knows and what might be missing.

```json
{
  "name": "memory_status",
  "description": "Get the current status of all memory subsystems including storage usage, entity counts, and tier distributions.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 4.2 Resources

**`memory://core`** — The current Tier 3 core memory block. Always available. The host application can inject this into the system prompt automatically.

**`memory://session/current`** — The current session's Tier 0 working memory.

**`memory://graph/schema`** — The current entity types and relation types in the knowledge graph, so the LLM can formulate effective graph queries.

**`memory://stats`** — Memory usage statistics (subscribable, updates on every write).

### 4.3 Prompts

**`extract_facts`** — A prompt template that instructs the LLM to extract structured facts from a conversation turn and output them as JSON triples suitable for `memory_store`.

**`compress_session`** — A prompt template for generating Tier 1 session summaries from Tier 0 working memory, with instructions to preserve key decisions, emotional context, unresolved items, and action items while aggressively compressing redundant dialogue.

**`consolidate_epoch`** — A prompt template for merging multiple Tier 1 summaries into a Tier 2 epoch summary.

---

## 5. Implementation Plan

### 5.1 Technology Stack

**Runtime:** Node.js (TypeScript) or Python. TypeScript is recommended for MCP server development due to mature SDK support (`@modelcontextprotocol/sdk`).

**Storage Backend (Knowledge Graph):** `better-sqlite3` with the triple-store schema defined above. SQLite is chosen for zero-configuration deployment, single-file portability, and sufficient performance for personal/single-user agent workloads (tens of thousands of facts).

**Vector Index:** `vectordb` (LanceDB) for embedded vector search with no external dependencies, or `hnswlib-node` for raw HNSW index. LanceDB is preferred because it persists to disk and supports metadata filtering natively.

**Embedding Model:** Two strategies supported via configuration. **Local** uses `@xenova/transformers` (ONNX runtime in Node.js) with `all-MiniLM-L6-v2` for zero-network-dependency operation. **Remote** calls an embedding API (OpenAI, Voyage, or Anthropic's future embedding endpoint) for higher quality embeddings at the cost of network latency.

**Token Counting:** `tiktoken` (or `js-tiktoken`) for accurate token budget management. The server must count tokens precisely to maximize budget utilization.

### 5.2 Project Structure

```
latentcontext-mcp/
├── src/
│   ├── index.ts                 # MCP server entry point
│   ├── server.ts                # Server setup, tool/resource/prompt registration
│   ├── memory/
│   │   ├── manager.ts           # Orchestrates all memory subsystems
│   │   ├── knowledge-graph.ts   # SQLite triple store operations
│   │   ├── vector-index.ts      # Embedding + vector search
│   │   ├── summary-tiers.ts     # Tiered summary management
│   │   └── compressor.ts        # Compression/summarization logic
│   ├── context/
│   │   ├── assembler.ts         # Context assembly + ranking algorithm
│   │   ├── ranker.ts            # Relevance scoring functions
│   │   └── deduplicator.ts      # Cross-source deduplication
│   ├── tools/
│   │   ├── memory-store.ts      # memory_store tool handler
│   │   ├── memory-retrieve.ts   # memory_retrieve tool handler
│   │   ├── graph-query.ts       # graph_query tool handler
│   │   ├── memory-compress.ts   # memory_compress tool handler
│   │   ├── memory-forget.ts     # memory_forget tool handler
│   │   └── memory-status.ts     # memory_status tool handler
│   ├── resources/
│   │   └── memory-resources.ts  # Resource endpoint handlers
│   ├── prompts/
│   │   ├── extract-facts.ts     # Fact extraction prompt template
│   │   ├── compress-session.ts  # Session compression template
│   │   └── consolidate-epoch.ts # Epoch consolidation template
│   ├── embeddings/
│   │   ├── local.ts             # Local ONNX embedding
│   │   └── remote.ts            # API-based embedding
│   └── utils/
│       ├── token-counter.ts     # Token counting utilities
│       └── config.ts            # Configuration management
├── data/                        # Runtime data directory
│   ├── memory.db                # SQLite database
│   └── vectors/                 # LanceDB vector storage
├── package.json
├── tsconfig.json
└── README.md
```

### 5.3 Development Phases

**Phase 1 — Foundation (MVP):** Implement the MCP server skeleton with `memory_store` (text only, stored in SQLite + vector index) and `memory_retrieve` (vector similarity search, basic ranking). No knowledge graph, no tiered summaries. This phase validates the MCP integration works end-to-end.

**Phase 2 — Knowledge Graph:** Add the triple store schema, the `graph_query` tool, and the `extract_facts` prompt template. Implement automatic fact extraction on `memory_store` calls when `memory_type` is "fact."

**Phase 3 — Tiered Summaries:** Implement the four-tier summary system with automatic promotion and compression. Add the `memory_compress` tool and the `compress_session` prompt template. Implement token budget management in the context assembler.

**Phase 4 — Intelligent Ranking:** Implement the full context assembly algorithm with composite relevance scoring, recency decay, deduplication, and budget-aware selection. This is the phase where the system becomes significantly better than naive retrieval.

**Phase 5 — Lifecycle Management:** Add `memory_forget`, confidence decay over time, conflict resolution (when new facts contradict old ones), and the `consolidate_epoch` prompt for Tier 2 compression. Add `memory_status` for introspection.

---

## 6. Configuration

The server should be configurable via a JSON configuration file or environment variables:

```json
{
  "storage": {
    "dataDir": "./data",
    "sqliteFile": "memory.db",
    "vectorDir": "vectors"
  },
  "embedding": {
    "provider": "local",
    "model": "all-MiniLM-L6-v2",
    "dimensions": 384
  },
  "tokenBudgets": {
    "tier0_working": 2000,
    "tier1_session": 500,
    "tier2_epoch": 300,
    "tier3_core": 200,
    "retrieval": 2000,
    "graphFacts": 500,
    "default_retrieve_budget": 3000
  },
  "compression": {
    "tier0_overflow_threshold": 2500,
    "tier1_consolidation_count": 10,
    "confidence_decay_rate": 0.01,
    "confidence_decay_interval_hours": 24
  },
  "ranking": {
    "semantic_weight": 0.4,
    "recency_weight": 0.3,
    "priority_weight": 0.2,
    "frequency_weight": 0.1,
    "dedup_similarity_threshold": 0.85
  }
}
```

---

## 7. Critical Design Considerations

### 7.1 The Compression-Fidelity Tradeoff

Every compression step risks information loss. The system mitigates this through three mechanisms. First, the knowledge graph preserves discrete facts losslessly — "John lives in Paris" never gets fuzzy through summarization because it is stored as a structured triple. Second, the vector index preserves semantic access to all stored content, so even if a summary omits a detail, a sufficiently specific query can surface the original. Third, confidence scoring means the system knows what it knows well versus what it has only weakly retained, and can communicate this uncertainty to the LLM.

### 7.2 Cold Start Problem

On first use, all memory tiers are empty. The system must degrade gracefully: `memory_retrieve` returns an empty context block, and the LLM operates normally using only the conversation history provided by the host. As interactions accumulate, the memory system progressively activates. The system should **not** require any pre-population to be useful.

### 7.3 Consistency and Conflict Resolution

When a new fact contradicts an existing one (e.g., "John moved from Paris to London"), the system must update rather than accumulate. The `memory_store` tool should check for existing triples with the same subject-predicate pair and either update them or create a new entry with a temporal marker while deprecating the old one. The knowledge graph's `temporal_start` and `temporal_end` fields support this.

### 7.4 Privacy and Data Lifecycle

All data is stored locally on the user's machine. The server stores no data externally. If using a local embedding model, no data leaves the machine at all. The configuration should support a maximum retention period after which memories are automatically pruned, and a full `memory_wipe` tool for complete data deletion.

### 7.5 The Summarization Dependency

Tier promotions (0→1, 1→2) require summarization, which itself requires an LLM call. This creates a dependency: the MCP server may need to call an LLM to compress context before it can serve context to an LLM. Two approaches exist. **Internal compression** uses the host LLM itself via the prompt templates — the host application orchestrates the compression by calling the `compress_session` prompt and feeding the result back to `memory_store`. **External compression** uses a small, fast, local model (e.g., a quantized Llama or Phi model) running on the server for summarization tasks, keeping the critical path independent. The first approach is simpler and recommended for Phase 1–3; the second is a Phase 5+ optimization.

---

## 8. Example Interaction Flow

```
User: "Hey, remember last week we talked about my trip to Japan? 
       What restaurant did I say I wanted to visit in Tokyo?"

Host App → LLM: [system prompt + user message]

LLM decides to call: memory_retrieve({
  query: "trip to Japan, restaurant in Tokyo, user wanted to visit",
  token_budget: 2000
})

LatentContext Server:
  1. Embeds query → vector search → finds Session Summary from 7 days ago
     mentioning Japan trip planning (similarity: 0.89)
  2. Queries knowledge graph: entity="Japan" → finds triples:
     - (User, planning_trip_to, Japan, confidence: 0.95)
     - (User, wants_to_visit, "Sukiyabashi Jiro", confidence: 0.88)
     - (Trip, destination_city, Tokyo, confidence: 0.95)
  3. Assembles context block:
     "[Core Memory] User enjoys travel and Japanese cuisine.
      [Session: Jan 15] Discussed planning a trip to Japan in March.
      User specifically wants to visit Sukiyabashi Jiro restaurant in
      Tokyo. Also considering visiting Kyoto for 2 days.
      [Fact] User → wants_to_visit → Sukiyabashi Jiro (confidence: 0.88)"
  4. Returns assembled context (147 tokens instead of ~3000 tokens
     of raw conversation history)

LLM → User: "Yes! You mentioned you wanted to visit Sukiyabashi Jiro
             in Tokyo. We also talked about spending 2 days in Kyoto."
```

---

## 9. Success Metrics

**Token Efficiency Ratio:** Measure the ratio of tokens in the assembled context versus tokens in the equivalent raw conversation history. Target: 5:1 or better compression.

**Retrieval Precision:** When the LLM calls `memory_retrieve`, what percentage of returned context chunks are actually used in the response? Measure via automated evaluation. Target: >70%.

**Fact Accuracy:** For facts stored in the knowledge graph, what percentage are correctly recalled when queried? Target: >95%.

**Latency:** `memory_retrieve` should complete in under 200ms for local embedding, under 500ms for remote embedding. The MCP tool call overhead should not be perceptible to the user.

**Graceful Degradation:** With empty memory stores, the system adds zero latency and returns empty results without errors. The LLM experience is identical to having no MCP server.

---

## 10. Future Extensions

**Cross-Session Reasoning Chains:** Store not just facts but reasoning chains — *why* a conclusion was reached — enabling the LLM to reconstruct its previous reasoning rather than re-derive it.

**Multi-Agent Memory Sharing:** Multiple MCP clients (different LLM agents) can connect to the same LatentContext server, sharing a common knowledge graph while maintaining separate working memories.

**Adaptive Compression:** The server learns which types of information the user queries most frequently and adjusts compression aggressiveness per category — heavily compressing rarely-accessed information while preserving detail for frequently-accessed topics.

**Embedding Space Alignment:** As embedding models improve, implement a migration pipeline that re-embeds all stored vectors with a new model without losing the structured data.

---

*This document is intended to be fed directly to an LLM with coding capabilities to implement the LatentContext MCP Server. All architectural decisions, tool schemas, and algorithms are specified with sufficient precision for code generation. Begin with Phase 1 and iterate.*