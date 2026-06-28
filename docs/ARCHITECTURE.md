# Architecture

This document covers the two architectural pillars of the project: the **GraphRAG**
retrieval design (the _what_) and the **clean-architecture modular monolith** on the
backend (the _how_). For screenshots and quick start, see the [README](../README.md).

## 1. GraphRAG design

GraphRAG can be built as an escalating ladder of sophistication. The last column maps each
rung to where this repo stands:

| #   | Architecture                                                                                | Status here     |
| --- | ------------------------------------------------------------------------------------------- | --------------- |
| 0   | **Baseline vector RAG** — `:Chunk` + vector index, top-k → prompt                           | ✅ implemented  |
| 1   | **Hybrid search** — union the vector and full-text indexes, normalise scores                | ✅ implemented  |
| 2   | **Advanced retrieval** — parent-document retriever, step-back / HyDE query rewriting        | ⬜ roadmap      |
| 3   | **Text2Cypher** — LLM generates Cypher for filtering/counting/aggregation                   | ⬜ roadmap      |
| 4   | **Agentic / router RAG** — retriever router + specialised templates + answer critic         | ⬜ roadmap      |
| 5   | **LLM knowledge-graph construction** — typed `:Entity` + `[:RELATED_TO]`, entity resolution | 🟡 simplified   |
| 6   | **Microsoft-style GraphRAG** — entity-anchored local search ✅; community detection + summaries / global search ⬜ | 🟡 simplified   |

Where this repo sits: **levels 0 and 1 in full**, a **simplified level 5** — it LLM-extracts a
real entity/relationship graph from every chunk, deduplicates entities case-insensitively, and
LLM-merges the descriptions a name collects into one summary (mention linking is still
substring-based; full entity resolution is not done) — and **part of level 6**: entity-anchored
_local search_ (entities are embedded into their own vector index and matched directly to the
query), plus one-hop `[:RELATED_TO]` graph expansion. It deliberately skips levels 2–4 and the
heavy parts of 5–6 (full entity resolution, community detection / global search) — all additive,
none requiring a re-architecture.

### The single most important rule

> Use the **same embedding model and dimension** at ingest time and query time.

`nomic-embed-text` produces 768-dimensional vectors. The vector index is created with
`vector.dimensions: 768`. If you switch embedding models, update `EMBEDDING_DIM` and reset
the database — vectors from different models are not comparable.

### Ingestion (`IngestTextUseCase`)

1. **Chunk** the text with a sliding window over word boundaries (`chunk_size=800`,
   `overlap=100`). Words are never split.
2. **Embed** all chunks in one batched Ollama call.
3. **Extract** a knowledge graph from **each chunk** using Ollama's JSON-schema-constrained
   output (`format`). Entities are deduplicated case-insensitively across chunks; when a name
   gathers several descriptions, they are LLM-merged into one. Relationships whose endpoints
   are not real extracted entities are dropped.
4. **Embed entities** (name + description) so the query can match them directly (local search).
5. **Persist** atomically in Neo4j: `:Document`, `:Chunk` (with embeddings), `:Entity` (with
   embeddings), `[:RELATED_TO]`, and `[:MENTIONS]` links (a chunk mentions an entity when the
   entity's name appears in the chunk text).

Extraction is best-effort: if it fails the system degrades gracefully to plain vector RAG.

<p align="center"><img src="img/flow-ingest.png" alt="Knowledge-creation sequence" width="880"></p>

### Retrieval (`AnswerQuestionUseCase`)

1. **Embed** the question (same model) — this query vector is the input to the vector search.
2. **Hybrid search** the top-`k` chunks: union `db.index.vector.queryNodes` with
   `db.index.fulltext.queryNodes`, normalise each branch's scores by its own max, and merge.
   The question is sanitised of Lucene reserved characters first; if it has no usable
   keywords, retrieval falls back to vector-only.
3. **Entity entry points** ("local search"): match the query vector against the
   `entity_embeddings` index to find the most relevant entities directly — not only the
   ones the retrieved chunks happen to mention.
4. **Graph expansion**: union the chunk-mentioned entities with those entry-point entities,
   then fetch their one-hop `[:RELATED_TO]` relationships as "graph facts".
5. **Assemble** a context of passages + facts and ask the LLM to answer using _only_ that
   context (grounding prompt). If nothing is retrieved, return a safe "I don't know".

<p align="center"><img src="img/flow-query.png" alt="Question-answering sequence" width="880"></p>

This is why the graph matters: two facts that live in different chunks but are connected
through a shared entity can both reach the answer, which pure top-k vector RAG would miss.

### Neo4j schema

A single Neo4j database is **both** the vector store and the knowledge graph:

```cypher
(:Document {id, title})-[:HAS_CHUNK]->(:Chunk {id, index, text, embedding})
(:Chunk)-[:MENTIONS]->(:Entity {name, type, description, embedding})
(:Entity)-[:RELATED_TO {type, description}]->(:Entity)
```

Indexes and constraints are created idempotently on boot by `ensure_schema`:

- **Vector index** `chunk_embeddings` on `Chunk.embedding` (cosine, 768-dim) — semantic search.
- **Full-text index** `chunk_fulltext` on `Chunk.text` — the keyword half of hybrid search.
- **Vector index** `entity_embeddings` on `Entity.embedding` (cosine, 768-dim) — entity-anchored
  ("local search") entry points.
- **Uniqueness constraints** on `Document.id`, `Chunk.id`, `Entity.name` (the last lets
  `MERGE (:Entity {name})` deduplicate entities across documents).

## 2. Clean architecture (backend)

The backend is a **modular monolith** organised by the Dependency Rule: **dependencies
point inward**, toward the domain. The domain knows nothing about Flask, Neo4j, or Ollama.

```
            ┌──────────────────────────────────────────────┐
            │                    api/                       │  Flask blueprints (controllers)
            │  routes.py, serializers.py                    │  → depends on application
            ├──────────────────────────────────────────────┤
            │                application/                   │  use cases (orchestration)
            │  ingest_text.py, answer_question.py,          │  → depends on domain ports
            │  chunking.py, prompts.py                      │
            ├──────────────────────────────────────────────┤
            │                  domain/                      │  models + ports (interfaces)
            │  models.py (dataclasses), ports.py (ABCs)     │  → depends on NOTHING
            ├──────────────────────────────────────────────┤
            │               infrastructure/                 │  adapters implementing ports
            │  ollama/ (LLM, embeddings), neo4j/ (repo)     │  → depends on domain
            └──────────────────────────────────────────────┘
                       container.py  ── wires it all together (composition root)
```

The Dependency Inversion Principle in one picture — the application's use cases **«use»** the
ports (interfaces) defined in `domain/`; the infrastructure adapters **implement** them. The
domain never imports infrastructure, so every dependency arrow points inward:

<p align="center"><img src="img/hexagonal.png" alt="Hexagonal ports and adapters" width="860"></p>

### Layers

- **`domain/`** — `models.py` holds plain frozen dataclasses (`Chunk`, `Entity`,
  `Answer`, …). `ports.py` declares abstract interfaces: `EmbeddingProvider`,
  `LLMProvider`, `GraphRepository`. No third-party imports. This is the stable core.
- **`application/`** — use cases that orchestrate the domain through the ports. They
  receive their dependencies via constructor injection and contain the RAG logic. Pure
  enough to unit-test without any infrastructure (see `tests/`).
- **`infrastructure/`** — concrete adapters: `OllamaEmbeddingProvider`,
  `OllamaLLMProvider`, `Neo4jGraphRepository`. Swapping Neo4j for another store, or Ollama
  for a hosted model, means writing one new adapter — nothing in the core changes.
- **`api/`** — thin Flask controllers that parse requests, call a use case, and serialise
  the result. No business logic.
- **`container.py`** — the composition root. The only place that imports both application
  and infrastructure, building the object graph from `Config`.

### Why this shape

- **Testability** — use cases depend on interfaces, so they're tested with fakes; pure
  logic (chunking, extraction parsing) is tested directly. No DB or LLM needed.
- **Swappability** — the "graph database" and "Ollama" requirements are honoured behind
  ports. A future Postgres+pgvector or hosted-LLM variant is an adapter, not a rewrite.
- **Modularity** — ingestion and retrieval are independent use cases; new capabilities
  (Text2Cypher, community summaries) slot in as new use cases + ports without touching the
  existing ones.

> **Runtime configuration.** Tuning parameters (chunk size, top-K, extraction toggle, …)
> live in a mutable `RuntimeSettings` object, editable via `PUT /api/settings` (the Settings
> tab), separate from the immutable boot `Config`. Because those settings are held in process
> memory, the backend runs as a **single gunicorn worker** — multiple workers would each keep
> their own copy and updates would be inconsistent.

## 3. Deployment topology

Five services, wired by Docker Compose:

<p align="center"><img src="img/architecture.png" alt="Deployment architecture" width="880"></p>

- **`neo4j`** — graph + vector store. Exposes Bolt (7687) and Browser (7474).
- **`ollama`** — serves the LLM and embedding models on 11434.
- **`ollama-init`** — a one-shot job that pulls the models, then exits; the backend waits
  for it to complete successfully before starting.
- **`backend`** — Flask/gunicorn (single worker) on 8000; creates the Neo4j schema on boot
  (with retry while Neo4j warms up).
- **`frontend`** — nginx serving the built React app on 3000 and reverse-proxying `/api`
  to the backend (so the browser never needs CORS or a separate API host).
