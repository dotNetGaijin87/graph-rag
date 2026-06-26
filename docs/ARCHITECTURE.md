# Architecture

This document explains the two architectural pillars of the project: the **GraphRAG**
retrieval design (the *what*) and the **clean-architecture modular monolith** on the
backend (the *how*).

## 1. GraphRAG design

GraphRAG can be built as an escalating ladder of sophistication. The columns map each rung
to where this repo stands:

| # | Architecture | Status here |
|---|--------------|-------------|
| 0 | **Baseline vector RAG** — `:Chunk` + vector index, top-k → prompt | ✅ implemented |
| 1 | **Hybrid search** — union the vector and full-text indexes, normalise scores | ✅ implemented |
| 2 | **Advanced retrieval** — parent-document retriever, step-back / HyDE query rewriting | ⬜ roadmap |
| 3 | **Text2Cypher** — LLM generates Cypher for filtering/counting/aggregation | ⬜ roadmap |
| 4 | **Agentic / router RAG** — retriever router + specialised templates + answer critic | ⬜ roadmap |
| 5 | **LLM knowledge-graph construction** — typed `:Entity` + `[:RELATED_TO]`, entity resolution | 🟡 simplified |
| 6 | **Microsoft-style GraphRAG** — community detection + summaries, global/local search | ⬜ roadmap |

Where this repo sits: **levels 0 and 1 in full**, plus a **simplified level 5** — it does
LLM-extract a real entity/relationship graph and link chunks via `[:MENTIONS]`, but without
entity resolution, without embedding entities, and with substring-based mention linking. On
top of that it adds a **hand-rolled graph expansion** at query time (vector hit → one-hop
`[:RELATED_TO]` facts) that echoes level-6 *local search* without its machinery (no entity
vector index, communities, or ranking). It deliberately skips levels 2–4 and the heavy parts
of 5–6 — all additive, none requiring a re-architecture.

### The single most important rule

> Use the **same embedding model and dimension** at ingest time and query time.

`nomic-embed-text` produces 768-dimensional vectors. The vector index is created with
`vector.dimensions: 768`. If you switch embedding models, update `EMBEDDING_DIM` and reset
the database — vectors from different models are not comparable.

### Ingestion (`IngestTextUseCase`)

1. **Chunk** the text with a sliding window over word boundaries (`chunk_size=800`,
   `overlap=100`). Words are never split.
2. **Embed** all chunks in one batched Ollama call.
3. **Extract** a knowledge graph with one LLM call using Ollama's JSON-schema-constrained
   output (`format`), yielding entities and relationships. Relationships whose endpoints
   are not real extracted entities are dropped.
4. **Persist** atomically in Neo4j: `:Document`, `:Chunk` (with embeddings), `:Entity`,
   `[:RELATED_TO]`, and `[:MENTIONS]` links (a chunk mentions an entity when the entity's
   name appears in the chunk text).

Extraction is best-effort: if it fails the system degrades gracefully to plain vector RAG.

### Retrieval (`AnswerQuestionUseCase`)

1. **Embed** the question (same model).
2. **Hybrid search** the top-`k` chunks: union `db.index.vector.queryNodes` with
   `db.index.fulltext.queryNodes`, normalise each branch's scores by its own max, and merge.
   The question is sanitised of Lucene reserved characters first; if it has no usable
   keywords, retrieval falls back to vector-only.
3. **Graph expansion**: collect the entities those chunks mention, then fetch their
   one-hop `[:RELATED_TO]` relationships as "graph facts".
4. **Assemble** a context of passages + facts and ask the LLM to answer using *only* that
   context (grounding prompt). If nothing is retrieved, return a safe "I don't know".

This is why the graph matters: two facts that live in different chunks but are connected
through a shared entity can both reach the answer, which pure top-k vector RAG would miss.

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

## 3. Deployment topology

Four containers wired by Docker Compose:

- **`neo4j`** — graph + vector store. Exposes Bolt (7687) and Browser (7474).
- **`ollama`** — serves the LLM and embedding models on 11434.
- **`ollama-init`** — a one-shot job that pulls the models, then exits; the backend waits
  for it to complete successfully before starting.
- **`backend`** — Flask/gunicorn on 8000; creates the Neo4j schema on boot (with retry
  while Neo4j warms up).
- **`frontend`** — nginx serving the built React app on 3000 and reverse-proxying `/api`
  to the backend (so the browser never needs CORS or a separate API host).
