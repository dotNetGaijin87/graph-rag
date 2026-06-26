# Local GraphRAG Knowledge Base

A self-hosted, **fully offline** Retrieval-Augmented Generation (RAG) system that uses a
**graph database** for semantic storage. Paste text into a web UI, and it is chunked,
embedded, and turned into a knowledge graph. Then ask questions and get grounded answers
from a **local LLM** — nothing ever leaves your machine.

> **Proof of concept.** This is the core end-to-end loop (ingest → graph → retrieve →
> answer). The architecture is designed to grow toward full GraphRAG (see
> [Roadmap](#roadmap)).

```
┌──────────────┐   paste text    ┌─────────────────────────────┐   Cypher    ┌────────────┐
│   React UI   │ ──────────────► │        Flask backend        │ ──────────► │   Neo4j    │
│ (TypeScript) │ ◄────────────── │   (clean architecture)      │ ◄────────── │  graph +   │
└──────────────┘   answer +      │  ingest / retrieve use-cases│   vectors   │  vectors   │
                   sources       └──────────────┬──────────────┘             └────────────┘
                                                │ embeddings + chat
                                                ▼
                                       ┌──────────────────┐
                                       │      Ollama      │
                                       │ nomic-embed-text │
                                       │     llama3.2     │
                                       └──────────────────┘
```

## Why a graph database?

This project uses the **GraphRAG** approach. Instead of a plain vector store, Neo4j holds
**both**:

1. **Vectors** — each text chunk is a `:Chunk` node with an `embedding`, served by Neo4j's
   native vector index for semantic search.
2. **A knowledge graph** — the LLM extracts `:Entity` nodes and `[:RELATED_TO]`
   relationships, and links each chunk to the entities it `[:MENTIONS]`.

Retrieval is therefore **hybrid**: vector search finds the most relevant chunks, then the
graph is traversed one hop to pull in connected facts. That extra structured context is
what plain vector RAG cannot give you.

## Architecture

The retrieval design is the **baseline + entity-graph** level of GraphRAG — the simplest
approach that genuinely uses the graph rather than treating Neo4j as just a vector store.

### Ingestion pipeline
```
text ─► chunk (sliding window) ─► embed chunks (Ollama) ─► :Chunk nodes + vector index
                                                        └─► LLM extracts entities/relations
                                                            ─► :Entity nodes, [:RELATED_TO],
                                                               [:MENTIONS] links
```

### Retrieval pipeline
```
question ─► embed ─► vector search top-k :Chunk ─► collect mentioned :Entity
                                               └─► expand [:RELATED_TO] (graph facts)
         ─► assemble context (passages + facts) ─► LLM answers, grounded only in context
```

### Neo4j schema
```
(:Document {id, title})-[:HAS_CHUNK]->(:Chunk {id, text, index, embedding})
(:Chunk)-[:MENTIONS]->(:Entity {name, type, description})
(:Entity)-[:RELATED_TO {type, description}]->(:Entity)
```
Indexes: a **vector index** on `Chunk.embedding` (cosine) and a **full-text index** on
`Chunk.text`. Uniqueness constraints on `Document.id`, `Chunk.id`, `Entity.name`.

## Tech stack

| Layer       | Choice                                            |
|-------------|---------------------------------------------------|
| Frontend    | React 18 + TypeScript + Vite + Material UI (MUI)  |
| Backend     | Python + Flask, **clean architecture / modular monolith** |
| Graph DB    | Neo4j 5 (graph + native vector index)             |
| LLM & embed | Ollama (`llama3.2`, `nomic-embed-text`) — open source, local |
| Orchestration | Docker Compose                                  |

## Prerequisites

- **Docker** and **Docker Compose** (v2).
- ~6 GB free disk for the model weights and images.
- First run downloads the Ollama models (~2.5 GB) — this takes a few minutes.
- CPU-only works; a GPU makes the LLM much faster (uncomment the `deploy` block for
  `ollama` in [docker-compose.yml](docker-compose.yml)).

## Quick start

```bash
# from the repository root
docker compose up --build
```

Then open:

| Service        | URL                                              |
|----------------|--------------------------------------------------|
| **Web UI**     | http://localhost:3000                            |
| Backend API    | http://localhost:8000/api/health                 |
| Neo4j Browser  | http://localhost:7474  (user `neo4j`, pass `password123`) |

The first startup waits for `ollama-init` to pull the models before the backend starts.
Watch the logs — once you see `RAG backend ready.` you're good to go.

### Using it
The UI has two tabs:

**Knowledge base** tab:
1. Paste any text (an article, notes, documentation) into the **left** panel and click
   *Add to knowledge base*.
2. Ask a question in the **right** panel. The answer comes back with expandable
   **sources** — the retrieved passages and the knowledge-graph facts used.

**Knowledge graph** tab:
3. An interactive Cytoscape view of the extracted entities (coloured by type) and their
   relationships. Drag nodes, scroll to zoom, and hit *Refresh* after adding text.

**Settings** tab:
4. Tune the RAG pipeline live — chunk size/overlap, top‑K, max extraction characters, and
   the entity-extraction toggle. Changes apply immediately (chunking affects newly added
   text) and persist in memory until the backend restarts.

You can also inspect the graph directly in the Neo4j Browser, e.g.:
   ```cypher
   MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c, e LIMIT 50;
   MATCH p=(:Entity)-[:RELATED_TO]->(:Entity) RETURN p LIMIT 50;
   ```

## Configuration

These env vars set the **startup defaults**; the chunking/retrieval ones can then be
adjusted at runtime from the **Settings** tab (or `PUT /api/settings`). All have sensible
defaults and can be overridden via a `.env` file (copy [.env.example](.env.example)):

| Variable                   | Default            | Meaning                                  |
|----------------------------|--------------------|------------------------------------------|
| `LLM_MODEL`                | `llama3.2`         | Ollama chat model for extraction/answers |
| `EMBEDDING_MODEL`          | `nomic-embed-text` | Ollama embedding model                   |
| `EMBEDDING_DIM`            | `768`              | Must match the embedding model           |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100`  | Chunking window (characters)             |
| `TOP_K`                    | `5`                | Chunks retrieved per question            |
| `ENABLE_ENTITY_EXTRACTION` | `true`             | Off = plain vector RAG (faster ingest)   |

> Changing `EMBEDDING_MODEL` changes the vector dimension — set `EMBEDDING_DIM` to match
> and reset the knowledge base, because old and new vectors are not comparable.

## Troubleshooting

**SSL certificate errors behind a corporate proxy / antivirus (TLS inspection).**
Your host trusts the proxy's root certificate, but the containers do not, so downloads
fail certificate verification. There are two download phases, fixed separately:

1. **Image build** (pip + npm) — `CERTIFICATE_VERIFY_FAILED` / `unable to get local
   issuer certificate`. Fix: set `INSECURE_TLS=1` in your `.env` (relaxes cert checks for
   the package registries during the build only).

2. **Ollama model pull** — `tls: failed to verify certificate: x509: certificate signed
   by unknown authority`. Ollama cannot skip verification, so it must *trust* your proxy's
   root CA. Export your trusted roots into [certs/](certs/):

   ```powershell
   powershell -ExecutionPolicy Bypass -File certs\export-windows-cas.ps1
   ```

   (See [certs/README.md](certs/README.md) for details / non-Windows.) Ollama loads them
   via `SSL_CERT_DIR=/certs`.

Then rebuild and start:

```bash
docker compose down
docker compose up --build
```

On a normal network without TLS inspection, neither step is needed (`INSECURE_TLS=0`, no
certs).

**The backend keeps retrying to reach Neo4j** — expected on first boot; Neo4j takes ~30 s
to become healthy. The backend retries for ~60 s.

**The first question is slow** — the local LLM is loading into memory on the first call.
Subsequent calls are faster. A GPU helps a lot (see prerequisites).

## API reference

| Method | Path             | Body                       | Description                       |
|--------|------------------|----------------------------|-----------------------------------|
| GET    | `/api/health`    | —                          | Liveness check                    |
| GET    | `/api/stats`     | —                          | Counts of docs/chunks/entities    |
| GET    | `/api/graph`     | `?limit=N`                 | Entities + relationships for the graph view |
| GET    | `/api/settings`  | —                          | Current RAG tuning parameters     |
| PUT    | `/api/settings`  | `{ "chunk_size", ... }`    | Update tuning parameters at runtime |
| POST   | `/api/documents` | `{ "text", "title?" }`     | Ingest text into the graph        |
| POST   | `/api/query`     | `{ "question" }`           | Ask a question, get a grounded answer |
| POST   | `/api/reset`     | —                          | Wipe all stored knowledge         |

## Project structure

```
.
├── docker-compose.yml          # Neo4j + Ollama + backend + frontend
├── backend/                    # Flask, clean architecture
│   └── app/
│       ├── domain/             # models + ports (no framework deps)
│       ├── application/        # use cases: ingest_text, answer_question
│       ├── infrastructure/     # adapters: ollama/, neo4j/
│       ├── api/                # Flask blueprints (controllers)
│       ├── config.py
│       └── container.py        # dependency-injection wiring
└── frontend/                   # React + TypeScript (Vite)
    └── src/
        ├── api/                # typed API client
        └── components/         # IngestPanel, QueryPanel, AnswerView, StatsBar, GraphView, SettingsPanel
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the clean-architecture rationale and
the GraphRAG design in depth.

## Local development (without Docker)

You still need Neo4j and Ollama running locally (or just `docker compose up neo4j ollama
ollama-init`).

```bash
# backend
cd backend
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements-dev.txt
NEO4J_URI=bolt://localhost:7687 OLLAMA_BASE_URL=http://localhost:11434 python wsgi.py
pytest                                                   # run unit tests

# frontend (separate terminal) — proxies /api to localhost:8000
cd frontend
npm install
npm run dev      # http://localhost:3000
```

## Roadmap

The PoC implements baseline GraphRAG + entity extraction. Natural next steps, in order
of value:

- [ ] **Hybrid search** — union the vector index with the full-text index (already created).
- [ ] **Text2Cypher** — route counting/aggregation questions to LLM-generated Cypher.
- [ ] **Entity resolution** — merge duplicate entities ("ACME"/"ACME Ltd").
- [ ] **Parent-document retrieval** — embed small chunks, return their parent for context.
- [ ] **Community summaries** (Microsoft GraphRAG) — Louvain clustering + per-community
      summaries for broad "what is this corpus about" questions.
- [x] **Graph visualisation** in the UI (interactive Cytoscape view of entities/relations).
- [ ] Streaming answers; document management; click-a-node-to-filter.

## License

MIT — see [LICENSE](LICENSE).
