"""Neo4j adapter — graph store and vector store in one."""

from __future__ import annotations

import logging
import re
import time

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from ...domain.models import (
    Chunk,
    ExtractionResult,
    GraphFact,
    RetrievedChunk,
)
from ...domain.ports import GraphRepository

logger = logging.getLogger(__name__)

CHUNK_VECTOR_INDEX = "chunk_embeddings"
CHUNK_FULLTEXT_INDEX = "chunk_fulltext"
ENTITY_VECTOR_INDEX = "entity_embeddings"

# Lucene reserved characters — stripped from the user's question before it is handed
# to the full-text index, so a stray "?" or ":" can't break the query parser.
_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def lucene_query(text: str) -> str:
    cleaned = _LUCENE_SPECIAL.sub(" ", text or "")
    return " ".join(term for term in cleaned.split() if term)


# Union the vector and full-text indexes, normalising each branch's scores by its own max.
_HYBRID_CYPHER = f"""
CALL {{
    CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $k, $embedding) YIELD node, score
    WITH collect({{node: node, score: score}}) AS hits, max(score) AS maxScore
    UNWIND hits AS hit
    RETURN hit.node AS node,
           CASE WHEN maxScore > 0 THEN hit.score / maxScore ELSE 0.0 END AS score
  UNION
    CALL db.index.fulltext.queryNodes('{CHUNK_FULLTEXT_INDEX}', $keywords, {{limit: $k}})
    YIELD node, score
    WITH collect({{node: node, score: score}}) AS hits, max(score) AS maxScore
    UNWIND hits AS hit
    RETURN hit.node AS node,
           CASE WHEN maxScore > 0 THEN hit.score / maxScore ELSE 0.0 END AS score
}}
WITH node, max(score) AS score
ORDER BY score DESC
LIMIT $k
OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
RETURN node.id AS chunk_id,
       node.document_id AS document_id,
       node.text AS text,
       score AS score,
       collect(DISTINCT e.name) AS entities
ORDER BY score DESC
"""

_VECTOR_ONLY_CYPHER = f"""
CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $k, $embedding) YIELD node, score
OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
RETURN node.id AS chunk_id,
       node.document_id AS document_id,
       node.text AS text,
       score AS score,
       collect(DISTINCT e.name) AS entities
ORDER BY score DESC
"""

_ENTITY_SEARCH_CYPHER = f"""
CALL db.index.vector.queryNodes('{ENTITY_VECTOR_INDEX}', $k, $embedding) YIELD node, score
RETURN node.name AS name
ORDER BY score DESC
"""


class Neo4jGraphRepository(GraphRepository):
    def __init__(self, uri: str, user: str, password: str, embedding_dim: int) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._embedding_dim = embedding_dim

    def ensure_schema(self, retries: int = 30, delay: float = 2.0) -> None:
        """Wait for Neo4j, then create constraints and indexes idempotently."""
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self._driver.verify_connectivity()
                break
            except (ServiceUnavailable, OSError) as exc:  # Neo4j still starting up
                last_error = exc
                logger.info("Waiting for Neo4j (%d/%d)...", attempt, retries)
                time.sleep(delay)
        else:
            raise RuntimeError(f"Neo4j not reachable: {last_error}")

        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            # Dimension cannot be parameterised in DDL; it comes from trusted config.
            session.run(
                f"CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {int(self._embedding_dim)}, "
                "`vector.similarity_function`: 'cosine'}}"
            )
            session.run(
                f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text]"
            )
            session.run(
                f"CREATE VECTOR INDEX {ENTITY_VECTOR_INDEX} IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {int(self._embedding_dim)}, "
                "`vector.similarity_function`: 'cosine'}}"
            )
        logger.info("Neo4j schema ensured.")

    def save_document(
        self,
        document_id: str,
        title: str,
        chunks: list[Chunk],
        extraction: ExtractionResult,
        entity_embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        entity_embeddings = entity_embeddings or {}
        chunk_rows = [
            {
                "id": c.id,
                "text": c.text,
                "index": c.index,
                "embedding": c.embedding,
            }
            for c in chunks
        ]
        entity_rows = [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "embedding": entity_embeddings.get(e.name),
            }
            for e in extraction.entities
        ]
        relationship_rows = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "description": r.description,
            }
            for r in extraction.relationships
        ]
        entity_names = [e.name for e in extraction.entities]

        with self._driver.session() as session:
            session.execute_write(
                self._write_document,
                document_id,
                title,
                chunk_rows,
                entity_rows,
                relationship_rows,
                entity_names,
            )

    @staticmethod
    def _write_document(
        tx,
        document_id: str,
        title: str,
        chunk_rows: list[dict],
        entity_rows: list[dict],
        relationship_rows: list[dict],
        entity_names: list[str],
    ) -> None:
        tx.run(
            """
            MERGE (d:Document {id: $document_id})
            SET d.title = $title
            WITH d
            UNWIND $chunks AS chunk
            CREATE (c:Chunk {id: chunk.id})
            SET c.text = chunk.text,
                c.index = chunk.index,
                c.document_id = $document_id,
                c.embedding = chunk.embedding
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=document_id,
            title=title,
            chunks=chunk_rows,
        )

        if entity_rows:
            tx.run(
                """
                UNWIND $entities AS entity
                MERGE (e:Entity {name: entity.name})
                SET e.type = entity.type,
                    e.description = CASE
                        WHEN entity.description <> '' THEN entity.description
                        ELSE e.description END,
                    e.embedding = CASE
                        WHEN entity.embedding IS NULL THEN e.embedding
                        ELSE entity.embedding END
                """,
                entities=entity_rows,
            )

        # Single RELATED_TO type; the semantic verb is stored as a property.
        if relationship_rows:
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (a:Entity {name: rel.source})
                MATCH (b:Entity {name: rel.target})
                MERGE (a)-[r:RELATED_TO {type: rel.type}]->(b)
                SET r.description = rel.description
                """,
                relationships=relationship_rows,
            )

        # Link chunks to the entities they mention (substring match, scoped to this doc).
        if entity_names:
            tx.run(
                """
                MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                UNWIND $names AS name
                MATCH (e:Entity {name: name})
                WHERE toLower(c.text) CONTAINS toLower(name)
                MERGE (c)-[:MENTIONS]->(e)
                """,
                document_id=document_id,
                names=entity_names,
            )

    def search_chunks(
        self, query_text: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval (vector + full-text). Falls back to vector-only if the
        question has no usable keywords."""
        keywords = lucene_query(query_text)
        with self._driver.session() as session:
            if keywords:
                records = session.run(
                    _HYBRID_CYPHER, k=k, embedding=query_embedding, keywords=keywords
                )
            else:
                records = session.run(_VECTOR_ONLY_CYPHER, k=k, embedding=query_embedding)
            return [
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    text=r["text"],
                    score=float(r["score"]),
                    entities=[name for name in r["entities"] if name],
                )
                for r in records
            ]

    def search_entities(self, query_embedding: list[float], k: int) -> list[str]:
        with self._driver.session() as session:
            records = session.run(_ENTITY_SEARCH_CYPHER, k=k, embedding=query_embedding)
            return [r["name"] for r in records if r["name"]]

    def graph_facts_for_entities(self, entity_names: list[str], limit: int) -> list[GraphFact]:
        if not entity_names:
            return []
        with self._driver.session() as session:
            records = session.run(
                """
                UNWIND $names AS name
                MATCH (e:Entity {name: name})-[r:RELATED_TO]-(other:Entity)
                WITH DISTINCT startNode(r) AS s, r, endNode(r) AS t
                RETURN s.name AS source, r.type AS type, t.name AS target,
                       r.description AS description
                LIMIT $limit
                """,
                names=entity_names,
                limit=limit,
            )
            return [
                GraphFact(
                    source=r["source"],
                    type=r["type"] or "RELATED_TO",
                    target=r["target"],
                    description=r["description"] or "",
                )
                for r in records
            ]

    def graph_overview(self, limit: int) -> dict:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        with self._driver.session() as session:
            for r in session.run(
                """
                MATCH (a:Entity)-[rel:RELATED_TO]->(b:Entity)
                RETURN a.name AS source, a.type AS source_type, a.description AS source_desc,
                       b.name AS target, b.type AS target_type, b.description AS target_desc,
                       rel.type AS type, rel.description AS description
                LIMIT $limit
                """,
                limit=limit,
            ):
                nodes.setdefault(
                    r["source"],
                    {"id": r["source"], "type": r["source_type"] or "Concept",
                     "description": r["source_desc"] or ""},
                )
                nodes.setdefault(
                    r["target"],
                    {"id": r["target"], "type": r["target_type"] or "Concept",
                     "description": r["target_desc"] or ""},
                )
                edges.append(
                    {
                        "source": r["source"],
                        "target": r["target"],
                        "type": r["type"] or "RELATED_TO",
                        "description": r["description"] or "",
                    }
                )

            # Standalone entities that have no relationships yet.
            for r in session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATED_TO]-()
                RETURN e.name AS id, e.type AS type, e.description AS description
                LIMIT $limit
                """,
                limit=limit,
            ):
                nodes.setdefault(
                    r["id"],
                    {"id": r["id"], "type": r["type"] or "Concept",
                     "description": r["description"] or ""},
                )

        return {"nodes": list(nodes.values()), "edges": edges}

    def stats(self) -> dict:
        with self._driver.session() as session:
            record = session.run(
                """
                OPTIONAL MATCH (d:Document) WITH count(d) AS documents
                OPTIONAL MATCH (c:Chunk) WITH documents, count(c) AS chunks
                OPTIONAL MATCH (e:Entity) WITH documents, chunks, count(e) AS entities
                OPTIONAL MATCH ()-[r:RELATED_TO]->()
                RETURN documents, chunks, entities, count(r) AS relationships
                """
            ).single()
            if record is None:
                return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}
            return {
                "documents": record["documents"],
                "chunks": record["chunks"],
                "entities": record["entities"],
                "relationships": record["relationships"],
            }

    def reset(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()
