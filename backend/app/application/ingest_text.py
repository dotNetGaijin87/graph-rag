"""Ingest text: chunk, embed, extract a graph, and store it."""

from __future__ import annotations

import logging
import uuid

from ..domain.models import Chunk, Entity, ExtractionResult, IngestionReport, Relationship
from ..domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from ..settings import RuntimeSettings
from .chunking import chunk_text

logger = logging.getLogger(__name__)


def _normalize_ws(name: str) -> str:
    return " ".join(name.split())


def _entity_key(name: str) -> str:
    return _normalize_ws(name).casefold()


class IngestTextUseCase:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        llm: LLMProvider,
        graph: GraphRepository,
        settings: RuntimeSettings,
    ) -> None:
        self._embeddings = embeddings
        self._llm = llm
        self._graph = graph
        self._settings = settings

    def execute(self, text: str, title: str | None = None) -> IngestionReport:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot ingest empty text.")

        document_id = uuid.uuid4().hex
        title = (title or "").strip() or f"Document {document_id[:8]}"

        raw_chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        logger.info("Ingest %s: %d chunks", document_id, len(raw_chunks))

        embeddings = self._embeddings.embed_documents(raw_chunks)
        chunks = [
            Chunk(
                id=f"{document_id}-{i}",
                document_id=document_id,
                index=i,
                text=chunk_text_value,
                embedding=embedding,
            )
            for i, (chunk_text_value, embedding) in enumerate(zip(raw_chunks, embeddings))
        ]

        extraction = self._extract(raw_chunks)
        entity_embeddings = self._embed_entities(extraction.entities)

        self._graph.save_document(document_id, title, chunks, extraction, entity_embeddings)

        report = IngestionReport(
            document_id=document_id,
            title=title,
            chunk_count=len(chunks),
            entity_count=len(extraction.entities),
            relationship_count=len(extraction.relationships),
        )
        logger.info(
            "Ingest %s done: %d chunks, %d entities, %d rels",
            document_id,
            report.chunk_count,
            report.entity_count,
            report.relationship_count,
        )
        return report

    def _extract(self, chunks: list[str]) -> ExtractionResult:
        """Extract a knowledge graph from every chunk and merge the results.

        Extracting per chunk (rather than from a single document prefix) means the
        whole document contributes to the graph, not just its opening. Entities are
        deduplicated case-insensitively, keeping the first display name and filling
        in a description when a later mention supplies one. A single chunk failing
        never aborts the rest, nor breaks ingestion.
        """
        if not self._settings.enable_entity_extraction:
            return ExtractionResult()

        limit = self._settings.max_extraction_chars
        entities: dict[str, Entity] = {}
        relationships: dict[tuple[str, str, str], Relationship] = {}

        for i, chunk in enumerate(chunks):
            try:
                result = self._llm.extract_graph(chunk[:limit])
            except Exception:  # extraction must never break ingestion
                logger.exception("Entity extraction failed for chunk %d; skipping it", i)
                continue

            for entity in result.entities:
                key = _entity_key(entity.name)
                if not key:
                    continue
                existing = entities.get(key)
                if existing is None:
                    entities[key] = Entity(
                        name=_normalize_ws(entity.name),
                        type=entity.type,
                        description=entity.description,
                    )
                elif not existing.description and entity.description:
                    entities[key] = Entity(
                        name=existing.name,
                        type=existing.type,
                        description=entity.description,
                    )

            for rel in result.relationships:
                source_key, target_key = _entity_key(rel.source), _entity_key(rel.target)
                if source_key not in entities or target_key not in entities:
                    continue
                relationships.setdefault(
                    (source_key, rel.type, target_key),
                    Relationship(
                        source=entities[source_key].name,
                        target=entities[target_key].name,
                        type=rel.type,
                        description=rel.description,
                    ),
                )

        return ExtractionResult(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
        )

    def _embed_entities(self, entities: list[Entity]) -> dict[str, list[float]]:
        if not entities:
            return {}
        texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
        vectors = self._embeddings.embed_documents(texts)
        return {entity.name: vector for entity, vector in zip(entities, vectors)}
