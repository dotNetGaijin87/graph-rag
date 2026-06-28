"""Ingest text: chunk, embed, extract a graph, and store it."""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace

from ..domain.models import Chunk, Entity, ExtractionResult, IngestionReport, Relationship
from ..domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from ..settings import RuntimeSettings
from .chunking import chunk_text
from .prompts import SUMMARIZE_SYSTEM, summarize_prompt

logger = logging.getLogger(__name__)


def _normalize_ws(name: str) -> str:
    return " ".join(name.split())


def _entity_key(name: str) -> str:
    return _normalize_ws(name).casefold()


def _add_description(accumulated: list[str], description: str) -> None:
    description = (description or "").strip()
    if description and description not in accumulated:
        accumulated.append(description)


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
        deduplicated case-insensitively; the descriptions their mentions provide are
        accumulated and, when there is more than one, merged into a single coherent
        summary by the LLM. A chunk failing never aborts the rest, nor breaks ingestion.
        """
        if not self._settings.enable_entity_extraction:
            return ExtractionResult()

        limit = self._settings.max_extraction_chars
        entities: dict[str, Entity] = {}
        entity_descriptions: dict[str, list[str]] = {}
        relationships: dict[tuple[str, str, str], Relationship] = {}
        rel_descriptions: dict[tuple[str, str, str], list[str]] = {}

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
                if key not in entities:
                    entities[key] = Entity(name=_normalize_ws(entity.name), type=entity.type)
                    entity_descriptions[key] = []
                _add_description(entity_descriptions[key], entity.description)

            for rel in result.relationships:
                source_key, target_key = _entity_key(rel.source), _entity_key(rel.target)
                if source_key not in entities or target_key not in entities:
                    continue
                rel_key = (source_key, rel.type, target_key)
                if rel_key not in relationships:
                    relationships[rel_key] = Relationship(
                        source=entities[source_key].name,
                        target=entities[target_key].name,
                        type=rel.type,
                    )
                    rel_descriptions[rel_key] = []
                _add_description(rel_descriptions[rel_key], rel.description)

        return ExtractionResult(
            entities=[
                replace(entity, description=self._summarize(entity.name, entity_descriptions[key]))
                for key, entity in entities.items()
            ],
            relationships=[
                replace(
                    rel,
                    description=self._summarize(f"{rel.source} and {rel.target}", rel_descriptions[key]),
                )
                for key, rel in relationships.items()
            ],
        )

    def _summarize(self, subject: str, descriptions: list[str]) -> str:
        """Merge multiple descriptions of one entity/relationship into a single summary."""
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]
        try:
            summary = self._llm.generate(
                SUMMARIZE_SYSTEM, summarize_prompt(subject, descriptions)
            ).strip()
        except Exception:  # summarization is best-effort; never break ingestion
            logger.exception("Description summarization failed; keeping the first description")
            return descriptions[0]
        return summary or descriptions[0]

    def _embed_entities(self, entities: list[Entity]) -> dict[str, list[float]]:
        if not entities:
            return {}
        texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
        vectors = self._embeddings.embed_documents(texts)
        return {entity.name: vector for entity, vector in zip(entities, vectors)}
