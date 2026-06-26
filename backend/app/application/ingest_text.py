"""Use case: ingest pasted text into the GraphRAG knowledge base.

Pipeline (baseline GraphRAG + entity extraction):
    text -> chunk -> embed chunks -> extract entities/relationships -> store in graph
"""

from __future__ import annotations

import logging
import uuid

from ..domain.models import Chunk, ExtractionResult, IngestionReport
from ..domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from ..settings import RuntimeSettings
from .chunking import chunk_text

logger = logging.getLogger(__name__)


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

        # 1. Chunk
        raw_chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        logger.info("Ingest %s: %d chunks", document_id, len(raw_chunks))

        # 2. Embed (batch)
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

        # 3. Extract knowledge graph (one LLM call over the document)
        extraction = self._extract(text)

        # 4. Persist everything in the graph database
        self._graph.save_document(document_id, title, chunks, extraction)

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

    def _extract(self, text: str) -> ExtractionResult:
        if not self._settings.enable_entity_extraction:
            return ExtractionResult()
        try:
            snippet = text[: self._settings.max_extraction_chars]
            return self._llm.extract_graph(snippet)
        except Exception:  # extraction must never break ingestion
            logger.exception("Entity extraction failed; falling back to vector-only RAG")
            return ExtractionResult()
