"""Interfaces for embeddings, the LLM, and graph persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Chunk, ExtractionResult, GraphFact, RetrievedChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str) -> str: ...

    @abstractmethod
    def extract_graph(self, text: str) -> ExtractionResult: ...


class GraphRepository(ABC):
    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def save_document(
        self,
        document_id: str,
        title: str,
        chunks: list[Chunk],
        extraction: ExtractionResult,
        entity_embeddings: dict[str, list[float]] | None = None,
    ) -> None: ...

    @abstractmethod
    def search_chunks(
        self, query_text: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def search_entities(self, query_embedding: list[float], k: int) -> list[str]: ...

    @abstractmethod
    def graph_facts_for_entities(self, entity_names: list[str], limit: int) -> list[GraphFact]: ...

    @abstractmethod
    def graph_overview(self, limit: int) -> dict: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
