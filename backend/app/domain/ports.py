"""Ports — abstract interfaces the application layer depends on.

Following the Dependency Inversion Principle, the application/domain layers depend
on these abstractions, and the infrastructure layer (Ollama, Neo4j) provides the
concrete implementations. This keeps the core logic testable and swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import (
    Chunk,
    ExtractionResult,
    GraphFact,
    RetrievedChunk,
)


class EmbeddingProvider(ABC):
    """Turns text into dense vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class LLMProvider(ABC):
    """A chat/instruct language model."""

    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        """Generate a free-text completion."""

    @abstractmethod
    def extract_graph(self, text: str) -> ExtractionResult:
        """Extract entities and relationships as a knowledge graph."""


class GraphRepository(ABC):
    """Persistence port backed by the graph database (Neo4j)."""

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create constraints and vector/full-text indexes idempotently."""

    @abstractmethod
    def save_document(
        self,
        document_id: str,
        title: str,
        chunks: list[Chunk],
        extraction: ExtractionResult,
    ) -> None:
        """Persist a document, its chunks (with embeddings) and extracted graph."""

    @abstractmethod
    def search_chunks(
        self, query_text: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks (hybrid vector + full-text),
        enriched with their entities."""

    @abstractmethod
    def graph_facts_for_entities(self, entity_names: list[str], limit: int) -> list[GraphFact]:
        """Return relationships connected to the given entities (one-hop expansion)."""

    @abstractmethod
    def graph_overview(self, limit: int) -> dict:
        """Return entities and relationships for visualisation: {nodes, edges}."""

    @abstractmethod
    def stats(self) -> dict:
        """Return counts of documents/chunks/entities/relationships."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all data (used for demos / a clean slate)."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection/driver."""
