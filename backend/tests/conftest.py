"""Shared fake port adapters and fixtures for the application and API tests."""

from __future__ import annotations

import pytest

from app.application.answer_question import AnswerQuestionUseCase
from app.application.ingest_text import IngestTextUseCase
from app.config import Config
from app.domain.models import (
    ExtractionResult,
    GraphFact,
    RetrievedChunk,
)
from app.domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from app.settings import RuntimeSettings


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic stand-in for the Ollama embedder — no network."""

    def __init__(self) -> None:
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return [float(len(text)), 0.0, 1.0]


class FakeLLMProvider(LLMProvider):
    """Stand-in for the Ollama chat/LLM — returns canned, configurable output."""

    def __init__(
        self,
        *,
        answer: str = "  Marie Curie discovered radium.  ",
        extraction: ExtractionResult | None = None,
        extract_error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._extraction = extraction or ExtractionResult()
        self._extract_error = extract_error
        self.generate_calls: list[tuple[str, str]] = []
        self.extract_calls: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.generate_calls.append((system, prompt))
        return self._answer

    def extract_graph(self, text: str) -> ExtractionResult:
        self.extract_calls.append(text)
        if self._extract_error is not None:
            raise self._extract_error
        return self._extraction


class FakeGraphRepository(GraphRepository):
    """In-memory graph store. Records writes; returns configurable read results."""

    def __init__(
        self,
        *,
        chunks: list[RetrievedChunk] | None = None,
        facts: list[GraphFact] | None = None,
        seed_entities: list[str] | None = None,
    ) -> None:
        self._seed_entities = seed_entities or []
        self._chunks = chunks if chunks is not None else [
            RetrievedChunk(
                chunk_id="doc-0",
                document_id="doc",
                text="Marie Curie discovered radium in 1898.",
                score=0.97,
                entities=["Marie Curie"],
            )
        ]
        self._facts = facts if facts is not None else [
            GraphFact("Marie Curie", "DISCOVERED", "Radium", "in 1898")
        ]
        self.saved_documents: list[dict] = []
        self.search_calls: list[dict] = []
        self.entity_search_calls: list[dict] = []
        self.facts_calls: list[dict] = []
        self.overview_limits: list[int] = []
        self.reset_count = 0
        self.ensure_schema_count = 0

    def ensure_schema(self) -> None:
        self.ensure_schema_count += 1

    def save_document(self, document_id, title, chunks, extraction, entity_embeddings=None) -> None:
        self.saved_documents.append(
            {
                "document_id": document_id,
                "title": title,
                "chunks": chunks,
                "extraction": extraction,
                "entity_embeddings": entity_embeddings or {},
            }
        )

    def search_chunks(self, query_text, query_embedding, k) -> list[RetrievedChunk]:
        self.search_calls.append({"query": query_text, "embedding": query_embedding, "k": k})
        return list(self._chunks)

    def search_entities(self, query_embedding, k) -> list[str]:
        self.entity_search_calls.append({"embedding": query_embedding, "k": k})
        return list(self._seed_entities)

    def graph_facts_for_entities(self, entity_names, limit) -> list[GraphFact]:
        self.facts_calls.append({"names": entity_names, "limit": limit})
        return list(self._facts)

    def graph_overview(self, limit) -> dict:
        self.overview_limits.append(limit)
        return {"nodes": [{"id": "Marie Curie"}], "edges": []}

    def stats(self) -> dict:
        return {
            "documents": len(self.saved_documents),
            "chunks": sum(len(d["chunks"]) for d in self.saved_documents),
            "entities": 0,
            "relationships": 0,
        }

    def reset(self) -> None:
        self.reset_count += 1
        self.saved_documents.clear()

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def graph() -> FakeGraphRepository:
    return FakeGraphRepository()


@pytest.fixture
def settings() -> RuntimeSettings:
    return RuntimeSettings(Config())


@pytest.fixture
def ingest_uc(embeddings, llm, graph, settings) -> IngestTextUseCase:
    return IngestTextUseCase(embeddings=embeddings, llm=llm, graph=graph, settings=settings)


@pytest.fixture
def answer_uc(embeddings, llm, graph, settings) -> AnswerQuestionUseCase:
    return AnswerQuestionUseCase(embeddings=embeddings, llm=llm, graph=graph, settings=settings)


class FakeContainer:
    """Mirrors the real Container but wired to fake adapters."""

    def __init__(self, embeddings, llm, graph, settings) -> None:
        self.embeddings = embeddings
        self.llm = llm
        self.graph = graph
        self.settings = settings
        self.ingest_text = IngestTextUseCase(
            embeddings=embeddings, llm=llm, graph=graph, settings=settings
        )
        self.answer_question = AnswerQuestionUseCase(
            embeddings=embeddings, llm=llm, graph=graph, settings=settings
        )


@pytest.fixture
def fake_container(embeddings, llm, graph, settings) -> FakeContainer:
    return FakeContainer(embeddings, llm, graph, settings)


@pytest.fixture
def app(monkeypatch, fake_container):
    """Real Flask app from the factory, with Neo4j stubbed and a fake container."""
    from app import create_app

    monkeypatch.setattr(
        "app.infrastructure.neo4j.repository.Neo4jGraphRepository.ensure_schema",
        lambda self, *a, **k: None,
    )
    application = create_app(Config())
    # Close the real (unused) driver before swapping in the fake, to avoid a GC warning.
    application.config["CONTAINER"].graph.close()
    application.config["CONTAINER"] = fake_container
    return application


@pytest.fixture
def client(app):
    return app.test_client()
