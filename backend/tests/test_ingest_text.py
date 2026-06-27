"""Unit tests for the ingestion use case."""

import pytest

from app.domain.models import Entity, ExtractionResult, Relationship
from app.settings import RuntimeSettings
from app.config import Config
from app.application.ingest_text import IngestTextUseCase

from tests.conftest import FakeEmbeddingProvider, FakeGraphRepository, FakeLLMProvider


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(ingest_uc, blank):
    with pytest.raises(ValueError):
        ingest_uc.execute(text=blank)


def test_ingests_chunks_and_reports_extraction_counts(ingest_uc, graph):
    extraction = ExtractionResult(
        entities=[Entity("Marie Curie"), Entity("Radium")],
        relationships=[Relationship("Marie Curie", "Radium", "DISCOVERED")],
    )
    ingest_uc._llm._extraction = extraction

    report = ingest_uc.execute(text="Marie Curie discovered radium.", title="Curie")

    assert report.title == "Curie"
    assert report.chunk_count >= 1
    assert report.entity_count == 2
    assert report.relationship_count == 1

    saved = graph.saved_documents[-1]
    assert saved["title"] == "Curie"
    assert len(saved["chunks"]) == report.chunk_count
    assert saved["chunks"][0].id == f"{report.document_id}-0"
    assert saved["chunks"][0].embedding is not None


def test_blank_title_falls_back_to_generated_name(ingest_uc):
    report = ingest_uc.execute(text="Some content.", title="   ")

    assert report.title.startswith("Document ")
    assert report.document_id[:8] in report.title


def test_extraction_can_be_disabled(embeddings, llm, graph):
    settings = RuntimeSettings(Config())
    settings.enable_entity_extraction = False
    use_case = IngestTextUseCase(
        embeddings=embeddings, llm=llm, graph=graph, settings=settings
    )

    report = use_case.execute(text="Marie Curie discovered radium.")

    assert report.entity_count == 0
    assert report.relationship_count == 0
    assert llm.extract_calls == []


def test_extraction_failure_does_not_break_ingestion():
    llm = FakeLLMProvider(extract_error=RuntimeError("ollama down"))
    graph = FakeGraphRepository()
    settings = RuntimeSettings(Config())
    use_case = IngestTextUseCase(
        embeddings=FakeEmbeddingProvider(), llm=llm, graph=graph, settings=settings
    )

    report = use_case.execute(text="Marie Curie discovered radium.")

    assert report.chunk_count >= 1
    assert report.entity_count == 0
    assert len(graph.saved_documents) == 1
