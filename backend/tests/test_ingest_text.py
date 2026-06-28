"""Unit tests for the ingestion use case."""

import pytest

from app.domain.models import Entity, ExtractionResult, Relationship
from app.settings import RuntimeSettings
from app.config import Config
from app.application.ingest_text import IngestTextUseCase
from app.domain.ports import LLMProvider

from tests.conftest import FakeEmbeddingProvider, FakeGraphRepository, FakeLLMProvider


class _SeqLLM(LLMProvider):
    """Returns a queued ExtractionResult per call (repeats the last)."""

    def __init__(self, results, summary="MERGED", generate_error=None):
        self._results = list(results)
        self._summary = summary
        self._generate_error = generate_error
        self.calls = 0
        self.generate_calls = []

    def generate(self, system, prompt):
        self.generate_calls.append((system, prompt))
        if self._generate_error is not None:
            raise self._generate_error
        return self._summary

    def extract_graph(self, text):
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


def _use_case(llm):
    return IngestTextUseCase(
        embeddings=FakeEmbeddingProvider(),
        llm=llm,
        graph=FakeGraphRepository(),
        settings=RuntimeSettings(Config()),
    )


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


def test_multiple_descriptions_are_merged_by_the_llm():
    first = ExtractionResult(entities=[Entity("Marie Curie", description="A physicist")])
    second = ExtractionResult(entities=[Entity("Marie Curie", description="Discovered radium")])
    llm = _SeqLLM([first, second], summary="Marie Curie, a physicist, discovered radium.")

    result = _use_case(llm)._extract(["chunk a", "chunk b"])

    assert result.entities[0].description == "Marie Curie, a physicist, discovered radium."
    assert len(llm.generate_calls) == 1


def test_relationship_descriptions_are_merged_across_chunks():
    entities = [Entity("Marie Curie"), Entity("Radium")]
    first = ExtractionResult(
        entities=entities,
        relationships=[Relationship("Marie Curie", "Radium", "DISCOVERED", "found it")],
    )
    second = ExtractionResult(
        entities=entities,
        relationships=[Relationship("Marie Curie", "Radium", "DISCOVERED", "isolated it in 1898")],
    )
    llm = _SeqLLM([first, second], summary="Marie Curie discovered and isolated radium in 1898.")

    result = _use_case(llm)._extract(["chunk a", "chunk b"])

    assert len(result.relationships) == 1
    assert result.relationships[0].description == "Marie Curie discovered and isolated radium in 1898."


def test_single_description_is_not_summarized():
    extraction = ExtractionResult(entities=[Entity("Marie Curie", description="A physicist")])
    llm = _SeqLLM([extraction])

    result = _use_case(llm)._extract(["only chunk"])

    assert result.entities[0].description == "A physicist"
    assert llm.generate_calls == []


def test_summarization_failure_keeps_the_first_description():
    first = ExtractionResult(entities=[Entity("X", description="d1")])
    second = ExtractionResult(entities=[Entity("X", description="d2")])
    llm = _SeqLLM([first, second], generate_error=RuntimeError("boom"))

    result = _use_case(llm)._extract(["a", "b"])

    assert result.entities[0].description == "d1"


def test_empty_summary_falls_back_to_first_description():
    first = ExtractionResult(entities=[Entity("X", description="d1")])
    second = ExtractionResult(entities=[Entity("X", description="d2")])
    llm = _SeqLLM([first, second], summary="   ")

    result = _use_case(llm)._extract(["a", "b"])

    assert result.entities[0].description == "d1"


def test_ingest_embeds_entities_for_local_search():
    llm = _SeqLLM([ExtractionResult(entities=[Entity("Marie Curie"), Entity("Radium")])])
    graph = FakeGraphRepository()
    use_case = IngestTextUseCase(
        embeddings=FakeEmbeddingProvider(),
        llm=llm,
        graph=graph,
        settings=RuntimeSettings(Config()),
    )

    use_case.execute(text="Marie Curie discovered radium.")

    assert set(graph.saved_documents[-1]["entity_embeddings"]) == {"Marie Curie", "Radium"}


def test_extraction_merges_case_and_whitespace_variants_across_chunks():
    first = ExtractionResult(entities=[Entity("NASA", description="")])
    second = ExtractionResult(
        entities=[Entity(" nasa ", type="Organization", description="Space agency")]
    )
    use_case = _use_case(_SeqLLM([first, second]))

    result = use_case._extract(["chunk one", "chunk two"])

    assert [e.name for e in result.entities] == ["NASA"]
    assert result.entities[0].description == "Space agency"


def test_extraction_merges_relationships_across_chunks():
    first = ExtractionResult(
        entities=[Entity("Ada Lovelace"), Entity("Analytical Engine")],
        relationships=[Relationship("Ada Lovelace", "Analytical Engine", "WROTE_FOR")],
    )
    second = ExtractionResult(
        entities=[Entity("Charles Babbage"), Entity("Analytical Engine")],
        relationships=[Relationship("Charles Babbage", "Analytical Engine", "DESIGNED")],
    )
    use_case = _use_case(_SeqLLM([first, second]))

    result = use_case._extract(["chunk one", "chunk two"])

    assert {e.name for e in result.entities} == {
        "Ada Lovelace",
        "Analytical Engine",
        "Charles Babbage",
    }
    assert {r.type for r in result.relationships} == {"WROTE_FOR", "DESIGNED"}


def test_extraction_skips_blank_entity_names():
    extraction = ExtractionResult(entities=[Entity("   "), Entity("Real")])

    result = _use_case(_SeqLLM([extraction]))._extract(["only chunk"])

    assert [e.name for e in result.entities] == ["Real"]


def test_extraction_drops_relationship_with_unknown_endpoint():
    extraction = ExtractionResult(
        entities=[Entity("A")],
        relationships=[Relationship("A", "Ghost", "KNOWS")],
    )

    result = _use_case(_SeqLLM([extraction]))._extract(["only chunk"])

    assert result.relationships == []


def test_extraction_continues_after_a_chunk_fails():
    class _FlakyLLM(LLMProvider):
        def __init__(self):
            self.calls = 0

        def generate(self, system, prompt):
            return ""

        def extract_graph(self, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return ExtractionResult(entities=[Entity("Survivor")])

    result = _use_case(_FlakyLLM())._extract(["bad chunk", "good chunk"])

    assert [e.name for e in result.entities] == ["Survivor"]
