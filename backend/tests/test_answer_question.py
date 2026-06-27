"""Unit tests for the question-answering use case."""

import pytest

from app.application.answer_question import AnswerQuestionUseCase
from app.config import Config
from app.domain.models import GraphFact, RetrievedChunk
from app.settings import RuntimeSettings

from tests.conftest import FakeEmbeddingProvider, FakeGraphRepository, FakeLLMProvider


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_empty_question_is_rejected(answer_uc, blank):
    with pytest.raises(ValueError):
        answer_uc.execute(question=blank)


def test_answers_from_retrieved_context(answer_uc, llm, graph):
    answer = answer_uc.execute(question="What did Curie discover?")

    assert answer.answer == "Marie Curie discovered radium."
    assert answer.question == "What did Curie discover?"
    assert len(answer.context.chunks) == 1
    assert len(answer.context.facts) == 1
    assert graph.facts_calls[-1]["names"] == ["Marie Curie"]


def test_returns_canned_reply_and_skips_llm_when_nothing_is_found():
    graph = FakeGraphRepository(chunks=[], facts=[])
    llm = FakeLLMProvider()
    use_case = AnswerQuestionUseCase(
        embeddings=FakeEmbeddingProvider(),
        llm=llm,
        graph=graph,
        settings=RuntimeSettings(Config()),
    )

    answer = use_case.execute(question="Something unknown?")

    assert "don't have any information" in answer.answer
    assert answer.context.chunks == []
    assert llm.generate_calls == []


def test_chunks_without_entities_skip_the_graph_fact_lookup():
    chunk = RetrievedChunk(
        chunk_id="doc-0", document_id="doc", text="Some text.", score=0.5, entities=[]
    )
    graph = FakeGraphRepository(chunks=[chunk], facts=[GraphFact("X", "Y", "Z")])
    use_case = AnswerQuestionUseCase(
        embeddings=FakeEmbeddingProvider(),
        llm=FakeLLMProvider(),
        graph=graph,
        settings=RuntimeSettings(Config()),
    )

    answer = use_case.execute(question="Anything?")

    assert answer.context.facts == []
    assert graph.facts_calls == []


def test_retrieval_uses_top_k_from_settings(graph):
    settings = RuntimeSettings(Config())
    settings.top_k = 9
    use_case = AnswerQuestionUseCase(
        embeddings=FakeEmbeddingProvider(),
        llm=FakeLLMProvider(),
        graph=graph,
        settings=settings,
    )

    use_case.execute(question="What did Curie discover?")

    assert graph.search_calls[-1]["k"] == 9
