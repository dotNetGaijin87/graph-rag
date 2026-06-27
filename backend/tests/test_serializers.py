"""Unit tests for the HTTP serialisers (domain -> JSON-friendly dicts)."""

from app.api.serializers import serialize_answer, serialize_report
from app.domain.models import (
    Answer,
    GraphFact,
    IngestionReport,
    RetrievalContext,
    RetrievedChunk,
)


def test_serialize_report_maps_all_counts():
    report = IngestionReport(
        document_id="abc123",
        title="Curie",
        chunk_count=3,
        entity_count=5,
        relationship_count=2,
    )

    assert serialize_report(report) == {
        "document_id": "abc123",
        "title": "Curie",
        "chunk_count": 3,
        "entity_count": 5,
        "relationship_count": 2,
    }


def test_serialize_answer_rounds_scores_and_renders_fact_sentences():
    answer = Answer(
        question="What did Curie discover?",
        answer="Radium.",
        context=RetrievalContext(
            chunks=[
                RetrievedChunk(
                    chunk_id="doc-0",
                    document_id="doc",
                    text="Radium is radioactive.",
                    score=0.123456,
                    entities=["Radium"],
                )
            ],
            facts=[GraphFact("Marie Curie", "DISCOVERED", "Radium", "in 1898")],
        ),
    )

    data = serialize_answer(answer)

    assert data["question"] == "What did Curie discover?"
    assert data["answer"] == "Radium."
    assert data["context"]["chunks"][0]["score"] == 0.1235
    assert data["context"]["chunks"][0]["entities"] == ["Radium"]
    fact = data["context"]["facts"][0]
    assert fact["source"] == "Marie Curie"
    assert fact["sentence"] == "Marie Curie discovered Radium (in 1898)"
