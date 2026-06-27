"""Unit tests for domain model behaviour (the only model with logic)."""

from app.domain.models import GraphFact


def test_as_sentence_renders_relationship_in_natural_language():
    fact = GraphFact(source="Marie Curie", type="WORKED_WITH", target="Pierre Curie")

    sentence = fact.as_sentence()

    assert sentence == "Marie Curie worked with Pierre Curie"


def test_as_sentence_appends_description_in_parentheses():
    fact = GraphFact("Marie Curie", "DISCOVERED", "Radium", description="in 1898")

    assert fact.as_sentence() == "Marie Curie discovered Radium (in 1898)"
