"""Unit tests for the full-text query sanitiser used by hybrid retrieval."""

from app.infrastructure.neo4j.repository import lucene_query


def test_strips_lucene_special_chars():
    assert lucene_query("What are the top 3 movies?") == "What are the top 3 movies"


def test_keeps_apostrophes_but_drops_reserved_symbols():
    assert lucene_query("Spielberg's best (rated) films!") == "Spielberg's best rated films"


def test_collapses_whitespace():
    assert lucene_query("  hello   world  ") == "hello world"


def test_symbols_only_returns_empty():
    assert lucene_query("") == ""
    assert lucene_query("?:*+-/\\") == ""
