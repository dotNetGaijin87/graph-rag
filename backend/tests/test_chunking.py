"""Unit tests for the chunker — pure logic, no infrastructure required."""

from app.application.chunking import _overlap_tail, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_long_text_is_split_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, 50, 10)
    assert len(chunks) > 1
    # No chunk meaningfully exceeds the target size.
    assert all(len(c) <= 60 for c in chunks)


def test_chunks_never_split_words():
    text = "supercalifragilistic expialidocious antidisestablishmentarianism"
    for chunk in chunk_text(text, 20, 5):
        for token in chunk.split():
            assert token in text.split()


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", 10, 10)


def test_overlap_tail_keeps_all_words_when_they_fit_the_budget():
    tail, length = _overlap_tail(["a", "b", "c"], overlap=100)

    assert tail == ["a", "b", "c"]
    assert length == len("a b c")
