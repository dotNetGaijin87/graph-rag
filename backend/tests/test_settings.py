"""Unit tests for runtime settings validation."""

import pytest

from app.config import Config
from app.settings import RuntimeSettings


def _settings() -> RuntimeSettings:
    return RuntimeSettings(Config())


def test_to_dict_includes_editable_and_readonly():
    data = _settings().to_dict()
    assert "chunk_size" in data
    assert "llm_model" in data
    assert "embedding_dim" in data


def test_valid_update_applies():
    s = _settings()
    out = s.update({"chunk_size": 1000, "top_k": 7})
    assert s.chunk_size == 1000
    assert s.top_k == 7
    assert out["chunk_size"] == 1000


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        _settings().update({"chunk_size": 500, "chunk_overlap": 500})


def test_rejects_unknown_or_readonly_field():
    with pytest.raises(ValueError):
        _settings().update({"embedding_model": "other"})


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        _settings().update({"top_k": 0})


def test_rejects_non_bool_extraction_flag():
    with pytest.raises(ValueError):
        _settings().update({"enable_entity_extraction": "yes"})


def test_partial_update_leaves_others_unchanged():
    s = _settings()
    original_top_k = s.top_k
    s.update({"chunk_size": 1234})
    assert s.chunk_size == 1234
    assert s.top_k == original_top_k
