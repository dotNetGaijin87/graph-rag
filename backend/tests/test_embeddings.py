"""Unit tests for the Ollama embedding adapter (requests mocked at the HTTP boundary)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.infrastructure.ollama.embeddings import OllamaEmbeddingProvider


def _response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _provider(base_url="http://ollama:11434"):
    return OllamaEmbeddingProvider(base_url=base_url, model="nomic-embed-text", timeout=30)


def test_embed_documents_short_circuits_on_empty_input():
    with patch("app.infrastructure.ollama.embeddings.requests.post") as post:
        result = _provider().embed_documents([])

    assert result == []
    post.assert_not_called()


def test_embed_documents_returns_vectors_from_the_api():
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    with patch(
        "app.infrastructure.ollama.embeddings.requests.post",
        return_value=_response({"embeddings": vectors}),
    ):
        result = _provider().embed_documents(["a", "b"])

    assert result == vectors


def test_embed_query_returns_the_single_vector():
    with patch(
        "app.infrastructure.ollama.embeddings.requests.post",
        return_value=_response({"embeddings": [[1.0, 2.0, 3.0]]}),
    ):
        result = _provider().embed_query("hello")

    assert result == [1.0, 2.0, 3.0]


def test_posts_to_the_embed_endpoint_without_double_slash():
    with patch(
        "app.infrastructure.ollama.embeddings.requests.post",
        return_value=_response({"embeddings": [[0.0]]}),
    ) as post:
        _provider(base_url="http://ollama:11434/").embed_query("hi")

    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "http://ollama:11434/api/embed"
    assert kwargs["json"] == {"model": "nomic-embed-text", "input": ["hi"]}
    assert kwargs["timeout"] == 30


def test_missing_embeddings_raises_runtime_error():
    with patch(
        "app.infrastructure.ollama.embeddings.requests.post",
        return_value=_response({"error": "no model"}),
    ):
        with pytest.raises(RuntimeError, match="no embeddings"):
            _provider().embed_query("hi")


def test_http_errors_propagate():
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("app.infrastructure.ollama.embeddings.requests.post", return_value=resp):
        with pytest.raises(requests.HTTPError):
            _provider().embed_query("hi")
