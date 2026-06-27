"""Unit tests for the Ollama chat/LLM adapter's HTTP behaviour (requests mocked)."""

from unittest.mock import MagicMock, patch

from app.infrastructure.ollama.llm import OllamaLLMProvider


def _response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _provider():
    return OllamaLLMProvider(base_url="http://ollama:11434", model="llama3.2", timeout=30)


def test_generate_returns_message_content():
    with patch(
        "app.infrastructure.ollama.llm.requests.post",
        return_value=_response({"message": {"content": "The answer is 42."}}),
    ):
        result = _provider().generate("system", "prompt")

    assert result == "The answer is 42."


def test_generate_returns_empty_string_when_content_is_missing():
    with patch(
        "app.infrastructure.ollama.llm.requests.post",
        return_value=_response({}),
    ):
        assert _provider().generate("system", "prompt") == ""


def test_generate_payload_omits_format_and_options():
    with patch(
        "app.infrastructure.ollama.llm.requests.post",
        return_value=_response({"message": {"content": "ok"}}),
    ) as post:
        _provider().generate("sys", "hello")

    url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url == "http://ollama:11434/api/chat"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "format" not in payload
    assert "options" not in payload


def test_extract_graph_requests_structured_output_and_parses_it():
    content = """
    {
      "entities": [{"name": "Marie Curie"}, {"name": "Radium"}],
      "relationships": [{"source": "Marie Curie", "target": "Radium", "type": "discovered"}]
    }
    """
    with patch(
        "app.infrastructure.ollama.llm.requests.post",
        return_value=_response({"message": {"content": content}}),
    ) as post:
        result = _provider().extract_graph("Marie Curie discovered radium.")

    payload = post.call_args.kwargs["json"]
    assert "format" in payload
    assert payload["options"] == {"temperature": 0}

    assert {e.name for e in result.entities} == {"Marie Curie", "Radium"}
    assert result.relationships[0].type == "DISCOVERED"


def test_extract_graph_handles_missing_content_gracefully():
    with patch(
        "app.infrastructure.ollama.llm.requests.post",
        return_value=_response({}),
    ):
        result = _provider().extract_graph("anything")

    assert result.entities == []
    assert result.relationships == []
