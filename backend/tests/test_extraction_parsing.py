"""Unit tests for LLM extraction JSON parsing (the pure static method)."""

from app.infrastructure.ollama.llm import OllamaLLMProvider


def test_parses_valid_extraction():
    content = """
    {
      "entities": [
        {"name": "Marie Curie", "type": "Person", "description": "Physicist"},
        {"name": "Radium", "type": "Concept", "description": "Element"}
      ],
      "relationships": [
        {"source": "Marie Curie", "target": "Radium", "type": "discovered", "description": ""}
      ]
    }
    """
    result = OllamaLLMProvider._parse_extraction(content)
    assert {e.name for e in result.entities} == {"Marie Curie", "Radium"}
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "DISCOVERED"  # normalised to UPPER_SNAKE


def test_drops_relationships_with_unknown_endpoints():
    content = """
    {
      "entities": [{"name": "A"}],
      "relationships": [{"source": "A", "target": "Ghost", "type": "knows"}]
    }
    """
    result = OllamaLLMProvider._parse_extraction(content)
    assert result.relationships == []


def test_invalid_json_returns_empty_result():
    result = OllamaLLMProvider._parse_extraction("not json at all")
    assert result.entities == []
    assert result.relationships == []


def test_entities_without_a_name_are_skipped():
    content = """
    {
      "entities": [
        {"name": "", "type": "Person"},
        {"type": "Concept"},
        {"name": "Radium"}
      ],
      "relationships": []
    }
    """
    result = OllamaLLMProvider._parse_extraction(content)
    assert [e.name for e in result.entities] == ["Radium"]
