"""Unit tests for prompt building — assert content is included, not exact wording."""

from app.application.prompts import answer_prompt, extraction_prompt, summarize_prompt
from app.domain.models import GraphFact, RetrievalContext, RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c", document_id="d", text=text, score=1.0)


def test_extraction_prompt_embeds_the_source_text():
    prompt = extraction_prompt("Marie Curie won two Nobel Prizes.")

    assert "Marie Curie won two Nobel Prizes." in prompt


def test_answer_prompt_includes_facts_passages_and_question():
    context = RetrievalContext(
        chunks=[_chunk("Radium is radioactive.")],
        facts=[GraphFact("Marie Curie", "DISCOVERED", "Radium")],
    )

    prompt = answer_prompt("What did Curie discover?", context)

    assert "Marie Curie discovered Radium" in prompt
    assert "Radium is radioactive." in prompt
    assert "What did Curie discover?" in prompt


def test_answer_prompt_uses_none_placeholder_when_context_is_empty():
    prompt = answer_prompt("Anything?", RetrievalContext(chunks=[], facts=[]))

    assert prompt.count("(none)") == 2


def test_summarize_prompt_includes_subject_and_descriptions():
    prompt = summarize_prompt("Marie Curie", ["A physicist", "Discovered radium"])

    assert "Marie Curie" in prompt
    assert "A physicist" in prompt
    assert "Discovered radium" in prompt
