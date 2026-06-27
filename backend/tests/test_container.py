"""Smoke test for dependency-injection wiring (no connections opened)."""

from app.config import Config
from app.container import Container


def test_container_wires_all_components_and_closes_cleanly():
    container = Container(Config())

    assert container.embeddings is not None
    assert container.llm is not None
    assert container.graph is not None
    assert container.settings is not None
    assert container.ingest_text._graph is container.graph
    assert container.answer_question._embeddings is container.embeddings

    container.close()
