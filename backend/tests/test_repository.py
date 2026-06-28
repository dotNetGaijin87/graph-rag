"""Unit tests for the Neo4j adapter (driver mocked at the boundary)."""

from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ServiceUnavailable

import app.infrastructure.neo4j.repository as repo_module
from app.domain.models import Chunk, Entity, ExtractionResult, Relationship
from app.infrastructure.neo4j.repository import Neo4jGraphRepository


@pytest.fixture
def driver():
    return MagicMock()


@pytest.fixture
def session(driver):
    """The session yielded by ``with driver.session() as session``."""
    sess = MagicMock()
    driver.session.return_value.__enter__.return_value = sess
    return sess


@pytest.fixture
def repo(monkeypatch, driver):
    monkeypatch.setattr(repo_module.GraphDatabase, "driver", lambda *a, **k: driver)
    return Neo4jGraphRepository(uri="bolt://x", user="u", password="p", embedding_dim=768)


# --- schema ---------------------------------------------------------------


def test_ensure_schema_creates_constraints_and_indexes(repo, driver, session):
    repo.ensure_schema()

    driver.verify_connectivity.assert_called_once()
    statements = " ".join(call.args[0] for call in session.run.call_args_list)
    assert "CREATE CONSTRAINT" in statements
    assert "VECTOR INDEX" in statements
    assert "768" in statements
    assert "FULLTEXT INDEX" in statements
    assert repo_module.ENTITY_VECTOR_INDEX in statements


def test_ensure_schema_retries_then_succeeds(repo, driver, session, monkeypatch):
    monkeypatch.setattr(repo_module.time, "sleep", lambda _s: None)
    driver.verify_connectivity.side_effect = [ServiceUnavailable("starting"), None]

    repo.ensure_schema(retries=5, delay=0)

    assert driver.verify_connectivity.call_count == 2


def test_ensure_schema_raises_after_exhausting_retries(repo, driver, monkeypatch):
    monkeypatch.setattr(repo_module.time, "sleep", lambda _s: None)
    driver.verify_connectivity.side_effect = ServiceUnavailable("never up")

    with pytest.raises(RuntimeError, match="not reachable"):
        repo.ensure_schema(retries=3, delay=0)


# --- writes ---------------------------------------------------------------


def _run_write(session):
    """Make ``execute_write`` invoke the unit-of-work with a fake tx and return it."""
    tx = MagicMock()
    session.execute_write.side_effect = lambda fn, *args, **kw: fn(tx, *args, **kw)
    return tx


def test_save_document_writes_chunks_entities_and_relationships(repo, session):
    tx = _run_write(session)
    chunks = [Chunk(id="doc-0", document_id="doc", index=0, text="t", embedding=[0.1])]
    extraction = ExtractionResult(
        entities=[Entity("Marie Curie"), Entity("Radium")],
        relationships=[Relationship("Marie Curie", "Radium", "DISCOVERED")],
    )

    repo.save_document("doc", "Curie", chunks, extraction)

    assert tx.run.call_count == 4


def test_save_document_writes_only_chunks_when_nothing_was_extracted(repo, session):
    tx = _run_write(session)
    chunks = [Chunk(id="doc-0", document_id="doc", index=0, text="t", embedding=[0.1])]

    repo.save_document("doc", "Curie", chunks, ExtractionResult())

    assert tx.run.call_count == 1


def test_save_document_carries_entity_embeddings(repo, session):
    tx = _run_write(session)
    chunks = [Chunk(id="doc-0", document_id="doc", index=0, text="t", embedding=[0.1])]
    extraction = ExtractionResult(entities=[Entity("Marie Curie")])

    repo.save_document("doc", "Curie", chunks, extraction, {"Marie Curie": [0.5, 0.6]})

    entity_write = next(c for c in tx.run.call_args_list if "entities" in c.kwargs)
    assert entity_write.kwargs["entities"][0]["embedding"] == [0.5, 0.6]


# --- retrieval ------------------------------------------------------------


def _chunk_record(entities):
    return {
        "chunk_id": "doc-0",
        "document_id": "doc",
        "text": "Marie Curie discovered radium.",
        "score": 0.8,
        "entities": entities,
    }


def test_search_chunks_uses_hybrid_query_when_keywords_exist(repo, session):
    session.run.return_value = [_chunk_record(["Marie Curie", None, ""])]

    results = repo.search_chunks("What did Curie discover?", [0.1, 0.2], k=5)

    assert session.run.call_args.args[0] == repo_module._HYBRID_CYPHER
    assert len(results) == 1
    assert results[0].chunk_id == "doc-0"
    assert results[0].score == 0.8
    assert results[0].entities == ["Marie Curie"]


def test_search_chunks_falls_back_to_vector_only_without_keywords(repo, session):
    session.run.return_value = [_chunk_record(["Radium"])]

    results = repo.search_chunks("?:*+-", [0.1, 0.2], k=5)

    assert session.run.call_args.args[0] == repo_module._VECTOR_ONLY_CYPHER
    assert results[0].entities == ["Radium"]


def test_search_entities_returns_matched_names(repo, session):
    session.run.return_value = [{"name": "Marie Curie"}, {"name": ""}]

    results = repo.search_entities([0.1, 0.2], k=5)

    assert session.run.call_args.args[0] == repo_module._ENTITY_SEARCH_CYPHER
    assert results == ["Marie Curie"]


def test_graph_facts_for_empty_names_returns_empty_without_querying(repo, session):
    assert repo.graph_facts_for_entities([], limit=10) == []
    session.run.assert_not_called()


def test_graph_facts_maps_records_and_coalesces_nulls(repo, session):
    session.run.return_value = [
        {"source": "Marie Curie", "type": None, "target": "Radium", "description": None}
    ]

    facts = repo.graph_facts_for_entities(["Marie Curie"], limit=10)

    assert len(facts) == 1
    assert facts[0].type == "RELATED_TO"
    assert facts[0].description == ""


def test_graph_overview_builds_unique_nodes_and_edges(repo, session):
    related = [
        {
            "source": "Marie Curie", "source_type": "Person", "source_desc": "",
            "target": "Radium", "target_type": "Concept", "target_desc": "",
            "type": "DISCOVERED", "description": "",
        }
    ]
    standalone = [{"id": "Bismuth", "type": "Concept", "description": ""}]
    session.run.side_effect = [related, standalone]

    overview = repo.graph_overview(limit=50)

    node_ids = {n["id"] for n in overview["nodes"]}
    assert node_ids == {"Marie Curie", "Radium", "Bismuth"}
    assert overview["edges"][0]["type"] == "DISCOVERED"


# --- stats / lifecycle ----------------------------------------------------


def test_stats_maps_the_single_record(repo, session):
    session.run.return_value.single.return_value = {
        "documents": 2, "chunks": 9, "entities": 4, "relationships": 3
    }

    assert repo.stats() == {"documents": 2, "chunks": 9, "entities": 4, "relationships": 3}


def test_stats_returns_zeros_when_there_is_no_record(repo, session):
    session.run.return_value.single.return_value = None

    assert repo.stats() == {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}


def test_reset_detach_deletes_everything(repo, session):
    repo.reset()

    assert "DETACH DELETE" in session.run.call_args.args[0]


def test_close_closes_the_driver(repo, driver):
    repo.close()

    driver.close.assert_called_once()
