"""HTTP route tests using Flask's test client and a fake container."""


def test_health_returns_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_stats_returns_graph_counts(client):
    response = client.get("/api/stats")

    assert response.status_code == 200
    assert set(response.get_json()) == {"documents", "chunks", "entities", "relationships"}


def test_graph_clamps_limit_to_upper_bound(client, graph):
    client.get("/api/graph?limit=99999")

    assert graph.overview_limits[-1] == 1000


def test_graph_clamps_limit_to_lower_bound(client, graph):
    client.get("/api/graph?limit=-5")

    assert graph.overview_limits[-1] == 1


def test_ingest_document_returns_201_with_report(client, graph):
    response = client.post("/api/documents", json={"text": "Curie discovered radium.", "title": "C"})

    assert response.status_code == 201
    body = response.get_json()
    assert body["title"] == "C"
    assert body["chunk_count"] >= 1
    assert len(graph.saved_documents) == 1


def test_ingest_document_rejects_empty_text(client):
    response = client.post("/api/documents", json={"text": "   "})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_ingest_document_rejects_non_string_text(client):
    response = client.post("/api/documents", json={"text": 123})

    assert response.status_code == 400


def test_query_returns_answer(client):
    response = client.post("/api/query", json={"question": "What did Curie discover?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "Marie Curie discovered radium."
    assert "chunks" in body["context"]


def test_query_rejects_empty_question(client):
    response = client.post("/api/query", json={"question": ""})

    assert response.status_code == 400


def test_get_settings_returns_current_values(client):
    response = client.get("/api/settings")

    assert response.status_code == 200
    assert "chunk_size" in response.get_json()


def test_update_settings_applies_valid_change(client):
    response = client.put("/api/settings", json={"top_k": 8})

    assert response.status_code == 200
    assert response.get_json()["top_k"] == 8


def test_update_settings_rejects_invalid_value(client):
    response = client.put("/api/settings", json={"top_k": 0})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_reset_clears_the_graph(client, graph):
    response = client.post("/api/reset")

    assert response.status_code == 200
    assert response.get_json() == {"status": "reset"}
    assert graph.reset_count == 1
