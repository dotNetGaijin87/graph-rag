"""Tests for the app factory's error handlers: ValueError->400, RequestException->502, Exception->500."""

import requests


def test_app_factory_wires_blueprint_and_serves_requests(client):
    assert client.get("/api/health").status_code == 200


def test_value_error_is_rendered_as_400(client, fake_container, monkeypatch):
    monkeypatch.setattr(
        fake_container.answer_question,
        "execute",
        lambda question: (_ for _ in ()).throw(ValueError("bad input")),
    )

    response = client.post("/api/query", json={"question": "anything"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad input"


def test_upstream_request_exception_is_rendered_as_502(client, fake_container, monkeypatch):
    monkeypatch.setattr(
        fake_container.answer_question,
        "execute",
        lambda question: (_ for _ in ()).throw(requests.ConnectionError("ollama down")),
    )

    response = client.post("/api/query", json={"question": "anything"})

    assert response.status_code == 502
    assert "language model service is unavailable" in response.get_json()["error"]


def test_unhandled_exception_is_rendered_as_500(client, fake_container, monkeypatch):
    monkeypatch.setattr(
        fake_container.answer_question,
        "execute",
        lambda question: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post("/api/query", json={"question": "anything"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal server error."
