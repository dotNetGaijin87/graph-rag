"""HTTP routes — the interface/delivery layer.

Thin controllers: parse the request, call a use case, serialize the result.
No business logic lives here.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from .serializers import serialize_answer, serialize_report

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)


def _container():
    return current_app.config["CONTAINER"]


@api.get("/health")
def health():
    return jsonify({"status": "ok"})


@api.get("/stats")
def stats():
    return jsonify(_container().graph.stats())


@api.get("/graph")
def graph():
    limit = request.args.get("limit", default=200, type=int)
    limit = max(1, min(limit, 1000))
    return jsonify(_container().graph.graph_overview(limit))


@api.post("/documents")
def ingest_document():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    title = payload.get("title")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400

    report = _container().ingest_text.execute(text=text, title=title)
    return jsonify(serialize_report(report)), 201


@api.post("/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Field 'question' is required and must be non-empty."}), 400

    answer = _container().answer_question.execute(question=question)
    return jsonify(serialize_answer(answer))


@api.get("/settings")
def get_settings():
    return jsonify(_container().settings.to_dict())


@api.put("/settings")
def update_settings():
    payload = request.get_json(silent=True) or {}
    updated = _container().settings.update(payload)  # raises ValueError -> 400
    return jsonify(updated)


@api.post("/reset")
def reset():
    _container().graph.reset()
    return jsonify({"status": "reset"})
