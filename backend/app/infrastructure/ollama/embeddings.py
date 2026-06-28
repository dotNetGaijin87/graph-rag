"""Ollama embedding adapter."""

from __future__ import annotations

import logging

import requests

from ...domain.ports import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, timeout: int = 300) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": inputs},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embeddings: {data}")
        return embeddings
