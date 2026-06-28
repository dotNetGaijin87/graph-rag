"""Ollama chat/LLM adapter."""

from __future__ import annotations

import json
import logging

import requests

from ...domain.models import Entity, ExtractionResult, Relationship
from ...domain.ports import LLMProvider
from ...application.prompts import EXTRACTION_SYSTEM, extraction_prompt

logger = logging.getLogger(__name__)

# JSON schema passed to Ollama's `format` field to force structured extraction output.
_EXTRACTION_FORMAT = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


class OllamaLLMProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        num_ctx: int = 4096,
        answer_temperature: float = 0.2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._answer_temperature = answer_temperature

    def generate(self, system: str, prompt: str) -> str:
        data = self._chat(
            system,
            prompt,
            options={"temperature": self._answer_temperature, "num_ctx": self._num_ctx},
        )
        return data.get("message", {}).get("content", "")

    def extract_graph(self, text: str) -> ExtractionResult:
        data = self._chat(
            EXTRACTION_SYSTEM,
            extraction_prompt(text),
            response_format=_EXTRACTION_FORMAT,
            options={"temperature": 0, "num_ctx": self._num_ctx},
        )
        content = data.get("message", {}).get("content", "{}")
        return self._parse_extraction(content)

    def _chat(
        self,
        system: str,
        prompt: str,
        response_format: dict | None = None,
        options: dict | None = None,
    ) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format
        if options is not None:
            payload["options"] = options

        response = requests.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_extraction(content: str) -> ExtractionResult:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Could not parse extraction JSON: %s", content[:200])
            return ExtractionResult()

        entities: list[Entity] = []
        valid_names: set[str] = set()
        for item in parsed.get("entities", []) or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            entities.append(
                Entity(
                    name=name,
                    type=(item.get("type") or "Concept").strip() or "Concept",
                    description=(item.get("description") or "").strip(),
                )
            )
            valid_names.add(name)

        relationships: list[Relationship] = []
        for item in parsed.get("relationships", []) or []:
            source = (item.get("source") or "").strip()
            target = (item.get("target") or "").strip()
            # Keep only relationships whose endpoints are real, extracted entities.
            if not source or not target or source not in valid_names or target not in valid_names:
                continue
            rel_type = (item.get("type") or "RELATED_TO").strip().upper().replace(" ", "_")
            relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    type=rel_type or "RELATED_TO",
                    description=(item.get("description") or "").strip(),
                )
            )

        return ExtractionResult(entities=entities, relationships=relationships)
