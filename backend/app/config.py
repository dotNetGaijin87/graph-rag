"""Application configuration, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password123")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    # Ollama's default context window is small and silently truncates long prompts.
    num_ctx: int = int(os.getenv("NUM_CTX", "4096"))
    # Near-greedy decoding keeps grounded answers factual (extraction always uses 0).
    answer_temperature: float = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    enable_entity_extraction: bool = _get_bool("ENABLE_ENTITY_EXTRACTION", True)
    max_extraction_chars: int = int(os.getenv("MAX_EXTRACTION_CHARS", "8000"))

    @staticmethod
    def from_env() -> "Config":
        return Config()
