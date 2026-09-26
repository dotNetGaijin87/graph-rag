"""Runtime-editable RAG settings (in-memory; reset to env defaults on restart)."""

from __future__ import annotations

from threading import RLock

from .config import Config

EDITABLE_FIELDS = (
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "enable_entity_extraction",
    "enable_reranking",
    "max_extraction_chars",
)


def _as_int(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool):  # bool is a subclass of int — reject it explicitly
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if not lo <= result <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return result


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


class RuntimeSettings:
    def __init__(self, config: Config) -> None:
        self._lock = RLock()
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.top_k = config.top_k
        self.enable_entity_extraction = config.enable_entity_extraction
        self.enable_reranking = config.enable_reranking
        self.max_extraction_chars = config.max_extraction_chars
        # Read-only (informational) — changing the embedding model/dim would invalidate
        # the existing vector index, so these are fixed at boot via env.
        self.llm_model = config.llm_model
        self.embedding_model = config.embedding_model
        self.embedding_dim = config.embedding_dim

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "enable_entity_extraction": self.enable_entity_extraction,
            "enable_reranking": self.enable_reranking,
            "max_extraction_chars": self.max_extraction_chars,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }

    def update(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        unknown = set(data) - set(EDITABLE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown or read-only field(s): {', '.join(sorted(unknown))}")

        pending: dict = {}
        if "chunk_size" in data:
            pending["chunk_size"] = _as_int(data["chunk_size"], "chunk_size", 50, 20000)
        if "chunk_overlap" in data:
            pending["chunk_overlap"] = _as_int(data["chunk_overlap"], "chunk_overlap", 0, 19999)
        if "top_k" in data:
            pending["top_k"] = _as_int(data["top_k"], "top_k", 1, 50)
        if "max_extraction_chars" in data:
            pending["max_extraction_chars"] = _as_int(
                data["max_extraction_chars"], "max_extraction_chars", 200, 200000
            )
        if "enable_entity_extraction" in data:
            pending["enable_entity_extraction"] = _as_bool(
                data["enable_entity_extraction"], "enable_entity_extraction"
            )
        if "enable_reranking" in data:
            pending["enable_reranking"] = _as_bool(data["enable_reranking"], "enable_reranking")

        # Check and apply under one lock so concurrent updates can't jointly break the
        # overlap < size invariant.
        with self._lock:
            chunk_size = pending.get("chunk_size", self.chunk_size)
            chunk_overlap = pending.get("chunk_overlap", self.chunk_overlap)
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size.")
            for key, value in pending.items():
                setattr(self, key, value)
            return self.to_dict()
