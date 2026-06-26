"""Composition root — wires concrete adapters to use cases (dependency injection).

This is the only place that knows about both the application layer and the
infrastructure layer, keeping the wiring out of the framework and the core.
"""

from __future__ import annotations

from .application.answer_question import AnswerQuestionUseCase
from .application.ingest_text import IngestTextUseCase
from .config import Config
from .infrastructure.neo4j.repository import Neo4jGraphRepository
from .infrastructure.ollama.embeddings import OllamaEmbeddingProvider
from .infrastructure.ollama.llm import OllamaLLMProvider
from .settings import RuntimeSettings


class Container:
    """Builds and holds the application's wired dependencies."""

    def __init__(self, config: Config) -> None:
        self.config = config

        self.embeddings = OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model=config.embedding_model,
            timeout=config.ollama_timeout,
        )
        self.llm = OllamaLLMProvider(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            timeout=config.ollama_timeout,
        )
        self.graph = Neo4jGraphRepository(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            embedding_dim=config.embedding_dim,
        )

        # Runtime-adjustable tuning parameters (editable via /api/settings).
        self.settings = RuntimeSettings(config)

        self.ingest_text = IngestTextUseCase(
            embeddings=self.embeddings,
            llm=self.llm,
            graph=self.graph,
            settings=self.settings,
        )
        self.answer_question = AnswerQuestionUseCase(
            embeddings=self.embeddings,
            llm=self.llm,
            graph=self.graph,
            settings=self.settings,
        )

    def close(self) -> None:
        self.graph.close()
