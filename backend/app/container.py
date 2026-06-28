"""Dependency-injection wiring."""

from __future__ import annotations

from .application.answer_question import AnswerQuestionUseCase
from .application.ingest_text import IngestTextUseCase
from .config import Config
from .infrastructure.neo4j.repository import Neo4jGraphRepository
from .infrastructure.ollama.embeddings import OllamaEmbeddingProvider
from .infrastructure.ollama.llm import OllamaLLMProvider
from .settings import RuntimeSettings


class Container:
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
            num_ctx=config.num_ctx,
            answer_temperature=config.answer_temperature,
        )
        self.graph = Neo4jGraphRepository(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            embedding_dim=config.embedding_dim,
        )

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
