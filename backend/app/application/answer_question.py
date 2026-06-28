"""Answer a question: retrieve chunks, expand the graph, and generate a grounded reply."""

from __future__ import annotations

import logging

from ..domain.models import Answer, RetrievalContext
from ..domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from ..settings import RuntimeSettings
from .prompts import ANSWER_SYSTEM, answer_prompt

logger = logging.getLogger(__name__)


class AnswerQuestionUseCase:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        llm: LLMProvider,
        graph: GraphRepository,
        settings: RuntimeSettings,
        *,
        max_facts: int = 25,
    ) -> None:
        self._embeddings = embeddings
        self._llm = llm
        self._graph = graph
        self._settings = settings
        self._max_facts = max_facts

    def execute(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        # Must use the same embedding model as ingestion, or the vectors won't compare.
        query_embedding = self._embeddings.embed_query(question)

        chunks = self._graph.search_chunks(question, query_embedding, self._settings.top_k)

        # "Local search": match entities directly to the query so the graph surfaces
        # relevant facts even when chunk retrieval misses the text.
        seed_entities = self._graph.search_entities(query_embedding, self._settings.top_k)
        entity_names = sorted(
            {name for chunk in chunks for name in chunk.entities} | set(seed_entities)
        )
        facts = (
            self._graph.graph_facts_for_entities(entity_names, self._max_facts)
            if entity_names
            else []
        )

        context = RetrievalContext(chunks=chunks, facts=facts)
        logger.info(
            "Query answered with %d chunks, %d entities, %d facts",
            len(chunks),
            len(entity_names),
            len(facts),
        )

        if not chunks and not facts:
            return Answer(
                question=question,
                answer=(
                    "I don't have any information about that yet. "
                    "Try adding some text to the knowledge base first."
                ),
                context=context,
            )

        answer_text = self._llm.generate(ANSWER_SYSTEM, answer_prompt(question, context))
        return Answer(question=question, answer=answer_text.strip(), context=context)
