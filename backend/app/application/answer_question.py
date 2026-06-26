"""Use case: answer a question with GraphRAG retrieval.

Pipeline:
    question -> embed -> vector search chunks -> expand graph (entities -> facts)
    -> assemble context -> LLM generates grounded answer
"""

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

        # 1. Embed the query with the SAME model used at ingest time.
        query_embedding = self._embeddings.embed_query(question)

        # 2. Vector search for the most relevant chunks.
        chunks = self._graph.vector_search(query_embedding, self._settings.top_k)

        # 3. Graph expansion: pull relationships around the entities those chunks mention.
        entity_names = sorted({name for chunk in chunks for name in chunk.entities})
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

        # 4. Generate a grounded answer.
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
