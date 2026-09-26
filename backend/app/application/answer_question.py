"""Answer a question: retrieve chunks, expand the graph, and generate a grounded reply."""

from __future__ import annotations

import logging
import re

from ..domain.models import Answer, RetrievalContext, RetrievedChunk
from ..domain.ports import EmbeddingProvider, GraphRepository, LLMProvider
from ..settings import RuntimeSettings
from .prompts import ANSWER_SYSTEM, RERANK_SYSTEM, answer_prompt, rerank_prompt

logger = logging.getLogger(__name__)


def _parse_rank_order(text: str, n: int) -> list[int]:
    """Pull the passage numbers out of the LLM's ranking, as distinct 0-based indices."""
    order: list[int] = []
    for token in re.findall(r"\d+", text or ""):
        index = int(token) - 1
        if 0 <= index < n and index not in order:
            order.append(index)
    return order


class AnswerQuestionUseCase:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        llm: LLMProvider,
        graph: GraphRepository,
        settings: RuntimeSettings,
        *,
        max_facts: int = 25,
        rerank_overfetch: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._llm = llm
        self._graph = graph
        self._settings = settings
        self._max_facts = max_facts
        self._rerank_overfetch = rerank_overfetch

    def execute(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        # Must use the same embedding model as ingestion, or the vectors won't compare.
        query_embedding = self._embeddings.embed_query(question)

        if self._settings.enable_reranking:
            pool = self._graph.search_chunks(
                question, query_embedding, self._settings.top_k * self._rerank_overfetch
            )
            chunks = self._rerank(question, pool)
        else:
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

    def _rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Have the LLM reorder the candidate chunks by relevance, keeping top_k.

        Best-effort: if there is nothing to rerank, or the LLM fails or returns an
        unparseable ranking, fall back to the original retrieval order.
        """
        k = self._settings.top_k
        if len(candidates) <= k:
            return candidates[:k]
        try:
            ranking = self._llm.generate(
                RERANK_SYSTEM, rerank_prompt(question, [c.text for c in candidates])
            )
        except Exception:  # reranking is best-effort; never break answering
            logger.exception("Reranking failed; using retrieval order")
            return candidates[:k]

        order = _parse_rank_order(ranking, len(candidates))
        if not order:
            return candidates[:k]
        ranked_set = set(order)
        ranked = [candidates[i] for i in order]
        ranked += [c for j, c in enumerate(candidates) if j not in ranked_set]
        return ranked[:k]
