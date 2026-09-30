"""Non-agentic baselines used by the evaluation harness."""

from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.generation import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    citations_from_hits,
    ensure_citation_labels,
)
from agentic_rag.llm import LanguageModel
from agentic_rag.models import AgentResponse
from agentic_rag.vector_store import VectorStore


class NaiveRAG:
    """Single-shot retrieval that always searches with the original question."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        retrieval_limit: int = 5,
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be positive")
        self._language_model = language_model
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._retrieval_limit = retrieval_limit

    def answer(self, question: str) -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        query_vectors = self._embeddings.embed([question])
        if len(query_vectors) != 1:
            raise ValueError("The embedding provider must return one query embedding")
        hits = self._vector_store.query(query_vectors[0], limit=self._retrieval_limit)
        if not hits:
            return AgentResponse(
                answer="I could not find relevant evidence in the uploaded documents.",
                retrieval_used=True,
                retrieval_query=question,
            )

        citations = citations_from_hits(hits)
        answer = self._language_model.complete(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=build_answer_prompt(question, hits),
        )
        return AgentResponse(
            answer=ensure_citation_labels(answer, citations),
            retrieval_used=True,
            retrieval_query=question,
            citations=citations,
        )
