"""Agentic retrieval planning and cited answer generation."""

from pydantic import ValidationError

from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.generation import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    citations_from_hits,
    ensure_citation_labels,
)
from agentic_rag.llm import LanguageModel
from agentic_rag.models import AgentResponse, RetrievalPlan
from agentic_rag.vector_store import VectorStore

_PLANNER_SYSTEM_PROMPT = """You are a retrieval planner.
Decide whether answering the user's message requires evidence from the uploaded documents.
Return only JSON matching:
{"needs_retrieval": true|false, "query": "concise standalone search query or null"}
Use retrieval for questions about document-specific facts, summaries, comparisons, or claims.
Do not retrieve for greetings, conversational acknowledgements, or general capability questions."""


class AgentDecisionError(ValueError):
    """Raised when the planner returns an invalid retrieval decision."""


class RetrievalAgent:
    """Plans retrieval, searches local evidence, and generates cited answers."""

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
        """Answer a user question, retrieving document evidence when planned."""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        plan = self._plan(question)
        if not plan.needs_retrieval:
            answer = self._language_model.complete(
                system_prompt=ANSWER_SYSTEM_PROMPT,
                user_prompt=question,
            )
            return AgentResponse(answer=answer, retrieval_used=False)

        query = (plan.query or question).strip()
        query_vectors = self._embeddings.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("The embedding provider must return one query embedding")
        hits = self._vector_store.query(query_vectors[0], limit=self._retrieval_limit)
        if not hits:
            return AgentResponse(
                answer="I could not find relevant evidence in the uploaded documents.",
                retrieval_used=True,
                retrieval_query=query,
            )

        citations = citations_from_hits(hits)
        answer = self._language_model.complete(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=build_answer_prompt(question, hits),
        )
        answer = ensure_citation_labels(answer, citations)
        return AgentResponse(
            answer=answer,
            retrieval_used=True,
            retrieval_query=query,
            citations=citations,
        )

    def _plan(self, question: str) -> RetrievalPlan:
        raw_plan = self._language_model.complete(
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            user_prompt=question,
        )
        try:
            plan = RetrievalPlan.model_validate_json(raw_plan)
        except ValidationError as error:
            raise AgentDecisionError("The retrieval planner returned invalid JSON") from error
        if plan.needs_retrieval and plan.query is not None and not plan.query.strip():
            raise AgentDecisionError("The retrieval query must not be blank")
        return plan
