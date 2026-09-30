import pytest

from agentic_rag.agent import AgentDecisionError, RetrievalAgent
from agentic_rag.models import DocumentChunk, RetrievedChunk
from tests.fakes import (
    KeywordEmbeddingProvider,
    ReverseReranker,
    ScriptedLanguageModel,
    StubVectorStore,
)


def _retrieval_hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="chunk-1",
            text="Agentic RAG decides whether retrieval is needed before searching.",
            source="guide.pdf",
            page=3,
            chunk_index=0,
        ),
        score=0.93,
    )


def test_plans_retrieval_and_returns_cited_answer() -> None:
    language_model = ScriptedLanguageModel(
        [
            '{"needs_retrieval": true, "query": "agentic retrieval decision"}',
            "The system plans retrieval before searching [S1].",
        ]
    )
    vector_store = StubVectorStore([_retrieval_hit()])
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
        retrieval_limit=3,
    )

    response = agent.answer("How does the system use retrieval?")

    assert response.retrieval_used is True
    assert response.retrieval_query == "agentic retrieval decision"
    assert response.citations[0].source == "guide.pdf"
    assert response.citations[0].page == 3
    assert "[S1]" in response.answer
    assert vector_store.last_limit == 3
    assert "[S1] guide.pdf, page 3" in language_model.prompts[1][1]


def test_reranks_a_wider_candidate_set_before_generation() -> None:
    hits = [
        _retrieval_hit(),
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="chunk-2",
                text="Vector retrieval finds an initial candidate set.",
                source="retrieval.txt",
                chunk_index=0,
            ),
            score=0.88,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="chunk-3",
                text="A cross-encoder reranks candidates before generation.",
                source="reranking.txt",
                chunk_index=0,
            ),
            score=0.82,
        ),
    ]
    language_model = ScriptedLanguageModel(
        [
            '{"needs_retrieval": true, "query": "reranking pipeline"}',
            "The cross-encoder reranks the candidate set [S1].",
        ]
    )
    vector_store = StubVectorStore(hits)
    reranker = ReverseReranker()
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
        reranker=reranker,
        retrieval_limit=2,
        reranking_candidate_limit=4,
    )

    response = agent.answer("What happens after vector retrieval?")

    assert vector_store.last_limit == 4
    assert reranker.last_query == "reranking pipeline"
    assert reranker.last_hits == hits
    assert reranker.last_limit == 2
    assert [citation.chunk_id for citation in response.citations] == ["chunk-3", "chunk-2"]
    assert "[S1] reranking.txt" in language_model.prompts[1][1]


def test_skips_retrieval_for_conversational_question() -> None:
    language_model = ScriptedLanguageModel(
        [
            '{"needs_retrieval": false, "query": null}',
            "Hello! Ask me about an uploaded document.",
        ]
    )
    vector_store = StubVectorStore([])
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
    )

    response = agent.answer("Hello")

    assert response.retrieval_used is False
    assert response.citations == []
    assert len(language_model.prompts) == 2
    assert vector_store.last_embedding is None
    assert "Available document chunks: 0" in language_model.prompts[0][1]


def test_reports_missing_evidence_without_generation_call() -> None:
    language_model = ScriptedLanguageModel(['{"needs_retrieval": true, "query": "missing topic"}'])
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=StubVectorStore([]),
    )

    response = agent.answer("What does the document say about the missing topic?")

    assert response.retrieval_used is True
    assert response.citations == []
    assert "could not find relevant evidence" in response.answer
    assert len(language_model.prompts) == 1


def test_adds_source_labels_when_model_omits_them() -> None:
    language_model = ScriptedLanguageModel(
        [
            '{"needs_retrieval": true, "query": "agentic retrieval"}',
            "The system decides before searching.",
        ]
    )
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=StubVectorStore([_retrieval_hit()]),
    )

    response = agent.answer("How does retrieval work?")

    assert response.answer.endswith("Sources: [S1]")


def test_removes_source_labels_from_no_retrieval_answer() -> None:
    agent = RetrievalAgent(
        language_model=ScriptedLanguageModel(
            [
                '{"needs_retrieval": false, "query": null}',
                "Hello! [S1]",
            ]
        ),
        embeddings=KeywordEmbeddingProvider(),
        vector_store=StubVectorStore([]),
    )

    response = agent.answer("Hello")

    assert response.answer == "Hello!"
    assert response.citations == []


def test_rejects_hits_below_minimum_relevance() -> None:
    weak_hit = _retrieval_hit().model_copy(update={"score": 0.19})
    language_model = ScriptedLanguageModel(
        ['{"needs_retrieval": true, "query": "company revenue"}']
    )
    agent = RetrievalAgent(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=StubVectorStore([weak_hit]),
    )

    response = agent.answer("What was the company's revenue?")

    assert response.citations == []
    assert "could not find relevant evidence" in response.answer
    assert len(language_model.prompts) == 1


def test_rejects_invalid_planner_output() -> None:
    agent = RetrievalAgent(
        language_model=ScriptedLanguageModel(["retrieve probably"]),
        embeddings=KeywordEmbeddingProvider(),
        vector_store=StubVectorStore([]),
    )

    with pytest.raises(AgentDecisionError, match="invalid JSON"):
        agent.answer("Summarize the document")


def test_rejects_candidate_limit_smaller_than_answer_limit() -> None:
    with pytest.raises(ValueError, match="at least retrieval_limit"):
        RetrievalAgent(
            language_model=ScriptedLanguageModel([]),
            embeddings=KeywordEmbeddingProvider(),
            vector_store=StubVectorStore([]),
            reranker=ReverseReranker(),
            retrieval_limit=5,
            reranking_candidate_limit=4,
        )
