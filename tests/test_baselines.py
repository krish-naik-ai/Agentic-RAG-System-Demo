from agentic_rag.baselines import NaiveRAG
from agentic_rag.models import DocumentChunk, RetrievedChunk
from tests.fakes import KeywordEmbeddingProvider, ScriptedLanguageModel, StubVectorStore


def test_naive_rag_always_queries_with_original_question() -> None:
    language_model = ScriptedLanguageModel(["The answer is grounded [S1]."])
    vector_store = StubVectorStore(
        [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="chunk-1",
                    text="Grounded evidence.",
                    source="source.txt",
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]
    )
    baseline = NaiveRAG(
        language_model=language_model,
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
    )

    response = baseline.answer("What is grounded?")

    assert response.retrieval_query == "What is grounded?"
    assert response.citations[0].source == "source.txt"
    assert "[S1]" in response.answer


def test_naive_rag_rejects_hits_below_minimum_relevance() -> None:
    vector_store = StubVectorStore(
        [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="chunk-1",
                    text="Unrelated evidence.",
                    source="source.txt",
                    chunk_index=0,
                ),
                score=0.19,
            )
        ]
    )
    baseline = NaiveRAG(
        language_model=ScriptedLanguageModel([]),
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
    )

    response = baseline.answer("What was the company's revenue?")

    assert response.citations == []
    assert "could not find relevant evidence" in response.answer


def test_naive_rag_rejects_fabricated_source_label() -> None:
    vector_store = StubVectorStore(
        [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="chunk-1",
                    text="Grounded evidence.",
                    source="source.txt",
                    chunk_index=0,
                ),
                score=0.9,
            )
        ]
    )
    baseline = NaiveRAG(
        language_model=ScriptedLanguageModel(["Unsupported claim [S2]."]),
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
    )

    response = baseline.answer("What is grounded?")

    assert "citation-valid answer" in response.answer
    assert "[S2]" not in response.answer
    assert response.answer.endswith("Sources reviewed: [S1]")
