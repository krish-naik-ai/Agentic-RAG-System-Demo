from pathlib import Path

import pytest

from agentic_rag.models import DocumentChunk
from agentic_rag.vector_store import ChromaVectorStore
from tests.fakes import KeywordEmbeddingProvider


def test_queries_most_similar_chunk(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="retrieval",
            text="Agents decide when retrieval is required.",
            source="rag.txt",
            chunk_index=0,
            page=1,
        ),
        DocumentChunk(
            chunk_id="cooking",
            text="This cooking recipe uses tomatoes.",
            source="food.txt",
            chunk_index=1,
        ),
    ]
    embeddings = KeywordEmbeddingProvider()
    vector_store = ChromaVectorStore(tmp_path / "chroma")
    vector_store.replace(chunks, embeddings.embed([chunk.text for chunk in chunks]))

    results = vector_store.query(embeddings.embed(["retrieval question"])[0], limit=2)

    assert results[0].chunk.chunk_id == "retrieval"
    assert results[0].chunk.page == 1
    assert results[0].score > results[1].score


def test_rejects_embedding_count_mismatch(tmp_path: Path) -> None:
    vector_store = ChromaVectorStore(tmp_path / "chroma")
    chunk = DocumentChunk(
        chunk_id="one",
        text="text",
        source="notes.txt",
        chunk_index=0,
    )

    with pytest.raises(ValueError, match="exactly one embedding"):
        vector_store.replace([chunk], [])


def test_separate_persist_paths_do_not_share_chunks(tmp_path: Path) -> None:
    embeddings = KeywordEmbeddingProvider()
    first_store = ChromaVectorStore(tmp_path / "sessions" / "first")
    second_store = ChromaVectorStore(tmp_path / "sessions" / "second")
    chunk = DocumentChunk(
        chunk_id="private",
        text="Session-private retrieval evidence.",
        source="private.txt",
        chunk_index=0,
    )

    first_store.replace([chunk], embeddings.embed([chunk.text]))

    assert first_store.count() == 1
    assert second_store.count() == 0
