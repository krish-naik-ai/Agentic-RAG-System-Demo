from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_rag.ingestion import DocumentIngestor
from agentic_rag.ingestion_limits import DEFAULT_INGESTION_LIMITS, IngestionLimitError
from agentic_rag.vector_store import ChromaVectorStore
from tests.fakes import KeywordEmbeddingProvider


class UnexpectedEmbeddingProvider:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError(f"Embedding should not run for {len(texts)} oversized chunks")


def test_ingests_text_into_persistent_vector_store(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.txt"
    document_path.write_text(
        "Agentic retrieval decides when evidence is needed. " * 8,
        encoding="utf-8",
    )
    vector_store = ChromaVectorStore(tmp_path / "chroma")
    ingestor = DocumentIngestor(
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
        chunk_size=100,
        chunk_overlap=20,
    )

    result = ingestor.ingest(document_path)

    assert result.source == "notes.txt"
    assert result.section_count == 1
    assert result.chunk_count > 1
    assert vector_store.count() == result.chunk_count


def test_reingestion_replaces_existing_source_chunks(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.txt"
    vector_store = ChromaVectorStore(tmp_path / "chroma")
    ingestor = DocumentIngestor(
        embeddings=KeywordEmbeddingProvider(),
        vector_store=vector_store,
        chunk_size=80,
        chunk_overlap=10,
    )
    document_path.write_text("Agentic retrieval. " * 20, encoding="utf-8")
    first_result = ingestor.ingest(document_path)
    document_path.write_text("Agentic retrieval.", encoding="utf-8")

    second_result = ingestor.ingest(document_path)

    assert first_result.chunk_count > second_result.chunk_count
    assert vector_store.count() == second_result.chunk_count


def test_rejects_excessive_chunks_before_embedding(tmp_path: Path) -> None:
    document_path = tmp_path / "large.txt"
    document_path.write_text("word " * 100, encoding="utf-8")
    vector_store = ChromaVectorStore(tmp_path / "chroma")
    limits = replace(
        DEFAULT_INGESTION_LIMITS,
        max_chunks=2,
        max_extracted_characters=1_000,
    )
    ingestor = DocumentIngestor(
        embeddings=UnexpectedEmbeddingProvider(),
        vector_store=vector_store,
        chunk_size=20,
        chunk_overlap=0,
        limits=limits,
    )

    with pytest.raises(IngestionLimitError, match="more than 2 chunks"):
        ingestor.ingest(document_path)

    assert vector_store.count() == 0
