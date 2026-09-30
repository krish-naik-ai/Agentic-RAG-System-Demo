"""Document ingestion orchestration."""

from pathlib import Path

from agentic_rag.chunking import chunk_sections
from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.loaders import load_document
from agentic_rag.models import IngestionResult
from agentic_rag.vector_store import ChromaVectorStore


class DocumentIngestor:
    """Loads, chunks, embeds, and persists supported documents."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, path: Path) -> IngestionResult:
        sections = load_document(path)
        chunks = chunk_sections(
            sections,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        vectors = self._embeddings.embed([chunk.text for chunk in chunks])
        self._vector_store.replace(chunks, vectors)
        return IngestionResult(
            source=path.name,
            section_count=len(sections),
            chunk_count=len(chunks),
        )
