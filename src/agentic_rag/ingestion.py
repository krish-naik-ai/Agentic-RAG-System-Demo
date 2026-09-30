"""Document ingestion orchestration."""

from pathlib import Path

from agentic_rag.chunking import chunk_sections
from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.ingestion_limits import (
    DEFAULT_INGESTION_LIMITS,
    IngestionLimitError,
    IngestionLimits,
)
from agentic_rag.loaders import load_document
from agentic_rag.models import IngestionResult
from agentic_rag.vector_store import VectorStore


class DocumentIngestor:
    """Loads, chunks, embeds, and persists supported documents."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
        limits: IngestionLimits = DEFAULT_INGESTION_LIMITS,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._limits = limits

    def ingest(self, path: Path) -> IngestionResult:
        sections = load_document(path, limits=self._limits)
        if len(sections) > self._limits.max_sections:
            raise IngestionLimitError(
                f"Document contains more than {self._limits.max_sections} sections."
            )
        extracted_characters = sum(len(section.text) for section in sections)
        if extracted_characters > self._limits.max_extracted_characters:
            raise IngestionLimitError(
                "Document text exceeds the safe ingestion limit. Upload a smaller document."
            )
        chunks = chunk_sections(
            sections,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            max_chunks=self._limits.max_chunks,
        )
        vectors = self._embeddings.embed([chunk.text for chunk in chunks])
        self._vector_store.replace(chunks, vectors)
        return IngestionResult(
            source=path.name,
            section_count=len(sections),
            chunk_count=len(chunks),
        )
