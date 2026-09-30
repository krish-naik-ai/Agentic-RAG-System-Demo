"""Deterministic source-aware text chunking."""

from hashlib import sha256

from agentic_rag.models import DocumentChunk, LoadedSection


def chunk_sections(
    sections: list[LoadedSection],
    *,
    chunk_size: int = 1_000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Split loaded sections into overlapping chunks with stable identifiers."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for section in sections:
        normalized = " ".join(section.text.split())
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + (chunk_size // 2), end)
                if boundary > start:
                    end = boundary

            text = normalized[start:end].strip()
            if text:
                chunk_index = len(chunks)
                identifier = _chunk_id(
                    source=section.source,
                    page=section.page,
                    chunk_index=chunk_index,
                    text=text,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=identifier,
                        text=text,
                        source=section.source,
                        page=section.page,
                        chunk_index=chunk_index,
                    )
                )

            if end == len(normalized):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks


def _chunk_id(*, source: str, page: int | None, chunk_index: int, text: str) -> str:
    value = f"{source}:{page}:{chunk_index}:{text}".encode()
    return sha256(value).hexdigest()
