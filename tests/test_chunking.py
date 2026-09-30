import pytest

from agentic_rag.chunking import chunk_sections
from agentic_rag.models import LoadedSection


def test_chunks_text_with_overlap_and_source_metadata() -> None:
    section = LoadedSection(
        text="alpha beta gamma delta epsilon zeta eta theta",
        source="notes.txt",
        page=2,
    )

    chunks = chunk_sections([section], chunk_size=24, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 24 for chunk in chunks)
    assert all(chunk.source == "notes.txt" and chunk.page == 2 for chunk in chunks)
    assert (
        chunks[0].chunk_id == chunk_sections([section], chunk_size=24, chunk_overlap=5)[0].chunk_id
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_rejects_invalid_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_sections([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
