from pathlib import Path

import pytest

from agentic_rag.loaders import UnsupportedDocumentError
from agentic_rag.models import Citation
from agentic_rag.ui import format_citation, save_uploaded_document


def test_save_uploaded_document_uses_safe_basename(tmp_path: Path) -> None:
    path = save_uploaded_document(
        filename="../../reference.txt",
        content=b"Agentic retrieval uses a planner.",
        upload_directory=tmp_path,
    )

    assert path == tmp_path / "reference.txt"
    assert path.read_bytes() == b"Agentic retrieval uses a planner."


def test_save_uploaded_document_rejects_unsupported_type(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDocumentError, match="Unsupported document type"):
        save_uploaded_document(
            filename="archive.zip",
            content=b"not a document",
            upload_directory=tmp_path,
        )


def test_save_uploaded_document_rejects_empty_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_uploaded_document(
            filename="empty.pdf",
            content=b"",
            upload_directory=tmp_path,
        )


def test_format_citation_includes_optional_page_and_score() -> None:
    citation = Citation(
        label="S1",
        source="guide.pdf",
        chunk_id="chunk-1",
        page=3,
        score=0.876,
    )

    assert format_citation(citation) == "S1 · guide.pdf · page 3 · relevance 0.88"
