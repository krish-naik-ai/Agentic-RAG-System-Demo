from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document

from agentic_rag.loaders import UnsupportedDocumentError, load_document


def test_loads_text_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("  Agentic retrieval uses tools.  ", encoding="utf-8")

    sections = load_document(path)

    assert [section.text for section in sections] == ["Agentic retrieval uses tools."]
    assert sections[0].source == "notes.txt"


def test_loads_docx_file(tmp_path: Path) -> None:
    path = tmp_path / "guide.docx"
    document = Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.save(str(path))

    sections = load_document(path)

    assert sections[0].text == "First paragraph.\n\nSecond paragraph."
    assert sections[0].source == "guide.docx"


def test_loads_non_empty_pdf_pages(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.touch()
    first_page = MagicMock()
    first_page.extract_text.return_value = "Page one"
    empty_page = MagicMock()
    empty_page.extract_text.return_value = None

    with patch("agentic_rag.loaders.PdfReader") as reader:
        reader.return_value.pages = [first_page, empty_page]
        sections = load_document(path)

    assert len(sections) == 1
    assert sections[0].text == "Page one"
    assert sections[0].page == 1


def test_rejects_unsupported_document(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.touch()

    with pytest.raises(UnsupportedDocumentError, match="\\.zip"):
        load_document(path)
