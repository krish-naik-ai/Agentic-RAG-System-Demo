"""Document loaders for the supported source formats."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader

from agentic_rag.models import LoadedSection


class UnsupportedDocumentError(ValueError):
    """Raised when a document extension is not supported."""


def load_document(path: Path) -> list[LoadedSection]:
    """Load a PDF, DOCX, or plain-text document into source sections."""

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in {".txt", ".md"}:
        return _load_text(path)
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix or '<none>'}")


def _load_pdf(path: Path) -> list[LoadedSection]:
    reader = PdfReader(str(path))
    sections: list[LoadedSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(LoadedSection(text=text, source=path.name, page=page_number))
    return sections


def _load_docx(path: Path) -> list[LoadedSection]:
    document = Document(str(path))
    text = "\n\n".join(
        paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
    )
    return [LoadedSection(text=text, source=path.name)] if text else []


def _load_text(path: Path) -> list[LoadedSection]:
    text = path.read_text(encoding="utf-8").strip()
    return [LoadedSection(text=text, source=path.name)] if text else []
