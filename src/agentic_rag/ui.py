"""Pure helpers used by the Streamlit interface."""

import re
from pathlib import Path

from agentic_rag.loaders import UnsupportedDocumentError
from agentic_rag.models import Citation

SUPPORTED_UPLOAD_SUFFIXES = {".docx", ".md", ".pdf", ".txt"}
_SESSION_NAMESPACE_PATTERN = re.compile(r"[0-9a-f]{32}")


def session_data_directory(base_directory: Path, session_namespace: str) -> Path:
    """Return an isolated data directory for an application-generated session ID."""

    if _SESSION_NAMESPACE_PATTERN.fullmatch(session_namespace) is None:
        raise ValueError("Invalid session namespace")
    return base_directory / session_namespace


def save_uploaded_document(
    *,
    filename: str,
    content: bytes,
    upload_directory: Path,
) -> Path:
    """Validate and persist one uploaded document under its basename."""

    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("The uploaded document must have a filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
        raise UnsupportedDocumentError(f"Unsupported document type. Choose one of: {supported}")
    if not content:
        raise ValueError("The uploaded document is empty")

    upload_directory.mkdir(parents=True, exist_ok=True)
    path = upload_directory / safe_name
    path.write_bytes(content)
    return path


def format_citation(citation: Citation) -> str:
    """Render citation metadata for the source details panel."""

    page = f" · page {citation.page}" if citation.page is not None else ""
    return f"{citation.label} · {citation.source}{page} · relevance {citation.score:.2f}"
