"""Document loaders for the supported source formats."""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document
from pypdf import Configuration, PdfReader, apply_configuration
from pypdf.errors import LimitReachedError
from pypdf.generic import DictionaryObject, PdfObject, StreamObject

from agentic_rag.ingestion_limits import (
    DEFAULT_INGESTION_LIMITS,
    IngestionLimitError,
    IngestionLimits,
)
from agentic_rag.models import LoadedSection


class UnsupportedDocumentError(ValueError):
    """Raised when a document extension is not supported."""


def load_document(
    path: Path,
    *,
    limits: IngestionLimits = DEFAULT_INGESTION_LIMITS,
) -> list[LoadedSection]:
    """Load a PDF, DOCX, or plain-text document into source sections."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > limits.max_source_bytes:
        raise IngestionLimitError(
            f"Document exceeds the {limits.max_upload_mebibytes} MB upload limit."
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        sections = _load_pdf(path, limits)
    elif suffix == ".docx":
        sections = _load_docx(path, limits)
    elif suffix in {".txt", ".md"}:
        sections = _load_text(path, limits)
    elif suffix != ".pdf":
        raise UnsupportedDocumentError(f"Unsupported document type: {suffix or '<none>'}")

    if len(sections) > limits.max_sections:
        raise IngestionLimitError(
            f"Document contains more than {limits.max_sections} pages or sections."
        )
    return sections


def _load_pdf(path: Path, limits: IngestionLimits) -> list[LoadedSection]:
    configuration = Configuration(
        maximum_declared_stream_length=limits.max_pdf_stream_bytes,
        array_based_stream_maximum_output_length=limits.max_pdf_decompressed_bytes,
        jbig2_maximum_output_length=limits.max_pdf_stream_bytes,
        lzw_maximum_output_length=limits.max_pdf_stream_bytes,
        run_length_maximum_output_length=limits.max_pdf_stream_bytes,
        zlib_maximum_output_length=limits.max_pdf_stream_bytes,
        zlib_maximum_recovery_input_length=limits.max_pdf_stream_bytes,
        image_maximum_buffer_size=limits.max_pdf_stream_bytes,
        xmp_maximum_input_length=limits.max_pdf_stream_bytes,
        page_tree_maximum_entries=limits.max_sections,
        xform_maximum_invocations_per_extraction=limits.max_pdf_form_invocations,
    )
    try:
        with apply_configuration(configuration):
            reader = PdfReader(str(path))
            if len(reader.pages) > limits.max_sections:
                raise IngestionLimitError(
                    f"Document contains more than {limits.max_sections} pages."
                )

            sections: list[LoadedSection] = []
            extracted_characters = 0
            decompressed_bytes = 0
            seen_form_streams: set[int] = set()
            for page_number, page in enumerate(reader.pages, start=1):
                contents = page.get_contents()
                if contents is not None:
                    decompressed_bytes = _add_pdf_bytes(
                        decompressed_bytes,
                        len(contents.get_data()),
                        limits,
                    )
                resources = _resolve_dictionary(page.get("/Resources"))
                decompressed_bytes = _add_pdf_form_bytes(
                    resources,
                    seen_form_streams,
                    decompressed_bytes,
                    limits,
                )

                text = (page.extract_text() or "").strip()
                extracted_characters += len(text)
                if extracted_characters > limits.max_extracted_characters:
                    raise IngestionLimitError(
                        "Document text exceeds the safe ingestion limit. Upload a smaller document."
                    )
                if text:
                    sections.append(LoadedSection(text=text, source=path.name, page=page_number))
            return sections
    except LimitReachedError as error:
        raise IngestionLimitError(
            "PDF content expands beyond the safe processing limit."
        ) from error


def _load_docx(path: Path, limits: IngestionLimits) -> list[LoadedSection]:
    _validate_docx_archive(path, limits)
    document = Document(str(path))
    paragraphs: list[str] = []
    extracted_characters = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        extracted_characters += len(text) + (2 if paragraphs else 0)
        if extracted_characters > limits.max_extracted_characters:
            raise IngestionLimitError(
                "Document text exceeds the safe ingestion limit. Upload a smaller document."
            )
        paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    return [LoadedSection(text=text, source=path.name)] if text else []


def _load_text(path: Path, limits: IngestionLimits) -> list[LoadedSection]:
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > limits.max_extracted_characters:
        raise IngestionLimitError(
            "Document text exceeds the safe ingestion limit. Upload a smaller document."
        )
    return [LoadedSection(text=text, source=path.name)] if text else []


def _validate_docx_archive(path: Path, limits: IngestionLimits) -> None:
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > limits.max_docx_entries:
                raise IngestionLimitError(
                    "DOCX contains too many archived entries to process safely."
                )

            total_uncompressed_bytes = 0
            for entry in entries:
                total_uncompressed_bytes += entry.file_size
                if total_uncompressed_bytes > limits.max_docx_uncompressed_bytes:
                    raise IngestionLimitError("DOCX expands beyond the safe processing limit.")
                if (
                    entry.filename == "word/document.xml"
                    and entry.file_size > limits.max_docx_document_xml_bytes
                ):
                    raise IngestionLimitError(
                        "DOCX document text expands beyond the safe processing limit."
                    )
    except BadZipFile as error:
        raise ValueError("The uploaded DOCX file is invalid.") from error


def _resolve_dictionary(value: PdfObject | None) -> DictionaryObject | None:
    if value is None:
        return None
    resolved = value.get_object()
    return resolved if isinstance(resolved, DictionaryObject) else None


def _add_pdf_form_bytes(
    resources: DictionaryObject | None,
    seen_streams: set[int],
    current_bytes: int,
    limits: IngestionLimits,
) -> int:
    if resources is None:
        return current_bytes
    xobjects = _resolve_dictionary(resources.get("/XObject"))
    if xobjects is None:
        return current_bytes

    total_bytes = current_bytes
    for value in xobjects.values():
        resolved = value.get_object()
        if not isinstance(resolved, StreamObject) or resolved.get("/Subtype") != "/Form":
            continue
        stream_id = id(resolved)
        if stream_id in seen_streams:
            continue
        seen_streams.add(stream_id)
        total_bytes = _add_pdf_bytes(total_bytes, len(resolved.get_data()), limits)
        nested_resources = _resolve_dictionary(resolved.get("/Resources"))
        total_bytes = _add_pdf_form_bytes(
            nested_resources,
            seen_streams,
            total_bytes,
            limits,
        )
    return total_bytes


def _add_pdf_bytes(
    current_bytes: int,
    additional_bytes: int,
    limits: IngestionLimits,
) -> int:
    total_bytes = current_bytes + additional_bytes
    if total_bytes > limits.max_pdf_decompressed_bytes:
        raise IngestionLimitError("PDF content expands beyond the safe processing limit.")
    return total_bytes
