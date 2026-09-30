"""Resource limits for document ingestion."""

from dataclasses import dataclass

MEBIBYTE = 1024 * 1024


class IngestionLimitError(ValueError):
    """Raised when a document exceeds a safe ingestion limit."""


@dataclass(frozen=True)
class IngestionLimits:
    """Upper bounds applied before expensive parsing and embedding work."""

    max_source_bytes: int = 10 * MEBIBYTE
    max_sections: int = 200
    max_extracted_characters: int = 400_000
    max_chunks: int = 512
    max_docx_entries: int = 1_000
    max_docx_uncompressed_bytes: int = 32 * MEBIBYTE
    max_docx_document_xml_bytes: int = 8 * MEBIBYTE
    max_pdf_stream_bytes: int = 4 * MEBIBYTE
    max_pdf_decompressed_bytes: int = 16 * MEBIBYTE
    max_pdf_form_invocations: int = 100

    def __post_init__(self) -> None:
        values = (
            self.max_source_bytes,
            self.max_sections,
            self.max_extracted_characters,
            self.max_chunks,
            self.max_docx_entries,
            self.max_docx_uncompressed_bytes,
            self.max_docx_document_xml_bytes,
            self.max_pdf_stream_bytes,
            self.max_pdf_decompressed_bytes,
            self.max_pdf_form_invocations,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Ingestion limits must be positive")
        if self.max_docx_document_xml_bytes > self.max_docx_uncompressed_bytes:
            raise ValueError("DOCX XML limit cannot exceed the archive limit")
        if self.max_pdf_stream_bytes > self.max_pdf_decompressed_bytes:
            raise ValueError("PDF stream limit cannot exceed the document limit")

    @property
    def max_upload_mebibytes(self) -> int:
        """Return the upload limit in whole MiB for Streamlit."""

        return max(1, (self.max_source_bytes + MEBIBYTE - 1) // MEBIBYTE)


DEFAULT_INGESTION_LIMITS = IngestionLimits()
