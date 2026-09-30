"""Shared domain models."""

from pydantic import BaseModel, ConfigDict, Field


class LoadedSection(BaseModel):
    """Text extracted from one logical section of a source document."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class DocumentChunk(BaseModel):
    """A source-backed text chunk ready for embedding."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)


class IngestionResult(BaseModel):
    """Summary returned after a document is stored."""

    model_config = ConfigDict(frozen=True)

    source: str
    section_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
