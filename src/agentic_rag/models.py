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


class RetrievedChunk(BaseModel):
    """A stored chunk returned by vector similarity search."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class RetrievalPlan(BaseModel):
    """The agent's decision about whether and how to retrieve evidence."""

    model_config = ConfigDict(frozen=True)

    needs_retrieval: bool
    query: str | None = None


class Citation(BaseModel):
    """Source metadata exposed with an agent answer."""

    model_config = ConfigDict(frozen=True)

    label: str
    source: str
    chunk_id: str
    page: int | None = None
    score: float


class AgentResponse(BaseModel):
    """Answer and retrieval trace returned to callers."""

    model_config = ConfigDict(frozen=True)

    answer: str
    retrieval_used: bool
    retrieval_query: str | None = None
    citations: list[Citation] = Field(default_factory=list)
