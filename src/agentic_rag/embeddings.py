"""Embedding provider interfaces and OpenAI implementation."""

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    """Converts text batches into numeric vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts while preserving input order."""


class OpenAIEmbeddingProvider:
    """OpenAI embeddings adapter."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
