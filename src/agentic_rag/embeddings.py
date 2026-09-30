"""Embedding provider interfaces and OpenAI implementation."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from openai import OpenAI

DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_CONCURRENCY = 4
MAX_BATCH_SIZE = 2_048


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
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._model = model
        self._client = client or OpenAI()
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        text_list = list(texts)
        batches = [
            text_list[start : start + self._batch_size]
            for start in range(0, len(text_list), self._batch_size)
        ]
        if len(batches) == 1 or self._max_concurrency == 1:
            results = [self._embed_batch(batch) for batch in batches]
        else:
            worker_count = min(self._max_concurrency, len(batches))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(self._embed_batch, batches))
        return [embedding for result in results for embedding in result]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
