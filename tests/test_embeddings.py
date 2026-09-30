import threading
import time
from dataclasses import dataclass
from typing import cast

import pytest
from openai import OpenAI

from agentic_rag.embeddings import OpenAIEmbeddingProvider


@dataclass(frozen=True)
class FakeEmbedding:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class FakeEmbeddingResponse:
    data: list[FakeEmbedding]


class FakeEmbeddingsAPI:
    def __init__(self, delay: float = 0.0) -> None:
        self.inputs: list[list[str]] = []
        self._delay = delay
        self._lock = threading.Lock()
        self._active_requests = 0
        self.max_active_requests = 0

    def create(self, *, model: str, input: list[str]) -> FakeEmbeddingResponse:
        with self._lock:
            self.inputs.append(input.copy())
            self._active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self._active_requests,
            )
        time.sleep(self._delay)
        with self._lock:
            self._active_requests -= 1
        embeddings = [
            FakeEmbedding(index=index, embedding=[float(len(text))])
            for index, text in enumerate(input)
        ]
        return FakeEmbeddingResponse(data=list(reversed(embeddings)))


class FakeOpenAI:
    def __init__(self, delay: float = 0.0) -> None:
        self.embeddings = FakeEmbeddingsAPI(delay)


def make_provider(
    client: FakeOpenAI,
    *,
    batch_size: int = 256,
    max_concurrency: int = 4,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        client=cast(OpenAI, client),
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )


def test_embed_batches_requests_and_preserves_order() -> None:
    client = FakeOpenAI()
    provider = make_provider(client, batch_size=2, max_concurrency=3)

    embeddings = provider.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert embeddings == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert sorted(client.embeddings.inputs) == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]


def test_embed_limits_concurrent_requests() -> None:
    client = FakeOpenAI(delay=0.05)
    provider = make_provider(client, batch_size=1, max_concurrency=4)

    provider.embed([str(index) for index in range(8)])

    assert 1 < client.embeddings.max_active_requests <= 4


def test_embed_skips_empty_input() -> None:
    client = FakeOpenAI()

    assert make_provider(client).embed([]) == []
    assert client.embeddings.inputs == []


@pytest.mark.parametrize(
    ("batch_size", "max_concurrency"),
    [(0, 1), (2_049, 1), (1, 0)],
)
def test_rejects_invalid_batching(batch_size: int, max_concurrency: int) -> None:
    with pytest.raises(ValueError):
        make_provider(
            FakeOpenAI(),
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
