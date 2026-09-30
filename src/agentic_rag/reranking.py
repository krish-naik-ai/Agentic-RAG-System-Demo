"""Open-source cross-encoder reranking for retrieved document chunks."""

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Protocol, TypedDict, cast

from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]

from agentic_rag.models import RetrievedChunk

DEFAULT_RERANKING_MODEL = "ms-marco-TinyBERT-L-2-v2"


class Reranker(Protocol):
    """Orders retrieved chunks by their relevance to a query."""

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Return the strongest hits first, capped at the requested limit."""


class _RankedPassage(TypedDict):
    id: int
    text: str


class _RankerBackend(Protocol):
    def rerank(self, request: object) -> list[_RankedPassage]:
        """Score and order passages for one query."""


class FlashRankReranker:
    """Runs a lightweight FlashRank cross-encoder locally on CPU."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        model_name: str = DEFAULT_RERANKING_MODEL,
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._max_length = max_length
        self._ranker: _RankerBackend | None = None
        self._ranker_lock = Lock()

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Rerank vector-search candidates while preserving their metadata."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not hits:
            return []

        passages = [
            {
                "id": index,
                "text": hit.chunk.text,
            }
            for index, hit in enumerate(hits)
        ]
        result = self._get_ranker().rerank(RerankRequest(query=query, passages=passages))

        reranked: list[RetrievedChunk] = []
        seen_indices: set[int] = set()
        for passage in result:
            index = passage["id"]
            if index < 0 or index >= len(hits) or index in seen_indices:
                raise ValueError("The reranker returned invalid passage identifiers")
            seen_indices.add(index)
            reranked.append(hits[index])
            if len(reranked) == limit:
                break
        return reranked

    def _get_ranker(self) -> _RankerBackend:
        ranker = self._ranker
        if ranker is not None:
            return ranker

        with self._ranker_lock:
            ranker = self._ranker
            if ranker is None:
                try:
                    ranker = cast(
                        _RankerBackend,
                        Ranker(
                            model_name=self._model_name,
                            cache_dir=str(self._cache_dir),
                            max_length=self._max_length,
                        ),
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"Unable to load reranking model {self._model_name!r}: {error}"
                    ) from error
                self._ranker = ranker
        return ranker
