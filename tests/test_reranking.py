from pathlib import Path
from typing import Protocol, cast

import pytest

from agentic_rag import reranking
from agentic_rag.models import DocumentChunk, RetrievedChunk
from agentic_rag.reranking import FlashRankReranker


class _Request(Protocol):
    query: str | None
    passages: list[dict[str, object]]


def _hit(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source="guide.txt",
            chunk_index=0,
        ),
        score=score,
    )


def test_flashrank_adapter_preserves_hits_in_model_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializations: list[tuple[str, str, int]] = []
    requests: list[_Request] = []

    class FakeRanker:
        def __init__(self, *, model_name: str, cache_dir: str, max_length: int) -> None:
            initializations.append((model_name, cache_dir, max_length))

        def rerank(self, request: object) -> list[dict[str, object]]:
            parsed_request = cast(_Request, request)
            requests.append(parsed_request)
            return list(reversed(parsed_request.passages))

    monkeypatch.setattr(reranking, "Ranker", FakeRanker)
    reranker = FlashRankReranker(
        tmp_path,
        model_name="test-model",
        max_length=256,
    )
    first = _hit("chunk-1", "Initial vector match.", 0.91)
    second = _hit("chunk-2", "Stronger semantic match.", 0.83)

    result = reranker.rerank("semantic match", [first, second], limit=1)

    assert initializations == [("test-model", str(tmp_path), 256)]
    assert requests[0].query == "semantic match"
    assert requests[0].passages == [
        {"id": 0, "text": "Initial vector match."},
        {"id": 1, "text": "Stronger semantic match."},
    ]
    assert result == [second]
    assert result[0].score == 0.83


def test_flashrank_adapter_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRanker:
        def __init__(self, *, model_name: str, cache_dir: str, max_length: int) -> None:
            pass

        def rerank(self, request: object) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(reranking, "Ranker", FakeRanker)
    reranker = FlashRankReranker(tmp_path)

    with pytest.raises(ValueError, match="query must not be empty"):
        reranker.rerank(" ", [], limit=1)
    with pytest.raises(ValueError, match="limit must be positive"):
        reranker.rerank("query", [], limit=0)
