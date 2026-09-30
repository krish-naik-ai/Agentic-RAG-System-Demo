from collections.abc import Sequence

from agentic_rag.models import DocumentChunk, RetrievedChunk


class KeywordEmbeddingProvider:
    """Deterministic embeddings for offline tests."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [
                float("retrieval" in text.lower() or "agent" in text.lower()),
                float("cooking" in text.lower() or "recipe" in text.lower()),
                0.1,
            ]
            for text in texts
        ]


class ScriptedLanguageModel:
    """Returns predefined responses and records prompts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return next(self._responses)


class StubVectorStore:
    """In-memory vector-store test double."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits
        self.last_embedding: list[float] | None = None
        self.last_limit: int | None = None

    def replace(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], *, limit: int = 5) -> list[RetrievedChunk]:
        self.last_embedding = embedding
        self.last_limit = limit
        return self._hits[:limit]

    def count(self) -> int:
        return len(self._hits)
