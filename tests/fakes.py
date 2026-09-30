from collections.abc import Sequence


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
