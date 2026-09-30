"""Language model interfaces and OpenAI implementation."""

from typing import Protocol

from openai import OpenAI


class LanguageModel(Protocol):
    """Minimal text generation boundary used by the retrieval agent."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response."""


class OpenAILanguageModel:
    """OpenAI Responses API adapter."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI()

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text.strip()
