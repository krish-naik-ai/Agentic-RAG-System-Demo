"""Language model interfaces and provider implementations."""

import os
from collections.abc import Mapping
from typing import Literal, Protocol

from openai import OpenAI

LanguageModelProvider = Literal["openai", "gemini"]
GeminiModel = Literal["gemini-2.5-flash", "gemini-3.5-flash"]

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SUPPORTED_GEMINI_MODELS: tuple[GeminiModel, ...] = (
    "gemini-2.5-flash",
    "gemini-3.5-flash",
)
SUPPORTED_PROVIDERS: tuple[LanguageModelProvider, ...] = ("openai", "gemini")


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


class GeminiLanguageModel:
    """Gemini adapter using Google's OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        if model not in SUPPORTED_GEMINI_MODELS:
            supported = ", ".join(SUPPORTED_GEMINI_MODELS)
            raise ValueError(f"Unsupported Gemini model {model!r}. Choose one of: {supported}")
        self._model = model
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY to use the Gemini language model")
            client = OpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE_URL)
        self._client = client

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if content is None:
            raise RuntimeError("Gemini returned an empty response")
        return content.strip()


def create_language_model(environ: Mapping[str, str] | None = None) -> LanguageModel:
    """Build the language model selected by ``LLM_PROVIDER`` and ``LLM_MODEL``."""

    env = os.environ if environ is None else environ
    provider = env.get("LLM_PROVIDER", "openai").strip().lower()
    model = env.get("LLM_MODEL", "").strip()
    if provider == "openai":
        return OpenAILanguageModel(model=model) if model else OpenAILanguageModel()
    if provider == "gemini":
        return GeminiLanguageModel(
            model=model or "gemini-2.5-flash",
            api_key=env.get("GEMINI_API_KEY"),
        )
    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise ValueError(f"Unsupported LLM_PROVIDER {provider!r}. Choose one of: {supported}")


def required_api_keys(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return environment variables required by the configured providers."""

    env = os.environ if environ is None else environ
    keys = ["OPENAI_API_KEY"]
    if env.get("LLM_PROVIDER", "openai").strip().lower() == "gemini":
        keys.append("GEMINI_API_KEY")
    return keys
