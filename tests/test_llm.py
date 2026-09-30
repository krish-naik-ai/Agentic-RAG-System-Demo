from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import pytest
from openai import OpenAI

from agentic_rag.llm import (
    GEMINI_OPENAI_BASE_URL,
    GeminiLanguageModel,
    OpenAILanguageModel,
    create_language_model,
    required_api_keys,
)


class CompletionMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class CompletionCall(TypedDict):
    model: str
    messages: list[CompletionMessage]


@dataclass(frozen=True)
class FakeMessage:
    content: str | None


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeChatResponse:
    choices: list[FakeChoice]


class RecordingCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.calls: list[CompletionCall] = []

    def create(self, *, model: str, messages: list[CompletionMessage]) -> FakeChatResponse:
        self.calls.append({"model": model, "messages": messages})
        return FakeChatResponse(choices=[FakeChoice(message=FakeMessage(self._content))])


@dataclass(frozen=True)
class FakeChat:
    completions: RecordingCompletions


@dataclass(frozen=True)
class FakeOpenAI:
    chat: FakeChat


def fake_client(completions: RecordingCompletions) -> OpenAI:
    return cast(OpenAI, FakeOpenAI(chat=FakeChat(completions=completions)))


def test_gemini_model_sends_system_and_user_messages() -> None:
    completions = RecordingCompletions("  Grounded answer [S1].  ")
    model = GeminiLanguageModel(model="gemini-2.5-flash", client=fake_client(completions))

    answer = model.complete(system_prompt="Be precise.", user_prompt="What is RAG?")

    assert answer == "Grounded answer [S1]."
    assert completions.calls == [
        {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "What is RAG?"},
            ],
        }
    ]


def test_gemini_model_rejects_empty_response() -> None:
    model = GeminiLanguageModel(client=fake_client(RecordingCompletions(None)))

    with pytest.raises(RuntimeError, match="empty response"):
        model.complete(system_prompt="system", user_prompt="user")


def test_gemini_model_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported Gemini model"):
        GeminiLanguageModel(model="gemini-2.5-pro", api_key="test-key")


def test_gemini_model_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLanguageModel()


def test_gemini_client_targets_openai_compatible_endpoint() -> None:
    model = GeminiLanguageModel(model="gemini-3.5-flash", api_key="test-key")

    client = model._client
    assert str(client.base_url) == GEMINI_OPENAI_BASE_URL
    assert client.api_key == "test-key"


def test_factory_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert isinstance(create_language_model({}), OpenAILanguageModel)


def test_factory_builds_gemini_from_environment() -> None:
    model = create_language_model(
        {"LLM_PROVIDER": "Gemini", "LLM_MODEL": "gemini-3.5-flash", "GEMINI_API_KEY": "key"}
    )

    assert isinstance(model, GeminiLanguageModel)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        create_language_model({"LLM_PROVIDER": "anthropic"})


def test_gemini_still_requires_openai_key_for_embeddings() -> None:
    assert required_api_keys({}) == ["OPENAI_API_KEY"]
    assert required_api_keys({"LLM_PROVIDER": "gemini"}) == ["OPENAI_API_KEY", "GEMINI_API_KEY"]
