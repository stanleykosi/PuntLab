"""Tests for PuntLab's canonical multi-provider LLM setup.

Purpose: verify task-aware model settings, provider fallback behavior, and
Langfuse callback construction without calling real LLM services.
Scope: unit tests for `src.llm.providers`.
Dependencies: pytest, the shared settings model, and lightweight constructor
stubs that replace the real LangChain and Langfuse classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from src.config import Settings
from src.llm import providers
from src.llm.providers import AllProvidersFailedError, get_langfuse_handler, get_llm


@dataclass
class FakeChatModel:
    """Simple stand-in for a LangChain chat model constructor result."""

    provider_label: str
    kwargs: dict[str, Any]


class FakeCallbackHandler:
    """Test double that records the credentials used for Langfuse setup."""

    def __init__(self, *, public_key: str) -> None:
        """Persist constructor arguments for later assertions."""

        self.public_key = public_key


class FakeLangfuseClient:
    """Test double that records the Langfuse client bootstrap arguments."""

    created_instances: list[dict[str, str]] = []

    def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
        """Persist constructor arguments for later assertions."""

        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.created_instances.append(
            {
                "public_key": public_key,
                "secret_key": secret_key,
                "host": host,
            }
        )


def build_settings(**overrides: Any) -> Settings:
    """Construct test settings with isolated in-memory credentials."""

    base_values: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": None,
        "anthropic_api_key": None,
        "openrouter_api_key": None,
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "database_url": "postgresql://puntlab:puntlab@localhost:5432/puntlab",
    }
    base_values.update(overrides)
    return Settings(**base_values)


def install_fake_model_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace provider SDK constructors with pure-Python test doubles."""

    def build_openai_model(**kwargs: Any) -> FakeChatModel:
        """Record the kwargs that would be passed to ChatOpenAI."""

        return FakeChatModel(provider_label="openai", kwargs=kwargs)

    def build_anthropic_model(**kwargs: Any) -> FakeChatModel:
        """Record the kwargs that would be passed to ChatAnthropic."""

        return FakeChatModel(provider_label="anthropic", kwargs=kwargs)

    monkeypatch.setattr(providers, "ChatOpenAI", build_openai_model)
    monkeypatch.setattr(providers, "ChatAnthropic", build_anthropic_model)


@pytest.mark.asyncio
async def test_get_llm_prefers_openai_primary_for_research_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI should be selected first with the research task token budget."""

    install_fake_model_factories(monkeypatch)
    settings = build_settings(openai_api_key="sk-openai")

    llm = await get_llm("research", settings=settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.provider_label == "openai"
    assert llm.kwargs["model"] == "gpt-4o"
    assert llm.kwargs["temperature"] == pytest.approx(0.3)
    assert llm.kwargs["max_tokens"] == 500
    assert llm.kwargs["callbacks"] == []
    assert llm.kwargs["metadata"]["task"] == "research"


@pytest.mark.asyncio
async def test_get_llm_falls_back_to_anthropic_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secondary Anthropic provider should be used when OpenAI is unavailable."""

    install_fake_model_factories(monkeypatch)
    settings = build_settings(anthropic_api_key="sk-ant")

    llm = await get_llm("qualitative_assessment", settings=settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.provider_label == "anthropic"
    assert llm.kwargs["model"] == "claude-sonnet-4-20250514"
    assert llm.kwargs["temperature"] == pytest.approx(0.3)
    assert llm.kwargs["max_tokens"] == 300
    assert llm.kwargs["metadata"]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_get_llm_uses_openrouter_with_openai_compatible_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback provider should target OpenRouter's OpenAI-compatible endpoint."""

    install_fake_model_factories(monkeypatch)
    settings = build_settings(openrouter_api_key="sk-or")

    llm = await get_llm("accumulator_rationale", settings=settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.provider_label == "openai"
    assert llm.kwargs["model"] == "meta-llama/llama-3-70b-instruct"
    assert llm.kwargs["base_url"] == providers.OPENROUTER_BASE_URL
    assert llm.kwargs["default_headers"] == {"X-Title": "puntlab-agent"}
    assert llm.kwargs["max_tokens"] == 150


@pytest.mark.asyncio
async def test_get_llm_raises_clear_error_when_no_provider_credentials_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider construction should fail fast with explicit diagnostics."""

    install_fake_model_factories(monkeypatch)
    settings = build_settings()

    with pytest.raises(
        AllProvidersFailedError,
        match="Attempted providers: openai, anthropic, openrouter",
    ):
        await get_llm("leg_rationale", settings=settings)


def test_get_langfuse_handler_returns_none_without_complete_credentials() -> None:
    """Tracing should stay disabled when Langfuse keys are not fully configured."""

    settings = build_settings(langfuse_public_key="pk-lf")

    assert get_langfuse_handler(settings) is None


def test_get_langfuse_handler_builds_cached_callback_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse handler construction should reuse a cached handler instance."""

    monkeypatch.setattr(providers, "CallbackHandler", FakeCallbackHandler)
    monkeypatch.setattr(providers, "Langfuse", FakeLangfuseClient)
    providers._build_langfuse_handler.cache_clear()
    FakeLangfuseClient.created_instances.clear()

    settings = build_settings(
        langfuse_public_key="pk-lf",
        langfuse_secret_key="sk-lf",
    )

    first_handler = get_langfuse_handler(settings)
    second_handler = get_langfuse_handler(settings)

    assert isinstance(first_handler, FakeCallbackHandler)
    assert first_handler is second_handler
    assert first_handler.public_key == "pk-lf"
    assert FakeLangfuseClient.created_instances == [
        {
            "public_key": "pk-lf",
            "secret_key": "sk-lf",
            "host": "https://cloud.langfuse.com",
        }
    ]


@pytest.mark.asyncio
async def test_get_llm_rejects_unknown_task_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected task names should fail fast instead of silently defaulting."""

    install_fake_model_factories(monkeypatch)
    settings = build_settings(openai_api_key="sk-openai")

    with pytest.raises(ValueError, match="Unknown LLM task"):
        await get_llm("made_up_task", settings=settings)
