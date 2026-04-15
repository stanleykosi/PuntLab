"""Canonical multi-provider LLM configuration for PuntLab.

Purpose: centralize task-aware model settings, provider selection order, and
Langfuse callback wiring for every LangChain-backed LLM call.
Scope: configurable OpenRouter/OpenAI/Anthropic provider fallbacks via
LangChain adapters plus fail-fast diagnostics when no provider can be
constructed for a requested task.
Dependencies: `src.config` for validated credentials and model ordering,
`langchain-openai`/`langchain-anthropic` for chat models, and `langfuse` for
optional tracing callbacks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "anthropic", "openrouter"})
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 30.0
DEFAULT_PROVIDER_MAX_RETRIES = 2


@dataclass(frozen=True, slots=True)
class LLMTaskConfig:
    """Stable task-level defaults for one category of LLM work.

    Inputs:
        Canonical task definitions declared in `TASK_LLM_CONFIGS`.

    Outputs:
        A validated configuration object that determines max tokens and
        sampling temperature for the requested pipeline task.
    """

    task_key: str
    temperature: float
    max_tokens: int
    description: str

    def __post_init__(self) -> None:
        """Reject invalid task-level model settings at import time."""

        if not self.task_key.strip():
            raise ValueError("LLM task keys must not be blank.")
        if not 0 <= self.temperature <= 1:
            raise ValueError("LLM task temperatures must be between 0.0 and 1.0.")
        if self.max_tokens <= 0:
            raise ValueError("LLM task max_tokens must be a positive integer.")
        if not self.description.strip():
            raise ValueError("LLM task descriptions must not be blank.")


@dataclass(frozen=True, slots=True)
class ProviderCandidate:
    """Resolved provider slot ready for runtime model construction.

    Inputs:
        One configured provider slot from the current agent settings.

    Outputs:
        A normalized provider description with the credential and model name
        needed to instantiate a LangChain chat model.
    """

    key: str
    model: str
    api_key: str | None
    priority_label: str

    def __post_init__(self) -> None:
        """Validate that provider candidates are structurally usable."""

        if self.key not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{self.key}'. "
                f"Expected one of: {sorted(SUPPORTED_LLM_PROVIDERS)}."
            )
        if not self.model.strip():
            raise ValueError("LLM provider model names must not be blank.")
        if not self.priority_label.strip():
            raise ValueError("LLM provider priority labels must not be blank.")


TASK_LLM_CONFIGS: dict[str, LLMTaskConfig] = {
    "default": LLMTaskConfig(
        task_key="default",
        temperature=0.3,
        max_tokens=500,
        description="General-purpose fallback settings for uncategorized agent calls.",
    ),
    "research": LLMTaskConfig(
        task_key="research",
        temperature=0.3,
        max_tokens=500,
        description="News-context analysis for fixture narratives and qualitative signals.",
    ),
    "qualitative_assessment": LLMTaskConfig(
        task_key="qualitative_assessment",
        temperature=0.3,
        max_tokens=300,
        description="Compact qualitative scoring based on fixture context and research.",
    ),
    "leg_rationale": LLMTaskConfig(
        task_key="leg_rationale",
        temperature=0.2,
        max_tokens=100,
        description="Short rationale generation for one accumulator leg.",
    ),
    "accumulator_rationale": LLMTaskConfig(
        task_key="accumulator_rationale",
        temperature=0.2,
        max_tokens=150,
        description="Short rationale generation for a full accumulator slip.",
    ),
}

TASK_ALIASES: dict[str, str] = {
    "news_context_analysis": "research",
    "context_analysis": "research",
    "qualitative": "qualitative_assessment",
    "qualitative_score": "qualitative_assessment",
    "leg_explanation": "leg_rationale",
    "accumulator_explanation": "accumulator_rationale",
}


class AllProvidersFailedError(RuntimeError):
    """Raised when no configured LLM provider can satisfy a requested task."""

    def __init__(self, task: str, attempted_providers: tuple[str, ...], reasons: tuple[str, ...]):
        """Capture the requested task and the concrete provider diagnostics."""

        provider_summary = ", ".join(attempted_providers) or "none"
        reason_summary = " | ".join(reasons) if reasons else "No providers were configured."
        super().__init__(
            f"Unable to construct an LLM for task '{task}'. "
            f"Attempted providers: {provider_summary}. Diagnostics: {reason_summary}"
        )
        self.task = task
        self.attempted_providers = attempted_providers
        self.reasons = reasons


def resolve_task_config(task: str = "default") -> LLMTaskConfig:
    """Return the canonical LLM settings for one pipeline task.

    Inputs:
        task: Stable task key used by pipeline stages, such as `research` or
            `leg_rationale`. Blank values resolve to `default`.

    Outputs:
        The validated `LLMTaskConfig` for the requested task.
    """

    normalized_task = task.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized_task:
        normalized_task = "default"

    resolved_key = TASK_ALIASES.get(normalized_task, normalized_task)
    if resolved_key not in TASK_LLM_CONFIGS:
        raise ValueError(
            f"Unknown LLM task '{task}'. Expected one of: {sorted(TASK_LLM_CONFIGS)}."
        )

    return TASK_LLM_CONFIGS[resolved_key]


def _resolve_provider_candidates(settings: Settings) -> tuple[ProviderCandidate, ...]:
    """Translate configured provider slots into normalized runtime candidates."""

    configured_slots = (
        ProviderCandidate(
            key=settings.llm.primary_provider.strip().lower(),
            model=settings.llm.primary_model.strip(),
            api_key=settings.llm.openai_api_key
            if settings.llm.primary_provider.strip().lower() == "openai"
            else settings.llm.anthropic_api_key
            if settings.llm.primary_provider.strip().lower() == "anthropic"
            else settings.llm.openrouter_api_key,
            priority_label="primary",
        ),
        ProviderCandidate(
            key=settings.llm.secondary_provider.strip().lower(),
            model=settings.llm.secondary_model.strip(),
            api_key=settings.llm.openai_api_key
            if settings.llm.secondary_provider.strip().lower() == "openai"
            else settings.llm.anthropic_api_key
            if settings.llm.secondary_provider.strip().lower() == "anthropic"
            else settings.llm.openrouter_api_key,
            priority_label="secondary",
        ),
        ProviderCandidate(
            key=settings.llm.fallback_provider.strip().lower(),
            model=settings.llm.fallback_model.strip(),
            api_key=settings.llm.openai_api_key
            if settings.llm.fallback_provider.strip().lower() == "openai"
            else settings.llm.anthropic_api_key
            if settings.llm.fallback_provider.strip().lower() == "anthropic"
            else settings.llm.openrouter_api_key,
            priority_label="fallback",
        ),
    )

    provider_keys = tuple(candidate.key for candidate in configured_slots)
    if len(provider_keys) != len(set(provider_keys)):
        raise ValueError(
            "LLM provider slots must be unique. "
            f"Configured providers: {provider_keys}."
        )

    return configured_slots


@lru_cache(maxsize=4)
def _build_langfuse_handler(
    public_key: str,
    secret_key: str,
    host: str,
) -> CallbackHandler:
    """Instantiate and cache one Langfuse callback handler per credential set."""

    Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    return CallbackHandler(public_key=public_key)


def get_langfuse_handler(settings: Settings | None = None) -> CallbackHandler | None:
    """Return the shared Langfuse callback handler when credentials are present.

    Inputs:
        settings: Optional preloaded runtime settings for tests or custom flows.

    Outputs:
        A cached `CallbackHandler` when both Langfuse keys are configured;
        otherwise `None`, allowing LLM calls to proceed without tracing.
    """

    runtime_settings = settings or get_settings()
    public_key = runtime_settings.langfuse.public_key
    secret_key = runtime_settings.langfuse.secret_key

    if not public_key or not secret_key:
        return None

    return _build_langfuse_handler(public_key, secret_key, runtime_settings.langfuse.host)


def _build_callbacks(settings: Settings) -> list[CallbackHandler]:
    """Return the callback list attached to every constructed chat model."""

    handler = get_langfuse_handler(settings)
    return [handler] if handler is not None else []


def _base_model_kwargs(
    *,
    provider_key: str,
    provider_model: str,
    task_config: LLMTaskConfig,
    settings: Settings,
) -> dict[str, Any]:
    """Build shared model kwargs applied across all provider implementations."""

    return {
        "model": provider_model,
        "temperature": task_config.temperature,
        "max_tokens": task_config.max_tokens,
        "timeout": DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_PROVIDER_MAX_RETRIES,
        "callbacks": _build_callbacks(settings),
        "tags": [f"provider:{provider_key}", f"task:{task_config.task_key}"],
        "metadata": {
            "provider": provider_key,
            "task": task_config.task_key,
            "app": settings.app.app_name,
        },
    }


def _build_secret_api_key(api_key: str) -> SecretStr:
    """Wrap raw provider keys in `SecretStr` for LangChain client constructors."""

    return SecretStr(api_key)


def _require_api_key(candidate: ProviderCandidate) -> str:
    """Return a non-null provider key after the caller's availability checks."""

    if candidate.api_key is None:
        raise ValueError(
            f"LLM provider '{candidate.key}' cannot be initialized without an API key."
        )
    return candidate.api_key


def _create_openai_model(
    candidate: ProviderCandidate,
    task_config: LLMTaskConfig,
    settings: Settings,
) -> BaseChatModel:
    """Construct the OpenAI primary provider model."""

    return ChatOpenAI(
        api_key=_build_secret_api_key(_require_api_key(candidate)),
        **_base_model_kwargs(
            provider_key=candidate.key,
            provider_model=candidate.model,
            task_config=task_config,
            settings=settings,
        ),
    )


def _create_anthropic_model(
    candidate: ProviderCandidate,
    task_config: LLMTaskConfig,
    settings: Settings,
) -> BaseChatModel:
    """Construct the Anthropic secondary provider model."""

    return ChatAnthropic(
        api_key=_build_secret_api_key(_require_api_key(candidate)),
        **_base_model_kwargs(
            provider_key=candidate.key,
            provider_model=candidate.model,
            task_config=task_config,
            settings=settings,
        ),
    )


def _create_openrouter_model(
    candidate: ProviderCandidate,
    task_config: LLMTaskConfig,
    settings: Settings,
) -> BaseChatModel:
    """Construct the OpenRouter fallback using the OpenAI-compatible adapter."""

    return ChatOpenAI(
        api_key=_build_secret_api_key(_require_api_key(candidate)),
        base_url=OPENROUTER_BASE_URL,
        default_headers={"X-Title": settings.app.app_name},
        **_base_model_kwargs(
            provider_key=candidate.key,
            provider_model=candidate.model,
            task_config=task_config,
            settings=settings,
        ),
    )


PROVIDER_FACTORIES: Mapping[
    str,
    Callable[[ProviderCandidate, LLMTaskConfig, Settings], BaseChatModel],
] = {
    "openai": _create_openai_model,
    "anthropic": _create_anthropic_model,
    "openrouter": _create_openrouter_model,
}


async def get_llm(task: str = "default", *, settings: Settings | None = None) -> BaseChatModel:
    """Return a configured LangChain chat model with provider fallback.

    Inputs:
        task: Stable task key that selects the canonical max-token and
            temperature profile for this model instance.
        settings: Optional preloaded settings object used primarily in tests.

    Outputs:
        The first successfully constructed `BaseChatModel` from the configured
        primary → secondary → fallback provider order.
    """

    runtime_settings = settings or get_settings()
    task_config = resolve_task_config(task)
    provider_candidates = _resolve_provider_candidates(runtime_settings)

    attempted_providers: list[str] = []
    failure_reasons: list[str] = []

    for candidate in provider_candidates:
        attempted_providers.append(candidate.key)
        if not candidate.api_key:
            reason = (
                f"{candidate.priority_label} provider '{candidate.key}' skipped because its "
                "API key is not configured."
            )
            logger.warning("Skipping LLM provider: %s", reason)
            failure_reasons.append(reason)
            continue

        try:
            llm = PROVIDER_FACTORIES[candidate.key](candidate, task_config, runtime_settings)
            logger.info(
                "Selected LLM provider '%s' for task '%s' using model '%s'.",
                candidate.key,
                task_config.task_key,
                candidate.model,
            )
            return llm
        except Exception as exc:  # pragma: no cover - defensive guard around SDK init
            reason = (
                f"{candidate.priority_label} provider '{candidate.key}' failed to initialize "
                f"model '{candidate.model}': {exc}"
            )
            logger.exception("Failed to initialize LLM provider '%s'.", candidate.key)
            failure_reasons.append(reason)

    raise AllProvidersFailedError(
        task=task_config.task_key,
        attempted_providers=tuple(attempted_providers),
        reasons=tuple(failure_reasons),
    )


__all__ = [
    "AllProvidersFailedError",
    "DEFAULT_PROVIDER_MAX_RETRIES",
    "DEFAULT_PROVIDER_TIMEOUT_SECONDS",
    "LLMTaskConfig",
    "OPENROUTER_BASE_URL",
    "PROVIDER_FACTORIES",
    "SUPPORTED_LLM_PROVIDERS",
    "TASK_LLM_CONFIGS",
    "get_langfuse_handler",
    "get_llm",
    "resolve_task_config",
]
