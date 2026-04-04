"""LLM integration package for PuntLab.

Purpose: expose the canonical provider selection, prompt, and structured-output
building blocks used by research, scoring, and explanation stages.
Scope: provider fallback configuration for this step, plus future prompt and
schema modules that share the same namespace.
Dependencies: `src.llm.providers` for runtime model construction.
"""

from src.llm.providers import (
    TASK_LLM_CONFIGS,
    AllProvidersFailedError,
    LLMTaskConfig,
    get_langfuse_handler,
    get_llm,
    resolve_task_config,
)

__all__ = [
    "AllProvidersFailedError",
    "LLMTaskConfig",
    "TASK_LLM_CONFIGS",
    "get_langfuse_handler",
    "get_llm",
    "resolve_task_config",
]
