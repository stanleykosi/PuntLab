"""LLM integration package for PuntLab.

Purpose: expose the canonical provider selection, prompt, and structured-output
building blocks used by research, scoring, and explanation stages.
Scope: provider fallback configuration for this step, plus future prompt and
schema modules that share the same namespace.
Dependencies: `src.llm.providers` for runtime model construction.
"""

from src.llm.prompts import (
    ACCUMULATOR_RATIONALE_PROMPT,
    NEWS_CONTEXT_ANALYSIS_PROMPT,
    PROMPT_REGISTRY,
    PROMPT_TASK_KEYS,
    QUALITATIVE_ASSESSMENT_PROMPT,
    get_prompt,
    resolve_prompt_task,
)
from src.llm.providers import (
    TASK_LLM_CONFIGS,
    AllProvidersFailedError,
    LLMTaskConfig,
    get_langfuse_handler,
    get_llm,
    resolve_task_config,
)
from src.llm.schemas import (
    AccumulatorRationale,
    MatchContext,
    QualitativeScore,
)

__all__ = [
    "ACCUMULATOR_RATIONALE_PROMPT",
    "AccumulatorRationale",
    "AllProvidersFailedError",
    "LLMTaskConfig",
    "MatchContext",
    "NEWS_CONTEXT_ANALYSIS_PROMPT",
    "PROMPT_REGISTRY",
    "PROMPT_TASK_KEYS",
    "QualitativeScore",
    "QUALITATIVE_ASSESSMENT_PROMPT",
    "TASK_LLM_CONFIGS",
    "get_langfuse_handler",
    "get_llm",
    "get_prompt",
    "resolve_prompt_task",
    "resolve_task_config",
]
