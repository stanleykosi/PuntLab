"""Explanation node for PuntLab's LangGraph pipeline.

Purpose: enrich generated accumulator slips with concise slip-level rationales
before the approval stage.
Scope: task-aware LLM invocation for explanation prompts, deterministic
JSON response validation, and `ExplainedAccumulator` output assembly for
downstream delivery.
Dependencies: `src.llm` for prompt and model access, `src.pipeline.state` for
validated LangGraph state exchange, and accumulator schemas for leg/slip IO.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import MarketType
from src.llm import (
    AccumulatorRationale,
    get_llm,
    get_prompt,
)
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorLeg, AccumulatorSlip, ExplainedAccumulator

DEFAULT_MAX_RATIONALE_LENGTH = 320
type ExplanationProgressCallback = Callable[[Mapping[str, object]], None]


async def explanation_node(
    state: PipelineState | Mapping[str, Any],
    *,
    accumulator_llm: BaseChatModel | None = None,
    progress_callback: ExplanationProgressCallback | None = None,
) -> dict[str, object]:
    """Execute the explanation stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        accumulator_llm: Optional injected LLM used for full-slip rationales.
        progress_callback: Optional callback receiving stage progress updates
            suitable for live CLI dashboards.

    Outputs:
        A partial LangGraph update containing delivery-ready
        `ExplainedAccumulator` rows, merged diagnostics, and the next stage
        marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    if not validated_state.accumulators:
        return {
            "current_stage": PipelineStage.APPROVAL,
            "explained_accumulators": [],
            "errors": list(validated_state.errors),
        }

    resolved_accumulator_llm = accumulator_llm
    total_slips = len(validated_state.accumulators)
    _emit_progress(
        progress_callback,
        event="stage_started",
        total_slips=total_slips,
    )

    if resolved_accumulator_llm is None:
        resolved_accumulator_llm = await get_llm("accumulator_rationale")

    explained_accumulators: list[ExplainedAccumulator] = []
    for slip_index, slip in enumerate(validated_state.accumulators, start=1):
        _emit_progress(
            progress_callback,
            event="slip_started",
            slip_number=slip.slip_number,
            slip_index=slip_index,
            total_slips=total_slips,
            leg_count=len(slip.legs),
        )
        _emit_progress(
            progress_callback,
            event="slip_rationale_started",
            slip_number=slip.slip_number,
            slip_index=slip_index,
            total_slips=total_slips,
            leg_count=len(slip.legs),
        )
        slip_started_at = perf_counter()
        (
            accumulator_rationale,
            accumulator_diagnostics,
            disable_accumulator_llm,
        ) = await _explain_accumulator(
            slip,
            llm=resolved_accumulator_llm,
        )
        if disable_accumulator_llm:
            raise RuntimeError("Accumulator rationale LLM was disabled unexpectedly.")

        explained_accumulators.append(
            ExplainedAccumulator.model_validate(
                {
                    **slip.model_dump(),
                    "legs": slip.legs,
                    "rationale": accumulator_rationale,
                }
            )
        )
        _emit_progress(
            progress_callback,
            event="slip_completed",
            slip_number=slip.slip_number,
            slip_index=slip_index,
            total_slips=total_slips,
            leg_count=len(slip.legs),
            duration_seconds=perf_counter() - slip_started_at,
            had_error=bool(accumulator_diagnostics),
        )

    _emit_progress(
        progress_callback,
        event="stage_completed",
        total_slips=total_slips,
    )

    return {
        "current_stage": PipelineStage.APPROVAL,
        "explained_accumulators": explained_accumulators,
        "errors": list(validated_state.errors),
    }


def _emit_progress(
    callback: ExplanationProgressCallback | None,
    /,
    **payload: object,
) -> None:
    """Emit one progress update without allowing callback failures to bubble.

    Inputs:
        callback: Optional progress callback supplied by the runtime.
        payload: Structured progress update payload.

    Outputs:
        None. Callback failures are swallowed to keep the node resilient.
    """

    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        return


async def _explain_accumulator(
    slip: AccumulatorSlip,
    *,
    llm: BaseChatModel | None,
) -> tuple[str, tuple[str, ...], bool]:
    """Generate one accumulator rationale via LLM only.

    Inputs:
        slip: Accumulator slip whose legs already contain their final leg-level
            rationales.
        llm: Optional explanation-model instance. Missing models fail fast.

    Outputs:
        A tuple of `(rationale, diagnostics, disabled)` where diagnostics and
        disabled are retained for progress event shape but remain empty/false.
    """

    if llm is None:
        raise RuntimeError(
            "accumulator rationale requires an LLM; deterministic fallback is disabled."
        )

    prompt = get_prompt("accumulator_rationale")
    prompt_messages = prompt.format_messages(
        slip_summary=_render_slip_summary(slip),
        legs_summary=_render_legs_summary(slip.legs),
        confidence_summary=_render_confidence_summary(slip),
        portfolio_note=_render_portfolio_note(slip),
    )

    rationale_result = await _invoke_accumulator_rationale_model(
        llm=llm,
        prompt_messages=prompt_messages,
        slip_number=slip.slip_number,
    )

    return (
        _combine_rationale_and_risk(
            rationale_result.rationale,
            rationale_result.shared_risk,
            risk_label="Shared risk",
            max_length=DEFAULT_MAX_RATIONALE_LENGTH,
        ),
        (),
        False,
    )


async def _invoke_accumulator_rationale_model(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    slip_number: int,
) -> AccumulatorRationale:
    """Invoke the accumulator-rationale model with resilient payload handling."""

    result = await invoke_json_schema(
        llm=llm,
        prompt_messages=prompt_messages,
        schema=AccumulatorRationale,
        instruction=_rationale_json_instruction(),
    )
    return result.model_copy(update={"slip_number": result.slip_number or slip_number})


def _rationale_json_instruction() -> str:
    """Return the JSON-only instruction for one explanation schema."""

    return (
        "Return only one valid JSON object. Do not use markdown. "
        "Keys: slip_number, rationale, shared_risk. Use null for shared_risk "
        "when there is no material shared caveat. Keep rationale under 70 words."
    )


def _display_market_label(leg: AccumulatorLeg) -> str:
    """Return the best available market label for explanation prompts."""

    if leg.market_label:
        return leg.market_label
    if leg.canonical_market is not None:
        return leg.canonical_market.value.replace("_", " ")
    return leg.market.replace("_", " ")


def _render_slip_summary(slip: AccumulatorSlip) -> str:
    """Render one high-level slip summary for accumulator rationale prompts."""

    strategy_label = slip.strategy.value if slip.strategy is not None else "unspecified"
    return (
        f"Slip #{slip.slip_number} with {slip.leg_count} legs, total odds {slip.total_odds:.2f}, "
        f"confidence {slip.confidence:.2f}, strategy {strategy_label}."
    )


def _render_legs_summary(legs: Sequence[AccumulatorLeg]) -> str:
    """Render the slip's legs into an ordered multi-line prompt summary."""

    return "\n".join(
        (
            f"{leg.leg_number}. {leg.fixture_label()} - {leg.selection} "
            f"({_display_market_label(leg)}) @ {leg.odds:.2f}; "
            f"confidence {leg.confidence:.2f}; "
            f"edge: {leg.rationale or 'No prior rationale available.'}"
        )
        for leg in legs
    )


def _render_confidence_summary(slip: AccumulatorSlip) -> str:
    """Describe the slip's current confidence posture for the LLM."""

    return (
        f"The slip grades as { _describe_confidence_band(slip.confidence) } confidence "
        f"with {slip.leg_count} legs and combined odds of {slip.total_odds:.2f}."
    )


def _render_portfolio_note(slip: AccumulatorSlip) -> str:
    """Summarize shared slip composition and risk posture for the LLM."""

    competitions = sorted({leg.competition for leg in slip.legs})
    resolution_counts = Counter(leg.resolution_source.value for leg in slip.legs)
    competition_summary = (
        "1 competition"
        if len(competitions) == 1
        else f"{len(competitions)} competitions"
    )
    source_summary = ", ".join(
        f"{source} x{count}" for source, count in sorted(resolution_counts.items())
    )
    shared_risk = _derive_accumulator_risk(slip)
    return (
        f"Composition spans {competition_summary}; "
        f"resolution mix: {source_summary or 'unknown'}; "
        f"largest shared risk: {shared_risk or 'standard multi-leg variance'}."
    )


def _derive_accumulator_risk(slip: AccumulatorSlip) -> str | None:
    """Infer one concise shared-risk note from the slip's composition."""

    if slip.leg_count >= 5:
        return "More legs increase the chance that one volatile result breaks the slip."
    if any(leg.resolution_source.value == "sportybet_browser" for leg in slip.legs):
        return "At least one leg depended on SportyBet browser fallback instead of the API path."
    if any(leg.market in _LINE_MARKETS for leg in slip.legs):
        return "Line-based totals or handicap legs can turn on narrow margins."
    if slip.total_odds >= 8.0:
        return "The higher combined price leaves less room for error across the slip."
    return None


def _combine_rationale_and_risk(
    rationale: str,
    risk: str | None,
    *,
    risk_label: str,
    max_length: int,
) -> str:
    """Combine rationale text with an optional risk note within schema limits."""

    normalized_rationale = _ensure_sentence(rationale)
    if risk is None:
        return normalized_rationale

    combined = f"{normalized_rationale} {risk_label}: {_ensure_sentence(risk, force_period=False)}"
    if len(combined) <= max_length:
        return combined
    return normalized_rationale[:max_length].rstrip()


def _ensure_sentence(text: str, *, force_period: bool = True) -> str:
    """Normalize sentence-style explanation text and ensure punctuation."""

    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ""
    if force_period and normalized[-1] not in {".", "!", "?"}:
        return f"{normalized}."
    return normalized


def _describe_confidence_band(confidence: float) -> str:
    """Translate a numeric confidence score into a short qualitative label."""

    if confidence >= 0.78:
        return "high-confidence"
    if confidence >= 0.62:
        return "steady"
    return "measured"


_LINE_MARKETS = frozenset(
    {
        MarketType.ASIAN_HANDICAP,
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
    }
)


__all__ = ["explanation_node"]
