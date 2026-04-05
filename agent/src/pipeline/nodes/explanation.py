"""Explanation node for PuntLab's LangGraph pipeline.

Purpose: enrich generated accumulator slips with concise leg-level and
slip-level rationales before the approval stage.
Scope: task-aware LLM invocation for explanation prompts, deterministic
fallback rationale generation when providers are unavailable, and validated
`ExplainedAccumulator` output assembly for downstream delivery.
Dependencies: `src.llm` for prompt and model access, `src.pipeline.state` for
validated LangGraph state exchange, and accumulator schemas for leg/slip IO.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import MarketType
from src.llm import (
    AccumulatorRationale,
    AllProvidersFailedError,
    LegRationale,
    get_llm,
    get_prompt,
)
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorLeg, AccumulatorSlip, ExplainedAccumulator

DEFAULT_MAX_RATIONALE_LENGTH = 320
DEFAULT_MAX_LEG_RATIONALE_LENGTH = 240


async def explanation_node(
    state: PipelineState | Mapping[str, Any],
    *,
    leg_llm: BaseChatModel | None = None,
    accumulator_llm: BaseChatModel | None = None,
) -> dict[str, object]:
    """Execute the explanation stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        leg_llm: Optional injected LLM used for leg-level rationales.
        accumulator_llm: Optional injected LLM used for full-slip rationales.

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

    diagnostics: list[str] = []
    resolved_leg_llm = leg_llm
    resolved_accumulator_llm = accumulator_llm

    if resolved_leg_llm is None:
        try:
            resolved_leg_llm = await get_llm("leg_rationale")
        except AllProvidersFailedError as exc:
            diagnostics.append(str(exc))
            resolved_leg_llm = None

    if resolved_accumulator_llm is None:
        try:
            resolved_accumulator_llm = await get_llm("accumulator_rationale")
        except AllProvidersFailedError as exc:
            diagnostics.append(str(exc))
            resolved_accumulator_llm = None

    explained_accumulators: list[ExplainedAccumulator] = []
    for slip in validated_state.accumulators:
        explained_legs: list[AccumulatorLeg] = []

        for leg in slip.legs:
            rationale, leg_diagnostics = await _explain_leg(
                leg,
                llm=resolved_leg_llm,
            )
            explained_legs.append(leg.model_copy(update={"rationale": rationale}))
            diagnostics.extend(leg_diagnostics)

        slip_with_leg_rationales = slip.model_copy(update={"legs": tuple(explained_legs)})
        accumulator_rationale, accumulator_diagnostics = await _explain_accumulator(
            slip_with_leg_rationales,
            llm=resolved_accumulator_llm,
        )
        diagnostics.extend(accumulator_diagnostics)

        explained_accumulators.append(
            ExplainedAccumulator.model_validate(
                {
                    **slip_with_leg_rationales.model_dump(),
                    "legs": tuple(explained_legs),
                    "rationale": accumulator_rationale,
                }
            )
        )

    return {
        "current_stage": PipelineStage.APPROVAL,
        "explained_accumulators": explained_accumulators,
        "errors": _merge_diagnostics(validated_state.errors, diagnostics),
    }


async def _explain_leg(
    leg: AccumulatorLeg,
    *,
    llm: BaseChatModel | None,
) -> tuple[str, tuple[str, ...]]:
    """Generate one leg rationale, falling back deterministically on failure.

    Inputs:
        leg: Canonical accumulator leg awaiting a delivery-ready rationale.
        llm: Optional explanation-model instance. Missing models trigger the
            deterministic fallback rationale path.

    Outputs:
        A tuple of `(rationale, diagnostics)` where `rationale` is always
        non-blank and `diagnostics` captures any recoverable explanation
        issues encountered while attempting LLM generation.
    """

    fallback_rationale = _build_fallback_leg_rationale(leg)
    if llm is None:
        return fallback_rationale, ()

    prompt = get_prompt("leg_rationale")
    prompt_messages = prompt.format_messages(
        fixture_summary=_render_fixture_summary(leg),
        selection_summary=_render_selection_summary(leg),
        score_summary=_render_leg_score_summary(leg),
        risk_notes=_render_leg_risk_notes(leg),
    )

    try:
        structured_llm = llm.with_structured_output(LegRationale)
        raw_result = await structured_llm.ainvoke(prompt_messages)
        rationale_result = (
            raw_result
            if isinstance(raw_result, LegRationale)
            else LegRationale.model_validate(raw_result)
        )
    except Exception as exc:
        return (
            fallback_rationale,
            (f"Leg explanation failed for {leg.fixture_ref}: {exc}",),
        )

    return (
        _combine_rationale_and_risk(
            rationale_result.rationale,
            rationale_result.key_risk,
            risk_label="Risk",
            max_length=DEFAULT_MAX_LEG_RATIONALE_LENGTH,
        ),
        (),
    )


async def _explain_accumulator(
    slip: AccumulatorSlip,
    *,
    llm: BaseChatModel | None,
) -> tuple[str, tuple[str, ...]]:
    """Generate one accumulator rationale with deterministic fallback support.

    Inputs:
        slip: Accumulator slip whose legs already contain their final leg-level
            rationales.
        llm: Optional explanation-model instance. Missing models trigger the
            deterministic fallback rationale path.

    Outputs:
        A tuple of `(rationale, diagnostics)` where `rationale` is always
        non-blank and `diagnostics` captures any recoverable slip-level
        explanation issues.
    """

    fallback_rationale = _build_fallback_accumulator_rationale(slip)
    if llm is None:
        return fallback_rationale, ()

    prompt = get_prompt("accumulator_rationale")
    prompt_messages = prompt.format_messages(
        slip_summary=_render_slip_summary(slip),
        legs_summary=_render_legs_summary(slip.legs),
        confidence_summary=_render_confidence_summary(slip),
        portfolio_note=_render_portfolio_note(slip),
    )

    try:
        structured_llm = llm.with_structured_output(AccumulatorRationale)
        raw_result = await structured_llm.ainvoke(prompt_messages)
        rationale_result = (
            raw_result
            if isinstance(raw_result, AccumulatorRationale)
            else AccumulatorRationale.model_validate(raw_result)
        )
    except Exception as exc:
        return (
            fallback_rationale,
            (f"Accumulator explanation failed for slip {slip.slip_number}: {exc}",),
        )

    return (
        _combine_rationale_and_risk(
            rationale_result.rationale,
            rationale_result.shared_risk,
            risk_label="Shared risk",
            max_length=DEFAULT_MAX_RATIONALE_LENGTH,
        ),
        (),
    )


def _render_fixture_summary(leg: AccumulatorLeg) -> str:
    """Build a compact fixture label suitable for explanation prompts."""

    return (
        f"{leg.home_team} vs {leg.away_team} "
        f"({leg.competition}, {leg.sport.value})"
    )


def _render_selection_summary(leg: AccumulatorLeg) -> str:
    """Render one human-readable selection summary for prompt grounding."""

    line_suffix = f" line {leg.line:+.1f}" if leg.line is not None else ""
    market_label = leg.market_label or leg.market.value.replace("_", " ")
    return (
        f"{market_label}: {leg.selection} @ {leg.odds:.2f} odds"
        f"{line_suffix} via {leg.provider}"
    )


def _render_leg_score_summary(leg: AccumulatorLeg) -> str:
    """Render the strongest structured leg evidence for rationale prompts."""

    evidence = leg.rationale or (
        f"This leg carries a { _describe_confidence_band(leg.confidence) } confidence signal."
    )
    return (
        f"Leg confidence {leg.confidence:.2f}; "
        f"resolution source {leg.resolution_source.value}; "
        f"existing edge summary: {evidence}"
    )


def _render_leg_risk_notes(leg: AccumulatorLeg) -> str:
    """Render concise leg-specific risk notes for prompt grounding."""

    risk = _derive_leg_risk(leg)
    return risk or "No material additional caveat beyond standard market variance."


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
            f"({leg.market.value}) @ {leg.odds:.2f}; "
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


def _build_fallback_leg_rationale(leg: AccumulatorLeg) -> str:
    """Construct one deterministic leg rationale when LLM output is unavailable."""

    base_rationale = leg.rationale or (
        f"{leg.selection.title()} is the preferred angle because this leg still carries a "
        f"{_describe_confidence_band(leg.confidence)} confidence signal at {leg.odds:.2f} odds."
    )
    return _combine_rationale_and_risk(
        base_rationale,
        _derive_leg_risk(leg),
        risk_label="Risk",
        max_length=DEFAULT_MAX_LEG_RATIONALE_LENGTH,
    )


def _build_fallback_accumulator_rationale(slip: AccumulatorSlip) -> str:
    """Construct one deterministic accumulator rationale without LLM help."""

    competitions = len({leg.competition for leg in slip.legs})
    competition_label = "competition" if competitions == 1 else "competitions"
    rationale = (
        f"This slip leans on {slip.leg_count} { _describe_confidence_band(slip.confidence) } "
        f"legs across {competitions} {competition_label}, aiming to keep the overall profile "
        f"coherent at {slip.total_odds:.2f} combined odds."
    )
    return _combine_rationale_and_risk(
        rationale,
        _derive_accumulator_risk(slip),
        risk_label="Shared risk",
        max_length=DEFAULT_MAX_RATIONALE_LENGTH,
    )


def _derive_leg_risk(leg: AccumulatorLeg) -> str | None:
    """Infer one concise leg-level caveat from the structured leg metadata."""

    if leg.confidence < 0.60:
        return "Confidence is lower than the slate's top legs."
    if leg.resolution_source.value == "external_odds":
        return "This pick relies on external odds fallback rather than direct SportyBet resolution."
    if leg.line is not None and leg.market in _LINE_MARKETS:
        return "Line-based markets can swing on narrow late margins."
    if leg.odds >= 2.40:
        return "Higher odds improve payout but add more variance."
    return None


def _derive_accumulator_risk(slip: AccumulatorSlip) -> str | None:
    """Infer one concise shared-risk note from the slip's composition."""

    if slip.leg_count >= 5:
        return "More legs increase the chance that one volatile result breaks the slip."
    if any(leg.resolution_source.value == "external_odds" for leg in slip.legs):
        return (
            "At least one leg depends on external-odds fallback instead of direct "
            "SportyBet data."
        )
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


def _merge_diagnostics(existing_errors: Sequence[str], diagnostics: Sequence[str]) -> list[str]:
    """Append new diagnostics to the pipeline error list without duplicates."""

    merged_errors = list(existing_errors)
    seen_messages = {message.casefold() for message in merged_errors}

    for diagnostic in diagnostics:
        lookup_key = diagnostic.casefold()
        if lookup_key in seen_messages:
            continue
        seen_messages.add(lookup_key)
        merged_errors.append(diagnostic)

    return merged_errors


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
