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

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
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
_LEG_FAILURE_PATTERN = re.compile(
    r"^Leg explanation failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_ACCUMULATOR_FAILURE_PATTERN = re.compile(
    r"^Accumulator explanation failed for slip (?P<slip>\d+): (?P<reason>.+)$"
)
type ExplanationProgressCallback = Callable[[Mapping[str, object]], None]


async def explanation_node(
    state: PipelineState | Mapping[str, Any],
    *,
    leg_llm: BaseChatModel | None = None,
    accumulator_llm: BaseChatModel | None = None,
    progress_callback: ExplanationProgressCallback | None = None,
) -> dict[str, object]:
    """Execute the explanation stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        leg_llm: Optional injected LLM used for leg-level rationales.
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

    diagnostics: list[str] = []
    resolved_leg_llm = leg_llm
    resolved_accumulator_llm = accumulator_llm
    leg_llm_disabled_reason_emitted = False
    accumulator_llm_disabled_reason_emitted = False
    total_slips = len(validated_state.accumulators)
    _emit_progress(
        progress_callback,
        event="stage_started",
        total_slips=total_slips,
    )

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
    for slip_index, slip in enumerate(validated_state.accumulators, start=1):
        _emit_progress(
            progress_callback,
            event="slip_started",
            slip_number=slip.slip_number,
            slip_index=slip_index,
            total_slips=total_slips,
            leg_count=len(slip.legs),
        )
        explained_legs: list[AccumulatorLeg] = []

        for leg_index, leg in enumerate(slip.legs, start=1):
            _emit_progress(
                progress_callback,
                event="leg_started",
                slip_number=slip.slip_number,
                slip_index=slip_index,
                total_slips=total_slips,
                leg_index=leg_index,
                leg_number=leg.leg_number,
                slip_total_legs=len(slip.legs),
                fixture_ref=leg.fixture_ref,
            )
            leg_started_at = perf_counter()
            rationale, leg_diagnostics, disable_leg_llm = await _explain_leg(
                leg,
                llm=resolved_leg_llm,
            )
            explained_legs.append(leg.model_copy(update={"rationale": rationale}))
            diagnostics.extend(leg_diagnostics)
            if disable_leg_llm:
                resolved_leg_llm = None
                if not leg_llm_disabled_reason_emitted:
                    diagnostics.append(
                        "Leg rationale LLM was disabled for the remaining slips due to "
                        "upstream rate limiting."
                    )
                    leg_llm_disabled_reason_emitted = True
            _emit_progress(
                progress_callback,
                event="leg_completed",
                slip_number=slip.slip_number,
                slip_index=slip_index,
                total_slips=total_slips,
                leg_index=leg_index,
                leg_number=leg.leg_number,
                slip_total_legs=len(slip.legs),
                fixture_ref=leg.fixture_ref,
                duration_seconds=perf_counter() - leg_started_at,
                had_error=bool(leg_diagnostics),
            )

        slip_with_leg_rationales = slip.model_copy(update={"legs": tuple(explained_legs)})
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
            slip_with_leg_rationales,
            llm=resolved_accumulator_llm,
        )
        diagnostics.extend(accumulator_diagnostics)
        if disable_accumulator_llm:
            resolved_accumulator_llm = None
            if not accumulator_llm_disabled_reason_emitted:
                diagnostics.append(
                    "Accumulator rationale LLM was disabled for the remaining slips due to "
                    "upstream rate limiting."
                )
                accumulator_llm_disabled_reason_emitted = True

        explained_accumulators.append(
            ExplainedAccumulator.model_validate(
                {
                    **slip_with_leg_rationales.model_dump(),
                    "legs": tuple(explained_legs),
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
        "errors": _merge_diagnostics(
            validated_state.errors,
            _summarize_explanation_diagnostics(diagnostics),
        ),
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


async def _explain_leg(
    leg: AccumulatorLeg,
    *,
    llm: BaseChatModel | None,
) -> tuple[str, tuple[str, ...], bool]:
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
        return fallback_rationale, (), False

    prompt = get_prompt("leg_rationale")
    prompt_messages = prompt.format_messages(
        fixture_summary=_render_fixture_summary(leg),
        selection_summary=_render_selection_summary(leg),
        score_summary=_render_leg_score_summary(leg),
        risk_notes=_render_leg_risk_notes(leg),
    )

    try:
        rationale_result = await _invoke_leg_rationale_model(
            llm=llm,
            prompt_messages=prompt_messages,
            fixture_ref=leg.fixture_ref,
        )
    except Exception as exc:
        disable_llm = _is_rate_limit_error(exc)
        normalized_reason = _normalize_explanation_error_reason(exc)
        return (
            fallback_rationale,
            (f"Leg explanation failed for {leg.fixture_ref}: {normalized_reason}",),
            disable_llm,
        )

    return (
        _combine_rationale_and_risk(
            rationale_result.rationale,
            rationale_result.key_risk,
            risk_label="Risk",
            max_length=DEFAULT_MAX_LEG_RATIONALE_LENGTH,
        ),
        (),
        False,
    )


async def _explain_accumulator(
    slip: AccumulatorSlip,
    *,
    llm: BaseChatModel | None,
) -> tuple[str, tuple[str, ...], bool]:
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
        return fallback_rationale, (), False

    prompt = get_prompt("accumulator_rationale")
    prompt_messages = prompt.format_messages(
        slip_summary=_render_slip_summary(slip),
        legs_summary=_render_legs_summary(slip.legs),
        confidence_summary=_render_confidence_summary(slip),
        portfolio_note=_render_portfolio_note(slip),
    )

    try:
        rationale_result = await _invoke_accumulator_rationale_model(
            llm=llm,
            prompt_messages=prompt_messages,
            slip_number=slip.slip_number,
        )
    except Exception as exc:
        disable_llm = _is_rate_limit_error(exc)
        normalized_reason = _normalize_explanation_error_reason(exc)
        return (
            fallback_rationale,
            (f"Accumulator explanation failed for slip {slip.slip_number}: {normalized_reason}",),
            disable_llm,
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


async def _invoke_leg_rationale_model(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    fixture_ref: str,
) -> LegRationale:
    """Invoke the leg-rationale model and normalize provider payload variants."""

    if _is_openrouter_model(llm):
        return await _invoke_openrouter_leg_rationale_model(
            llm=llm,
            prompt_messages=prompt_messages,
            fixture_ref=fixture_ref,
        )

    structured_llm = llm.with_structured_output(LegRationale)
    raw_result = await structured_llm.ainvoke(prompt_messages)
    if isinstance(raw_result, LegRationale):
        return raw_result
    normalized_payload = _normalize_leg_payload(raw_result, fixture_ref=fixture_ref)
    return LegRationale.model_validate(normalized_payload)


async def _invoke_accumulator_rationale_model(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    slip_number: int,
) -> AccumulatorRationale:
    """Invoke the accumulator-rationale model with resilient payload handling."""

    if _is_openrouter_model(llm):
        return await _invoke_openrouter_accumulator_rationale_model(
            llm=llm,
            prompt_messages=prompt_messages,
            slip_number=slip_number,
        )

    structured_llm = llm.with_structured_output(AccumulatorRationale)
    raw_result = await structured_llm.ainvoke(prompt_messages)
    if isinstance(raw_result, AccumulatorRationale):
        return raw_result
    normalized_payload = _normalize_accumulator_payload(
        raw_result,
        slip_number=slip_number,
    )
    return AccumulatorRationale.model_validate(normalized_payload)


def _is_openrouter_model(llm: BaseChatModel) -> bool:
    """Return whether the resolved LLM instance points to OpenRouter."""

    endpoint = str(
        getattr(llm, "openai_api_base", None)
        or getattr(llm, "base_url", "")
        or ""
    )
    return "openrouter.ai" in endpoint.casefold()


async def _invoke_openrouter_leg_rationale_model(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    fixture_ref: str,
) -> LegRationale:
    """Invoke OpenRouter rationale generation via documented JSON schema modes.

    Inputs:
        llm: Configured OpenRouter-backed chat model.
        prompt_messages: Prompt message sequence for leg rationale generation.
        fixture_ref: Fixture identifier injected into normalized rationale rows.

    Outputs:
        A validated `LegRationale` record.
    """

    raw_result = await _invoke_openrouter_structured_output(
        llm=llm,
        schema=LegRationale,
        prompt_messages=prompt_messages,
    )
    normalized_payload = _normalize_leg_payload(raw_result, fixture_ref=fixture_ref)
    return LegRationale.model_validate(normalized_payload)


async def _invoke_openrouter_accumulator_rationale_model(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
    slip_number: int,
) -> AccumulatorRationale:
    """Invoke OpenRouter accumulator rationale via documented JSON schema modes.

    Inputs:
        llm: Configured OpenRouter-backed chat model.
        prompt_messages: Prompt message sequence for slip rationale generation.
        slip_number: Slip identifier injected into normalized rationale rows.

    Outputs:
        A validated `AccumulatorRationale` record.
    """

    raw_result = await _invoke_openrouter_structured_output(
        llm=llm,
        schema=AccumulatorRationale,
        prompt_messages=prompt_messages,
    )
    normalized_payload = _normalize_accumulator_payload(
        raw_result,
        slip_number=slip_number,
    )
    return AccumulatorRationale.model_validate(normalized_payload)


async def _invoke_openrouter_structured_output(
    *,
    llm: BaseChatModel,
    schema: type[LegRationale] | type[AccumulatorRationale],
    prompt_messages: Sequence[Any],
) -> object:
    """Run OpenRouter structured-output calls with JSON-schema-first strategy.

    Inputs:
        llm: Configured OpenRouter-backed chat model.
        schema: Pydantic schema expected from the model output.
        prompt_messages: Prompt message sequence.

    Outputs:
        Raw model output object from the first successful structured strategy.

    Raises:
        Exception: If both JSON schema and JSON mode fail.
    """

    json_schema_error: Exception | None = None
    try:
        json_schema_runner = llm.with_structured_output(schema, method="json_schema")
    except TypeError:
        json_schema_runner = None

    if json_schema_runner is not None:
        try:
            return await json_schema_runner.ainvoke(prompt_messages)
        except Exception as exc:
            json_schema_error = exc
            if not _should_retry_openrouter_structured_output(exc):
                raise

    try:
        json_mode_runner = llm.with_structured_output(schema, method="json_mode")
    except TypeError:
        json_mode_runner = None

    if json_mode_runner is None:
        if json_schema_error is not None:
            raise json_schema_error
        raise RuntimeError(
            "OpenRouter structured-output runner is unavailable: "
            "neither json_schema nor json_mode is supported by this model adapter."
        )

    return await json_mode_runner.ainvoke(prompt_messages)


def _should_retry_openrouter_structured_output(error: Exception) -> bool:
    """Return whether OpenRouter structured-output should fallback to JSON mode.

    Inputs:
        error: Exception raised while invoking JSON-schema structured output.

    Outputs:
        `True` when the error indicates unsupported strict-schema mode or known
        parser instability that JSON mode can avoid.
    """

    message = " ".join(str(error).split()).casefold()
    return any(
        token in message
        for token in (
            "response_format",
            "json_schema",
            "structured output",
            "does not support",
            "doesn't support",
            "unsupported",
            "nonetype",
            "tool_calls",
            "schema validation",
        )
    )


def _is_rate_limit_error(error: Exception) -> bool:
    """Return whether an exception indicates upstream LLM rate limiting."""

    message = " ".join(str(error).split()).casefold()
    return "429" in message or "rate limit" in message or "rate-limited" in message


def _normalize_explanation_error_reason(error: Exception) -> str:
    """Normalize verbose explanation errors into stable operator-facing reasons."""

    message = " ".join(str(error).split())
    lowered = message.casefold()
    if _is_rate_limit_error(error):
        return "LLM provider rate limit reached."
    if "nonetype" in lowered and "iterable" in lowered:
        return "LLM output schema validation failed."
    if "validation error" in lowered or "input should be an object" in lowered:
        return "LLM output schema validation failed."
    if "timed out" in lowered or "timeout" in lowered:
        return "LLM response timed out."
    return message


def _normalize_leg_payload(
    payload: object,
    *,
    fixture_ref: str,
) -> dict[str, object]:
    """Normalize provider payload variants into the `LegRationale` shape."""

    if isinstance(payload, LegRationale):
        return payload.model_dump(mode="python")
    if isinstance(payload, Mapping):
        rationale_value = payload.get("rationale")
        if not isinstance(rationale_value, str):
            rationale_value = payload.get("text")
        key_risk_value = payload.get("key_risk")
        if key_risk_value is not None and not isinstance(key_risk_value, str):
            key_risk_value = str(key_risk_value)
        fixture_value = payload.get("fixture_ref")
        if fixture_value is not None and not isinstance(fixture_value, str):
            fixture_value = str(fixture_value)
        return {
            "fixture_ref": fixture_value or fixture_ref,
            "rationale": rationale_value,
            "key_risk": key_risk_value,
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if payload and isinstance(payload[0], str):
            return {
                "fixture_ref": fixture_ref,
                "rationale": payload[0],
                "key_risk": None,
            }
    if isinstance(payload, str):
        return {
            "fixture_ref": fixture_ref,
            "rationale": payload,
            "key_risk": None,
        }
    raise ValueError("Leg rationale payload is not a supported object shape.")


def _normalize_accumulator_payload(
    payload: object,
    *,
    slip_number: int,
) -> dict[str, object]:
    """Normalize provider payload variants into `AccumulatorRationale` shape."""

    if isinstance(payload, AccumulatorRationale):
        return payload.model_dump(mode="python")
    if isinstance(payload, Mapping):
        rationale_value = payload.get("rationale")
        if not isinstance(rationale_value, str):
            rationale_value = payload.get("text")
        shared_risk_value = payload.get("shared_risk")
        if shared_risk_value is not None and not isinstance(shared_risk_value, str):
            shared_risk_value = str(shared_risk_value)
        slip_value = payload.get("slip_number")
        if slip_value is not None and not isinstance(slip_value, int):
            try:
                slip_value = int(str(slip_value))
            except ValueError:
                slip_value = slip_number
        return {
            "slip_number": slip_value or slip_number,
            "rationale": rationale_value,
            "shared_risk": shared_risk_value,
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if payload and isinstance(payload[0], str):
            return {
                "slip_number": slip_number,
                "rationale": payload[0],
                "shared_risk": None,
            }
    if isinstance(payload, str):
        return {
            "slip_number": slip_number,
            "rationale": payload,
            "shared_risk": None,
        }
    raise ValueError("Accumulator rationale payload is not a supported object shape.")


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


def _summarize_explanation_diagnostics(diagnostics: Sequence[str]) -> tuple[str, ...]:
    """Compress repetitive leg/slip explanation diagnostics by root cause."""

    summarized: list[str] = []
    leg_failures_by_reason: dict[str, list[str]] = {}
    slip_failures_by_reason: dict[str, list[str]] = {}

    for diagnostic in diagnostics:
        leg_match = _LEG_FAILURE_PATTERN.match(diagnostic)
        if leg_match is not None:
            reason = leg_match.group("reason").strip()
            fixture_ref = leg_match.group("fixture").strip()
            leg_failures_by_reason.setdefault(reason, []).append(fixture_ref)
            continue

        accumulator_match = _ACCUMULATOR_FAILURE_PATTERN.match(diagnostic)
        if accumulator_match is not None:
            reason = accumulator_match.group("reason").strip()
            slip_number = accumulator_match.group("slip").strip()
            slip_failures_by_reason.setdefault(reason, []).append(slip_number)
            continue

        summarized.append(diagnostic)

    for reason, fixture_refs in leg_failures_by_reason.items():
        if len(fixture_refs) == 1:
            summarized.append(f"Leg explanation failed for {fixture_refs[0]}: {reason}")
            continue
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Leg explanation failed for "
            f"{len(fixture_refs)} legs due to: {reason.rstrip('.')}. "
            f"Sample fixtures: {sample_refs}."
        )

    for reason, slip_numbers in slip_failures_by_reason.items():
        if len(slip_numbers) == 1:
            summarized.append(
                f"Accumulator explanation failed for slip {slip_numbers[0]}: {reason}"
            )
            continue
        sample_slips = ", ".join(slip_numbers[:4])
        summarized.append(
            "Accumulator explanation failed for "
            f"{len(slip_numbers)} slips due to: {reason.rstrip('.')}. "
            f"Sample slips: {sample_slips}."
        )

    return tuple(summarized)


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
