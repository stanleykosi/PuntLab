"""Accumulator-building node for PuntLab's LangGraph pipeline.

Purpose: transform resolved ranked-match opportunities into the canonical
daily accumulator slip slate consumed by explanation and delivery stages.
Scope: invoke the shared accumulator builder, pass through the current run
date and optional UUID run identifier, and surface stage-level diagnostics
when no slips can be generated from the available candidate pool.
Dependencies: `src.accumulators.AccumulatorBuilder` for slip generation and
`src.pipeline.state.PipelineState` for validated LangGraph state exchange.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from src.accumulators import AccumulatorBuilder
from src.pipeline.state import PipelineStage, PipelineState


async def accumulator_building_node(
    state: PipelineState | Mapping[str, Any],
    *,
    builder: AccumulatorBuilder | None = None,
) -> dict[str, object]:
    """Execute the accumulator-building stage and return LangGraph updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        builder: Optional injected accumulator builder for tests or explicit
            runtime wiring.

    Outputs:
        A partial LangGraph update containing the generated accumulator slips,
        merged diagnostics, and the next stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    stage_builder = builder or AccumulatorBuilder()
    diagnostics: list[str] = []

    try:
        accumulators = list(
            stage_builder.build_accumulators(
                validated_state.ranked_matches,
                validated_state.resolved_markets,
                slip_date=validated_state.run_date,
                run_id=_coerce_run_uuid(validated_state.run_id),
            )
        )
    except (TypeError, ValueError) as exc:
        diagnostics.append(
            f"Accumulator building failed for run {validated_state.run_id}: {exc}"
        )
        accumulators = []

    return {
        "current_stage": PipelineStage.EXPLANATION,
        "accumulators": accumulators,
        "errors": _merge_diagnostics(validated_state.errors, diagnostics),
    }


def _coerce_run_uuid(run_id: str) -> UUID | None:
    """Parse UUID-like run identifiers while tolerating descriptive run IDs.

    Inputs:
        run_id: Pipeline run identifier stored in state as a non-blank string.

    Outputs:
        A parsed UUID when the run identifier uses UUID formatting, otherwise
        `None` so accumulator generation can proceed without synthetic IDs.
    """

    try:
        return UUID(run_id)
    except ValueError:
        return None


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


__all__ = ["accumulator_building_node"]
