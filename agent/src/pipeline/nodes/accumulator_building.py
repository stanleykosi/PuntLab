"""LLM accumulator-building node for PuntLab's LangGraph pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from src.accumulators import AccumulatorBuilder
from src.llm import get_llm, get_prompt
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorLeg, AccumulatorSlip, ResolvedMarket
from src.schemas.analysis import RankedMatch
from src.schemas.llm_decisions import LLMAccumulatorPortfolio


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
    if not validated_state.ranked_matches or not validated_state.resolved_markets:
        raise RuntimeError("accumulator_building_node requires ranked and resolved LLM outputs.")

    target_count = (
        builder.target_count if builder is not None else AccumulatorBuilder().target_count
    )
    llm = await get_llm("accumulator_builder")
    prompt = get_prompt("accumulator_builder")
    portfolio = await invoke_json_schema(
        llm=llm,
        prompt_messages=prompt.format_messages(
            run_date=validated_state.run_date.isoformat(),
            target_count=target_count,
            resolved_leg_menu=_render_resolved_leg_menu(
                validated_state.ranked_matches,
                validated_state.resolved_markets,
            ),
        ),
        schema=LLMAccumulatorPortfolio,
        instruction=(
            "Return only this JSON shape: {\"slips\":[{\"slip_number\":1,"
            "\"leg_fixture_refs\":[\"fixture-ref-a\",\"fixture-ref-b\"],"
            "\"confidence\":0.72,\"strategy\":\"balanced\",\"rationale\":\"short reason\"}]}. "
            "Use only supplied fixture refs."
        ),
    )
    accumulators = _build_accumulators_from_llm(
        portfolio=portfolio,
        ranked_matches=validated_state.ranked_matches,
        resolved_markets=validated_state.resolved_markets,
        slip_date=validated_state.run_date,
        run_id=_coerce_run_uuid(validated_state.run_id),
    )

    return {
        "current_stage": PipelineStage.EXPLANATION,
        "accumulators": accumulators,
        "errors": list(validated_state.errors),
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


def _render_resolved_leg_menu(
    ranked_matches: Sequence[RankedMatch],
    resolved_markets: Sequence[ResolvedMarket],
) -> str:
    """Render resolved legs for the accumulator-builder prompt."""

    ranked_by_ref = {match.fixture_ref: match for match in ranked_matches}
    lines: list[str] = []
    for market in resolved_markets:
        ranked = ranked_by_ref.get(market.fixture_ref)
        if ranked is None:
            continue
        lines.append(
            f"- fixture_ref={market.fixture_ref}; rank={ranked.rank}; "
            f"{market.home_team} vs {market.away_team}; league={market.competition}; "
            f"market={market.market_label or market.provider_market_name} [key={market.market}]; "
            f"selection={market.provider_selection_name}; odds={market.odds:.2f}; "
            f"line={market.line}; confidence={ranked.confidence:.2f}; "
            f"score={ranked.composite_score:.2f}"
        )
    return "\n".join(lines)


def _build_accumulators_from_llm(
    *,
    portfolio: LLMAccumulatorPortfolio,
    ranked_matches: Sequence[RankedMatch],
    resolved_markets: Sequence[ResolvedMarket],
    slip_date: object,
    run_id: UUID | None,
) -> list[AccumulatorSlip]:
    """Translate LLM-selected fixture combinations into validated slips."""

    ranked_by_ref = {match.fixture_ref: match for match in ranked_matches}
    market_by_ref = {market.fixture_ref: market for market in resolved_markets}
    accumulators: list[AccumulatorSlip] = []
    for slip_choice in portfolio.slips:
        legs: list[AccumulatorLeg] = []
        for leg_number, fixture_ref in enumerate(slip_choice.leg_fixture_refs, start=1):
            ranked = ranked_by_ref.get(fixture_ref)
            market = market_by_ref.get(fixture_ref)
            if ranked is None or market is None:
                raise ValueError(
                    f"LLM accumulator selected unknown or unresolved fixture {fixture_ref!r}."
                )
            legs.append(
                AccumulatorLeg(
                    leg_number=leg_number,
                    fixture_ref=fixture_ref,
                    sport=ranked.sport,
                    competition=ranked.competition,
                    home_team=ranked.home_team,
                    away_team=ranked.away_team,
                    market=market.market,
                    canonical_market=market.canonical_market,
                    selection=market.provider_selection_name,
                    odds=market.odds,
                    provider=market.provider,
                    confidence=ranked.confidence,
                    resolution_source=market.resolution_source,
                    market_label=market.market_label,
                    line=market.line,
                    rationale=ranked.qualitative_summary or slip_choice.rationale,
                    sportybet_url=market.sportybet_url,
                )
            )
        total_odds = 1.0
        for leg in legs:
            total_odds *= leg.odds
        accumulators.append(
            AccumulatorSlip(
                run_id=run_id,
                slip_date=slip_date,
                slip_number=slip_choice.slip_number,
                legs=tuple(legs),
                total_odds=round(total_odds, 3),
                leg_count=len(legs),
                confidence=slip_choice.confidence,
                strategy=slip_choice.strategy,
            )
        )
    return accumulators


__all__ = ["accumulator_building_node"]
