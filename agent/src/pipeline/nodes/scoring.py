"""LLM scoring node for PuntLab's LangGraph pipeline.

Purpose: translate SportyBet fixtures, fixture details, markets, injuries, and
research-stage contexts into model-selected `MatchScore` outputs for ranking.
Scope: fixture-by-fixture LLM market scoring with prompt-level JSON validation
and fail-fast behavior.
Dependencies: `src.llm`, SportyBet market snapshots, and validated pipeline
state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.llm import get_llm, get_prompt
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.nodes.research import (
    _build_fixture_summary,
    _render_fixture_details,
    _render_known_absences,
)
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import build_fixture_market_snapshots
from src.schemas.analysis import MatchContext, MatchScore, ScoreFactorBreakdown
from src.schemas.stats import InjuryData
from src.scoring import ScoringEngine


async def scoring_node(
    state: PipelineState | Mapping[str, Any],
    *,
    engine: ScoringEngine | None = None,
) -> dict[str, object]:
    """Execute the scoring stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        engine: Optional injected scoring engine used by tests or custom
            runtime wiring.

    Outputs:
        A partial state update containing ordered `MatchScore` records, merged
        diagnostics, and the next stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    if engine is not None:
        raise RuntimeError(
            "scoring_node is LLM-led; deterministic ScoringEngine injection is disabled."
        )

    context_by_fixture = _context_index(validated_state.match_contexts)
    fixture_details_by_fixture = {
        details.fixture_ref: details for details in validated_state.fixture_details
    }
    market_snapshots = build_fixture_market_snapshots(
        validated_state.fixtures,
        validated_state.odds_market_catalog,
    )
    market_snapshot_by_fixture = {
        snapshot.fixture_ref: snapshot for snapshot in market_snapshots
    }
    resolved_llm = await get_llm("market_scoring")
    prompt = get_prompt("market_scoring")
    match_scores: list[MatchScore] = []

    for fixture in validated_state.fixtures:
        fixture_context = context_by_fixture.get(fixture.get_fixture_ref())
        if fixture_context is None:
            raise RuntimeError(
                f"LLM scoring requires research context for {fixture.get_fixture_ref()}."
            )
        fixture_injuries = _fixture_injuries(fixture, validated_state.injuries)
        prompt_messages = prompt.format_messages(
            fixture_summary=_build_fixture_summary(fixture),
            match_context_summary=_render_match_context_summary(fixture_context),
            fixture_details=_render_fixture_details(
                fixture_details_by_fixture.get(fixture.get_fixture_ref())
            ),
            known_absences=_render_known_absences(fixture, fixture_injuries or ()),
            market_menu=_render_scoring_market_menu(
                market_snapshot_by_fixture.get(fixture.get_fixture_ref())
            ),
        )
        score = await invoke_json_schema(
            llm=resolved_llm,
            prompt_messages=prompt_messages,
            schema=MatchScore,
            instruction=_match_score_json_instruction(),
        )
        match_scores.append(
            score.model_copy(
                update={
                    "fixture_ref": fixture.get_fixture_ref(),
                    "sport": fixture.sport,
                    "competition": fixture.competition,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "factors": score.factors
                    if isinstance(score.factors, ScoreFactorBreakdown)
                    else ScoreFactorBreakdown.model_validate(score.factors),
                }
            )
        )

    return {
        "current_stage": PipelineStage.RANKING,
        "match_scores": match_scores,
        "errors": list(validated_state.errors),
    }


def _context_index(contexts: Sequence[MatchContext]) -> dict[str, MatchContext]:
    """Index research contexts by fixture reference, keeping first occurrence.

    Inputs:
        contexts: Ordered context outputs from the research stage.

    Outputs:
        A dictionary keyed by `fixture_ref` for fast fixture-to-context lookup.
    """

    indexed_contexts: dict[str, MatchContext] = {}
    for context in contexts:
        if context.fixture_ref is None:
            continue
        indexed_contexts.setdefault(context.fixture_ref, context)
    return indexed_contexts


def _fixture_injuries(
    fixture: object,
    injuries: Sequence[InjuryData],
) -> tuple[InjuryData, ...] | None:
    """Return fixture-scoped injuries or `None` when no structured rows exist.

    Inputs:
        fixture: Fixture currently being scored.
        injuries: Full state-level structured injury slate.

    Outputs:
        Fixture-scoped injuries when structured availability rows exist, or
        `None` when the fixture has no structured injury data and the scoring
        engine should remain conservative.
    """

    fixture_ref = fixture.get_fixture_ref()
    fixture_rows = tuple(injury for injury in injuries if injury.fixture_ref == fixture_ref)
    return fixture_rows or None


def _render_match_context_summary(context: MatchContext) -> str:
    """Render one research context for the market-scoring prompt."""

    parts = [
        f"fixture_ref={context.fixture_ref}",
        f"qualitative_score={context.qualitative_score:.2f}",
        f"fixture_detail_summary={context.fixture_detail_summary}",
    ]
    optional_fields = (
        ("tactical_context", context.tactical_context),
        ("statistical_context", context.statistical_context),
        ("availability_context", context.availability_context),
        ("market_context", context.market_context),
        ("supplemental_news_context", context.supplemental_news_context),
    )
    for label, value in optional_fields:
        if value:
            parts.append(f"{label}={value}")
    parts.append(f"data_sources={', '.join(context.data_sources)}")
    return "; ".join(parts)


def _match_score_json_instruction() -> str:
    """Return the strict JSON contract for model-led market scoring."""

    return (
        "Return a MatchScore JSON object with keys: fixture_ref, sport, competition, "
        "home_team, away_team, composite_score, confidence, factors, recommended_market, "
        "recommended_market_label, recommended_canonical_market, recommended_selection, "
        "recommended_odds, recommended_line, qualitative_summary. factors must include "
        "form, h2h, injury_impact, odds_value, context, venue, statistical. Use numbers "
        "between 0 and 1 for score fields. recommended_market must be an exact provider "
        "market key from the menu. recommended_selection and recommended_odds must match "
        "one exact menu selection. Use null only for recommended_canonical_market or "
        "recommended_line when unavailable."
    )


def _render_scoring_market_menu(snapshot: object | None) -> str:
    """Render a compact market menu for LLM scoring."""

    if snapshot is None or getattr(snapshot, "fetched_market_count", 0) == 0:
        return "No SportyBet market snapshot was available for this fixture."

    priority_tokens = (
        "1x2",
        "full_time",
        "result",
        "double",
        "draw",
        "btts",
        "both",
        "over",
        "under",
        "total",
        "goals",
        "handicap",
    )
    market_entries: list[object] = []
    for group in snapshot.market_groups:
        for market in group.markets:
            key_text = (
                f"{market.provider_market_key} {market.market_label} "
                f"{market.provider_market_name}"
            ).casefold()
            priority = 0 if any(token in key_text for token in priority_tokens) else 1
            market_entries.append((priority, market))

    ordered_entries = [market for _, market in sorted(market_entries, key=lambda item: item[0])]
    lines = [
        (
            "SportyBet compact scoring menu: "
            f"{snapshot.fetched_market_count} markets fetched; showing "
            f"{min(len(ordered_entries), 30)} prioritized markets."
        )
    ]
    for market in ordered_entries[:30]:
        selections = " | ".join(
            _format_scoring_selection(selection)
            for selection in market.selections[:6]
        )
        lines.append(f"- {market.market_label} [key={market.provider_market_key}]: {selections}")
    if len(ordered_entries) > 30:
        lines.append(f"- {len(ordered_entries) - 30} additional markets omitted.")
    return "\n".join(lines)


def _format_scoring_selection(selection: object) -> str:
    """Render one selection for the compact scoring menu."""

    label = selection.provider_selection_name
    if selection.line is not None and str(selection.line) not in label:
        label += f" ({selection.line:g})"
    return f"{label} {selection.odds:.2f}"


__all__ = ["scoring_node"]
