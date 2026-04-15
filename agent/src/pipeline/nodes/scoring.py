"""Scoring node for PuntLab's LangGraph pipeline.

Purpose: translate ingested fixtures, canonically mapped odds, team stats, and
research-stage contexts into validated `MatchScore` outputs for ranking.
Scope: fixture-by-fixture scoring orchestration, conservative injury/context
matching, and recoverable diagnostics when individual fixtures cannot be
scored cleanly.
Dependencies: `src.scoring.ScoringEngine` for composite score calculation and
`src.pipeline.state.PipelineState` for validated state IO between LangGraph
stages.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchContext, MatchScore
from src.schemas.fixtures import NormalizedFixture
from src.schemas.stats import InjuryData, TeamStats
from src.scoring import ScoringEngine

_SCORING_FAILURE_PATTERN = re.compile(
    r"^Scoring failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_FIXTURE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:sr:match:\d+|football-data:\d+|api-football:\d+|balldontlie:\d+|the-odds-api:[a-z0-9_-]+)\b",
    flags=re.IGNORECASE,
)


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
    scoring_engine = engine or ScoringEngine()

    context_by_fixture = _context_index(validated_state.match_contexts)
    diagnostics: list[str] = []
    match_scores: list[MatchScore] = []

    for fixture in validated_state.fixtures:
        team_stats = _fixture_team_stats(fixture, validated_state.team_stats)
        fixture_context = context_by_fixture.get(fixture.get_fixture_ref())
        fixture_injuries = _fixture_injuries(fixture, validated_state.injuries)

        try:
            match_scores.append(
                scoring_engine.calculate_match_score(
                    fixture,
                    team_stats,
                    validated_state.odds_data,
                    context=fixture_context,
                    injuries=fixture_injuries,
                    h2h_data=None,
                )
            )
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                f"Scoring failed for {fixture.get_fixture_ref()}: {exc}"
            )

    return {
        "current_stage": PipelineStage.RANKING,
        "match_scores": match_scores,
        "errors": _merge_diagnostics(
            validated_state.errors,
            _summarize_scoring_diagnostics(diagnostics),
        ),
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


def _fixture_team_stats(
    fixture: NormalizedFixture,
    team_stats: Sequence[TeamStats],
) -> tuple[TeamStats, ...]:
    """Return same-sport team stats that could plausibly match the fixture.

    Inputs:
        fixture: Fixture currently being scored.
        team_stats: Full state-level team-stat slate gathered during ingestion.

    Outputs:
        A tuple of same-sport team snapshots, prioritizing rows that match the
        fixture's team IDs or team names while preserving source order.
    """

    relevant_stats = tuple(stats for stats in team_stats if stats.sport == fixture.sport)
    if not relevant_stats:
        return ()

    exact_matches = tuple(
        stats
        for stats in relevant_stats
        if _team_stats_match_fixture_team(stats, fixture)
    )
    return exact_matches or relevant_stats


def _team_stats_match_fixture_team(
    stats: TeamStats,
    fixture: NormalizedFixture,
) -> bool:
    """Report whether a team-stat row likely belongs to either fixture team."""

    team_name = stats.team_name.casefold()
    if team_name in {fixture.home_team.casefold(), fixture.away_team.casefold()}:
        return True
    if stats.team_id in {fixture.home_team_id, fixture.away_team_id}:
        return True
    return False


def _fixture_injuries(
    fixture: NormalizedFixture,
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

    fixture_rows = tuple(
        injury for injury in injuries if injury.fixture_ref == fixture.get_fixture_ref()
    )
    return fixture_rows or None


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


def _summarize_scoring_diagnostics(diagnostics: Sequence[str]) -> tuple[str, ...]:
    """Compress repetitive fixture-level scoring failures by shared reason.

    Inputs:
        diagnostics: Ordered scoring diagnostics collected per fixture.

    Outputs:
        A tuple where repeated fixture-specific failures are grouped into one
        message per root cause, while one-off diagnostics remain unchanged.
    """

    summarized: list[str] = []
    failures_by_reason: dict[str, list[str]] = {}
    reason_examples: dict[str, str] = {}

    for diagnostic in diagnostics:
        match = _SCORING_FAILURE_PATTERN.match(diagnostic)
        if match is None:
            summarized.append(diagnostic)
            continue
        raw_reason = match.group("reason").strip()
        reason = _normalize_scoring_failure_reason(raw_reason)
        fixture_ref = match.group("fixture").strip()
        failures_by_reason.setdefault(reason, []).append(fixture_ref)
        reason_examples.setdefault(reason, raw_reason)

    for reason, fixture_refs in failures_by_reason.items():
        if len(fixture_refs) == 1:
            reason_for_single = reason_examples.get(reason, reason)
            summarized.append(
                f"Scoring failed for {fixture_refs[0]}: {reason_for_single}"
            )
            continue
        normalized_reason = reason.rstrip(".")
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Scoring failed for "
            f"{len(fixture_refs)} fixtures due to: {normalized_reason}. "
            f"Sample fixtures: {sample_refs}."
        )

    return tuple(summarized)


def _normalize_scoring_failure_reason(reason: str) -> str:
    """Normalize variable fixture-specific fragments in scoring failures.

    Inputs:
        reason: Raw exception text raised while scoring one fixture.

    Outputs:
        A normalized root-cause string that keeps semantic detail while
        removing fixture-specific identifiers that fragment aggregation.
    """

    normalized = " ".join(reason.split())
    normalized = _FIXTURE_IDENTIFIER_PATTERN.sub("<fixture>", normalized)
    return normalized


__all__ = ["scoring_node"]
