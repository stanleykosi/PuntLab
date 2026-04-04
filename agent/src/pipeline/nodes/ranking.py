"""Ranking node for PuntLab's LangGraph pipeline.

Purpose: transform scored fixtures into the canonical globally ranked slate
consumed by market resolution and accumulator building.
Scope: validate scoring-stage outputs, apply deterministic global ordering,
and assign 1-based ranks for every scoreable match in the daily run.
Dependencies: `src.pipeline.state.PipelineState` for validated state IO and
`src.schemas.analysis` for `MatchScore` to `RankedMatch` conversion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchScore, RankedMatch


async def ranking_node(state: PipelineState | Mapping[str, Any]) -> dict[str, object]:
    """Execute the ranking stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state, either as a validated `PipelineState`
            instance or a raw mapping that can be validated into one.

    Outputs:
        A partial LangGraph update containing globally ordered
        `RankedMatch` rows plus the next-stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    ordered_scores = _order_match_scores(validated_state.match_scores)
    ranked_matches = [
        RankedMatch.model_validate(
            {
                **score.model_dump(),
                "rank": rank,
            }
        )
        for rank, score in enumerate(ordered_scores, start=1)
    ]

    return {
        "current_stage": PipelineStage.MARKET_RESOLUTION,
        "ranked_matches": ranked_matches,
        "errors": list(validated_state.errors),
    }


def _order_match_scores(match_scores: Sequence[MatchScore]) -> list[MatchScore]:
    """Return deterministically ordered match scores for global ranking.

    Inputs:
        match_scores: Ordered scoring-stage results for the current run.

    Outputs:
        A new list sorted primarily by descending composite score, then by
        descending confidence, and finally by stable fixture identity fields
        so ties resolve reproducibly across runs.

    Raises:
        TypeError: If any supplied item is not a canonical `MatchScore`.
    """

    normalized_scores = list(match_scores)
    for score in normalized_scores:
        if not isinstance(score, MatchScore):
            raise TypeError("ranking_node expects MatchScore instances only.")

    # Deterministic tie-breakers keep downstream slips stable when fixtures
    # land on the same model score and confidence.
    normalized_scores.sort(key=_ranking_sort_key)
    return normalized_scores


def _ranking_sort_key(score: MatchScore) -> tuple[float, float, str, str, str, str]:
    """Build the canonical sort key for one scored fixture."""

    return (
        -score.composite_score,
        -score.confidence,
        score.fixture_ref.casefold(),
        score.competition.casefold(),
        score.home_team.casefold(),
        score.away_team.casefold(),
    )


__all__ = ["ranking_node"]
