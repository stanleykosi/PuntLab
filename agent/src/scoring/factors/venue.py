"""Venue scoring factor for PuntLab's deterministic scoring engine.

Purpose: convert home/away split performance into a bounded venue-advantage
score for one fixture.
Scope: resolve the fixture's home and away teams from canonical team-stat
snapshots, favor strong home edges, and penalize resilient away travelers.
Dependencies: relies on the canonical `NormalizedFixture` and `TeamStats`
schemas from `src.schemas`.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Final

from src.schemas.fixtures import NormalizedFixture
from src.schemas.stats import TeamStats

_GOAL_DIFF_RANGE_PER_MATCH: Final[float] = 2.5
_MATCH_SAMPLE_TARGET: Final[int] = 10

type TeamStatsInput = TeamStats | Sequence[TeamStats]


def analyze_venue(
    fixture: NormalizedFixture,
    team_stats: TeamStatsInput,
) -> float:
    """Score the fixture's venue and home/away split dynamics.

    Inputs:
        `fixture`: the candidate match being scored.
        `team_stats`: one or more canonical team-stat snapshots that should
        include the fixture's home side, away side, or both.

    Outputs:
        A bounded score from `0.0` to `1.0`, where higher values indicate a
        stronger home-venue edge and a weaker away-travel profile.

    Raises:
        TypeError: If any team-stat item is not a `TeamStats` record.
        ValueError: If no supplied team stats can be matched to the fixture or
            if the sport does not match the fixture.
    """

    normalized_team_stats = _normalize_team_stats_input(fixture, team_stats)
    home_stats = _select_best_team_snapshot(
        normalized_team_stats,
        team_id=fixture.home_team_id,
        team_name=fixture.home_team,
    )
    away_stats = _select_best_team_snapshot(
        normalized_team_stats,
        team_id=fixture.away_team_id,
        team_name=fixture.away_team,
    )

    if home_stats is None and away_stats is None:
        raise ValueError(
            "analyze_venue could not match any TeamStats records to the fixture teams."
        )

    if home_stats is not None and away_stats is not None:
        home_advantage = _home_advantage_score(home_stats)
        away_resilience = _away_resilience_score(away_stats)
        strength_gap = _strength_gap_score(home_stats, away_stats)
        reliability = fmean((_sample_reliability(home_stats), _sample_reliability(away_stats)))
        return _clamp(
            (home_advantage * 0.50)
            + ((1.0 - away_resilience) * 0.20)
            + (strength_gap * 0.20)
            + (reliability * 0.10)
        )

    if home_stats is not None:
        return _clamp(0.35 + (_home_advantage_score(home_stats) * 0.65))

    assert away_stats is not None
    return _clamp(0.35 + ((1.0 - _away_resilience_score(away_stats)) * 0.65))


def _normalize_team_stats_input(
    fixture: NormalizedFixture,
    team_stats: TeamStatsInput,
) -> tuple[TeamStats, ...]:
    """Validate the venue-factor input and normalize it into a tuple.

    Args:
        fixture: Fixture whose sport and team identities must be respected.
        team_stats: One or more team-stat snapshots.

    Returns:
        A validated tuple of canonical `TeamStats` objects.

    Raises:
        TypeError: If any input item is not a `TeamStats`.
        ValueError: If no stats are provided or the sport is incompatible.
    """

    normalized: tuple[TeamStats, ...]
    if isinstance(team_stats, TeamStats):
        normalized = (team_stats,)
    else:
        normalized = tuple(team_stats)

    if not normalized:
        raise ValueError("analyze_venue requires at least one TeamStats record.")

    for stats in normalized:
        if not isinstance(stats, TeamStats):
            raise TypeError("analyze_venue expects TeamStats instances only.")
        if stats.sport != fixture.sport:
            raise ValueError(
                "analyze_venue requires team stats from the same sport as the fixture."
            )

    return normalized


def _select_best_team_snapshot(
    team_stats: Sequence[TeamStats],
    *,
    team_id: str | None,
    team_name: str,
) -> TeamStats | None:
    """Select the richest matching team snapshot for one fixture side."""

    matching_rows = tuple(
        stats
        for stats in team_stats
        if _matches_fixture_team(stats, team_id=team_id, team_name=team_name)
    )
    if not matching_rows:
        return None
    return max(matching_rows, key=_team_snapshot_quality)


def _matches_fixture_team(
    team_stats: TeamStats,
    *,
    team_id: str | None,
    team_name: str,
) -> bool:
    """Report whether a canonical team snapshot belongs to one fixture team."""

    if team_id is not None and team_stats.team_id == team_id:
        return True
    return team_stats.team_name.casefold() == team_name.casefold()


def _team_snapshot_quality(team_stats: TeamStats) -> float:
    """Estimate the usefulness of one team snapshot for venue analysis."""

    quality = float(team_stats.matches_played)
    if team_stats.form:
        quality += min(len(team_stats.form), 10) * 0.5
    if team_stats.avg_goals_scored is not None:
        quality += 1.0
    if team_stats.avg_goals_conceded is not None:
        quality += 1.0
    if team_stats.home_wins > 0 or team_stats.away_wins > 0:
        quality += 2.0
    return quality


def _home_advantage_score(team_stats: TeamStats) -> float:
    """Return one team's strength profile when playing at home."""

    estimated_home_matches = max(team_stats.matches_played / 2.0, 1.0)
    home_win_rate = _clamp(team_stats.home_wins / estimated_home_matches)
    home_share = 0.5 if team_stats.wins == 0 else _clamp(team_stats.home_wins / team_stats.wins)
    return _clamp(
        (home_win_rate * 0.50)
        + (home_share * 0.25)
        + (_goal_balance_score(team_stats) * 0.25)
    )


def _away_resilience_score(team_stats: TeamStats) -> float:
    """Return one team's strength profile when travelling away from home."""

    estimated_away_matches = max(team_stats.matches_played / 2.0, 1.0)
    away_win_rate = _clamp(team_stats.away_wins / estimated_away_matches)
    away_share = 0.5 if team_stats.wins == 0 else _clamp(team_stats.away_wins / team_stats.wins)
    return _clamp(
        (away_win_rate * 0.50)
        + (away_share * 0.25)
        + (_goal_balance_score(team_stats) * 0.25)
    )


def _strength_gap_score(home_stats: TeamStats, away_stats: TeamStats) -> float:
    """Return the overall strength gap between the home and away teams."""

    home_strength = _overall_strength_score(home_stats)
    away_strength = _overall_strength_score(away_stats)
    return _clamp(0.5 + ((home_strength - away_strength) / 2.0))


def _overall_strength_score(team_stats: TeamStats) -> float:
    """Return a bounded overall-strength estimate for one team snapshot."""

    matches_played = max(team_stats.matches_played, 1)
    win_rate = team_stats.wins / matches_played
    return _clamp((win_rate * 0.60) + (_goal_balance_score(team_stats) * 0.40))


def _goal_balance_score(team_stats: TeamStats) -> float:
    """Normalize overall goal or point differential into a bounded score."""

    if team_stats.avg_goals_scored is not None and team_stats.avg_goals_conceded is not None:
        goal_difference_per_match = team_stats.avg_goals_scored - team_stats.avg_goals_conceded
    else:
        matches_played = max(team_stats.matches_played, 1)
        goal_difference_per_match = (
            team_stats.goals_for - team_stats.goals_against
        ) / matches_played

    normalized = (goal_difference_per_match + _GOAL_DIFF_RANGE_PER_MATCH) / (
        _GOAL_DIFF_RANGE_PER_MATCH * 2
    )
    return _clamp(normalized)


def _sample_reliability(team_stats: TeamStats) -> float:
    """Estimate how trustworthy a team snapshot is from sample size."""

    return _clamp(team_stats.matches_played / _MATCH_SAMPLE_TARGET)


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


__all__ = ["analyze_venue"]
