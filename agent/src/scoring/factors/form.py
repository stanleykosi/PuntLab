"""Recent-form scoring factor for PuntLab's deterministic scoring engine.

Purpose: convert one or two normalized team-stat snapshots into a bounded
recent-form score that can be combined with the other deterministic factors.
Scope: evaluate win rate, compact form strings, and goal balance across the
last 5-10 matches without assuming provider-specific raw payloads.
Dependencies: relies on the canonical `TeamStats` schema from `src.schemas`.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Final

from src.schemas.stats import TeamStats

_MAX_FORM_WINDOW: Final[int] = 10
_GOAL_DIFF_RANGE_PER_MATCH: Final[float] = 2.5
_MATCH_SAMPLE_TARGET: Final[int] = 10
_RECENT_FORM_WEIGHT: Final[float] = 0.45
_WIN_RATE_WEIGHT: Final[float] = 0.30
_GOAL_BALANCE_WEIGHT: Final[float] = 0.25

type TeamStatsInput = TeamStats | Sequence[TeamStats]


def analyze_form(team_stats: TeamStatsInput) -> float:
    """Score recent form for one team or the form edge in a two-team matchup.

    Inputs:
        `team_stats`: either one canonical `TeamStats` snapshot or a sequence
        containing the two teams from a single fixture.

    Outputs:
        A bounded score from `0.0` to `1.0`, where higher values indicate
        stronger recent form or a clearer form separation between the teams.

    Raises:
        TypeError: If any provided item is not a `TeamStats` instance.
        ValueError: If no team stats are supplied, more than two are supplied,
            or the supplied snapshots mix sports.
    """

    normalized_team_stats = _normalize_team_stats_input(team_stats)
    intrinsic_scores = tuple(
        _calculate_intrinsic_form_score(stats) for stats in normalized_team_stats
    )
    reliability_scores = tuple(_sample_reliability(stats) for stats in normalized_team_stats)

    if len(normalized_team_stats) == 1:
        # Single-team form is slightly dampened when the available sample is
        # very small so sparse snapshots do not look deceptively decisive.
        return _clamp(intrinsic_scores[0] * (0.7 + (0.3 * reliability_scores[0])))

    dominant_score = max(intrinsic_scores)
    separation_score = abs(intrinsic_scores[0] - intrinsic_scores[1])
    reliability_score = fmean(reliability_scores)

    matchup_score = (
        (dominant_score * 0.45)
        + (separation_score * 0.40)
        + (reliability_score * 0.15)
    )
    return _clamp(matchup_score)


def _normalize_team_stats_input(team_stats: TeamStatsInput) -> tuple[TeamStats, ...]:
    """Validate and normalize the public `analyze_form` input shape.

    Args:
        team_stats: Single team snapshot or a sequence of one or two snapshots.

    Returns:
        A validated tuple of canonical `TeamStats` items.

    Raises:
        TypeError: If a sequence member is not a `TeamStats`.
        ValueError: If the input is empty, oversized, or mixes sports.
    """

    normalized: tuple[TeamStats, ...]
    if isinstance(team_stats, TeamStats):
        normalized = (team_stats,)
    else:
        normalized = tuple(team_stats)

    if not normalized:
        raise ValueError("analyze_form requires at least one TeamStats record.")
    if len(normalized) > 2:
        raise ValueError("analyze_form accepts at most two TeamStats records per fixture.")

    for stats in normalized:
        if not isinstance(stats, TeamStats):
            raise TypeError("analyze_form expects TeamStats instances only.")

    sports = {stats.sport for stats in normalized}
    if len(sports) > 1:
        raise ValueError("analyze_form requires team snapshots from the same sport.")

    return normalized


def _calculate_intrinsic_form_score(team_stats: TeamStats) -> float:
    """Derive one team's intrinsic form score from canonical stat inputs.

    Args:
        team_stats: Canonical team snapshot for one competition sample.

    Returns:
        A bounded intrinsic form score for the team alone.
    """

    matches_played = max(team_stats.matches_played, 1)
    form_score = _form_points_score(team_stats)
    win_rate_score = team_stats.wins / matches_played
    goal_balance_score = _goal_balance_score(team_stats)

    combined_score = (
        (form_score * _RECENT_FORM_WEIGHT)
        + (win_rate_score * _WIN_RATE_WEIGHT)
        + (goal_balance_score * _GOAL_BALANCE_WEIGHT)
    )
    return _clamp(combined_score)


def _form_points_score(team_stats: TeamStats) -> float:
    """Convert compact `W/D/L` form strings or season records into a score.

    Args:
        team_stats: Canonical team snapshot with optional recent form markers.

    Returns:
        A normalized points-per-match style score from `0.0` to `1.0`.
    """

    recent_results = tuple((team_stats.form or "")[-_MAX_FORM_WINDOW:])
    if recent_results:
        weighted_points = 0.0
        weighted_max = 0.0
        for index, result in enumerate(recent_results, start=1):
            points = 3 if result == "W" else 1 if result == "D" else 0
            weighted_points += points * index
            weighted_max += 3 * index
        return _clamp(weighted_points / weighted_max)

    matches_played = max(team_stats.matches_played, 1)
    season_points = (team_stats.wins * 3) + team_stats.draws
    return _clamp(season_points / (matches_played * 3))


def _goal_balance_score(team_stats: TeamStats) -> float:
    """Normalize attacking-versus-defensive balance into a bounded score.

    Args:
        team_stats: Canonical team snapshot with goals or points scored and conceded.

    Returns:
        A score where `0.5` is neutral and stronger positive balance trends
        toward `1.0`.
    """

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
    """Estimate how trustworthy a team-form snapshot is from sample size.

    Args:
        team_stats: Canonical team snapshot to assess.

    Returns:
        A bounded reliability score based on matches played and explicit form data.
    """

    sample_score = _clamp(team_stats.matches_played / _MATCH_SAMPLE_TARGET)
    if team_stats.form:
        return sample_score
    return _clamp(sample_score * 0.75)


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


__all__ = ["analyze_form"]
