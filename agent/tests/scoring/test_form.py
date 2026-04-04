"""Tests for PuntLab's recent-form scoring factor.

Purpose: verify that the form factor rewards strong recent trajectories while
failing fast on invalid scoring inputs.
Scope: unit tests for `src.scoring.factors.form`.
Dependencies: pytest, canonical `TeamStats`, and the shared sport enum.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.stats import TeamStats
from src.scoring.factors.form import analyze_form


def build_team_stats(
    *,
    team_id: str,
    team_name: str,
    sport: SportName = SportName.SOCCER,
    matches_played: int = 10,
    wins: int = 5,
    draws: int = 2,
    losses: int = 3,
    goals_for: int = 16,
    goals_against: int = 12,
    form: str | None = "WWDDL",
    avg_goals_scored: float | None = 1.6,
    avg_goals_conceded: float | None = 1.2,
) -> TeamStats:
    """Build a canonical `TeamStats` snapshot for factor tests."""

    return TeamStats(
        team_id=team_id,
        team_name=team_name,
        sport=sport,
        source_provider="test-suite",
        fetched_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        competition="Premier League" if sport == SportName.SOCCER else "NBA",
        season="2025-26",
        matches_played=matches_played,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        form=form,
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
    )


def test_analyze_form_rewards_clear_recent_form_edge() -> None:
    """A dominant-versus-struggling pairing should score well above a balanced matchup."""

    dominant = build_team_stats(
        team_id="arsenal",
        team_name="Arsenal",
        wins=8,
        draws=1,
        losses=1,
        goals_for=24,
        goals_against=8,
        form="WWWWDWWLWW",
        avg_goals_scored=2.4,
        avg_goals_conceded=0.8,
    )
    struggling = build_team_stats(
        team_id="chelsea",
        team_name="Chelsea",
        wins=2,
        draws=2,
        losses=6,
        goals_for=10,
        goals_against=19,
        form="LLWLDDLLLL",
        avg_goals_scored=1.0,
        avg_goals_conceded=1.9,
    )
    balanced_home = build_team_stats(
        team_id="villa",
        team_name="Aston Villa",
        wins=5,
        draws=3,
        losses=2,
    )
    balanced_away = build_team_stats(
        team_id="newcastle",
        team_name="Newcastle United",
        wins=5,
        draws=2,
        losses=3,
        goals_for=17,
        goals_against=13,
        form="WDWLWDWLDL",
        avg_goals_scored=1.7,
        avg_goals_conceded=1.3,
    )

    dominant_score = analyze_form((dominant, struggling))
    balanced_score = analyze_form((balanced_home, balanced_away))

    assert 0.0 <= dominant_score <= 1.0
    assert dominant_score > 0.75
    assert dominant_score > balanced_score


def test_analyze_form_uses_season_record_when_compact_form_is_missing() -> None:
    """Single-team form should still be scoreable when only season totals exist."""

    team_stats = build_team_stats(
        team_id="lakers",
        team_name="Los Angeles Lakers",
        sport=SportName.BASKETBALL,
        wins=7,
        draws=0,
        losses=3,
        goals_for=1180,
        goals_against=1102,
        form=None,
        avg_goals_scored=118.0,
        avg_goals_conceded=110.2,
    )

    score = analyze_form(team_stats)

    assert 0.6 < score < 1.0


def test_analyze_form_rejects_empty_and_mixed_sport_inputs() -> None:
    """The factor should fail fast on empty inputs and cross-sport comparisons."""

    soccer_stats = build_team_stats(team_id="arsenal", team_name="Arsenal")
    nba_stats = build_team_stats(
        team_id="celtics",
        team_name="Boston Celtics",
        sport=SportName.BASKETBALL,
        draws=0,
        form="WWLWWWWLWW",
    )

    with pytest.raises(ValueError, match="at least one TeamStats"):
        analyze_form(())

    with pytest.raises(ValueError, match="same sport"):
        analyze_form((soccer_stats, nba_stats))
