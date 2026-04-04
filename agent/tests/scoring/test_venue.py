"""Tests for PuntLab's venue scoring factor.

Purpose: verify that the venue factor rewards strong home edges and penalizes
resilient away travelers while validating fixture-team matching.
Scope: unit tests for `src.scoring.factors.venue`.
Dependencies: pytest plus the canonical fixture and team-stat schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.stats import TeamStats
from src.scoring.factors.venue import analyze_venue


def build_fixture() -> NormalizedFixture:
    """Build a canonical Arsenal-Chelsea fixture for venue tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        source_provider="test-suite",
        source_id="fixture-7001",
        country="England",
        home_team_id="1",
        away_team_id="2",
        status=FixtureStatus.SCHEDULED,
    )


def build_team_stats(
    *,
    team_id: str,
    team_name: str,
    wins: int,
    draws: int,
    losses: int,
    home_wins: int,
    away_wins: int,
    avg_goals_scored: float,
    avg_goals_conceded: float,
) -> TeamStats:
    """Build a canonical team snapshot for venue-factor tests."""

    return TeamStats(
        team_id=team_id,
        team_name=team_name,
        sport=SportName.SOCCER,
        source_provider="test-suite",
        fetched_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        competition="Premier League",
        season="2025-26",
        matches_played=10,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=int(round(avg_goals_scored * 10)),
        goals_against=int(round(avg_goals_conceded * 10)),
        home_wins=home_wins,
        away_wins=away_wins,
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
        form="WWDWLDWWDL",
    )


def test_analyze_venue_rewards_home_strength_against_weak_away_travel() -> None:
    """A strong home side facing a weak traveler should score well above parity."""

    fixture = build_fixture()
    home_stats = build_team_stats(
        team_id="1",
        team_name="Arsenal",
        wins=7,
        draws=2,
        losses=1,
        home_wins=5,
        away_wins=2,
        avg_goals_scored=2.0,
        avg_goals_conceded=0.8,
    )
    away_stats = build_team_stats(
        team_id="2",
        team_name="Chelsea",
        wins=3,
        draws=3,
        losses=4,
        home_wins=3,
        away_wins=0,
        avg_goals_scored=1.1,
        avg_goals_conceded=1.5,
    )
    resilient_away_stats = build_team_stats(
        team_id="2",
        team_name="Chelsea",
        wins=7,
        draws=1,
        losses=2,
        home_wins=3,
        away_wins=4,
        avg_goals_scored=1.8,
        avg_goals_conceded=0.9,
    )

    strong_home_score = analyze_venue(fixture, (home_stats, away_stats))
    resilient_away_score = analyze_venue(fixture, (home_stats, resilient_away_stats))

    assert strong_home_score > 0.7
    assert strong_home_score > resilient_away_score


def test_analyze_venue_selects_the_best_matching_snapshot_per_team() -> None:
    """Multiple provider snapshots for one side should resolve to the richest record."""

    fixture = build_fixture()
    sparse_home_stats = build_team_stats(
        team_id="1",
        team_name="Arsenal",
        wins=2,
        draws=1,
        losses=1,
        home_wins=1,
        away_wins=1,
        avg_goals_scored=1.4,
        avg_goals_conceded=1.0,
    ).model_copy(update={"matches_played": 4, "form": None})
    rich_home_stats = build_team_stats(
        team_id="1",
        team_name="Arsenal",
        wins=7,
        draws=2,
        losses=1,
        home_wins=5,
        away_wins=2,
        avg_goals_scored=2.0,
        avg_goals_conceded=0.8,
    )
    away_stats = build_team_stats(
        team_id="2",
        team_name="Chelsea",
        wins=3,
        draws=3,
        losses=4,
        home_wins=3,
        away_wins=0,
        avg_goals_scored=1.1,
        avg_goals_conceded=1.5,
    )

    score = analyze_venue(fixture, (sparse_home_stats, rich_home_stats, away_stats))

    assert score > 0.7


def test_analyze_venue_rejects_non_matching_fixture_team_stats() -> None:
    """Venue analysis should fail fast when no stats belong to the fixture teams."""

    fixture = build_fixture()
    unrelated_stats = build_team_stats(
        team_id="99",
        team_name="Liverpool",
        wins=8,
        draws=1,
        losses=1,
        home_wins=5,
        away_wins=3,
        avg_goals_scored=2.3,
        avg_goals_conceded=0.7,
    )

    with pytest.raises(ValueError, match="could not match any TeamStats"):
        analyze_venue(fixture, (unrelated_stats,))
