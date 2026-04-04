"""Tests for PuntLab's head-to-head scoring factor.

Purpose: verify that the H2H factor rewards dense, recent matchup history and
rejects unusable historical inputs.
Scope: unit tests for `src.scoring.factors.h2h`.
Dependencies: pytest, canonical `NormalizedFixture`, and the shared sport enum.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.scoring.factors.h2h import analyze_h2h


def build_fixture(
    *,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    competition: str = "Premier League",
    sport: SportName = SportName.SOCCER,
    kickoff: datetime | None = None,
    status: FixtureStatus = FixtureStatus.FINISHED,
    source_id: str = "fixture-1",
) -> NormalizedFixture:
    """Build a canonical fixture for H2H factor tests."""

    return NormalizedFixture(
        sportradar_id=None,
        home_team=home_team,
        away_team=away_team,
        competition=competition,
        sport=sport,
        kickoff=kickoff or datetime(2026, 3, 1, 19, 45, tzinfo=UTC),
        source_provider="test-suite",
        source_id=source_id,
        country="England" if sport == SportName.SOCCER else "United States",
        home_team_id="1",
        away_team_id="2",
        status=status,
    )


def test_analyze_h2h_rewards_recent_dense_matchup_history() -> None:
    """Several recent finished meetings should produce a strong H2H score."""

    upcoming_fixture = build_fixture(
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        status=FixtureStatus.SCHEDULED,
        source_id="fixture-upcoming",
    )
    history = (
        build_fixture(kickoff=datetime(2026, 2, 15, 17, 30, tzinfo=UTC), source_id="h2h-1"),
        build_fixture(kickoff=datetime(2025, 12, 1, 20, 0, tzinfo=UTC), source_id="h2h-2"),
        build_fixture(
            home_team="Chelsea",
            away_team="Arsenal",
            kickoff=datetime(2025, 10, 12, 16, 30, tzinfo=UTC),
            source_id="h2h-3",
        ),
        build_fixture(kickoff=datetime(2025, 5, 5, 19, 45, tzinfo=UTC), source_id="h2h-4"),
        build_fixture(kickoff=datetime(2024, 12, 20, 19, 45, tzinfo=UTC), source_id="h2h-5"),
    )

    score = analyze_h2h(upcoming_fixture, history)

    assert 0.0 <= score <= 1.0
    assert score > 0.8


def test_analyze_h2h_penalizes_sparse_or_stale_history() -> None:
    """One older meeting should be meaningfully weaker than a recent dense sample."""

    upcoming_fixture = build_fixture(
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        status=FixtureStatus.SCHEDULED,
        source_id="fixture-upcoming",
    )
    sparse_history = (
        build_fixture(
            competition="FA Cup",
            kickoff=datetime(2023, 1, 15, 19, 45, tzinfo=UTC),
            source_id="old-h2h",
        ),
    )

    score = analyze_h2h(upcoming_fixture, sparse_history)

    assert 0.0 <= score <= 1.0
    assert score < 0.5


def test_analyze_h2h_rejects_unusable_history_rows() -> None:
    """The factor should fail fast when no valid finished meetings remain."""

    upcoming_fixture = build_fixture(
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        status=FixtureStatus.SCHEDULED,
        source_id="fixture-upcoming",
    )
    invalid_history = (
        build_fixture(
            home_team="Arsenal",
            away_team="Tottenham",
            kickoff=datetime(2026, 2, 1, 16, 30, tzinfo=UTC),
            source_id="wrong-pair",
        ),
        build_fixture(
            kickoff=datetime(2026, 4, 20, 19, 45, tzinfo=UTC),
            status=FixtureStatus.FINISHED,
            source_id="future-row",
        ),
    )

    with pytest.raises(ValueError, match="No finished historical meetings"):
        analyze_h2h(upcoming_fixture, invalid_history)
