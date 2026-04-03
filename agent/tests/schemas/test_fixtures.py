"""Tests for PuntLab's normalized fixture schema.

Purpose: lock down fixture validation, fallback identity behavior, and the
derived SportyBet lookup metadata needed by later scraping steps.
Scope: unit tests for `src.schemas.fixtures.NormalizedFixture`.
Dependencies: pytest plus the shared sport enum and fixture schema.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.fixtures import FixtureStatus, NormalizedFixture


def test_normalized_fixture_derives_fixture_reference_and_league_slug() -> None:
    """Fixtures should expose a fallback ref and a canonical league slug."""

    fixture = NormalizedFixture(
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 3, 19, 45, tzinfo=UTC),
        source_provider="api-football",
        source_id="fixture-1001",
        country="England",
    )

    dumped = fixture.model_dump(mode="json")

    assert fixture.get_fixture_ref() == "api-football:fixture-1001"
    assert fixture.league == "premier-league"
    assert fixture.get_sportybet_country_slug() == "england"
    assert fixture.get_sportybet_league_slug() == "premier-league"
    assert dumped["sport"] == "soccer"
    assert dumped["status"] == "scheduled"


def test_normalized_fixture_prefers_sportradar_reference() -> None:
    """Sportradar IDs should become the canonical fixture reference when present."""

    fixture = NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Lakers",
        away_team="Celtics",
        competition="NBA",
        sport=SportName.BASKETBALL,
        kickoff=datetime(2026, 4, 3, 1, 0, tzinfo=UTC),
        source_provider="balldontlie",
        source_id="game-5001",
        league="nba",
        status=FixtureStatus.SCHEDULED,
    )

    assert fixture.get_fixture_ref() == "sr:match:61301159"


def test_normalized_fixture_rejects_identical_teams_and_naive_kickoff() -> None:
    """Fixtures should fail fast on ambiguous timing and invalid participants."""

    with pytest.raises(ValueError, match="timezone information"):
        NormalizedFixture(
            home_team="Arsenal",
            away_team="Chelsea",
            competition="Premier League",
            sport=SportName.SOCCER,
            kickoff=datetime(2026, 4, 3, 19, 45),
            source_provider="api-football",
            source_id="fixture-1001",
        )

    with pytest.raises(ValueError, match="different teams"):
        NormalizedFixture(
            home_team="Arsenal",
            away_team="arsenal",
            competition="Premier League",
            sport=SportName.SOCCER,
            kickoff=datetime(2026, 4, 3, 19, 45, tzinfo=UTC),
            source_provider="api-football",
            source_id="fixture-1001",
        )
