"""Tests for PuntLab's accumulator and market-resolution schemas.

Purpose: verify resolved-market enrichment and accumulator integrity checks
before downstream builder, explanation, and delivery stages depend on them.
Scope: unit tests for `src.schemas.accumulators`.
Dependencies: pytest plus the shared market, sport, and accumulator schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorSlip,
    ExplainedAccumulator,
    ResolutionSource,
    ResolvedMarket,
)


def test_resolved_market_serializes_resolution_metadata() -> None:
    """Resolved markets should preserve enum values and sportsbook metadata."""

    market = ResolvedMarket(
        fixture_ref="sr:match:61301159",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.82,
        provider="sportybet",
        provider_market_name="Full Time Result",
        provider_selection_name="Home",
        sportybet_available=True,
        resolution_source=ResolutionSource.SPORTYBET_API,
        sportybet_market_id=45,
        sportybet_url="https://www.sportybet.com/ng/sport/football/england/premier-league/example/sr:match:61301159",
        resolved_at=datetime(2026, 4, 3, 6, 20, tzinfo=UTC),
    )

    dumped = market.model_dump(mode="json")

    assert dumped["resolution_source"] == "sportybet_api"
    assert dumped["sport"] == "soccer"
    assert dumped["sportybet_market_id"] == 45


def test_accumulator_slip_validates_leg_count_numbering_and_total_odds() -> None:
    """Accumulator slips should reject malformed leg ordering and odds totals."""

    legs = (
        AccumulatorLeg(
            leg_number=1,
            fixture_ref="sr:match:61301159",
            sport=SportName.SOCCER,
            competition="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            market=MarketType.OVER_UNDER_25,
            selection="Over",
            odds=1.85,
            provider="sportybet",
            confidence=0.72,
            resolution_source=ResolutionSource.SPORTYBET_API,
            rationale="Both attacks are creating consistently strong chances.",
        ),
        AccumulatorLeg(
            leg_number=2,
            fixture_ref="sr:match:61301160",
            sport=SportName.BASKETBALL,
            competition="NBA",
            home_team="Lakers",
            away_team="Celtics",
            market=MarketType.TOTAL_POINTS,
            selection="Over",
            odds=1.91,
            provider="the-odds-api",
            confidence=0.68,
            resolution_source=ResolutionSource.EXTERNAL_ODDS,
            line=220.5,
        ),
    )

    slip = ExplainedAccumulator(
        slip_date=date(2026, 4, 3),
        slip_number=1,
        legs=legs,
        total_odds=round(1.85 * 1.91, 3),
        leg_count=2,
        confidence=0.7,
        rationale="High-tempo slate with two strong attacking environments.",
    )

    assert slip.legs[0].fixture_label() == "Arsenal vs Chelsea"

    with pytest.raises(ValueError, match="leg_count must match"):
        AccumulatorSlip(
            slip_date=date(2026, 4, 3),
            slip_number=1,
            legs=legs,
            total_odds=round(1.85 * 1.91, 3),
            leg_count=3,
            confidence=0.7,
        )

    with pytest.raises(ValueError, match="total_odds must match"):
        AccumulatorSlip(
            slip_date=date(2026, 4, 3),
            slip_number=1,
            legs=legs,
            total_odds=12.0,
            leg_count=2,
            confidence=0.7,
        )
