"""Tests for PuntLab's normalized odds schema.

Purpose: verify broad market preservation, canonical line handling, and JSON
serialization for normalized odds rows.
Scope: unit tests for `src.schemas.odds.NormalizedOdds`.
Dependencies: pytest and the shared market taxonomy from `src.config`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import MarketType
from src.schemas.odds import NormalizedOdds


def test_normalized_odds_serializes_market_enum_and_provider_metadata() -> None:
    """Odds rows should preserve canonical mappings and provider metadata."""

    odds = NormalizedOdds(
        fixture_ref="sr:match:61301159",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.82,
        provider="sportybet",
        provider_market_name="Full Time Result",
        provider_selection_name="Home",
        sportybet_available=True,
        market_label="Full Time Result",
        provider_market_id="1x2",
        last_updated=datetime(2026, 4, 3, 6, 30, tzinfo=UTC),
    )

    dumped = odds.model_dump(mode="json")

    assert dumped["market"] == "1x2"
    assert dumped["provider_market_id"] == "1x2"
    assert dumped["provider_market_key"] == "full_time_result"
    assert dumped["provider_selection_key"] == "home"
    assert dumped["sportybet_available"] is True


def test_normalized_odds_requires_line_for_line_based_markets() -> None:
    """Spread and totals markets should fail fast when line data is missing."""

    with pytest.raises(ValueError, match="line is required"):
        NormalizedOdds(
            fixture_ref="sr:match:61301159",
            market=MarketType.POINT_SPREAD,
            selection="home",
            odds=1.91,
            provider="sportybet",
            provider_market_name="Point Spread",
            provider_selection_name="Home",
        )


def test_normalized_odds_rejects_non_finite_values() -> None:
    """Odds validation should reject invalid floating-point values."""

    with pytest.raises(ValueError, match="finite number"):
        NormalizedOdds(
            fixture_ref="sr:match:61301159",
            market=MarketType.MATCH_RESULT,
            selection="home",
            odds=float("inf"),
            provider="sportybet",
            provider_market_name="Match Winner",
            provider_selection_name="Home",
        )


def test_normalized_odds_preserves_unmapped_provider_markets() -> None:
    """Unmapped provider markets should remain ingestible with `market=None`."""

    odds = NormalizedOdds(
        fixture_ref="api-football:501",
        market=None,
        selection="Home",
        odds=2.4,
        provider="Bet365",
        provider_market_name="Team To Score First",
        provider_selection_name="Home",
        participant_scope="team",
        raw_metadata={"canonical_market_supported": False},
    )

    dumped = odds.model_dump(mode="json")

    assert dumped["market"] is None
    assert dumped["market_label"] == "Team To Score First"
    assert dumped["provider_market_key"] == "team_to_score_first"
    assert dumped["provider_selection_key"] == "home"
