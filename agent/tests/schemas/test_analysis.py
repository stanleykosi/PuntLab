"""Tests for PuntLab's analysis and scoring schemas.

Purpose: lock down qualitative context validation, scoring-weight checks, and
ranked match recommendation integrity before pipeline nodes depend on them.
Scope: unit tests for `src.schemas.analysis`.
Dependencies: pytest plus the shared analysis, market, and sport schemas.
"""

from __future__ import annotations

import pytest
from src.config import MarketType, SportName
from src.schemas.analysis import (
    MatchContext,
    MatchScore,
    RankedMatch,
    ScoreFactorBreakdown,
    ScoringWeights,
)


def test_match_context_deduplicates_sources_and_serializes_fixture_ref() -> None:
    """Match context should preserve bounded signals and unique source labels."""

    context = MatchContext(
        fixture_ref="sr:match:61301159",
        morale_home=0.8,
        morale_away=0.52,
        rivalry_factor=0.75,
        pressure_home=0.61,
        pressure_away=0.67,
        key_narrative="Title pressure is high and both teams arrive with strong momentum.",
        qualitative_score=0.72,
        data_sources=("BBC Sport", "bbc sport", "Tavily"),
    )

    dumped = context.model_dump(mode="json")

    assert context.data_sources == ("BBC Sport", "Tavily")
    assert dumped["fixture_ref"] == "sr:match:61301159"


def test_scoring_weights_must_sum_to_one() -> None:
    """Scoring weights should fail fast when the configured total drifts."""

    with pytest.raises(ValueError, match="sum to 1.0"):
        ScoringWeights(statistical=0.2)


def test_ranked_match_requires_coherent_recommendation_fields() -> None:
    """Ranked matches should reject incomplete recommendation metadata."""

    factors = ScoreFactorBreakdown(
        form=0.8,
        h2h=0.55,
        injury_impact=0.7,
        odds_value=0.68,
        context=0.74,
        venue=0.6,
        statistical=0.51,
    )

    ranked_match = RankedMatch(
        fixture_ref="sr:match:61301159",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        composite_score=0.73,
        confidence=0.69,
        factors=factors,
        recommended_market=MarketType.OVER_UNDER_25,
        recommended_selection="Over",
        recommended_odds=1.85,
        qualitative_summary="Both sides are creating high-quality chances consistently.",
        rank=1,
    )

    dumped = ranked_match.model_dump(mode="json")

    assert dumped["recommended_market"] == "over_under_2.5"
    assert dumped["rank"] == 1

    with pytest.raises(ValueError, match="recommended_selection is required"):
        MatchScore(
            fixture_ref="sr:match:61301159",
            sport=SportName.SOCCER,
            competition="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            composite_score=0.73,
            confidence=0.69,
            factors=factors,
            recommended_market=MarketType.BTTS,
        )
