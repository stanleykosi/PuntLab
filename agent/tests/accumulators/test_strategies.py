"""Tests for PuntLab's accumulator leg-selection strategies.

Purpose: verify dynamic leg counts, strategy rotation, and deterministic leg
selection before the higher-level accumulator builder depends on them.
Scope: unit tests for `src.accumulators.strategies`.
Dependencies: pytest plus shared ranked-match and resolved-market schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.accumulators.strategies import determine_leg_count, get_strategy, select_legs
from src.config import MarketType, SportName
from src.schemas.accumulators import AccumulatorStrategy, ResolutionSource, ResolvedMarket
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown


def build_ranked_match(
    *,
    fixture_ref: str,
    rank: int,
    confidence: float,
    composite_score: float,
    sport: SportName = SportName.SOCCER,
    competition: str = "Premier League",
    home_team: str | None = None,
    away_team: str | None = None,
    recommended_market: MarketType = MarketType.MATCH_RESULT,
    recommended_selection: str = "home",
    recommended_odds: float = 1.72,
) -> RankedMatch:
    """Build a canonical ranked-match record for strategy tests."""

    home_label = home_team or f"Home {rank}"
    away_label = away_team or f"Away {rank}"
    return RankedMatch(
        fixture_ref=fixture_ref,
        sport=sport,
        competition=competition,
        home_team=home_label,
        away_team=away_label,
        composite_score=composite_score,
        confidence=confidence,
        factors=ScoreFactorBreakdown(
            form=0.75,
            h2h=0.55,
            injury_impact=0.68,
            odds_value=0.71,
            context=0.63,
            venue=0.66,
            statistical=0.61,
        ),
        recommended_market=recommended_market,
        recommended_selection=recommended_selection,
        recommended_odds=recommended_odds,
        qualitative_summary=f"{home_label} carries the stronger recent edge.",
        rank=rank,
    )


def build_resolved_market(
    *,
    fixture_ref: str,
    sport: SportName = SportName.SOCCER,
    competition: str = "Premier League",
    home_team: str,
    away_team: str,
    market: MarketType = MarketType.MATCH_RESULT,
    selection: str = "home",
    odds: float = 1.72,
    resolution_source: ResolutionSource = ResolutionSource.SPORTYBET_API,
    provider: str = "sportybet",
) -> ResolvedMarket:
    """Build a canonical resolved market for strategy tests."""

    return ResolvedMarket(
        fixture_ref=fixture_ref,
        sport=sport,
        competition=competition,
        home_team=home_team,
        away_team=away_team,
        market=market,
        selection=selection,
        odds=odds,
        provider=provider,
        provider_market_name="Full Time Result",
        provider_selection_name=selection,
        provider_market_id="full_time_result",
        resolution_source=resolution_source,
        market_label="Full Time Result",
        resolved_at=datetime(2026, 4, 4, 7, 30, tzinfo=UTC),
    )


def test_determine_leg_count_matches_strategy_windows() -> None:
    """Leg counts should follow the spec's bounded formulas for each strategy."""

    assert determine_leg_count(0.0, AccumulatorStrategy.CONFIDENT) == 2
    assert determine_leg_count(1.0, AccumulatorStrategy.CONFIDENT) == 5
    assert determine_leg_count(0.6, AccumulatorStrategy.BALANCED) == 5
    assert determine_leg_count(0.95, AccumulatorStrategy.AGGRESSIVE) == 9


def test_determine_leg_count_rejects_invalid_inputs() -> None:
    """The public leg-count helper should fail fast on invalid inputs."""

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        determine_leg_count(1.2, AccumulatorStrategy.CONFIDENT)

    with pytest.raises(TypeError, match="finite numeric value"):
        determine_leg_count(float("nan"), AccumulatorStrategy.CONFIDENT)

    with pytest.raises(ValueError, match="strategy must be one of"):
        determine_leg_count(0.5, "speculative")


def test_get_strategy_cycles_in_canonical_order() -> None:
    """Strategy assignment should rotate confidently, then balanced, then aggressive."""

    assert get_strategy(0) is AccumulatorStrategy.CONFIDENT
    assert get_strategy(1) is AccumulatorStrategy.BALANCED
    assert get_strategy(2) is AccumulatorStrategy.AGGRESSIVE
    assert get_strategy(3) is AccumulatorStrategy.CONFIDENT


def test_select_legs_prefers_recommended_markets_and_numbers_output() -> None:
    """Selected legs should keep ranking order and prefer recommended fixture markets."""

    ranked_matches = tuple(
        build_ranked_match(
            fixture_ref=f"sr:match:{index}",
            rank=index,
            confidence=0.95 - (index * 0.04),
            composite_score=0.90 - (index * 0.03),
            home_team=f"Team {index}A",
            away_team=f"Team {index}B",
        )
        for index in range(1, 6)
    )
    resolved_markets: tuple[ResolvedMarket, ...] = tuple(
        build_resolved_market(
            fixture_ref=match.fixture_ref,
            sport=match.sport,
            competition=match.competition,
            home_team=match.home_team,
            away_team=match.away_team,
            market=match.recommended_market or MarketType.MATCH_RESULT,
            selection=match.recommended_selection or "home",
            odds=1.55 + (match.rank * 0.08),
        )
        for match in ranked_matches
    ) + (
        build_resolved_market(
            fixture_ref=ranked_matches[0].fixture_ref,
            home_team=ranked_matches[0].home_team,
            away_team=ranked_matches[0].away_team,
            market=MarketType.OVER_UNDER_25,
            selection="over",
            odds=2.20,
            resolution_source=ResolutionSource.EXTERNAL_ODDS,
            provider="the-odds-api",
        ),
    )

    selected = select_legs(
        ranked_matches,
        resolved_markets,
        strategy=AccumulatorStrategy.CONFIDENT,
    )

    assert len(selected) == 4
    assert tuple(leg.leg_number for leg in selected) == (1, 2, 3, 4)
    assert selected[0].fixture_ref == ranked_matches[0].fixture_ref
    assert all(leg.market is MarketType.MATCH_RESULT for leg in selected)
    assert selected[0].selection == "home"


def test_select_legs_respects_excluded_fixture_combinations() -> None:
    """Selection should rotate to a different fixture set when the best combo is excluded."""

    ranked_matches = tuple(
        build_ranked_match(
            fixture_ref=f"sr:match:{index}",
            rank=index,
            confidence=0.93 - (index * 0.03),
            composite_score=0.88 - (index * 0.02),
            home_team=f"Club {index}A",
            away_team=f"Club {index}B",
            competition="Premier League" if index < 4 else "La Liga",
        )
        for index in range(1, 9)
    )
    resolved_markets = tuple(
        build_resolved_market(
            fixture_ref=match.fixture_ref,
            competition=match.competition,
            home_team=match.home_team,
            away_team=match.away_team,
            odds=1.60 + (match.rank * 0.07),
        )
        for match in ranked_matches
    )

    initial_selection = select_legs(
        ranked_matches,
        resolved_markets,
        strategy=AccumulatorStrategy.BALANCED,
    )
    excluded_combination = frozenset(leg.fixture_ref for leg in initial_selection)

    alternative_selection = select_legs(
        ranked_matches,
        resolved_markets,
        exclude_combinations=(excluded_combination,),
        strategy=AccumulatorStrategy.BALANCED,
    )

    assert frozenset(leg.fixture_ref for leg in alternative_selection) != excluded_combination
    assert len(alternative_selection) == len(initial_selection)


def test_select_legs_requires_matching_resolved_market_candidates() -> None:
    """Leg selection should fail fast when ranking and market inputs cannot be joined."""

    ranked_matches = (
        build_ranked_match(
            fixture_ref="sr:match:1001",
            rank=1,
            confidence=0.9,
            composite_score=0.85,
            home_team="Arsenal",
            away_team="Chelsea",
        ),
    )
    resolved_markets = (
        build_resolved_market(
            fixture_ref="sr:match:9999",
            home_team="Barcelona",
            away_team="Real Madrid",
        ),
    )

    with pytest.raises(
        ValueError,
        match="at least one ranked match with a matching resolved market",
    ):
        select_legs(ranked_matches, resolved_markets)
