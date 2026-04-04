"""Tests for PuntLab's accumulator builder.

Purpose: verify slip construction, uniqueness enforcement, confidence ordering,
and fail-fast validation for `src.accumulators.builder`.
Scope: unit tests for builder behavior on ranked matches and resolved markets.
Dependencies: pytest plus canonical accumulator, ranking, and market schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from src.accumulators.builder import AccumulatorBuilder
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
    """Build one canonical ranked match for builder tests."""

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
            form=0.74,
            h2h=0.58,
            injury_impact=0.67,
            odds_value=0.72,
            context=0.61,
            venue=0.69,
            statistical=0.63,
        ),
        recommended_market=recommended_market,
        recommended_selection=recommended_selection,
        recommended_odds=recommended_odds,
        qualitative_summary=f"{home_label} carries the stronger setup.",
        rank=rank,
    )


def build_resolved_market(
    *,
    fixture_ref: str,
    home_team: str,
    away_team: str,
    sport: SportName = SportName.SOCCER,
    competition: str = "Premier League",
    market: MarketType = MarketType.MATCH_RESULT,
    selection: str = "home",
    odds: float = 1.72,
    resolution_source: ResolutionSource = ResolutionSource.SPORTYBET_API,
    provider: str = "sportybet",
) -> ResolvedMarket:
    """Build one canonical resolved market for builder tests."""

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
        resolved_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
    )


def build_candidate_sets(
    count: int = 8,
) -> tuple[tuple[RankedMatch, ...], tuple[ResolvedMarket, ...]]:
    """Build a fixture slate large enough to produce multiple unique slips."""

    ranked_matches = tuple(
        build_ranked_match(
            fixture_ref=f"sr:match:{1000 + index}",
            rank=index,
            confidence=0.95 - (index * 0.035),
            composite_score=0.90 - (index * 0.025),
            competition="Premier League" if index < 4 else "La Liga",
            home_team=f"Club {index}A",
            away_team=f"Club {index}B",
        )
        for index in range(1, count + 1)
    )
    resolved_markets = tuple(
        build_resolved_market(
            fixture_ref=match.fixture_ref,
            sport=match.sport,
            competition=match.competition,
            home_team=match.home_team,
            away_team=match.away_team,
            market=match.recommended_market or MarketType.MATCH_RESULT,
            selection=match.recommended_selection or "home",
            odds=1.55 + (match.rank * 0.09),
        )
        for match in ranked_matches
    )
    return ranked_matches, resolved_markets


def test_build_accumulators_creates_unique_sorted_slips() -> None:
    """Builder output should be unique, sorted by confidence, and renumbered."""

    ranked_matches, resolved_markets = build_candidate_sets()
    builder = AccumulatorBuilder(target_count=4)
    run_id = uuid4()

    slips = builder.build_accumulators(
        ranked_matches,
        resolved_markets,
        slip_date=date(2026, 4, 4),
        run_id=run_id,
    )

    assert len(slips) == 4
    assert [slip.slip_number for slip in slips] == [1, 2, 3, 4]
    assert all(slip.run_id == run_id for slip in slips)
    assert all(slip.slip_date == date(2026, 4, 4) for slip in slips)
    assert all(slip.strategy in set(AccumulatorStrategy) for slip in slips)
    assert [slip.confidence for slip in slips] == sorted(
        (slip.confidence for slip in slips),
        reverse=True,
    )
    assert len({frozenset(leg.fixture_ref for leg in slip.legs) for slip in slips}) == len(slips)


def test_build_accumulators_returns_fewer_slips_when_unique_combos_run_out() -> None:
    """The builder should stop at the unique-slip ceiling instead of duplicating combos."""

    ranked_matches, resolved_markets = build_candidate_sets(count=5)
    builder = AccumulatorBuilder(target_count=6)

    slips = builder.build_accumulators(
        ranked_matches,
        resolved_markets,
        slip_date=date(2026, 4, 4),
    )

    assert 1 <= len(slips) < 6
    assert len({frozenset(leg.fixture_ref for leg in slip.legs) for slip in slips}) == len(slips)


def test_calculate_acca_confidence_penalizes_longer_slips() -> None:
    """Longer slips should receive lower confidence when leg confidence is otherwise equal."""

    ranked_matches, resolved_markets = build_candidate_sets(count=8)
    builder = AccumulatorBuilder()

    shorter_slip = builder.build_accumulators(
        ranked_matches,
        resolved_markets,
        slip_date=date(2026, 4, 4),
        target_count=1,
    )[0]
    longer_slip = builder.build_accumulators(
        ranked_matches,
        resolved_markets,
        slip_date=date(2026, 4, 4),
        target_count=3,
    )[-1]

    assert shorter_slip.leg_count <= longer_slip.leg_count
    assert shorter_slip.confidence >= longer_slip.confidence


def test_build_accumulators_rejects_empty_candidate_pool() -> None:
    """The builder should fail fast when no unique accumulator can be created."""

    builder = AccumulatorBuilder()

    with pytest.raises(ValueError, match="could not produce any unique accumulator slips"):
        builder.build_accumulators((), (), slip_date=date(2026, 4, 4))


def test_build_accumulators_rejects_datetime_slip_date() -> None:
    """Slip dates should stay as pure dates rather than timezone-bearing datetimes."""

    ranked_matches, resolved_markets = build_candidate_sets(count=4)
    builder = AccumulatorBuilder(target_count=1)

    with pytest.raises(TypeError, match="slip_date must be a date instance"):
        builder.build_accumulators(
            ranked_matches,
            resolved_markets,
            slip_date=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        )
