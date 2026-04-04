"""Tests for PuntLab's accumulator tier distribution logic.

Purpose: verify that generated accumulator slips are sliced consistently for
Free, Plus, and Elite users.
Scope: unit tests for `src.accumulators.distributor`.
Dependencies: pytest plus canonical accumulator and user-tier schemas.
"""

from __future__ import annotations

from datetime import date
from math import prod

import pytest
from src.accumulators.distributor import distribute_to_tiers
from src.config import MarketType, SportName
from src.schemas.accumulators import AccumulatorLeg, AccumulatorSlip, ResolutionSource
from src.schemas.users import SubscriptionTier


def build_leg(
    *,
    leg_number: int,
    fixture_ref: str,
    confidence: float,
    odds: float,
) -> AccumulatorLeg:
    """Build one canonical accumulator leg for distributor tests."""

    return AccumulatorLeg(
        leg_number=leg_number,
        fixture_ref=fixture_ref,
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {leg_number}",
        away_team=f"Away {leg_number}",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=odds,
        provider="sportybet",
        confidence=confidence,
        resolution_source=ResolutionSource.SPORTYBET_API,
        market_label="Full Time Result",
    )


def build_slip(
    *,
    slip_number: int,
    confidence: float,
    leg_count: int = 2,
) -> AccumulatorSlip:
    """Build one canonical accumulator slip for tier-slicing tests."""

    legs = tuple(
        build_leg(
            leg_number=index,
            fixture_ref=f"sr:match:{slip_number:02d}{index:02d}",
            confidence=confidence,
            odds=1.55 + (index * 0.1),
        )
        for index in range(1, leg_count + 1)
    )

    return AccumulatorSlip(
        slip_date=date(2026, 4, 4),
        slip_number=slip_number,
        legs=legs,
        total_odds=round(prod(leg.odds for leg in legs), 3),
        leg_count=leg_count,
        confidence=confidence,
    )


def test_distribute_to_tiers_sorts_and_slices_by_entitlement() -> None:
    """Distribution should rank by confidence and slice each tier correctly."""

    accumulators = tuple(
            build_slip(
                slip_number=index,
                confidence=0.55 + (index * 0.02),
                leg_count=2 + (index % 3),
            )
            for index in range(1, 13)
    )
    unsorted_accumulators = tuple(reversed(accumulators))

    distributed = distribute_to_tiers(unsorted_accumulators)

    assert set(distributed) == {
        SubscriptionTier.FREE,
        SubscriptionTier.PLUS,
        SubscriptionTier.ELITE,
    }
    assert len(distributed[SubscriptionTier.FREE]) == 1
    assert len(distributed[SubscriptionTier.PLUS]) == 10
    assert len(distributed[SubscriptionTier.ELITE]) == 12
    assert distributed[SubscriptionTier.FREE][0].confidence == max(
        slip.confidence for slip in accumulators
    )
    assert distributed[SubscriptionTier.PLUS] == distributed[SubscriptionTier.ELITE][:10]


def test_distribute_to_tiers_returns_all_available_slips_for_small_slates() -> None:
    """Small daily slates should still include all slips for Plus and Elite."""

    accumulators = (
        build_slip(slip_number=3, confidence=0.74),
        build_slip(slip_number=1, confidence=0.81),
        build_slip(slip_number=2, confidence=0.79),
    )

    distributed = distribute_to_tiers(accumulators)

    assert tuple(slip.slip_number for slip in distributed[SubscriptionTier.FREE]) == (1,)
    assert tuple(slip.slip_number for slip in distributed[SubscriptionTier.PLUS]) == (1, 2, 3)
    assert tuple(slip.slip_number for slip in distributed[SubscriptionTier.ELITE]) == (1, 2, 3)


def test_distribute_to_tiers_rejects_empty_and_invalid_inputs() -> None:
    """The distributor should fail fast on empty or malformed inputs."""

    with pytest.raises(ValueError, match="at least one accumulator slip"):
        distribute_to_tiers(())

    with pytest.raises(TypeError, match="AccumulatorSlip instances only"):
        distribute_to_tiers((object(),))  # type: ignore[arg-type]
