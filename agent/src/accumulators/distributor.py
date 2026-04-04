"""Tier-based accumulator distribution helpers for PuntLab.

Purpose: map the generated daily accumulator slate into the canonical Free,
Plus, and Elite entitlement windows used by delivery surfaces.
Scope: validate accumulator inputs, normalize sort order, and expose one
current-state tier distribution function for downstream delivery logic.
Dependencies: accumulator slip schemas plus subscription-tier enums from
`src.schemas.users`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from src.schemas.accumulators import AccumulatorSlip
from src.schemas.users import SubscriptionTier

_FREE_LIMIT: Final[int] = 1
_PLUS_LIMIT: Final[int] = 10

type TierDistribution = dict[SubscriptionTier, tuple[AccumulatorSlip, ...]]


def distribute_to_tiers(accumulators: Sequence[AccumulatorSlip]) -> TierDistribution:
    """Distribute generated accumulator slips into subscription-tier slices.

    Inputs:
        accumulators: Generated daily accumulator slips from the builder
            stage. They may arrive unsorted, but they must already be valid
            canonical `AccumulatorSlip` instances.

    Outputs:
        A dictionary keyed by `SubscriptionTier` where:
        - `free` receives the single best slip
        - `plus` receives up to the best ten slips
        - `elite` receives the full ordered slate

    Raises:
        ValueError: If no accumulators are supplied.
        TypeError: If any supplied item is not an `AccumulatorSlip`.
    """

    normalized_accumulators = _normalize_accumulators(accumulators)
    ordered_accumulators = tuple(
        sorted(
            normalized_accumulators,
            key=lambda slip: (
                -slip.confidence,
                slip.leg_count,
                slip.slip_number,
            ),
        )
    )

    # Delivery layers can rely on every tier key always existing even when the
    # available slip count is smaller than a tier's maximum allowance.
    return {
        SubscriptionTier.FREE: ordered_accumulators[:_FREE_LIMIT],
        SubscriptionTier.PLUS: ordered_accumulators[:_PLUS_LIMIT],
        SubscriptionTier.ELITE: ordered_accumulators,
    }


def _normalize_accumulators(accumulators: Sequence[AccumulatorSlip]) -> tuple[AccumulatorSlip, ...]:
    """Validate the distributor input shape before tier slicing begins."""

    normalized_accumulators = tuple(accumulators)
    if not normalized_accumulators:
        raise ValueError("distribute_to_tiers requires at least one accumulator slip.")

    for slip in normalized_accumulators:
        if not isinstance(slip, AccumulatorSlip):
            raise TypeError("distribute_to_tiers expects AccumulatorSlip instances only.")

    return normalized_accumulators


__all__ = ["TierDistribution", "distribute_to_tiers"]
