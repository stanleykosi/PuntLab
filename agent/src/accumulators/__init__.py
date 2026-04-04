"""Accumulator-building package for PuntLab.

Purpose: expose the canonical leg-selection and tiering utilities used by the
builder stage.
Scope: generation of daily accumulator slips from ranked match opportunities.
Dependencies: consumed after market resolution and scoring are in place.
"""

from src.accumulators.builder import AccumulatorBuilder
from src.accumulators.distributor import TierDistribution, distribute_to_tiers
from src.accumulators.strategies import determine_leg_count, get_strategy, select_legs

__all__ = [
    "AccumulatorBuilder",
    "TierDistribution",
    "determine_leg_count",
    "distribute_to_tiers",
    "get_strategy",
    "select_legs",
]
