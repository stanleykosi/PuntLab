"""Canonical accumulator builder for PuntLab's daily recommendation stage.

Purpose: assemble ranked match opportunities and resolved markets into
validated accumulator slips using the configured strategy rotation.
Scope: deterministic slip generation, uniqueness enforcement, odds aggregation,
and slip-level confidence calculation for up to the requested target count.
Dependencies: accumulator schemas plus the strategy helpers defined in
`src.accumulators.strategies`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from math import prod
from statistics import fmean
from typing import Final
from uuid import UUID

from src.accumulators.strategies import get_strategy, select_legs
from src.schemas.accumulators import AccumulatorLeg, AccumulatorSlip, ResolvedMarket
from src.schemas.analysis import RankedMatch

_DEFAULT_TARGET_COUNT: Final[int] = 15
_LEG_COUNT_PENALTY_PER_EXTRA_LEG: Final[float] = 0.045
_MAX_LEG_COUNT_PENALTY: Final[float] = 0.40


class AccumulatorBuilder:
    """Build the daily set of accumulator slips from ranked market candidates.

    Inputs:
        Ranked matches from the ranking stage plus resolved markets from the
        market-resolution stage.

    Outputs:
        A deterministic, confidence-sorted tuple of validated
        `AccumulatorSlip` instances ready for explanation and delivery stages.
    """

    def __init__(self, *, target_count: int = _DEFAULT_TARGET_COUNT) -> None:
        """Initialize the builder with a canonical default generation limit.

        Inputs:
            target_count: Maximum number of unique slips to attempt per daily
                build. Must be a positive integer.

        Outputs:
            A reusable accumulator builder instance.
        """

        self.target_count = self._require_positive_int(target_count, "target_count")

    def build_accumulators(
        self,
        ranked_matches: Sequence[RankedMatch],
        resolved_markets: Sequence[ResolvedMarket],
        *,
        slip_date: date,
        run_id: UUID | None = None,
        target_count: int | None = None,
    ) -> tuple[AccumulatorSlip, ...]:
        """Build up to the requested number of unique accumulator slips.

        Inputs:
            ranked_matches: Ranked match recommendations for the build window.
            resolved_markets: Resolved fixture markets eligible for leg
                construction.
            slip_date: Date the resulting slips belong to.
            run_id: Optional pipeline run identifier to stamp on every slip.
            target_count: Optional per-call override for the number of slips to
                attempt. When omitted, the builder instance default is used.

        Outputs:
            A tuple of unique accumulator slips sorted by confidence
            descending and renumbered into their final public order.

        Raises:
            ValueError: If no valid unique accumulator could be generated or if
                the supplied build parameters are invalid.
            TypeError: If schema inputs use unexpected types.
        """

        build_date = self._validate_slip_date(slip_date)
        normalized_target_count = (
            self.target_count
            if target_count is None
            else self._require_positive_int(target_count, "target_count")
        )

        provisional_slips: list[AccumulatorSlip] = []
        used_combinations: set[frozenset[str]] = set()

        for strategy_index in range(normalized_target_count):
            strategy = get_strategy(strategy_index)
            try:
                legs = select_legs(
                    ranked_matches,
                    resolved_markets,
                    exclude_combinations=used_combinations,
                    strategy=strategy,
                )
            except ValueError:
                # Later strategies can still produce viable unique slips when a
                # specific strategy and exclusion set cannot.
                continue

            fixture_combination = frozenset(leg.fixture_ref for leg in legs)
            if fixture_combination in used_combinations:
                continue

            provisional_slips.append(
                AccumulatorSlip(
                    run_id=run_id,
                    slip_date=build_date,
                    slip_number=len(provisional_slips) + 1,
                    legs=legs,
                    total_odds=round(prod(leg.odds for leg in legs), 3),
                    leg_count=len(legs),
                    confidence=self.calculate_acca_confidence(legs),
                    strategy=strategy,
                )
            )
            used_combinations.add(fixture_combination)

        if not provisional_slips:
            raise ValueError("build_accumulators could not produce any unique accumulator slips.")

        ordered_slips = sorted(
            provisional_slips,
            key=lambda slip: (
                -slip.confidence,
                slip.leg_count,
                -self._average_leg_confidence(slip.legs),
                slip.slip_number,
            ),
        )
        return tuple(
            slip.model_copy(update={"slip_number": index})
            for index, slip in enumerate(ordered_slips, start=1)
        )

    def calculate_acca_confidence(self, legs: Sequence[AccumulatorLeg]) -> float:
        """Calculate slip confidence from leg confidence and leg-count penalty.

        Inputs:
            legs: Ordered validated accumulator legs that belong to one slip.

        Outputs:
            A bounded slip-level confidence score where additional legs lower
            confidence even when the underlying legs are individually strong.

        Raises:
            ValueError: If no legs are supplied.
            TypeError: If any supplied item is not an `AccumulatorLeg`.
        """

        normalized_legs = self._normalize_legs(legs)
        average_leg_confidence = self._average_leg_confidence(normalized_legs)
        penalty = min(
            max((len(normalized_legs) - 1) * _LEG_COUNT_PENALTY_PER_EXTRA_LEG, 0.0),
            _MAX_LEG_COUNT_PENALTY,
        )
        confidence = average_leg_confidence * (1.0 - penalty)
        return max(0.0, min(1.0, round(confidence, 4)))

    def _average_leg_confidence(self, legs: Sequence[AccumulatorLeg]) -> float:
        """Return the arithmetic mean leg confidence for one slip candidate."""

        return fmean(leg.confidence for leg in self._normalize_legs(legs))

    def _normalize_legs(self, legs: Sequence[AccumulatorLeg]) -> tuple[AccumulatorLeg, ...]:
        """Validate and normalize leg collections for slip calculations."""

        normalized_legs = tuple(legs)
        if not normalized_legs:
            raise ValueError("Accumulator confidence requires at least one leg.")
        for leg in normalized_legs:
            if not isinstance(leg, AccumulatorLeg):
                raise TypeError("Accumulator confidence expects AccumulatorLeg instances only.")
        return normalized_legs

    @staticmethod
    def _validate_slip_date(value: date) -> date:
        """Require a concrete date object and reject datetime instances."""

        if isinstance(value, datetime) or not isinstance(value, date):
            raise TypeError("slip_date must be a date instance.")
        return value

    @staticmethod
    def _require_positive_int(value: int, field_name: str) -> int:
        """Validate positive integer builder configuration values."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than zero.")
        return value


__all__ = ["AccumulatorBuilder"]
