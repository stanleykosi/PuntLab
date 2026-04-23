"""Accumulator leg-selection strategies for PuntLab's builder stage.

Purpose: provide the canonical strategy rotation, dynamic leg-count rules, and
deterministic leg selection used by the accumulator builder.
Scope: convert ranked matches plus resolved markets into validated accumulator
legs while enforcing fail-fast validation and basic within-slip diversity.
Dependencies: ranked-match and resolved-market schemas from `src.schemas`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Final

from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorStrategy,
    ResolutionSource,
    ResolvedMarket,
)
from src.schemas.analysis import RankedMatch

_STRATEGY_ROTATION: Final[tuple[AccumulatorStrategy, ...]] = (
    AccumulatorStrategy.CONFIDENT,
    AccumulatorStrategy.BALANCED,
    AccumulatorStrategy.AGGRESSIVE,
)
_STRATEGY_WINDOWS: Final[dict[AccumulatorStrategy, tuple[int, int, int]]] = {
    AccumulatorStrategy.CONFIDENT: (2, 3, 5),
    AccumulatorStrategy.BALANCED: (3, 4, 7),
    AccumulatorStrategy.AGGRESSIVE: (5, 5, 10),
}
_UTILITY_WEIGHTS: Final[
    dict[AccumulatorStrategy, tuple[float, float, float, float]]
] = {
    AccumulatorStrategy.CONFIDENT: (0.62, 0.20, 0.14, 0.04),
    AccumulatorStrategy.BALANCED: (0.48, 0.20, 0.12, 0.20),
    AccumulatorStrategy.AGGRESSIVE: (0.34, 0.16, 0.08, 0.42),
}
_RESOLUTION_PRIORITY: Final[dict[ResolutionSource, int]] = {
    ResolutionSource.SPORTYBET_API: 2,
    ResolutionSource.SPORTYBET_BROWSER: 1,
}
_MAX_SELECTION_ATTEMPTS: Final[int] = 9

type _ExcludedCombination = frozenset[str]


@dataclass(frozen=True, slots=True)
class _LegCandidate:
    """Internal candidate record pairing ranking data with a resolved market."""

    ranked_match: RankedMatch
    resolved_market: ResolvedMarket
    utility_score: float

    @property
    def fixture_ref(self) -> str:
        """Expose the fixture reference for deterministic selection handling."""

        return self.ranked_match.fixture_ref


def determine_leg_count(
    confidence_pool: float,
    strategy: AccumulatorStrategy | str,
) -> int:
    """Return the canonical dynamic leg count for a strategy and confidence pool.

    Inputs:
        confidence_pool: Normalized confidence quality for the day's available
            leg pool. Must already be bounded between `0.0` and `1.0`.
        strategy: Target accumulator strategy name or enum value.

    Outputs:
        The integer leg count defined by the technical specification.

    Raises:
        ValueError: If the confidence pool is out of range or the strategy is
            unknown.
        TypeError: If the confidence pool is not a finite real number.
    """

    if not isinstance(confidence_pool, int | float):
        raise TypeError("confidence_pool must be a finite numeric value.")
    normalized_confidence = float(confidence_pool)
    if not isfinite(normalized_confidence):
        raise TypeError("confidence_pool must be a finite numeric value.")
    if normalized_confidence < 0.0 or normalized_confidence > 1.0:
        raise ValueError("confidence_pool must be between 0.0 and 1.0 inclusive.")

    normalized_strategy = _coerce_strategy(strategy)
    base, multiplier, maximum = _STRATEGY_WINDOWS[normalized_strategy]
    return min(base + int(normalized_confidence * multiplier), maximum)


def get_strategy(index: int) -> AccumulatorStrategy:
    """Return the canonical strategy rotation entry for a slip index.

    Inputs:
        index: Zero-based slip index during one builder run.

    Outputs:
        The strategy enum to use for that slip position.

    Raises:
        TypeError: If `index` is not an integer.
        ValueError: If `index` is negative.
    """

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise ValueError("index must be zero or greater.")
    return _STRATEGY_ROTATION[index % len(_STRATEGY_ROTATION)]


def select_legs(
    ranked_matches: Sequence[RankedMatch],
    resolved_markets: Sequence[ResolvedMarket],
    *,
    exclude_combinations: Collection[Collection[str]] | None = None,
    strategy: AccumulatorStrategy | str = AccumulatorStrategy.CONFIDENT,
) -> tuple[AccumulatorLeg, ...]:
    """Select accumulator legs for one slip using the requested strategy.

    Inputs:
        ranked_matches: Globally ranked match recommendations for the day.
        resolved_markets: Fixture-scoped resolved market rows available for leg
            construction.
        exclude_combinations: Prior fixture combinations already used by other
            generated slips and therefore ineligible for reuse.
        strategy: Strategy profile controlling leg-count and candidate utility.

    Outputs:
        An ordered tuple of validated `AccumulatorLeg` records ready for the
        builder to place into an accumulator slip.

    Raises:
        ValueError: If no usable candidates exist or no unique combination can
            be selected under the requested constraints.
        TypeError: If any supplied ranked match or resolved market has the
            wrong schema type.
    """

    normalized_strategy = _coerce_strategy(strategy)
    excluded = _normalize_excluded_combinations(exclude_combinations)
    candidates = _build_candidates(
        ranked_matches=ranked_matches,
        resolved_markets=resolved_markets,
        strategy=normalized_strategy,
    )
    if not candidates:
        raise ValueError(
            "select_legs requires at least one ranked match with a matching resolved market."
        )

    confidence_pool = _derive_confidence_pool(candidates)
    target_leg_count = min(
        determine_leg_count(confidence_pool, normalized_strategy),
        len(candidates),
    )
    if target_leg_count <= 0:
        raise ValueError("select_legs could not derive a valid target leg count.")

    selected_candidates = _select_candidate_combination(
        candidates=candidates,
        target_leg_count=target_leg_count,
        excluded=excluded,
    )
    if len(selected_candidates) != target_leg_count:
        raise ValueError("select_legs could not select the requested number of unique legs.")

    ordered_candidates = tuple(
        sorted(
            selected_candidates,
            key=lambda candidate: (
                candidate.ranked_match.rank,
                -candidate.utility_score,
                candidate.fixture_ref,
            ),
        )
    )
    return tuple(
        _build_accumulator_leg(candidate, leg_number=index)
        for index, candidate in enumerate(ordered_candidates, start=1)
    )


def _coerce_strategy(strategy: AccumulatorStrategy | str) -> AccumulatorStrategy:
    """Normalize strategy inputs into the canonical enum form."""

    try:
        if isinstance(strategy, AccumulatorStrategy):
            return strategy
        return AccumulatorStrategy(strategy)
    except ValueError as error:
        raise ValueError(
            "strategy must be one of: confident, balanced, aggressive."
        ) from error


def _normalize_excluded_combinations(
    exclude_combinations: Collection[Collection[str]] | None,
) -> frozenset[_ExcludedCombination]:
    """Validate fixture-combination exclusions for deterministic reuse checks."""

    if exclude_combinations is None:
        return frozenset()

    normalized_exclusions: set[_ExcludedCombination] = set()
    for combination in exclude_combinations:
        normalized_fixture_refs = frozenset(
            fixture_ref.strip()
            for fixture_ref in combination
            if isinstance(fixture_ref, str) and fixture_ref.strip()
        )
        if not normalized_fixture_refs:
            continue
        normalized_exclusions.add(normalized_fixture_refs)
    return frozenset(normalized_exclusions)


def _build_candidates(
    *,
    ranked_matches: Sequence[RankedMatch],
    resolved_markets: Sequence[ResolvedMarket],
    strategy: AccumulatorStrategy,
) -> tuple[_LegCandidate, ...]:
    """Project ranked matches plus resolved markets into unique leg candidates."""

    resolved_by_fixture: dict[str, list[ResolvedMarket]] = {}
    for market in resolved_markets:
        if not isinstance(market, ResolvedMarket):
            raise TypeError("select_legs expects ResolvedMarket instances only.")
        resolved_by_fixture.setdefault(market.fixture_ref, []).append(market)

    candidates: list[_LegCandidate] = []
    seen_fixture_refs: set[str] = set()

    for ranked_match in ranked_matches:
        if not isinstance(ranked_match, RankedMatch):
            raise TypeError("select_legs expects RankedMatch instances only.")
        if ranked_match.fixture_ref in seen_fixture_refs:
            continue

        markets = resolved_by_fixture.get(ranked_match.fixture_ref)
        if not markets:
            continue

        selected_market = _select_market_for_ranked_match(ranked_match, tuple(markets))
        utility_score = _score_candidate(ranked_match, selected_market, strategy)
        candidates.append(
            _LegCandidate(
                ranked_match=ranked_match,
                resolved_market=selected_market,
                utility_score=utility_score,
            )
        )
        seen_fixture_refs.add(ranked_match.fixture_ref)

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.utility_score,
                -candidate.ranked_match.confidence,
                candidate.ranked_match.rank,
                candidate.fixture_ref,
            ),
        )
    )


def _select_market_for_ranked_match(
    ranked_match: RankedMatch,
    markets: Sequence[ResolvedMarket],
) -> ResolvedMarket:
    """Choose the resolved market that best matches the ranked recommendation."""

    def ranking_key(market: ResolvedMarket) -> tuple[int, int, float]:
        market_match = int(market.market == ranked_match.recommended_market)
        selection_match = int(market.selection == ranked_match.recommended_selection)
        resolution_priority = _RESOLUTION_PRIORITY[market.resolution_source]
        return (
            (market_match * 2) + selection_match,
            resolution_priority,
            market.odds,
        )

    return max(markets, key=ranking_key)


def _score_candidate(
    ranked_match: RankedMatch,
    resolved_market: ResolvedMarket,
    strategy: AccumulatorStrategy,
) -> float:
    """Calculate strategy-specific candidate utility for greedy leg selection."""

    confidence_weight, composite_weight, safety_weight, upside_weight = _UTILITY_WEIGHTS[
        strategy
    ]
    safety_score = _calculate_safety_score(resolved_market.odds)
    upside_score = _calculate_upside_score(resolved_market.odds)
    utility = (
        (ranked_match.confidence * confidence_weight)
        + (ranked_match.composite_score * composite_weight)
        + (safety_score * safety_weight)
        + (upside_score * upside_weight)
    )
    return min(max(utility, 0.0), 1.0)


def _calculate_safety_score(odds: float) -> float:
    """Reward markets that sit in PuntLab's safer accumulator-odds band."""

    distance_from_target = abs(odds - 1.75)
    return max(0.0, 1.0 - (distance_from_target / 1.75))


def _calculate_upside_score(odds: float) -> float:
    """Reward higher decimal odds without allowing extreme longshots to dominate."""

    capped_odds = min(max(odds, 1.20), 4.20)
    return (capped_odds - 1.20) / 3.0


def _derive_confidence_pool(candidates: Sequence[_LegCandidate]) -> float:
    """Estimate the day's usable confidence pool from the top candidate window."""

    candidate_window = tuple(
        candidate.ranked_match.confidence for candidate in candidates[: min(len(candidates), 8)]
    )
    return fmean(candidate_window)


def _select_candidate_combination(
    *,
    candidates: Sequence[_LegCandidate],
    target_leg_count: int,
    excluded: Collection[_ExcludedCombination],
) -> tuple[_LegCandidate, ...]:
    """Select a unique candidate combination using deterministic greedy passes."""

    candidate_orderings = (
        tuple(candidates),
        tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.ranked_match.confidence,
                    -candidate.utility_score,
                    candidate.ranked_match.rank,
                    candidate.fixture_ref,
                ),
            )
        ),
        tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.resolved_market.odds,
                    -candidate.utility_score,
                    candidate.ranked_match.rank,
                    candidate.fixture_ref,
                ),
            )
        ),
    )

    for ordering in candidate_orderings:
        attempt_limit = min(len(ordering), _MAX_SELECTION_ATTEMPTS)
        for skipped_candidates in range(attempt_limit):
            pruned_ordering = ordering[skipped_candidates:]
            if len(pruned_ordering) < target_leg_count:
                continue
            selected = _greedy_select(pruned_ordering, target_leg_count)
            if len(selected) != target_leg_count:
                continue
            fixture_refs = frozenset(candidate.fixture_ref for candidate in selected)
            if fixture_refs in excluded:
                continue
            return selected

    raise ValueError("select_legs could not produce a unique leg combination.")


def _greedy_select(
    candidates: Sequence[_LegCandidate],
    target_leg_count: int,
) -> tuple[_LegCandidate, ...]:
    """Greedily select the strongest diverse subset from an ordered candidate list."""

    selected: list[_LegCandidate] = []
    used_fixture_refs: set[str] = set()
    competition_counts: Counter[str] = Counter()
    sport_counts: Counter[str] = Counter()

    remaining = list(candidates)
    while remaining and len(selected) < target_leg_count:
        next_candidate = max(
            remaining,
            key=lambda candidate: _greedy_priority(
                candidate,
                competition_counts=competition_counts,
                sport_counts=sport_counts,
            ),
        )
        remaining.remove(next_candidate)
        if next_candidate.fixture_ref in used_fixture_refs:
            continue

        selected.append(next_candidate)
        used_fixture_refs.add(next_candidate.fixture_ref)
        competition_counts[next_candidate.ranked_match.competition] += 1
        sport_counts[next_candidate.ranked_match.sport.value] += 1

    return tuple(selected)


def _greedy_priority(
    candidate: _LegCandidate,
    *,
    competition_counts: Counter[str],
    sport_counts: Counter[str],
) -> float:
    """Apply soft diversity penalties while preserving strategy utility ordering."""

    competition_penalty = 0.08 * competition_counts[candidate.ranked_match.competition]
    sport_penalty = 0.04 * sport_counts[candidate.ranked_match.sport.value]
    return candidate.utility_score - competition_penalty - sport_penalty


def _build_accumulator_leg(candidate: _LegCandidate, *, leg_number: int) -> AccumulatorLeg:
    """Translate one selected candidate into the public accumulator-leg schema."""

    ranked_match = candidate.ranked_match
    resolved_market = candidate.resolved_market

    return AccumulatorLeg(
        leg_number=leg_number,
        fixture_ref=ranked_match.fixture_ref,
        sport=ranked_match.sport,
        competition=ranked_match.competition,
        home_team=ranked_match.home_team,
        away_team=ranked_match.away_team,
        market=resolved_market.market,
        canonical_market=resolved_market.canonical_market,
        selection=resolved_market.selection,
        odds=resolved_market.odds,
        provider=resolved_market.provider,
        confidence=ranked_match.confidence,
        resolution_source=resolved_market.resolution_source,
        market_label=resolved_market.market_label,
        line=resolved_market.line,
        rationale=ranked_match.qualitative_summary,
        sportybet_url=resolved_market.sportybet_url,
    )


__all__ = ["determine_leg_count", "get_strategy", "select_legs"]
