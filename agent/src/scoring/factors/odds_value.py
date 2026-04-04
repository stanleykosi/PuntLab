"""Odds-value scoring factor for PuntLab's deterministic scoring engine.

Purpose: compare available bookmaker prices against a consensus market
probability estimate and translate the best detected edge into a bounded score.
Scope: canonicalize scoreable odds rows, normalize provider overrounds within
each market, and reward prices that are materially better than the consensus.
Dependencies: relies on the canonical `NormalizedOdds` schema and the odds
mapping helpers under `src.providers.odds_mapping`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from src.config import MarketType
from src.providers.odds_mapping import filter_scoreable_odds
from src.schemas.odds import NormalizedOdds

type MarketKey = tuple[MarketType, float | None, str | None, str | None]
type ProviderBookKey = tuple[MarketType, float | None, str | None, str | None, str]
type SelectionKey = tuple[MarketType, float | None, str | None, str | None, str]


@dataclass(frozen=True)
class _ProviderMarketBook:
    """Normalized provider probabilities for one fixture market.

    Inputs:
        Scoreable odds rows for one provider's view of a single canonical
        market and line.

    Outputs:
        Overround-adjusted per-selection probabilities plus the offered odds
        used for consensus comparison.
    """

    market_key: MarketKey
    provider: str
    normalized_probabilities: dict[str, float]
    offered_odds: dict[str, float]


def analyze_odds_value(odds: Sequence[NormalizedOdds]) -> float:
    """Score the best value edge currently visible in the odds market.

    Inputs:
        `odds`: fixture-scoped odds rows across one or more providers.

    Outputs:
        A bounded score from `0.1` to `0.9`, following the threshold mapping
        in the technical specification for strong, moderate, slight, marginal,
        and absent value edges.

    Raises:
        TypeError: If any item is not a `NormalizedOdds` record.
        ValueError: If no odds are supplied or the rows span multiple fixtures.
    """

    normalized_rows = _normalize_odds_rows(odds)
    scoreable_rows = filter_scoreable_odds(normalized_rows)
    if not scoreable_rows:
        return 0.1

    provider_books = _build_provider_books(scoreable_rows)
    if not provider_books:
        return 0.1

    selection_edges = _calculate_adjusted_edges(provider_books)
    if not selection_edges:
        return 0.1

    return _score_edge(max(selection_edges))


def _normalize_odds_rows(odds: Sequence[NormalizedOdds]) -> tuple[NormalizedOdds, ...]:
    """Validate the public odds-value input shape.

    Args:
        odds: Raw odds rows passed to the public factor.

    Returns:
        A validated tuple of canonical odds rows.

    Raises:
        TypeError: If any input item is not a `NormalizedOdds`.
        ValueError: If the input is empty or spans multiple fixtures.
    """

    normalized_rows = tuple(odds)
    if not normalized_rows:
        raise ValueError("analyze_odds_value requires at least one NormalizedOdds record.")

    fixture_refs = {row.fixture_ref for row in normalized_rows}
    if len(fixture_refs) != 1:
        raise ValueError("analyze_odds_value expects odds rows for exactly one fixture.")

    for row in normalized_rows:
        if not isinstance(row, NormalizedOdds):
            raise TypeError("analyze_odds_value expects NormalizedOdds instances only.")

    return normalized_rows


def _build_provider_books(
    odds_rows: Sequence[NormalizedOdds],
) -> tuple[_ProviderMarketBook, ...]:
    """Build overround-adjusted provider books for each scoreable market."""

    grouped_rows: dict[ProviderBookKey, dict[str, float]] = {}
    for row in odds_rows:
        if row.market is None:
            continue
        group_key = (
            row.market,
            row.line,
            row.period,
            row.participant_scope,
            row.provider,
        )
        selection_odds = grouped_rows.setdefault(group_key, {})
        current_best = selection_odds.get(row.selection)
        if current_best is None or row.odds > current_best:
            selection_odds[row.selection] = row.odds

    provider_books: list[_ProviderMarketBook] = []
    for group_key, selection_odds in grouped_rows.items():
        if len(selection_odds) < 2:
            continue

        implied_total = sum(1.0 / offered_odds for offered_odds in selection_odds.values())
        if implied_total <= 0.0:
            continue

        market_key: MarketKey = group_key[:4]
        provider_books.append(
            _ProviderMarketBook(
                market_key=market_key,
                provider=group_key[4],
                normalized_probabilities={
                    selection: ((1.0 / offered_odds) / implied_total)
                    for selection, offered_odds in selection_odds.items()
                },
                offered_odds=selection_odds,
            )
        )

    return tuple(provider_books)


def _calculate_adjusted_edges(
    provider_books: Sequence[_ProviderMarketBook],
) -> tuple[float, ...]:
    """Return the adjusted value edge for every scoreable market selection."""

    books_by_market: dict[MarketKey, list[_ProviderMarketBook]] = {}
    for book in provider_books:
        books_by_market.setdefault(book.market_key, []).append(book)

    adjusted_edges: list[float] = []
    for market_key, market_books in books_by_market.items():
        del market_key
        provider_count = len(market_books)
        probabilities_by_selection: dict[str, list[float]] = {}
        odds_by_selection: dict[str, list[float]] = {}

        for book in market_books:
            for selection, probability in book.normalized_probabilities.items():
                probabilities_by_selection.setdefault(selection, []).append(probability)
                offered_price = book.offered_odds.get(selection)
                if offered_price is not None:
                    odds_by_selection.setdefault(selection, []).append(offered_price)

        for selection, probabilities in probabilities_by_selection.items():
            selection_odds = odds_by_selection.get(selection)
            if not selection_odds:
                continue

            consensus_probability = fmean(probabilities)
            best_odds = max(selection_odds)
            best_implied_probability = 1.0 / best_odds
            raw_edge = max(0.0, consensus_probability - best_implied_probability)

            probability_range = max(probabilities) - min(probabilities)
            consensus_tightness = _clamp(1.0 - (probability_range / 0.20))
            provider_coverage = _clamp(len(probabilities) / provider_count)
            provider_reliability = _clamp(provider_count / 3.0)
            adjustment_multiplier = (
                provider_coverage
                * provider_reliability
                * (0.70 + (0.30 * consensus_tightness))
            )

            adjusted_edges.append(raw_edge * adjustment_multiplier)

    return tuple(adjusted_edges)


def _score_edge(edge: float) -> float:
    """Map a probability edge into the spec's bounded odds-value score."""

    if edge > 0.10:
        return 0.90
    if edge > 0.05:
        return 0.70
    if edge > 0.02:
        return 0.50
    if edge > 0.0:
        return 0.30
    return 0.10


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into a bounded 0.0-1.0 interval."""

    return max(0.0, min(1.0, value))


__all__ = ["analyze_odds_value"]
