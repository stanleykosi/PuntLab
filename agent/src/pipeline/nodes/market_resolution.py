"""Market-resolution node for PuntLab's LangGraph pipeline.

Purpose: resolve ranked match recommendations into sportsbook-ready markets
using the canonical resolver fallback chain before accumulator building.
Scope: fixture lookup, deterministic resolver orchestration, full odds-catalog
fallback support, and recoverable diagnostics for per-fixture resolution
failures that should not block the rest of the slate.
Dependencies: `src.scrapers.resolver.MarketResolver` for source fallback,
`src.pipeline.state.PipelineState` for validated state IO, and the shared
fixture, ranking, and resolved-market schemas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.pipeline.state import PipelineStage, PipelineState
from src.providers.base import ProviderError
from src.schemas.accumulators import ResolvedMarket
from src.schemas.analysis import RankedMatch
from src.schemas.fixtures import NormalizedFixture
from src.scrapers.resolver import MarketResolver


async def market_resolution_node(
    state: PipelineState | Mapping[str, Any],
    *,
    resolver: MarketResolver | None = None,
) -> dict[str, object]:
    """Execute the market-resolution stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        resolver: Optional injected market resolver for tests or explicit
            runtime wiring.

    Outputs:
        A partial LangGraph update containing the successfully resolved market
        rows, merged diagnostics, and the next stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    fixture_index = _index_fixtures(validated_state.fixtures)
    external_odds = validated_state.odds_market_catalog.all_rows()
    diagnostics: list[str] = []
    resolved_markets: list[ResolvedMarket] = []

    stage_resolver = resolver or MarketResolver()
    try:
        for ranked_match in _normalize_ranked_matches(validated_state.ranked_matches):
            fixture = fixture_index.get(ranked_match.fixture_ref)
            if fixture is None:
                diagnostics.append(
                    "Market resolution skipped for "
                    f"{ranked_match.fixture_ref}: no matching fixture exists in pipeline state."
                )
                continue

            try:
                resolved_markets.append(
                    await stage_resolver.resolve(
                        fixture,
                        ranked_match,
                        external_odds=external_odds,
                    )
                )
            except (ProviderError, ValueError) as exc:
                diagnostics.append(
                    "Market resolution failed for "
                    f"{ranked_match.fixture_ref}: {_format_resolution_error(exc)}"
                )
    finally:
        await stage_resolver.aclose()

    return {
        "current_stage": PipelineStage.ACCUMULATOR_BUILDING,
        "resolved_markets": resolved_markets,
        "errors": _merge_diagnostics(validated_state.errors, diagnostics),
    }


def _index_fixtures(fixtures: Sequence[NormalizedFixture]) -> dict[str, NormalizedFixture]:
    """Index canonical fixtures by `fixture_ref`, keeping the first occurrence.

    Inputs:
        fixtures: Ordered fixture slate already validated in pipeline state.

    Outputs:
        A fixture lookup keyed by canonical fixture reference.
    """

    indexed_fixtures: dict[str, NormalizedFixture] = {}
    for fixture in fixtures:
        indexed_fixtures.setdefault(fixture.get_fixture_ref(), fixture)
    return indexed_fixtures


def _normalize_ranked_matches(ranked_matches: Sequence[RankedMatch]) -> tuple[RankedMatch, ...]:
    """Validate ranked-match inputs and deduplicate by fixture reference.

    Inputs:
        ranked_matches: Ordered ranked recommendations from the ranking stage.

    Outputs:
        A tuple preserving the first occurrence of each fixture reference so
        the highest-ranked recommendation for a fixture is the one resolved.

    Raises:
        TypeError: If any supplied item is not a canonical `RankedMatch`.
    """

    normalized_matches: list[RankedMatch] = []
    seen_fixture_refs: set[str] = set()

    for ranked_match in ranked_matches:
        if not isinstance(ranked_match, RankedMatch):
            raise TypeError("market_resolution_node expects RankedMatch instances only.")
        if ranked_match.fixture_ref in seen_fixture_refs:
            continue
        seen_fixture_refs.add(ranked_match.fixture_ref)
        normalized_matches.append(ranked_match)

    return tuple(normalized_matches)


def _merge_diagnostics(existing_errors: Sequence[str], diagnostics: Sequence[str]) -> list[str]:
    """Append new diagnostics to the pipeline error list without duplicates."""

    merged_errors = list(existing_errors)
    seen_messages = {message.casefold() for message in merged_errors}

    for diagnostic in diagnostics:
        lookup_key = diagnostic.casefold()
        if lookup_key in seen_messages:
            continue
        seen_messages.add(lookup_key)
        merged_errors.append(diagnostic)

    return merged_errors


def _format_resolution_error(error: ProviderError | ValueError) -> str:
    """Render one resolution error into an operator-friendly diagnostic string."""

    if isinstance(error, ProviderError):
        return f"[{error.provider}] {error}"
    return str(error)


__all__ = ["market_resolution_node"]
