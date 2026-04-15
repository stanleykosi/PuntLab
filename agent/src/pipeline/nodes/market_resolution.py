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

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.pipeline.state import PipelineStage, PipelineState
from src.providers.base import ProviderError
from src.schemas.accumulators import ResolvedMarket
from src.schemas.analysis import RankedMatch
from src.schemas.fixtures import NormalizedFixture
from src.scrapers.resolver import MarketResolver

_RESOLUTION_FAILURE_PATTERN = re.compile(
    r"^Market resolution failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_RESOLUTION_SKIPPED_PATTERN = re.compile(
    r"^Market resolution skipped for (?P<fixture>.+?): (?P<reason>.+)$"
)
_FIXTURE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:sr:match:\d+|football-data:\d+|api-football:\d+|balldontlie:\d+|the-odds-api:[a-z0-9_-]+)\b",
    flags=re.IGNORECASE,
)


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
            if (
                ranked_match.recommended_market is None
                or ranked_match.recommended_selection is None
            ):
                diagnostics.append(
                    "Market resolution skipped for "
                    f"{ranked_match.fixture_ref}: ranked match lacks a canonical "
                    "recommended market and selection."
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
        "errors": _merge_diagnostics(
            validated_state.errors,
            _summarize_resolution_diagnostics(diagnostics),
        ),
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
        detail = str(error)
        if error.provider == "market-resolver":
            return _normalize_market_resolver_detail(detail)
        return f"[{error.provider}] {detail}"
    return str(error)


def _normalize_market_resolver_detail(detail: str) -> str:
    """Collapse verbose resolver diagnostics to stable operator-facing causes.

    Inputs:
        detail: Raw `ProviderError` text emitted by `MarketResolver`.

    Outputs:
        A concise root-cause message with fixture-specific details removed so
        repeated resolver failures aggregate cleanly in node diagnostics.
    """

    normalized = " ".join(detail.split())
    if (
        "External odds fallback did not contain a compatible canonical market for the fixture."
        in normalized
    ):
        return (
            "[market-resolver] no compatible canonical odds were found for the "
            "recommended market and selection."
        )
    normalized = _FIXTURE_IDENTIFIER_PATTERN.sub("<fixture>", normalized)
    return f"[market-resolver] {normalized}"


def _summarize_resolution_diagnostics(diagnostics: Sequence[str]) -> tuple[str, ...]:
    """Compress repeated fixture-level market-resolution diagnostics.

    Inputs:
        diagnostics: Ordered per-fixture market-resolution diagnostics.

    Outputs:
        A tuple where repeated failures/skips with the same reason are grouped
        into concise root-cause summaries.
    """

    summarized: list[str] = []
    failures_by_reason: dict[str, list[str]] = {}
    failure_reason_examples: dict[str, str] = {}
    skips_by_reason: dict[str, list[str]] = {}
    skip_reason_examples: dict[str, str] = {}

    for diagnostic in diagnostics:
        failure_match = _RESOLUTION_FAILURE_PATTERN.match(diagnostic)
        if failure_match is not None:
            raw_reason = failure_match.group("reason").strip()
            reason = _normalize_resolution_reason(raw_reason)
            fixture_ref = failure_match.group("fixture").strip()
            failures_by_reason.setdefault(reason, []).append(fixture_ref)
            failure_reason_examples.setdefault(reason, raw_reason)
            continue

        skipped_match = _RESOLUTION_SKIPPED_PATTERN.match(diagnostic)
        if skipped_match is not None:
            raw_reason = skipped_match.group("reason").strip()
            reason = _normalize_resolution_reason(raw_reason)
            fixture_ref = skipped_match.group("fixture").strip()
            skips_by_reason.setdefault(reason, []).append(fixture_ref)
            skip_reason_examples.setdefault(reason, raw_reason)
            continue

        summarized.append(diagnostic)

    for reason, fixture_refs in failures_by_reason.items():
        if len(fixture_refs) == 1:
            reason_for_single = failure_reason_examples.get(reason, reason)
            summarized.append(
                f"Market resolution failed for {fixture_refs[0]}: {reason_for_single}"
            )
            continue
        normalized_reason = reason.rstrip(".")
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Market resolution failed for "
            f"{len(fixture_refs)} fixtures due to: {normalized_reason}. "
            f"Sample fixtures: {sample_refs}."
        )

    for reason, fixture_refs in skips_by_reason.items():
        if len(fixture_refs) == 1:
            reason_for_single = skip_reason_examples.get(reason, reason)
            summarized.append(
                f"Market resolution skipped for {fixture_refs[0]}: {reason_for_single}"
            )
            continue
        normalized_reason = reason.rstrip(".")
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Market resolution skipped for "
            f"{len(fixture_refs)} fixtures due to: {normalized_reason}. "
            f"Sample fixtures: {sample_refs}."
        )

    return tuple(summarized)


def _normalize_resolution_reason(reason: str) -> str:
    """Normalize fixture-specific identifiers from resolution reasons.

    Inputs:
        reason: Raw resolution failure/skip reason.

    Outputs:
        A stable reason string that enables root-cause aggregation.
    """

    normalized = " ".join(reason.split())
    normalized = _FIXTURE_IDENTIFIER_PATTERN.sub("<fixture>", normalized)
    return normalized


__all__ = ["market_resolution_node"]
