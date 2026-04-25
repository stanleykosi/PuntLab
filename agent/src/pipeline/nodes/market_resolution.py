"""LLM market-resolution node for PuntLab's LangGraph pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.llm import get_llm, get_prompt
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import ResolutionSource, ResolvedMarket
from src.schemas.analysis import RankedMatch
from src.schemas.fixtures import NormalizedFixture
from src.schemas.llm_decisions import LLMResolvedMarketChoice
from src.schemas.odds import NormalizedOdds
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
    prefetched_rows = validated_state.odds_market_catalog.all_rows()
    resolved_markets: list[ResolvedMarket] = []

    if resolver is not None:
        await resolver.aclose()
        raise RuntimeError("market_resolution_node is LLM-led; resolver fallback is disabled.")

    llm = await get_llm("market_resolution")
    prompt = get_prompt("market_resolution")
    for ranked_match in _normalize_ranked_matches(validated_state.ranked_matches):
        fixture = fixture_index.get(ranked_match.fixture_ref)
        if fixture is None:
            raise RuntimeError(
                f"Market resolution requires fixture {ranked_match.fixture_ref} in state."
            )
        fixture_rows = tuple(
            row for row in prefetched_rows if row.fixture_ref == ranked_match.fixture_ref
        )
        if not fixture_rows:
            raise RuntimeError(
                f"Market resolution requires SportyBet odds rows for {ranked_match.fixture_ref}."
            )
        fixture_rows = _candidate_rows_for_resolution(ranked_match, fixture_rows)
        row_by_id = {
            _row_id(index): row for index, row in enumerate(fixture_rows, start=1)
        }
        decision = await invoke_json_schema(
            llm=llm,
            prompt_messages=prompt.format_messages(
                ranked_match_summary=_render_ranked_match_summary(ranked_match),
                row_menu=_render_row_menu(row_by_id),
            ),
            schema=LLMResolvedMarketChoice,
            instruction=(
                "Return JSON with fixture_ref, row_id, confidence, and rationale. "
                "The row_id must be copied exactly from the supplied row menu."
            ),
        )
        if decision.fixture_ref != ranked_match.fixture_ref:
            raise ValueError(
                "LLM market resolution returned fixture_ref "
                f"{decision.fixture_ref!r}; expected {ranked_match.fixture_ref!r}."
            )
        selected_row = row_by_id.get(decision.row_id)
        if selected_row is None:
            raise ValueError(
                f"LLM market resolution selected unknown row_id {decision.row_id!r} "
                f"for {ranked_match.fixture_ref}."
            )
        resolved_markets.append(_build_resolved_market(selected_row, fixture))

    return {
        "current_stage": PipelineStage.ACCUMULATOR_BUILDING,
        "resolved_markets": resolved_markets,
        "errors": list(validated_state.errors),
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


def _row_id(index: int) -> str:
    """Return a stable prompt row id."""

    return f"row_{index}"


def _candidate_rows_for_resolution(
    ranked_match: RankedMatch,
    fixture_rows: Sequence[NormalizedOdds],
) -> tuple[NormalizedOdds, ...]:
    """Return the compact row set the LLM should resolve from."""

    market_key = (ranked_match.recommended_market or "").casefold()
    selection_key = (ranked_match.recommended_selection or "").casefold()
    exact_rows = tuple(
        row
        for row in fixture_rows
        if (
            market_key
            and (
                (row.provider_market_key or "").casefold() == market_key
                or row.provider_market_name.casefold() == market_key
                or (row.market_label or "").casefold() == market_key
            )
        )
    )
    if exact_rows:
        selection_rows = tuple(
            row
            for row in exact_rows
            if selection_key
            and (
                row.provider_selection_name.casefold() == selection_key
                or row.selection.casefold() == selection_key
                or (row.provider_selection_key or "").casefold() == selection_key
            )
        )
        return selection_rows or exact_rows
    return tuple(fixture_rows[:200])


def _render_ranked_match_summary(ranked_match: RankedMatch) -> str:
    """Render one ranked recommendation for the market-resolution prompt."""

    return (
        f"rank={ranked_match.rank}; fixture_ref={ranked_match.fixture_ref}; "
        f"{ranked_match.home_team} vs {ranked_match.away_team}; "
        f"competition={ranked_match.competition}; score={ranked_match.composite_score:.2f}; "
        f"confidence={ranked_match.confidence:.2f}; recommended_market="
        f"{ranked_match.recommended_market_label} [key={ranked_match.recommended_market}]; "
        f"selection={ranked_match.recommended_selection}; odds={ranked_match.recommended_odds}; "
        f"line={ranked_match.recommended_line}; summary={ranked_match.qualitative_summary or '-'}"
    )


def _render_row_menu(row_by_id: Mapping[str, NormalizedOdds]) -> str:
    """Render concrete SportyBet rows with exact row ids."""

    lines: list[str] = []
    for row_id, row in row_by_id.items():
        lines.append(
            f"- {row_id}: market={row.market_label or row.provider_market_name} "
            f"[key={row.provider_market_key} id={row.provider_market_id}]; "
            f"selection={row.provider_selection_name} [key={row.provider_selection_key}]; "
            f"odds={row.odds:.2f}; line={row.line}; canonical_market="
            f"{row.market.value if row.market is not None else None}; period={row.period}; "
            f"group={row.raw_metadata.get('market_group_name')}"
        )
    return "\n".join(lines)


def _build_resolved_market(row: NormalizedOdds, fixture: NormalizedFixture) -> ResolvedMarket:
    """Build the final resolved market from the LLM-selected SportyBet row."""

    return ResolvedMarket.model_validate(
        {
            **row.model_dump(),
            "market": row.provider_market_key or row.provider_market_name,
            "canonical_market": row.market,
            "resolution_source": ResolutionSource.SPORTYBET_API,
            "sport": fixture.sport,
            "competition": fixture.competition,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "sportybet_market_id": row.provider_market_id
            if isinstance(row.provider_market_id, int)
            else None,
            "sportybet_url": getattr(fixture, "sportybet_url", None),
        }
    )


__all__ = ["market_resolution_node"]
