"""Ingestion node for PuntLab's LangGraph pipeline.

Purpose: collect the canonical daily slate of fixtures, full odds-market
catalogs, stats, injuries, and news before downstream research and scoring.
Scope: orchestrator-driven stage execution, deterministic state updates, and
diagnostic propagation for partial provider coverage.
Dependencies: `src.providers.orchestrator.ProviderOrchestrator` for upstream
data access plus `src.pipeline.state.PipelineState` for validated state IO.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import OddsMarketCatalog
from src.providers.orchestrator import (
    FixtureDetailsFetchResult,
    InjuryFetchResult,
    OddsFetchResult,
    ProviderOrchestrator,
    StatsFetchResult,
)
from src.schemas.news import NewsArticle


def _deduplicate_articles(articles: Sequence[NewsArticle]) -> list[NewsArticle]:
    """Return articles in source order with URL-first deduplication applied.

    Inputs:
        articles: News rows collected from the main news pipeline and injury
            fallback research.

    Outputs:
        A list preserving the first occurrence of each article URL or source ID.
    """

    deduplicated: list[NewsArticle] = []
    seen_keys: set[tuple[str, str]] = set()

    for article in articles:
        source_id = article.source_id or ""
        dedupe_key = ("source_id", source_id.casefold()) if source_id else (
            "url",
            str(article.url).casefold(),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduplicated.append(article)

    return deduplicated


def _extend_errors(state_errors: Sequence[str], *diagnostic_groups: Sequence[str]) -> list[str]:
    """Merge new stage diagnostics into the state error list without duplicates.

    Inputs:
        state_errors: Existing errors already carried by the pipeline state.
        diagnostic_groups: Ordered collections of warnings or recoverable
            errors returned by the provider orchestrator.

    Outputs:
        A deduplicated list preserving the original message order.
    """

    merged_errors: list[str] = list(state_errors)
    seen_messages = {message.casefold() for message in merged_errors}

    for diagnostics in diagnostic_groups:
        for message in diagnostics:
            lookup_key = message.casefold()
            if lookup_key in seen_messages:
                continue
            seen_messages.add(lookup_key)
            merged_errors.append(message)

    return merged_errors

async def ingestion_node(
    state: PipelineState | Mapping[str, Any],
    *,
    orchestrator: ProviderOrchestrator | None = None,
) -> dict[str, object]:
    """Execute the canonical ingestion stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state, either as the validated model or a raw
            mapping LangGraph can deserialize into that model.
        orchestrator: Optional injected provider orchestrator for tests or
            custom runtime wiring.

    Outputs:
        A partial state update containing fixtures, the full odds catalog, the
        scoreable odds subset, team/player stats, injuries, merged news, stage
        diagnostics, and the next pipeline stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    stage_orchestrator = orchestrator or ProviderOrchestrator()

    fixtures = await stage_orchestrator.fetch_fixtures(
        run_date=validated_state.run_date,
    )
    odds_result = await stage_orchestrator.fetch_odds(fixtures=fixtures)
    fixture_details_result = await stage_orchestrator.fetch_fixture_details(
        fixtures=fixtures,
    )
    stats_result = await stage_orchestrator.fetch_stats(fixtures=fixtures)
    injuries_result = await stage_orchestrator.fetch_injuries(fixtures=fixtures)
    news_articles = await stage_orchestrator.fetch_news(fixtures=fixtures)

    merged_news_articles = _deduplicate_articles(
        (
            *news_articles,
            *injuries_result.supporting_articles,
        )
    )

    diagnostics = _build_ingestion_diagnostics(
        odds_result=odds_result,
        stats_result=stats_result,
        fixture_details_result=fixture_details_result,
        injuries_result=injuries_result,
        fixture_count=len(fixtures),
        run_date=validated_state.run_date.isoformat(),
    )

    return {
        "current_stage": PipelineStage.RESEARCH,
        "fixtures": list(fixtures),
        "odds_market_catalog": (
            odds_result.catalog if fixtures else OddsMarketCatalog()
        ),
        "odds_data": list(odds_result.catalog.scoreable_rows()),
        "team_stats": list(stats_result.team_stats),
        "player_stats": list(stats_result.player_stats),
        "fixture_details": list(fixture_details_result.fixture_details),
        "injuries": list(injuries_result.injuries),
        "news_articles": merged_news_articles,
        "errors": _extend_errors(
            validated_state.errors,
            diagnostics,
        ),
    }


def _build_ingestion_diagnostics(
    *,
    odds_result: OddsFetchResult,
    stats_result: StatsFetchResult,
    fixture_details_result: FixtureDetailsFetchResult,
    injuries_result: InjuryFetchResult,
    fixture_count: int,
    run_date: str,
) -> tuple[str, ...]:
    """Assemble ordered ingestion diagnostics for partial or empty coverage.

    Inputs:
        odds_result: Odds provider result bundle.
        stats_result: Statistics provider result bundle.
        injuries_result: Injury provider result bundle.
        fixture_count: Number of fixtures returned for the run.
        run_date: ISO date string for human-readable diagnostics.

    Outputs:
        An ordered tuple of deduplicated diagnostics suitable for
        `PipelineState.errors`.
    """

    diagnostics: list[str] = []

    if fixture_count == 0:
        diagnostics.append(f"No eligible fixtures were ingested for {run_date}.")

    diagnostics.extend(odds_result.warnings)
    diagnostics.extend(stats_result.warnings)
    diagnostics.extend(fixture_details_result.warnings)
    diagnostics.extend(injuries_result.warnings)

    if odds_result.unmatched_fixture_refs:
        diagnostics.append(
            "Odds coverage is incomplete for fixtures: "
            + ", ".join(sorted(odds_result.unmatched_fixture_refs))
        )

    if odds_result.catalog.unmapped_rows():
        diagnostics.append(
            "Preserved unmapped odds markets are available to research and provider-native "
            "resolution, but they remain outside the deterministic scoreable taxonomy."
        )

    # The node surfaces provider warnings as explicit diagnostics because later
    # stages need operator-visible context when the slate is only partially supported.
    return tuple(diagnostics)


__all__ = ["ingestion_node"]
