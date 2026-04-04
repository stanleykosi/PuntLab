"""Tests for PuntLab's ingestion pipeline node.

Purpose: verify the ingestion stage gathers the expected normalized datasets,
preserves the full odds-market universe, and forwards recoverable diagnostics.
Scope: unit tests for `src.pipeline.nodes.ingestion`.
Dependencies: pytest plus lightweight async orchestrator stubs and shared
schema helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.ingestion import ingestion_node
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import OddsMarketCatalog, build_odds_market_catalog
from src.providers.orchestrator import (
    InjuryFetchResult,
    OddsFetchResult,
    StatsFetchResult,
)
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, PlayerStats, TeamStats


def build_fixture() -> NormalizedFixture:
    """Create a representative fixture for ingestion-node tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 0, tzinfo=UTC),
        source_provider="api-football",
        source_id="7001",
        country="England",
        home_team_id="42",
        away_team_id="49",
    )


def build_state() -> PipelineState:
    """Create a minimal validated pipeline state for ingestion execution."""

    return PipelineState(
        run_id="run-2026-04-04-main",
        run_date=date(2026, 4, 4),
        started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        errors=["Existing upstream notice."],
    )


@dataclass(slots=True)
class StubProviderOrchestrator:
    """Async stub matching the ingestion node's provider-orchestrator contract."""

    fixtures: tuple[NormalizedFixture, ...]
    odds_result: OddsFetchResult
    stats_result: StatsFetchResult
    injuries_result: InjuryFetchResult
    news_articles: tuple[NewsArticle, ...]

    async def fetch_fixtures(
        self,
        *,
        run_date: date,
        competitions: tuple[object, ...] | None = None,
        season_overrides: dict[str, int] | None = None,
    ) -> tuple[NormalizedFixture, ...]:
        """Return the configured fixture slate."""

        del run_date, competitions, season_overrides
        return self.fixtures

    async def fetch_odds(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        markets_by_sport: dict[SportName, tuple[str, ...]] | None = None,
    ) -> OddsFetchResult:
        """Return the configured odds result bundle."""

        del fixtures, markets_by_sport
        return self.odds_result

    async def fetch_stats(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        season_overrides: dict[str, int] | None = None,
        include_player_stats: bool = True,
    ) -> StatsFetchResult:
        """Return the configured stats result bundle."""

        del fixtures, season_overrides, include_player_stats
        return self.stats_result

    async def fetch_injuries(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        season_overrides: dict[str, int] | None = None,
    ) -> InjuryFetchResult:
        """Return the configured injury result bundle."""

        del fixtures, season_overrides
        return self.injuries_result

    async def fetch_news(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[NewsArticle, ...]:
        """Return the configured news results."""

        del fixtures
        return self.news_articles


@pytest.mark.asyncio
async def test_ingestion_node_preserves_full_odds_catalog_and_merges_news() -> None:
    """The ingestion node should store both scoreable odds and the full catalog."""

    fixture = build_fixture()
    full_odds_catalog = build_odds_market_catalog(
        (
            NormalizedOdds(
                fixture_ref=fixture.get_fixture_ref(),
                market=None,
                selection="Arsenal",
                odds=1.92,
                provider="the-odds-api",
                provider_market_name="Match Winner",
                provider_selection_name="Arsenal",
                period="match",
                participant_scope="match",
                raw_metadata={
                    "sport_key": "soccer_epl",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                },
            ),
            NormalizedOdds(
                fixture_ref=fixture.get_fixture_ref(),
                market=None,
                selection="Arsenal Over 1.5",
                odds=2.20,
                provider="the-odds-api",
                provider_market_name="Team Totals",
                provider_selection_name="Arsenal Over 1.5",
                line=1.5,
                period="match",
                participant_scope="team",
                raw_metadata={"sport_key": "soccer_epl"},
            ),
        ),
        sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
    )
    supporting_article = NewsArticle(
        headline="Chelsea monitor late fitness concerns before derby",
        url="https://example.com/articles/chelsea-fitness",
        published_at=datetime(2026, 4, 4, 8, 15, tzinfo=UTC),
        source="BBC Sport",
        source_provider="tavily",
        fixture_ref=fixture.get_fixture_ref(),
        teams=("Chelsea",),
    )
    primary_article = NewsArticle(
        headline="Arsenal look to extend home winning streak",
        url="https://example.com/articles/arsenal-home-streak",
        published_at=datetime(2026, 4, 4, 7, 45, tzinfo=UTC),
        source="Sky Sports",
        source_provider="rss",
        fixture_ref=fixture.get_fixture_ref(),
        teams=("Arsenal",),
    )

    result = await ingestion_node(
        build_state(),
        orchestrator=StubProviderOrchestrator(
            fixtures=(fixture,),
            odds_result=OddsFetchResult(
                catalog=full_odds_catalog,
                matched_rows=full_odds_catalog.all_rows(),
                unmatched_fixture_refs=(),
                providers_attempted=("the-odds-api",),
            ),
            stats_result=StatsFetchResult(
                team_stats=(
                    TeamStats(
                        team_id="42",
                        team_name="Arsenal",
                        sport=SportName.SOCCER,
                        source_provider="api-football",
                        fetched_at=datetime(2026, 4, 4, 7, 5, tzinfo=UTC),
                        competition="Premier League",
                        matches_played=30,
                        wins=21,
                        draws=5,
                        losses=4,
                    ),
                ),
                player_stats=(
                    PlayerStats(
                        player_id="saka",
                        player_name="Bukayo Saka",
                        team_id="42",
                        sport=SportName.SOCCER,
                        source_provider="api-football",
                        fetched_at=datetime(2026, 4, 4, 7, 6, tzinfo=UTC),
                        team_name="Arsenal",
                        appearances=26,
                        starts=25,
                        metrics={"goals": 13.0},
                    ),
                ),
                providers_attempted=("api-football",),
            ),
            injuries_result=InjuryFetchResult(
                injuries=(
                    InjuryData(
                        fixture_ref=fixture.get_fixture_ref(),
                        team_id="49",
                        player_name="Reece James",
                        source_provider="api-football",
                        injury_type=InjuryType.INJURY,
                        reported_at=datetime(2026, 4, 4, 6, 50, tzinfo=UTC),
                    ),
                ),
                supporting_articles=(supporting_article,),
                providers_attempted=("api-football", "tavily"),
            ),
            news_articles=(primary_article,),
        ),
    )

    assert result["current_stage"] == PipelineStage.RESEARCH
    assert len(result["fixtures"]) == 1
    assert isinstance(result["odds_market_catalog"], OddsMarketCatalog)
    assert len(result["odds_market_catalog"].all_rows()) == 2
    assert [(row.market, row.selection) for row in result["odds_data"]] == [
        (MarketType.MATCH_RESULT, "home")
    ]
    assert [article.headline for article in result["news_articles"]] == [
        "Arsenal look to extend home winning streak",
        "Chelsea monitor late fitness concerns before derby",
    ]
    assert (
        "Preserved unmapped odds markets remain outside the current canonical "
        "scoring taxonomy."
    ) in result["errors"]
    assert result["errors"][0] == "Existing upstream notice."


@pytest.mark.asyncio
async def test_ingestion_node_records_empty_fixture_and_provider_warnings() -> None:
    """The ingestion node should surface empty slates and partial odds coverage."""

    result = await ingestion_node(
        build_state(),
        orchestrator=StubProviderOrchestrator(
            fixtures=(),
            odds_result=OddsFetchResult(
                catalog=OddsMarketCatalog(),
                matched_rows=(),
                unmatched_fixture_refs=("sr:match:missing-1",),
                providers_attempted=("the-odds-api",),
                warnings=("The Odds API odds fetch failed: quota exhausted",),
            ),
            stats_result=StatsFetchResult(
                team_stats=(),
                player_stats=(),
                providers_attempted=("api-football",),
                warnings=("API-Football stats fetch failed for england_premier_league: timeout",),
            ),
            injuries_result=InjuryFetchResult(
                injuries=(),
                supporting_articles=(),
                providers_attempted=("api-football",),
                warnings=("API-Football injury fetch failed for england_premier_league: timeout",),
            ),
            news_articles=(),
        ),
    )

    assert result["fixtures"] == []
    assert result["odds_data"] == []
    assert result["news_articles"] == []
    assert result["errors"] == [
        "Existing upstream notice.",
        "No eligible fixtures were ingested for 2026-04-04.",
        "The Odds API odds fetch failed: quota exhausted",
        "API-Football stats fetch failed for england_premier_league: timeout",
        "API-Football injury fetch failed for england_premier_league: timeout",
        "Odds coverage is incomplete for fixtures: sr:match:missing-1",
    ]
