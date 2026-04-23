"""Tests for PuntLab's provider orchestrator.

Purpose: verify that the orchestration layer applies the configured fallback
chain, reconciles cross-provider fixture identities, and preserves the full
odds market catalog without live upstream dependencies.
Scope: unit tests for fixture fallback, odds matching and fallback behavior,
news fallback, injury fallback, and explicit SportyBet diagnostics.
Dependencies: pytest, the orchestrator module, and lightweight async stub
providers that emulate the concrete provider interfaces used in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pytest
from src.config import SUPPORTED_COMPETITIONS, CompetitionConfig, MarketType, SportName
from src.providers.base import ProviderError
from src.providers.orchestrator import ProviderOrchestrator
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, TeamStats
from src.scrapers.sportybet_fixture_probe import build_fixture_page_url
from src.scrapers.sportybet_fixture_stats import (
    SportyBetFixtureStatsResult,
    SportyBetFixtureStatsWidget,
)
from src.scrapers.sportybet_today import (
    SportyBetTodayEvent,
    SportyBetTodaySlate,
    SportyBetTodaySport,
)


def build_soccer_fixture(
    *,
    source_provider: str = "api-football",
    source_id: str = "501",
    sportradar_id: str = "sr:match:9001",
) -> NormalizedFixture:
    """Create a canonical Arsenal-Chelsea fixture used across tests."""

    return NormalizedFixture(
        sportradar_id=sportradar_id,
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        source_provider=source_provider,
        source_id=source_id,
        country="England",
        home_team_id="42",
        away_team_id="49",
    )


def build_nba_fixture() -> NormalizedFixture:
    """Create a canonical Lakers-Celtics fixture used in news fallback tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:9901",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        competition="NBA",
        sport=SportName.BASKETBALL,
        kickoff=datetime(2026, 4, 4, 2, 0, tzinfo=UTC),
        source_provider="balldontlie",
        source_id="15907925",
        country="United States",
        home_team_id="14",
        away_team_id="2",
    )


def get_competition(key: str) -> CompetitionConfig:
    """Return one configured competition record by stable key."""

    return next(competition for competition in SUPPORTED_COMPETITIONS if competition.key == key)


@dataclass
class StubAPIFootballProvider:
    """Async stub covering the API-Football methods used by the orchestrator."""

    fixtures_by_league: dict[tuple[int, date, int], list[NormalizedFixture]] = field(
        default_factory=dict
    )
    injuries_by_league: dict[tuple[int, int, date], list[InjuryData]] = field(default_factory=dict)
    fail_fixture_fetch: bool = False
    fixture_requests: list[tuple[int, date, int]] = field(default_factory=list)
    injury_requests: list[tuple[int, int, date]] = field(default_factory=list)

    provider_name: str = "api-football"

    async def fetch_fixtures_by_date(
        self,
        *,
        run_date: date,
        league_id: int,
        season: int,
        timezone: str | None = None,
    ) -> list[NormalizedFixture]:
        """Return configured fixtures or raise a forced upstream failure."""

        del timezone
        self.fixture_requests.append((league_id, run_date, season))
        if self.fail_fixture_fetch:
            raise ProviderError(self.provider_name, "primary provider outage")
        return list(self.fixtures_by_league.get((league_id, run_date, season), ()))

    async def fetch_injuries(
        self,
        *,
        fixture_id: int | None = None,
        league_id: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        player_id: int | None = None,
        report_date: date | None = None,
        timezone: str | None = None,
    ) -> list[InjuryData]:
        """Return configured injury rows for a league/day request."""

        del fixture_id, team_id, player_id, timezone
        assert league_id is not None
        assert season is not None
        assert report_date is not None
        self.injury_requests.append((league_id, season, report_date))
        return list(self.injuries_by_league.get((league_id, season, report_date), ()))

    async def fetch_head_to_head(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
        last: int | None = None,
        league_id: int | None = None,
        season: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[NormalizedFixture]:
        """Return no H2H rows in orchestrator tests."""

        del home_team_id, away_team_id, last, league_id, season, from_date, to_date
        return []


@dataclass
class StubSportyBetAPIClient:
    """Async stub for SportyBet API fixture-market fetches."""

    rows_by_sportradar_id: dict[str, list[NormalizedOdds]] = field(default_factory=dict)
    errors_by_sportradar_id: dict[str, ProviderError] = field(default_factory=dict)
    fetch_calls: list[str] = field(default_factory=list)

    async def fetch_markets(
        self,
        sportradar_id: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Return configured rows or raise a configured SportyBet API failure."""

        del fixture, use_cache
        self.fetch_calls.append(sportradar_id)
        if sportradar_id in self.errors_by_sportradar_id:
            raise self.errors_by_sportradar_id[sportradar_id]
        return tuple(self.rows_by_sportradar_id.get(sportradar_id, ()))

    def build_sportybet_url(self, fixture: NormalizedFixture) -> str:
        """Return a deterministic public SportyBet URL for the fixture."""

        assert fixture.sportradar_id is not None
        return (
            "https://www.sportybet.com/ng/sport/football/england/"
            f"premier-league/{fixture.home_team}_vs_{fixture.away_team}/{fixture.sportradar_id}"
        )

    async def aclose(self) -> None:
        """Satisfy the async cleanup protocol used by the orchestrator."""


@dataclass
class StubSportyBetBrowserScraper:
    """Async stub for SportyBet browser fallback fixture-market fetches."""

    rows_by_url: dict[str, list[NormalizedOdds]] = field(default_factory=dict)
    errors_by_url: dict[str, ProviderError] = field(default_factory=dict)
    scrape_calls: list[str] = field(default_factory=list)

    async def scrape_markets(
        self,
        url: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Return configured rows or raise a configured browser fallback failure."""

        del fixture, use_cache
        self.scrape_calls.append(url)
        if url in self.errors_by_url:
            raise self.errors_by_url[url]
        return tuple(self.rows_by_url.get(url, ()))

    async def aclose(self) -> None:
        """Satisfy the async cleanup protocol used by the orchestrator."""


@dataclass
class StubSportyBetTodayCollector:
    """Async stub for the SportyBet Today Games fixture collector."""

    slate: SportyBetTodaySlate
    collect_calls: list[tuple[str, ...]] = field(default_factory=list)

    async def collect(
        self,
        *,
        sports: tuple[str, ...] = (),
    ) -> SportyBetTodaySlate:
        """Return the configured slate and record the requested sport filter."""

        self.collect_calls.append(sports)
        return self.slate


@dataclass
class StubSportyBetFixtureStatsScraper:
    """Async stub for SportyBet fixture-page detail fetches."""

    results_by_url: dict[str, SportyBetFixtureStatsResult] = field(default_factory=dict)
    errors_by_url: dict[str, RuntimeError] = field(default_factory=dict)
    transient_errors_by_url: dict[str, list[RuntimeError]] = field(default_factory=dict)
    fetch_calls: list[str] = field(default_factory=list)

    async def fetch_fixture_stats(
        self,
        *,
        fixture_url: str,
        output_dir: object | None = None,
    ) -> SportyBetFixtureStatsResult:
        """Return configured fixture details or raise a configured failure."""

        del output_dir
        self.fetch_calls.append(fixture_url)
        transient_errors = self.transient_errors_by_url.get(fixture_url)
        if transient_errors:
            raise transient_errors.pop(0)
        if fixture_url in self.errors_by_url:
            raise self.errors_by_url[fixture_url]
        return self.results_by_url[fixture_url]


@dataclass
class StubRSSFeedProvider:
    """Async stub for RSS news collection."""

    articles: list[NewsArticle] = field(default_factory=list)
    fail: bool = False

    async def fetch_news(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        lookback_days: int = 3,
        max_entries_per_feed: int = 25,
    ) -> list[NewsArticle]:
        """Return configured articles or raise a forced failure."""

        del fixtures, lookback_days, max_entries_per_feed
        if self.fail:
            raise ProviderError("rss-feeds", "rss feed outage")
        return list(self.articles)


@dataclass
class StubTavilySearchProvider:
    """Async stub for Tavily match-news and injury fallback calls."""

    match_articles_by_fixture: dict[str, list[NewsArticle]] = field(default_factory=dict)
    injury_articles_by_fixture: dict[str, list[NewsArticle]] = field(default_factory=dict)
    match_calls: list[str] = field(default_factory=list)
    injury_calls: list[str] = field(default_factory=list)
    provider_name: str = "tavily"

    async def search_match_news(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int | None = None,
        search_depth: str | None = None,
        lookback_days: int = 7,
        include_domains: tuple[str, ...] | None = None,
        exclude_domains: tuple[str, ...] | None = None,
    ) -> list[NewsArticle]:
        """Return configured match-news fallback articles."""

        del max_results, search_depth, lookback_days, include_domains, exclude_domains
        fixture_ref = fixture.get_fixture_ref()
        self.match_calls.append(fixture_ref)
        return list(self.match_articles_by_fixture.get(fixture_ref, ()))

    async def search_injury_updates(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int | None = None,
        search_depth: str | None = None,
        lookback_days: int = 7,
        include_domains: tuple[str, ...] | None = None,
        exclude_domains: tuple[str, ...] | None = None,
    ) -> list[NewsArticle]:
        """Return configured injury fallback articles."""

        del max_results, search_depth, lookback_days, include_domains, exclude_domains
        fixture_ref = fixture.get_fixture_ref()
        self.injury_calls.append(fixture_ref)
        return list(self.injury_articles_by_fixture.get(fixture_ref, ()))


def build_sportybet_row(
    *,
    fixture: NormalizedFixture,
    market: MarketType,
    selection: str,
    provider_selection_name: str,
    odds: float,
    provider_market_id: int = 1,
    line: float | None = None,
    fetch_source: str = "api",
) -> NormalizedOdds:
    """Create one SportyBet-like odds row for orchestrator tests."""

    return NormalizedOdds(
        fixture_ref=fixture.get_fixture_ref(),
        market=market,
        selection=selection,
        odds=odds,
        provider="sportybet",
        provider_market_name="SportyBet Market",
        provider_selection_name=provider_selection_name,
        sportybet_available=True,
        provider_market_id=provider_market_id,
        line=line,
        raw_metadata={
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "requested_sportradar_id": fixture.sportradar_id,
            "market_group_id": "1001",
            "market_group_name": "Main",
            "sportybet_fetch_source": fetch_source,
        },
        last_updated=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
    )


def build_news_article(*, fixture: NormalizedFixture, url: str, source: str) -> NewsArticle:
    """Create a canonical news article tied to one fixture."""

    return NewsArticle(
        headline=f"{fixture.home_team} vs {fixture.away_team} preview",
        url=url,
        published_at=datetime(2026, 4, 4, 6, 0, tzinfo=UTC),
        source=source,
        source_provider=source.lower(),
        summary="Match preview",
        sport=fixture.sport,
        competition=fixture.competition,
        teams=(fixture.home_team, fixture.away_team),
        fixture_ref=fixture.get_fixture_ref(),
        relevance_score=0.9,
    )


def build_today_slate(
    *,
    events: tuple[SportyBetTodayEvent, ...],
) -> SportyBetTodaySlate:
    """Create one canonical today slate for orchestrator fixture tests."""

    return SportyBetTodaySlate(
        run_date=date(2026, 4, 4),
        fetched_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        timezone="Africa/Lagos",
        sports=(
            SportyBetTodaySport(
                sport_id="sr:sport:1",
                sport_name="Football",
                route_slug="football",
                page_url="https://www.sportybet.com/ng/sport/football/",
                event_size_hint=len(events),
            ),
        ),
        events=events,
    )


@pytest.mark.asyncio
async def test_fetch_fixtures_uses_sportybet_today_games_as_canonical_source() -> None:
    """Fixture ingestion should start from SportyBet Today Games, not provider leagues."""

    today_collector = StubSportyBetTodayCollector(
        slate=build_today_slate(
            events=(
                SportyBetTodayEvent(
                    event_id="sr:match:9001",
                    game_id="501",
                    sport_id="sr:sport:1",
                    sport_name="Football",
                    category_id="sr:category:30",
                    category_name="England",
                    tournament_id="sr:tournament:17",
                    tournament_name="Premier League",
                    kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
                    home_team_name="Arsenal",
                    away_team_name="Chelsea",
                    total_market_size=42,
                    market_count=2,
                    status=0,
                    match_status="Not start",
                ),
                SportyBetTodayEvent(
                    event_id="sr:match:9999",
                    game_id="999",
                    sport_id="sr:sport:1",
                    sport_name="Football",
                    category_id="sr:category:44",
                    category_name="Spain",
                    tournament_id="sr:tournament:8",
                    tournament_name="La Liga",
                    kickoff=datetime(2026, 4, 4, 20, 0, tzinfo=UTC),
                    home_team_name="Barcelona",
                    away_team_name="Valencia",
                    total_market_size=40,
                    market_count=2,
                    status=0,
                    match_status="Not start",
                ),
            ),
        )
    )

    orchestrator = ProviderOrchestrator(
        sportybet_today_collector=today_collector,
    )

    fixtures = await orchestrator.fetch_fixtures(
        run_date=date(2026, 4, 4),
        competitions=(get_competition("england_premier_league"),),
    )

    assert len(fixtures) == 1
    assert fixtures[0].get_fixture_ref() == "sr:match:9001"
    assert fixtures[0].source_provider == "sportybet"
    assert today_collector.collect_calls == [("football",)]


@pytest.mark.asyncio
async def test_fetch_odds_uses_sportybet_api_rows_and_builds_catalog() -> None:
    """SportyBet API rows should become the canonical odds catalog."""

    fixture = build_soccer_fixture()
    api_client = StubSportyBetAPIClient(
        rows_by_sportradar_id={
            fixture.sportradar_id: [
                build_sportybet_row(
                    fixture=fixture,
                    market=MarketType.MATCH_RESULT,
                    selection="home",
                    provider_selection_name="Home",
                    odds=1.88,
                )
            ]
        }
    )
    browser_scraper = StubSportyBetBrowserScraper()

    orchestrator = ProviderOrchestrator(
        sportybet_api_client=api_client,
        sportybet_browser_scraper=browser_scraper,
    )

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("sportybet_api",)
    assert result.catalog.all_rows()[0].fixture_ref == fixture.get_fixture_ref()
    assert result.catalog.scoreable_rows()[0].market == MarketType.MATCH_RESULT
    assert result.catalog.all_rows()[0].raw_metadata["sportybet_fetch_source"] == "api"
    assert result.catalog.all_rows()[0].raw_metadata["market_group_id"] == "1001"
    assert api_client.fetch_calls == ["sr:match:9001"]
    assert browser_scraper.scrape_calls == []


@pytest.mark.asyncio
async def test_fetch_odds_falls_back_to_sportybet_browser_when_api_fails() -> None:
    """Browser fallback should run when the SportyBet API cannot resolve a fixture."""

    fixture = build_soccer_fixture()
    api_client = StubSportyBetAPIClient(
        errors_by_sportradar_id={
            fixture.sportradar_id: ProviderError("sportybet", "api outage")
        }
    )
    sportybet_url = (
        "https://www.sportybet.com/ng/sport/football/england/"
        "premier-league/Arsenal_vs_Chelsea/sr:match:9001"
    )
    browser_scraper = StubSportyBetBrowserScraper(
        rows_by_url={
            sportybet_url: [
                build_sportybet_row(
                    fixture=fixture,
                    market=MarketType.OVER_UNDER_25,
                    selection="over",
                    provider_selection_name="Over 2.5",
                    odds=1.91,
                    provider_market_id=18,
                    line=2.5,
                    fetch_source="browser",
                )
            ]
        }
    )

    orchestrator = ProviderOrchestrator(
        sportybet_api_client=api_client,
        sportybet_browser_scraper=browser_scraper,
    )

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("sportybet_api", "sportybet_browser")
    assert result.catalog.scoreable_rows()[0].fixture_ref == "sr:match:9001"
    assert result.catalog.all_rows()[0].raw_metadata["sportybet_fetch_source"] == "browser"
    assert api_client.fetch_calls == ["sr:match:9001"]
    assert browser_scraper.scrape_calls == [sportybet_url]
    assert not result.warnings


@pytest.mark.asyncio
async def test_fetch_odds_reports_missing_sportradar_ids_without_attempting_fetch() -> None:
    """Fixtures without a Sportradar ID should be reported explicitly."""

    fixture = NormalizedFixture(
        sportradar_id=None,
        home_team="Sacramento Kings",
        away_team="LA Clippers",
        competition="NBA",
        sport=SportName.BASKETBALL,
        kickoff=datetime(2026, 4, 6, 1, 0, tzinfo=UTC),
        source_provider="balldontlie",
        source_id="18447959",
        country="United States",
        home_team_id="30",
        away_team_id="13",
    )
    api_client = StubSportyBetAPIClient()
    browser_scraper = StubSportyBetBrowserScraper()
    orchestrator = ProviderOrchestrator(
        sportybet_api_client=api_client,
        sportybet_browser_scraper=browser_scraper,
    )

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ("balldontlie:18447959",)
    assert result.providers_attempted == ()
    assert result.catalog.all_rows() == ()
    assert api_client.fetch_calls == []
    assert browser_scraper.scrape_calls == []
    assert "fixture.sportradar_id is required for canonical SportyBet odds fetching" in (
        result.warnings[0]
    )


@pytest.mark.asyncio
async def test_fetch_news_uses_tavily_for_fixtures_without_rss_coverage() -> None:
    """News should combine RSS coverage with Tavily fallback for gaps."""

    soccer_fixture = build_soccer_fixture()
    nba_fixture = build_nba_fixture()
    rss_article = build_news_article(
        fixture=soccer_fixture,
        url="https://www.bbc.com/sport/football/preview-1",
        source="BBC Sport",
    )
    tavily_article = build_news_article(
        fixture=nba_fixture,
        url="https://www.espn.com/nba/story/_/id/9002/lakers-celtics-preview",
        source="ESPN",
    )
    rss_provider = StubRSSFeedProvider(articles=[rss_article])
    tavily_provider = StubTavilySearchProvider(
        match_articles_by_fixture={nba_fixture.get_fixture_ref(): [tavily_article]}
    )

    orchestrator = ProviderOrchestrator(
        rss_feeds=rss_provider,
        tavily_search=tavily_provider,
    )

    articles = await orchestrator.fetch_news(fixtures=(soccer_fixture, nba_fixture))

    assert len(articles) == 2
    assert {article.fixture_ref for article in articles} == {
        soccer_fixture.get_fixture_ref(),
        nba_fixture.get_fixture_ref(),
    }
    assert tavily_provider.match_calls == [nba_fixture.get_fixture_ref()]


@pytest.mark.asyncio
async def test_fetch_stats_uses_sportybet_fixture_details_as_canonical_soccer_source() -> None:
    """Soccer team stats should now be derived from SportyBet fixture details."""

    fixture = build_soccer_fixture(source_provider="sportybet")
    fixture_url = build_fixture_page_url(
        event_id=fixture.sportradar_id or "",
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        country=fixture.country or "",
        competition=fixture.competition,
        sport="football",
    )
    scraper_result = SportyBetFixtureStatsResult(
        fixture_url=fixture_url,
        final_url=fixture_url,
        event_id=fixture.sportradar_id or "",
        match_id="9001",
        page_title="Arsenal vs Chelsea",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        home_team_uid="sportybet-home-42",
        away_team_uid="sportybet-away-49",
        widget_loader_status="loaded",
        team_stats=(
            TeamStats(
                team_id="sportybet-home-42",
                team_name="Arsenal FC",
                sport=SportName.SOCCER,
                source_provider="sportybet_fixture_stats",
                fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                competition="Premier League",
                season="25/26",
                matches_played=32,
                wins=21,
                draws=6,
                losses=5,
                goals_for=62,
                goals_against=28,
                form="WWDWWL",
                position=2,
                points=69,
                home_wins=12,
                away_wins=9,
                avg_goals_scored=1.9375,
                avg_goals_conceded=0.875,
            ),
            TeamStats(
                team_id="sportybet-away-49",
                team_name="Chelsea FC",
                sport=SportName.SOCCER,
                source_provider="sportybet_fixture_stats",
                fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                competition="Premier League",
                season="25/26",
                matches_played=32,
                wins=15,
                draws=8,
                losses=9,
                goals_for=51,
                goals_against=37,
                form="DLWWDW",
                position=6,
                points=53,
                home_wins=9,
                away_wins=6,
                avg_goals_scored=1.59375,
                avg_goals_conceded=1.15625,
            ),
        ),
        widgets=(),
        responses=(),
    )
    scraper = StubSportyBetFixtureStatsScraper(
        results_by_url={fixture_url: scraper_result}
    )
    orchestrator = ProviderOrchestrator(
        sportybet_fixture_stats_scraper=scraper,  # type: ignore[arg-type]
    )

    result = await orchestrator.fetch_stats(fixtures=(fixture,))

    assert result.providers_attempted == ("sportybet_fixture_stats",)
    assert result.player_stats == ()
    assert result.warnings == ()
    assert scraper.fetch_calls == [fixture_url]
    assert {stats.team_name for stats in result.team_stats} == {"Arsenal", "Chelsea"}
    home_stats = next(stats for stats in result.team_stats if stats.team_name == "Arsenal")
    away_stats = next(stats for stats in result.team_stats if stats.team_name == "Chelsea")
    assert home_stats.team_id == "42"
    assert away_stats.team_id == "49"
    assert home_stats.form == "WWDWWL"
    assert away_stats.form == "DLWWDW"


@pytest.mark.asyncio
async def test_fetch_stats_reuses_cached_fixture_detail_fetch_after_details_stage() -> None:
    """A fixture-details fetch should seed the later stats stage without another browser run."""

    fixture = build_soccer_fixture(source_provider="sportybet")
    fixture_url = build_fixture_page_url(
        event_id=fixture.sportradar_id or "",
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        country=fixture.country or "",
        competition=fixture.competition,
        sport="football",
    )
    scraper_result = SportyBetFixtureStatsResult(
        fixture_url=fixture_url,
        final_url=fixture_url,
        event_id=fixture.sportradar_id or "",
        match_id="9001",
        page_title="Arsenal vs Chelsea",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        home_team_uid="sportybet-home-42",
        away_team_uid="sportybet-away-49",
        widget_loader_status="loaded",
        team_stats=(
            TeamStats(
                team_id="sportybet-home-42",
                team_name="Arsenal FC",
                sport=SportName.SOCCER,
                source_provider="sportybet_fixture_stats",
                fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                competition="Premier League",
                matches_played=32,
                wins=21,
                draws=6,
                losses=5,
            ),
            TeamStats(
                team_id="sportybet-away-49",
                team_name="Chelsea FC",
                sport=SportName.SOCCER,
                source_provider="sportybet_fixture_stats",
                fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
                competition="Premier League",
                matches_played=32,
                wins=15,
                draws=8,
                losses=9,
            ),
        ),
        widgets=(
            SportyBetFixtureStatsWidget(
                widget_key="teamInfo",
                widget_type="team.info",
                status="mounted",
                content_lines=("Home manager: Mikel Arteta",),
            ),
        ),
        responses=(),
    )
    scraper = StubSportyBetFixtureStatsScraper(
        results_by_url={fixture_url: scraper_result}
    )
    orchestrator = ProviderOrchestrator(
        sportybet_fixture_stats_scraper=scraper,  # type: ignore[arg-type]
    )

    details_result = await orchestrator.fetch_fixture_details(fixtures=(fixture,))
    stats_result = await orchestrator.fetch_stats(fixtures=(fixture,))

    assert len(details_result.fixture_details) == 1
    assert len(stats_result.team_stats) == 2
    assert scraper.fetch_calls == [fixture_url]


@pytest.mark.asyncio
async def test_fetch_fixture_details_retries_transient_sportybet_failures() -> None:
    """The canonical fixture-details path should retry transient SportyBet failures."""

    fixture = build_soccer_fixture(source_provider="sportybet")
    fixture_url = build_fixture_page_url(
        event_id=fixture.sportradar_id or "",
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        country=fixture.country or "",
        competition=fixture.competition,
        sport="football",
    )
    scraper_result = SportyBetFixtureStatsResult(
        fixture_url=fixture_url,
        final_url=fixture_url,
        event_id=fixture.sportradar_id or "",
        match_id="9001",
        page_title="Arsenal vs Chelsea",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        widget_loader_status="loaded",
        widgets=(),
        responses=(),
    )
    scraper = StubSportyBetFixtureStatsScraper(
        results_by_url={fixture_url: scraper_result},
        transient_errors_by_url={fixture_url: [RuntimeError("temporary timeout")]},
    )
    orchestrator = ProviderOrchestrator(
        sportybet_fixture_stats_scraper=scraper,  # type: ignore[arg-type]
        sportybet_fixture_detail_retry_backoff_seconds=0.0,
    )

    result = await orchestrator.fetch_fixture_details(fixtures=(fixture,))

    assert len(result.fixture_details) == 1
    assert result.warnings == ()
    assert scraper.fetch_calls == [fixture_url, fixture_url]


@pytest.mark.asyncio
async def test_fetch_fixture_details_uses_sportybet_fixture_page_scraper() -> None:
    """Fixture details should be fetched from SportyBet fixture pages by Sportradar ID."""

    fixture = build_soccer_fixture(source_provider="sportybet")
    fixture_url = build_fixture_page_url(
        event_id=fixture.sportradar_id or "",
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        country=fixture.country or "",
        competition=fixture.competition,
        sport="football",
    )
    scraper_result = SportyBetFixtureStatsResult(
        fixture_url=fixture_url,
        final_url=fixture_url,
        event_id=fixture.sportradar_id or "",
        match_id="9001",
        page_title="Arsenal vs Chelsea",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        widget_loader_status="loaded",
        widgets=(
            SportyBetFixtureStatsWidget(
                widget_key="teamInfo",
                widget_type="team.info",
                status="mounted",
                content_lines=("Home manager: Mikel Arteta",),
            ),
        ),
    )
    scraper = StubSportyBetFixtureStatsScraper(
        results_by_url={fixture_url: scraper_result}
    )
    orchestrator = ProviderOrchestrator(
        sportybet_fixture_stats_scraper=scraper,  # type: ignore[arg-type]
    )

    result = await orchestrator.fetch_fixture_details(fixtures=(fixture,))

    assert scraper.fetch_calls == [fixture_url]
    assert result.providers_attempted == ("sportybet_fixture_stats",)
    assert result.warnings == ()
    assert len(result.fixture_details) == 1
    assert result.fixture_details[0].fixture_ref == fixture.get_fixture_ref()
    assert result.fixture_details[0].sections[0].content_lines == (
        "Home manager: Mikel Arteta",
    )


@pytest.mark.asyncio
async def test_fetch_injuries_uses_tavily_when_structured_provider_has_no_match() -> None:
    """Injury fallback should preserve Tavily articles for unresolved fixtures."""

    soccer_fixture = build_soccer_fixture()
    nba_fixture = build_nba_fixture()
    api_injury = InjuryData(
        fixture_ref="api-football:501",
        team_id="42",
        player_name="Bukayo Saka",
        source_provider="api-football",
        injury_type=InjuryType.INJURY,
        team_name="Arsenal",
        reported_at=datetime(2026, 4, 4, 5, 0, tzinfo=UTC),
    )
    api_provider = StubAPIFootballProvider(
        fixtures_by_league={
            (
                39,
                date(2026, 4, 4),
                2025,
            ): [
                build_soccer_fixture(
                    source_provider="api-football",
                    sportradar_id=None,
                )
            ]
        },
        injuries_by_league={(39, 2025, date(2026, 4, 4)): [api_injury]},
    )
    tavily_article = build_news_article(
        fixture=nba_fixture,
        url="https://www.espn.com/nba/story/_/id/9003/lakers-celtics-injury-report",
        source="ESPN",
    )
    tavily_provider = StubTavilySearchProvider(
        injury_articles_by_fixture={nba_fixture.get_fixture_ref(): [tavily_article]}
    )

    orchestrator = ProviderOrchestrator(
        api_football=api_provider,
        tavily_search=tavily_provider,
    )

    result = await orchestrator.fetch_injuries(fixtures=(soccer_fixture, nba_fixture))

    assert result.injuries[0].fixture_ref == soccer_fixture.get_fixture_ref()
    assert result.supporting_articles[0].fixture_ref == nba_fixture.get_fixture_ref()
    assert tavily_provider.injury_calls == [nba_fixture.get_fixture_ref()]


@pytest.mark.asyncio
async def test_fetch_injuries_warns_for_today_slate_competitions_without_provider_routing() -> None:
    """Unsupported today-slate competitions should still reach injury fallback with a warning."""

    unsupported_fixture = NormalizedFixture(
        sportradar_id="sr:match:9903",
        home_team="FC Noah",
        away_team="Pyunik",
        competition="Armenian Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 12, 30, tzinfo=UTC),
        source_provider="sportybet",
        source_id="9903",
        country="Armenia",
    )
    tavily_article = build_news_article(
        fixture=unsupported_fixture,
        url="https://example.com/armenia/injuries",
        source="Armenia Sports",
    )
    tavily_provider = StubTavilySearchProvider(
        injury_articles_by_fixture={unsupported_fixture.get_fixture_ref(): [tavily_article]}
    )
    orchestrator = ProviderOrchestrator(
        api_football=StubAPIFootballProvider(),
        tavily_search=tavily_provider,
    )

    result = await orchestrator.fetch_injuries(fixtures=(unsupported_fixture,))

    assert result.injuries == ()
    assert result.supporting_articles == (tavily_article,)
    assert result.warnings == (
        "Structured injury coverage is unavailable for 1 today-slate fixture from "
        "competitions without provider routing: Armenian Premier League (Armenia).",
    )
    assert tavily_provider.injury_calls == [unsupported_fixture.get_fixture_ref()]


@pytest.mark.asyncio
async def test_fetch_markets_reuses_canonical_sportybet_flow() -> None:
    """Market fetching should use the same canonical SportyBet path as fetch_odds."""

    fixture = build_soccer_fixture()
    api_client = StubSportyBetAPIClient(
        rows_by_sportradar_id={
            fixture.sportradar_id: [
                build_sportybet_row(
                    fixture=fixture,
                    market=MarketType.MATCH_RESULT,
                    selection="home",
                    provider_selection_name="Home",
                    odds=1.86,
                )
            ]
        }
    )
    orchestrator = ProviderOrchestrator(
        sportybet_api_client=api_client,
        sportybet_browser_scraper=StubSportyBetBrowserScraper(),
    )

    result = await orchestrator.fetch_markets(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("sportybet_api",)
    assert result.catalog.scoreable_rows()[0].market == MarketType.MATCH_RESULT
