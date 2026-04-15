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
from src.schemas.stats import InjuryData, InjuryType


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
    odds_by_fixture_id: dict[int, list[NormalizedOdds]] = field(default_factory=dict)
    injuries_by_league: dict[tuple[int, int, date], list[InjuryData]] = field(default_factory=dict)
    fail_fixture_fetch: bool = False
    fixture_requests: list[tuple[int, date, int]] = field(default_factory=list)
    odds_requests: list[int] = field(default_factory=list)
    injury_requests: list[tuple[int, int, date]] = field(default_factory=list)
    player_stats_requests: list[tuple[int, int, int]] = field(default_factory=list)

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

    async def fetch_odds_by_fixture(
        self,
        *,
        fixture_id: int,
        bookmaker_id: int | None = None,
        timezone: str | None = None,
    ) -> list[NormalizedOdds]:
        """Return configured fallback odds for one API-Football fixture."""

        del bookmaker_id, timezone
        self.odds_requests.append(fixture_id)
        return list(self.odds_by_fixture_id.get(fixture_id, ()))

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

    async def fetch_standings(self, *, league_id: int, season: int) -> list[object]:
        """Return an empty standings response for tests that do not need stats."""

        del league_id, season
        return []

    async def fetch_player_stats(
        self,
        *,
        season: int,
        team_id: int | None = None,
        league_id: int | None = None,
        player_id: int | None = None,
        search: str | None = None,
    ) -> list[object]:
        """Record player-stat requests and return no rows."""

        del player_id, search
        assert team_id is not None
        assert league_id is not None
        self.player_stats_requests.append((season, team_id, league_id))
        return []

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
class StubFootballDataProvider:
    """Async stub for Football-Data.org fixture fallback behavior."""

    fixtures_by_competition: dict[tuple[str, date, int | None], list[NormalizedFixture]] = field(
        default_factory=dict
    )
    fixture_requests: list[tuple[str, date, int | None]] = field(default_factory=list)
    provider_name: str = "football-data"

    async def fetch_fixtures_by_date(
        self,
        *,
        run_date: date,
        competition_code: str,
        season: int | None = None,
    ) -> list[NormalizedFixture]:
        """Return configured fallback fixtures for one competition/date."""

        self.fixture_requests.append((competition_code, run_date, season))
        return list(self.fixtures_by_competition.get((competition_code, run_date, season), ()))

    async def fetch_standings(
        self,
        *,
        competition_code: str,
        season: int | None = None,
        matchday: int | None = None,
        as_of_date: date | None = None,
    ) -> list[object]:
        """Return no standings rows for tests that only cover fixtures."""

        del competition_code, season, matchday, as_of_date
        return []


@dataclass
class StubTheOddsAPIProvider:
    """Async stub for The Odds API sport-level odds requests."""

    odds_by_sport: dict[str, list[NormalizedOdds]] = field(default_factory=dict)
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    fail: bool = False
    provider_name: str = "the-odds-api"

    async def fetch_odds(
        self,
        *,
        sport_key: str,
        markets: tuple[str, ...] = ("h2h",),
        regions: tuple[str, ...] | None = None,
        bookmakers: tuple[str, ...] | None = None,
        commence_time_from: datetime | None = None,
        commence_time_to: datetime | None = None,
    ) -> list[NormalizedOdds]:
        """Return configured odds rows or raise a forced failure."""

        del regions, bookmakers, commence_time_from, commence_time_to
        self.calls.append((sport_key, markets))
        if self.fail:
            raise ProviderError(self.provider_name, "quota exhausted")
        return list(self.odds_by_sport.get(sport_key, ()))


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


def build_the_odds_row() -> NormalizedOdds:
    """Create one The Odds API row that should match the Arsenal fixture."""

    return NormalizedOdds(
        fixture_ref="the-odds-api:event-123",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.88,
        provider="Pinnacle",
        provider_market_name="h2h",
        provider_selection_name="Arsenal",
        sportybet_available=False,
        raw_metadata={
            "event_id": "event-123",
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-04-04T19:45:00+00:00",
        },
        last_updated=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
    )


def build_api_fallback_row() -> NormalizedOdds:
    """Create one API-Football odds row used by the fallback-path test."""

    return NormalizedOdds(
        fixture_ref="api-football:501",
        market=MarketType.OVER_UNDER_25,
        selection="over",
        odds=1.91,
        provider="Bet365",
        provider_market_name="Goals Over/Under",
        provider_selection_name="Over 2.5",
        sportybet_available=False,
        line=2.5,
        raw_metadata={"canonical_market_supported": True},
        last_updated=datetime(2026, 4, 4, 11, 30, tzinfo=UTC),
    )


def build_nba_the_odds_row() -> NormalizedOdds:
    """Create one NBA odds row that uses a long-form away-team alias."""

    return NormalizedOdds(
        fixture_ref="the-odds-api:event-nba-1",
        market=MarketType.MONEYLINE,
        selection="away",
        odds=2.02,
        provider="Pinnacle",
        provider_market_name="h2h",
        provider_selection_name="Los Angeles Clippers",
        sportybet_available=False,
        raw_metadata={
            "event_id": "event-nba-1",
            "sport_key": "basketball_nba",
            "home_team": "Sacramento Kings",
            "away_team": "Los Angeles Clippers",
            "commence_time": "2026-04-06T01:10:00+00:00",
        },
        last_updated=datetime(2026, 4, 5, 20, 0, tzinfo=UTC),
    )


def build_serie_a_alias_row() -> NormalizedOdds:
    """Create one Serie A row with a short-form team label alias."""

    return NormalizedOdds(
        fixture_ref="the-odds-api:event-sa-1",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.95,
        provider="Pinnacle",
        provider_market_name="h2h",
        provider_selection_name="Inter Milan",
        sportybet_available=False,
        raw_metadata={
            "event_id": "event-sa-1",
            "sport_key": "soccer_italy_serie_a",
            "home_team": "Inter Milan",
            "away_team": "AS Roma",
            "commence_time": "2026-04-05T18:45:00+00:00",
        },
        last_updated=datetime(2026, 4, 5, 18, 0, tzinfo=UTC),
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


@pytest.mark.asyncio
async def test_fetch_fixtures_falls_back_to_football_data_when_api_football_fails() -> None:
    """Fixtures should fall back to Football-Data.org after a primary failure."""

    fallback_fixture = build_soccer_fixture(source_provider="football-data", source_id="fd-501")
    api_provider = StubAPIFootballProvider(fail_fixture_fetch=True)
    football_data = StubFootballDataProvider(
        fixtures_by_competition={("PL", date(2026, 4, 4), 2025): [fallback_fixture]}
    )

    orchestrator = ProviderOrchestrator(
        api_football=api_provider,
        football_data=football_data,
    )

    fixtures = await orchestrator.fetch_fixtures(
        run_date=date(2026, 4, 4),
        competitions=(get_competition("england_premier_league"),),
    )

    assert fixtures == (fallback_fixture,)
    assert api_provider.fixture_requests[0] == (39, date(2026, 4, 4), 2025)
    assert football_data.fixture_requests == [("PL", date(2026, 4, 4), 2025)]


@pytest.mark.asyncio
async def test_fetch_odds_matches_the_odds_rows_and_builds_catalog() -> None:
    """Primary odds should be matched back onto canonical fixture refs."""

    fixture = build_soccer_fixture()
    the_odds_api = StubTheOddsAPIProvider(odds_by_sport={"soccer_epl": [build_the_odds_row()]})

    orchestrator = ProviderOrchestrator(
        api_football=StubAPIFootballProvider(),
        the_odds_api=the_odds_api,
    )

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("the-odds-api",)
    assert result.catalog.all_rows()[0].fixture_ref == "sr:match:9001"
    assert result.catalog.scoreable_rows()[0].market == MarketType.MATCH_RESULT
    assert the_odds_api.calls == [
        (
            "soccer_epl",
            (
                "h2h",
                "totals",
                "spreads",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_fetch_odds_falls_back_to_api_football_and_reports_sportybet_gap() -> None:
    """Unmatched primary odds should fall back to API-Football before warning."""

    fixture = build_soccer_fixture()
    api_provider = StubAPIFootballProvider(odds_by_fixture_id={501: [build_api_fallback_row()]})
    the_odds_api = StubTheOddsAPIProvider(odds_by_sport={"soccer_epl": []})

    orchestrator = ProviderOrchestrator(
        api_football=api_provider,
        the_odds_api=the_odds_api,
    )

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("the-odds-api", "api-football")
    assert result.catalog.scoreable_rows()[0].fixture_ref == "sr:match:9001"
    assert api_provider.odds_requests == [501]
    assert not result.warnings


@pytest.mark.asyncio
async def test_fetch_odds_matches_nba_rows_when_team_names_use_common_aliases() -> None:
    """NBA odds matching should handle `LA` vs `Los Angeles` team aliases."""

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
    the_odds_api = StubTheOddsAPIProvider(
        odds_by_sport={"basketball_nba": [build_nba_the_odds_row()]}
    )
    orchestrator = ProviderOrchestrator(the_odds_api=the_odds_api)

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("the-odds-api",)
    assert len(result.catalog.all_rows()) == 1
    assert result.catalog.all_rows()[0].fixture_ref == "balldontlie:18447959"
    assert result.catalog.scoreable_rows()[0].market == MarketType.MONEYLINE


@pytest.mark.asyncio
async def test_fetch_odds_matches_soccer_rows_with_short_form_team_aliases() -> None:
    """Soccer matching should accept short-form aliases with exact kickoff match."""

    fixture = NormalizedFixture(
        sportradar_id=None,
        home_team="FC Internazionale Milano",
        away_team="AS Roma",
        competition="Serie A",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 5, 18, 45, tzinfo=UTC),
        source_provider="football-data",
        source_id="537116",
        country="Italy",
        home_team_id="108",
        away_team_id="100",
    )
    the_odds_api = StubTheOddsAPIProvider(
        odds_by_sport={"soccer_italy_serie_a": [build_serie_a_alias_row()]}
    )
    orchestrator = ProviderOrchestrator(the_odds_api=the_odds_api)

    result = await orchestrator.fetch_odds(fixtures=(fixture,))

    assert result.unmatched_fixture_refs == ()
    assert result.providers_attempted == ("the-odds-api",)
    assert len(result.catalog.all_rows()) == 1
    assert result.catalog.all_rows()[0].fixture_ref == "football-data:537116"
    assert result.catalog.scoreable_rows()[0].market == MarketType.MATCH_RESULT


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
async def test_fetch_markets_fails_fast_until_sportybet_steps_exist() -> None:
    """Market fetching should stay explicit until the scraper steps are built."""

    orchestrator = ProviderOrchestrator()

    with pytest.raises(ProviderError, match="Steps 22-24"):
        await orchestrator.fetch_markets(fixtures=(build_soccer_fixture(),))
