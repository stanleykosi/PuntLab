"""Tests for PuntLab's SportyBet market resolver.

Purpose: verify the resolver's source fallback order, canonical market
matching, external-odds fixture reconciliation, and error handling without
live SportyBet or odds-provider dependencies.
Scope: unit tests for `src.scrapers.resolver`.
Dependencies: pytest, the shared schemas, and lightweight async stub scraper
implementations that emulate the resolver's collaborators.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import MarketType, SportName
from src.providers.base import ProviderError
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.scrapers.resolver import MarketResolver


class StubSportyBetAPIClient:
    """Async SportyBet API stub used to drive resolver source-fallback tests."""

    def __init__(
        self,
        *,
        rows: tuple[NormalizedOdds, ...] = (),
        error: ProviderError | None = None,
    ) -> None:
        """Capture the rows or error this stub should return."""

        self._rows = rows
        self._error = error
        self.fetch_calls: list[tuple[str, bool]] = []

    async def fetch_markets(
        self,
        sportradar_id: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Return configured rows or raise the configured provider failure."""

        del fixture
        self.fetch_calls.append((sportradar_id, use_cache))
        if self._error is not None:
            raise self._error
        return self._rows

    def build_sportybet_url(self, fixture: NormalizedFixture) -> str:
        """Return a deterministic public SportyBet URL for the fixture."""

        assert fixture.sportradar_id is not None
        return (
            "https://www.sportybet.com/ng/sport/football/england/"
            f"premier-league/{fixture.home_team}_vs_{fixture.away_team}/{fixture.sportradar_id}"
        )

    async def aclose(self) -> None:
        """Satisfy the async cleanup protocol used by the resolver."""


class StubSportyBetBrowserScraper:
    """Async SportyBet browser stub used to verify fallback behavior."""

    def __init__(
        self,
        *,
        rows: tuple[NormalizedOdds, ...] = (),
        error: ProviderError | None = None,
    ) -> None:
        """Capture the rows or error this stub should return."""

        self._rows = rows
        self._error = error
        self.scrape_calls: list[tuple[str, bool]] = []

    async def scrape_markets(
        self,
        url: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Return configured rows or raise the configured provider failure."""

        del fixture
        self.scrape_calls.append((url, use_cache))
        if self._error is not None:
            raise self._error
        return self._rows

    async def aclose(self) -> None:
        """Satisfy the async cleanup protocol used by the resolver."""


def build_soccer_fixture() -> NormalizedFixture:
    """Create a canonical Arsenal-Chelsea fixture used in resolver tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
        source_provider="api-football",
        source_id="501",
        country="England",
    )


def build_basketball_fixture() -> NormalizedFixture:
    """Create a canonical Lakers-Celtics fixture for line-based fallback tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:15907925",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        competition="NBA",
        sport=SportName.BASKETBALL,
        kickoff=datetime(2026, 4, 4, 2, 0, tzinfo=UTC),
        source_provider="balldontlie",
        source_id="15907925",
        country="United States",
    )


def build_ranked_match(
    *,
    fixture: NormalizedFixture,
    market: MarketType | None,
    selection: str | None,
    recommended_odds: float | None = None,
) -> RankedMatch:
    """Create one ranked match recommendation for resolver inputs."""

    return RankedMatch(
        fixture_ref=fixture.get_fixture_ref(),
        sport=fixture.sport,
        competition=fixture.competition,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        composite_score=0.78,
        confidence=0.71,
        factors=ScoreFactorBreakdown(
            form=0.82,
            h2h=0.56,
            injury_impact=0.66,
            odds_value=0.74,
            context=0.7,
            venue=0.61,
            statistical=0.58,
        ),
        recommended_market=market,
        recommended_selection=selection,
        recommended_odds=recommended_odds,
        qualitative_summary="Resolver test recommendation.",
        rank=1,
    )


def build_sportybet_row(
    *,
    fixture: NormalizedFixture,
    market: MarketType,
    selection: str,
    provider_selection_name: str,
    odds: float,
    line: float | None = None,
    provider_market_id: int = 45,
) -> NormalizedOdds:
    """Create one SportyBet-like normalized odds row for resolver tests."""

    return NormalizedOdds(
        fixture_ref=fixture.get_fixture_ref(),
        market=market,
        selection=selection,
        odds=odds,
        provider="sportybet",
        provider_market_name="SportyBet Market",
        provider_selection_name=provider_selection_name,
        sportybet_available=True,
        line=line,
        provider_market_id=provider_market_id,
        raw_metadata={
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "requested_sportradar_id": fixture.sportradar_id,
        },
        last_updated=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
    )


def build_external_row(
    *,
    fixture: NormalizedFixture,
    market: MarketType,
    selection: str,
    provider_selection_name: str,
    odds: float,
    line: float | None = None,
) -> NormalizedOdds:
    """Create one external fallback odds row keyed by provider event metadata."""

    return NormalizedOdds(
        fixture_ref="the-odds-api:event-123",
        market=market,
        selection=selection,
        odds=odds,
        provider="Pinnacle",
        provider_market_name="spreads" if market == MarketType.POINT_SPREAD else "totals",
        provider_selection_name=provider_selection_name,
        sportybet_available=False,
        line=line,
        raw_metadata={
            "sport_key": "basketball_nba",
            "event_id": "event-123",
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "commence_time": fixture.kickoff.isoformat(),
        },
        last_updated=datetime(2026, 4, 4, 11, 55, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_resolve_returns_api_match_without_using_browser() -> None:
    """The resolver should stop at the API interceptor when it finds a match."""

    fixture = build_soccer_fixture()
    analysis = build_ranked_match(
        fixture=fixture,
        market=MarketType.MATCH_RESULT,
        selection="home",
        recommended_odds=1.86,
    )
    api_client = StubSportyBetAPIClient(
        rows=(
            build_sportybet_row(
                fixture=fixture,
                market=MarketType.MATCH_RESULT,
                selection="Home",
                provider_selection_name="Home",
                odds=1.86,
            ),
            build_sportybet_row(
                fixture=fixture,
                market=MarketType.MATCH_RESULT,
                selection="Draw",
                provider_selection_name="Draw",
                odds=3.5,
            ),
        )
    )
    browser_scraper = StubSportyBetBrowserScraper()
    resolver = MarketResolver(api_client=api_client, browser_scraper=browser_scraper)

    resolved = await resolver.resolve(fixture, analysis)

    assert resolved.resolution_source.value == "sportybet_api"
    assert resolved.selection == "home"
    assert resolved.sportybet_market_id == 45
    assert browser_scraper.scrape_calls == []


@pytest.mark.asyncio
async def test_resolve_falls_back_to_browser_when_api_rows_do_not_match() -> None:
    """Browser scraping should run when the API returns markets but not the right one."""

    fixture = build_soccer_fixture()
    analysis = build_ranked_match(
        fixture=fixture,
        market=MarketType.OVER_UNDER_25,
        selection="Over",
        recommended_odds=1.91,
    )
    api_client = StubSportyBetAPIClient(
        rows=(
            build_sportybet_row(
                fixture=fixture,
                market=MarketType.MATCH_RESULT,
                selection="Home",
                provider_selection_name="Home",
                odds=1.86,
            ),
        )
    )
    browser_scraper = StubSportyBetBrowserScraper(
        rows=(
            build_sportybet_row(
                fixture=fixture,
                market=MarketType.OVER_UNDER_25,
                selection="Over 2.5",
                provider_selection_name="Over 2.5",
                odds=1.91,
                line=2.5,
                provider_market_id=47,
            ),
        )
    )
    resolver = MarketResolver(api_client=api_client, browser_scraper=browser_scraper)

    resolved = await resolver.resolve(fixture, analysis)

    assert resolved.resolution_source.value == "sportybet_browser"
    assert resolved.selection == "over"
    assert len(browser_scraper.scrape_calls) == 1


@pytest.mark.asyncio
async def test_resolve_falls_back_to_external_odds_and_matches_fixture_metadata() -> None:
    """External odds should be matched by preserved fixture metadata when needed."""

    fixture = build_basketball_fixture()
    analysis = build_ranked_match(
        fixture=fixture,
        market=MarketType.POINT_SPREAD,
        selection="Los Angeles Lakers -4.5",
        recommended_odds=1.91,
    )
    api_client = StubSportyBetAPIClient(
        error=ProviderError("sportybet", "api outage")
    )
    browser_scraper = StubSportyBetBrowserScraper(
        error=ProviderError("sportybet", "browser outage")
    )
    resolver = MarketResolver(api_client=api_client, browser_scraper=browser_scraper)

    resolved = await resolver.resolve(
        fixture,
        analysis,
        external_odds=(
            build_external_row(
                fixture=fixture,
                market=MarketType.POINT_SPREAD,
                selection="Los Angeles Lakers",
                provider_selection_name="Los Angeles Lakers",
                odds=1.91,
                line=-4.5,
            ),
            build_external_row(
                fixture=fixture,
                market=MarketType.POINT_SPREAD,
                selection="Los Angeles Lakers",
                provider_selection_name="Los Angeles Lakers",
                odds=2.35,
                line=-8.5,
            ),
        ),
    )

    assert resolved.resolution_source.value == "external_odds"
    assert resolved.provider == "Pinnacle"
    assert resolved.selection == "home"
    assert resolved.line == pytest.approx(-4.5)


@pytest.mark.asyncio
async def test_resolve_raises_clear_error_when_no_sources_match() -> None:
    """Failures across every source should surface one actionable resolver error."""

    fixture = build_soccer_fixture()
    analysis = build_ranked_match(
        fixture=fixture,
        market=MarketType.BTTS,
        selection="Yes",
        recommended_odds=1.8,
    )
    resolver = MarketResolver(
        api_client=StubSportyBetAPIClient(
            error=ProviderError("sportybet", "api outage")
        ),
        browser_scraper=StubSportyBetBrowserScraper(
            error=ProviderError("sportybet", "browser outage")
        ),
    )

    with pytest.raises(ProviderError, match="Could not resolve a bookmaker market"):
        await resolver.resolve(fixture, analysis)


@pytest.mark.asyncio
async def test_resolve_requires_recommendation_metadata() -> None:
    """The resolver should fail fast when the score lacks a chosen market."""

    fixture = build_soccer_fixture()
    analysis = build_ranked_match(
        fixture=fixture,
        market=None,
        selection=None,
    )
    resolver = MarketResolver(
        api_client=StubSportyBetAPIClient(),
        browser_scraper=StubSportyBetBrowserScraper(),
    )

    with pytest.raises(ValueError, match="recommended_market and recommended_selection"):
        await resolver.resolve(fixture, analysis)
