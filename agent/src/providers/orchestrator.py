"""Provider orchestration layer for PuntLab's ingestion workflows.

Purpose: coordinate the current-state provider integrations behind one
canonical policy for SportyBet fixture/market/detail ingestion plus the
remaining structured data sources used for injuries, head-to-head history,
basketball stats, and news.
Scope: provider construction, competition-to-provider routing metadata,
cross-provider fixture matching, odds catalog generation, and explicit
diagnostics for partial coverage.
Dependencies: the concrete provider classes under `src.providers`, the odds
catalog helpers in `src.providers.odds_mapping`, runtime settings from
`src.config`, and the shared ingestion schemas under `src.schemas`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Final

from src.cache.client import RedisClient
from src.config import (
    SUPPORTED_COMPETITIONS,
    CompetitionConfig,
    SportName,
    get_settings,
)
from src.providers.api_football import APIFootballProvider
from src.providers.balldontlie import BallDontLieProvider
from src.providers.base import ProviderError, RateLimitedClient
from src.providers.odds_mapping import OddsMarketCatalog, build_odds_market_catalog
from src.providers.rss_feeds import RSSFeedProvider
from src.providers.tavily_search import TavilySearchProvider
from src.schemas.fixture_details import FixtureDetails
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, PlayerStats, TeamStats
from src.scrapers.sportybet_api import SportyBetAPIClient
from src.scrapers.sportybet_browser import SportyBetBrowserScraper
from src.scrapers.sportybet_fixture_probe import build_fixture_page_url
from src.scrapers.sportybet_fixture_stats import (
    SportyBetFixtureStatsResult,
    SportyBetFixtureStatsScraper,
    build_fixture_details_snapshot,
)
from src.scrapers.sportybet_today import (
    SportyBetTodayGamesCollector,
    build_fixtures_from_today_slate,
)

logger = logging.getLogger(__name__)

_SOCCER_DEFAULT_SEASON_START_MONTH: Final[int] = 7
_NBA_DEFAULT_SEASON_START_MONTH: Final[int] = 10
_FIXTURE_MATCH_TIME_WINDOW: Final[timedelta] = timedelta(hours=18)
_MATCH_NEWS_LOOKBACK_DAYS: Final[int] = 7
_DEFAULT_SPORTYBET_FIXTURE_DETAIL_RETRIES: Final[int] = 2
_DEFAULT_SPORTYBET_FIXTURE_DETAIL_RETRY_BACKOFF_SECONDS: Final[float] = 1.0
_TEAM_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "ac",
        "afc",
        "as",
        "athletic",
        "basketball",
        "bc",
        "bk",
        "ca",
        "cd",
        "cf",
        "city",
        "club",
        "county",
        "de",
        "e",
        "fc",
        "football",
        "gd",
        "olympique",
        "sc",
        "sd",
        "sporting",
        "the",
        "town",
        "ud",
        "united",
        "us",
    }
)
_TEAM_TOKEN_EXPANSIONS: Final[dict[str, tuple[str, ...]]] = {
    # Common abbreviation expansions that appear across basketball and soccer
    # feeds and would otherwise fail strict token overlap checks.
    "la": ("los", "angeles"),
    "ny": ("new", "york"),
    "st": ("saint",),
}


@dataclass(frozen=True, slots=True)
class CompetitionProviderRoute:
    """Provider-specific routing metadata for one supported competition.

    Inputs:
        Stable per-competition identifiers for upstream providers that cannot
        infer each other automatically.

    Outputs:
        An immutable route definition the orchestrator can use to execute the
        configured fallback chain deterministically.
    """

    competition_key: str
    api_football_league_id: int | None = None
    season_start_month: int = _SOCCER_DEFAULT_SEASON_START_MONTH


@dataclass(frozen=True, slots=True)
class OddsFetchResult:
    """Structured odds-ingestion result returned by `ProviderOrchestrator`.

    Inputs:
        The matched odds rows returned by upstream providers after cross-
        provider fixture reconciliation.

    Outputs:
        A lossless market catalog plus diagnostics about which providers were
        used and which fixtures still lack odds coverage.
    """

    catalog: OddsMarketCatalog
    matched_rows: tuple[NormalizedOdds, ...]
    unmatched_fixture_refs: tuple[str, ...]
    providers_attempted: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StatsFetchResult:
    """Structured statistics result returned by `ProviderOrchestrator`.

    Inputs:
        Aggregated team and player statistics fetched across one fixture slate.

    Outputs:
        Canonical team/player stat bundles plus diagnostics describing any
        partial coverage or provider gaps.
    """

    team_stats: tuple[TeamStats, ...]
    player_stats: tuple[PlayerStats, ...]
    providers_attempted: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FixtureDetailsFetchResult:
    """Structured SportyBet fixture-page details returned by the orchestrator.

    Inputs:
        SportyBet Today Games fixtures with Sportradar identifiers.

    Outputs:
        Compact per-fixture details for research plus diagnostics for missing
        fixture-page coverage.
    """

    fixture_details: tuple[FixtureDetails, ...]
    providers_attempted: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InjuryFetchResult:
    """Structured injury-ingestion result with qualitative fallback support.

    Inputs:
        Structured injury records when available plus fallback Tavily articles
        when a sports-data provider cannot supply the same coverage.

    Outputs:
        A canonical result that keeps structured injuries and news-only
        recovery paths separate for downstream scoring logic.
    """

    injuries: tuple[InjuryData, ...]
    supporting_articles: tuple[NewsArticle, ...]
    providers_attempted: tuple[str, ...]
    warnings: tuple[str, ...] = ()


_ROUTES: Final[dict[str, CompetitionProviderRoute]] = {
    "england_premier_league": CompetitionProviderRoute(
        competition_key="england_premier_league",
        api_football_league_id=39,
    ),
    "spain_la_liga": CompetitionProviderRoute(
        competition_key="spain_la_liga",
        api_football_league_id=140,
    ),
    "italy_serie_a": CompetitionProviderRoute(
        competition_key="italy_serie_a",
        api_football_league_id=135,
    ),
    "germany_bundesliga": CompetitionProviderRoute(
        competition_key="germany_bundesliga",
        api_football_league_id=78,
    ),
    "france_ligue_1": CompetitionProviderRoute(
        competition_key="france_ligue_1",
        api_football_league_id=61,
    ),
    "netherlands_eredivisie": CompetitionProviderRoute(
        competition_key="netherlands_eredivisie",
        api_football_league_id=88,
    ),
    "portugal_primeira_liga": CompetitionProviderRoute(
        competition_key="portugal_primeira_liga",
        api_football_league_id=94,
    ),
    "belgium_pro_league": CompetitionProviderRoute(
        competition_key="belgium_pro_league",
        api_football_league_id=144,
    ),
    "turkey_super_lig": CompetitionProviderRoute(
        competition_key="turkey_super_lig",
        api_football_league_id=203,
    ),
    "scotland_premiership": CompetitionProviderRoute(
        competition_key="scotland_premiership",
        api_football_league_id=179,
    ),
    "austria_bundesliga": CompetitionProviderRoute(
        competition_key="austria_bundesliga",
        api_football_league_id=218,
    ),
    "czech_republic_first_league": CompetitionProviderRoute(
        competition_key="czech_republic_first_league",
        api_football_league_id=345,
    ),
    "switzerland_super_league": CompetitionProviderRoute(
        competition_key="switzerland_super_league",
        api_football_league_id=207,
    ),
    "greece_super_league": CompetitionProviderRoute(
        competition_key="greece_super_league",
        api_football_league_id=197,
    ),
    "denmark_superliga": CompetitionProviderRoute(
        competition_key="denmark_superliga",
        api_football_league_id=119,
    ),
    "serbia_superliga": CompetitionProviderRoute(
        competition_key="serbia_superliga",
        api_football_league_id=286,
    ),
    "norway_eliteserien": CompetitionProviderRoute(
        competition_key="norway_eliteserien",
        api_football_league_id=103,
        season_start_month=1,
    ),
    "croatia_hnl": CompetitionProviderRoute(
        competition_key="croatia_hnl",
        api_football_league_id=210,
    ),
    "ukraine_premier_league": CompetitionProviderRoute(
        competition_key="ukraine_premier_league",
        api_football_league_id=333,
    ),
    "israel_premier_league": CompetitionProviderRoute(
        competition_key="israel_premier_league",
        api_football_league_id=242,
    ),
    "uefa_champions_league": CompetitionProviderRoute(
        competition_key="uefa_champions_league",
        api_football_league_id=2,
    ),
    "uefa_europa_league": CompetitionProviderRoute(
        competition_key="uefa_europa_league",
        api_football_league_id=3,
    ),
    "uefa_conference_league": CompetitionProviderRoute(
        competition_key="uefa_conference_league",
        api_football_league_id=848,
    ),
    "nba": CompetitionProviderRoute(
        competition_key="nba",
        season_start_month=_NBA_DEFAULT_SEASON_START_MONTH,
    ),
}


class ProviderOrchestrator:
    """Coordinate PuntLab's provider fallback chains behind one API.

    Inputs:
        Optional pre-built provider instances for tests or custom runtime
        wiring. When omitted, the orchestrator constructs the providers that
        can be configured from the current environment.

    Outputs:
        Canonical, route-aware fetch methods used by the ingestion stage to
        gather fixtures, odds, stats, injuries, H2H history, and news.
    """

    def __init__(
        self,
        *,
        api_football: APIFootballProvider | None = None,
        sportybet_today_collector: SportyBetTodayGamesCollector | None = None,
        sportybet_api_client: SportyBetAPIClient | None = None,
        sportybet_browser_scraper: SportyBetBrowserScraper | None = None,
        sportybet_fixture_stats_scraper: SportyBetFixtureStatsScraper | None = None,
        balldontlie: BallDontLieProvider | None = None,
        rss_feeds: RSSFeedProvider | None = None,
        tavily_search: TavilySearchProvider | None = None,
        client: RateLimitedClient | None = None,
        cache: RedisClient | None = None,
        sportybet_fixture_detail_retries: int = _DEFAULT_SPORTYBET_FIXTURE_DETAIL_RETRIES,
        sportybet_fixture_detail_retry_backoff_seconds: float = (
            _DEFAULT_SPORTYBET_FIXTURE_DETAIL_RETRY_BACKOFF_SECONDS
        ),
        sleep: Callable[[float], Awaitable[object]] | None = None,
    ) -> None:
        """Initialize provider instances and the route catalog.

        Args:
            api_football: Optional API-Football provider override.
            sportybet_today_collector: Optional SportyBet today-games fixture collector.
            sportybet_api_client: Optional SportyBet API odds/markets override.
            sportybet_browser_scraper: Optional SportyBet browser fallback override.
            sportybet_fixture_stats_scraper: Optional SportyBet fixture-page detail scraper.
            balldontlie: Optional BALLDONTLIE provider override.
            rss_feeds: Optional RSS provider override.
            tavily_search: Optional Tavily provider override.
            client: Optional shared `RateLimitedClient` used when providers
                must be constructed automatically.
            cache: Optional Redis cache wrapper used only when `client` is not
                supplied and automatic provider construction is needed.
            sportybet_fixture_detail_retries: Number of retry attempts for one
                SportyBet fixture-detail fetch after the initial attempt.
            sportybet_fixture_detail_retry_backoff_seconds: Delay between
                SportyBet fixture-detail retries.
            sleep: Optional async sleep function override used for tests.
        """

        if sportybet_fixture_detail_retries < 0:
            raise ValueError("sportybet_fixture_detail_retries must be zero or positive.")
        if sportybet_fixture_detail_retry_backoff_seconds < 0:
            raise ValueError(
                "sportybet_fixture_detail_retry_backoff_seconds must be zero or positive."
            )

        self._settings = get_settings()
        self._cache = cache
        self._client = client
        self._owns_cache = False
        self._owns_client = False
        self._sleep = sleep or asyncio.sleep
        self._sportybet_fixture_detail_retries = sportybet_fixture_detail_retries
        self._sportybet_fixture_detail_retry_backoff_seconds = (
            sportybet_fixture_detail_retry_backoff_seconds
        )

        if self._cache is None and any(
            provider is None
            for provider in (
                api_football,
                sportybet_api_client,
                sportybet_browser_scraper,
                balldontlie,
                rss_feeds,
                tavily_search,
            )
        ):
            self._cache = RedisClient()
            self._owns_cache = True

        if self._client is None and any(
            provider is None
            for provider in (
                api_football,
                balldontlie,
                rss_feeds,
                tavily_search,
            )
        ):
            if self._cache is None:
                self._cache = RedisClient()
                self._owns_cache = True
            self._client = RateLimitedClient(self._cache)
            self._owns_client = True

        self._api_football = api_football or self._build_api_football()
        self._sportybet_today_collector = (
            sportybet_today_collector or SportyBetTodayGamesCollector()
        )
        self._owns_sportybet_api_client = sportybet_api_client is None
        self._sportybet_api_client = sportybet_api_client or SportyBetAPIClient(self._cache)
        self._owns_sportybet_browser_scraper = sportybet_browser_scraper is None
        self._sportybet_browser_scraper = sportybet_browser_scraper or SportyBetBrowserScraper(
            self._cache
        )
        self._sportybet_fixture_stats_scraper = (
            sportybet_fixture_stats_scraper or SportyBetFixtureStatsScraper()
        )
        self._balldontlie = balldontlie or self._build_balldontlie()
        self._rss_feeds = rss_feeds or self._build_rss_feeds()
        self._tavily_search = tavily_search or self._build_tavily_search()

        self._routes = dict(_ROUTES)
        self._competition_by_key = {
            competition.key: competition for competition in SUPPORTED_COMPETITIONS
        }
        self._competition_identity_index = self._build_competition_identity_index()
        self._api_fixture_cache: dict[tuple[str, date, int], tuple[NormalizedFixture, ...]] = {}
        self._sportybet_fixture_stats_cache: dict[str, SportyBetFixtureStatsResult] = {}

    async def fetch_fixtures(
        self,
        *,
        run_date: date,
        competitions: tuple[CompetitionConfig, ...] | None = None,
        season_overrides: dict[str, int] | None = None,
    ) -> tuple[NormalizedFixture, ...]:
        """Fetch one day's fixtures from SportyBet's Today Games slate.

        Args:
            run_date: Date of the slate being analyzed.
            competitions: Optional competition subset. When omitted, the
                orchestrator keeps the full Today Games slate for the
                requested sports.
            season_overrides: Optional season-year overrides keyed by
                `competition.key`.

        Returns:
            A deduplicated tuple of normalized fixtures.
        """

        del season_overrides
        selected_competitions = competitions
        requested_sports = self._today_sports_for_competitions(selected_competitions)
        today_slate = await self._sportybet_today_collector.collect(sports=requested_sports)
        collected_fixtures = list(build_fixtures_from_today_slate(today_slate))
        if selected_competitions is not None:
            collected_fixtures = self._filter_fixtures_to_competitions(
                tuple(collected_fixtures),
                selected_competitions,
            )
        return self._deduplicate_fixtures(collected_fixtures)

    async def fetch_odds(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        markets_by_sport: dict[SportName, tuple[str, ...]] | None = None,
    ) -> OddsFetchResult:
        """Fetch canonical SportyBet odds for a fixture slate.

        Args:
            fixtures: Canonical fixtures for the current run.
            markets_by_sport: Ignored. SportyBet exposes one canonical market
                universe per fixture and no longer uses sport-level market-key
                requests in this orchestrator.

        Returns:
            A structured SportyBet odds result containing the full market
            catalog and unresolved fixture diagnostics.
        """

        del markets_by_sport
        return await self._fetch_sportybet_market_result(fixtures=fixtures)

    async def fetch_stats(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        season_overrides: dict[str, int] | None = None,
        include_player_stats: bool = True,
    ) -> StatsFetchResult:
        """Fetch team and player stats using the canonical current-state sources.

        Args:
            fixtures: Canonical fixtures for the current run.
            season_overrides: Optional season-year overrides keyed by
                `competition.key`.
            include_player_stats: Whether the orchestrator should attempt
                player-level lookups in addition to team-level summaries.

        Returns:
            A structured statistics result with diagnostics for partial
            provider coverage.
        """

        team_stats: list[TeamStats] = []
        player_stats: list[PlayerStats] = []
        warnings: list[str] = []
        providers_attempted: list[str] = []

        soccer_fixtures = tuple(
            fixture for fixture in fixtures if fixture.sport == SportName.SOCCER
        )
        basketball_fixtures = tuple(
            fixture for fixture in fixtures if fixture.sport == SportName.BASKETBALL
        )

        if soccer_fixtures:
            providers_attempted.append("sportybet_fixture_stats")
            for fixture in soccer_fixtures:
                fixture_ref = fixture.get_fixture_ref()
                try:
                    stats_result = await self._get_or_fetch_sportybet_fixture_stats_result(
                        fixture
                    )
                except ValueError as exc:
                    warning = f"SportyBet stats skipped for {fixture_ref}: {exc}"
                    warnings.append(warning)
                    continue
                except Exception as exc:
                    warning = f"SportyBet stats fetch failed for {fixture_ref}: {exc}"
                    logger.warning(warning)
                    warnings.append(warning)
                    continue

                derived_team_stats = self._bind_sportybet_team_stats_to_fixture(
                    fixture=fixture,
                    stats_result=stats_result,
                )
                if not derived_team_stats:
                    warning = (
                        f"SportyBet stats fetch returned no scoreable team snapshots for "
                        f"{fixture_ref}."
                    )
                    logger.warning(warning)
                    warnings.append(warning)
                    continue
                team_stats.extend(derived_team_stats)

        if basketball_fixtures and self._balldontlie is not None:
            providers_attempted.append(self._balldontlie.provider_name)
            run_date = min(
                fixture.kickoff.astimezone(UTC).date()
                for fixture in basketball_fixtures
            )
            season = self._resolve_season(
                competition=self._competition_by_key["nba"],
                run_date=run_date,
                season_overrides=season_overrides,
                route=self._routes["nba"],
            )
            try:
                team_ids = self._extract_numeric_team_ids(basketball_fixtures)
                if team_ids:
                    team_stats.extend(
                        await self._balldontlie.fetch_team_season_averages(
                            season=season,
                            team_ids=team_ids,
                        )
                    )
                    if include_player_stats:
                        player_stats.extend(
                            await self._balldontlie.fetch_players(team_ids=team_ids)
                        )
            except (ProviderError, ValueError) as exc:
                warning = f"BALLDONTLIE stats fetch failed for NBA slate: {exc}"
                logger.warning(warning)
                warnings.append(warning)

        return StatsFetchResult(
            team_stats=tuple(self._deduplicate_team_stats(team_stats)),
            player_stats=tuple(player_stats),
            providers_attempted=tuple(providers_attempted),
            warnings=tuple(warnings),
        )

    async def fetch_fixture_details(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> FixtureDetailsFetchResult:
        """Fetch SportyBet fixture-page context for today's football slate.

        Args:
            fixtures: Canonical fixtures for the current run.

        Returns:
            Compact SportyBet fixture details plus explicit diagnostics for any
            fixture that cannot be opened through the public fixture page.
        """

        football_fixtures = tuple(
            fixture for fixture in fixtures if fixture.sport == SportName.SOCCER
        )
        if not football_fixtures:
            return FixtureDetailsFetchResult(
                fixture_details=(),
                providers_attempted=(),
                warnings=(),
            )

        providers_attempted = ("sportybet_fixture_stats",)
        fixture_details: list[FixtureDetails] = []
        warnings: list[str] = []

        for fixture in football_fixtures:
            fixture_ref = fixture.get_fixture_ref()
            try:
                stats_result = await self._get_or_fetch_sportybet_fixture_stats_result(
                    fixture
                )
            except ValueError as exc:
                warnings.append(f"SportyBet fixture details skipped for {fixture_ref}: {exc}")
                continue
            except Exception as exc:
                warning = (
                    f"SportyBet fixture details fetch failed for {fixture_ref}: {exc}"
                )
                logger.warning(warning)
                warnings.append(warning)
                continue

            fixture_details.append(
                build_fixture_details_snapshot(
                    stats_result,
                    fixture_ref=fixture_ref,
                )
            )

        return FixtureDetailsFetchResult(
            fixture_details=tuple(fixture_details),
            providers_attempted=providers_attempted,
            warnings=tuple(warnings),
        )

    async def _get_or_fetch_sportybet_fixture_stats_result(
        self,
        fixture: NormalizedFixture,
    ) -> SportyBetFixtureStatsResult:
        """Return one cached or freshly fetched SportyBet fixture-stats result."""

        fixture_ref = fixture.get_fixture_ref()
        cached_result = self._sportybet_fixture_stats_cache.get(fixture_ref)
        if cached_result is not None:
            return cached_result

        fixture_url = self._build_sportybet_fixture_page_url(fixture)
        attempt_count = self._sportybet_fixture_detail_retries + 1
        last_error: Exception | None = None

        for attempt_index in range(attempt_count):
            try:
                result = await self._sportybet_fixture_stats_scraper.fetch_fixture_stats(
                    fixture_url=fixture_url,
                )
            except Exception as exc:
                last_error = exc
                if attempt_index >= self._sportybet_fixture_detail_retries:
                    break
                if self._sportybet_fixture_detail_retry_backoff_seconds > 0:
                    await self._sleep(
                        self._sportybet_fixture_detail_retry_backoff_seconds
                    )
                continue

            self._sportybet_fixture_stats_cache[fixture_ref] = result
            return result

        raise RuntimeError(
            "Could not fetch SportyBet fixture details after "
            f"{attempt_count} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _build_sportybet_fixture_page_url(fixture: NormalizedFixture) -> str:
        """Build the canonical SportyBet fixture page URL for one soccer fixture."""

        if fixture.sportradar_id is None:
            raise ValueError("missing Sportradar ID.")
        if fixture.country is None:
            raise ValueError("missing country.")
        return build_fixture_page_url(
            event_id=fixture.sportradar_id,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            country=fixture.country,
            competition=fixture.competition,
            sport="football",
        )

    def _bind_sportybet_team_stats_to_fixture(
        self,
        *,
        fixture: NormalizedFixture,
        stats_result: SportyBetFixtureStatsResult,
    ) -> tuple[TeamStats, ...]:
        """Normalize derived SportyBet team stats to the canonical fixture labels."""

        bound_team_stats: list[TeamStats] = []
        for team_stats in stats_result.team_stats:
            update: dict[str, object] = {"competition": fixture.competition}

            if (
                stats_result.home_team_uid is not None
                and team_stats.team_id == stats_result.home_team_uid
            ):
                update["team_name"] = fixture.home_team
                if fixture.home_team_id is not None:
                    update["team_id"] = fixture.home_team_id
            elif (
                stats_result.away_team_uid is not None
                and team_stats.team_id == stats_result.away_team_uid
            ):
                update["team_name"] = fixture.away_team
                if fixture.away_team_id is not None:
                    update["team_id"] = fixture.away_team_id
            elif self._team_similarity(team_stats.team_name, fixture.home_team) >= 0.93:
                update["team_name"] = fixture.home_team
                if fixture.home_team_id is not None:
                    update["team_id"] = fixture.home_team_id
            elif self._team_similarity(team_stats.team_name, fixture.away_team) >= 0.93:
                update["team_name"] = fixture.away_team
                if fixture.away_team_id is not None:
                    update["team_id"] = fixture.away_team_id
            else:
                continue

            bound_team_stats.append(team_stats.model_copy(update=update))

        return tuple(bound_team_stats)

    @staticmethod
    def _deduplicate_team_stats(team_stats: list[TeamStats]) -> list[TeamStats]:
        """Keep the richest team-stat snapshot per sport/team/competition key."""

        deduplicated: dict[tuple[SportName, str, str], TeamStats] = {}
        for stats in team_stats:
            team_key = (
                stats.sport,
                (stats.team_id or stats.team_name).casefold(),
                (stats.competition or "").casefold(),
            )
            current = deduplicated.get(team_key)
            if current is None:
                deduplicated[team_key] = stats
                continue
            if ProviderOrchestrator._team_stats_quality(
                stats
            ) >= ProviderOrchestrator._team_stats_quality(current):
                deduplicated[team_key] = stats
        return list(deduplicated.values())

    @staticmethod
    def _team_stats_quality(team_stats: TeamStats) -> float:
        """Estimate how useful one team-stat snapshot is for deterministic scoring."""

        quality = float(team_stats.matches_played)
        if team_stats.form:
            quality += min(len(team_stats.form), 10) * 0.5
        if team_stats.avg_goals_scored is not None:
            quality += 1.0
        if team_stats.avg_goals_conceded is not None:
            quality += 1.0
        if team_stats.position is not None:
            quality += 0.5
        if team_stats.advanced_metrics:
            quality += min(len(team_stats.advanced_metrics), 4) * 0.5
        return quality

    async def fetch_injuries(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        season_overrides: dict[str, int] | None = None,
    ) -> InjuryFetchResult:
        """Fetch structured injuries with Tavily article fallback.

        Args:
            fixtures: Canonical fixtures for the current run.
            season_overrides: Optional season-year overrides keyed by
                `competition.key`.

        Returns:
            Structured injuries when available plus Tavily fallback articles for
            fixtures that still need availability context.
        """

        injuries: list[InjuryData] = []
        supporting_articles: list[NewsArticle] = []
        warnings: list[str] = []
        providers_attempted: list[str] = []
        unresolved_fixtures = {fixture.get_fixture_ref(): fixture for fixture in fixtures}

        soccer_fixtures = tuple(
            fixture for fixture in fixtures if fixture.sport == SportName.SOCCER
        )
        soccer_competitions, unsupported_soccer_fixtures = (
            self._partition_fixtures_by_competition_key(soccer_fixtures)
        )
        unsupported_injury_warning = self._unsupported_competition_warning(
            capability="injury",
            fixtures=unsupported_soccer_fixtures,
        )
        if unsupported_injury_warning is not None:
            warnings.append(unsupported_injury_warning)
        if soccer_competitions and self._api_football is not None:
            providers_attempted.append(self._api_football.provider_name)
            for competition_key, competition_fixtures in soccer_competitions.items():
                competition = self._competition_by_key[competition_key]
                route = self._route_for_competition(competition)
                if route is None or route.api_football_league_id is None:
                    continue

                run_date = min(
                    fixture.kickoff.astimezone(UTC).date() for fixture in competition_fixtures
                )
                season = self._resolve_season(
                    competition=competition,
                    run_date=run_date,
                    season_overrides=season_overrides,
                    route=route,
                )
                try:
                    raw_injuries = await self._api_football.fetch_injuries(
                        league_id=route.api_football_league_id,
                        season=season,
                        report_date=run_date,
                    )
                except (ProviderError, ValueError) as exc:
                    warning = f"API-Football injury fetch failed for {competition_key}: {exc}"
                    logger.warning(warning)
                    warnings.append(warning)
                    continue

                remapped_injuries = await self._remap_api_football_injuries(
                    injuries=raw_injuries,
                    fixtures=competition_fixtures,
                    route=route,
                    run_date=run_date,
                    season=season,
                )
                injuries.extend(remapped_injuries)
                for injury in remapped_injuries:
                    unresolved_fixtures.pop(injury.fixture_ref, None)

        if unresolved_fixtures and self._tavily_search is not None:
            providers_attempted.append(self._tavily_search.provider_name)
            for fixture in unresolved_fixtures.values():
                try:
                    supporting_articles.extend(
                        await self._tavily_search.search_injury_updates(
                            fixture=fixture,
                            lookback_days=_MATCH_NEWS_LOOKBACK_DAYS,
                        )
                    )
                except ProviderError as exc:
                    warning = (
                        "Tavily injury fallback failed for "
                        f"{fixture.get_fixture_ref()}: {exc}"
                    )
                    logger.warning(warning)
                    warnings.append(warning)

        return InjuryFetchResult(
            injuries=tuple(injuries),
            supporting_articles=tuple(self._deduplicate_articles(supporting_articles)),
            providers_attempted=tuple(providers_attempted),
            warnings=tuple(warnings),
        )

    async def fetch_news(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[NewsArticle, ...]:
        """Fetch fixture-aware news with RSS-first fallback behavior.

        Args:
            fixtures: Canonical fixtures for the current run.

        Returns:
            A deduplicated tuple of relevant news articles.
        """

        if not fixtures:
            return ()

        articles: list[NewsArticle] = []
        fixtures_without_articles = {fixture.get_fixture_ref(): fixture for fixture in fixtures}

        if self._rss_feeds is not None:
            try:
                rss_articles = await self._rss_feeds.fetch_news(fixtures=fixtures)
            except ProviderError as exc:
                logger.warning("RSS news fetch failed: %s", exc)
            else:
                articles.extend(rss_articles)
                for article in rss_articles:
                    if article.fixture_ref is not None:
                        fixtures_without_articles.pop(article.fixture_ref, None)

        if fixtures_without_articles and self._tavily_search is not None:
            for fixture in fixtures_without_articles.values():
                try:
                    articles.extend(
                        await self._tavily_search.search_match_news(
                            fixture=fixture,
                            lookback_days=_MATCH_NEWS_LOOKBACK_DAYS,
                        )
                    )
                except ProviderError as exc:
                    logger.warning(
                        "Tavily match-news fallback failed for fixture_ref=%s: %s",
                        fixture.get_fixture_ref(),
                        exc,
                    )

        return tuple(self._deduplicate_articles(articles))

    async def fetch_head_to_head(
        self,
        *,
        fixture: NormalizedFixture,
        last: int = 5,
        season_override: int | None = None,
    ) -> tuple[NormalizedFixture, ...]:
        """Fetch recent head-to-head history for one soccer fixture.

        Args:
            fixture: Canonical fixture used to locate the provider team IDs.
            last: Number of recent meetings to request.
            season_override: Optional explicit season year.

        Returns:
            A tuple of normalized historical fixtures.

        Raises:
            ProviderError: If API-Football is unavailable or the fixture cannot
                be matched to canonical API-Football team identifiers.
        """

        if fixture.sport != SportName.SOCCER:
            raise ProviderError(
                "api-football",
                "Head-to-head history is only supported for soccer fixtures.",
            )
        if self._api_football is None:
            raise ProviderError(
                "api-football",
                "API-Football is not configured, so head-to-head history is unavailable.",
            )

        api_fixture = await self._resolve_api_football_fixture_for_fixture(fixture)
        if (
            api_fixture is None
            or api_fixture.home_team_id is None
            or api_fixture.away_team_id is None
        ):
            raise ProviderError(
                "api-football",
                (
                    "Could not resolve API-Football team IDs for fixture "
                    f"{fixture.get_fixture_ref()}."
                ),
            )

        competition = self._competition_for_fixture(fixture)
        route = self._route_for_competition(competition) if competition is not None else None
        season = season_override or self._resolve_season(
            competition=competition or self._competition_by_key["england_premier_league"],
            run_date=fixture.kickoff.astimezone(UTC).date(),
            season_overrides=None,
            route=route,
        )

        try:
            h2h_fixtures = await self._api_football.fetch_head_to_head(
                home_team_id=int(api_fixture.home_team_id),
                away_team_id=int(api_fixture.away_team_id),
                last=last,
                league_id=route.api_football_league_id if route is not None else None,
                season=season,
            )
        except (ProviderError, ValueError) as exc:
            raise ProviderError("api-football", f"Head-to-head fetch failed: {exc}") from exc

        return tuple(h2h_fixtures)

    async def fetch_markets(self, *, fixtures: tuple[NormalizedFixture, ...]) -> OddsFetchResult:
        """Fetch canonical SportyBet markets for the supplied fixture slate.

        Args:
            fixtures: Canonical fixtures whose SportyBet markets are requested.
        """

        return await self._fetch_sportybet_market_result(fixtures=fixtures)

    async def aclose(self) -> None:
        """Close the owned provider helpers constructed by the orchestrator."""

        if self._owns_sportybet_browser_scraper:
            await self._sportybet_browser_scraper.aclose()
        if self._owns_sportybet_api_client:
            await self._sportybet_api_client.aclose()
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._cache is not None and self._owns_cache:
            await self._cache.close()

    def _build_api_football(self) -> APIFootballProvider | None:
        """Construct API-Football when the environment is configured for it."""

        if self._client is None or self._settings.data_providers.api_football_key is None:
            return None
        return APIFootballProvider(self._client)

    def _build_balldontlie(self) -> BallDontLieProvider | None:
        """Construct BALLDONTLIE when a key is available."""

        if self._client is None or self._settings.data_providers.balldontlie_api_key is None:
            return None
        return BallDontLieProvider(self._client)

    def _build_rss_feeds(self) -> RSSFeedProvider | None:
        """Construct the RSS provider when the shared client exists."""

        if self._client is None:
            return None
        return RSSFeedProvider(self._client)

    def _build_tavily_search(self) -> TavilySearchProvider | None:
        """Construct Tavily when the environment is configured for it."""

        if self._client is None or self._settings.data_providers.tavily_api_key is None:
            return None
        return TavilySearchProvider(self._client)

    @staticmethod
    def _apply_competition_identity(
        fixture: NormalizedFixture,
        competition: CompetitionConfig,
    ) -> NormalizedFixture:
        """Stamp canonical competition identity onto provider fixture rows.

        Football providers may expose alternate competition labels (for example
        `Primera Division` instead of `La Liga`). Normalizing to the configured
        competition identity keeps downstream routing deterministic.
        """

        return fixture.model_copy(
            update={
                "competition": competition.name,
                "country": competition.country,
                "league": competition.slug,
            }
        )

    async def _fetch_sportybet_market_result(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> OddsFetchResult:
        """Fetch SportyBet markets for every fixture and build one catalog."""

        if not fixtures:
            empty_catalog = build_odds_market_catalog(())
            return OddsFetchResult(
                catalog=empty_catalog,
                matched_rows=(),
                unmatched_fixture_refs=(),
                providers_attempted=(),
            )

        matched_rows: list[NormalizedOdds] = []
        warnings: list[str] = []
        providers_attempted: list[str] = []
        unmatched_fixture_refs: list[str] = []
        fixture_map = {fixture.get_fixture_ref(): fixture for fixture in fixtures}

        for fixture in fixtures:
            try:
                fixture_rows = await self._fetch_sportybet_rows_for_fixture(
                    fixture=fixture,
                    providers_attempted=providers_attempted,
                )
            except ProviderError as exc:
                warning = (
                    "SportyBet odds fetch failed for "
                    f"{fixture.get_fixture_ref()}: {exc}"
                )
                logger.warning(warning)
                warnings.append(warning)
                unmatched_fixture_refs.append(fixture.get_fixture_ref())
                continue

            matched_rows.extend(fixture_rows)

        sport_by_fixture = {
            fixture_ref: fixture.sport for fixture_ref, fixture in fixture_map.items()
        }
        catalog = build_odds_market_catalog(
            tuple(matched_rows),
            sport_by_fixture=sport_by_fixture,
        )
        return OddsFetchResult(
            catalog=catalog,
            matched_rows=tuple(catalog.all_rows()),
            unmatched_fixture_refs=tuple(sorted(unmatched_fixture_refs)),
            providers_attempted=tuple(providers_attempted),
            warnings=tuple(warnings),
        )

    async def _fetch_sportybet_rows_for_fixture(
        self,
        *,
        fixture: NormalizedFixture,
        providers_attempted: list[str],
    ) -> tuple[NormalizedOdds, ...]:
        """Fetch one fixture's SportyBet rows using API-first browser fallback."""

        if fixture.sportradar_id is None:
            raise ProviderError(
                "sportybet",
                (
                    "fixture.sportradar_id is required for canonical SportyBet odds "
                    f"fetching ({fixture.get_fixture_ref()})."
                ),
            )

        self._append_provider_attempt(providers_attempted, "sportybet_api")
        api_error: ProviderError | None = None
        try:
            return await self._sportybet_api_client.fetch_markets(
                fixture.sportradar_id,
                fixture=fixture,
            )
        except ProviderError as exc:
            api_error = exc

        self._append_provider_attempt(providers_attempted, "sportybet_browser")
        sportybet_url = self._sportybet_api_client.build_sportybet_url(fixture)
        try:
            return await self._sportybet_browser_scraper.scrape_markets(
                sportybet_url,
                fixture=fixture,
            )
        except ProviderError as browser_error:
            raise ProviderError(
                "sportybet",
                (
                    f"SportyBet API failed: {api_error}. "
                    f"SportyBet browser fallback failed: {browser_error}"
                ),
            ) from browser_error

    @staticmethod
    def _append_provider_attempt(providers_attempted: list[str], provider_name: str) -> None:
        """Record one provider attempt once while preserving first-seen order."""

        if provider_name not in providers_attempted:
            providers_attempted.append(provider_name)

    async def _remap_api_football_injuries(
        self,
        *,
        injuries: list[InjuryData],
        fixtures: tuple[NormalizedFixture, ...],
        route: CompetitionProviderRoute,
        run_date: date,
        season: int,
    ) -> tuple[InjuryData, ...]:
        """Rewrite API-Football injury fixture refs to the slate's canonical refs."""

        api_fixtures = await self._get_or_fetch_api_fixtures_for_route(
            route=route,
            run_date=run_date,
            season=season,
        )
        api_fixture_by_ref = {fixture.get_fixture_ref(): fixture for fixture in api_fixtures}
        fixture_match_map = self._match_provider_fixtures_to_canonical(
            provider_fixtures=api_fixtures,
            canonical_fixtures=fixtures,
        )

        remapped: list[InjuryData] = []
        for injury in injuries:
            provider_fixture = api_fixture_by_ref.get(injury.fixture_ref)
            if provider_fixture is None:
                continue
            matched_fixture = fixture_match_map.get(provider_fixture.get_fixture_ref())
            if matched_fixture is None:
                continue
            remapped.append(
                injury.model_copy(update={"fixture_ref": matched_fixture.get_fixture_ref()})
            )
        return tuple(remapped)

    async def _resolve_api_football_fixture_for_fixture(
        self,
        fixture: NormalizedFixture,
    ) -> NormalizedFixture | None:
        """Resolve one canonical fixture back to an API-Football fixture."""

        if fixture.source_provider == "api-football":
            return fixture

        competition = self._competition_for_fixture(fixture)
        if competition is None:
            return None
        route = self._route_for_competition(competition)
        if route is None or route.api_football_league_id is None:
            return None

        run_date = fixture.kickoff.astimezone(UTC).date()
        season = self._resolve_season(
            competition=competition,
            run_date=run_date,
            season_overrides=None,
            route=route,
        )
        api_fixtures = await self._get_or_fetch_api_fixtures_for_route(
            route=route,
            run_date=run_date,
            season=season,
        )
        best_match = self._find_best_fixture_match(
            reference_home_team=fixture.home_team,
            reference_away_team=fixture.away_team,
            reference_kickoff=fixture.kickoff,
            candidates=api_fixtures,
            sport=fixture.sport,
            competition=fixture.competition,
            country=fixture.country,
        )
        return best_match

    async def _get_or_fetch_api_fixtures_for_route(
        self,
        *,
        route: CompetitionProviderRoute,
        run_date: date,
        season: int,
    ) -> tuple[NormalizedFixture, ...]:
        """Return cached API-Football fixtures for one route/date/season tuple."""

        cache_key = (route.competition_key, run_date, season)
        if cache_key in self._api_fixture_cache:
            return self._api_fixture_cache[cache_key]

        if self._api_football is None or route.api_football_league_id is None:
            return ()
        fixtures = tuple(
            await self._api_football.fetch_fixtures_by_date(
                run_date=run_date,
                league_id=route.api_football_league_id,
                season=season,
            )
        )
        self._api_fixture_cache[cache_key] = fixtures
        return fixtures

    def _match_provider_fixtures_to_canonical(
        self,
        *,
        provider_fixtures: tuple[NormalizedFixture, ...],
        canonical_fixtures: tuple[NormalizedFixture, ...],
    ) -> dict[str, NormalizedFixture]:
        """Build a provider-fixture to canonical-fixture mapping."""

        matches: dict[str, NormalizedFixture] = {}
        for provider_fixture in provider_fixtures:
            best_match = self._find_best_fixture_match(
                reference_home_team=provider_fixture.home_team,
                reference_away_team=provider_fixture.away_team,
                reference_kickoff=provider_fixture.kickoff,
                candidates=canonical_fixtures,
                sport=provider_fixture.sport,
                competition=provider_fixture.competition,
                country=provider_fixture.country,
            )
            if best_match is not None:
                matches[provider_fixture.get_fixture_ref()] = best_match
        return matches

    def _find_best_fixture_match(
        self,
        *,
        reference_home_team: str | None,
        reference_away_team: str | None,
        reference_kickoff: datetime | None,
        candidates: tuple[NormalizedFixture, ...],
        sport: SportName | None,
        competition: str | None,
        country: str | None,
    ) -> NormalizedFixture | None:
        """Select the best fixture match across provider datasets.

        Team-name normalization and kickoff proximity provide the primary match
        score. Competition and country are used as soft tie-breakers rather
        than hard requirements because upstream feeds sometimes omit or rename
        them differently.
        """

        if reference_home_team is None or reference_away_team is None:
            return None

        best_fixture: NormalizedFixture | None = None
        best_score = 0.0
        for candidate in candidates:
            if sport is not None and candidate.sport != sport:
                continue
            time_penalty = self._kickoff_penalty(reference_kickoff, candidate.kickoff)
            if time_penalty is None:
                continue
            home_score = self._team_similarity(reference_home_team, candidate.home_team)
            away_score = self._team_similarity(reference_away_team, candidate.away_team)
            if not self._passes_team_match_gate(
                home_score=home_score,
                away_score=away_score,
                time_penalty=time_penalty,
            ):
                continue

            score = (home_score + away_score) - time_penalty
            if competition is not None and (
                self._normalize_phrase(competition)
                == self._normalize_phrase(candidate.competition)
            ):
                score += 0.05
            if country is not None and candidate.country is not None:
                if self._normalize_phrase(country) == self._normalize_phrase(candidate.country):
                    score += 0.03

            if score > best_score:
                best_score = score
                best_fixture = candidate

        return best_fixture

    @staticmethod
    def _passes_team_match_gate(
        *,
        home_score: float,
        away_score: float,
        time_penalty: float,
    ) -> bool:
        """Return whether one candidate pair is strong enough to accept.

        Inputs:
            home_score: Similarity score between reference and candidate home
                teams.
            away_score: Similarity score between reference and candidate away
                teams.
            time_penalty: Normalized kickoff mismatch penalty where `0.0` is an
                exact time match and `1.0` is the maximum accepted distance.

        Outputs:
            `True` when the pair has strong bilateral similarity or one very
            strong team-name match plus a near-identical kickoff timestamp.
        """

        if home_score >= 0.72 and away_score >= 0.72:
            return True
        if time_penalty <= 0.05 and home_score >= 0.60 and away_score >= 0.60:
            return True
        if time_penalty <= 0.08 and max(home_score, away_score) >= 0.82 and min(
            home_score, away_score
        ) >= 0.45:
            return True
        return False

    def _build_competition_identity_index(
        self,
    ) -> dict[tuple[str, str, str | None], CompetitionConfig]:
        """Build a lookup index for mapping fixtures back to configured competitions."""

        index: dict[tuple[str, str, str | None], CompetitionConfig] = {}
        for competition in SUPPORTED_COMPETITIONS:
            index[
                (
                    competition.sport.value,
                    self._normalize_phrase(competition.name),
                    self._normalize_optional_phrase(competition.country),
                )
            ] = competition
        return index

    def _competition_for_fixture(self, fixture: NormalizedFixture) -> CompetitionConfig | None:
        """Resolve the configured competition metadata for one normalized fixture."""

        lookup_keys = [
            (
                fixture.sport.value,
                self._normalize_phrase(fixture.competition),
                self._normalize_optional_phrase(fixture.country),
            ),
            (
                fixture.sport.value,
                self._normalize_phrase(fixture.competition),
                None,
            ),
        ]
        for key in lookup_keys:
            competition = self._competition_identity_index.get(key)
            if competition is not None:
                return competition
        return None

    def _route_for_fixture(self, fixture: NormalizedFixture) -> CompetitionProviderRoute | None:
        """Resolve the provider route for one normalized fixture."""

        competition = self._competition_for_fixture(fixture)
        if competition is None:
            return None
        return self._route_for_competition(competition)

    def _route_for_competition(
        self,
        competition: CompetitionConfig,
    ) -> CompetitionProviderRoute | None:
        """Return provider routing metadata for one configured competition."""

        return self._routes.get(competition.key)

    def _resolve_season(
        self,
        *,
        competition: CompetitionConfig,
        run_date: date,
        season_overrides: dict[str, int] | None,
        route: CompetitionProviderRoute | None,
    ) -> int:
        """Resolve the provider season year for one run date and competition."""

        if season_overrides is not None and competition.key in season_overrides:
            return season_overrides[competition.key]

        season_start_month = (
            route.season_start_month
            if route is not None
            else (
                _NBA_DEFAULT_SEASON_START_MONTH
                if competition.sport == SportName.BASKETBALL
                else _SOCCER_DEFAULT_SEASON_START_MONTH
            )
        )
        return run_date.year if run_date.month >= season_start_month else run_date.year - 1

    def _partition_fixtures_by_competition_key(
        self,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[dict[str, tuple[NormalizedFixture, ...]], tuple[NormalizedFixture, ...]]:
        """Group supported fixtures by competition key and return unsupported rows."""

        grouped: dict[str, list[NormalizedFixture]] = {}
        unsupported: list[NormalizedFixture] = []
        for fixture in fixtures:
            competition = self._competition_for_fixture(fixture)
            if competition is None:
                unsupported.append(fixture)
                continue
            grouped.setdefault(competition.key, []).append(fixture)
        return {key: tuple(values) for key, values in grouped.items()}, tuple(unsupported)

    @staticmethod
    def _today_sports_for_competitions(
        competitions: tuple[CompetitionConfig, ...] | None,
    ) -> tuple[str, ...]:
        """Resolve the SportyBet today-games sport selectors for fixture collection."""

        if competitions is None:
            return ("football",)

        ordered_sports: list[str] = []
        seen: set[str] = set()
        for competition in competitions:
            sport_selector = "football" if competition.sport == SportName.SOCCER else "basketball"
            if sport_selector in seen:
                continue
            seen.add(sport_selector)
            ordered_sports.append(sport_selector)
        return tuple(ordered_sports) or ("football",)

    def _filter_fixtures_to_competitions(
        self,
        fixtures: tuple[NormalizedFixture, ...],
        competitions: tuple[CompetitionConfig, ...],
    ) -> list[NormalizedFixture]:
        """Keep only fixtures whose competition identity matches the requested subset."""

        allowed_keys = {competition.key for competition in competitions}
        filtered_fixtures: list[NormalizedFixture] = []
        for fixture in fixtures:
            competition = self._competition_for_fixture(fixture)
            if competition is None:
                continue
            if competition.key not in allowed_keys:
                continue
            filtered_fixtures.append(fixture)
        return filtered_fixtures

    def _unsupported_competition_warning(
        self,
        *,
        capability: str,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> str | None:
        """Describe unsupported today-slate competitions for operator diagnostics."""

        if not fixtures:
            return None

        competition_labels: list[str] = []
        seen_labels: set[tuple[str, str | None]] = set()
        for fixture in fixtures:
            normalized_label = (
                self._normalize_phrase(fixture.competition),
                self._normalize_optional_phrase(fixture.country),
            )
            if normalized_label in seen_labels:
                continue
            seen_labels.add(normalized_label)
            if fixture.country is not None:
                competition_labels.append(f"{fixture.competition} ({fixture.country})")
            else:
                competition_labels.append(fixture.competition)

        preview_labels = competition_labels[:6]
        if len(competition_labels) > len(preview_labels):
            preview_labels.append(f"+{len(competition_labels) - len(preview_labels)} more")

        fixture_noun = "fixture" if len(fixtures) == 1 else "fixtures"
        return (
            f"Structured {capability} coverage is unavailable for {len(fixtures)} "
            f"today-slate {fixture_noun} from competitions without provider routing: "
            f"{', '.join(preview_labels)}."
        )

    def _deduplicate_fixtures(
        self,
        fixtures: list[NormalizedFixture],
    ) -> tuple[NormalizedFixture, ...]:
        """Deduplicate fixtures while preserving the first-seen canonical record."""

        deduplicated: list[NormalizedFixture] = []
        seen_keys: set[tuple[str, str, str, str, datetime]] = set()
        for fixture in fixtures:
            dedupe_key = (
                fixture.sport.value,
                self._normalize_phrase(fixture.competition),
                self._normalize_phrase(fixture.home_team),
                self._normalize_phrase(fixture.away_team),
                fixture.kickoff.astimezone(UTC),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduplicated.append(fixture)
        return tuple(deduplicated)

    def _deduplicate_articles(
        self,
        articles: list[NewsArticle],
    ) -> list[NewsArticle]:
        """Deduplicate articles by URL while preserving the highest-signal row."""

        deduplicated: dict[str, NewsArticle] = {}
        for article in sorted(
            articles,
            key=lambda item: ((item.relevance_score or 0.0), item.published_at),
            reverse=True,
        ):
            deduplicated.setdefault(str(article.url), article)
        return list(deduplicated.values())

    def _extract_numeric_team_ids(
        self,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[int, ...]:
        """Extract unique numeric team IDs from a fixture slate."""

        team_ids: list[int] = []
        seen: set[int] = set()
        for fixture in fixtures:
            for team_id in (fixture.home_team_id, fixture.away_team_id):
                if team_id is None:
                    continue
                try:
                    numeric_team_id = int(team_id)
                except ValueError:
                    continue
                if numeric_team_id in seen:
                    continue
                seen.add(numeric_team_id)
                team_ids.append(numeric_team_id)
        return tuple(team_ids)

    @staticmethod
    def _kickoff_penalty(
        reference_kickoff: datetime | None,
        candidate_kickoff: datetime,
    ) -> float | None:
        """Convert kickoff mismatch into a soft penalty, or reject distant matches."""

        if reference_kickoff is None:
            return 0.0
        delta = abs(
            reference_kickoff.astimezone(UTC) - candidate_kickoff.astimezone(UTC)
        )
        if delta > _FIXTURE_MATCH_TIME_WINDOW:
            return None
        return delta.total_seconds() / _FIXTURE_MATCH_TIME_WINDOW.total_seconds()

    @classmethod
    def _team_similarity(cls, left: str, right: str) -> float:
        """Estimate similarity between two provider team labels."""

        normalized_left = cls._normalize_phrase(left)
        normalized_right = cls._normalize_phrase(right)
        if normalized_left == normalized_right:
            return 1.0

        tokenized_left = cls._normalize_team_tokens(left)
        tokenized_right = cls._normalize_team_tokens(right)
        if tokenized_left and tokenized_left == tokenized_right:
            return 0.97

        overlap = len(tokenized_left & tokenized_right)
        denominator = max(len(tokenized_left), len(tokenized_right), 1)
        token_score = overlap / denominator
        phrase_score = SequenceMatcher(None, normalized_left, normalized_right).ratio()
        return max(token_score, phrase_score)

    @classmethod
    def _normalize_team_tokens(cls, value: str) -> set[str]:
        """Normalize one team label into significant comparison tokens."""

        tokens = {
            token
            for token in cls._normalize_phrase(value).split()
            if token not in _TEAM_STOP_WORDS
        }
        expanded_tokens = set(tokens)
        for token in tokens:
            expansion = _TEAM_TOKEN_EXPANSIONS.get(token)
            if expansion is not None:
                expanded_tokens.update(expansion)
        return expanded_tokens

    @staticmethod
    def _normalize_phrase(value: str) -> str:
        """Normalize free-text labels for deterministic matching."""

        cleaned = []
        for character in value.lower():
            cleaned.append(character if character.isalnum() else " ")
        return " ".join("".join(cleaned).split())

    @classmethod
    def _normalize_optional_phrase(cls, value: str | None) -> str | None:
        """Normalize optional text while preserving `None`."""

        if value is None:
            return None
        normalized = cls._normalize_phrase(value)
        return normalized or None


__all__ = [
    "CompetitionProviderRoute",
    "FixtureDetailsFetchResult",
    "InjuryFetchResult",
    "OddsFetchResult",
    "ProviderOrchestrator",
    "StatsFetchResult",
]
