"""Provider orchestration layer for PuntLab's ingestion workflows.

Purpose: coordinate the concrete provider integrations behind one canonical
fallback policy for fixtures, odds, stats, injuries, head-to-head history, and
news ingestion.
Scope: provider construction, competition-to-provider routing metadata,
cross-provider fixture matching, odds catalog generation, and explicit
diagnostics when a planned fallback such as SportyBet is not yet implemented.
Dependencies: the concrete provider classes under `src.providers`, the odds
catalog helpers in `src.providers.odds_mapping`, runtime settings from
`src.config`, and the shared ingestion schemas under `src.schemas`.
"""

from __future__ import annotations

import logging
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
from src.providers.football_data import FootballDataProvider
from src.providers.odds_mapping import OddsMarketCatalog, build_odds_market_catalog
from src.providers.rss_feeds import RSSFeedProvider
from src.providers.tavily_search import TavilySearchProvider
from src.providers.the_odds_api import TheOddsAPIProvider
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, PlayerStats, TeamStats

logger = logging.getLogger(__name__)

_SOCCER_DEFAULT_SEASON_START_MONTH: Final[int] = 7
_NBA_DEFAULT_SEASON_START_MONTH: Final[int] = 10
_FIXTURE_MATCH_TIME_WINDOW: Final[timedelta] = timedelta(hours=18)
_MATCH_NEWS_LOOKBACK_DAYS: Final[int] = 7
_SOCCER_ODDS_MARKETS: Final[tuple[str, ...]] = (
    "h2h",
    "totals",
    "spreads",
    "btts",
    "draw_no_bet",
    "double_chance",
    "correct_score",
)
_NBA_ODDS_MARKETS: Final[tuple[str, ...]] = ("h2h", "spreads", "totals")
_TEAM_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "afc",
        "athletic",
        "basketball",
        "bc",
        "bk",
        "cf",
        "city",
        "club",
        "county",
        "fc",
        "football",
        "sc",
        "sporting",
        "the",
        "town",
        "united",
    }
)


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
    football_data_code: str | None = None
    the_odds_sport_key: str | None = None
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
        football_data_code="PL",
        the_odds_sport_key="soccer_epl",
    ),
    "spain_la_liga": CompetitionProviderRoute(
        competition_key="spain_la_liga",
        api_football_league_id=140,
        football_data_code="PD",
        the_odds_sport_key="soccer_spain_la_liga",
    ),
    "italy_serie_a": CompetitionProviderRoute(
        competition_key="italy_serie_a",
        api_football_league_id=135,
        football_data_code="SA",
        the_odds_sport_key="soccer_italy_serie_a",
    ),
    "germany_bundesliga": CompetitionProviderRoute(
        competition_key="germany_bundesliga",
        api_football_league_id=78,
        football_data_code="BL1",
        the_odds_sport_key="soccer_germany_bundesliga",
    ),
    "france_ligue_1": CompetitionProviderRoute(
        competition_key="france_ligue_1",
        api_football_league_id=61,
        football_data_code="FL1",
        the_odds_sport_key="soccer_france_ligue_one",
    ),
    "netherlands_eredivisie": CompetitionProviderRoute(
        competition_key="netherlands_eredivisie",
        api_football_league_id=88,
        football_data_code="DED",
        the_odds_sport_key="soccer_netherlands_eredivisie",
    ),
    "portugal_primeira_liga": CompetitionProviderRoute(
        competition_key="portugal_primeira_liga",
        api_football_league_id=94,
        football_data_code="PPL",
        the_odds_sport_key="soccer_portugal_primeira_liga",
    ),
    "belgium_pro_league": CompetitionProviderRoute(
        competition_key="belgium_pro_league",
        api_football_league_id=144,
        the_odds_sport_key="soccer_belgium_first_div",
    ),
    "turkey_super_lig": CompetitionProviderRoute(
        competition_key="turkey_super_lig",
        api_football_league_id=203,
        the_odds_sport_key="soccer_turkey_super_league",
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
        the_odds_sport_key="soccer_norway_eliteserien",
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
        football_data_code="CL",
        the_odds_sport_key="soccer_uefa_champs_league",
    ),
    "uefa_europa_league": CompetitionProviderRoute(
        competition_key="uefa_europa_league",
        api_football_league_id=3,
        football_data_code="EL",
        the_odds_sport_key="soccer_uefa_europa_league",
    ),
    "uefa_conference_league": CompetitionProviderRoute(
        competition_key="uefa_conference_league",
        api_football_league_id=848,
        the_odds_sport_key="soccer_uefa_europa_conference_league",
    ),
    "nba": CompetitionProviderRoute(
        competition_key="nba",
        the_odds_sport_key="basketball_nba",
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
        football_data: FootballDataProvider | None = None,
        the_odds_api: TheOddsAPIProvider | None = None,
        balldontlie: BallDontLieProvider | None = None,
        rss_feeds: RSSFeedProvider | None = None,
        tavily_search: TavilySearchProvider | None = None,
        client: RateLimitedClient | None = None,
        cache: RedisClient | None = None,
    ) -> None:
        """Initialize provider instances and the route catalog.

        Args:
            api_football: Optional API-Football provider override.
            football_data: Optional Football-Data.org provider override.
            the_odds_api: Optional The Odds API provider override.
            balldontlie: Optional BALLDONTLIE provider override.
            rss_feeds: Optional RSS provider override.
            tavily_search: Optional Tavily provider override.
            client: Optional shared `RateLimitedClient` used when providers
                must be constructed automatically.
            cache: Optional Redis cache wrapper used only when `client` is not
                supplied and automatic provider construction is needed.
        """

        self._settings = get_settings()
        self._cache = cache
        self._client = client
        self._owns_client = False

        if self._client is None and any(
            provider is None
            for provider in (
                api_football,
                football_data,
                the_odds_api,
                balldontlie,
                rss_feeds,
                tavily_search,
            )
        ):
            self._cache = self._cache or RedisClient()
            self._client = RateLimitedClient(self._cache)
            self._owns_client = True

        self._api_football = api_football or self._build_api_football()
        self._football_data = football_data or self._build_football_data()
        self._the_odds_api = the_odds_api or self._build_the_odds_api()
        self._balldontlie = balldontlie or self._build_balldontlie()
        self._rss_feeds = rss_feeds or self._build_rss_feeds()
        self._tavily_search = tavily_search or self._build_tavily_search()

        self._routes = dict(_ROUTES)
        self._competition_by_key = {
            competition.key: competition for competition in SUPPORTED_COMPETITIONS
        }
        self._competition_identity_index = self._build_competition_identity_index()
        self._api_fixture_cache: dict[tuple[str, date, int], tuple[NormalizedFixture, ...]] = {}

    async def fetch_fixtures(
        self,
        *,
        run_date: date,
        competitions: tuple[CompetitionConfig, ...] | None = None,
        season_overrides: dict[str, int] | None = None,
    ) -> tuple[NormalizedFixture, ...]:
        """Fetch one day's fixtures using the configured provider chain.

        Args:
            run_date: Date of the slate being analyzed.
            competitions: Optional competition subset. When omitted, the
                orchestrator uses the canonical supported competition catalog.
            season_overrides: Optional season-year overrides keyed by
                `competition.key`.

        Returns:
            A deduplicated tuple of normalized fixtures.
        """

        selected_competitions = competitions or SUPPORTED_COMPETITIONS
        collected_fixtures: list[NormalizedFixture] = []

        for competition in selected_competitions:
            route = self._route_for_competition(competition)
            season = self._resolve_season(
                competition=competition,
                run_date=run_date,
                season_overrides=season_overrides,
                route=route,
            )

            if competition.sport == SportName.BASKETBALL:
                if self._balldontlie is None:
                    logger.warning(
                        "Skipping fixture fetch for %s because BALLDONTLIE is unavailable.",
                        competition.key,
                    )
                    continue
                try:
                    basketball_run_fixtures = await self._balldontlie.fetch_games_by_date(
                        run_date=run_date,
                        seasons=(season,),
                    )
                except ProviderError as exc:
                    logger.warning(
                        "BALLDONTLIE fixture fetch failed for competition=%s date=%s: %s",
                        competition.key,
                        run_date,
                        exc,
                    )
                    continue
                collected_fixtures.extend(basketball_run_fixtures)
                continue

            soccer_run_fixtures = await self._fetch_soccer_fixtures_for_competition(
                competition=competition,
                route=route,
                run_date=run_date,
                season=season,
            )
            collected_fixtures.extend(soccer_run_fixtures)

        return self._deduplicate_fixtures(collected_fixtures)

    async def fetch_odds(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        markets_by_sport: dict[SportName, tuple[str, ...]] | None = None,
    ) -> OddsFetchResult:
        """Fetch odds for a fixture slate and build a lossless market catalog.

        Args:
            fixtures: Canonical fixtures for the current run.
            markets_by_sport: Optional provider market key overrides grouped by
                sport for The Odds API.

        Returns:
            A structured odds result containing the full market catalog and
            unresolved fixture diagnostics.
        """

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
        fixture_map = {fixture.get_fixture_ref(): fixture for fixture in fixtures}
        unmatched_fixtures = {fixture.get_fixture_ref(): fixture for fixture in fixtures}

        if self._the_odds_api is not None:
            providers_attempted.append(self._the_odds_api.provider_name)
            try:
                the_odds_rows = await self._fetch_the_odds_rows(
                    fixtures=fixtures,
                    markets_by_sport=markets_by_sport,
                )
            except ProviderError as exc:
                warning = f"The Odds API odds fetch failed: {exc}"
                logger.warning(warning)
                warnings.append(warning)
            else:
                matched_from_primary, unmatched_primary = self._match_odds_rows_to_fixtures(
                    odds_rows=the_odds_rows,
                    fixtures=tuple(unmatched_fixtures.values()),
                )
                matched_rows.extend(matched_from_primary)
                for fixture_ref in unmatched_primary:
                    logger.info(
                        "No The Odds API event match found for fixture_ref=%s; falling back.",
                        fixture_ref,
                    )
                for matched_row in matched_from_primary:
                    unmatched_fixtures.pop(matched_row.fixture_ref, None)

        if unmatched_fixtures and self._api_football is not None:
            providers_attempted.append(self._api_football.provider_name)
            api_rows = await self._fetch_api_football_fallback_odds(
                fixtures=tuple(unmatched_fixtures.values()),
            )
            if api_rows:
                matched_rows.extend(api_rows)
                for row in api_rows:
                    unmatched_fixtures.pop(row.fixture_ref, None)

        if unmatched_fixtures:
            warning = (
                "SportyBet odds fallback is not available yet. Remaining fixtures lack odds for: "
                + ", ".join(sorted(unmatched_fixtures))
            )
            logger.warning(warning)
            warnings.append(warning)

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
            unmatched_fixture_refs=tuple(sorted(unmatched_fixtures)),
            providers_attempted=tuple(providers_attempted),
            warnings=tuple(warnings),
        )

    async def fetch_stats(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        season_overrides: dict[str, int] | None = None,
        include_player_stats: bool = True,
    ) -> StatsFetchResult:
        """Fetch team and player stats using sport-specific provider routing.

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

        soccer_competitions = self._group_fixtures_by_competition_key(soccer_fixtures)
        for competition_key, competition_fixtures in soccer_competitions.items():
            competition = self._competition_by_key[competition_key]
            route = self._route_for_competition(competition)
            run_date = min(
                fixture.kickoff.astimezone(UTC).date()
                for fixture in competition_fixtures
            )
            season = self._resolve_season(
                competition=competition,
                run_date=run_date,
                season_overrides=season_overrides,
                route=route,
            )

            if (
                route is not None
                and self._api_football is not None
                and route.api_football_league_id is not None
            ):
                providers_attempted.append(self._api_football.provider_name)
                try:
                    team_stats.extend(
                        await self._api_football.fetch_standings(
                            league_id=route.api_football_league_id,
                            season=season,
                        )
                    )
                    if include_player_stats:
                        player_stats.extend(
                            await self._fetch_api_football_player_stats(
                                fixtures=competition_fixtures,
                                route=route,
                                season=season,
                            )
                        )
                    continue
                except (ProviderError, ValueError) as exc:
                    warning = f"API-Football stats fetch failed for {competition_key}: {exc}"
                    logger.warning(warning)
                    warnings.append(warning)

            if (
                route is not None
                and self._football_data is not None
                and route.football_data_code is not None
            ):
                providers_attempted.append(self._football_data.provider_name)
                try:
                    team_stats.extend(
                        await self._football_data.fetch_standings(
                            competition_code=route.football_data_code,
                            season=season,
                        )
                    )
                except ProviderError as exc:
                    warning = f"Football-Data.org stats fetch failed for {competition_key}: {exc}"
                    logger.warning(warning)
                    warnings.append(warning)

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
            team_stats=tuple(team_stats),
            player_stats=tuple(player_stats),
            providers_attempted=tuple(providers_attempted),
            warnings=tuple(warnings),
        )

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
        soccer_competitions = self._group_fixtures_by_competition_key(soccer_fixtures)
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
        """Fail fast until the SportyBet market resolvers exist.

        Args:
            fixtures: Canonical fixtures whose SportyBet markets are requested.

        Raises:
            ProviderError: Always, until the dedicated SportyBet scraper steps
                are implemented.
        """

        raise ProviderError(
            "sportybet",
            (
                "SportyBet market orchestration is not available yet. "
                "Implement Steps 22-24 before calling fetch_markets()."
            ),
        )

    async def aclose(self) -> None:
        """Close the shared HTTP client when the orchestrator created it."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()

    def _build_api_football(self) -> APIFootballProvider | None:
        """Construct API-Football when the environment is configured for it."""

        if self._client is None or self._settings.data_providers.api_football_key is None:
            return None
        return APIFootballProvider(self._client)

    def _build_football_data(self) -> FootballDataProvider | None:
        """Construct Football-Data.org when the environment is configured for it."""

        if self._client is None or self._settings.data_providers.football_data_api_key is None:
            return None
        return FootballDataProvider(self._client)

    def _build_the_odds_api(self) -> TheOddsAPIProvider | None:
        """Construct The Odds API provider when a key is available."""

        if self._client is None or self._settings.data_providers.the_odds_api_key is None:
            return None
        return TheOddsAPIProvider(self._client)

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

    async def _fetch_soccer_fixtures_for_competition(
        self,
        *,
        competition: CompetitionConfig,
        route: CompetitionProviderRoute | None,
        run_date: date,
        season: int,
    ) -> tuple[NormalizedFixture, ...]:
        """Fetch one soccer competition's fixtures with provider fallback."""

        if route is None:
            logger.warning(
                "Skipping fixture fetch for competition=%s because no route is configured.",
                competition.key,
            )
            return ()

        if self._api_football is not None and route.api_football_league_id is not None:
            try:
                api_fixtures = await self._api_football.fetch_fixtures_by_date(
                    run_date=run_date,
                    league_id=route.api_football_league_id,
                    season=season,
                )
            except ProviderError as exc:
                logger.warning(
                    "API-Football fixture fetch failed for competition=%s date=%s: %s",
                    competition.key,
                    run_date,
                    exc,
                )
            else:
                self._api_fixture_cache[(competition.key, run_date, season)] = tuple(api_fixtures)
                return tuple(api_fixtures)

        if self._football_data is not None and route.football_data_code is not None:
            try:
                return tuple(
                    await self._football_data.fetch_fixtures_by_date(
                        run_date=run_date,
                        competition_code=route.football_data_code,
                        season=season,
                    )
                )
            except ProviderError as exc:
                logger.warning(
                    "Football-Data.org fixture fetch failed for competition=%s date=%s: %s",
                    competition.key,
                    run_date,
                    exc,
                )
        return ()

    async def _fetch_the_odds_rows(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        markets_by_sport: dict[SportName, tuple[str, ...]] | None,
    ) -> tuple[NormalizedOdds, ...]:
        """Fetch The Odds API rows grouped by sport-key route."""

        if self._the_odds_api is None:
            return ()

        grouped_fixtures: dict[str, list[NormalizedFixture]] = {}
        for fixture in fixtures:
            route = self._route_for_fixture(fixture)
            if route is None or route.the_odds_sport_key is None:
                continue
            grouped_fixtures.setdefault(route.the_odds_sport_key, []).append(fixture)

        odds_rows: list[NormalizedOdds] = []
        for sport_key, sport_fixtures in grouped_fixtures.items():
            earliest_kickoff = min(
                fixture.kickoff.astimezone(UTC) for fixture in sport_fixtures
            ) - timedelta(hours=6)
            latest_kickoff = max(
                fixture.kickoff.astimezone(UTC) for fixture in sport_fixtures
            ) + timedelta(hours=6)
            sport = sport_fixtures[0].sport
            configured_market_keys = (
                markets_by_sport.get(sport)
                if markets_by_sport is not None and sport in markets_by_sport
                else None
            )
            market_keys = configured_market_keys or (
                _SOCCER_ODDS_MARKETS if sport == SportName.SOCCER else _NBA_ODDS_MARKETS
            )
            odds_rows.extend(
                await self._the_odds_api.fetch_odds(
                    sport_key=sport_key,
                    markets=market_keys,
                    commence_time_from=earliest_kickoff,
                    commence_time_to=latest_kickoff,
                )
            )

        return tuple(odds_rows)

    async def _fetch_api_football_fallback_odds(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[NormalizedOdds, ...]:
        """Fetch API-Football odds for fixtures unresolved by the primary odds feed."""

        if self._api_football is None:
            return ()

        rows: list[NormalizedOdds] = []
        for fixture in fixtures:
            api_fixture = await self._resolve_api_football_fixture_for_fixture(fixture)
            if api_fixture is None:
                logger.warning(
                    "Skipping API-Football odds fallback because fixture=%s could not be "
                    "matched to an API-Football fixture.",
                    fixture.get_fixture_ref(),
                )
                continue

            try:
                fetched_rows = await self._api_football.fetch_odds_by_fixture(
                    fixture_id=int(api_fixture.source_id)
                )
            except (ProviderError, ValueError) as exc:
                logger.warning(
                    "API-Football odds fallback failed for fixture=%s api_fixture=%s: %s",
                    fixture.get_fixture_ref(),
                    api_fixture.source_id,
                    exc,
                )
                continue

            rows.extend(
                row.model_copy(update={"fixture_ref": fixture.get_fixture_ref()})
                for row in fetched_rows
            )
        return tuple(rows)

    async def _fetch_api_football_player_stats(
        self,
        *,
        fixtures: tuple[NormalizedFixture, ...],
        route: CompetitionProviderRoute,
        season: int,
    ) -> tuple[PlayerStats, ...]:
        """Fetch player stats for the unique API-Football team IDs in a slate."""

        if self._api_football is None or route.api_football_league_id is None:
            return ()

        player_rows: list[PlayerStats] = []
        seen_team_ids: set[int] = set()
        for fixture in fixtures:
            api_fixture = await self._resolve_api_football_fixture_for_fixture(fixture)
            if api_fixture is None:
                continue
            for team_id in (api_fixture.home_team_id, api_fixture.away_team_id):
                if team_id is None:
                    continue
                try:
                    numeric_team_id = int(team_id)
                except ValueError:
                    continue
                if numeric_team_id in seen_team_ids:
                    continue
                seen_team_ids.add(numeric_team_id)
                try:
                    player_rows.extend(
                        await self._api_football.fetch_player_stats(
                            season=season,
                            team_id=numeric_team_id,
                            league_id=route.api_football_league_id,
                        )
                    )
                except (ProviderError, ValueError) as exc:
                    logger.warning(
                        "API-Football player stats fetch failed for team_id=%s competition=%s: %s",
                        numeric_team_id,
                        route.competition_key,
                        exc,
                    )
        return tuple(player_rows)

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

    def _match_odds_rows_to_fixtures(
        self,
        *,
        odds_rows: tuple[NormalizedOdds, ...],
        fixtures: tuple[NormalizedFixture, ...],
    ) -> tuple[tuple[NormalizedOdds, ...], tuple[str, ...]]:
        """Match provider odds rows to canonical fixtures by event identity."""

        event_groups: dict[str, list[NormalizedOdds]] = {}
        for row in odds_rows:
            raw_event_id = row.raw_metadata.get("event_id")
            group_key = str(raw_event_id) if raw_event_id is not None else row.fixture_ref
            event_groups.setdefault(group_key, []).append(row)

        matched_rows: list[NormalizedOdds] = []
        matched_fixture_refs: set[str] = set()
        for group_rows in event_groups.values():
            sample_row = group_rows[0]
            home_team = self._string_metadata(sample_row, "home_team")
            away_team = self._string_metadata(sample_row, "away_team")
            commence_time = self._datetime_metadata(sample_row, "commence_time")
            matched_fixture = self._find_best_fixture_match(
                reference_home_team=home_team,
                reference_away_team=away_team,
                reference_kickoff=commence_time,
                candidates=fixtures,
                sport=self._infer_fixture_sport(group_rows, fixtures),
                competition=None,
                country=None,
            )
            if matched_fixture is None:
                continue

            matched_fixture_ref = matched_fixture.get_fixture_ref()
            matched_fixture_refs.add(matched_fixture_ref)
            matched_rows.extend(
                row.model_copy(update={"fixture_ref": matched_fixture_ref}) for row in group_rows
            )

        unmatched_fixture_refs = tuple(
            sorted(
                fixture.get_fixture_ref()
                for fixture in fixtures
                if fixture.get_fixture_ref() not in matched_fixture_refs
            )
        )
        return tuple(matched_rows), unmatched_fixture_refs

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
            if home_score < 0.72 or away_score < 0.72:
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

    def _group_fixtures_by_competition_key(
        self,
        fixtures: tuple[NormalizedFixture, ...],
    ) -> dict[str, tuple[NormalizedFixture, ...]]:
        """Group fixtures by resolved competition key while preserving order."""

        grouped: dict[str, list[NormalizedFixture]] = {}
        for fixture in fixtures:
            competition = self._competition_for_fixture(fixture)
            if competition is None:
                logger.warning(
                    "Skipping fixture=%s because no competition mapping is configured.",
                    fixture.get_fixture_ref(),
                )
                continue
            grouped.setdefault(competition.key, []).append(fixture)
        return {key: tuple(values) for key, values in grouped.items()}

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

    def _infer_fixture_sport(
        self,
        group_rows: list[NormalizedOdds],
        fixtures: tuple[NormalizedFixture, ...],
    ) -> SportName | None:
        """Infer the sport for one odds event group from metadata or fixtures."""

        sample_row = group_rows[0]
        raw_sport_key = self._string_metadata(sample_row, "sport_key")
        if raw_sport_key is not None:
            normalized_sport_key = raw_sport_key.lower()
            if normalized_sport_key.startswith("soccer_"):
                return SportName.SOCCER
            if normalized_sport_key.startswith("basketball_"):
                return SportName.BASKETBALL

        for fixture in fixtures:
            if fixture.get_fixture_ref() == sample_row.fixture_ref:
                return fixture.sport
        return None

    @staticmethod
    def _string_metadata(row: NormalizedOdds, key: str) -> str | None:
        """Extract one string metadata field from a normalized odds row."""

        value = row.raw_metadata.get(key)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _datetime_metadata(row: NormalizedOdds, key: str) -> datetime | None:
        """Extract one timezone-aware datetime from odds metadata when present."""

        value = row.raw_metadata.get(key)
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

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
        return tokens

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
    "InjuryFetchResult",
    "OddsFetchResult",
    "ProviderOrchestrator",
    "StatsFetchResult",
]
