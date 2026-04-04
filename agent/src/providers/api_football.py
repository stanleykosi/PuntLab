"""API-Football provider implementation for PuntLab's soccer ingestion layer.

Purpose: connect the canonical provider infrastructure to the current
API-Football v3 endpoints for fixtures, odds, standings, player stats,
injuries, and head-to-head history.
Scope: authenticated HTTP requests, paginated API response handling, and
normalization into PuntLab's shared fixture, odds, and stats schemas.
Dependencies: `src.providers.base` for shared HTTP behavior, `src.config` for
runtime settings and market enums, and the shared Pydantic schemas under
`src.schemas`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import date, datetime
from math import isfinite
from typing import Final, cast

from src.cache.client import (
    API_FOOTBALL_FIXTURES_TTL_SECONDS,
    API_ODDS_TTL_SECONDS,
    API_STATS_TTL_SECONDS,
)
from src.config import MarketType, SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, PlayerStats, TeamStats

logger = logging.getLogger(__name__)

_API_DATE_FORMAT: Final[str] = "%Y-%m-%d"
_MATCH_STATUS_MAP: Final[dict[str, FixtureStatus]] = {
    "TBD": FixtureStatus.SCHEDULED,
    "NS": FixtureStatus.SCHEDULED,
    "1H": FixtureStatus.LIVE,
    "HT": FixtureStatus.LIVE,
    "2H": FixtureStatus.LIVE,
    "ET": FixtureStatus.LIVE,
    "BT": FixtureStatus.LIVE,
    "P": FixtureStatus.LIVE,
    "INT": FixtureStatus.LIVE,
    "FT": FixtureStatus.FINISHED,
    "AET": FixtureStatus.FINISHED,
    "PEN": FixtureStatus.FINISHED,
    "PST": FixtureStatus.POSTPONED,
    "SUSP": FixtureStatus.CANCELLED,
    "CANC": FixtureStatus.CANCELLED,
    "ABD": FixtureStatus.CANCELLED,
    "AWD": FixtureStatus.CANCELLED,
    "WO": FixtureStatus.CANCELLED,
}
_TEAM_SELECTION_MAP: Final[dict[str, str]] = {
    "home": "home",
    "1": "home",
    "away": "away",
    "2": "away",
    "draw": "draw",
    "x": "draw",
    "yes": "yes",
    "no": "no",
    "1x": "1X",
    "12": "12",
    "x2": "X2",
    "home/draw": "1X",
    "home/away": "12",
    "draw/away": "X2",
    "team1 or draw": "1X",
    "team1 or team2": "12",
    "draw or team2": "X2",
}
_HT_FT_TOKEN_MAP: Final[dict[str, str]] = {
    "home": "1",
    "draw": "X",
    "away": "2",
    "1": "1",
    "x": "X",
    "2": "2",
}
_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?P<line>[+-]?\d+(?:\.\d+)?)")
_NUMERIC_TEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_PERCENT_TEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[+-]?\d+(?:\.\d+)?%$")
_NON_METRIC_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_SUPPORTED_TOTAL_MARKET_LINES: Final[dict[float, MarketType]] = {
    0.5: MarketType.OVER_UNDER_05,
    1.5: MarketType.OVER_UNDER_15,
    2.5: MarketType.OVER_UNDER_25,
    3.5: MarketType.OVER_UNDER_35,
}


class APIFootballProvider(DataProvider):
    """Concrete API-Football v3 integration used for soccer ingestion.

    Inputs:
        A shared rate-limited client and an API key from either the constructor
        or runtime settings.

    Outputs:
        Typed provider methods that return normalized PuntLab schemas for the
        canonical ingestion flow.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        api_key: str | None = None,
        default_timezone: str | None = None,
    ) -> None:
        """Initialize API credentials and the default response timezone.

        Args:
            client: Shared `RateLimitedClient` instance used by the provider
                base class for cached, retried HTTP requests.
            api_key: Optional explicit API-Football key. When omitted, the
                provider falls back to `API_FOOTBALL_KEY` from settings.
            default_timezone: Optional timezone name passed to endpoints that
                support kickoff localization.

        Raises:
            ValueError: If no usable API key is available.
        """

        super().__init__(client)
        resolved_api_key = (api_key or get_settings().data_providers.api_football_key or "").strip()
        if not resolved_api_key:
            raise ValueError(
                "API-Football requires `API_FOOTBALL_KEY` or an explicit `api_key`."
            )

        self._api_key = resolved_api_key
        self._default_timezone = (default_timezone or get_settings().app.timezone_name).strip()
        if not self._default_timezone:
            raise ValueError("default_timezone must not be blank.")

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in cache keys and logs."""

        return "api-football"

    @property
    def base_url(self) -> str:
        """Return the current official API-Football v3 base URL."""

        return "https://v3.football.api-sports.io"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return the free-tier API-Football request budget from the spec."""

        return RateLimitPolicy(limit=100)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the canonical authentication and content negotiation headers."""

        return {
            "x-apisports-key": self._api_key,
            "Accept": "application/json",
        }

    async def fetch_fixtures_by_date(
        self,
        *,
        run_date: date,
        league_id: int,
        season: int,
        timezone: str | None = None,
    ) -> list[NormalizedFixture]:
        """Fetch one league's fixtures for a specific date.

        Args:
            run_date: Calendar date to query in the provider API.
            league_id: API-Football competition identifier.
            season: API-Football season year for the competition.
            timezone: Optional override for kickoff localization.

        Returns:
            A list of normalized fixtures for the requested competition date.
        """

        payload = await self._fetch_json(
            "/fixtures",
            params={
                "league": self._require_positive_int("league_id", league_id),
                "season": self._require_positive_int("season", season),
                "date": run_date.strftime(_API_DATE_FORMAT),
                "timezone": self._resolve_timezone(timezone),
            },
            cache_ttl_seconds=API_FOOTBALL_FIXTURES_TTL_SECONDS,
        )
        return [
            self._normalize_fixture(response_item)
            for response_item in self._extract_response_items(payload)
        ]

    async def fetch_odds_by_fixture(
        self,
        *,
        fixture_id: int,
        bookmaker_id: int | None = None,
        timezone: str | None = None,
    ) -> list[NormalizedOdds]:
        """Fetch and normalize the supported pre-match odds for one fixture.

        Args:
            fixture_id: Provider fixture identifier used by the `/odds` endpoint.
            bookmaker_id: Optional bookmaker filter to reduce response volume.
            timezone: Optional override for response timestamps.

        Returns:
            A list of normalized odds rows across all supported market families.
        """

        params: dict[str, object] = {
            "fixture": self._require_positive_int("fixture_id", fixture_id),
            "timezone": self._resolve_timezone(timezone),
        }
        if bookmaker_id is not None:
            params["bookmaker"] = self._require_positive_int("bookmaker_id", bookmaker_id)

        payloads = await self._fetch_paginated_json(
            "/odds",
            params=params,
            cache_ttl_seconds=API_ODDS_TTL_SECONDS,
        )

        normalized_odds: list[NormalizedOdds] = []
        for payload in payloads:
            for response_item in self._extract_response_items(payload):
                normalized_odds.extend(self._normalize_odds_fixture_response(response_item))
        return normalized_odds

    async def fetch_standings(
        self,
        *,
        league_id: int,
        season: int,
    ) -> list[TeamStats]:
        """Fetch normalized table snapshots for one league season.

        Args:
            league_id: API-Football competition identifier.
            season: API-Football season year for the competition.

        Returns:
            A flat list of team snapshots derived from the standings table.
        """

        payload = await self._fetch_json(
            "/standings",
            params={
                "league": self._require_positive_int("league_id", league_id),
                "season": self._require_positive_int("season", season),
            },
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
        )

        standings: list[TeamStats] = []
        fetched_at = self._response_timestamp_or_now(payload.get("response"))
        for league_wrapper in self._extract_response_items(payload):
            league_data = self._require_mapping(league_wrapper.get("league"), "league")
            groups = league_data.get("standings")
            if not isinstance(groups, list):
                raise ProviderError(
                    self.provider_name,
                    "API-Football standings response is missing `league.standings`.",
                )

            for group in groups:
                if not isinstance(group, list):
                    raise ProviderError(
                        self.provider_name,
                        "API-Football standings groups must be arrays of team rows.",
                    )
                for row in group:
                    standings.append(
                        self._normalize_standings_row(
                            row,
                            league_data=league_data,
                            fetched_at=fetched_at,
                        )
                    )

        return standings

    async def fetch_player_stats(
        self,
        *,
        season: int,
        team_id: int | None = None,
        league_id: int | None = None,
        player_id: int | None = None,
        search: str | None = None,
    ) -> list[PlayerStats]:
        """Fetch normalized season-level player statistics.

        Args:
            season: API-Football season year.
            team_id: Optional team filter used to limit the player pool.
            league_id: Optional competition filter used alongside team or broad
                season queries.
            player_id: Optional specific player identifier.
            search: Optional provider search term for one player.

        Returns:
            A flattened list of normalized player stat snapshots, one per player
            and competition statistics bundle returned by the provider.

        Raises:
            ValueError: If no narrowing filter is supplied.
        """

        if all(value is None for value in (team_id, league_id, player_id, search)):
            raise ValueError(
                "fetch_player_stats requires at least one of `team_id`, `league_id`, "
                "`player_id`, or `search` to avoid accidental full-dataset crawls."
            )

        params: dict[str, object] = {
            "season": self._require_positive_int("season", season),
        }
        if team_id is not None:
            params["team"] = self._require_positive_int("team_id", team_id)
        if league_id is not None:
            params["league"] = self._require_positive_int("league_id", league_id)
        if player_id is not None:
            params["id"] = self._require_positive_int("player_id", player_id)
        if search is not None:
            normalized_search = search.strip()
            if not normalized_search:
                raise ValueError("search must not be blank.")
            params["search"] = normalized_search

        payloads = await self._fetch_paginated_json(
            "/players",
            params=params,
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
        )

        player_stats: list[PlayerStats] = []
        for payload in payloads:
            fetched_at = self._response_timestamp_or_now(payload.get("response"))
            for response_item in self._extract_response_items(payload):
                player_stats.extend(
                    self._normalize_player_response(
                        response_item,
                        fetched_at=fetched_at,
                    )
                )
        return player_stats

    async def fetch_injuries(
        self,
        *,
        fixture_id: int | None = None,
        league_id: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        player_id: int | None = None,
        report_date: date | None = None,
    ) -> list[InjuryData]:
        """Fetch normalized injury and suspension signals.

        Args:
            fixture_id: Optional fixture filter for match-specific availability.
            league_id: Optional competition filter. Requires `season`.
            season: Optional season filter used with `league_id`.
            team_id: Optional team filter for club-specific injuries.
            player_id: Optional player filter for one athlete.
            report_date: Optional date filter for one injury snapshot day.

        Returns:
            A list of normalized injury or suspension records.

        Raises:
            ValueError: If no usable query selector is supplied.
        """

        if all(
            value is None
            for value in (fixture_id, league_id, team_id, player_id, report_date)
        ):
            raise ValueError(
                "fetch_injuries requires at least one selector such as `fixture_id`, "
                "`league_id`, `team_id`, `player_id`, or `report_date`."
            )
        if league_id is not None and season is None:
            raise ValueError("season is required when filtering injuries by league_id.")

        params: dict[str, object] = {}
        if fixture_id is not None:
            params["fixture"] = self._require_positive_int("fixture_id", fixture_id)
        if league_id is not None:
            params["league"] = self._require_positive_int("league_id", league_id)
        if season is not None:
            params["season"] = self._require_positive_int("season", season)
        if team_id is not None:
            params["team"] = self._require_positive_int("team_id", team_id)
        if player_id is not None:
            params["player"] = self._require_positive_int("player_id", player_id)
        if report_date is not None:
            params["date"] = report_date.strftime(_API_DATE_FORMAT)

        payload = await self._fetch_json(
            "/injuries",
            params=params,
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
        )

        injuries: list[InjuryData] = []
        for response_item in self._extract_response_items(payload):
            injuries.append(self._normalize_injury(response_item))
        return injuries

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
        """Fetch historical head-to-head fixtures between two teams.

        Args:
            home_team_id: First team identifier in the `h2h` query.
            away_team_id: Second team identifier in the `h2h` query.
            last: Optional limit for the most recent N meetings.
            league_id: Optional competition filter.
            season: Optional season filter.
            from_date: Optional inclusive start date filter.
            to_date: Optional inclusive end date filter.

        Returns:
            A list of normalized fixtures using the same contract as `/fixtures`.
        """

        params: dict[str, object] = {
            "h2h": (
                f"{self._require_positive_int('home_team_id', home_team_id)}-"
                f"{self._require_positive_int('away_team_id', away_team_id)}"
            )
        }
        if last is not None:
            params["last"] = self._require_positive_int("last", last)
        if league_id is not None:
            params["league"] = self._require_positive_int("league_id", league_id)
        if season is not None:
            params["season"] = self._require_positive_int("season", season)
        if from_date is not None:
            params["from"] = from_date.strftime(_API_DATE_FORMAT)
        if to_date is not None:
            params["to"] = to_date.strftime(_API_DATE_FORMAT)

        payload = await self._fetch_json(
            "/fixtures/headtohead",
            params=params,
            cache_ttl_seconds=API_FOOTBALL_FIXTURES_TTL_SECONDS,
        )

        return [
            self._normalize_fixture(response_item)
            for response_item in self._extract_response_items(payload)
        ]

    async def _fetch_json(
        self,
        path: str,
        *,
        params: Mapping[str, object],
        cache_ttl_seconds: int,
    ) -> dict[str, object]:
        """Fetch one API-Football JSON payload and validate its envelope."""

        response = await self.fetch(
            "GET",
            path,
            params=params,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProviderError(
                self.provider_name,
                f"API-Football returned a non-object JSON payload for {path}.",
            )

        self._raise_on_api_errors(payload)
        return cast(dict[str, object], payload)

    async def _fetch_paginated_json(
        self,
        path: str,
        *,
        params: Mapping[str, object],
        cache_ttl_seconds: int,
    ) -> list[dict[str, object]]:
        """Fetch every page for paginated API-Football endpoints."""

        page = 1
        payloads: list[dict[str, object]] = []

        while True:
            request_params = dict(params)
            request_params["page"] = page
            payload = await self._fetch_json(
                path,
                params=request_params,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            payloads.append(payload)

            paging = payload.get("paging")
            if not isinstance(paging, Mapping):
                break

            current_page = self._coerce_positive_int(
                paging.get("current"),
                field_name="paging.current",
                allow_none=True,
            ) or page
            total_pages = self._coerce_positive_int(
                paging.get("total"),
                field_name="paging.total",
                allow_none=True,
            ) or current_page
            if current_page >= total_pages:
                break
            page = current_page + 1

        return payloads

    def _raise_on_api_errors(self, payload: Mapping[str, object]) -> None:
        """Raise a provider error when the API envelope reports failures."""

        raw_errors = payload.get("errors")
        if raw_errors in (None, [], {}):
            return

        if isinstance(raw_errors, list):
            error_text = "; ".join(str(item).strip() for item in raw_errors if str(item).strip())
        elif isinstance(raw_errors, Mapping):
            fragments = [
                f"{key}: {value}"
                for key, value in raw_errors.items()
                if str(value).strip()
            ]
            error_text = "; ".join(fragments)
        else:
            error_text = str(raw_errors).strip()

        raise ProviderError(
            self.provider_name,
            f"API-Football reported upstream errors: {error_text or 'unknown error'}.",
        )

    def _extract_response_items(self, payload: Mapping[str, object]) -> list[dict[str, object]]:
        """Return the typed `response` array from one API-Football payload."""

        response_items = payload.get("response")
        if not isinstance(response_items, list):
            raise ProviderError(
                self.provider_name,
                "API-Football payload is missing the `response` array.",
            )

        typed_items: list[dict[str, object]] = []
        for item in response_items:
            if not isinstance(item, dict):
                raise ProviderError(
                    self.provider_name,
                    "API-Football response items must be JSON objects.",
                )
            typed_items.append(cast(dict[str, object], item))
        return typed_items

    def _normalize_fixture(self, response_item: Mapping[str, object]) -> NormalizedFixture:
        """Normalize one API-Football fixture payload into PuntLab's schema."""

        fixture_data = self._require_mapping(response_item.get("fixture"), "fixture")
        teams_data = self._require_mapping(response_item.get("teams"), "teams")
        league_data = self._require_mapping(response_item.get("league"), "league")
        home_team = self._require_mapping(teams_data.get("home"), "teams.home")
        away_team = self._require_mapping(teams_data.get("away"), "teams.away")
        status_data = self._require_mapping(fixture_data.get("status"), "fixture.status")
        venue_data = fixture_data.get("venue")
        typed_venue_data = (
            self._require_mapping(venue_data, "fixture.venue")
            if isinstance(venue_data, Mapping)
            else None
        )

        kickoff = self._parse_datetime(
            fixture_data.get("date"),
            field_name="fixture.date",
        )
        fixture_id = self._required_positive_int(
            fixture_data.get("id"),
            field_name="fixture.id",
        )
        short_status = str(status_data.get("short", "")).strip().upper()

        return NormalizedFixture(
            home_team=self._require_text(home_team.get("name"), "teams.home.name"),
            away_team=self._require_text(away_team.get("name"), "teams.away.name"),
            competition=self._require_text(league_data.get("name"), "league.name"),
            sport=SportName.SOCCER,
            kickoff=kickoff,
            source_provider=self.provider_name,
            source_id=str(fixture_id),
            country=self._optional_text(league_data.get("country")),
            home_team_id=self._optional_identifier(home_team.get("id")),
            away_team_id=self._optional_identifier(away_team.get("id")),
            venue=(
                self._optional_text(typed_venue_data.get("name"))
                if typed_venue_data is not None
                else None
            ),
            status=_MATCH_STATUS_MAP.get(short_status, FixtureStatus.SCHEDULED),
        )

    def _normalize_odds_fixture_response(
        self,
        response_item: Mapping[str, object],
    ) -> list[NormalizedOdds]:
        """Normalize one fixture odds response across bookmakers and bet types."""

        fixture_data = self._require_mapping(response_item.get("fixture"), "fixture")
        fixture_id = self._required_positive_int(
            fixture_data.get("id"),
            field_name="fixture.id",
        )
        fixture_ref = self._build_fixture_ref(fixture_id)
        last_updated = self._optional_datetime(
            response_item.get("update"),
            field_name="response.update",
        )

        normalized_odds: list[NormalizedOdds] = []
        bookmakers = response_item.get("bookmakers")
        if not isinstance(bookmakers, list):
            raise ProviderError(
                self.provider_name,
                "API-Football odds response is missing the `bookmakers` array.",
            )

        for bookmaker in bookmakers:
            bookmaker_data = self._require_mapping(bookmaker, "bookmaker")
            bookmaker_name = self._require_text(
                bookmaker_data.get("name"),
                "bookmaker.name",
            )
            bets = bookmaker_data.get("bets")
            if not isinstance(bets, list):
                raise ProviderError(
                    self.provider_name,
                    "API-Football odds bookmaker entries must contain `bets` arrays.",
                )

            for bet in bets:
                bet_data = self._require_mapping(bet, "bet")
                bet_name = self._require_text(bet_data.get("name"), "bet.name")
                provider_market_id = bet_data.get("id")
                values = bet_data.get("values")
                if not isinstance(values, list):
                    raise ProviderError(
                        self.provider_name,
                        "API-Football odds bet entries must contain `values` arrays.",
                    )

                for value_entry in values:
                    normalized_odds.append(
                        self._parse_odds_market(
                        fixture_ref=fixture_ref,
                        bookmaker_name=bookmaker_name,
                        bet_name=bet_name,
                        provider_market_id=provider_market_id,
                        value_entry=self._require_mapping(value_entry, "bet.value"),
                        last_updated=last_updated,
                        )
                    )

        return normalized_odds

    def _parse_odds_market(
        self,
        *,
        fixture_ref: str,
        bookmaker_name: str,
        bet_name: str,
        provider_market_id: object,
        value_entry: Mapping[str, object],
        last_updated: datetime | None,
    ) -> NormalizedOdds:
        """Map one API-Football odds value into the broad normalized odds schema."""

        raw_selection = self._require_text(value_entry.get("value"), "bet.value.value")
        odds_value = self._coerce_decimal_odds(
            value_entry.get("odd"),
            field_name="bet.value.odd",
        )

        normalized_bet_name = self._normalize_phrase(bet_name)
        market: MarketType | None = None
        selection = raw_selection
        line = self._extract_market_line(
            normalized_bet_name,
            raw_selection=raw_selection,
            bet_name=bet_name,
        )

        if normalized_bet_name in {"match winner", "winner"}:
            market = MarketType.MATCH_RESULT
            selection = self._normalize_match_result_selection(raw_selection)
        elif "both teams" in normalized_bet_name and "score" in normalized_bet_name:
            market = MarketType.BTTS
            selection = self._normalize_yes_no_selection(raw_selection)
        elif "double chance" in normalized_bet_name:
            market = MarketType.DOUBLE_CHANCE
            selection = self._normalize_double_chance_selection(raw_selection)
        elif "draw no bet" in normalized_bet_name:
            market = MarketType.DRAW_NO_BET
            selection = self._normalize_match_result_selection(raw_selection)
        elif "asian handicap" in normalized_bet_name:
            market = MarketType.ASIAN_HANDICAP
            selection, line = self._normalize_line_selection(raw_selection)
        elif "correct score" in normalized_bet_name:
            market = MarketType.CORRECT_SCORE
            selection = raw_selection.strip()
        elif "halftime" in normalized_bet_name and "fulltime" in normalized_bet_name:
            market = MarketType.HT_FT
            selection = self._normalize_ht_ft_selection(raw_selection)
        elif "over" in normalized_bet_name and "under" in normalized_bet_name:
            if line is not None:
                market = _SUPPORTED_TOTAL_MARKET_LINES.get(line)
            selection = self._normalize_over_under_selection(raw_selection)

        if market is None:
            logger.debug(
                "Preserving unmapped API-Football market bet=%s selection=%s",
                bet_name,
                raw_selection,
            )

        return NormalizedOdds(
            fixture_ref=fixture_ref,
            market=market,
            selection=selection,
            odds=odds_value,
            provider=bookmaker_name,
            provider_market_name=bet_name,
            provider_selection_name=raw_selection,
            sportybet_available=False,
            market_label=bet_name,
            line=line,
            period=self._infer_period(bet_name),
            participant_scope=self._infer_participant_scope(bet_name),
            provider_market_id=self._optional_market_id(provider_market_id),
            raw_metadata={
                "canonical_market_supported": market is not None,
            },
            last_updated=last_updated,
        )

    def _normalize_standings_row(
        self,
        row: object,
        *,
        league_data: Mapping[str, object],
        fetched_at: datetime,
    ) -> TeamStats:
        """Normalize one table row from the standings endpoint."""

        row_data = self._require_mapping(row, "league.standings.row")
        team_data = self._require_mapping(row_data.get("team"), "standings.team")
        all_data = self._require_mapping(row_data.get("all"), "standings.all")
        home_data = self._require_mapping(row_data.get("home"), "standings.home")
        away_data = self._require_mapping(row_data.get("away"), "standings.away")
        goals_data = self._require_mapping(all_data.get("goals"), "standings.all.goals")

        matches_played = self._required_non_negative_int(
            all_data.get("played"),
            field_name="standings.all.played",
        )
        goals_for = self._required_non_negative_int(
            goals_data.get("for"),
            field_name="standings.all.goals.for",
        )
        goals_against = self._required_non_negative_int(
            goals_data.get("against"),
            field_name="standings.all.goals.against",
        )

        avg_goals_scored = (
            goals_for / matches_played if matches_played > 0 else None
        )
        avg_goals_conceded = (
            goals_against / matches_played if matches_played > 0 else None
        )

        return TeamStats(
            team_id=self._optional_identifier(team_data.get("id"))
            or self._require_text(team_data.get("name"), "standings.team.name"),
            team_name=self._require_text(team_data.get("name"), "standings.team.name"),
            sport=SportName.SOCCER,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            competition=self._require_text(league_data.get("name"), "league.name"),
            season=str(
                self._required_positive_int(
                    league_data.get("season"),
                    field_name="league.season",
                )
            ),
            matches_played=matches_played,
            wins=self._required_non_negative_int(
                all_data.get("win"),
                field_name="standings.all.win",
            ),
            draws=self._required_non_negative_int(
                all_data.get("draw"),
                field_name="standings.all.draw",
            ),
            losses=self._required_non_negative_int(
                all_data.get("lose"),
                field_name="standings.all.lose",
            ),
            goals_for=goals_for,
            goals_against=goals_against,
            form=self._optional_text(row_data.get("form")),
            position=self._coerce_positive_int(
                row_data.get("rank"),
                field_name="standings.rank",
                allow_none=True,
            ),
            points=self._coerce_non_negative_int(
                row_data.get("points"),
                field_name="standings.points",
                allow_none=True,
            ),
            home_wins=self._required_non_negative_int(
                home_data.get("win"),
                field_name="standings.home.win",
            ),
            away_wins=self._required_non_negative_int(
                away_data.get("win"),
                field_name="standings.away.win",
            ),
            avg_goals_scored=avg_goals_scored,
            avg_goals_conceded=avg_goals_conceded,
            advanced_metrics=self._collect_numeric_metrics(
                {
                    "goal_difference": row_data.get("goalsDiff"),
                }
            ),
        )

    def _normalize_player_response(
        self,
        response_item: Mapping[str, object],
        *,
        fetched_at: datetime,
    ) -> list[PlayerStats]:
        """Flatten one API-Football player response into stat snapshots."""

        player_data = self._require_mapping(response_item.get("player"), "player")
        statistics_data = response_item.get("statistics")
        if not isinstance(statistics_data, list):
            raise ProviderError(
                self.provider_name,
                "API-Football player response is missing the `statistics` array.",
            )

        player_stats: list[PlayerStats] = []
        player_id = self._optional_identifier(player_data.get("id"))
        player_name = self._require_text(player_data.get("name"), "player.name")

        for statistics_item in statistics_data:
            stats_bundle = self._require_mapping(statistics_item, "player.statistics")
            team_data = self._require_mapping(stats_bundle.get("team"), "player.statistics.team")
            league_data = self._require_mapping(
                stats_bundle.get("league"),
                "player.statistics.league",
            )
            games_data = self._require_mapping(
                stats_bundle.get("games"),
                "player.statistics.games",
            )

            player_stats.append(
                PlayerStats(
                    player_id=player_id or player_name,
                    player_name=player_name,
                    team_id=self._optional_identifier(team_data.get("id"))
                    or self._require_text(team_data.get("name"), "player.statistics.team.name"),
                    sport=SportName.SOCCER,
                    source_provider=self.provider_name,
                    fetched_at=fetched_at,
                    team_name=self._optional_text(team_data.get("name")),
                    competition=self._optional_text(league_data.get("name")),
                    season=self._stringify_optional(league_data.get("season")),
                    position=self._optional_text(games_data.get("position")),
                    appearances=self._required_non_negative_int(
                        games_data.get("appearences"),
                        field_name="player.statistics.games.appearences",
                    ),
                    starts=self._required_non_negative_int(
                        games_data.get("lineups"),
                        field_name="player.statistics.games.lineups",
                    ),
                    minutes_played=self._required_non_negative_int(
                        games_data.get("minutes"),
                        field_name="player.statistics.games.minutes",
                    ),
                    metrics=self._collect_numeric_metrics(stats_bundle),
                )
            )

        return player_stats

    def _normalize_injury(self, response_item: Mapping[str, object]) -> InjuryData:
        """Normalize one API-Football injury record."""

        player_data = self._require_mapping(response_item.get("player"), "player")
        team_data = self._require_mapping(response_item.get("team"), "team")
        fixture_data = self._require_mapping(response_item.get("fixture"), "fixture")

        fixture_id = self._required_positive_int(
            fixture_data.get("id"),
            field_name="fixture.id",
        )

        return InjuryData(
            fixture_ref=self._build_fixture_ref(fixture_id),
            team_id=self._optional_identifier(team_data.get("id"))
            or self._require_text(team_data.get("name"), "team.name"),
            player_name=self._require_text(player_data.get("name"), "player.name"),
            source_provider=self.provider_name,
            injury_type=self._normalize_injury_type(response_item.get("type")),
            team_name=self._optional_text(team_data.get("name")),
            player_id=self._optional_identifier(player_data.get("id")),
            reason=self._optional_text(response_item.get("reason")),
            is_key_player=False,
            expected_return=self._optional_date(response_item.get("end"), field_name="injury.end"),
            reported_at=self._optional_datetime(
                response_item.get("date"),
                field_name="injury.date",
            ),
        )

    def _response_timestamp_or_now(self, response_items: object) -> datetime:
        """Infer a stable fetched timestamp from the payload or current time."""

        if isinstance(response_items, list):
            for item in response_items:
                if not isinstance(item, Mapping):
                    continue
                for candidate_key in ("update", "date"):
                    parsed = self._optional_datetime(
                        item.get(candidate_key),
                        field_name=candidate_key,
                    )
                    if parsed is not None:
                        return parsed
        return datetime.now(get_settings().timezone)

    def _normalize_match_result_selection(self, selection: str) -> str:
        """Normalize home/draw/away selections across match result markets."""

        normalized = self._normalize_phrase(selection)
        if normalized not in _TEAM_SELECTION_MAP:
            raise ProviderError(
                self.provider_name,
                f"Unsupported match-result selection from API-Football: {selection}.",
            )
        return _TEAM_SELECTION_MAP[normalized]

    def _normalize_yes_no_selection(self, selection: str) -> str:
        """Normalize yes/no markets such as BTTS."""

        normalized = self._normalize_phrase(selection)
        if normalized not in {"yes", "no"}:
            raise ProviderError(
                self.provider_name,
                f"Unsupported yes/no selection from API-Football: {selection}.",
            )
        return normalized

    def _normalize_double_chance_selection(self, selection: str) -> str:
        """Normalize double-chance labels into the canonical 1X/12/X2 form."""

        normalized = self._normalize_phrase(selection)
        mapped = _TEAM_SELECTION_MAP.get(normalized)
        if mapped is None or mapped not in {"1X", "12", "X2"}:
            raise ProviderError(
                self.provider_name,
                f"Unsupported double-chance selection from API-Football: {selection}.",
            )
        return mapped

    def _normalize_over_under_selection(self, selection: str) -> str:
        """Normalize total-market side labels into `over` or `under`."""

        normalized = self._normalize_phrase(selection)
        if normalized.startswith("over"):
            return "over"
        if normalized.startswith("under"):
            return "under"
        raise ProviderError(
            self.provider_name,
            f"Unsupported over/under selection from API-Football: {selection}.",
        )

    def _normalize_line_selection(self, selection: str) -> tuple[str, float]:
        """Normalize line-based selections such as Asian handicaps."""

        line = self._extract_line(selection)
        if line is None:
            raise ProviderError(
                self.provider_name,
                f"API-Football line-based selection is missing a numeric line: {selection}.",
            )

        normalized = self._normalize_phrase(selection)
        if normalized.startswith(("home", "1")):
            return "home", line
        if normalized.startswith(("away", "2")):
            return "away", line
        raise ProviderError(
            self.provider_name,
            f"Unsupported line-based selection from API-Football: {selection}.",
        )

    def _normalize_ht_ft_selection(self, selection: str) -> str:
        """Normalize halftime/fulltime labels into compact `1/X/2` tokens."""

        parts = [part for part in re.split(r"\s*/\s*", selection.strip()) if part]
        if len(parts) != 2:
            raise ProviderError(
                self.provider_name,
                f"Unsupported halftime/fulltime selection from API-Football: {selection}.",
            )

        normalized_parts: list[str] = []
        for part in parts:
            token = _HT_FT_TOKEN_MAP.get(self._normalize_phrase(part))
            if token is None:
                raise ProviderError(
                    self.provider_name,
                    f"Unsupported halftime/fulltime selection token: {part}.",
                )
            normalized_parts.append(token)
        return "/".join(normalized_parts)

    def _infer_period(self, market_name: str) -> str | None:
        """Infer the broad match period associated with one provider market."""

        normalized = self._normalize_phrase(market_name)
        if "1st half" in normalized or "first half" in normalized:
            return "first_half"
        if "2nd half" in normalized or "second half" in normalized:
            return "second_half"
        if "extra time" in normalized:
            return "extra_time"
        if "penalt" in normalized:
            return "penalties"
        return "match"

    def _infer_participant_scope(self, market_name: str) -> str | None:
        """Infer whether a market is match-, team-, or player-oriented."""

        normalized = self._normalize_phrase(market_name)
        if "player" in normalized:
            return "player"
        if "team" in normalized:
            return "team"
        return "match"

    def _normalize_injury_type(self, value: object) -> InjuryType:
        """Map provider injury labels into PuntLab's canonical enum."""

        normalized = self._normalize_phrase(self._require_text(value, "injury.type"))
        if "susp" in normalized:
            return InjuryType.SUSPENSION
        if "doubt" in normalized:
            return InjuryType.DOUBTFUL
        if "question" in normalized:
            return InjuryType.QUESTIONABLE
        if "ill" in normalized:
            return InjuryType.ILLNESS
        if "rest" in normalized:
            return InjuryType.REST
        if "inj" in normalized:
            return InjuryType.INJURY
        return InjuryType.OTHER

    def _collect_numeric_metrics(self, payload: Mapping[str, object]) -> dict[str, float]:
        """Flatten nested provider statistics into a finite metric map."""

        metrics: dict[str, float] = {}

        def visit(node: object, prefix: str) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    if key in {"team", "league", "player"}:
                        continue
                    child_key = self._normalize_metric_key(str(key))
                    if not child_key:
                        continue
                    visit(value, f"{prefix}_{child_key}" if prefix else child_key)
                return

            if isinstance(node, bool):
                return
            if isinstance(node, (int, float)):
                if isfinite(float(node)):
                    metrics[prefix] = float(node)
                return
            if isinstance(node, str):
                parsed = self._parse_optional_numeric_text(node)
                if parsed is not None:
                    metrics[prefix] = parsed

        for key, value in payload.items():
            if key in {"team", "league", "player"}:
                continue
            root_key = self._normalize_metric_key(str(key))
            if root_key:
                visit(value, root_key)

        return metrics

    @staticmethod
    def _build_fixture_ref(fixture_id: int) -> str:
        """Build the canonical provider-scoped fixture reference."""

        return f"api-football:{fixture_id}"

    @staticmethod
    def _normalize_phrase(value: str) -> str:
        """Normalize free-text provider labels for deterministic matching."""

        return " ".join(value.strip().lower().replace("-", " ").split())

    @staticmethod
    def _normalize_metric_key(value: str) -> str:
        """Convert nested provider metric keys into stable snake_case fragments."""

        normalized = _NON_METRIC_KEY_PATTERN.sub("_", value.strip().lower()).strip("_")
        if normalized == "percentage":
            return "pct"
        return normalized

    @staticmethod
    def _extract_line(value: str) -> float | None:
        """Extract the first numeric line from a market or selection label."""

        match = _LINE_PATTERN.search(value)
        if match is None:
            return None
        try:
            return float(match.group("line"))
        except ValueError:
            return None

    def _extract_market_line(
        self,
        normalized_bet_name: str,
        *,
        raw_selection: str,
        bet_name: str,
    ) -> float | None:
        """Extract a numeric line only for markets that look line-based."""

        line_indicators = ("over", "under", "handicap", "spread", "total")
        if not any(indicator in normalized_bet_name for indicator in line_indicators):
            return None
        return self._extract_line(raw_selection) or self._extract_line(bet_name)

    @staticmethod
    def _parse_optional_numeric_text(value: str) -> float | None:
        """Parse simple numeric or percentage strings into floats."""

        normalized = value.strip()
        if not normalized:
            return None
        if _PERCENT_TEXT_PATTERN.fullmatch(normalized):
            return float(normalized[:-1])
        if _NUMERIC_TEXT_PATTERN.fullmatch(normalized):
            return float(normalized)
        return None

    @staticmethod
    def _optional_text(value: object) -> str | None:
        """Return a trimmed string or `None` when the value is blank."""

        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _stringify_optional(value: object) -> str | None:
        """Convert optional scalar values into trimmed strings."""

        if value is None:
            return None
        return str(value).strip() or None

    @staticmethod
    def _optional_identifier(value: object) -> str | None:
        """Convert provider IDs into canonical string identifiers."""

        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _optional_market_id(value: object) -> int | str | None:
        """Normalize optional provider market IDs for odds traceability."""

        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        normalized = str(value).strip()
        if not normalized:
            return None
        return int(normalized) if normalized.isdigit() else normalized

    @staticmethod
    def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
        """Require a JSON object for nested provider payload sections."""

        if not isinstance(value, Mapping):
            raise ProviderError(
                "api-football",
                f"API-Football payload is missing object field `{field_name}`.",
            )
        return cast(Mapping[str, object], value)

    @staticmethod
    def _require_text(value: object, field_name: str) -> str:
        """Require a non-blank text value from the provider payload."""

        normalized = "" if value is None else str(value).strip()
        if not normalized:
            raise ProviderError(
                "api-football",
                f"API-Football payload is missing text field `{field_name}`.",
            )
        return normalized

    def _parse_datetime(self, value: object, *, field_name: str) -> datetime:
        """Parse required ISO datetime strings from provider responses."""

        parsed = self._optional_datetime(value, field_name=field_name)
        if parsed is None:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must contain an ISO datetime.",
            )
        return parsed

    def _optional_datetime(self, value: object, *, field_name: str) -> datetime | None:
        """Parse optional ISO datetime strings from provider responses."""

        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` is not a valid ISO datetime: {normalized}.",
            ) from exc
        if parsed.tzinfo is None:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must include timezone information.",
            )
        return parsed

    def _optional_date(self, value: object, *, field_name: str) -> date | None:
        """Parse optional ISO dates used by injury return estimates."""

        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized or normalized.lower() == "unknown":
            return None
        try:
            return date.fromisoformat(normalized)
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` is not a valid ISO date: {normalized}.",
            ) from exc

    def _coerce_decimal_odds(self, value: object, *, field_name: str) -> float:
        """Convert provider odds text into canonical decimal odds."""

        normalized = self._require_text(value, field_name)
        try:
            odds = float(normalized)
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` is not a decimal odds value: {normalized}.",
            ) from exc
        if not isfinite(odds) or odds <= 1.0:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a finite decimal odds value > 1.0.",
            )
        return odds

    @staticmethod
    def _require_positive_int(field_name: str, value: int) -> int:
        """Validate positive integer query arguments before outbound requests."""

        if value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    def _coerce_positive_int(
        self,
        value: object,
        *,
        field_name: str,
        allow_none: bool = False,
    ) -> int | None:
        """Parse positive integers from provider payloads."""

        if value is None:
            if allow_none:
                return None
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a positive integer.",
            )
        if isinstance(value, bool):
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a positive integer.",
            )
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a positive integer.",
            ) from exc
        if parsed <= 0:
            if allow_none:
                return None
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a positive integer.",
            )
        return parsed

    def _required_positive_int(self, value: object, *, field_name: str) -> int:
        """Parse one required positive integer from a provider payload."""

        parsed = self._coerce_positive_int(value, field_name=field_name)
        if parsed is None:
            raise AssertionError(f"{field_name} unexpectedly resolved to `None`.")
        return parsed

    def _coerce_non_negative_int(
        self,
        value: object,
        *,
        field_name: str,
        allow_none: bool = False,
    ) -> int | None:
        """Parse non-negative integers from provider payloads."""

        if value is None:
            if allow_none:
                return None
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a non-negative integer.",
            )
        if isinstance(value, bool):
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a non-negative integer.",
            )
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a non-negative integer.",
            ) from exc
        if parsed < 0:
            raise ProviderError(
                self.provider_name,
                f"API-Football field `{field_name}` must be a non-negative integer.",
            )
        return parsed

    def _required_non_negative_int(self, value: object, *, field_name: str) -> int:
        """Parse one required non-negative integer from a provider payload."""

        parsed = self._coerce_non_negative_int(value, field_name=field_name)
        if parsed is None:
            raise AssertionError(f"{field_name} unexpectedly resolved to `None`.")
        return parsed

    def _resolve_timezone(self, timezone: str | None) -> str:
        """Return the provider request timezone after validation."""

        candidate = (timezone or self._default_timezone).strip()
        if not candidate:
            raise ValueError("timezone must not be blank.")
        return candidate


__all__ = ["APIFootballProvider"]
