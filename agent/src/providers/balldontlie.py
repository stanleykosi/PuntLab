"""BALLDONTLIE NBA provider implementation for PuntLab's basketball ingestion.

Purpose: connect BALLDONTLIE's NBA endpoints to PuntLab's canonical provider
infrastructure for fixtures, roster data, player stat lines, and season
averages.
Scope: authenticated HTTP requests, cursor-based pagination, fail-fast
validation of upstream response shapes, and normalization into shared fixture
and stats schemas.
Dependencies: `src.providers.base` for shared HTTP behavior, `src.cache.client`
for cache TTL constants, `src.config` for credentials and sport enums, and the
shared schemas under `src.schemas`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from math import isfinite
from typing import Final, cast

from src.cache.client import API_FOOTBALL_FIXTURES_TTL_SECONDS, API_STATS_TTL_SECONDS
from src.config import SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.stats import PlayerStats, TeamStats

_API_PREFIX: Final[str] = "/nba/v1"
_TEAM_METADATA_TTL_SECONDS: Final[int] = 24 * 60 * 60
_PLAYER_PROFILE_TTL_SECONDS: Final[int] = 12 * 60 * 60
_MAX_PER_PAGE: Final[int] = 100
_PLAYER_SEASON_AVERAGE_TYPES: Final[frozenset[str]] = frozenset(
    {"general", "clutch", "defense", "shooting"}
)
_TEAM_SEASON_AVERAGE_CATEGORIES: Final[dict[str, frozenset[str] | None]] = {
    "general": frozenset(
        {"base", "advanced", "scoring", "misc", "opponent", "defense", "violations"}
    ),
    "clutch": frozenset({"base", "advanced", "misc", "scoring"}),
    "shooting": frozenset(
        {"by_zone_base", "by_zone_opponent", "5ft_range_base", "5ft_range_opponent"}
    ),
    "playtype": frozenset(
        {
            "cut",
            "handoff",
            "isolation",
            "offrebound",
            "offscreen",
            "postup",
            "prballhandler",
            "prrollman",
            "spotup",
            "transition",
            "misc",
        }
    ),
    "tracking": frozenset(
        {
            "painttouch",
            "efficiency",
            "speeddistance",
            "defense",
            "elbowtouch",
            "posttouch",
            "passing",
            "drives",
            "rebounding",
            "catchshoot",
            "pullupshot",
            "possessions",
        }
    ),
    "hustle": None,
    "shotdashboard": frozenset(
        {"overall", "pullups", "catch_and_shoot", "less_than_10_ft"}
    ),
}
_TEAM_SEASON_TYPES: Final[frozenset[str]] = frozenset({"regular", "playoffs", "ist"})
_PLAYER_SEASON_TYPES: Final[frozenset[str]] = frozenset(
    {"regular", "playoffs", "ist", "playin"}
)
_SCHEDULED_STATUS_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "scheduled",
        "pre",
        "pregame",
        "tbd",
    }
)
_LIVE_STATUS_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "1st qtr",
        "2nd qtr",
        "3rd qtr",
        "4th qtr",
        "halftime",
        "ot",
        "final/ot",
    }
)


class BallDontLieProvider(DataProvider):
    """Concrete BALLDONTLIE NBA integration for PuntLab's basketball data layer.

    Inputs:
        A shared `RateLimitedClient` plus a BALLDONTLIE API key from either the
        constructor or runtime settings.

    Outputs:
        Typed provider methods that normalize NBA fixtures and stats into
        PuntLab's shared schemas without exposing upstream response details to
        downstream pipeline stages.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        api_key: str | None = None,
    ) -> None:
        """Initialize the provider with validated credentials.

        Args:
            client: Shared `RateLimitedClient` used for cached and retried HTTP
                requests.
            api_key: Optional explicit API key. When omitted, the provider
                falls back to `BALLDONTLIE_API_KEY` from settings.

        Raises:
            ValueError: If no usable API key is available.
        """

        super().__init__(client)
        resolved_api_key = (
            api_key or get_settings().data_providers.balldontlie_api_key or ""
        ).strip()
        if not resolved_api_key:
            raise ValueError(
                "BALLDONTLIE requires `BALLDONTLIE_API_KEY` or an explicit `api_key`."
            )

        self._api_key = resolved_api_key

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in cache keys and logs."""

        return "balldontlie"

    @property
    def base_url(self) -> str:
        """Return the BALLDONTLIE production base URL."""

        return "https://api.balldontlie.io"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return PuntLab's configured BALLDONTLIE request budget.

        The local product spec currently targets 30 requests per minute for the
        NBA provider integration, so the shared rate-limiter enforces that
        canonical ceiling.
        """

        return RateLimitPolicy(limit=30, window_seconds=60)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the canonical authentication and content negotiation headers."""

        return {
            "Authorization": self._api_key,
            "Accept": "application/json",
        }

    async def fetch_games_by_date(
        self,
        *,
        run_date: date,
        team_ids: Sequence[int] | None = None,
        seasons: Sequence[int] | None = None,
        postseason: bool | None = None,
        per_page: int = _MAX_PER_PAGE,
        max_pages: int | None = 1,
    ) -> list[NormalizedFixture]:
        """Fetch NBA games for one calendar date and normalize them to fixtures.

        Args:
            run_date: Date to query in the provider API.
            team_ids: Optional team filters scoped to BALLDONTLIE team IDs.
            seasons: Optional NBA season-year filters such as `2025`.
            postseason: Optional playoff flag filter.
            per_page: Page size requested from BALLDONTLIE.
            max_pages: Maximum number of cursor pages to fetch. Use `None` to
                exhaust all pages.

        Returns:
            A list of normalized NBA fixtures for the requested date.
        """

        params = self._build_sequence_params(
            ("dates[]", (run_date.isoformat(),)),
            ("team_ids[]", self._normalize_int_sequence("team_ids", team_ids)),
            ("seasons[]", self._normalize_int_sequence("seasons", seasons)),
        )
        if postseason is not None:
            params.append(("postseason", "true" if postseason else "false"))

        game_rows = await self._fetch_paginated_collection(
            f"{_API_PREFIX}/games",
            params=params,
            cache_ttl_seconds=API_FOOTBALL_FIXTURES_TTL_SECONDS,
            per_page=per_page,
            max_pages=max_pages,
        )
        return [self._normalize_game(game_row) for game_row in game_rows]

    async def fetch_teams(
        self,
        *,
        conference: str | None = None,
        division: str | None = None,
    ) -> list[TeamStats]:
        """Fetch NBA teams and normalize them into minimal `TeamStats` rows.

        Args:
            conference: Optional conference filter such as `East`.
            division: Optional division filter such as `Pacific`.

        Returns:
            A list of canonical team metadata snapshots.
        """

        params: list[tuple[str, str]] = []
        if conference is not None:
            params.append(("conference", self._require_text(conference, "conference")))
        if division is not None:
            params.append(("division", self._require_text(division, "division")))

        payload = await self._fetch_json_mapping(
            f"{_API_PREFIX}/teams",
            params=params,
            cache_ttl_seconds=_TEAM_METADATA_TTL_SECONDS,
        )
        fetched_at = self._now()
        return [
            self._normalize_team_metadata(team_row, fetched_at=fetched_at)
            for team_row in self._extract_data_list(payload, field_name="teams.data")
        ]

    async def fetch_players(
        self,
        *,
        search: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        team_ids: Sequence[int] | None = None,
        player_ids: Sequence[int] | None = None,
        per_page: int = _MAX_PER_PAGE,
        max_pages: int | None = 1,
    ) -> list[PlayerStats]:
        """Fetch NBA players and normalize them into canonical player bundles.

        Args:
            search: Optional free-text player search.
            first_name: Optional exact first-name filter.
            last_name: Optional exact last-name filter.
            team_ids: Optional team filters scoped to BALLDONTLIE team IDs.
            player_ids: Optional explicit player ID filters.
            per_page: Page size requested from BALLDONTLIE.
            max_pages: Maximum number of cursor pages to fetch. Use `None` to
                exhaust all pages.

        Returns:
            A list of minimal canonical player stat rows preserving player and
            team identity metadata.
        """

        params = self._build_sequence_params(
            ("team_ids[]", self._normalize_int_sequence("team_ids", team_ids)),
            ("player_ids[]", self._normalize_int_sequence("player_ids", player_ids)),
        )
        if search is not None:
            params.append(("search", self._require_text(search, "search")))
        if first_name is not None:
            params.append(("first_name", self._require_text(first_name, "first_name")))
        if last_name is not None:
            params.append(("last_name", self._require_text(last_name, "last_name")))

        player_rows = await self._fetch_paginated_collection(
            f"{_API_PREFIX}/players",
            params=params,
            cache_ttl_seconds=_PLAYER_PROFILE_TTL_SECONDS,
            per_page=per_page,
            max_pages=max_pages,
        )
        fetched_at = self._now()
        return [
            self._normalize_player_profile(player_row, fetched_at=fetched_at)
            for player_row in player_rows
        ]

    async def fetch_box_scores_by_date(
        self,
        *,
        run_date: date,
    ) -> list[PlayerStats]:
        """Fetch one day's NBA box scores and flatten them into player stat rows.

        Args:
            run_date: Calendar date for the requested game slate.

        Returns:
            A flat list of canonical per-player game stat bundles.

        Raises:
            ProviderError: If the upstream response shape is invalid or the
                account tier does not include box-score access.
        """

        payload = await self._fetch_json_mapping(
            f"{_API_PREFIX}/box_scores",
            params=[("date", run_date.isoformat())],
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
            endpoint_label="box scores",
            tier_guidance=(
                "Verify that the supplied NBA API key is valid and that the "
                "account tier includes BOX SCORES access."
            ),
        )

        fetched_at = self._now()
        player_rows: list[PlayerStats] = []
        for box_score_row in self._extract_data_list(payload, field_name="box_scores.data"):
            box_score_mapping = self._require_mapping(box_score_row, "box_scores.data[]")
            player_rows.extend(
                self._normalize_box_score_players(
                    box_score_mapping,
                    team_side="home_team",
                    fetched_at=fetched_at,
                )
            )
            player_rows.extend(
                self._normalize_box_score_players(
                    box_score_mapping,
                    team_side="visitor_team",
                    fetched_at=fetched_at,
                )
            )
        return player_rows

    async def fetch_season_averages(
        self,
        *,
        season: int,
        stats_type: str,
        average_type: str = "general",
        season_type: str = "regular",
        player_ids: Sequence[int] | None = None,
        per_page: int = _MAX_PER_PAGE,
        max_pages: int | None = 1,
    ) -> list[PlayerStats]:
        """Fetch NBA player season averages and normalize them to `PlayerStats`.

        Args:
            season: NBA season start year such as `2025`.
            average_type: Provider path category such as `general` or
                `shooting`.
            stats_type: Category-specific stats subtype accepted by the
                upstream provider.
            season_type: Season partition such as `regular` or `playoffs`.
            player_ids: Optional explicit player filters.
            per_page: Page size requested from BALLDONTLIE.
            max_pages: Maximum number of cursor pages to fetch. Use `None` to
                exhaust all pages.

        Returns:
            A list of canonical player stat bundles representing season
            averages.

        Raises:
            ValueError: If the category or season type is invalid.
            ProviderError: If the provider rejects the request or the response
                shape is invalid.
        """

        normalized_average_type = self._normalize_season_average_type(average_type)
        normalized_season_type = self._normalize_choice(
            "season_type",
            season_type,
            _PLAYER_SEASON_TYPES,
        )
        params = self._build_sequence_params(
            ("player_ids[]", self._normalize_int_sequence("player_ids", player_ids)),
        )
        params.extend(
            [
                ("season", str(self._require_positive_int("season", season))),
                ("season_type", normalized_season_type),
                ("type", self._require_text(stats_type, "stats_type")),
            ]
        )

        average_rows = await self._fetch_paginated_collection(
            f"{_API_PREFIX}/season_averages/{normalized_average_type}",
            params=params,
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
            per_page=per_page,
            max_pages=max_pages,
            endpoint_label="season averages",
            tier_guidance=(
                "Verify that the supplied NBA API key is valid and that the "
                "account tier includes SEASON AVERAGES access."
            ),
        )
        fetched_at = self._now()
        return [
            self._normalize_player_season_average(average_row, fetched_at=fetched_at)
            for average_row in average_rows
        ]

    async def fetch_team_season_averages(
        self,
        *,
        season: int,
        category: str = "general",
        season_type: str = "regular",
        stats_type: str | None = "base",
        team_ids: Sequence[int] | None = None,
        per_page: int = _MAX_PER_PAGE,
        max_pages: int | None = 1,
    ) -> list[TeamStats]:
        """Fetch NBA team season averages and normalize them to `TeamStats`.

        Args:
            season: NBA season start year such as `2025`.
            category: Team season-average category such as `general` or
                `tracking`.
            season_type: Season partition such as `regular` or `playoffs`.
            stats_type: Category-specific stats subtype. Omit only for
                `hustle`, which does not require a subtype.
            team_ids: Optional explicit team filters.
            per_page: Page size requested from BALLDONTLIE.
            max_pages: Maximum number of cursor pages to fetch. Use `None` to
                exhaust all pages.

        Returns:
            A list of canonical team stat rows.
        """

        normalized_category = self._normalize_team_season_average_category(category)
        normalized_season_type = self._normalize_choice(
            "season_type",
            season_type,
            _TEAM_SEASON_TYPES,
        )
        normalized_stats_type = self._normalize_team_season_average_stats_type(
            category=normalized_category,
            stats_type=stats_type,
        )

        params = self._build_sequence_params(
            ("team_ids[]", self._normalize_int_sequence("team_ids", team_ids)),
        )
        params.extend(
            [
                ("season", str(self._require_positive_int("season", season))),
                ("season_type", normalized_season_type),
            ]
        )
        if normalized_stats_type is not None:
            params.append(("type", normalized_stats_type))

        team_rows = await self._fetch_paginated_collection(
            f"{_API_PREFIX}/team_season_averages/{normalized_category}",
            params=params,
            cache_ttl_seconds=API_STATS_TTL_SECONDS,
            per_page=per_page,
            max_pages=max_pages,
            endpoint_label="team season averages",
            tier_guidance=(
                "Verify that the supplied NBA API key is valid and that the "
                "account tier includes TEAM SEASON AVERAGES access."
            ),
        )
        fetched_at = self._now()
        return [
            self._normalize_team_season_average(team_row, fetched_at=fetched_at)
            for team_row in team_rows
        ]

    async def _fetch_paginated_collection(
        self,
        path: str,
        *,
        params: Sequence[tuple[str, str]],
        cache_ttl_seconds: int,
        per_page: int,
        max_pages: int | None,
        endpoint_label: str | None = None,
        tier_guidance: str | None = None,
    ) -> list[Mapping[str, object]]:
        """Fetch one cursor-paginated BALLDONTLIE collection endpoint.

        Args:
            path: Provider API path.
            params: Base request parameters excluding pagination fields.
            cache_ttl_seconds: Cache TTL applied to every page.
            per_page: Requested page size.
            max_pages: Maximum number of pages to fetch, or `None` for all.
            endpoint_label: Human-readable endpoint name for improved errors.
            tier_guidance: Optional recovery guidance used for `401` failures.

        Returns:
            A flat list of provider response rows.
        """

        validated_per_page = self._validate_per_page(per_page)
        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be a positive integer or `None`.")

        collected_rows: list[Mapping[str, object]] = []
        cursor: int | None = None
        pages_fetched = 0

        while True:
            request_params = list(params)
            request_params.append(("per_page", str(validated_per_page)))
            if cursor is not None:
                request_params.append(("cursor", str(cursor)))

            payload = await self._fetch_json_mapping(
                path,
                params=request_params,
                cache_ttl_seconds=cache_ttl_seconds,
                endpoint_label=endpoint_label,
                tier_guidance=tier_guidance,
            )
            collected_rows.extend(self._extract_data_list(payload, field_name="data"))
            pages_fetched += 1
            if max_pages is not None and pages_fetched >= max_pages:
                break

            cursor = self._extract_next_cursor(payload)
            if cursor is None:
                break

        return collected_rows

    async def _fetch_json_mapping(
        self,
        path: str,
        *,
        params: Sequence[tuple[str, str]],
        cache_ttl_seconds: int,
        endpoint_label: str | None = None,
        tier_guidance: str | None = None,
    ) -> Mapping[str, object]:
        """Fetch one JSON object payload and validate its top-level shape."""

        try:
            response = await self.fetch(
                "GET",
                path,
                params=list(params),
                cache_ttl_seconds=cache_ttl_seconds,
            )
        except ProviderError as exc:
            if exc.status_code == 401 and tier_guidance is not None:
                label = endpoint_label or path
                raise ProviderError(
                    self.provider_name,
                    (
                        f"BALLDONTLIE rejected the {label} request. "
                        f"{tier_guidance}"
                    ),
                    status_code=exc.status_code,
                    cause=exc,
                ) from exc
            raise

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"BALLDONTLIE returned invalid JSON for path '{path}'.",
                cause=exc,
            ) from exc

        if not isinstance(payload, Mapping):
            raise ProviderError(
                self.provider_name,
                f"BALLDONTLIE must return a JSON object for path '{path}'.",
            )

        return cast(Mapping[str, object], payload)

    def _normalize_game(self, game_row: Mapping[str, object]) -> NormalizedFixture:
        """Normalize one BALLDONTLIE game row into a canonical fixture."""

        game_id = self._require_positive_int_from_value(game_row.get("id"), field_name="game.id")
        kickoff = self._parse_datetime(game_row.get("datetime"), field_name="game.datetime")
        home_team = self._require_mapping(game_row.get("home_team"), "game.home_team")
        visitor_team = self._require_mapping(game_row.get("visitor_team"), "game.visitor_team")

        return NormalizedFixture(
            sportradar_id=None,
            home_team=self._team_display_name(home_team),
            away_team=self._team_display_name(visitor_team),
            competition="NBA",
            sport=SportName.BASKETBALL,
            kickoff=kickoff,
            source_provider=self.provider_name,
            source_id=str(game_id),
            country="United States",
            league="nba",
            home_team_id=str(
                self._require_positive_int_from_value(
                    home_team.get("id"),
                    field_name="home_team.id",
                )
            ),
            away_team_id=str(
                self._require_positive_int_from_value(
                    visitor_team.get("id"),
                    field_name="visitor_team.id",
                )
            ),
            venue=None,
            status=self._normalize_game_status(game_row),
        )

    def _normalize_game_status(self, game_row: Mapping[str, object]) -> FixtureStatus:
        """Map BALLDONTLIE game status fields into PuntLab's fixture statuses."""

        postponed = game_row.get("postponed")
        if isinstance(postponed, bool) and postponed:
            return FixtureStatus.POSTPONED

        raw_status = self._normalize_optional_text(game_row.get("status"))
        normalized_status = raw_status.casefold() if raw_status is not None else ""
        if normalized_status == "final":
            return FixtureStatus.FINISHED
        if normalized_status in _LIVE_STATUS_TOKENS or normalized_status.endswith(" qtr"):
            return FixtureStatus.LIVE
        if normalized_status in _SCHEDULED_STATUS_TOKENS:
            return FixtureStatus.SCHEDULED

        # BALLDONTLIE uses pregame clock strings such as `7:00 pm ET` for
        # scheduled games. We treat any unrecognized status containing `ET`
        # before tip-off as scheduled.
        if "et" in normalized_status:
            return FixtureStatus.SCHEDULED

        period_value = game_row.get("period")
        if isinstance(period_value, int) and period_value > 0:
            return FixtureStatus.LIVE

        return FixtureStatus.SCHEDULED

    def _normalize_team_metadata(
        self,
        team_row: Mapping[str, object],
        *,
        fetched_at: datetime,
    ) -> TeamStats:
        """Normalize one team metadata row into a minimal `TeamStats` object."""

        team_id = self._require_positive_int_from_value(team_row.get("id"), field_name="team.id")
        return TeamStats(
            team_id=str(team_id),
            team_name=self._team_display_name(team_row),
            sport=SportName.BASKETBALL,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            competition="NBA",
        )

    def _normalize_team_season_average(
        self,
        team_row: Mapping[str, object],
        *,
        fetched_at: datetime,
    ) -> TeamStats:
        """Normalize one BALLDONTLIE team season-average row into `TeamStats`."""

        team_mapping = self._require_mapping(team_row.get("team"), "team_season_average.team")
        stats_mapping = self._require_mapping(team_row.get("stats"), "team_season_average.stats")
        season_year = self._require_positive_int_from_value(
            team_row.get("season"),
            field_name="team_season_average.season",
        )

        matches_played = self._metric_as_int(stats_mapping.get("gp"))
        wins = self._metric_as_int(stats_mapping.get("w"))
        losses = self._metric_as_int(stats_mapping.get("l"))
        avg_points_scored = self._metric_as_float(stats_mapping.get("pts"))
        avg_points_allowed = self._metric_as_float(
            stats_mapping.get("opp_pts") or stats_mapping.get("points_against")
        )
        advanced_metrics = self._extract_numeric_metrics(stats_mapping)

        goals_for = (
            int(round(avg_points_scored * matches_played))
            if avg_points_scored is not None and matches_played > 0
            else 0
        )
        goals_against = (
            int(round(avg_points_allowed * matches_played))
            if avg_points_allowed is not None and matches_played > 0
            else 0
        )

        return TeamStats(
            team_id=str(
                self._require_positive_int_from_value(team_mapping.get("id"), field_name="team.id")
            ),
            team_name=self._team_display_name(team_mapping),
            sport=SportName.BASKETBALL,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            competition="NBA",
            season=self._format_nba_season(season_year),
            matches_played=matches_played,
            wins=wins,
            draws=0,
            losses=losses,
            goals_for=goals_for,
            goals_against=goals_against,
            clean_sheets=0,
            home_wins=self._metric_as_int(
                stats_mapping.get("home_w") or stats_mapping.get("home_wins")
            ),
            away_wins=self._metric_as_int(
                stats_mapping.get("road_w") or stats_mapping.get("away_wins")
            ),
            avg_goals_scored=avg_points_scored,
            avg_goals_conceded=avg_points_allowed,
            advanced_metrics=advanced_metrics,
        )

    def _normalize_player_profile(
        self,
        player_row: Mapping[str, object],
        *,
        fetched_at: datetime,
    ) -> PlayerStats:
        """Normalize one player metadata row into a minimal `PlayerStats` object."""

        player_id = self._require_positive_int_from_value(
            player_row.get("id"),
            field_name="player.id",
        )
        team_mapping = self._optional_mapping(player_row.get("team"))
        team_id = self._resolve_player_team_id(player_row, team_mapping)

        return PlayerStats(
            player_id=str(player_id),
            player_name=self._player_display_name(player_row),
            team_id=str(team_id),
            sport=SportName.BASKETBALL,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            team_name=self._team_display_name(team_mapping) if team_mapping is not None else None,
            competition="NBA",
            position=self._normalize_optional_text(player_row.get("position")),
        )

    def _normalize_box_score_players(
        self,
        box_score_row: Mapping[str, object],
        *,
        team_side: str,
        fetched_at: datetime,
    ) -> list[PlayerStats]:
        """Flatten one box-score team section into canonical player stat rows."""

        team_wrapper = self._require_mapping(
            box_score_row.get(team_side),
            f"box_score.{team_side}",
        )
        team_mapping = self._require_mapping(
            team_wrapper.get("team"),
            f"box_score.{team_side}.team",
        )
        players = self._require_list(team_wrapper.get("players"), f"box_score.{team_side}.players")
        return [
            self._normalize_box_score_player(
                self._require_mapping(player_row, f"box_score.{team_side}.players[]"),
                team_mapping=team_mapping,
                box_score_row=box_score_row,
                fetched_at=fetched_at,
            )
            for player_row in players
        ]

    def _normalize_box_score_player(
        self,
        player_row: Mapping[str, object],
        *,
        team_mapping: Mapping[str, object],
        box_score_row: Mapping[str, object],
        fetched_at: datetime,
    ) -> PlayerStats:
        """Normalize one flattened box-score player row into `PlayerStats`."""

        player_mapping = self._require_mapping(player_row.get("player"), "box_score.player")
        season_year = self._require_positive_int_from_value(
            box_score_row.get("season"),
            field_name="box_score.season",
        )
        metrics = self._extract_numeric_metrics(player_row, excluded_keys={"player"})
        game_id = box_score_row.get("id") or box_score_row.get("game_id")
        if game_id is not None:
            metrics["game_id"] = float(
                self._require_positive_int_from_value(game_id, field_name="box_score.game_id")
            )

        return PlayerStats(
            player_id=str(
                self._require_positive_int_from_value(
                    player_mapping.get("id"),
                    field_name="player.id",
                )
            ),
            player_name=self._player_display_name(player_mapping),
            team_id=str(
                self._require_positive_int_from_value(team_mapping.get("id"), field_name="team.id")
            ),
            sport=SportName.BASKETBALL,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            team_name=self._team_display_name(team_mapping),
            competition="NBA",
            season=self._format_nba_season(season_year),
            position=self._normalize_optional_text(player_mapping.get("position")),
            appearances=1,
            starts=0,
            minutes_played=self._parse_minutes_played(player_row.get("min")),
            metrics=metrics,
        )

    def _normalize_player_season_average(
        self,
        average_row: Mapping[str, object],
        *,
        fetched_at: datetime,
    ) -> PlayerStats:
        """Normalize one BALLDONTLIE player season-average row into `PlayerStats`."""

        player_mapping = self._require_mapping(average_row.get("player"), "season_average.player")
        stats_mapping = self._require_mapping(average_row.get("stats"), "season_average.stats")
        season_year = self._require_positive_int_from_value(
            average_row.get("season"),
            field_name="season_average.season",
        )
        appearances = self._metric_as_int(
            stats_mapping.get("gp") or stats_mapping.get("games_played")
        )
        average_minutes = self._metric_as_float(stats_mapping.get("min"))
        metrics = self._extract_numeric_metrics(stats_mapping)
        player_team = self._optional_mapping(player_mapping.get("team"))

        return PlayerStats(
            player_id=str(
                self._require_positive_int_from_value(
                    player_mapping.get("id"),
                    field_name="player.id",
                )
            ),
            player_name=self._player_display_name(player_mapping),
            team_id=str(self._resolve_player_team_id(player_mapping, player_team)),
            sport=SportName.BASKETBALL,
            source_provider=self.provider_name,
            fetched_at=fetched_at,
            team_name=self._team_display_name(player_team) if player_team is not None else None,
            competition="NBA",
            season=self._format_nba_season(season_year),
            position=self._normalize_optional_text(player_mapping.get("position")),
            appearances=appearances,
            starts=0,
            minutes_played=self._estimate_total_minutes(
                appearances=appearances,
                average_minutes=average_minutes,
            ),
            metrics=metrics,
        )

    @staticmethod
    def _build_sequence_params(
        *pairs: tuple[str, Sequence[int] | Sequence[str] | None],
    ) -> list[tuple[str, str]]:
        """Expand sequence-valued request params into repeated query tuples."""

        params: list[tuple[str, str]] = []
        for key, values in pairs:
            if values is None:
                continue
            for value in values:
                params.append((key, str(value)))
        return params

    @staticmethod
    def _format_nba_season(season_year: int) -> str:
        """Format one NBA season year as `YYYY-YY`."""

        return f"{season_year}-{(season_year + 1) % 100:02d}"

    @staticmethod
    def _estimate_total_minutes(*, appearances: int, average_minutes: float | None) -> int:
        """Estimate total minutes from season-average appearances and `min`."""

        if appearances <= 0 or average_minutes is None or not isfinite(average_minutes):
            return 0
        return max(0, int(round(appearances * average_minutes)))

    def _extract_data_list(
        self,
        payload: Mapping[str, object],
        *,
        field_name: str,
    ) -> list[Mapping[str, object]]:
        """Extract and validate the top-level BALLDONTLIE `data` array."""

        data = payload.get("data")
        raw_rows = self._require_list(data, field_name)
        return [self._require_mapping(item, f"{field_name}[]") for item in raw_rows]

    def _extract_next_cursor(self, payload: Mapping[str, object]) -> int | None:
        """Extract one optional pagination cursor from a provider payload."""

        meta_mapping = self._optional_mapping(payload.get("meta"))
        if meta_mapping is None:
            return None

        next_cursor = meta_mapping.get("next_cursor")
        if next_cursor is None:
            return None
        if not isinstance(next_cursor, int) or next_cursor <= 0:
            raise ProviderError(
                self.provider_name,
                "BALLDONTLIE returned an invalid `meta.next_cursor` value.",
            )
        return next_cursor

    def _normalize_season_average_type(self, value: str) -> str:
        """Validate the BALLDONTLIE player season-average path category."""

        return self._normalize_choice(
            "average_type",
            value,
            _PLAYER_SEASON_AVERAGE_TYPES,
        )

    def _normalize_team_season_average_category(self, value: str) -> str:
        """Validate the BALLDONTLIE team season-average category."""

        normalized = self._require_text(value, "category").lower()
        if normalized not in _TEAM_SEASON_AVERAGE_CATEGORIES:
            raise ValueError(
                "category must be one of "
                f"{sorted(_TEAM_SEASON_AVERAGE_CATEGORIES)}."
            )
        return normalized

    def _normalize_team_season_average_stats_type(
        self,
        *,
        category: str,
        stats_type: str | None,
    ) -> str | None:
        """Validate the category-specific `type` for team season averages."""

        allowed_values = _TEAM_SEASON_AVERAGE_CATEGORIES[category]
        if allowed_values is None:
            if stats_type is None:
                return None
            normalized = self._require_text(stats_type, "stats_type").lower()
            return normalized

        if stats_type is None:
            raise ValueError(
                f"stats_type is required for category '{category}'."
            )

        normalized = self._require_text(stats_type, "stats_type").lower()
        if normalized not in allowed_values:
            raise ValueError(
                f"stats_type must be one of {sorted(allowed_values)} for category '{category}'."
            )
        return normalized

    @staticmethod
    def _normalize_choice(field_name: str, value: str, allowed: frozenset[str]) -> str:
        """Normalize one string choice against a fixed set of allowed values."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank.")
        if normalized not in allowed:
            raise ValueError(f"{field_name} must be one of {sorted(allowed)}.")
        return normalized

    def _normalize_int_sequence(
        self,
        field_name: str,
        values: Sequence[int] | None,
    ) -> tuple[int, ...] | None:
        """Validate one optional integer filter sequence."""

        if values is None:
            return None
        normalized: list[int] = []
        for value in values:
            normalized.append(self._require_positive_int(field_name, value))
        return tuple(normalized)

    @staticmethod
    def _validate_per_page(value: int) -> int:
        """Validate one BALLDONTLIE page size."""

        if value <= 0:
            raise ValueError("per_page must be a positive integer.")
        if value > _MAX_PER_PAGE:
            raise ValueError(f"per_page must not exceed {_MAX_PER_PAGE}.")
        return value

    @staticmethod
    def _parse_minutes_played(value: object) -> int:
        """Parse BALLDONTLIE minute strings such as `30` or `30:45`."""

        if value is None:
            return 0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not isfinite(float(value)) or float(value) < 0:
                raise ProviderError("balldontlie", "`min` must be a non-negative finite value.")
            return int(round(float(value)))
        if not isinstance(value, str):
            raise ProviderError("balldontlie", "`min` must be supplied as text or a number.")

        normalized = value.strip()
        if not normalized:
            return 0
        if ":" in normalized:
            minute_fragment, second_fragment = normalized.split(":", maxsplit=1)
            if not minute_fragment.isdigit() or not second_fragment.isdigit():
                raise ProviderError("balldontlie", f"Unsupported minute value '{value}'.")
            total_seconds = (int(minute_fragment) * 60) + int(second_fragment)
            return int(round(total_seconds / 60))

        try:
            numeric_minutes = float(normalized)
        except ValueError as exc:
            raise ProviderError("balldontlie", f"Unsupported minute value '{value}'.") from exc
        if not isfinite(numeric_minutes) or numeric_minutes < 0:
            raise ProviderError("balldontlie", "`min` must be a non-negative finite value.")
        return int(round(numeric_minutes))

    def _extract_numeric_metrics(
        self,
        metrics_row: Mapping[str, object],
        *,
        excluded_keys: set[str] | None = None,
    ) -> dict[str, float]:
        """Extract every finite numeric field from a provider stat mapping."""

        metrics: dict[str, float] = {}
        for key, raw_value in metrics_row.items():
            if excluded_keys is not None and key in excluded_keys:
                continue
            numeric_value = self._metric_as_float(raw_value)
            if numeric_value is None:
                continue
            metrics[self._require_text(key, "metric_key")] = numeric_value
        return metrics

    @staticmethod
    def _metric_as_int(value: object) -> int:
        """Convert one provider metric value into an integer when possible."""

        numeric_value = BallDontLieProvider._metric_as_float(value)
        if numeric_value is None:
            return 0
        return int(round(numeric_value))

    @staticmethod
    def _metric_as_float(value: object) -> float | None:
        """Convert one provider metric value into a finite float when possible."""

        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric_value = float(value)
            if not isfinite(numeric_value):
                raise ProviderError("balldontlie", "Provider metrics must be finite numbers.")
            return numeric_value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                numeric_value = float(normalized)
            except ValueError:
                return None
            if not isfinite(numeric_value):
                raise ProviderError("balldontlie", "Provider metrics must be finite numbers.")
            return numeric_value
        return None

    def _resolve_player_team_id(
        self,
        player_row: Mapping[str, object],
        team_mapping: Mapping[str, object] | None,
    ) -> int:
        """Resolve a required player team identifier from the available fields."""

        if team_mapping is not None:
            return self._require_positive_int_from_value(
                team_mapping.get("id"),
                field_name="team.id",
            )
        return self._require_positive_int_from_value(
            player_row.get("team_id"),
            field_name="player.team_id",
        )

    @staticmethod
    def _team_display_name(team_row: Mapping[str, object] | None) -> str:
        """Build the canonical display name for one BALLDONTLIE team."""

        if team_row is None:
            raise ProviderError("balldontlie", "Expected a team mapping but received `None`.")

        full_name = BallDontLieProvider._normalize_optional_text(team_row.get("full_name"))
        if full_name is not None:
            return full_name

        city = BallDontLieProvider._normalize_optional_text(team_row.get("city"))
        name = BallDontLieProvider._normalize_optional_text(team_row.get("name"))
        if city is not None and name is not None:
            return f"{city} {name}"
        if name is not None:
            return name
        raise ProviderError("balldontlie", "BALLDONTLIE team rows must include a display name.")

    @staticmethod
    def _player_display_name(player_row: Mapping[str, object]) -> str:
        """Build the canonical display name for one BALLDONTLIE player."""

        first_name = BallDontLieProvider._normalize_optional_text(player_row.get("first_name"))
        last_name = BallDontLieProvider._normalize_optional_text(player_row.get("last_name"))
        full_name = " ".join(part for part in (first_name, last_name) if part is not None).strip()
        if full_name:
            return full_name
        raise ProviderError("balldontlie", "BALLDONTLIE player rows must include a name.")

    def _parse_datetime(self, value: object, *, field_name: str) -> datetime:
        """Parse one BALLDONTLIE ISO-8601 timestamp into a timezone-aware value."""

        text_value = self._require_text(value, field_name)
        try:
            parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"{field_name} must be a valid ISO-8601 datetime string.",
                cause=exc,
            ) from exc
        if parsed.tzinfo is None:
            raise ProviderError(
                self.provider_name,
                f"{field_name} must include timezone information.",
            )
        return parsed

    @staticmethod
    def _now() -> datetime:
        """Return the current timezone-aware timestamp for provider snapshots."""

        return datetime.now(get_settings().timezone).astimezone(UTC)

    @staticmethod
    def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
        """Require one provider field to be a JSON object."""

        if not isinstance(value, Mapping):
            raise ProviderError("balldontlie", f"{field_name} must be a JSON object.")
        return cast(Mapping[str, object], value)

    @classmethod
    def _optional_mapping(cls, value: object) -> Mapping[str, object] | None:
        """Return one optional JSON object or `None`."""

        if value is None:
            return None
        return cls._require_mapping(value, "value")

    @staticmethod
    def _require_list(value: object, field_name: str) -> list[object]:
        """Require one provider field to be a JSON array."""

        if not isinstance(value, list):
            raise ProviderError("balldontlie", f"{field_name} must be a JSON array.")
        return value

    @staticmethod
    def _require_text(value: object, field_name: str) -> str:
        """Require one provider field to be a non-blank string."""

        if not isinstance(value, str):
            raise ProviderError("balldontlie", f"{field_name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ProviderError("balldontlie", f"{field_name} must not be blank.")
        return normalized

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        """Normalize one optional provider text field."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ProviderError("balldontlie", "Optional text fields must be strings when present.")
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _require_positive_int(field_name: str, value: int) -> int:
        """Require one caller-supplied integer argument to be positive."""

        if value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    @staticmethod
    def _require_positive_int_from_value(value: object, *, field_name: str) -> int:
        """Require one provider value to be a positive integer."""

        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ProviderError("balldontlie", f"{field_name} must be a positive integer.")
        return value


__all__ = ["BallDontLieProvider"]
