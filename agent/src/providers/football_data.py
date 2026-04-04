"""Football-Data.org provider implementation for PuntLab's soccer fallback layer.

Purpose: connect Football-Data.org's v4 API to PuntLab's canonical provider
infrastructure for fixture, standings, and team metadata fallback coverage.
Scope: authenticated HTTP requests, competition-scoped response parsing, and
normalization into shared fixture and team-stat schemas.
Dependencies: `src.providers.base` for caching and rate limiting, `src.config`
for provider credentials and sport enums, and the shared schemas under
`src.schemas`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Final, cast

from src.config import SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.stats import TeamStats

_COMPETITION_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]{2,8}$")
_FIXTURE_STATUS_MAP: Final[dict[str, FixtureStatus]] = {
    "SCHEDULED": FixtureStatus.SCHEDULED,
    "TIMED": FixtureStatus.SCHEDULED,
    "IN_PLAY": FixtureStatus.LIVE,
    "PAUSED": FixtureStatus.LIVE,
    "LIVE": FixtureStatus.LIVE,
    "FINISHED": FixtureStatus.FINISHED,
    "POSTPONED": FixtureStatus.POSTPONED,
    "SUSPENDED": FixtureStatus.CANCELLED,
    "CANCELLED": FixtureStatus.CANCELLED,
    "AWARDED": FixtureStatus.CANCELLED,
}
_FIXTURES_TTL_SECONDS: Final[int] = 2 * 60 * 60
_STANDINGS_TTL_SECONDS: Final[int] = 6 * 60 * 60
_TEAM_DATA_TTL_SECONDS: Final[int] = 24 * 60 * 60


class FootballDataProvider(DataProvider):
    """Concrete Football-Data.org integration for soccer fallback ingestion.

    Inputs:
        A shared `RateLimitedClient` plus a Football-Data.org API token from
        either the constructor or runtime settings.

    Outputs:
        Typed provider methods that return normalized PuntLab schemas for
        fixtures, standings, and competition team metadata.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        api_token: str | None = None,
    ) -> None:
        """Initialize the provider with a validated Football-Data.org token.

        Args:
            client: Shared `RateLimitedClient` used for cached and retried HTTP
                requests.
            api_token: Optional explicit token. When omitted, the provider
                falls back to `FOOTBALL_DATA_API_KEY` from settings.

        Raises:
            ValueError: If no usable API token is available.
        """

        super().__init__(client)
        resolved_token = (
            api_token or get_settings().data_providers.football_data_api_key or ""
        ).strip()
        if not resolved_token:
            raise ValueError(
                "Football-Data.org requires `FOOTBALL_DATA_API_KEY` or an explicit `api_token`."
            )
        self._api_token = resolved_token

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in cache keys and logs."""

        return "football-data"

    @property
    def base_url(self) -> str:
        """Return the current official Football-Data.org v4 base URL."""

        return "https://api.football-data.org/v4"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return the documented free-tier limit of ten requests per minute."""

        return RateLimitPolicy(limit=10, window_seconds=60)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the canonical authentication and content negotiation headers."""

        return {
            "X-Auth-Token": self._api_token,
            "Accept": "application/json",
        }

    async def fetch_fixtures_by_date(
        self,
        *,
        run_date: date,
        competition_code: str,
        season: int | None = None,
    ) -> list[NormalizedFixture]:
        """Fetch one competition's fixtures for a specific calendar date.

        Args:
            run_date: Date to query in the provider API.
            competition_code: Football-Data.org competition code such as `PL`
                or `FL1`.
            season: Optional season year override supported by the endpoint.

        Returns:
            A list of normalized fixtures for the requested competition date.
        """

        normalized_code = self._normalize_competition_code(competition_code)
        params: dict[str, object] = {
            "dateFrom": run_date.isoformat(),
            # Football-Data.org documents `dateTo` as an exclusive upper bound,
            # so querying a single day uses the following day as the boundary.
            "dateTo": (run_date + timedelta(days=1)).isoformat(),
        }
        if season is not None:
            params["season"] = self._require_positive_int("season", season)

        payload = await self._fetch_json(
            f"/competitions/{normalized_code}/matches",
            params=params,
            cache_ttl_seconds=_FIXTURES_TTL_SECONDS,
        )

        competition_data = self._require_mapping(payload.get("competition"), "competition")
        area_data = self._optional_mapping(payload.get("area"))
        return [
            self._normalize_fixture(
                self._require_mapping(match_payload, "matches[]"),
                competition_data=competition_data,
                area_data=area_data,
            )
            for match_payload in self._require_list(payload.get("matches"), "matches")
        ]

    async def fetch_standings(
        self,
        *,
        competition_code: str,
        season: int | None = None,
        matchday: int | None = None,
        as_of_date: date | None = None,
    ) -> list[TeamStats]:
        """Fetch normalized standings snapshots for one competition.

        Args:
            competition_code: Football-Data.org competition code such as `PL`.
            season: Optional season year supported by the endpoint.
            matchday: Optional historical matchday filter.
            as_of_date: Optional historical table snapshot date.

        Returns:
            A list of canonical `TeamStats` rows merged from the provider's
            `TOTAL`, `HOME`, and `AWAY` standings tables when available.
        """

        normalized_code = self._normalize_competition_code(competition_code)
        params: dict[str, object] = {}
        if season is not None:
            params["season"] = self._require_positive_int("season", season)
        if matchday is not None:
            params["matchday"] = self._require_positive_int("matchday", matchday)
        if as_of_date is not None:
            params["date"] = as_of_date.isoformat()

        payload = await self._fetch_json(
            f"/competitions/{normalized_code}/standings",
            params=params,
            cache_ttl_seconds=_STANDINGS_TTL_SECONDS,
        )

        standings_payload = self._require_list(payload.get("standings"), "standings")
        competition_data = self._require_mapping(payload.get("competition"), "competition")
        season_data = self._optional_mapping(payload.get("season"))
        season_label = self._build_season_label(
            season_data=season_data,
            fallback_season=season,
        )

        total_rows: list[Mapping[str, object]] = []
        home_wins_by_team: dict[str, int] = {}
        away_wins_by_team: dict[str, int] = {}

        for standings_block in standings_payload:
            block_mapping = self._require_mapping(standings_block, "standings[]")
            table_rows = self._require_list(block_mapping.get("table"), "standings[].table")
            block_type = self._normalize_optional_text(block_mapping.get("type"))

            if block_type == "TOTAL" or block_type is None:
                total_rows.extend(
                    self._require_mapping(row, "standings[].table[]") for row in table_rows
                )
                continue

            if block_type == "HOME":
                self._collect_split_wins(home_wins_by_team, table_rows)
                continue

            if block_type == "AWAY":
                self._collect_split_wins(away_wins_by_team, table_rows)

        if not total_rows:
            raise ProviderError(
                self.provider_name,
                "Football-Data.org standings response did not include a usable standings table.",
            )

        return [
            self._normalize_standings_row(
                row,
                competition_data=competition_data,
                season_label=season_label,
                home_wins_by_team=home_wins_by_team,
                away_wins_by_team=away_wins_by_team,
            )
            for row in total_rows
        ]

    async def fetch_teams(
        self,
        *,
        competition_code: str,
        season: int | None = None,
    ) -> list[TeamStats]:
        """Fetch team metadata for one competition and normalize it to `TeamStats`.

        Args:
            competition_code: Football-Data.org competition code such as `PL`.
            season: Optional season year used to scope the competition roster.

        Returns:
            A list of minimal canonical `TeamStats` snapshots that preserve team
            identifiers and display metadata for downstream fallback workflows.
        """

        normalized_code = self._normalize_competition_code(competition_code)
        params: dict[str, object] = {}
        if season is not None:
            params["season"] = self._require_positive_int("season", season)

        payload = await self._fetch_json(
            f"/competitions/{normalized_code}/teams",
            params=params,
            cache_ttl_seconds=_TEAM_DATA_TTL_SECONDS,
        )

        competition_data = self._require_mapping(payload.get("competition"), "competition")
        season_data = self._optional_mapping(payload.get("season"))
        season_label = self._build_season_label(
            season_data=season_data,
            fallback_season=season,
        )

        return [
            self._normalize_team_metadata(
                self._require_mapping(team_payload, "teams[]"),
                competition_data=competition_data,
                season_label=season_label,
            )
            for team_payload in self._require_list(payload.get("teams"), "teams")
        ]

    async def _fetch_json(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> Mapping[str, object]:
        """Fetch one JSON payload and validate the top-level response shape."""

        response = await self.fetch(
            "GET",
            path,
            params=params,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"Football-Data.org returned invalid JSON for path '{path}'.",
                cause=exc,
            ) from exc

        if not isinstance(payload, Mapping):
            raise ProviderError(
                self.provider_name,
                "Football-Data.org responses must be JSON objects at the top level.",
            )

        return cast(Mapping[str, object], payload)

    def _normalize_fixture(
        self,
        payload: Mapping[str, object],
        *,
        competition_data: Mapping[str, object],
        area_data: Mapping[str, object] | None,
    ) -> NormalizedFixture:
        """Normalize one Football-Data.org match object into `NormalizedFixture`."""

        match_id = self._required_positive_int(payload.get("id"), field_name="match.id")
        kickoff = self._parse_datetime(payload.get("utcDate"), field_name="match.utcDate")
        status = self._normalize_fixture_status(payload.get("status"))
        home_team = self._require_mapping(payload.get("homeTeam"), "match.homeTeam")
        away_team = self._require_mapping(payload.get("awayTeam"), "match.awayTeam")

        return NormalizedFixture(
            sportradar_id=None,
            home_team=self._require_text(home_team.get("name"), "match.homeTeam.name"),
            away_team=self._require_text(away_team.get("name"), "match.awayTeam.name"),
            competition=self._require_text(competition_data.get("name"), "competition.name"),
            sport=SportName.SOCCER,
            kickoff=kickoff,
            source_provider=self.provider_name,
            source_id=str(match_id),
            country=(
                self._require_text(area_data.get("name"), "area.name")
                if area_data is not None and area_data.get("name") is not None
                else None
            ),
            home_team_id=str(
                self._required_positive_int(home_team.get("id"), field_name="match.homeTeam.id")
            ),
            away_team_id=str(
                self._required_positive_int(away_team.get("id"), field_name="match.awayTeam.id")
            ),
            venue=self._normalize_optional_text(payload.get("venue")),
            status=status,
        )

    def _normalize_standings_row(
        self,
        payload: Mapping[str, object],
        *,
        competition_data: Mapping[str, object],
        season_label: str | None,
        home_wins_by_team: Mapping[str, int],
        away_wins_by_team: Mapping[str, int],
    ) -> TeamStats:
        """Normalize one merged standings row into the canonical `TeamStats` model."""

        team_data = self._require_mapping(payload.get("team"), "standings.table[].team")
        team_id = str(
            self._required_positive_int(team_data.get("id"), field_name="standings.table[].team.id")
        )
        matches_played = self._coerce_non_negative_int(
            payload.get("playedGames"),
            field_name="standings.table[].playedGames",
        )
        goals_for = self._coerce_non_negative_int(
            payload.get("goalsFor"),
            field_name="standings.table[].goalsFor",
        )
        goals_against = self._coerce_non_negative_int(
            payload.get("goalsAgainst"),
            field_name="standings.table[].goalsAgainst",
        )
        goal_difference = self._coerce_int(
            payload.get("goalDifference"),
            field_name="standings.table[].goalDifference",
            default=goals_for - goals_against,
        )

        avg_goals_scored = (
            goals_for / matches_played if matches_played > 0 else None
        )
        avg_goals_conceded = (
            goals_against / matches_played if matches_played > 0 else None
        )

        return TeamStats(
            team_id=team_id,
            team_name=self._require_text(team_data.get("name"), "standings.table[].team.name"),
            sport=SportName.SOCCER,
            source_provider=self.provider_name,
            fetched_at=self._response_timestamp_or_now(payload.get("lastUpdated")),
            competition=self._require_text(competition_data.get("name"), "competition.name"),
            season=season_label,
            matches_played=matches_played,
            wins=self._coerce_non_negative_int(
                payload.get("won"),
                field_name="standings.table[].won",
            ),
            draws=self._coerce_non_negative_int(
                payload.get("draw"),
                field_name="standings.table[].draw",
            ),
            losses=self._coerce_non_negative_int(
                payload.get("lost"),
                field_name="standings.table[].lost",
            ),
            goals_for=goals_for,
            goals_against=goals_against,
            clean_sheets=0,
            form=self._normalize_form(payload.get("form")),
            position=self._coerce_positive_int(
                payload.get("position"),
                field_name="standings.table[].position",
            ),
            points=self._coerce_non_negative_int(
                payload.get("points"),
                field_name="standings.table[].points",
            ),
            home_wins=home_wins_by_team.get(team_id, 0),
            away_wins=away_wins_by_team.get(team_id, 0),
            avg_goals_scored=avg_goals_scored,
            avg_goals_conceded=avg_goals_conceded,
            advanced_metrics={"goal_difference": float(goal_difference or 0)},
        )

    def _normalize_team_metadata(
        self,
        payload: Mapping[str, object],
        *,
        competition_data: Mapping[str, object],
        season_label: str | None,
    ) -> TeamStats:
        """Normalize one competition team record into a minimal `TeamStats` snapshot."""

        team_id = self._required_positive_int(payload.get("id"), field_name="teams[].id")
        return TeamStats(
            team_id=str(team_id),
            team_name=self._require_text(payload.get("name"), "teams[].name"),
            sport=SportName.SOCCER,
            source_provider=self.provider_name,
            fetched_at=self._response_timestamp_or_now(payload.get("lastUpdated")),
            competition=self._require_text(competition_data.get("name"), "competition.name"),
            season=season_label,
        )

    @classmethod
    def _collect_split_wins(
        cls,
        target: dict[str, int],
        table_rows: list[object],
    ) -> None:
        """Collect home or away win totals from split standings tables."""

        for row in table_rows:
            row_mapping = cls._require_mapping(row, "standings[].table[]")
            team_data = cls._require_mapping(row_mapping.get("team"), "standings[].table[].team")
            team_id = str(
                cls._required_positive_int(
                    team_data.get("id"),
                    field_name="standings[].table[].team.id",
                )
            )
            target[team_id] = cls._coerce_non_negative_int(
                row_mapping.get("won"),
                field_name="standings[].table[].won",
            )

    @staticmethod
    def _normalize_competition_code(competition_code: str) -> str:
        """Validate and uppercase one Football-Data.org competition code."""

        normalized = competition_code.strip().upper()
        if not normalized:
            raise ValueError("competition_code must not be blank.")
        if not _COMPETITION_CODE_PATTERN.fullmatch(normalized):
            raise ValueError(
                "competition_code must be a Football-Data.org code such as `PL` or `FL1`."
            )
        return normalized

    @classmethod
    def _normalize_fixture_status(cls, value: object) -> FixtureStatus:
        """Map Football-Data.org match statuses into PuntLab fixture statuses."""

        status_text = cls._require_text(value, "match.status").upper()
        return _FIXTURE_STATUS_MAP.get(status_text, FixtureStatus.SCHEDULED)

    @classmethod
    def _build_season_label(
        cls,
        *,
        season_data: Mapping[str, object] | None,
        fallback_season: int | None,
    ) -> str | None:
        """Build a display-friendly season label from season metadata when available."""

        if season_data is not None:
            start_date = cls._parse_iso_date_or_none(season_data.get("startDate"))
            end_date = cls._parse_iso_date_or_none(season_data.get("endDate"))
            if start_date is not None and end_date is not None:
                if start_date.year == end_date.year:
                    return str(start_date.year)
                return f"{start_date.year}-{end_date.year % 100:02d}"

        if fallback_season is not None:
            return str(fallback_season)
        return None

    @classmethod
    def _response_timestamp_or_now(cls, value: object) -> datetime:
        """Parse an optional provider timestamp or fall back to current UTC time."""

        if value is None:
            return datetime.now(UTC)
        return cls._parse_datetime(value, field_name="lastUpdated")

    @staticmethod
    def _parse_iso_date_or_none(value: object) -> date | None:
        """Parse an optional ISO date string into `date` when present."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ProviderError(
                "football-data",
                f"Expected an ISO date string, received {type(value).__name__}.",
            )
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return date.fromisoformat(normalized)
        except ValueError as exc:
            raise ProviderError(
                "football-data",
                f"Invalid ISO date value '{normalized}'.",
                cause=exc,
            ) from exc

    @staticmethod
    def _parse_datetime(value: object, *, field_name: str) -> datetime:
        """Parse one required ISO datetime string into a timezone-aware `datetime`."""

        if not isinstance(value, str):
            raise ProviderError(
                "football-data",
                f"{field_name} must be an ISO datetime string.",
            )
        normalized = value.strip()
        if not normalized:
            raise ProviderError("football-data", f"{field_name} must not be blank.")

        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProviderError(
                "football-data",
                f"{field_name} must be a valid ISO datetime string.",
                cause=exc,
            ) from exc

        if parsed.tzinfo is None:
            raise ProviderError("football-data", f"{field_name} must include timezone data.")
        return parsed

    @staticmethod
    def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
        """Validate that one payload node is a JSON object mapping."""

        if not isinstance(value, Mapping):
            raise ProviderError("football-data", f"{field_name} must be an object.")
        return cast(Mapping[str, object], value)

    @classmethod
    def _optional_mapping(cls, value: object) -> Mapping[str, object] | None:
        """Return an optional mapping node while rejecting invalid non-mappings."""

        if value is None:
            return None
        return cls._require_mapping(value, "optional mapping")

    @staticmethod
    def _require_list(value: object, field_name: str) -> list[object]:
        """Validate that one payload node is a JSON array."""

        if not isinstance(value, list):
            raise ProviderError("football-data", f"{field_name} must be an array.")
        return value

    @staticmethod
    def _require_text(value: object, field_name: str) -> str:
        """Validate and normalize one required text value from provider payloads."""

        if not isinstance(value, str):
            raise ProviderError("football-data", f"{field_name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ProviderError("football-data", f"{field_name} must not be blank.")
        return normalized

    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        """Normalize optional string fields and collapse blanks to `None`."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ProviderError(
                "football-data",
                f"Expected optional text to be a string, received {type(value).__name__}.",
            )
        normalized = value.strip()
        return normalized or None

    @classmethod
    def _normalize_form(cls, value: object) -> str | None:
        """Normalize Football-Data.org form strings such as `W,W,D,W,W`."""

        normalized = cls._normalize_optional_text(value)
        if normalized is None:
            return None
        return normalized.replace(",", "").replace(" ", "").upper()

    @classmethod
    def _require_positive_int(cls, field_name: str, value: int) -> int:
        """Validate one required positive integer provided by the caller."""

        if value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    @staticmethod
    def _required_positive_int(value: object, *, field_name: str) -> int:
        """Validate one provider payload field as a positive integer."""

        if not isinstance(value, int) or value <= 0:
            raise ProviderError("football-data", f"{field_name} must be a positive integer.")
        return value

    @classmethod
    def _coerce_non_negative_int(cls, value: object, *, field_name: str) -> int:
        """Coerce a provider payload field into a non-negative integer."""

        coerced = cls._coerce_int(value, field_name=field_name, default=0)
        if coerced is None:
            return 0
        if coerced < 0:
            raise ProviderError("football-data", f"{field_name} must be non-negative.")
        return coerced

    @classmethod
    def _coerce_positive_int(cls, value: object, *, field_name: str) -> int | None:
        """Coerce an optional provider payload field into a positive integer."""

        if value is None:
            return None
        coerced = cls._coerce_int(value, field_name=field_name, default=None)
        if coerced is None:
            return None
        if coerced <= 0:
            raise ProviderError("football-data", f"{field_name} must be positive.")
        return coerced

    @staticmethod
    def _coerce_int(
        value: object,
        *,
        field_name: str,
        default: int | None,
    ) -> int | None:
        """Coerce one optional provider payload number into an integer."""

        if value is None:
            return default
        if isinstance(value, bool):
            raise ProviderError("football-data", f"{field_name} must be an integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return default
            try:
                return int(normalized)
            except ValueError as exc:
                raise ProviderError(
                    "football-data",
                    f"{field_name} must be an integer.",
                    cause=exc,
                ) from exc
        raise ProviderError("football-data", f"{field_name} must be an integer.")


__all__ = ["FootballDataProvider"]
