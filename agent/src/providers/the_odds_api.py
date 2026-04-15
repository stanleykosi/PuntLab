"""The Odds API provider implementation for PuntLab's odds ingestion layer.

Purpose: connect The Odds API's v4 sportsbook endpoints to PuntLab's canonical
provider infrastructure for soccer and NBA odds ingestion.
Scope: authenticated odds requests by sport or event, conservative quota-aware
request shaping, and normalization into the shared `NormalizedOdds` schema
without discarding unsupported provider markets.
Dependencies: `src.providers.base` for shared HTTP behavior, `src.cache.client`
for cache TTLs, `src.config` for credentials and market taxonomy, and the
shared odds schema under `src.schemas`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Final, cast

import httpx

from src.cache.client import API_ODDS_TTL_SECONDS
from src.config import MarketType, SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.odds import NormalizedOdds

logger = logging.getLogger(__name__)

_DEFAULT_REGIONS: Final[tuple[str, ...]] = ("eu",)
_MONTH_WINDOW_SECONDS: Final[int] = 30 * 24 * 60 * 60
_LOW_QUOTA_WARNING_THRESHOLD: Final[int] = 25
_SUPPORTED_SOCCER_TOTAL_LINES: Final[dict[float, MarketType]] = {
    0.5: MarketType.OVER_UNDER_05,
    1.5: MarketType.OVER_UNDER_15,
    2.5: MarketType.OVER_UNDER_25,
    3.5: MarketType.OVER_UNDER_35,
}
_MARKET_LABELS: Final[dict[str, str]] = {
    "h2h": "Head to Head",
    "spreads": "Spreads",
    "alternate_spreads": "Alternate Spreads",
    "totals": "Totals",
    "alternate_totals": "Alternate Totals",
    "btts": "Both Teams To Score",
    "draw_no_bet": "Draw No Bet",
    "double_chance": "Double Chance",
    "correct_score": "Correct Score",
}
_MARKET_KEY_PREFIXES: Final[tuple[str, ...]] = (
    "alternate_spreads",
    "alternate_totals",
    "draw_no_bet",
    "double_chance",
    "correct_score",
    "spreads",
    "totals",
    "btts",
    "h2h",
)
_PERIOD_SUFFIXES: Final[dict[str, str]] = {
    "h1": "first_half",
    "h2": "second_half",
    "q1": "first_quarter",
    "q2": "second_quarter",
    "q3": "third_quarter",
    "q4": "fourth_quarter",
    "p1": "first_period",
    "p2": "second_period",
    "p3": "third_period",
}
_DOUBLE_CHANCE_SELECTIONS: Final[dict[str, str]] = {
    "home or draw": "1X",
    "draw or home": "1X",
    "home or away": "12",
    "away or home": "12",
    "draw or away": "X2",
    "away or draw": "X2",
}


class TheOddsAPIProvider(DataProvider):
    """Concrete The Odds API integration for bookmaker odds ingestion.

    Inputs:
        A shared `RateLimitedClient`, a The Odds API key, and an optional
        canonical default bookmaker region list.

    Outputs:
        Typed methods for sport-level and event-level odds retrieval that
        return normalized PuntLab odds rows.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        api_key: str | None = None,
        default_regions: Sequence[str] | None = None,
    ) -> None:
        """Initialize the provider with validated credentials and defaults.

        Args:
            client: Shared `RateLimitedClient` used for cached and retried HTTP
                requests.
            api_key: Optional explicit The Odds API key. When omitted, the
                provider falls back to `THE_ODDS_API_KEY` from settings.
            default_regions: Optional canonical region list used when callers do
                not provide `regions` or `bookmakers`.

        Raises:
            ValueError: If no usable API key is available or the region default
                list is empty.
        """

        super().__init__(client)
        resolved_api_key = (api_key or get_settings().data_providers.the_odds_api_key or "").strip()
        if not resolved_api_key:
            raise ValueError(
                "The Odds API requires `THE_ODDS_API_KEY` or an explicit `api_key`."
            )

        self._api_key = resolved_api_key
        self._default_regions = self._normalize_identifier_list(
            "default_regions",
            default_regions or _DEFAULT_REGIONS,
        )

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in cache keys and logs."""

        return "the-odds-api"

    @property
    def base_url(self) -> str:
        """Return the current official The Odds API v4 base URL."""

        return "https://api.the-odds-api.com/v4"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Approximate the documented 500-credit free tier with a monthly bucket."""

        return RateLimitPolicy(limit=500, window_seconds=_MONTH_WINDOW_SECONDS)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the canonical content negotiation headers for this provider."""

        return {"Accept": "application/json"}

    async def fetch_odds(
        self,
        *,
        sport_key: str,
        markets: Sequence[str] = ("h2h",),
        regions: Sequence[str] | None = None,
        bookmakers: Sequence[str] | None = None,
        commence_time_from: datetime | None = None,
        commence_time_to: datetime | None = None,
    ) -> list[NormalizedOdds]:
        """Fetch upcoming or live odds across all visible events in one sport.

        Args:
            sport_key: The Odds API sport identifier such as `soccer_epl` or
                `basketball_nba`.
            markets: One or more provider market keys to request.
            regions: Optional bookmaker region filters. Mutually exclusive with
                `bookmakers`.
            bookmakers: Optional explicit bookmaker keys. Mutually exclusive
                with `regions`.
            commence_time_from: Optional inclusive lower bound for event kickoff
                filtering.
            commence_time_to: Optional inclusive upper bound for event kickoff
                filtering.

        Returns:
            A flat list of normalized odds rows across every event returned for
            the sport query.
        """

        params = self._build_query_params(
            markets=markets,
            regions=regions,
            bookmakers=bookmakers,
        )
        if commence_time_from is not None:
            params["commenceTimeFrom"] = self._format_datetime(commence_time_from)
        if commence_time_to is not None:
            params["commenceTimeTo"] = self._format_datetime(commence_time_to)

        payload = await self._fetch_json_list(
            f"/sports/{self._require_text(sport_key, 'sport_key')}/odds",
            params=params,
        )

        normalized_rows: list[NormalizedOdds] = []
        for event_payload in payload:
            normalized_rows.extend(
                self._normalize_event_odds(
                    self._require_mapping(event_payload, "event"),
                    sport_key=sport_key,
                )
            )
        return normalized_rows

    async def fetch_event_odds(
        self,
        *,
        sport_key: str,
        event_id: str,
        markets: Sequence[str],
        regions: Sequence[str] | None = None,
        bookmakers: Sequence[str] | None = None,
        include_multipliers: bool = False,
    ) -> list[NormalizedOdds]:
        """Fetch odds for one event, including non-featured market keys.

        Args:
            sport_key: The Odds API sport identifier such as `basketball_nba`.
            event_id: Provider event identifier from the sport or events
                endpoints.
            markets: One or more provider market keys to request.
            regions: Optional bookmaker region filters. Mutually exclusive with
                `bookmakers`.
            bookmakers: Optional explicit bookmaker keys. Mutually exclusive
                with `regions`.
            include_multipliers: Whether DFS multiplier fields should be
                requested when supported by the upstream provider.

        Returns:
            A flat list of normalized odds rows for the requested event.
        """

        params = self._build_query_params(
            markets=markets,
            regions=regions,
            bookmakers=bookmakers,
        )
        params["includeMultipliers"] = "true" if include_multipliers else "false"

        payload = await self._fetch_json_object(
            (
                f"/sports/{self._require_text(sport_key, 'sport_key')}/events/"
                f"{self._require_text(event_id, 'event_id')}/odds"
            ),
            params=params,
        )
        return self._normalize_event_odds(payload, sport_key=sport_key)

    async def _fetch_json_list(
        self,
        path: str,
        *,
        params: Mapping[str, object],
    ) -> list[Mapping[str, object]]:
        """Fetch one JSON array payload and validate its top-level shape."""

        response = await self.fetch(
            "GET",
            path,
            params=params,
            cache_ttl_seconds=API_ODDS_TTL_SECONDS,
        )
        self._log_quota_snapshot(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"The Odds API returned invalid JSON for path '{path}'.",
                cause=exc,
            ) from exc

        if not isinstance(payload, list):
            raise ProviderError(
                self.provider_name,
                "The Odds API sport odds endpoint must return a JSON array.",
            )

        return [self._require_mapping(item, "event") for item in payload]

    async def _fetch_json_object(
        self,
        path: str,
        *,
        params: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Fetch one JSON object payload and validate its top-level shape."""

        response = await self.fetch(
            "GET",
            path,
            params=params,
            cache_ttl_seconds=API_ODDS_TTL_SECONDS,
        )
        self._log_quota_snapshot(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"The Odds API returned invalid JSON for path '{path}'.",
                cause=exc,
            ) from exc

        if not isinstance(payload, Mapping):
            raise ProviderError(
                self.provider_name,
                "The Odds API event odds endpoint must return a JSON object.",
            )

        return cast(Mapping[str, object], payload)

    def _build_query_params(
        self,
        *,
        markets: Sequence[str],
        regions: Sequence[str] | None,
        bookmakers: Sequence[str] | None,
    ) -> dict[str, str]:
        """Build one canonical request parameter set for The Odds API calls."""

        if regions is not None and bookmakers is not None:
            raise ValueError("regions and bookmakers cannot be supplied together.")

        normalized_markets = self._normalize_identifier_list("markets", markets)
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "markets": ",".join(normalized_markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        if bookmakers is not None:
            params["bookmakers"] = ",".join(
                self._normalize_identifier_list("bookmakers", bookmakers)
            )
            return params

        resolved_regions = (
            self._normalize_identifier_list("regions", regions)
            if regions is not None
            else self._default_regions
        )
        params["regions"] = ",".join(resolved_regions)
        return params

    def _normalize_event_odds(
        self,
        payload: Mapping[str, object],
        *,
        sport_key: str,
    ) -> list[NormalizedOdds]:
        """Flatten one The Odds API event payload into normalized odds rows."""

        event_id = self._require_text(payload.get("id"), "event.id")
        home_team = self._require_text(payload.get("home_team"), "event.home_team")
        away_team = self._require_text(payload.get("away_team"), "event.away_team")
        sport_title = self._optional_text(payload.get("sport_title"))
        commence_time = self._optional_datetime(payload.get("commence_time"), "event.commence_time")
        bookmakers = self._require_list(payload.get("bookmakers"), "event.bookmakers")

        normalized_rows: list[NormalizedOdds] = []
        for bookmaker in bookmakers:
            bookmaker_data = self._require_mapping(bookmaker, "event.bookmakers[]")
            bookmaker_name = self._require_text(
                bookmaker_data.get("title"),
                "event.bookmakers[].title",
            )
            bookmaker_key = self._optional_text(bookmaker_data.get("key"))
            last_updated = self._optional_datetime(
                bookmaker_data.get("last_update"),
                "event.bookmakers[].last_update",
            )
            markets = self._require_list(
                bookmaker_data.get("markets"),
                "event.bookmakers[].markets",
            )

            for market_payload in markets:
                market_data = self._require_mapping(
                    market_payload,
                    "event.bookmakers[].markets[]",
                )
                market_key = self._require_text(
                    market_data.get("key"),
                    "event.bookmakers[].markets[].key",
                )
                outcomes = self._require_list(
                    market_data.get("outcomes"),
                    "event.bookmakers[].markets[].outcomes",
                )
                for outcome_payload in outcomes:
                    typed_outcome = self._require_mapping(
                        outcome_payload,
                        "event.bookmakers[].markets[].outcomes[]",
                    )
                    try:
                        normalized_rows.append(
                            self._normalize_outcome(
                                event_id=event_id,
                                sport_key=sport_key,
                                sport_title=sport_title,
                                home_team=home_team,
                                away_team=away_team,
                                commence_time=commence_time,
                                bookmaker_name=bookmaker_name,
                                bookmaker_key=bookmaker_key,
                                market_key=market_key,
                                last_updated=last_updated,
                                outcome_payload=typed_outcome,
                            )
                        )
                    except ProviderError as exc:
                        # Upstream feeds occasionally include malformed outcome
                        # payloads for one market while the rest of the event
                        # is valid. Preserve the healthy rows instead of
                        # dropping the entire odds batch for one bad outcome.
                        logger.warning(
                            "Skipping malformed The Odds API outcome for event=%s "
                            "bookmaker=%s market=%s: %s",
                            event_id,
                            bookmaker_name,
                            market_key,
                            exc,
                        )
                        continue

        return normalized_rows

    def _normalize_outcome(
        self,
        *,
        event_id: str,
        sport_key: str,
        sport_title: str | None,
        home_team: str,
        away_team: str,
        commence_time: datetime | None,
        bookmaker_name: str,
        bookmaker_key: str | None,
        market_key: str,
        last_updated: datetime | None,
        outcome_payload: Mapping[str, object],
    ) -> NormalizedOdds:
        """Normalize one provider outcome into the canonical odds contract."""

        raw_selection = self._require_text(
            outcome_payload.get("name"),
            "event.bookmakers[].markets[].outcomes[].name",
        )
        odds_value = self._coerce_decimal_odds(
            outcome_payload.get("price"),
            field_name="event.bookmakers[].markets[].outcomes[].price",
        )
        line = self._coerce_optional_float(
            outcome_payload.get("point"),
            field_name="event.bookmakers[].markets[].outcomes[].point",
        )

        market, selection = self._map_market_and_selection(
            sport_key=sport_key,
            market_key=market_key,
            raw_selection=raw_selection,
            line=line,
            home_team=home_team,
            away_team=away_team,
        )

        raw_metadata: dict[str, str | int | float | bool | None] = {
            "canonical_market_supported": market is not None,
            "sport_key": sport_key,
            "sport_title": sport_title,
            "event_id": event_id,
            "bookmaker_key": bookmaker_key,
            "home_team": home_team,
            "away_team": away_team,
        }
        multiplier = outcome_payload.get("multiplier")
        if multiplier is not None:
            raw_metadata["multiplier"] = self._coerce_optional_float(
                multiplier,
                field_name="event.bookmakers[].markets[].outcomes[].multiplier",
            )
        if commence_time is not None:
            raw_metadata["commence_time"] = self._format_datetime(commence_time)

        return NormalizedOdds(
            fixture_ref=f"{self.provider_name}:{event_id}",
            market=market,
            selection=selection,
            odds=odds_value,
            provider=bookmaker_name,
            provider_market_name=market_key,
            provider_selection_name=raw_selection,
            sportybet_available=False,
            market_label=self._market_label(market_key),
            line=line,
            period=self._infer_period(market_key),
            participant_scope=self._infer_participant_scope(market_key),
            provider_market_id=market_key,
            raw_metadata=raw_metadata,
            last_updated=last_updated,
        )

    def _map_market_and_selection(
        self,
        *,
        sport_key: str,
        market_key: str,
        raw_selection: str,
        line: float | None,
        home_team: str,
        away_team: str,
    ) -> tuple[MarketType | None, str]:
        """Map one provider market and selection into PuntLab's taxonomy."""

        base_market_key = self._extract_base_market_key(market_key)
        sport = self._infer_sport_name(sport_key)
        selection = raw_selection
        market: MarketType | None = None

        if base_market_key == "h2h":
            if sport == SportName.SOCCER:
                market = MarketType.MATCH_RESULT
                selection = self._normalize_match_result_selection(
                    raw_selection,
                    home_team=home_team,
                    away_team=away_team,
                )
            elif sport == SportName.BASKETBALL:
                market = MarketType.MONEYLINE
                selection = self._normalize_side_selection(
                    raw_selection,
                    home_team=home_team,
                    away_team=away_team,
                )
        elif base_market_key in {"spreads", "alternate_spreads"} and line is not None:
            if sport == SportName.SOCCER:
                market = MarketType.ASIAN_HANDICAP
            elif sport == SportName.BASKETBALL:
                market = MarketType.POINT_SPREAD
            selection = self._normalize_side_selection(
                raw_selection,
                home_team=home_team,
                away_team=away_team,
            )
        elif base_market_key in {"totals", "alternate_totals"} and line is not None:
            selection = self._normalize_over_under_selection(raw_selection)
            if sport == SportName.SOCCER:
                market = _SUPPORTED_SOCCER_TOTAL_LINES.get(line)
            elif sport == SportName.BASKETBALL:
                market = MarketType.TOTAL_POINTS
        elif base_market_key == "btts":
            market = MarketType.BTTS
            selection = self._normalize_yes_no_selection(raw_selection)
        elif base_market_key == "draw_no_bet":
            market = MarketType.DRAW_NO_BET
            selection = self._normalize_match_result_selection(
                raw_selection,
                home_team=home_team,
                away_team=away_team,
                allow_draw=False,
            )
        elif base_market_key == "double_chance":
            market = MarketType.DOUBLE_CHANCE
            selection = self._normalize_double_chance_selection(
                raw_selection,
                home_team=home_team,
                away_team=away_team,
            )
        elif base_market_key == "correct_score":
            market = MarketType.CORRECT_SCORE
            selection = raw_selection.strip()

        if market is None:
            logger.debug(
                "Preserving unmapped The Odds API market=%s selection=%s",
                market_key,
                raw_selection,
            )

        return market, selection

    def _extract_base_market_key(self, market_key: str) -> str:
        """Return the mapped market prefix from one provider market key."""

        normalized_key = self._require_text(market_key, "market_key").lower()
        for prefix in _MARKET_KEY_PREFIXES:
            if normalized_key == prefix or normalized_key.startswith(f"{prefix}_"):
                return prefix
        return normalized_key

    def _market_label(self, market_key: str) -> str:
        """Return a human-readable label while preserving the raw market key."""

        base_market_key = self._extract_base_market_key(market_key)
        if base_market_key in _MARKET_LABELS:
            return _MARKET_LABELS[base_market_key]
        return " ".join(segment.capitalize() for segment in market_key.split("_"))

    def _infer_period(self, market_key: str) -> str:
        """Infer the market period from common The Odds API key suffixes."""

        normalized_key = self._require_text(market_key, "market_key").lower()
        for suffix, period in _PERIOD_SUFFIXES.items():
            if normalized_key.endswith(f"_{suffix}"):
                return period
        return "match"

    def _infer_participant_scope(self, market_key: str) -> str:
        """Infer the participant scope for one provider market key."""

        normalized_key = self._require_text(market_key, "market_key").lower()
        if normalized_key.startswith("player_"):
            return "player"
        if normalized_key.startswith("team_") or "_team_" in normalized_key:
            return "team"
        return "match"

    def _infer_sport_name(self, sport_key: str) -> SportName | None:
        """Map one provider sport key into PuntLab's supported sport enum."""

        normalized_sport_key = self._require_text(sport_key, "sport_key").lower()
        if normalized_sport_key.startswith("soccer_"):
            return SportName.SOCCER
        if normalized_sport_key.startswith("basketball_"):
            return SportName.BASKETBALL
        return None

    def _normalize_match_result_selection(
        self,
        raw_selection: str,
        *,
        home_team: str,
        away_team: str,
        allow_draw: bool = True,
    ) -> str:
        """Normalize team-name outcomes into `home`, `away`, or `draw`."""

        normalized = raw_selection.strip().casefold()
        if normalized == home_team.casefold():
            return "home"
        if normalized == away_team.casefold():
            return "away"
        if allow_draw and normalized == "draw":
            return "draw"
        return raw_selection.strip()

    def _normalize_side_selection(
        self,
        raw_selection: str,
        *,
        home_team: str,
        away_team: str,
    ) -> str:
        """Normalize side-based selections into `home` or `away` when possible."""

        normalized = raw_selection.strip().casefold()
        if normalized == home_team.casefold():
            return "home"
        if normalized == away_team.casefold():
            return "away"
        return raw_selection.strip()

    def _normalize_over_under_selection(self, raw_selection: str) -> str:
        """Normalize totals outcomes into `over` or `under`."""

        normalized = raw_selection.strip().casefold()
        if normalized == "over":
            return "over"
        if normalized == "under":
            return "under"
        return raw_selection.strip()

    def _normalize_yes_no_selection(self, raw_selection: str) -> str:
        """Normalize binary selections into `yes` or `no`."""

        normalized = raw_selection.strip().casefold()
        if normalized == "yes":
            return "yes"
        if normalized == "no":
            return "no"
        return raw_selection.strip()

    def _normalize_double_chance_selection(
        self,
        raw_selection: str,
        *,
        home_team: str,
        away_team: str,
    ) -> str:
        """Normalize double-chance outcomes into the canonical 1X/12/X2 tokens."""

        normalized = raw_selection.strip().casefold()
        normalized = normalized.replace(home_team.casefold(), "home")
        normalized = normalized.replace(away_team.casefold(), "away")
        if normalized in _DOUBLE_CHANCE_SELECTIONS:
            return _DOUBLE_CHANCE_SELECTIONS[normalized]
        return raw_selection.strip()

    def _log_quota_snapshot(self, response: httpx.Response) -> None:
        """Log current The Odds API quota headers for low-budget visibility."""

        headers = cast(Mapping[str, str], response.headers)
        remaining = self._coerce_optional_int(headers.get("x-requests-remaining"))
        used = self._coerce_optional_int(headers.get("x-requests-used"))
        last = self._coerce_optional_int(headers.get("x-requests-last"))

        if remaining is not None and remaining <= _LOW_QUOTA_WARNING_THRESHOLD:
            logger.warning(
                "The Odds API quota is running low: remaining=%s used=%s last_cost=%s",
                remaining,
                used,
                last,
            )
            return

        logger.debug(
            "The Odds API quota snapshot: remaining=%s used=%s last_cost=%s",
            remaining,
            used,
            last,
        )

    def _normalize_identifier_list(
        self,
        field_name: str,
        values: Sequence[str],
    ) -> tuple[str, ...]:
        """Trim, validate, and deduplicate one identifier sequence."""

        normalized_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._require_text(value, field_name).lower()
            if normalized in seen:
                continue
            normalized_values.append(normalized)
            seen.add(normalized)

        if not normalized_values:
            raise ValueError(f"{field_name} must contain at least one value.")
        return tuple(normalized_values)

    def _format_datetime(self, value: datetime) -> str:
        """Serialize one timezone-aware datetime to canonical ISO-8601 UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime values must include timezone information.")
        return (
            value.astimezone(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _optional_datetime(self, value: object, field_name: str) -> datetime | None:
        """Parse one optional ISO-8601 datetime string when present."""

        if value is None:
            return None
        raw_value = self._require_text(value, field_name)
        if raw_value.endswith("Z"):
            raw_value = f"{raw_value[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw_value)
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                f"{field_name} must be a valid ISO-8601 datetime.",
                cause=exc,
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ProviderError(
                self.provider_name,
                f"{field_name} must include timezone information.",
            )
        return parsed

    def _coerce_decimal_odds(self, value: object, *, field_name: str) -> float:
        """Coerce one provider odds value into a finite decimal float."""

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("+", "-")):
                try:
                    american_value = float(stripped)
                except ValueError:
                    # Fall through to canonical numeric validation below.
                    american_value = None
                if american_value is not None and abs(american_value) >= 100:
                    return self._convert_american_to_decimal(
                        american_value,
                        field_name=field_name,
                    )

        odds_value = self._coerce_float(value, field_name=field_name)
        if odds_value > 1.0:
            return odds_value

        if abs(odds_value) >= 100:
            return self._convert_american_to_decimal(
                odds_value,
                field_name=field_name,
            )

        raise ProviderError(
            self.provider_name,
            f"{field_name} must be a decimal odds value greater than 1.0.",
        )

    def _convert_american_to_decimal(self, value: float, *, field_name: str) -> float:
        """Convert one American odds value into decimal format.

        Some sportsbooks may return American prices even when decimal was
        requested. Converting keeps the ingestion path resilient and avoids
        dropping full events due to mixed odds formats.
        """

        if value == 0:
            raise ProviderError(
                self.provider_name,
                f"{field_name} must not be zero.",
            )

        decimal = (
            1.0 + (value / 100.0)
            if value > 0
            else 1.0 + (100.0 / abs(value))
        )
        if decimal <= 1.0:
            raise ProviderError(
                self.provider_name,
                f"{field_name} could not be converted to valid decimal odds.",
            )
        return decimal

    def _coerce_optional_float(self, value: object, *, field_name: str) -> float | None:
        """Parse an optional numeric field into a finite float."""

        if value is None:
            return None
        return self._coerce_float(value, field_name=field_name)

    def _coerce_float(self, value: object, *, field_name: str) -> float:
        """Parse one numeric provider field into a finite float."""

        if isinstance(value, bool):
            raise ProviderError(self.provider_name, f"{field_name} must be numeric.")

        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ProviderError(self.provider_name, f"{field_name} must not be blank.")
            try:
                numeric_value = float(stripped)
            except ValueError as exc:
                raise ProviderError(
                    self.provider_name,
                    f"{field_name} must be numeric.",
                    cause=exc,
                ) from exc
        else:
            raise ProviderError(self.provider_name, f"{field_name} must be numeric.")

        if not isfinite(numeric_value):
            raise ProviderError(self.provider_name, f"{field_name} must be finite.")
        return numeric_value

    def _coerce_optional_int(self, value: object) -> int | None:
        """Parse one optional integer-like header value when present."""

        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = int(stripped)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        return None

    def _require_mapping(self, value: object, field_name: str) -> Mapping[str, object]:
        """Require one JSON object from a provider response payload."""

        if not isinstance(value, Mapping):
            raise ProviderError(
                self.provider_name,
                f"{field_name} must be a JSON object in The Odds API responses.",
            )
        return cast(Mapping[str, object], value)

    def _require_list(self, value: object, field_name: str) -> list[object]:
        """Require one JSON array from a provider response payload."""

        if not isinstance(value, list):
            raise ProviderError(
                self.provider_name,
                f"{field_name} must be a JSON array in The Odds API responses.",
            )
        return cast(list[object], value)

    def _require_text(self, value: object, field_name: str) -> str:
        """Trim and validate one required text field from provider payloads."""

        if not isinstance(value, str):
            raise ProviderError(self.provider_name, f"{field_name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ProviderError(self.provider_name, f"{field_name} must not be blank.")
        return normalized

    def _optional_text(self, value: object) -> str | None:
        """Trim optional provider strings and collapse blanks to `None`."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ProviderError(self.provider_name, "Optional text fields must be strings.")
        normalized = value.strip()
        return normalized or None


__all__ = ["TheOddsAPIProvider"]
