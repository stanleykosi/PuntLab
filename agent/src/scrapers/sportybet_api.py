"""HTTP-based SportyBet market interceptor for PuntLab's resolver pipeline.

Purpose: fetch SportyBet market data through lightweight HTTP requests before
the slower browser fallback is attempted.
Scope: public SportyBet URL construction, rotating user-agent selection,
Redis-backed market caching, current API endpoint fallback, grouped event
catalog construction, and normalization into `NormalizedOdds`.
Dependencies: `httpx` transport via `RateLimitedClient`, shared Redis cache
helpers, normalized fixture and odds schemas, and canonical market enums.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, MarketType, SportName
from src.providers.base import ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds

SPORTYBET_BASE_URL: Final[str] = "https://www.sportybet.com"
SPORTYBET_COUNTRY_CODE: Final[str] = "ng"
SPORTYBET_API_PATH_TEMPLATES: Final[tuple[str, ...]] = (
    "/api/ng/factsCenter/event?eventId={sportradar_id}",
    # Legacy endpoints are kept as trailing fallbacks only because SportyBet
    # has historically changed routes without notice. The canonical path is
    # `factsCenter/event?eventId=sr:match:*`.
    "/api/ng/factsCenter/pc/matchDetail?eventId={sportradar_id}",
    "/api/ng/factsCenter/eventDetail?eventId={sportradar_id}",
)
SPORTYBET_DEFAULT_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-NG,en;q=0.9",
    "Origin": SPORTYBET_BASE_URL,
    "Referer": f"{SPORTYBET_BASE_URL}/{SPORTYBET_COUNTRY_CODE}/",
}
SPORTYBET_RATE_LIMIT_POLICY: Final[RateLimitPolicy] = RateLimitPolicy(
    limit=120,
    window_seconds=60,
)
SPORTYBET_MARKET_IDS: Final[dict[int, MarketType]] = {
    1: MarketType.MATCH_RESULT,
}
DEFAULT_USER_AGENTS: Final[tuple[str, ...]] = (
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
)
_LINE_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.ASIAN_HANDICAP,
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
    }
)


class SportyBetMarketCacheEntry(BaseModel):
    """Serializable cached SportyBet market snapshot.

    Inputs:
        One resolved SportyBet response normalized into market rows.

    Outputs:
        A cache-safe container that allows future requests for the same
        Sportradar fixture to be served without another network request.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sportradar_id: str = Field(description="Fixture Sportradar identifier.")
    fetched_at: datetime = Field(description="UTC timestamp of the fetch.")
    markets: tuple[NormalizedOdds, ...] = Field(
        default_factory=tuple,
        description="Normalized SportyBet market rows for the fixture.",
    )

    @field_validator("sportradar_id")
    @classmethod
    def validate_sportradar_id(cls, value: str) -> str:
        """Reject blank cache identifiers before they reach Redis."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("sportradar_id must not be blank.")
        return normalized

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware cache timestamps for deterministic expiry."""

        if value.utcoffset() is None:
            raise ValueError("fetched_at must include timezone information.")
        return value


@dataclass(frozen=True, slots=True)
class ParsedFixtureContext:
    """Resolved fixture metadata extracted from a SportyBet event payload.

    Inputs:
        One event object returned by a SportyBet match-detail endpoint.

    Outputs:
        Normalized identity fields reused for every parsed market selection.
    """

    fixture_ref: str
    sport: SportName
    competition: str
    home_team: str
    away_team: str


@dataclass(frozen=True, slots=True)
class SportyBetMarketGroup:
    """One SportyBet market group preserved from the event payload.

    Inputs:
        Raw grouped market nodes from SportyBet's current `factsCenter/event`
        payload.

    Outputs:
        A deterministic group container used to arrange and traverse the
        fetched market universe without losing SportyBet's own grouping.
    """

    group_id: str
    group_name: str
    markets: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class SportyBetEventCatalog:
    """Structured SportyBet event snapshot grouped by market sections.

    Inputs:
        One normalized SportyBet event payload with fixture metadata and raw
        grouped market nodes.

    Outputs:
        A grouped catalog that downstream normalization can flatten into odds
        rows while preserving group identifiers and labels in metadata.
    """

    fixture: ParsedFixtureContext
    requested_sportradar_id: str
    event_id: str | None
    game_id: str | int | None
    total_market_size: int | None
    groups: tuple[SportyBetMarketGroup, ...]

    def iter_market_nodes(self) -> Iterator[tuple[SportyBetMarketGroup, Mapping[str, object]]]:
        """Yield grouped market nodes in deterministic SportyBet source order."""

        for market_group in self.groups:
            for market_node in market_group.markets:
                yield (market_group, market_node)


class SportyBetAPIClient:
    """Primary HTTP scraper for SportyBet match-market resolution.

    Args:
        cache: Shared Redis cache wrapper used for SportyBet market snapshots.
        http_client: Optional injected `httpx.AsyncClient` for tests.
        rate_limit_policy: Optional request budget override.
        api_path_templates: Optional candidate endpoint templates. Each entry
            must include `{sportradar_id}` so the requested fixture can be
            interpolated directly into the URL.
        user_agents: Optional ordered user-agent pool used in round-robin
            fashion across requests.
        clock: Optional clock used for cache metadata and deterministic tests.
    """

    def __init__(
        self,
        cache: RedisClient,
        *,
        http_client: httpx.AsyncClient | None = None,
        rate_limit_policy: RateLimitPolicy = SPORTYBET_RATE_LIMIT_POLICY,
        api_path_templates: Sequence[str] = SPORTYBET_API_PATH_TEMPLATES,
        user_agents: Sequence[str] = DEFAULT_USER_AGENTS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the SportyBet client with canonical PuntLab defaults."""

        if not api_path_templates:
            raise ValueError("api_path_templates must contain at least one endpoint template.")
        if not user_agents:
            raise ValueError("user_agents must contain at least one value.")

        normalized_templates = tuple(
            self._normalize_template(template) for template in api_path_templates
        )
        normalized_user_agents = tuple(
            self._normalize_user_agent(agent) for agent in user_agents
        )

        self._cache = cache
        self._clock = clock or (lambda: datetime.now(WAT_TIMEZONE))
        self._rate_limit_policy = rate_limit_policy
        self._api_path_templates = normalized_templates
        self._user_agents: Iterator[str] = itertools.cycle(normalized_user_agents)
        self._client = RateLimitedClient(
            cache,
            http_client=http_client,
        )

    async def fetch_markets(
        self,
        sportradar_id: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Fetch and normalize SportyBet markets for one Sportradar fixture.

        Inputs:
            sportradar_id: Canonical `sr:match:*` identifier for the fixture.
            fixture: Optional normalized fixture metadata used to construct the
                public SportyBet URL and fill any missing display fields.
            use_cache: Whether Redis may satisfy the request before the network.

        Outputs:
            A tuple of normalized SportyBet market rows that preserve provider
            labels and IDs while exposing PuntLab's canonical market taxonomy
            when it can be inferred safely.

        Raises:
            ProviderError: If every candidate SportyBet endpoint fails or no
                parsable matching event can be extracted from the responses.
        """

        normalized_sportradar_id = self._normalize_sportradar_id(sportradar_id)
        if use_cache:
            cached_markets = await self._get_cached_markets(normalized_sportradar_id)
            if cached_markets is not None:
                return cached_markets

        attempted_urls: list[str] = []
        errors: list[str] = []
        for url in self.build_api_urls(normalized_sportradar_id):
            attempted_urls.append(url)
            headers = self._build_request_headers()
            try:
                response = await self._client.request(
                    "sportybet",
                    "GET",
                    url,
                    rate_limit_policy=self._rate_limit_policy,
                    use_cache=False,
                    headers=headers,
                )
            except ProviderError as exc:
                errors.append(f"{url} -> {exc!s}")
                continue

            try:
                payload = response.json()
            except json.JSONDecodeError:
                errors.append(f"{url} -> response body is not valid JSON")
                continue

            try:
                event = self._extract_matching_event(payload, normalized_sportradar_id)
                markets = self._tag_fetch_source(
                    self._parse_markets(
                        event,
                        sportradar_id=normalized_sportradar_id,
                        fixture=fixture,
                    ),
                    fetch_source="api",
                )
            except ProviderError as exc:
                errors.append(f"{url} -> {exc!s}")
                continue

            if not markets:
                errors.append(f"{url} -> no active markets were returned for the fixture")
                continue

            await self._cache_markets(normalized_sportradar_id, markets)
            return markets

        attempted_text = ", ".join(attempted_urls)
        detail = " | ".join(errors) if errors else "no endpoint attempts were executed"
        raise ProviderError(
            "sportybet",
            (
                "SportyBet API intercept could not resolve markets for "
                f"{normalized_sportradar_id}. Attempted URLs: {attempted_text}. "
                f"Failures: {detail}"
            ),
        )

    def build_sportybet_url(self, fixture: NormalizedFixture) -> str:
        """Construct the canonical public SportyBet match URL for a fixture."""

        if fixture.sportradar_id is None:
            raise ValueError("fixture.sportradar_id is required for SportyBet URL construction.")

        sport_segment = "football" if fixture.sport == SportName.SOCCER else "basketball"
        country_segment = fixture.get_sportybet_country_slug() or "international"
        league_segment = fixture.get_sportybet_league_slug()
        home_slug = fixture.home_team.strip().replace(" ", "_")
        away_slug = fixture.away_team.strip().replace(" ", "_")
        match_slug = f"{home_slug}_vs_{away_slug}"
        return (
            f"{SPORTYBET_BASE_URL}/{SPORTYBET_COUNTRY_CODE}/sport/{sport_segment}/"
            f"{country_segment}/{league_segment}/{match_slug}/{fixture.sportradar_id}"
        )

    def build_api_urls(self, sportradar_id: str) -> tuple[str, ...]:
        """Expand the configured SportyBet endpoint templates for one fixture."""

        normalized_sportradar_id = self._normalize_sportradar_id(sportradar_id)
        return tuple(
            str(
                httpx.URL(
                    SPORTYBET_BASE_URL
                    + template.format(sportradar_id=normalized_sportradar_id)
                )
            )
            for template in self._api_path_templates
        )

    async def aclose(self) -> None:
        """Close the underlying shared HTTP transport."""

        await self._client.aclose()

    async def _get_cached_markets(self, sportradar_id: str) -> tuple[NormalizedOdds, ...] | None:
        """Load one cached SportyBet market snapshot from Redis when present."""

        cache_key = RedisClient.build_sportybet_markets_key(sportradar_id)
        cached_snapshot = await self._cache.get(cache_key, model=SportyBetMarketCacheEntry)
        if not isinstance(cached_snapshot, SportyBetMarketCacheEntry):
            return None
        return cached_snapshot.markets

    async def _cache_markets(
        self,
        sportradar_id: str,
        markets: Sequence[NormalizedOdds],
    ) -> None:
        """Persist the normalized SportyBet response under the canonical cache key."""

        cache_key = RedisClient.build_sportybet_markets_key(sportradar_id)
        snapshot = SportyBetMarketCacheEntry(
            sportradar_id=sportradar_id,
            fetched_at=self._clock().astimezone(UTC),
            markets=tuple(markets),
        )
        await self._cache.set(cache_key, snapshot)

    def _build_request_headers(self) -> dict[str, str]:
        """Build one outbound request header set with a rotated user agent."""

        headers = dict(SPORTYBET_DEFAULT_HEADERS)
        headers["User-Agent"] = next(self._user_agents)
        return headers

    @staticmethod
    def _tag_fetch_source(
        markets: Sequence[NormalizedOdds],
        *,
        fetch_source: str,
    ) -> tuple[NormalizedOdds, ...]:
        """Stamp the SportyBet fetch source onto normalized odds metadata."""

        tagged_rows: list[NormalizedOdds] = []
        for market in markets:
            tagged_rows.append(
                market.model_copy(
                    update={
                        "raw_metadata": {
                            **market.raw_metadata,
                            "sportybet_fetch_source": fetch_source,
                        }
                    }
                )
            )
        return tuple(tagged_rows)

    def _extract_matching_event(
        self,
        payload: object,
        requested_sportradar_id: str,
    ) -> Mapping[str, object]:
        """Select the event object matching the requested Sportradar fixture."""

        candidates = tuple(self._iter_event_candidates(payload))
        if not candidates:
            raise ProviderError(
                "sportybet",
                "response does not contain any event object with markets.",
            )

        exact_match = next(
            (
                candidate
                for candidate in candidates
                if requested_sportradar_id in self._extract_event_identifiers(candidate)
            ),
            None,
        )
        if exact_match is not None:
            return exact_match

        if len(candidates) == 1 and not self._extract_event_identifiers(candidates[0]):
            return candidates[0]

        raise ProviderError(
            "sportybet",
            (
                "response contains market data, but none of the discovered events matched "
                f"{requested_sportradar_id}."
            ),
        )

    def _iter_event_candidates(self, node: object) -> Iterator[Mapping[str, object]]:
        """Yield every nested mapping that looks like a SportyBet event payload."""

        if isinstance(node, Mapping):
            markets = self._extract_markets_container(node)
            if markets:
                yield node
            for value in node.values():
                yield from self._iter_event_candidates(value)
            return

        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                yield from self._iter_event_candidates(item)

    def _parse_markets(
        self,
        event: Mapping[str, object],
        *,
        sportradar_id: str,
        fixture: NormalizedFixture | None,
    ) -> tuple[NormalizedOdds, ...]:
        """Normalize all active outcomes from one SportyBet event payload."""

        event_catalog = self._build_event_catalog(
            event,
            sportradar_id=sportradar_id,
            fixture=fixture,
        )
        parsed_rows: list[NormalizedOdds] = []
        for market_group, market_node in event_catalog.iter_market_nodes():
            market_id = self._extract_market_id(market_node)
            provider_market_name = self._extract_market_name(market_node, market_id)
            market_specifier = self._extract_market_specifier(market_node)
            market = self._resolve_market_type(
                sport=event_catalog.fixture.sport,
                market_id=market_id,
                market_name=provider_market_name,
                specifier=market_specifier,
            )
            line = self._extract_market_line(
                market=market,
                market_name=provider_market_name,
                specifier=market_specifier,
            )
            for outcome_node in self._iter_outcome_nodes(market_node):
                if not self._is_active_outcome(outcome_node):
                    continue
                provider_selection_name = self._extract_selection_name(outcome_node)
                decimal_odds = self._extract_decimal_odds(outcome_node)
                if decimal_odds is None:
                    continue

                normalized_row = NormalizedOdds(
                    fixture_ref=event_catalog.fixture.fixture_ref,
                    market=market,
                    selection=self._normalize_selection_label(
                        provider_selection_name,
                        market=market,
                    ),
                    odds=decimal_odds,
                    provider="sportybet",
                    provider_market_name=provider_market_name,
                    provider_selection_name=provider_selection_name,
                    sportybet_available=True,
                    line=line,
                    period="match",
                    participant_scope="match",
                    provider_market_id=market_id,
                    raw_metadata=self._build_raw_metadata(
                        market_node=market_node,
                        outcome_node=outcome_node,
                        competition=event_catalog.fixture.competition,
                        home_team=event_catalog.fixture.home_team,
                        away_team=event_catalog.fixture.away_team,
                        requested_sportradar_id=sportradar_id,
                        event_id=event_catalog.event_id,
                        game_id=event_catalog.game_id,
                        total_market_size=event_catalog.total_market_size,
                        market_group_id=market_group.group_id,
                        market_group_name=market_group.group_name,
                    ),
                    last_updated=self._clock().astimezone(UTC),
                )
                parsed_rows.append(normalized_row)

        return tuple(parsed_rows)

    def _build_event_catalog(
        self,
        event: Mapping[str, object],
        *,
        sportradar_id: str,
        fixture: NormalizedFixture | None,
    ) -> SportyBetEventCatalog:
        """Arrange one SportyBet event payload into deterministic market groups."""

        fixture_context = self._build_fixture_context(
            event,
            sportradar_id=sportradar_id,
            fixture=fixture,
        )
        event_id = self._extract_optional_identifier(event, ("eventId", "matchId", "id"))
        game_id = self._extract_optional_identifier(event, ("gameId",))
        total_market_size = self._extract_optional_int(event, ("totalMarketSize",))

        grouped_markets: dict[str, list[Mapping[str, object]]] = {}
        group_names: dict[str, str] = {}
        group_order: list[str] = []
        ungrouped_index = 0

        for market_node in self._iter_market_nodes(event):
            group_id = self._extract_group_id(market_node)
            if group_id is None:
                ungrouped_index += 1
                group_id = f"ungrouped:{ungrouped_index}"
            group_name = self._extract_group_name(market_node) or "Ungrouped"
            if group_id not in grouped_markets:
                grouped_markets[group_id] = []
                group_names[group_id] = group_name
                group_order.append(group_id)
            grouped_markets[group_id].append(market_node)

        return SportyBetEventCatalog(
            fixture=fixture_context,
            requested_sportradar_id=sportradar_id,
            event_id=event_id,
            game_id=game_id,
            total_market_size=total_market_size,
            groups=tuple(
                SportyBetMarketGroup(
                    group_id=group_id,
                    group_name=group_names[group_id],
                    markets=tuple(grouped_markets[group_id]),
                )
                for group_id in group_order
            ),
        )

    def _build_fixture_context(
        self,
        event: Mapping[str, object],
        *,
        sportradar_id: str,
        fixture: NormalizedFixture | None,
    ) -> ParsedFixtureContext:
        """Resolve event-level display metadata reused across market rows."""

        if fixture is not None:
            return ParsedFixtureContext(
                fixture_ref=fixture.get_fixture_ref(),
                sport=fixture.sport,
                competition=fixture.competition,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
            )

        home_team = self._extract_text(
            event,
            ("homeTeamName", "homeName", "homeTeam"),
        )
        away_team = self._extract_text(
            event,
            ("awayTeamName", "awayName", "awayTeam"),
        )
        competition = self._extract_competition_name(event)
        sport = self._extract_sport(event)
        return ParsedFixtureContext(
            fixture_ref=sportradar_id,
            sport=sport,
            competition=competition,
            home_team=home_team,
            away_team=away_team,
        )

    @staticmethod
    def _extract_markets_container(event: Mapping[str, object]) -> Sequence[object]:
        """Return the raw markets collection from a SportyBet event object."""

        markets = event.get("markets")
        if isinstance(markets, Mapping):
            return tuple(markets.values())
        if isinstance(markets, Sequence) and not isinstance(markets, (str, bytes, bytearray)):
            return tuple(markets)
        return ()

    def _iter_market_nodes(self, event: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
        """Yield every market mapping attached to the selected event payload."""

        for market_node in self._extract_markets_container(event):
            if isinstance(market_node, Mapping):
                yield market_node
                continue
            if isinstance(market_node, Sequence) and not isinstance(
                market_node,
                (str, bytes, bytearray),
            ):
                for nested_market in market_node:
                    if isinstance(nested_market, Mapping):
                        yield nested_market

    @staticmethod
    def _extract_group_id(market_node: Mapping[str, object]) -> str | None:
        """Return the stable SportyBet market-group identifier when present."""

        for key in ("groupId", "marketGroupId"):
            raw_value = market_node.get(key)
            if isinstance(raw_value, int) and not isinstance(raw_value, bool):
                return str(raw_value)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _extract_group_name(market_node: Mapping[str, object]) -> str | None:
        """Return the SportyBet market-group label when present."""

        for key in ("group", "groupName"):
            raw_value = market_node.get(key)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        return None

    def _iter_outcome_nodes(
        self,
        market_node: Mapping[str, object],
    ) -> Iterator[Mapping[str, object]]:
        """Yield all outcome mappings attached to a market object."""

        outcomes = market_node.get("outcomes")
        if isinstance(outcomes, Mapping):
            for outcome in outcomes.values():
                if isinstance(outcome, Mapping):
                    yield outcome
            return

        if isinstance(outcomes, Sequence) and not isinstance(outcomes, (str, bytes, bytearray)):
            for outcome in outcomes:
                if isinstance(outcome, Mapping):
                    yield outcome

    @staticmethod
    def _extract_event_identifiers(event: Mapping[str, object]) -> frozenset[str]:
        """Collect all recognizable event identifiers from one event payload."""

        identifiers: set[str] = set()
        for key in (
            "eventId",
            "matchId",
            "gameId",
            "srMatchId",
            "sportradarId",
            "id",
        ):
            raw_value = event.get(key)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    identifiers.add(normalized)
        return frozenset(identifiers)

    @staticmethod
    def _extract_market_id(market_node: Mapping[str, object]) -> int | None:
        """Return one market identifier when the payload provides a numeric ID."""

        for key in ("id", "marketId"):
            raw_value = market_node.get(key)
            if isinstance(raw_value, bool):
                continue
            if isinstance(raw_value, int):
                return raw_value
            if isinstance(raw_value, str) and raw_value.strip().isdigit():
                return int(raw_value.strip())
        return None

    def _extract_market_name(self, market_node: Mapping[str, object], market_id: int | None) -> str:
        """Build the provider-facing market label for one SportyBet market."""

        name = self._extract_optional_text(market_node, ("name", "desc", "marketName"))
        if name is not None:
            return name
        if market_id is None:
            return "SportyBet Market"
        return f"SportyBet Market {market_id}"

    @staticmethod
    def _extract_market_specifier(market_node: Mapping[str, object]) -> str | None:
        """Return the provider market specifier when SportyBet supplies one."""

        for key in ("specifier", "marketSpecifiers", "marketSpecifier"):
            raw_value = market_node.get(key)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _extract_selection_name(outcome_node: Mapping[str, object]) -> str:
        """Return the provider-facing label for one market outcome."""

        for key in ("desc", "name", "outcomeName"):
            raw_value = outcome_node.get(key)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        raise ProviderError("sportybet", "outcome is missing a non-blank selection label.")

    def _extract_decimal_odds(self, outcome_node: Mapping[str, object]) -> float | None:
        """Parse SportyBet odds into decimal odds compatible with `NormalizedOdds`."""

        raw_odds = outcome_node.get("odds")
        if raw_odds is None:
            raw_odds = outcome_node.get("oddValue")
        if raw_odds is None:
            raw_odds = outcome_node.get("price")

        if isinstance(raw_odds, bool):
            return None
        if isinstance(raw_odds, str):
            stripped = raw_odds.strip()
            if not stripped:
                return None
            try:
                numeric_odds = float(stripped)
            except ValueError:
                return None
        elif isinstance(raw_odds, (int, float)):
            numeric_odds = float(raw_odds)
        else:
            return None

        if numeric_odds >= 1000:
            numeric_odds = numeric_odds / 1000.0
        elif numeric_odds >= 100:
            numeric_odds = numeric_odds / 100.0

        if numeric_odds <= 1.0:
            return None
        return round(numeric_odds, 3)

    @staticmethod
    def _is_active_outcome(outcome_node: Mapping[str, object]) -> bool:
        """Treat current SportyBet `isActive` flags as authoritative when present."""

        for key in ("isActive", "status"):
            raw_status = outcome_node.get(key)
            if raw_status is None:
                continue
            if isinstance(raw_status, bool):
                return raw_status
            if isinstance(raw_status, int):
                if key == "isActive":
                    return raw_status == 1
                return raw_status == 1
            if isinstance(raw_status, str):
                stripped = raw_status.strip().lower()
                if stripped in {"1", "true", "active", "open", "enabled"}:
                    return True
                if stripped in {"0", "false", "inactive", "suspended", "closed", "disabled"}:
                    return False
        return True

    def _resolve_market_type(
        self,
        *,
        sport: SportName,
        market_id: int | None,
        market_name: str,
        specifier: str | None,
    ) -> MarketType | None:
        """Infer PuntLab's canonical market taxonomy conservatively."""

        if market_id is not None and market_id in SPORTYBET_MARKET_IDS:
            return SPORTYBET_MARKET_IDS[market_id]

        normalized_name = market_name.casefold()
        normalized_specifier = (specifier or "").casefold()
        combined = f"{normalized_name} {normalized_specifier}"

        if sport == SportName.BASKETBALL:
            if "moneyline" in combined:
                return MarketType.MONEYLINE
            if "spread" in combined or "handicap" in combined:
                return MarketType.POINT_SPREAD
            if "total" in combined or "over/under" in combined:
                return MarketType.TOTAL_POINTS

        if "double chance" in combined:
            return MarketType.DOUBLE_CHANCE
        if "draw no bet" in combined:
            return MarketType.DRAW_NO_BET
        if "both teams to score" in combined or "btts" in combined:
            return MarketType.BTTS
        if "correct score" in combined:
            return MarketType.CORRECT_SCORE
        if "half time/full time" in combined or "ht/ft" in combined:
            return MarketType.HT_FT
        if "asian handicap" in combined:
            return MarketType.ASIAN_HANDICAP
        if "1x2" in combined or "match result" in combined or "full time result" in combined:
            return MarketType.MATCH_RESULT
        if "over/under" in combined or "total=" in combined:
            if "0.5" in combined:
                return MarketType.OVER_UNDER_05
            if "1.5" in combined:
                return MarketType.OVER_UNDER_15
            if "2.5" in combined:
                return MarketType.OVER_UNDER_25
            if "3.5" in combined:
                return MarketType.OVER_UNDER_35
        return None

    def _extract_market_line(
        self,
        *,
        market: MarketType | None,
        market_name: str,
        specifier: str | None,
    ) -> float | None:
        """Extract numeric market lines from SportyBet labels or specifiers."""

        normalized_market = market
        if normalized_market not in _LINE_MARKETS:
            return None

        raw_fragments = (specifier or "", market_name)
        for fragment in raw_fragments:
            parsed_line = self._find_numeric_line(fragment)
            if parsed_line is not None:
                return parsed_line
        return None

    @staticmethod
    def _find_numeric_line(fragment: str) -> float | None:
        """Parse the first decimal or integer value embedded in a string."""

        cleaned_fragment = fragment.replace("|", "&")
        for token in cleaned_fragment.split("&"):
            compact = token.strip()
            if "=" in compact:
                compact = compact.split("=", maxsplit=1)[1].strip()
            try:
                return float(compact)
            except ValueError:
                continue
        return None

    @staticmethod
    def _normalize_selection_label(
        selection_name: str,
        *,
        market: MarketType | None,
    ) -> str:
        """Normalize common SportyBet outcome labels without hiding raw values."""

        normalized = selection_name.strip()
        if market == MarketType.MATCH_RESULT:
            aliases = {
                "home": "Home",
                "1": "1",
                "draw": "Draw",
                "x": "X",
                "away": "Away",
                "2": "2",
            }
            return aliases.get(normalized.casefold(), normalized)
        return normalized

    @staticmethod
    def _build_raw_metadata(
        *,
        market_node: Mapping[str, object],
        outcome_node: Mapping[str, object],
        competition: str,
        home_team: str,
        away_team: str,
        requested_sportradar_id: str,
        event_id: str | None,
        game_id: str | None,
        total_market_size: int | None,
        market_group_id: str,
        market_group_name: str,
    ) -> dict[str, str | int | float | bool | None]:
        """Preserve SportyBet-native context that is useful for later resolution."""

        metadata: dict[str, str | int | float | bool | None] = {
            "competition": competition,
            "home_team": home_team,
            "away_team": away_team,
            "requested_sportradar_id": requested_sportradar_id,
            "event_id": event_id,
            "game_id": game_id,
            "event_total_market_size": total_market_size,
            "market_group_id": market_group_id,
            "market_group_name": market_group_name,
        }
        for key in (
            "specifier",
            "marketSpecifiers",
            "marketName",
            "name",
            "desc",
            "group",
            "groupId",
            "title",
            "product",
            "sourceType",
            "favourite",
            "banned",
        ):
            raw_value = market_node.get(key)
            if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                metadata[f"market_{key}"] = raw_value
        for key in ("id", "outcomeId", "name", "desc", "status", "isActive"):
            raw_value = outcome_node.get(key)
            if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                metadata[f"outcome_{key}"] = raw_value
        return metadata

    def _extract_competition_name(self, event: Mapping[str, object]) -> str:
        """Resolve the best available competition label from a SportyBet event."""

        sport_node = event.get("sport")
        if isinstance(sport_node, Mapping):
            nested_category = sport_node.get("category")
            if isinstance(nested_category, Mapping):
                nested_tournament = nested_category.get("tournament")
                if isinstance(nested_tournament, Mapping):
                    tournament_name = self._extract_optional_text(nested_tournament, ("name",))
                    if tournament_name is not None:
                        return tournament_name
                category_name = self._extract_optional_text(nested_category, ("name",))
                if category_name is not None:
                    return category_name

        tournament = event.get("category")
        if isinstance(tournament, Mapping):
            nested_tournament = tournament.get("tournament")
            if isinstance(nested_tournament, Mapping):
                tournament_name = self._extract_optional_text(nested_tournament, ("name",))
                if tournament_name is not None:
                    return tournament_name
            category_name = self._extract_optional_text(tournament, ("name",))
            if category_name is not None:
                return category_name

        tournament_name = self._extract_optional_text(event, ("tournamentName", "competitionName"))
        if tournament_name is not None:
            return tournament_name
        raise ProviderError("sportybet", "event payload is missing competition metadata.")

    def _extract_sport(self, event: Mapping[str, object]) -> SportName:
        """Map SportyBet sport metadata onto PuntLab's canonical sport enum."""

        sport_node = event.get("sport")
        if isinstance(sport_node, Mapping):
            sport_name = self._extract_optional_text(sport_node, ("name", "sportName"))
            if sport_name is not None:
                return self._normalize_sport_name(sport_name)

        for key in ("sportName", "sport"):
            raw_value = event.get(key)
            if isinstance(raw_value, str):
                return self._normalize_sport_name(raw_value)

        raise ProviderError("sportybet", "event payload is missing sport metadata.")

    @staticmethod
    def _normalize_sport_name(value: str) -> SportName:
        """Convert SportyBet sport labels into PuntLab's internal enum."""

        normalized = value.strip().casefold()
        if normalized in {"football", "soccer"}:
            return SportName.SOCCER
        if normalized in {"basketball", "nba"}:
            return SportName.BASKETBALL
        raise ProviderError("sportybet", f"unsupported SportyBet sport '{value}'.")

    @staticmethod
    def _extract_text(event: Mapping[str, object], keys: Sequence[str]) -> str:
        """Require a non-blank text field from one event mapping."""

        value = SportyBetAPIClient._extract_optional_text(event, keys)
        if value is None:
            key_list = ", ".join(keys)
            raise ProviderError("sportybet", f"event payload is missing one of: {key_list}.")
        return value

    @staticmethod
    def _extract_optional_text(
        event: Mapping[str, object],
        keys: Sequence[str],
    ) -> str | None:
        """Return the first non-blank text field present in a mapping."""

        for key in keys:
            raw_value = event.get(key)
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _extract_optional_identifier(
        event: Mapping[str, object],
        keys: Sequence[str],
    ) -> str | int | None:
        """Return the first non-blank text or integer identifier in a mapping."""

        for key in keys:
            raw_value = event.get(key)
            if isinstance(raw_value, int) and not isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _extract_optional_int(
        event: Mapping[str, object],
        keys: Sequence[str],
    ) -> int | None:
        """Return the first non-negative integer value present in a mapping."""

        for key in keys:
            raw_value = event.get(key)
            if isinstance(raw_value, bool):
                continue
            if isinstance(raw_value, int):
                return raw_value if raw_value >= 0 else None
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                if normalized.isdigit():
                    return int(normalized)
        return None

    @staticmethod
    def _normalize_sportradar_id(value: str) -> str:
        """Validate one outbound Sportradar identifier before any requests."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("sportradar_id must not be blank.")
        if not normalized.startswith("sr:match:"):
            raise ValueError("sportradar_id must follow the pattern `sr:match:<id>`.")
        return normalized

    @staticmethod
    def _normalize_template(value: str) -> str:
        """Validate one configured endpoint template at construction time."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("endpoint templates must not be blank.")
        if "{sportradar_id}" not in normalized:
            raise ValueError(
                "endpoint templates must contain the `{sportradar_id}` placeholder."
            )
        return normalized

    @staticmethod
    def _normalize_user_agent(value: str) -> str:
        """Reject blank user-agent values before round-robin rotation begins."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("user-agent values must not be blank.")
        return normalized


__all__ = [
    "DEFAULT_USER_AGENTS",
    "SPORTYBET_API_PATH_TEMPLATES",
    "SPORTYBET_BASE_URL",
    "SPORTYBET_COUNTRY_CODE",
    "SPORTYBET_MARKET_IDS",
    "SPORTYBET_RATE_LIMIT_POLICY",
    "SportyBetAPIClient",
    "SportyBetEventCatalog",
    "SportyBetMarketCacheEntry",
    "SportyBetMarketGroup",
]
