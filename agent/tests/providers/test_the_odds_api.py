"""Tests for PuntLab's The Odds API provider integration.

Purpose: verify request shaping, quota-aware defaults, and normalization from
The Odds API payloads into PuntLab's shared odds schema without live network
calls.
Scope: unit tests for sport-level and event-level odds responses using
`httpx.MockTransport` and an in-memory Redis stub.
Dependencies: pytest, httpx, the shared `RedisClient`, and the concrete
`TheOddsAPIProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, MarketType
from src.providers.base import RateLimitedClient
from src.providers.the_odds_api import TheOddsAPIProvider


class FakeAsyncRedis:
    """Minimal async Redis stub for provider unit tests."""

    def __init__(self) -> None:
        """Initialize in-memory value and TTL stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return a stored string value for one key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Persist a string value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment a stored numeric string value."""

        current_value = int(self.values.get(name, "0"))
        next_value = current_value + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Persist a TTL for an existing key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the in-memory Redis store is always available."""

        return True

    async def aclose(self) -> None:
        """Close the fake client without side effects."""


def build_provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[TheOddsAPIProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = TheOddsAPIProvider(rate_limited_client, api_key="test-key")
    return provider, rate_limited_client


@pytest.mark.asyncio
async def test_fetch_odds_normalizes_soccer_markets_and_preserves_unmapped_rows() -> None:
    """Sport odds fetches should map supported soccer markets and keep the rest."""

    observed_query: dict[str, str] = {}
    observed_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_query, observed_path
        observed_query = dict(request.url.params.items())
        observed_path = request.url.path
        return httpx.Response(
            200,
            headers={
                "x-requests-remaining": "412",
                "x-requests-used": "88",
                "x-requests-last": "2",
            },
            json=[
                {
                    "id": "event-123",
                    "sport_key": "soccer_epl",
                    "sport_title": "Premier League",
                    "commence_time": "2026-04-03T19:45:00Z",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "bookmakers": [
                        {
                            "key": "pinnacle",
                            "title": "Pinnacle",
                            "last_update": "2026-04-03T11:30:00Z",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Arsenal", "price": 1.88},
                                        {"name": "Draw", "price": 3.55},
                                        {"name": "Chelsea", "price": 4.30},
                                    ],
                                },
                                {
                                    "key": "totals",
                                    "outcomes": [
                                        {"name": "Over", "price": 1.91, "point": 2.5},
                                        {"name": "Under", "price": 1.89, "point": 2.5},
                                    ],
                                },
                                {
                                    "key": "btts",
                                    "outcomes": [
                                        {"name": "Yes", "price": 1.73},
                                        {"name": "No", "price": 2.06},
                                    ],
                                },
                                {
                                    "key": "team_totals",
                                    "outcomes": [
                                        {"name": "Arsenal Over 1.5", "price": 2.10, "point": 1.5},
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ],
            request=request,
        )

    provider, client = build_provider(handler)

    odds_rows = await provider.fetch_odds(
        sport_key="soccer_epl",
        markets=("h2h", "totals"),
    )

    assert observed_path == "/v4/sports/soccer_epl/odds"
    assert observed_query == {
        "apiKey": "test-key",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "regions": "eu",
    }
    assert len(odds_rows) == 8
    assert {row.market for row in odds_rows} == {
        MarketType.MATCH_RESULT,
        MarketType.OVER_UNDER_25,
        MarketType.BTTS,
        None,
    }

    home_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.MATCH_RESULT and row.selection == "home"
    )
    over_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.OVER_UNDER_25 and row.selection == "over"
    )
    btts_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.BTTS and row.selection == "yes"
    )
    unmapped_row = next(row for row in odds_rows if row.provider_market_name == "team_totals")

    assert home_row.fixture_ref == "the-odds-api:event-123"
    assert home_row.provider == "Pinnacle"
    assert home_row.last_updated == datetime(2026, 4, 3, 11, 30, tzinfo=UTC)
    assert over_row.line == 2.5
    assert btts_row.provider_selection_name == "Yes"
    assert unmapped_row.market is None
    assert unmapped_row.selection == "Arsenal Over 1.5"
    assert unmapped_row.participant_scope == "team"
    assert unmapped_row.raw_metadata["sport_key"] == "soccer_epl"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_event_odds_normalizes_basketball_featured_and_player_markets() -> None:
    """Event odds fetches should support arbitrary event markets without dropping rows."""

    observed_query: dict[str, str] = {}
    observed_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_query, observed_path
        observed_query = dict(request.url.params.items())
        observed_path = request.url.path
        return httpx.Response(
            200,
            headers={
                "x-requests-remaining": "33",
                "x-requests-used": "467",
                "x-requests-last": "4",
            },
            json={
                "id": "nba-555",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2026-04-03T23:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": "2026-04-03T16:05:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "price": 1.72},
                                    {"name": "Boston Celtics", "price": 2.15},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "price": 1.91, "point": -4.5},
                                    {"name": "Boston Celtics", "price": 1.91, "point": 4.5},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": 1.95, "point": 228.5},
                                    {"name": "Under", "price": 1.87, "point": 228.5},
                                ],
                            },
                            {
                                "key": "player_points",
                                "outcomes": [
                                    {"name": "LeBron James Over 27.5", "price": 1.84},
                                ],
                            },
                        ],
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    odds_rows = await provider.fetch_event_odds(
        sport_key="basketball_nba",
        event_id="nba-555",
        markets=("h2h", "spreads", "totals", "player_points"),
        regions=("us",),
        include_multipliers=True,
    )

    assert observed_path == "/v4/sports/basketball_nba/events/nba-555/odds"
    assert observed_query == {
        "apiKey": "test-key",
        "markets": "h2h,spreads,totals,player_points",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "regions": "us",
        "includeMultipliers": "true",
    }
    assert len(odds_rows) == 7
    assert {row.market for row in odds_rows} == {
        MarketType.MONEYLINE,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
        None,
    }

    moneyline_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.MONEYLINE and row.selection == "home"
    )
    spread_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.POINT_SPREAD and row.selection == "home"
    )
    total_row = next(
        row
        for row in odds_rows
        if row.market == MarketType.TOTAL_POINTS and row.selection == "over"
    )
    player_prop_row = next(row for row in odds_rows if row.provider_market_name == "player_points")

    assert moneyline_row.fixture_ref == "the-odds-api:nba-555"
    assert spread_row.line == -4.5
    assert total_row.line == 228.5
    assert player_prop_row.market is None
    assert player_prop_row.participant_scope == "player"
    assert player_prop_row.provider_market_id == "player_points"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_event_odds_rejects_regions_and_bookmakers_together() -> None:
    """Canonical request shaping should reject ambiguous region and bookmaker filters."""

    provider, client = build_provider(
        lambda request: httpx.Response(200, json={}, request=request)
    )

    with pytest.raises(ValueError, match="cannot be supplied together"):
        await provider.fetch_event_odds(
            sport_key="basketball_nba",
            event_id="nba-555",
            markets=("h2h",),
            regions=("us",),
            bookmakers=("draftkings",),
        )

    await client.aclose()
