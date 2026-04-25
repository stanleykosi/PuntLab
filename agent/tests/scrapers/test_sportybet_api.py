"""Tests for PuntLab's primary SportyBet HTTP market interceptor.

Purpose: verify URL construction, endpoint fallback, response normalization,
user-agent rotation, and Redis-backed market caching without live requests.
Scope: pure unit tests for `src.scrapers.sportybet_api`.
Dependencies: pytest, httpx mock transports, the shared Redis cache wrapper,
and the normalized fixture and odds schemas used by the scraper.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import MarketType, SportName
from src.providers.base import ProviderError
from src.schemas.fixtures import NormalizedFixture
from src.scrapers.sportybet_api import (
    SPORTYBET_API_PATH_TEMPLATES,
    SportyBetAPIClient,
)


class FakeAsyncRedis:
    """Minimal async Redis stub for SportyBet scraper tests."""

    def __init__(self) -> None:
        """Initialize in-memory storage for values and expirations."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return one stored value by key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Store one value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment one numeric counter."""

        current_value = int(self.values.get(name, "0"))
        next_value = current_value + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Attach a TTL to one existing key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the stubbed Redis instance is always reachable."""

        return True

    async def aclose(self) -> None:
        """Match the Redis protocol expected by `RedisClient`."""


def build_fixture() -> NormalizedFixture:
    """Create a canonical Arsenal-Chelsea fixture for URL tests."""

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


def build_match_detail_payload() -> dict[str, object]:
    """Return a representative SportyBet match-detail response payload."""

    return {
        "data": {
            "event": {
                "eventId": "sr:match:61301159",
                "totalMarketSize": 42,
                "homeTeamName": "Arsenal",
                "awayTeamName": "Chelsea",
                "sport": {"id": "sr:sport:1", "name": "Football"},
                "category": {
                    "name": "England",
                    "tournament": {"id": "sr:tournament:17", "name": "Premier League"},
                },
                "markets": [
                    {
                        "id": 1,
                        "name": "1X2",
                        "groupId": "1001",
                        "group": "Main",
                        "outcomes": [
                            {"id": "1", "desc": "Home", "odds": 183, "isActive": True},
                            {"id": "2", "desc": "Draw", "odds": 350, "isActive": True},
                            {"id": "3", "desc": "Away", "odds": 420, "isActive": True},
                        ],
                    },
                    {
                        "id": 18,
                        "name": "Over/Under",
                        "specifier": "total=2.5",
                        "groupId": "1002",
                        "group": "Goals",
                        "outcomes": [
                            {"id": "425", "desc": "Over 2.5", "odds": 191, "isActive": 1},
                            {"id": "426", "desc": "Under 2.5", "odds": 182, "isActive": 1},
                        ],
                    },
                ],
            }
        }
    }


@pytest.mark.asyncio
async def test_build_sportybet_url_uses_fixture_segments() -> None:
    """The public SportyBet URL should follow the spec's canonical path pattern."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    client = SportyBetAPIClient(cache)

    assert (
        client.build_sportybet_url(build_fixture())
        == "https://www.sportybet.com/ng/sport/football/england/"
        "premier-league/Arsenal_vs_Chelsea/sr:match:61301159"
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_markets_normalizes_payload_and_reuses_cache() -> None:
    """Successful requests should normalize markets once and cache the result."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url.path == "/api/ng/factsCenter/event"
        assert request.url.params["eventId"] == "sr:match:61301159"
        return httpx.Response(200, json=build_match_detail_payload(), request=request)

    client = SportyBetAPIClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=lambda: datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
    )

    first_result = await client.fetch_markets("sr:match:61301159", fixture=build_fixture())
    second_result = await client.fetch_markets("sr:match:61301159", fixture=build_fixture())

    assert request_count == 1
    assert len(first_result) == 5
    assert second_result == first_result
    assert first_result[0].market == MarketType.MATCH_RESULT
    assert first_result[0].provider_market_id == 1
    assert first_result[0].odds == pytest.approx(1.83)
    assert first_result[0].raw_metadata["market_group_id"] == "1001"
    assert first_result[0].raw_metadata["market_group_name"] == "Main"
    assert first_result[0].raw_metadata["event_total_market_size"] == 42
    assert first_result[0].raw_metadata["sportybet_fetch_source"] == "api"
    assert first_result[3].market == MarketType.OVER_UNDER_25
    assert first_result[3].line == pytest.approx(2.5)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_markets_resolves_interval_goal_market_endpoint() -> None:
    """SportyBet interval labels must replace X with the fetched endpoint minute."""

    payload = build_match_detail_payload()
    payload["data"]["event"]["markets"].append(
        {
            "id": 99901,
            "name": "Total Goals from 1 to X min",
            "specifier": "from=1|to=15|total=0.5",
            "groupId": "1003",
            "group": "Intervals",
            "outcomes": [
                {"id": "u015", "desc": "Under 0.5", "odds": 112, "isActive": True},
                {"id": "o015", "desc": "Over 0.5", "odds": 550, "isActive": True},
            ],
        }
    )
    cache = RedisClient(redis_client=FakeAsyncRedis())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = SportyBetAPIClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    rows = await client.fetch_markets("sr:match:61301159", fixture=build_fixture())
    interval_rows = [
        row for row in rows if row.provider_market_name == "Total Goals from 1 to 15 min"
    ]

    assert len(interval_rows) == 2
    assert all(row.market_label == "Total Goals from 1 to 15 min" for row in interval_rows)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_markets_falls_back_to_second_endpoint_and_rotates_user_agents() -> None:
    """Endpoint fallback should continue after one failed template and rotate headers."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    requested_paths: list[str] = []
    request_user_agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query_fragment = request.url.query.decode()
        requested_paths.append(
            request.url.path + (f"?{query_fragment}" if query_fragment else "")
        )
        request_user_agents.append(request.headers["User-Agent"])
        if request.url.path == "/api/ng/factsCenter/event":
            return httpx.Response(404, json={"message": "missing"}, request=request)
        return httpx.Response(200, json=build_match_detail_payload(), request=request)

    client = SportyBetAPIClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        user_agents=("agent-one", "agent-two"),
    )

    result = await client.fetch_markets("sr:match:61301159")

    assert len(result) == 5
    assert requested_paths == [
        "/api/ng/factsCenter/event?eventId=sr:match:61301159",
        "/api/ng/factsCenter/pc/matchDetail?eventId=sr:match:61301159",
    ]
    assert request_user_agents == ["agent-one", "agent-two"]

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_markets_raises_when_no_matching_event_is_present() -> None:
    """Responses without the requested fixture should fail with a clear error."""

    cache = RedisClient(redis_client=FakeAsyncRedis())

    payload = {
        "data": {
            "event": {
                "eventId": "sr:match:99999999",
                "homeTeamName": "Other",
                "awayTeamName": "Fixture",
                "sport": {"name": "Football"},
                "category": {"tournament": {"name": "Premier League"}},
                "markets": [{"id": 1, "name": "1X2", "outcomes": [{"desc": "Home", "odds": 190}]}],
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = SportyBetAPIClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_path_templates=SPORTYBET_API_PATH_TEMPLATES[:1],
    )

    with pytest.raises(ProviderError, match="none of the discovered events matched"):
        await client.fetch_markets("sr:match:61301159")

    await client.aclose()
