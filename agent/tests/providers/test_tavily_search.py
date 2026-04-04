"""Tests for PuntLab's Tavily search provider integration.

Purpose: verify Tavily request shaping, sports-news normalization, and
malformed-response handling without live network calls.
Scope: unit tests for `src.providers.tavily_search.TavilySearchProvider`
backed by `httpx.MockTransport` and an in-memory Redis stub.
Dependencies: pytest, httpx, the shared `RedisClient`, and the concrete
Tavily provider implementation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, SportName
from src.providers.base import ProviderError, RateLimitedClient
from src.providers.tavily_search import TavilySearchProvider
from src.schemas.fixtures import NormalizedFixture


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
) -> tuple[TavilySearchProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = TavilySearchProvider(
        rate_limited_client,
        api_key="tvly-test-key",
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    return provider, rate_limited_client


def build_fixture() -> NormalizedFixture:
    """Create a canonical soccer fixture used by Tavily provider tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:9001",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 3, 19, 45, tzinfo=UTC),
        source_provider="api-football",
        source_id="9001",
        country="England",
    )


@pytest.mark.asyncio
async def test_search_match_news_shapes_request_and_normalizes_articles() -> None:
    """Fixture searches should shape Tavily requests and normalize news rows."""

    observed_headers: dict[str, str] = {}
    observed_payload: dict[str, object] = {}
    observed_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_headers, observed_payload, observed_path
        observed_headers = dict(request.headers.items())
        observed_payload = json.loads(request.content.decode("utf-8"))
        observed_path = request.url.path
        return httpx.Response(
            200,
            json={
                "request_id": "req-123",
                "results": [
                    {
                        "title": "Arsenal vs Chelsea preview with key lineup notes",
                        "url": "https://www.bbc.com/sport/football/preview-example",
                        "content": "Both clubs enter the match with strong recent attacking form.",
                        "score": 0.91,
                        "published_date": "2026-04-03",
                    },
                    {
                        "title": "Chelsea prepare for another London derby",
                        "url": "https://www.espn.com/soccer/story/_/id/fixture-preview",
                        "content": "Chelsea are monitoring midfield fitness ahead of kickoff.",
                        "score": 0.73,
                        "published_date": "2026-04-02T14:30:00Z",
                    },
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    articles = await provider.search_match_news(fixture=build_fixture())

    assert observed_path == "/search"
    assert observed_headers["authorization"] == "Bearer tvly-test-key"
    assert observed_payload == {
        "query": "Arsenal Chelsea Premier League match preview team news",
        "topic": "news",
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "exact_match": False,
        "start_date": "2026-03-28",
        "end_date": "2026-04-03",
    }

    assert len(articles) == 2
    assert articles[0].source_provider == "tavily"
    assert articles[0].source == "BBC Sport"
    assert articles[0].published_at == datetime(2026, 4, 3, 0, 0, tzinfo=UTC)
    assert articles[0].sport == SportName.SOCCER
    assert articles[0].competition == "Premier League"
    assert articles[0].teams == ("Arsenal", "Chelsea")
    assert articles[0].fixture_ref == "sr:match:9001"
    assert articles[0].relevance_score == pytest.approx(0.91)
    assert articles[1].published_at == datetime(2026, 4, 2, 14, 30, tzinfo=UTC)

    await client.aclose()


@pytest.mark.asyncio
async def test_search_injury_updates_normalizes_domains_and_skips_invalid_rows() -> None:
    """Injury searches should normalize domain filters and ignore bad result rows."""

    observed_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_payload
        observed_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "request_id": "req-456",
                "results": [
                    {
                        "title": "Chelsea face late fitness checks before Arsenal trip",
                        "url": "https://www.skysports.com/football/news/chelsea-fitness-checks",
                        "content": (
                            "Mauricio Pochettino will assess two midfielders before kickoff."
                        ),
                        "score": "0.67",
                        "published_date": "2026-04-01T08:15:00+00:00",
                    },
                    {
                        "title": "Malformed article missing publish date",
                        "url": "https://www.goal.com/en/news/malformed-item",
                        "content": (
                            "This row should be skipped because the provider omitted a date."
                        ),
                        "score": 0.4,
                    },
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    articles = await provider.search_injury_updates(
        fixture=build_fixture(),
        max_results=2,
        search_depth="advanced",
        lookback_days=3,
        include_domains=("https://www.bbc.com", "espn.com", "bbc.com"),
    )

    assert observed_payload == {
        "query": "Arsenal Chelsea Premier League injuries suspensions lineup team news",
        "topic": "news",
        "search_depth": "advanced",
        "max_results": 2,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "exact_match": False,
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "include_domains": ["bbc.com", "espn.com"],
    }
    assert len(articles) == 1
    assert articles[0].source == "Sky Sports"
    assert articles[0].relevance_score == pytest.approx(0.67)

    await client.aclose()


@pytest.mark.asyncio
async def test_search_breaking_news_raises_for_missing_results_list() -> None:
    """Malformed Tavily envelopes should fail fast with a provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"request_id": "req-789", "query": "NBA breaking news"},
            request=request,
        )

    provider, client = build_provider(handler)

    with pytest.raises(ProviderError, match="results"):
        await provider.search_breaking_news(sport=SportName.BASKETBALL)

    await client.aclose()
