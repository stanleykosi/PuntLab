"""Tests for PuntLab's RSS feed news provider.

Purpose: verify RSS feed parsing, fixture-aware article matching, duplicate
handling, and hard-failure behavior when all configured feeds are unavailable.
Scope: unit tests for `src.providers.rss_feeds.RSSFeedProvider` backed by
`httpx.MockTransport` and an in-memory Redis stub.
Dependencies: pytest, httpx, the shared `RedisClient`, and the concrete RSS
provider implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, SportName
from src.providers.base import ProviderError, RateLimitedClient
from src.providers.rss_feeds import RSSFeedDefinition, RSSFeedProvider
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
    *,
    feeds: tuple[RSSFeedDefinition, ...],
) -> tuple[RSSFeedProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 4, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = RSSFeedProvider(
        rate_limited_client,
        feeds=feeds,
        clock=lambda: datetime(2026, 4, 4, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    return provider, rate_limited_client


def build_fixtures() -> tuple[NormalizedFixture, ...]:
    """Create canonical fixtures used by the RSS provider tests."""

    return (
        NormalizedFixture(
            sportradar_id="sr:match:9001",
            home_team="Arsenal",
            away_team="Chelsea",
            competition="Premier League",
            sport=SportName.SOCCER,
            kickoff=datetime(2026, 4, 4, 19, 45, tzinfo=UTC),
            source_provider="api-football",
            source_id="9001",
            country="England",
        ),
        NormalizedFixture(
            sportradar_id="sr:match:9002",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            competition="NBA",
            sport=SportName.BASKETBALL,
            kickoff=datetime(2026, 4, 4, 2, 0, tzinfo=UTC),
            source_provider="balldontlie",
            source_id="9002",
            country="United States",
        ),
    )


@pytest.mark.asyncio
async def test_fetch_news_filters_fixture_relevant_articles_and_deduplicates_urls() -> None:
    """Relevant entries should be matched to fixtures and duplicate URLs collapsed."""

    soccer_feed = RSSFeedDefinition(
        source="BBC Sport",
        url="https://feeds.example.test/soccer.xml",
        sport=SportName.SOCCER,
    )
    nba_feed = RSSFeedDefinition(
        source="ESPN",
        url="https://feeds.example.test/nba.xml",
        sport=SportName.BASKETBALL,
        competition="NBA",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/soccer.xml":
            return httpx.Response(
                200,
                text="""<?xml version="1.0" encoding="UTF-8"?>
                <rss version="2.0">
                  <channel>
                    <title>BBC Sport Football</title>
                    <item>
                      <title>Arsenal vs Chelsea preview as title race tightens</title>
                      <link>https://www.bbc.com/sport/football/preview-1</link>
                      <pubDate>Sat, 04 Apr 2026 05:30:00 GMT</pubDate>
                      <description>Premier League preview with likely lineups.</description>
                      <author>BBC Sport staff</author>
                      <guid>preview-1</guid>
                    </item>
                    <item>
                      <title>Arsenal injury update before Chelsea clash</title>
                      <link>https://www.bbc.com/sport/football/preview-1</link>
                      <pubDate>Sat, 04 Apr 2026 04:45:00 GMT</pubDate>
                      <description>Premier League injury news for Arsenal.</description>
                      <guid>preview-1-duplicate</guid>
                    </item>
                    <item>
                      <title>Bundesliga roundup from Friday night</title>
                      <link>https://www.bbc.com/sport/football/roundup-1</link>
                      <pubDate>Sat, 04 Apr 2026 02:00:00 GMT</pubDate>
                      <description>Not relevant to today's tracked fixtures.</description>
                      <guid>roundup-1</guid>
                    </item>
                  </channel>
                </rss>""",
                headers={"Content-Type": "application/rss+xml"},
                request=request,
            )

        if request.url.path == "/nba.xml":
            return httpx.Response(
                200,
                text="""<?xml version="1.0" encoding="UTF-8"?>
                <rss version="2.0">
                  <channel>
                    <title>ESPN NBA</title>
                    <item>
                      <title>Lakers vs Celtics injury report ahead of NBA showdown</title>
                      <link>https://www.espn.com/nba/story/_/id/9002/lakers-celtics-injury-report</link>
                      <pubDate>Sat, 04 Apr 2026 03:30:00 GMT</pubDate>
                      <description>The NBA leaders could both miss starters tonight.</description>
                      <guid>nba-1</guid>
                    </item>
                  </channel>
                </rss>""",
                headers={"Content-Type": "application/rss+xml"},
                request=request,
            )

        raise AssertionError(f"Unexpected URL requested: {request.url}")

    provider, client = build_provider(handler, feeds=(soccer_feed, nba_feed))

    articles = await provider.fetch_news(fixtures=build_fixtures())

    assert len(articles) == 2

    soccer_article = next(
        article for article in articles if str(article.url) == "https://www.bbc.com/sport/football/preview-1"
    )
    assert soccer_article.source_provider == "rss"
    assert soccer_article.source == "BBC Sport"
    assert soccer_article.fixture_ref == "sr:match:9001"
    assert soccer_article.competition == "Premier League"
    assert soccer_article.sport == SportName.SOCCER
    assert soccer_article.teams == ("Arsenal", "Chelsea")
    assert soccer_article.author == "BBC Sport staff"
    assert soccer_article.relevance_score == pytest.approx(1.0)

    nba_article = next(
        article
        for article in articles
        if str(article.url)
        == "https://www.espn.com/nba/story/_/id/9002/lakers-celtics-injury-report"
    )
    assert nba_article.fixture_ref == "sr:match:9002"
    assert nba_article.competition == "NBA"
    assert nba_article.sport == SportName.BASKETBALL
    assert nba_article.teams == ("Los Angeles Lakers", "Boston Celtics")
    assert nba_article.relevance_score == pytest.approx(1.0)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_news_raises_when_every_feed_fails() -> None:
    """The provider should fail fast when no configured feed can be fetched."""

    broken_feed = RSSFeedDefinition(
        source="Broken Feed",
        url="https://feeds.example.test/broken.xml",
        sport=SportName.SOCCER,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable", request=request)

    provider, client = build_provider(handler, feeds=(broken_feed,))

    with pytest.raises(ProviderError, match="All configured RSS feeds failed"):
        await provider.fetch_news(fixtures=build_fixtures())

    await client.aclose()
