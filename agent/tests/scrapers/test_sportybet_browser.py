"""Tests for PuntLab's Playwright-based SportyBet browser fallback scraper.

Purpose: verify browser-page loading, embedded payload parsing, DOM-market
normalization, timeout handling, and Redis-backed caching without launching a
real browser in unit tests.
Scope: pure unit tests for `src.scrapers.sportybet_browser` using injected fake
page sessions and the shared normalized schemas.
Dependencies: pytest, the Redis cache wrapper, shared fixture schemas, and the
browser scraper itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal, cast

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from src.cache.client import RedisClient
from src.config import MarketType, SportName
from src.providers.base import ProviderError
from src.schemas.fixtures import NormalizedFixture
from src.scrapers.sportybet_browser import PageSessionFactory, SportyBetBrowserScraper


class FakeAsyncRedis:
    """Minimal async Redis stub for browser scraper tests."""

    def __init__(self) -> None:
        """Initialize in-memory storage for values and expirations."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return one stored value by key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Store one value and an optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment one numeric key and return the new value."""

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
        """Match the async Redis protocol expected by `RedisClient`."""


class FakePage:
    """Injected Playwright-like page stub used by browser scraper tests."""

    def __init__(
        self,
        snapshot: object,
        *,
        goto_error: Exception | None = None,
        wait_selector_error: Exception | None = None,
    ) -> None:
        """Capture the page snapshot and optional lifecycle failures."""

        self.snapshot = snapshot
        self.goto_error = goto_error
        self.wait_selector_error = wait_selector_error
        self.goto_calls: list[tuple[str, str | None, float | None]] = []
        self.load_state_calls: list[tuple[str, float | None]] = []
        self.wait_selector_calls: list[tuple[str, float | None]] = []
        self.evaluate_calls: list[str] = []

    async def goto(
        self,
        url: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] | None = None,
        referer: str | None = None,
    ) -> object:
        """Record page navigation and optionally raise a timeout."""

        del referer
        self.goto_calls.append((url, wait_until, timeout))
        if self.goto_error is not None:
            raise self.goto_error
        return {"url": url}

    async def wait_for_load_state(
        self,
        state: Literal["domcontentloaded", "load", "networkidle"] | None = None,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> None:
        """Record load-state waits without changing test behavior."""

        self.load_state_calls.append((state or "load", timeout))

    async def wait_for_selector(
        self,
        selector: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> object | None:
        """Record selector waits and optionally simulate a timeout."""

        self.wait_selector_calls.append((selector, timeout))
        if self.wait_selector_error is not None:
            raise self.wait_selector_error
        return {"selector": selector}

    async def evaluate(self, expression: str) -> object:
        """Return the injected browser evaluation snapshot."""

        self.evaluate_calls.append(expression)
        return self.snapshot


def build_fixture() -> NormalizedFixture:
    """Create a canonical Arsenal-Chelsea fixture for browser scraper tests."""

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


def build_event_snapshot() -> dict[str, object]:
    """Return a representative embedded SportyBet event payload snapshot."""

    return {
        "event_payloads": [
            {
                "eventId": "sr:match:61301159",
                "homeTeamName": "Arsenal",
                "awayTeamName": "Chelsea",
                "sport": {"id": "sr:sport:1", "name": "Football"},
                "category": {
                    "name": "England",
                    "tournament": {"id": "sr:tournament:17", "name": "Premier League"},
                },
                "markets": [
                    {
                        "id": 45,
                        "name": "1X2",
                        "outcomes": [
                            {"id": "1", "desc": "Home", "odds": 183, "status": 1},
                            {"id": "2", "desc": "Draw", "odds": 350, "status": 1},
                            {"id": "3", "desc": "Away", "odds": 420, "status": 1},
                        ],
                    },
                    {
                        "id": 47,
                        "name": "Over/Under",
                        "specifier": "total=2.5",
                        "outcomes": [
                            {"id": "425", "desc": "Over 2.5", "odds": 191, "status": 1},
                            {"id": "426", "desc": "Under 2.5", "odds": 182, "status": 1},
                        ],
                    },
                ],
            }
        ],
        "dom_markets": [],
        "title": "Arsenal vs Chelsea",
        "heading": "Arsenal vs Chelsea",
        "competition": "Premier League",
        "sport": "football",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    }


def build_dom_snapshot() -> dict[str, object]:
    """Return a DOM-only browser snapshot without embedded event payloads."""

    return {
        "event_payloads": [],
        "dom_markets": [
            {
                "market_id": "45",
                "market_name": "1X2",
                "outcomes": [
                    {
                        "selection_id": "1",
                        "selection_name": "Home",
                        "odds": "1.83",
                        "status": "active",
                    },
                    {
                        "selection_id": "2",
                        "selection_name": "Draw",
                        "odds": "3.50",
                        "status": "active",
                    },
                    {
                        "selection_id": "3",
                        "selection_name": "Away",
                        "odds": "4.20",
                        "status": "active",
                    },
                ],
            },
            {
                "market_id": "47",
                "market_name": "Over/Under",
                "specifier": "total=2.5",
                "outcomes": [
                    {
                        "selection_id": "425",
                        "selection_name": "Over 2.5",
                        "odds": "1.91",
                        "status": "active",
                    },
                    {
                        "selection_id": "426",
                        "selection_name": "Under 2.5",
                        "odds": "1.82",
                        "status": "active",
                    },
                ],
            },
        ],
        "title": "Arsenal vs Chelsea",
        "heading": "Arsenal vs Chelsea",
        "competition": "Premier League",
        "sport": "football",
        "home_team": "",
        "away_team": "",
    }


def build_page_factory(
    pages: list[FakePage],
    user_agents_seen: list[str],
) -> PageSessionFactory:
    """Create an injected page-session factory that yields fake pages in order."""

    @asynccontextmanager
    async def factory(user_agent: str) -> AsyncIterator[FakePage]:
        user_agents_seen.append(user_agent)
        if not pages:
            raise AssertionError("No fake pages remain for the requested browser session.")
        yield pages.pop(0)

    return cast(PageSessionFactory, factory)


@pytest.mark.asyncio
async def test_scrape_markets_uses_embedded_event_payloads_and_cache() -> None:
    """Embedded page-state events should normalize once and then reuse cache."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    pages = [FakePage(build_event_snapshot())]
    user_agents_seen: list[str] = []
    scraper = SportyBetBrowserScraper(
        cache,
        page_session_factory=build_page_factory(pages, user_agents_seen),
        clock=lambda: datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        user_agents=("ua-one", "ua-two"),
    )

    url = (
        "https://www.sportybet.com/ng/sport/football/england/premier-league/"
        "Arsenal_vs_Chelsea/sr:match:61301159"
    )
    first_result = await scraper.scrape_markets(url, fixture=build_fixture())
    second_result = await scraper.scrape_markets(url, fixture=build_fixture())

    assert len(first_result) == 5
    assert second_result == first_result
    assert user_agents_seen == ["ua-one"]
    assert first_result[0].market == MarketType.MATCH_RESULT
    assert first_result[3].market == MarketType.OVER_UNDER_25
    assert first_result[3].line == pytest.approx(2.5)

    await scraper.aclose()


@pytest.mark.asyncio
async def test_scrape_markets_uses_dom_fallback_after_selector_timeout() -> None:
    """Selector waits should fail softly when DOM snapshots still contain markets."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    timeout_error = PlaywrightTimeoutError("selector timeout")
    pages = [FakePage(build_dom_snapshot(), wait_selector_error=timeout_error)]
    user_agents_seen: list[str] = []
    scraper = SportyBetBrowserScraper(
        cache,
        page_session_factory=build_page_factory(pages, user_agents_seen),
        user_agents=("ua-one", "ua-two"),
    )

    url = (
        "https://www.sportybet.com/ng/sport/football/england/premier-league/"
        "Arsenal_vs_Chelsea/sr:match:61301159"
    )
    result = await scraper.scrape_markets(url, fixture=build_fixture(), use_cache=False)

    assert len(result) == 5
    assert user_agents_seen == ["ua-one"]
    assert result[0].provider == "sportybet"
    assert result[0].market == MarketType.MATCH_RESULT
    assert result[3].market == MarketType.OVER_UNDER_25
    assert result[4].selection == "Under 2.5"

    await scraper.aclose()


@pytest.mark.asyncio
async def test_scrape_markets_raises_provider_error_when_navigation_times_out() -> None:
    """Navigation failures should surface a clear provider error."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    pages = [
        FakePage(
            build_dom_snapshot(),
            goto_error=PlaywrightTimeoutError("navigation timeout"),
        )
    ]
    scraper = SportyBetBrowserScraper(
        cache,
        page_session_factory=build_page_factory(pages, []),
    )

    url = (
        "https://www.sportybet.com/ng/sport/football/england/premier-league/"
        "Arsenal_vs_Chelsea/sr:match:61301159"
    )
    with pytest.raises(ProviderError, match="Timed out while waiting for SportyBet page"):
        await scraper.scrape_markets(url, fixture=build_fixture(), use_cache=False)

    await scraper.aclose()
