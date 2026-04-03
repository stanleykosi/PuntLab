"""Tests for PuntLab's canonical provider base infrastructure.

Purpose: verify Redis-backed rate limiting, request caching, retry/backoff,
and the abstract provider fetch contract without live network access.
Scope: pure unit tests for `src.providers.base`.
Dependencies: pytest, httpx mock transports, and a lightweight in-memory Redis
stub compatible with `src.cache.client.RedisClient`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE
from src.providers.base import (
    DataProvider,
    RateLimitedClient,
    RateLimitExhausted,
    RateLimitPolicy,
    RetryConfig,
)


class FakeAsyncRedis:
    """Minimal in-memory async Redis stub for provider infrastructure tests."""

    def __init__(self) -> None:
        """Initialize value, TTL, and close-tracking stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.closed = False

    async def get(self, name: str) -> str | None:
        """Return a stored key value if present."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Persist a string value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment a numeric string key by the supplied amount."""

        current_value = int(self.values.get(name, "0"))
        next_value = current_value + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Attach a TTL to an existing stored key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the in-memory Redis stub is always healthy."""

        return True

    async def aclose(self) -> None:
        """Mark the fake Redis client as closed."""

        self.closed = True


class ExampleProvider(DataProvider):
    """Concrete provider used to exercise the shared `DataProvider` contract."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier."""

        return "example-provider"

    @property
    def base_url(self) -> str:
        """Return the provider base URL used for relative paths."""

        return "https://provider.test/api"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return a simple daily limit policy for test requests."""

        return RateLimitPolicy(limit=5, window_seconds=86_400)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the provider's default authentication header."""

        return {"Authorization": "Bearer secret-token"}

    @property
    def default_cache_ttl_seconds(self) -> int | None:
        """Enable cache-by-default for the test provider."""

        return 300


@pytest.mark.asyncio
async def test_request_caches_successful_responses_and_skips_duplicate_network_calls() -> None:
    """Repeated identical requests should be served from Redis after the first call."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={"provider": "api-football", "fixtures": 12},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
    )

    first_response = await client.request(
        "api-football",
        "GET",
        "https://provider.test/fixtures",
        rate_limit_policy=RateLimitPolicy(limit=5, window_seconds=86_400),
        cache_ttl_seconds=300,
        params={"date": "2026-04-03"},
    )
    second_response = await client.request(
        "api-football",
        "GET",
        "https://provider.test/fixtures",
        rate_limit_policy=RateLimitPolicy(limit=5, window_seconds=86_400),
        cache_ttl_seconds=300,
        params={"date": "2026-04-03"},
    )

    assert call_count == 1
    assert first_response.extensions["from_cache"] is False
    assert second_response.extensions["from_cache"] is True
    assert second_response.json() == {"provider": "api-football", "fixtures": 12}
    assert await cache.get_rate_count("api-football", for_date="2026-04-03") == 1

    await client.aclose()


@pytest.mark.asyncio
async def test_request_retries_transient_failures_with_exponential_backoff() -> None:
    """Transient retryable responses should back off and eventually recover."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    call_count = 0
    sleep_calls: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        """Record retry delays instead of waiting in real time."""

        sleep_calls.append(delay_seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        status_code = 503 if call_count < 3 else 200
        return httpx.Response(status_code, json={"attempt": call_count}, request=request)

    client = RateLimitedClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay_seconds=0.25,
            backoff_multiplier=2.0,
            max_delay_seconds=1.0,
        ),
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
        sleep=fake_sleep,
    )

    response = await client.request(
        "api-football",
        "GET",
        "https://provider.test/odds",
        rate_limit_policy=RateLimitPolicy(limit=10, window_seconds=86_400),
        use_cache=False,
    )

    assert call_count == 3
    assert sleep_calls == [0.25, 0.5]
    assert response.status_code == 200
    assert response.json() == {"attempt": 3}
    assert await cache.get_rate_count("api-football", for_date="2026-04-03") == 3

    await client.aclose()


@pytest.mark.asyncio
async def test_request_raises_before_network_when_local_rate_limit_is_exhausted() -> None:
    """Local Redis counters should fail fast before another network attempt is sent."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    network_attempted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_attempted
        network_attempted = True
        return httpx.Response(200, json={"ok": True}, request=request)

    rate_limit_key = RedisClient.build_rate_limit_key("api-football", "2026-04-03")
    await cache.increment(rate_limit_key, amount=2, ttl_seconds=86_400)
    client = RateLimitedClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=lambda: datetime(2026, 4, 3, 8, 0, tzinfo=WAT_TIMEZONE),
    )

    with pytest.raises(RateLimitExhausted, match="Rate limit exhausted"):
        await client.request(
            "api-football",
            "GET",
            "https://provider.test/fixtures",
            rate_limit_policy=RateLimitPolicy(limit=2, window_seconds=86_400),
            use_cache=False,
        )

    assert network_attempted is False

    await client.aclose()


@pytest.mark.asyncio
async def test_data_provider_fetch_builds_absolute_urls_and_merges_headers() -> None:
    """Concrete providers should inherit URL resolution and header merging via `fetch()`."""

    redis_stub = FakeAsyncRedis()
    cache = RedisClient(redis_client=redis_stub)
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = dict(request.headers)
        assert str(request.url) == "https://provider.test/api/fixtures?date=2026-04-03"
        return httpx.Response(200, json={"ok": True}, request=request)

    client = RateLimitedClient(
        cache,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=lambda: datetime(2026, 4, 3, 9, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = ExampleProvider(client)

    response = await provider.fetch(
        "GET",
        "/fixtures",
        headers={"X-Request-ID": "req-123"},
        params={"date": "2026-04-03"},
    )

    assert response.status_code == 200
    assert captured_headers["authorization"] == "Bearer secret-token"
    assert captured_headers["x-request-id"] == "req-123"
    assert await cache.get_rate_count("example-provider", for_date="2026-04-03") == 1

    await client.aclose()
