"""Tests for PuntLab's canonical Redis cache client.

Purpose: verify cache key generation, TTL resolution, JSON serialization, and
rate-limit tracking without requiring a live Redis server.
Scope: pure unit tests for `src.cache.client`.
Dependencies: pytest, Pydantic, and a lightweight in-memory async Redis stub.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel
from src.cache.client import (
    API_FOOTBALL_FIXTURES_TTL_SECONDS,
    LLM_CONTEXT_TTL_SECONDS,
    RATE_LIMIT_TTL_SECONDS,
    RedisClient,
)


class FakeAsyncRedis:
    """Minimal async Redis stub used to unit test the cache client."""

    def __init__(self) -> None:
        """Initialize the in-memory value and TTL stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.closed = False

    async def get(self, name: str) -> str | None:
        """Return a stored value by key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Store a value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment an integer counter stored as a string."""

        current_value = int(self.values.get(name, "0"))
        next_value = current_value + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Set a TTL for an existing key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the in-memory Redis stub is always healthy."""

        return True

    async def aclose(self) -> None:
        """Mark the stub client as closed."""

        self.closed = True


class CachedFixtureContext(BaseModel):
    """Simple Pydantic payload used to exercise typed Redis round-trips."""

    fixture_id: str
    narrative: str
    generated_at: datetime


def test_key_builders_follow_the_specification() -> None:
    """The canonical Redis keys should match Section 16 of the spec exactly."""

    assert RedisClient.build_api_football_fixtures_key(date(2026, 4, 3)) == (
        "api:football:fixtures:2026-04-03"
    )
    assert RedisClient.build_api_odds_key("fixture-123") == "api:odds:fixture-123"
    assert RedisClient.build_api_stats_key(44) == "api:stats:44"
    assert RedisClient.build_sportybet_markets_key("sr:match:61301159") == (
        "sportybet:markets:sr:match:61301159"
    )
    assert RedisClient.build_rate_limit_key("api-football", date(2026, 4, 3)) == (
        "ratelimit:api-football:2026-04-03"
    )
    assert RedisClient.build_llm_context_key("fixture-123") == "llm:context:fixture-123"
    assert RedisClient.build_pipeline_state_key("run-123") == "pipeline:state:run-123"


def test_resolve_ttl_seconds_matches_canonical_key_families() -> None:
    """Known key families should inherit their specification-defined TTLs."""

    client = RedisClient(redis_client=FakeAsyncRedis())

    assert client.resolve_ttl_seconds("api:football:fixtures:2026-04-03") == (
        API_FOOTBALL_FIXTURES_TTL_SECONDS
    )
    assert client.resolve_ttl_seconds("llm:context:fixture-123") == LLM_CONTEXT_TTL_SECONDS


def test_resolve_ttl_seconds_rejects_unknown_keys() -> None:
    """Unknown key prefixes should fail fast instead of silently picking a TTL."""

    client = RedisClient(redis_client=FakeAsyncRedis())

    with pytest.raises(ValueError, match="No default TTL is configured"):
        client.resolve_ttl_seconds("custom:namespace:key")


@pytest.mark.asyncio
async def test_set_and_get_round_trip_pydantic_models_with_default_ttls() -> None:
    """Cached Pydantic models should serialize to JSON and validate on read."""

    redis_stub = FakeAsyncRedis()
    client = RedisClient(redis_client=redis_stub)
    cache_key = RedisClient.build_llm_context_key("fixture-42")
    payload = CachedFixtureContext(
        fixture_id="fixture-42",
        narrative="The away side is missing two starting defenders.",
        generated_at=datetime(2026, 4, 3, 8, 30, tzinfo=UTC),
    )

    write_result = await client.set(cache_key, payload)
    cached_payload = await client.get(cache_key, model=CachedFixtureContext)

    assert write_result is True
    assert redis_stub.expirations[cache_key] == LLM_CONTEXT_TTL_SECONDS
    assert cached_payload == payload


@pytest.mark.asyncio
async def test_increment_tracks_rate_limits_and_sets_counter_ttl_once() -> None:
    """Daily provider counters should increment and inherit the rate-limit TTL."""

    redis_stub = FakeAsyncRedis()
    client = RedisClient(redis_client=redis_stub)
    cache_date = date(2026, 4, 3)
    rate_key = RedisClient.build_rate_limit_key("api-football", cache_date)

    first_count = await client.increment(rate_key)
    second_count = await client.increment(rate_key)
    stored_count = await client.get_rate_count("api-football", for_date=cache_date)
    is_limited = await client.is_rate_limited("api-football", limit=2, for_date=cache_date)

    assert first_count == 1
    assert second_count == 2
    assert stored_count == 2
    assert is_limited is True
    assert redis_stub.expirations[rate_key] == RATE_LIMIT_TTL_SECONDS


@pytest.mark.asyncio
async def test_close_delegates_to_the_underlying_async_client() -> None:
    """Closing the cache client should close the underlying Redis connection."""

    redis_stub = FakeAsyncRedis()
    client = RedisClient(redis_client=redis_stub)

    await client.close()

    assert redis_stub.closed is True
