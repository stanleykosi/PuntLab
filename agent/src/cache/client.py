"""Redis cache client for PuntLab's transient data layer.

Purpose: provide the canonical async Redis interface for provider response
caching, LLM output caching, pipeline checkpoints, and rate-limit tracking.
Scope: cache key construction, default TTL resolution, JSON serialization, and
basic async Redis operations used by later provider and pipeline steps.
Dependencies: relies on `redis.asyncio` for network I/O and `src.config` for
the default Redis connection URL and WAT-aware daily rate-limit keys.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Protocol, TypeVar, cast
from uuid import UUID

import redis.asyncio as redis_asyncio
from pydantic import BaseModel
from redis import exceptions as redis_exceptions

from src.config import get_settings

API_FOOTBALL_FIXTURES_TTL_SECONDS = 2 * 60 * 60
API_ODDS_TTL_SECONDS = 30 * 60
API_STATS_TTL_SECONDS = 6 * 60 * 60
SPORTYBET_MARKETS_TTL_SECONDS = 60 * 60
RATE_LIMIT_TTL_SECONDS = 24 * 60 * 60
LLM_CONTEXT_TTL_SECONDS = 4 * 60 * 60
PIPELINE_STATE_TTL_SECONDS = 12 * 60 * 60

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
CacheModelT = TypeVar("CacheModelT", bound=BaseModel)
Identifier = str | int | UUID
logger = logging.getLogger(__name__)
_TRANSIENT_REDIS_ERRORS = (
    redis_exceptions.ConnectionError,
    redis_exceptions.TimeoutError,
)


class SupportsRedisClient(Protocol):
    """Protocol describing the async Redis methods the cache client depends on."""

    async def get(self, name: str) -> str | bytes | None:
        """Return the string value for a Redis key, if present."""

    async def set(self, name: str, value: str, ex: int | None = None) -> bool | None:
        """Persist a string value under a Redis key with an optional TTL."""

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment a numeric key and return the new value."""

    async def expire(self, name: str, time: int) -> bool:
        """Set a TTL on an existing Redis key."""

    async def ping(self) -> bool:
        """Check whether the Redis server is reachable."""

    async def aclose(self) -> None:
        """Close the underlying client connection pool."""


@dataclass(frozen=True, slots=True)
class CacheTTLConfig:
    """Default TTL configuration for every canonical PuntLab cache key family."""

    api_football_fixtures: int = API_FOOTBALL_FIXTURES_TTL_SECONDS
    api_odds: int = API_ODDS_TTL_SECONDS
    api_stats: int = API_STATS_TTL_SECONDS
    sportybet_markets: int = SPORTYBET_MARKETS_TTL_SECONDS
    rate_limit: int = RATE_LIMIT_TTL_SECONDS
    llm_context: int = LLM_CONTEXT_TTL_SECONDS
    pipeline_state: int = PIPELINE_STATE_TTL_SECONDS


class RedisClient:
    """Typed async Redis wrapper used by provider and pipeline infrastructure.

    Args:
        redis_client: Optional injected async Redis client for tests or custom
            wiring. When omitted, the client is created from configuration.
        redis_url: Optional explicit Redis URL used only when `redis_client`
            is not supplied.
        ttl_config: Optional TTL overrides for the canonical key families.
    """

    def __init__(
        self,
        redis_client: SupportsRedisClient | None = None,
        *,
        redis_url: str | None = None,
        ttl_config: CacheTTLConfig | None = None,
    ) -> None:
        """Initialize the Redis cache wrapper with canonical defaults."""

        if redis_client is None:
            connection_url = redis_url or get_settings().redis.url
            redis_client = cast(
                SupportsRedisClient,
                redis_asyncio.Redis.from_url(
                    connection_url,
                    encoding="utf-8",
                    decode_responses=True,
                ),
            )

        self._redis = redis_client
        self._ttl_config = ttl_config or CacheTTLConfig()

    @staticmethod
    def build_api_football_fixtures_key(run_date: date | str) -> str:
        """Build the canonical fixtures cache key for API-Football data."""

        return f"api:football:fixtures:{RedisClient._normalize_date_fragment(run_date)}"

    @staticmethod
    def build_api_odds_key(fixture_id: Identifier) -> str:
        """Build the canonical odds cache key for one fixture."""

        return f"api:odds:{RedisClient._normalize_identifier('fixture_id', fixture_id)}"

    @staticmethod
    def build_api_stats_key(team_id: Identifier) -> str:
        """Build the canonical team statistics cache key for one team."""

        return f"api:stats:{RedisClient._normalize_identifier('team_id', team_id)}"

    @staticmethod
    def build_sportybet_markets_key(sportradar_id: str) -> str:
        """Build the canonical SportyBet market cache key for one fixture."""

        return f"sportybet:markets:{RedisClient._normalize_text('sportradar_id', sportradar_id)}"

    @staticmethod
    def build_rate_limit_key(provider: str, for_date: date | str | None = None) -> str:
        """Build the canonical daily rate-limit counter key for one provider."""

        normalized_provider = RedisClient._normalize_text("provider", provider).lower()
        normalized_date = RedisClient._normalize_date_fragment(
            for_date or RedisClient._current_wat_date()
        )
        return f"ratelimit:{normalized_provider}:{normalized_date}"

    @staticmethod
    def build_llm_context_key(fixture_id: Identifier) -> str:
        """Build the canonical key for LLM-generated fixture context."""

        return f"llm:context:{RedisClient._normalize_identifier('fixture_id', fixture_id)}"

    @staticmethod
    def build_pipeline_state_key(run_id: Identifier) -> str:
        """Build the canonical pipeline checkpoint key for one run."""

        return f"pipeline:state:{RedisClient._normalize_identifier('run_id', run_id)}"

    async def get(
        self,
        key: str,
        *,
        model: type[CacheModelT] | None = None,
    ) -> CacheModelT | JSONValue | None:
        """Fetch and deserialize a cached JSON value.

        Args:
            key: Redis key to load.
            model: Optional Pydantic model used to validate the decoded payload.

        Returns:
            The cached value, optionally validated into `model`, or `None` when
            the key is absent.

        Raises:
            ValueError: If the key is blank or the stored payload is not valid JSON.
        """

        normalized_key = self._normalize_text("key", key)
        try:
            payload = await self._redis.get(normalized_key)
        except _TRANSIENT_REDIS_ERRORS as exc:
            logger.warning("Retrying Redis GET for key '%s' after: %s", normalized_key, exc)
            payload = await self._redis.get(normalized_key)
        if payload is None:
            return None

        serialized = payload.decode("utf-8") if isinstance(payload, bytes) else payload

        try:
            decoded = cast(JSONValue, json.loads(serialized))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cached value for '{normalized_key}' is not valid JSON.") from exc

        if model is None:
            return decoded

        return model.model_validate(decoded)

    async def set(
        self,
        key: str,
        value: BaseModel | JSONValue,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Serialize and store a value under a Redis key.

        Args:
            key: Redis key to write.
            value: JSON-compatible data or a Pydantic model instance.
            ttl_seconds: Optional explicit TTL. When omitted, the TTL is
                resolved from the canonical key pattern defined in the spec.

        Returns:
            `True` when Redis accepted the write.

        Raises:
            ValueError: If the key is blank, the TTL is invalid, or no default
                TTL exists for the supplied key family.
            TypeError: If `value` cannot be serialized into JSON.
        """

        normalized_key = self._normalize_text("key", key)
        resolved_ttl = (
            self.resolve_ttl_seconds(normalized_key)
            if ttl_seconds is None
            else ttl_seconds
        )
        validated_ttl = self._validate_ttl_seconds(resolved_ttl)
        serialized = json.dumps(
            value,
            default=self._json_default,
            separators=(",", ":"),
        )

        try:
            result = await self._redis.set(normalized_key, serialized, ex=validated_ttl)
        except _TRANSIENT_REDIS_ERRORS as exc:
            logger.warning("Retrying Redis SET for key '%s' after: %s", normalized_key, exc)
            result = await self._redis.set(normalized_key, serialized, ex=validated_ttl)
        return bool(result)

    async def increment(
        self,
        key: str,
        *,
        amount: int = 1,
        ttl_seconds: int | None = None,
    ) -> int:
        """Increment a numeric Redis key and ensure it has a TTL.

        Args:
            key: Redis key to increment.
            amount: Positive integer increment size.
            ttl_seconds: Optional explicit TTL. Defaults to the canonical TTL
                for the key family when omitted.

        Returns:
            The new numeric value stored in Redis.

        Raises:
            ValueError: If the key is blank, `amount` is not positive, or the
                TTL is invalid / cannot be resolved for the key family.
        """

        normalized_key = self._normalize_text("key", key)
        if amount <= 0:
            raise ValueError("amount must be a positive integer.")

        resolved_ttl = (
            self.resolve_ttl_seconds(normalized_key)
            if ttl_seconds is None
            else ttl_seconds
        )
        validated_ttl = self._validate_ttl_seconds(resolved_ttl)
        try:
            current_value = await self._redis.incr(normalized_key, amount=amount)
        except _TRANSIENT_REDIS_ERRORS as exc:
            logger.warning("Retrying Redis INCR for key '%s' after: %s", normalized_key, exc)
            current_value = await self._redis.incr(normalized_key, amount=amount)

        # Only the first increment for a fresh key should attach the TTL so the
        # counter window remains anchored to the start of that rate-limit day.
        if current_value == amount:
            try:
                await self._redis.expire(normalized_key, validated_ttl)
            except _TRANSIENT_REDIS_ERRORS as exc:
                logger.warning(
                    "Retrying Redis EXPIRE for key '%s' after: %s",
                    normalized_key,
                    exc,
                )
                await self._redis.expire(normalized_key, validated_ttl)

        return current_value

    async def get_rate_count(self, provider: str, *, for_date: date | str | None = None) -> int:
        """Return the daily API call count for one provider."""

        key = self.build_rate_limit_key(provider, for_date)
        try:
            raw_value = await self._redis.get(key)
        except _TRANSIENT_REDIS_ERRORS as exc:
            logger.warning("Retrying Redis GET for key '%s' after: %s", key, exc)
            raw_value = await self._redis.get(key)
        if raw_value is None:
            return 0

        serialized = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
        try:
            return int(serialized)
        except ValueError as exc:
            raise ValueError(f"Rate-limit counter for '{key}' is not an integer.") from exc

    async def is_rate_limited(
        self,
        provider: str,
        *,
        limit: int,
        for_date: date | str | None = None,
    ) -> bool:
        """Return whether a provider has reached or exceeded its daily limit."""

        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        current_count = await self.get_rate_count(provider, for_date=for_date)
        return current_count >= limit

    async def ping(self) -> bool:
        """Check whether the underlying Redis server is reachable."""

        try:
            return await self._redis.ping()
        except _TRANSIENT_REDIS_ERRORS as exc:
            logger.warning("Retrying Redis PING after: %s", exc)
            return await self._redis.ping()

    async def close(self) -> None:
        """Close the underlying async Redis client."""

        await self._redis.aclose()

    def resolve_ttl_seconds(self, key: str) -> int:
        """Resolve the default TTL for a canonical cache key.

        Args:
            key: Redis key whose TTL should be inferred.

        Returns:
            The configured TTL in seconds for the matching cache family.

        Raises:
            ValueError: If the key is blank or does not match a canonical cache
                family from the technical specification.
        """

        normalized_key = self._normalize_text("key", key)

        key_prefix_to_ttl = (
            ("api:football:fixtures:", self._ttl_config.api_football_fixtures),
            ("api:odds:", self._ttl_config.api_odds),
            ("api:stats:", self._ttl_config.api_stats),
            ("sportybet:markets:", self._ttl_config.sportybet_markets),
            ("ratelimit:", self._ttl_config.rate_limit),
            ("llm:context:", self._ttl_config.llm_context),
            ("pipeline:state:", self._ttl_config.pipeline_state),
        )

        for prefix, ttl_seconds in key_prefix_to_ttl:
            if normalized_key.startswith(prefix):
                return self._validate_ttl_seconds(ttl_seconds)

        raise ValueError(
            "No default TTL is configured for cache key "
            f"'{normalized_key}'. Use a canonical key builder or pass ttl_seconds explicitly."
        )

    @staticmethod
    def _normalize_text(field_name: str, value: str) -> str:
        """Trim and validate a non-empty string fragment used in a Redis key."""

        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{field_name} must not be blank.")
        return candidate

    @staticmethod
    def _normalize_identifier(field_name: str, value: Identifier) -> str:
        """Convert a supported identifier into a safe non-empty key fragment."""

        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, int):
            return str(value)

        return RedisClient._normalize_text(field_name, value)

    @staticmethod
    def _normalize_date_fragment(value: date | str) -> str:
        """Convert a date object or ISO date string into a cache key fragment."""

        if isinstance(value, date):
            return value.isoformat()

        candidate = RedisClient._normalize_text("date", value)
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError as exc:
            raise ValueError("date must be a `date` object or ISO-8601 date string.") from exc

    @staticmethod
    def _current_wat_date() -> date:
        """Return the current calendar date in PuntLab's canonical WAT timezone."""

        return datetime.now(get_settings().timezone).date()

    @staticmethod
    def _validate_ttl_seconds(ttl_seconds: int) -> int:
        """Ensure TTL values are positive integers before sending them to Redis."""

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer.")
        return ttl_seconds

    @staticmethod
    def _json_default(value: object) -> JSONValue:
        """Serialize supported non-primitive objects into JSON-compatible data."""

        if isinstance(value, BaseModel):
            return cast(JSONValue, value.model_dump(mode="json"))

        if isinstance(value, (date, datetime)):
            return value.isoformat()

        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, Enum):
            enum_value = value.value
            if not isinstance(enum_value, (str, int, float, bool)) and enum_value is not None:
                raise TypeError(
                    "Enum values for Redis caching must be JSON-compatible, "
                    f"got {type(enum_value)!r}."
                )
            return cast(JSONValue, enum_value)

        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


__all__ = [
    "API_FOOTBALL_FIXTURES_TTL_SECONDS",
    "API_ODDS_TTL_SECONDS",
    "API_STATS_TTL_SECONDS",
    "CacheTTLConfig",
    "LLM_CONTEXT_TTL_SECONDS",
    "PIPELINE_STATE_TTL_SECONDS",
    "RATE_LIMIT_TTL_SECONDS",
    "RedisClient",
    "SPORTYBET_MARKETS_TTL_SECONDS",
]
