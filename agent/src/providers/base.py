"""Canonical provider infrastructure for PuntLab's ingestion integrations.

Purpose: provide the shared HTTP client wrapper, rate-limit enforcement, and
provider base class that all external sports, odds, and news integrations use.
Scope: Redis-backed response caching, retry/backoff behavior, provider-scoped
rate-limit tracking, and the abstract `DataProvider` contract.
Dependencies: `httpx` for async HTTP I/O, `src.cache.client.RedisClient` for
cache and counter persistence, and `src.config.get_settings` for WAT-aware
rate-limit window bucketing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.cache.client import RATE_LIMIT_TTL_SECONDS, RedisClient
from src.config import get_settings

logger = logging.getLogger(__name__)

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Provider-specific rate-limit configuration for network requests.

    Inputs:
        Static provider settings describing the maximum number of requests that
        may be executed within a fixed rolling bucket window.

    Outputs:
        A validated, immutable policy object consumed by `RateLimitedClient`
        before every outbound request.
    """

    limit: int
    window_seconds: int = RATE_LIMIT_TTL_SECONDS

    def __post_init__(self) -> None:
        """Reject invalid policy definitions at construction time."""

        if self.limit <= 0:
            raise ValueError("Rate-limit `limit` must be a positive integer.")
        if self.window_seconds <= 0:
            raise ValueError("Rate-limit `window_seconds` must be a positive integer.")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry policy used for transient provider errors and flaky upstreams.

    Inputs:
        Optional per-request overrides or a wrapper-level default.

    Outputs:
        A validated immutable retry configuration with exponential backoff.
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 8.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        """Validate retry configuration values before use."""

        if self.max_attempts <= 0:
            raise ValueError("Retry `max_attempts` must be a positive integer.")
        if self.initial_delay_seconds <= 0:
            raise ValueError("Retry `initial_delay_seconds` must be positive.")
        if self.backoff_multiplier < 1:
            raise ValueError("Retry `backoff_multiplier` must be at least 1.")
        if self.max_delay_seconds <= 0:
            raise ValueError("Retry `max_delay_seconds` must be positive.")
        if not self.retryable_status_codes:
            raise ValueError("Retry `retryable_status_codes` must not be empty.")


class CachedHTTPResponse(BaseModel):
    """Serializable HTTP response payload stored in Redis.

    Inputs:
        One successful provider response returned by `httpx`.

    Outputs:
        A JSON-safe model that can be reconstructed into an equivalent
        `httpx.Response` instance on cache reads.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    method: str = Field(description="Uppercase HTTP method used for the request.")
    url: str = Field(description="Fully-qualified request URL.")
    status_code: int = Field(ge=200, le=299, description="Successful HTTP status code.")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Serializable response headers preserved for cached reads.",
    )
    content_base64: str = Field(description="Base64-encoded raw response body bytes.")
    _REPLAY_UNSAFE_HEADERS: ClassVar[frozenset[str]] = frozenset(
        {"content-encoding", "transfer-encoding", "content-length"}
    )

    @field_validator("method", "url", "content_base64")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject blank strings in serialized response payloads."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("Cached HTTP response fields must not be blank.")
        return normalized

    @classmethod
    def from_response(cls, response: httpx.Response) -> CachedHTTPResponse:
        """Create a serializable cache payload from one successful response."""

        request = response.request
        content_bytes = response.content
        return cls(
            method=request.method.upper(),
            url=str(request.url),
            status_code=response.status_code,
            headers=cls._sanitize_replay_headers(response.headers),
            content_base64=base64.b64encode(content_bytes).decode("ascii"),
        )

    def to_response(self) -> httpx.Response:
        """Reconstruct a cached payload into an `httpx.Response` object."""

        request = httpx.Request(self.method, self.url)
        # Defensive normalization guards against legacy cache entries that were
        # persisted before header sanitization was introduced.
        replay_headers = self._sanitize_replay_headers(self.headers)
        response = httpx.Response(
            self.status_code,
            headers=replay_headers,
            content=base64.b64decode(self.content_base64.encode("ascii")),
            request=request,
        )
        response.extensions["from_cache"] = True
        return response

    @classmethod
    def _sanitize_replay_headers(cls, headers: Mapping[str, str]) -> dict[str, str]:
        """Drop headers that can make cached body replay invalid.

        Inputs:
            headers: Upstream response headers captured during network fetch.

        Outputs:
            A replay-safe header mapping without transport/compression metadata
            that can conflict with cached body bytes during reconstruction.
        """

        sanitized_headers: dict[str, str] = {}
        for key, value in headers.items():
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                continue
            if normalized_key.lower() in cls._REPLAY_UNSAFE_HEADERS:
                continue
            sanitized_headers[normalized_key] = normalized_value
        return sanitized_headers


class ProviderError(RuntimeError):
    """Base exception raised when a provider request cannot be completed."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Capture provider metadata alongside the failure message."""

        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.__cause__ = cause


class RateLimitExhausted(ProviderError):
    """Raised when local or upstream rate limits prevent a request."""

    def __init__(
        self,
        provider: str,
        *,
        limit: int | None = None,
        window_seconds: int | None = None,
        retry_after_seconds: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Build a helpful error message for rate-limit failures."""

        message_parts = [f"Rate limit exhausted for provider '{provider}'."]
        if limit is not None and window_seconds is not None:
            message_parts.append(
                f"Configured limit: {limit} requests per {window_seconds} seconds."
            )
        if retry_after_seconds is not None:
            message_parts.append(f"Retry after approximately {retry_after_seconds} seconds.")
        if detail is not None:
            message_parts.append(detail)

        super().__init__(
            provider,
            " ".join(message_parts),
            status_code=429,
        )
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds


class RateLimitedClient:
    """Shared async HTTP client with caching, retries, and provider limits.

    Args:
        cache: Canonical Redis cache wrapper used for response caching and
            provider request counters.
        http_client: Optional injected async HTTP client for tests or custom
            transports. Defaults to a shared `httpx.AsyncClient`.
        retry_config: Default retry policy applied when a request does not
            provide its own override.
        clock: Optional clock injector used for deterministic window keys.
        sleep: Optional async sleep function used for retry backoff.
    """

    def __init__(
        self,
        cache: RedisClient,
        *,
        http_client: httpx.AsyncClient | None = None,
        retry_config: RetryConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize the wrapper with canonical PuntLab defaults."""

        self._cache = cache
        self._http_client = http_client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=20.0,
        )
        self._retry_config = retry_config or RetryConfig()
        self._clock = clock or (lambda: datetime.now(get_settings().timezone))
        self._sleep = sleep

    async def request(
        self,
        provider: str,
        method: str,
        url: str,
        *,
        rate_limit_policy: RateLimitPolicy,
        cache_ttl_seconds: int | None = None,
        use_cache: bool = True,
        retry_config: RetryConfig | None = None,
        cache_vary_by_headers: Iterable[str] = (),
        **kwargs: object,
    ) -> httpx.Response:
        """Execute one provider request with cache lookup and limit enforcement.

        Inputs:
            provider: Stable provider identifier, such as `api-football`.
            method: HTTP method used for the request.
            url: Fully-qualified provider URL.
            rate_limit_policy: Provider limit definition applied before every
                network attempt.
            cache_ttl_seconds: Optional response cache TTL. When omitted, the
                request bypasses response caching.
            use_cache: Whether cached successful responses may satisfy the call.
            retry_config: Optional request-specific retry override.
            cache_vary_by_headers: Header names that should affect the cache
                key when a provider's response varies by header value.
            **kwargs: Additional `httpx.AsyncClient.request()` arguments.

        Outputs:
            An `httpx.Response` either from the network or reconstructed from
            Redis on a cache hit.

        Raises:
            RateLimitExhausted: When the local or upstream rate limit is hit.
            ProviderError: When the request fails permanently.
        """

        normalized_provider = self._normalize_text("provider", provider)
        normalized_method = self._normalize_text("method", method).upper()
        normalized_url = self._normalize_text("url", url)
        resolved_retry = retry_config or self._retry_config
        request_headers = self._extract_headers(kwargs)

        cache_key: str | None = None
        if use_cache and cache_ttl_seconds is not None:
            cache_key = self._build_response_cache_key(
                provider=normalized_provider,
                method=normalized_method,
                url=normalized_url,
                kwargs=kwargs,
                headers=request_headers,
                cache_vary_by_headers=cache_vary_by_headers,
            )
            cached_entry = await self._cache.get(cache_key, model=CachedHTTPResponse)
            if isinstance(cached_entry, CachedHTTPResponse):
                logger.debug(
                    "Cache hit for provider=%s url=%s",
                    normalized_provider,
                    normalized_url,
                )
                try:
                    return cached_entry.to_response()
                except (ValueError, httpx.DecodingError) as exc:
                    logger.warning(
                        "Discarding unreadable cached response for provider=%s url=%s: %s",
                        normalized_provider,
                        normalized_url,
                        exc,
                    )

        last_error: ProviderError | None = None
        for attempt in range(1, resolved_retry.max_attempts + 1):
            await self._assert_within_rate_limit(normalized_provider, rate_limit_policy)
            await self._increment_rate_counter(normalized_provider, rate_limit_policy)

            try:
                request_kwargs = cast(dict[str, Any], dict(kwargs))
                response = await self._http_client.request(
                    normalized_method,
                    normalized_url,
                    **request_kwargs,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "Provider request failed for provider=%s url=%s attempt=%s/%s: %s",
                    normalized_provider,
                    normalized_url,
                    attempt,
                    resolved_retry.max_attempts,
                    exc,
                )
                last_error = ProviderError(
                    normalized_provider,
                    (
                        f"Provider request failed for '{normalized_provider}' after "
                        f"attempt {attempt}: {exc!s}"
                    ),
                    cause=exc,
                )
                if attempt == resolved_retry.max_attempts:
                    raise last_error from exc
                await self._sleep(self._compute_backoff_delay(resolved_retry, attempt))
                continue

            response.extensions["from_cache"] = False
            if response.status_code == 429:
                retry_after_seconds = self._parse_retry_after_seconds(response)
                if attempt < resolved_retry.max_attempts:
                    delay_seconds = (
                        float(retry_after_seconds)
                        if retry_after_seconds is not None
                        else self._compute_backoff_delay(resolved_retry, attempt)
                    )
                    await self._sleep(delay_seconds)
                    continue
                raise RateLimitExhausted(
                    normalized_provider,
                    retry_after_seconds=retry_after_seconds,
                    detail="Upstream provider returned HTTP 429.",
                )

            if response.status_code in resolved_retry.retryable_status_codes:
                logger.warning(
                    "Transient provider response for provider=%s status=%s attempt=%s/%s",
                    normalized_provider,
                    response.status_code,
                    attempt,
                    resolved_retry.max_attempts,
                )
                if attempt < resolved_retry.max_attempts:
                    await self._sleep(self._compute_backoff_delay(resolved_retry, attempt))
                    continue

            if 200 <= response.status_code < 300:
                if cache_key is not None and cache_ttl_seconds is not None:
                    await self._cache.set(
                        cache_key,
                        CachedHTTPResponse.from_response(response),
                        ttl_seconds=cache_ttl_seconds,
                    )
                return response

            response_body = response.text[:200].strip()
            raise ProviderError(
                normalized_provider,
                (
                    f"Provider '{normalized_provider}' returned HTTP {response.status_code} "
                    f"for {normalized_method} {normalized_url}. Response snippet: {response_body}"
                ),
                status_code=response.status_code,
            )

        if last_error is None:
            raise ProviderError(
                normalized_provider,
                f"Provider '{normalized_provider}' failed without a captured error.",
            )
        raise last_error

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._http_client.aclose()

    async def _assert_within_rate_limit(
        self,
        provider: str,
        rate_limit_policy: RateLimitPolicy,
    ) -> None:
        """Fail fast when a provider's current counter has reached its limit."""

        rate_limit_key = self._build_rate_limit_key(provider, rate_limit_policy)
        raw_count = await self._cache.get(rate_limit_key)
        if raw_count is None:
            current_count = 0
        elif isinstance(raw_count, int):
            current_count = raw_count
        else:
            raise ValueError(
                "Rate-limit counters must deserialize as integers, "
                f"got {type(raw_count).__name__} for key '{rate_limit_key}'."
            )
        if current_count >= rate_limit_policy.limit:
            raise RateLimitExhausted(
                provider,
                limit=rate_limit_policy.limit,
                window_seconds=rate_limit_policy.window_seconds,
                detail="Local Redis-backed provider budget is exhausted.",
            )

    async def _increment_rate_counter(
        self,
        provider: str,
        rate_limit_policy: RateLimitPolicy,
    ) -> None:
        """Track one outbound API attempt against the provider budget."""

        rate_limit_key = self._build_rate_limit_key(provider, rate_limit_policy)
        await self._cache.increment(
            rate_limit_key,
            ttl_seconds=rate_limit_policy.window_seconds,
        )

    def _build_rate_limit_key(
        self,
        provider: str,
        rate_limit_policy: RateLimitPolicy,
    ) -> str:
        """Build the Redis key for the provider's active rate-limit bucket."""

        current_time = self._clock().astimezone(get_settings().timezone)
        if rate_limit_policy.window_seconds == RATE_LIMIT_TTL_SECONDS:
            return RedisClient.build_rate_limit_key(provider, current_time.date())

        bucket_start_epoch = (
            int(current_time.timestamp()) // rate_limit_policy.window_seconds
        ) * rate_limit_policy.window_seconds
        return (
            f"ratelimit:{provider}:{rate_limit_policy.window_seconds}:"
            f"{bucket_start_epoch}"
        )

    def _build_response_cache_key(
        self,
        *,
        provider: str,
        method: str,
        url: str,
        kwargs: Mapping[str, object],
        headers: Mapping[str, str],
        cache_vary_by_headers: Iterable[str],
    ) -> str:
        """Build a deterministic cache key for one provider request shape."""

        vary_headers: dict[str, JSONValue] = {
            header_name.lower(): headers[header_name]
            for header_name in headers
            if header_name.lower()
            in {name.strip().lower() for name in cache_vary_by_headers if name.strip()}
        }
        cache_payload: JSONValue = {
            "method": method,
            "url": url,
            "params": self._to_json_compatible(kwargs.get("params")),
            "json": self._to_json_compatible(kwargs.get("json")),
            "data": self._to_json_compatible(kwargs.get("data")),
            "content": self._to_json_compatible(kwargs.get("content")),
            "headers": vary_headers,
        }
        serialized_payload = json.dumps(cache_payload, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
        return f"provider:http:{provider}:{digest}"

    @staticmethod
    def _compute_backoff_delay(retry_config: RetryConfig, attempt: int) -> float:
        """Return the exponential backoff delay for a 1-based attempt number."""

        raw_delay = retry_config.initial_delay_seconds * (
            retry_config.backoff_multiplier ** (attempt - 1)
        )
        return min(raw_delay, retry_config.max_delay_seconds)

    @staticmethod
    def _extract_headers(kwargs: Mapping[str, object]) -> dict[str, str]:
        """Normalize request headers into a lower-noise string dictionary."""

        raw_headers = kwargs.get("headers")
        if raw_headers is None:
            return {}

        if isinstance(raw_headers, httpx.Headers):
            return {key.lower(): value for key, value in raw_headers.items()}

        if isinstance(raw_headers, Mapping):
            normalized: dict[str, str] = {}
            for key, value in raw_headers.items():
                normalized_key = RateLimitedClient._normalize_text("header name", str(key)).lower()
                normalized[normalized_key] = RateLimitedClient._normalize_text(
                    f"header '{normalized_key}'",
                    str(value),
                )
            return normalized

        raise TypeError("Request headers must be provided as a mapping.")

    @staticmethod
    def _parse_retry_after_seconds(response: httpx.Response) -> int | None:
        """Parse the provider's `Retry-After` header when present."""

        header_value = response.headers.get("Retry-After")
        if header_value is None:
            return None
        stripped = header_value.strip()
        if not stripped:
            return None
        try:
            parsed_value = int(stripped)
        except ValueError:
            return None
        return parsed_value if parsed_value > 0 else None

    @staticmethod
    def _normalize_text(field_name: str, value: str) -> str:
        """Trim and validate required text fragments used in keys and requests."""

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank.")
        return normalized

    @classmethod
    def _to_json_compatible(cls, value: object) -> JSONValue:
        """Convert supported request arguments into JSON-safe cache payloads."""

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")

        if isinstance(value, httpx.QueryParams):
            value = dict(value.multi_items())

        if isinstance(value, Mapping):
            normalized_mapping: dict[str, JSONValue] = {}
            for key, item in value.items():
                normalized_key = cls._normalize_text("mapping key", str(key))
                normalized_mapping[normalized_key] = cls._to_json_compatible(item)
            return normalized_mapping

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._to_json_compatible(item) for item in value]

        if isinstance(value, BaseModel):
            return cast(JSONValue, value.model_dump(mode="json"))

        raise TypeError(
            f"Unsupported request argument type for cache key generation: {type(value).__name__}."
        )


class DataProvider(ABC):
    """Abstract base class for all PuntLab external data providers.

    Inputs:
        A shared `RateLimitedClient` plus provider-specific metadata supplied by
        concrete subclasses.

    Outputs:
        A concrete provider implementation can call `fetch()` to inherit the
        canonical cache, retry, and rate-limit behavior.
    """

    def __init__(self, client: RateLimitedClient) -> None:
        """Store the shared rate-limited HTTP client for subclasses."""

        self._client = client

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the stable provider identifier used in logging and Redis keys."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the canonical base URL used to resolve relative request paths."""

    @property
    @abstractmethod
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return the provider's configured request budget."""

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return default request headers merged into every provider request."""

        return {}

    @property
    def default_cache_ttl_seconds(self) -> int | None:
        """Return the provider's default response cache TTL.

        Concrete providers can override this to enable cache-by-default for
        endpoints that are safe to reuse within a freshness window.
        """

        return None

    def build_url(self, path: str) -> str:
        """Resolve a relative provider path into an absolute request URL."""

        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path must not be blank.")

        if normalized_path.startswith(("http://", "https://")):
            return normalized_path

        return (
            f"{self.base_url.rstrip('/')}/{normalized_path.lstrip('/')}"
        )

    async def fetch(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        cache_ttl_seconds: int | None = None,
        use_cache: bool = True,
        retry_config: RetryConfig | None = None,
        cache_vary_by_headers: Iterable[str] = (),
        **kwargs: object,
    ) -> httpx.Response:
        """Execute one provider request through the canonical shared wrapper.

        Inputs:
            method: HTTP method for the provider request.
            path: Relative or absolute request URL.
            headers: Optional per-call headers merged over `default_headers`.
            cache_ttl_seconds: Optional cache TTL override for the response.
            use_cache: Whether cached responses may satisfy the call.
            retry_config: Optional per-call retry override.
            cache_vary_by_headers: Header names that should influence the cache
                key when response bodies vary by header.
            **kwargs: Additional `httpx` request arguments such as `params`,
                `json`, `timeout`, or `data`.

        Outputs:
            The provider response as an `httpx.Response`.
        """

        merged_headers = dict(self.default_headers)
        if headers is not None:
            merged_headers.update(
                {
                    self._normalize_header_name(name): self._normalize_header_value(name, value)
                    for name, value in headers.items()
                }
            )

        request_kwargs = dict(kwargs)
        if merged_headers:
            request_kwargs["headers"] = merged_headers

        return await self._client.request(
            self.provider_name,
            method,
            self.build_url(path),
            rate_limit_policy=self.rate_limit_policy,
            cache_ttl_seconds=(
                self.default_cache_ttl_seconds
                if cache_ttl_seconds is None
                else cache_ttl_seconds
            ),
            use_cache=use_cache,
            retry_config=retry_config,
            cache_vary_by_headers=cache_vary_by_headers,
            **request_kwargs,
        )

    @staticmethod
    def _normalize_header_name(name: str) -> str:
        """Normalize and validate one outbound header name."""

        normalized = name.strip()
        if not normalized:
            raise ValueError("header names must not be blank.")
        return normalized

    @staticmethod
    def _normalize_header_value(name: str, value: str) -> str:
        """Normalize and validate one outbound header value."""

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"header '{name}' must not be blank.")
        return normalized


__all__ = [
    "CachedHTTPResponse",
    "DataProvider",
    "ProviderError",
    "RateLimitExhausted",
    "RateLimitPolicy",
    "RateLimitedClient",
    "RetryConfig",
]
