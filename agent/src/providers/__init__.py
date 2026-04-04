"""External data provider integrations for PuntLab.

Purpose: expose the canonical shared provider infrastructure and reserve the
namespace for concrete sports, odds, and news integrations.
Scope: provider base classes plus future API-Football, Football-Data.org,
The Odds API, Tavily, RSS, and NBA provider implementations.
Dependencies: concrete providers reuse `src.providers.base` and are
orchestrated by the ingestion stage.
"""

from src.providers.api_football import APIFootballProvider
from src.providers.base import (
    CachedHTTPResponse,
    DataProvider,
    ProviderError,
    RateLimitedClient,
    RateLimitExhausted,
    RateLimitPolicy,
    RetryConfig,
)

__all__ = [
    "APIFootballProvider",
    "CachedHTTPResponse",
    "DataProvider",
    "ProviderError",
    "RateLimitExhausted",
    "RateLimitPolicy",
    "RateLimitedClient",
    "RetryConfig",
]
