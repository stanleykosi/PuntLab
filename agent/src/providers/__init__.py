"""External data provider integrations for PuntLab.

Purpose: expose the canonical shared provider infrastructure and the concrete
provider implementations that the ingestion stage composes.
Scope: provider base classes plus the current API-Football, Football-Data.org,
and The Odds API integrations.
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
from src.providers.football_data import FootballDataProvider
from src.providers.the_odds_api import TheOddsAPIProvider

__all__ = [
    "APIFootballProvider",
    "CachedHTTPResponse",
    "DataProvider",
    "FootballDataProvider",
    "ProviderError",
    "RateLimitExhausted",
    "RateLimitPolicy",
    "RateLimitedClient",
    "RetryConfig",
    "TheOddsAPIProvider",
]
