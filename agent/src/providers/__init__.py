"""External data provider integrations for PuntLab.

Purpose: expose the canonical shared provider infrastructure and the concrete
provider implementations that the ingestion stage still composes directly.
Scope: provider base classes plus the current API-Football, BALLDONTLIE, RSS
feeds, Tavily, and the orchestrator used by ingestion.
Dependencies: concrete providers reuse `src.providers.base` and are
orchestrated by the ingestion stage.
"""

from src.providers.api_football import APIFootballProvider
from src.providers.balldontlie import BallDontLieProvider
from src.providers.base import (
    CachedHTTPResponse,
    DataProvider,
    ProviderError,
    RateLimitedClient,
    RateLimitExhausted,
    RateLimitPolicy,
    RetryConfig,
)
from src.providers.orchestrator import (
    CompetitionProviderRoute,
    InjuryFetchResult,
    OddsFetchResult,
    ProviderOrchestrator,
    StatsFetchResult,
)
from src.providers.rss_feeds import DEFAULT_RSS_FEEDS, RSSFeedDefinition, RSSFeedProvider
from src.providers.tavily_search import TavilySearchProvider

__all__ = [
    "APIFootballProvider",
    "BallDontLieProvider",
    "CachedHTTPResponse",
    "CompetitionProviderRoute",
    "DataProvider",
    "DEFAULT_RSS_FEEDS",
    "InjuryFetchResult",
    "OddsFetchResult",
    "ProviderError",
    "ProviderOrchestrator",
    "RateLimitExhausted",
    "RateLimitPolicy",
    "RateLimitedClient",
    "RSSFeedDefinition",
    "RSSFeedProvider",
    "RetryConfig",
    "StatsFetchResult",
    "TavilySearchProvider",
]
