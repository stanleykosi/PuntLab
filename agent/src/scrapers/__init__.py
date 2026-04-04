"""SportyBet scraping and market-resolution namespace.

Purpose: groups API interception and browser-based scraping utilities.
Scope: bookmaker market lookup and normalization support.
Dependencies: used by the market-resolution stage.
"""

from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SPORTYBET_API_PATH_TEMPLATES,
    SPORTYBET_BASE_URL,
    SPORTYBET_COUNTRY_CODE,
    SPORTYBET_MARKET_IDS,
    SPORTYBET_RATE_LIMIT_POLICY,
    SportyBetAPIClient,
    SportyBetMarketCacheEntry,
)

__all__ = [
    "DEFAULT_USER_AGENTS",
    "SPORTYBET_API_PATH_TEMPLATES",
    "SPORTYBET_BASE_URL",
    "SPORTYBET_COUNTRY_CODE",
    "SPORTYBET_MARKET_IDS",
    "SPORTYBET_RATE_LIMIT_POLICY",
    "SportyBetAPIClient",
    "SportyBetMarketCacheEntry",
]
