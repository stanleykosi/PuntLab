"""SportyBet scraping and market-resolution namespace.

Purpose: groups API interception and browser-based scraping utilities.
Scope: bookmaker market lookup and normalization support.
Dependencies: used by the market-resolution stage.
"""

from src.scrapers.resolver import MarketResolver
from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SPORTYBET_API_PATH_TEMPLATES,
    SPORTYBET_BASE_URL,
    SPORTYBET_COUNTRY_CODE,
    SPORTYBET_MARKET_IDS,
    SPORTYBET_RATE_LIMIT_POLICY,
    SportyBetAPIClient,
    SportyBetEventCatalog,
    SportyBetMarketCacheEntry,
    SportyBetMarketGroup,
)
from src.scrapers.sportybet_browser import (
    DEFAULT_MARKET_WAIT_SELECTORS,
    DOM_MARKET_EXTRACTOR_SCRIPT,
    SportyBetBrowserScraper,
)
__all__ = [
    "DEFAULT_USER_AGENTS",
    "DEFAULT_MARKET_WAIT_SELECTORS",
    "DOM_MARKET_EXTRACTOR_SCRIPT",
    "MarketResolver",
    "SPORTYBET_API_PATH_TEMPLATES",
    "SPORTYBET_BASE_URL",
    "SPORTYBET_COUNTRY_CODE",
    "SPORTYBET_MARKET_IDS",
    "SPORTYBET_RATE_LIMIT_POLICY",
    "SportyBetAPIClient",
    "SportyBetEventCatalog",
    "SportyBetBrowserScraper",
    "SportyBetMarketCacheEntry",
    "SportyBetMarketGroup",
]
