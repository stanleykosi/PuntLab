"""Redis cache package for PuntLab.

Purpose: groups typed cache access helpers and rate-limit tracking.
Scope: provider caching, LLM caching, and transient coordination.
Dependencies: configured through Redis connection settings.
"""

from src.cache.client import (
    API_FOOTBALL_FIXTURES_TTL_SECONDS,
    API_ODDS_TTL_SECONDS,
    API_STATS_TTL_SECONDS,
    LLM_CONTEXT_TTL_SECONDS,
    PIPELINE_STATE_TTL_SECONDS,
    RATE_LIMIT_TTL_SECONDS,
    SPORTYBET_MARKETS_TTL_SECONDS,
    CacheTTLConfig,
    RedisClient,
)

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
