"""Shared Pydantic schema namespace for PuntLab.

Purpose: central location for normalized data contracts shared across stages.
Scope: fixtures, odds, analyses, accumulators, users, and delivery results.
Dependencies: imported by providers, scoring, the pipeline state, and the API layer.
"""

from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import MarketType, NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, PlayerStats, TeamStats

__all__ = [
    "FixtureStatus",
    "InjuryData",
    "InjuryType",
    "MarketType",
    "NewsArticle",
    "NormalizedFixture",
    "NormalizedOdds",
    "PlayerStats",
    "TeamStats",
]
