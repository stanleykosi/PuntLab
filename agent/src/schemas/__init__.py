"""Shared Pydantic schema namespace for PuntLab.

Purpose: central location for normalized data contracts shared across stages.
Scope: fixtures, odds, analyses, accumulators, users, and delivery results.
Dependencies: imported by providers, scoring, the pipeline state, and the API layer.
"""

from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorOutcome,
    AccumulatorSlip,
    AccumulatorStatus,
    AccumulatorStrategy,
    ExplainedAccumulator,
    LegOutcome,
    ResolutionSource,
    ResolvedMarket,
)
from src.schemas.analysis import (
    MatchContext,
    MatchScore,
    QualitativeScore,
    RankedMatch,
    ScoreFactorBreakdown,
    ScoringWeights,
)
from src.schemas.fixtures import FixtureStatus, NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import MarketType, NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, PlayerStats, TeamStats
from src.schemas.users import (
    DeliveryChannel,
    DeliveryResult,
    DeliveryStatus,
    SubscriptionStatus,
    SubscriptionTier,
    UserProfile,
)

__all__ = [
    "AccumulatorLeg",
    "AccumulatorOutcome",
    "AccumulatorSlip",
    "AccumulatorStatus",
    "AccumulatorStrategy",
    "DeliveryChannel",
    "DeliveryResult",
    "DeliveryStatus",
    "ExplainedAccumulator",
    "FixtureStatus",
    "InjuryData",
    "InjuryType",
    "LegOutcome",
    "MarketType",
    "MatchContext",
    "MatchScore",
    "NewsArticle",
    "NormalizedFixture",
    "NormalizedOdds",
    "PlayerStats",
    "QualitativeScore",
    "RankedMatch",
    "ResolutionSource",
    "ResolvedMarket",
    "ScoreFactorBreakdown",
    "ScoringWeights",
    "SubscriptionStatus",
    "SubscriptionTier",
    "TeamStats",
    "UserProfile",
]
