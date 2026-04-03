"""Database access package for PuntLab.

Purpose: expose the canonical SQLAlchemy async connection helpers and ORM
models used throughout the agent runtime.
Scope: persistence primitives for fixtures, analyses, accumulators, users, and
delivery logs.
Dependencies: configured through PostgreSQL connection settings and SQLAlchemy.
"""

from src.db.connection import (
    create_engine,
    create_session_factory,
    dispose_engine,
    get_engine,
    get_session,
    get_session_factory,
    normalize_database_url,
)
from src.db.models import (
    Accumulator,
    AccumulatorLeg,
    Base,
    Competition,
    DeliveryLog,
    Fixture,
    Injury,
    MatchAnalysis,
    Odds,
    Payment,
    PipelineRun,
    TeamStats,
    User,
)

__all__ = [
    "Accumulator",
    "AccumulatorLeg",
    "Base",
    "Competition",
    "DeliveryLog",
    "Fixture",
    "Injury",
    "MatchAnalysis",
    "Odds",
    "Payment",
    "PipelineRun",
    "TeamStats",
    "User",
    "create_engine",
    "create_session_factory",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
    "normalize_database_url",
]
