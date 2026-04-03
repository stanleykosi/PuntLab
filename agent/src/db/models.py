"""SQLAlchemy ORM models for PuntLab's canonical PostgreSQL schema.

Purpose: map the agent's persistence layer to the Supabase/PostgreSQL schema
defined in `supabase/migrations/001_initial_schema.sql`.
Scope: declarative table models, relationships, indexes, and constraints for
the Python agent runtime.
Dependencies: SQLAlchemy 2.x declarative ORM and PostgreSQL-specific column
types such as `JSONB` and UUID.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BIGINT,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, foreign, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all PuntLab ORM models."""


class Competition(Base):
    """Competition or league metadata for supported soccer and basketball slates."""

    __tablename__ = "competitions"
    __table_args__ = (
        CheckConstraint("sport IN ('soccer', 'basketball')"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str | None] = mapped_column(String(100))
    sport: Mapped[str] = mapped_column(String(50), nullable=False)
    league_code: Mapped[str | None] = mapped_column(String(50), unique=True)
    api_football_id: Mapped[int | None] = mapped_column(Integer)
    football_data_id: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    fixtures: Mapped[list[Fixture]] = relationship(back_populates="competition")
    team_stats: Mapped[list[TeamStats]] = relationship(back_populates="competition")


class Fixture(Base):
    """Scheduled or completed fixture analyzed by the recommendation pipeline."""

    __tablename__ = "fixtures"
    __table_args__ = (
        Index("idx_fixtures_date", "match_date"),
        Index("idx_fixtures_sportradar", "sportradar_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    competition_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("competitions.id"),
    )
    sportradar_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    home_team: Mapped[str] = mapped_column(String(255), nullable=False)
    away_team: Mapped[str] = mapped_column(String(255), nullable=False)
    home_team_id: Mapped[str | None] = mapped_column(String(100))
    away_team_id: Mapped[str | None] = mapped_column(String(100))
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    kickoff_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    venue: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'scheduled'"),
    )
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    api_football_id: Mapped[int | None] = mapped_column(Integer)
    sportybet_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    competition: Mapped[Competition | None] = relationship(back_populates="fixtures")
    odds: Mapped[list[Odds]] = relationship(back_populates="fixture")
    injuries: Mapped[list[Injury]] = relationship(back_populates="fixture")
    analyses: Mapped[list[MatchAnalysis]] = relationship(back_populates="fixture")
    accumulator_legs: Mapped[list[AccumulatorLeg]] = relationship(back_populates="fixture")


class Odds(Base):
    """Normalized odds row for a fixture, provider, market, and selection."""

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("fixture_id", "provider", "market_type", "selection"),
        Index("idx_odds_fixture", "fixture_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    fixture_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("fixtures.id"),
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    market_type: Mapped[str] = mapped_column(String(100), nullable=False)
    market_label: Mapped[str | None] = mapped_column(String(255))
    selection: Mapped[str] = mapped_column(String(255), nullable=False)
    odds_value: Mapped[Decimal] = mapped_column(Numeric(8, 3), nullable=False)
    sportybet_market_id: Mapped[int | None] = mapped_column(Integer)
    is_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    fixture: Mapped[Fixture | None] = relationship(back_populates="odds")


class TeamStats(Base):
    """Per-team statistical snapshot used during deterministic match scoring."""

    __tablename__ = "team_stats"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    team_id: Mapped[str] = mapped_column(String(100), nullable=False)
    team_name: Mapped[str] = mapped_column(String(255), nullable=False)
    competition_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("competitions.id"),
    )
    season: Mapped[str | None] = mapped_column(String(20))
    matches_played: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    wins: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    draws: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    losses: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    goals_for: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    goals_against: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    clean_sheets: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    form: Mapped[str | None] = mapped_column(String(50))
    position: Mapped[int | None] = mapped_column(Integer)
    points: Mapped[int | None] = mapped_column(Integer)
    home_wins: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    away_wins: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    avg_goals_scored: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    avg_goals_conceded: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    competition: Mapped[Competition | None] = relationship(back_populates="team_stats")


class Injury(Base):
    """Injury or suspension row scoped to a fixture and team."""

    __tablename__ = "injuries"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    fixture_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("fixtures.id"),
    )
    team_id: Mapped[str] = mapped_column(String(100), nullable=False)
    player_name: Mapped[str] = mapped_column(String(255), nullable=False)
    injury_type: Mapped[str | None] = mapped_column(String(100))
    reason: Mapped[str | None] = mapped_column(String(500))
    is_key_player: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    fixture: Mapped[Fixture | None] = relationship(back_populates="injuries")


class MatchAnalysis(Base):
    """Scoring output and recommended market for a single analyzed fixture."""

    __tablename__ = "match_analyses"
    __table_args__ = (
        Index("idx_analyses_run", "run_id"),
        Index("idx_analyses_date", "match_date"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    fixture_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("fixtures.id"),
    )
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    form_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    h2h_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    injury_impact_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    odds_value_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    context_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    venue_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 3))
    composite_score: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    global_rank: Mapped[int | None] = mapped_column(Integer)
    recommended_market: Mapped[str | None] = mapped_column(String(100))
    recommended_selection: Mapped[str | None] = mapped_column(String(255))
    recommended_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 3))
    news_summary: Mapped[str | None] = mapped_column(Text)
    context_notes: Mapped[str | None] = mapped_column(Text)
    llm_assessment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    fixture: Mapped[Fixture | None] = relationship(back_populates="analyses")
    pipeline_run: Mapped[PipelineRun | None] = relationship(
        back_populates="match_analyses",
        primaryjoin=lambda: foreign(MatchAnalysis.run_id) == PipelineRun.id,
    )
    accumulator_legs: Mapped[list[AccumulatorLeg]] = relationship(back_populates="analysis")


class Accumulator(Base):
    """Accumulator slip assembled from top-ranked market recommendations."""

    __tablename__ = "accumulators"
    __table_args__ = (
        Index("idx_accumulators_date", "slip_date"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    slip_date: Mapped[date] = mapped_column(Date, nullable=False)
    slip_number: Mapped[int] = mapped_column(Integer, nullable=False)
    total_odds: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    leg_count: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'pending'"),
    )
    outcome: Mapped[str | None] = mapped_column(String(50))
    is_published: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    pipeline_run: Mapped[PipelineRun | None] = relationship(
        back_populates="accumulators",
        primaryjoin=lambda: foreign(Accumulator.run_id) == PipelineRun.id,
    )
    legs: Mapped[list[AccumulatorLeg]] = relationship(
        back_populates="accumulator",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    delivery_logs: Mapped[list[DeliveryLog]] = relationship(back_populates="accumulator")


class AccumulatorLeg(Base):
    """A single market selection that belongs to an accumulator slip."""

    __tablename__ = "accumulator_legs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    accumulator_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("accumulators.id", ondelete="CASCADE"),
    )
    fixture_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("fixtures.id"),
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("match_analyses.id"),
    )
    leg_number: Mapped[int] = mapped_column(Integer, nullable=False)
    market_type: Mapped[str] = mapped_column(String(100), nullable=False)
    selection: Mapped[str] = mapped_column(String(255), nullable=False)
    odds_value: Mapped[Decimal] = mapped_column(Numeric(8, 3), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    accumulator: Mapped[Accumulator | None] = relationship(back_populates="legs")
    fixture: Mapped[Fixture | None] = relationship(back_populates="accumulator_legs")
    analysis: Mapped[MatchAnalysis | None] = relationship(back_populates="accumulator_legs")


class User(Base):
    """Telegram or web user record with subscription and admin metadata."""

    __tablename__ = "users"
    __table_args__ = (
        Index("idx_users_telegram", "telegram_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    telegram_id: Mapped[int | None] = mapped_column(BIGINT, unique=True)
    telegram_username: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    subscription_tier: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'free'"),
    )
    subscription_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'active'"),
    )
    subscription_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    paystack_customer_id: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    delivery_logs: Mapped[list[DeliveryLog]] = relationship(back_populates="user")
    payments: Mapped[list[Payment]] = relationship(back_populates="user")


class PipelineRun(Base):
    """Execution log for a full daily or manual pipeline run."""

    __tablename__ = "pipeline_runs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_date: Mapped[date] = mapped_column(Date, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'running'"),
    )
    trigger: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'scheduled'"),
    )
    fixtures_analyzed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    accumulators_generated: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    accumulators_published: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    errors: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    stage_timings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    llm_tokens_used: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    llm_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        nullable=False,
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # `run_id` is intentionally not enforced as a foreign key in the canonical
    # schema, but we still model the object graph explicitly for agent code.
    match_analyses: Mapped[list[MatchAnalysis]] = relationship(
        back_populates="pipeline_run",
        primaryjoin=lambda: PipelineRun.id == foreign(MatchAnalysis.run_id),
    )
    accumulators: Mapped[list[Accumulator]] = relationship(
        back_populates="pipeline_run",
        primaryjoin=lambda: PipelineRun.id == foreign(Accumulator.run_id),
    )


class DeliveryLog(Base):
    """Per-user delivery status for a published accumulator recommendation."""

    __tablename__ = "delivery_log"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    accumulator_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("accumulators.id"),
    )
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
    )
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    accumulator: Mapped[Accumulator | None] = relationship(back_populates="delivery_logs")
    user: Mapped[User | None] = relationship(back_populates="delivery_logs")


class Payment(Base):
    """Subscription payment transaction recorded against a PuntLab user."""

    __tablename__ = "payments"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
    )
    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'paystack'"),
    )
    provider_reference: Mapped[str | None] = mapped_column(String(255))
    amount_ngn: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    plan: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_days: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped[User | None] = relationship(back_populates="payments")
