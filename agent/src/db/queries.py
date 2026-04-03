"""Async database query helpers for PuntLab's agent pipeline.

Purpose: provide the canonical SQLAlchemy query layer used by the agent to
persist and retrieve competitions, fixtures, odds, analyses, accumulators,
users, pipeline runs, delivery logs, and payments.
Scope: validated repository-style helpers that operate on an `AsyncSession`
without committing transactions, so callers can compose larger workflows.
Dependencies: SQLAlchemy async ORM/session primitives and the declarative
models defined in `src.db.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Accumulator,
    AccumulatorLeg,
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

type SubscriptionTier = Literal["free", "plus", "elite"]
type SportName = Literal["soccer", "basketball"]
type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _utc_now() -> datetime:
    """Return the current UTC timestamp for application-managed defaults."""

    return datetime.now(UTC)


def _require_non_blank(value: str, field_name: str) -> str:
    """Validate that a string contains non-whitespace content.

    Args:
        value: Raw string input.
        field_name: Field name used in validation errors.

    Returns:
        The stripped string.

    Raises:
        ValueError: If the string is empty after trimming.
    """

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank.")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    """Normalize optional strings by trimming and collapsing empties to `None`."""

    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _require_non_negative_int(value: int, field_name: str) -> int:
    """Validate that an integer is zero or positive."""

    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    """Validate that an integer is strictly positive."""

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _require_positive_decimal(value: Decimal, field_name: str) -> Decimal:
    """Validate that a decimal field is strictly positive."""

    if value <= Decimal("0"):
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _require_non_negative_decimal(value: Decimal, field_name: str) -> Decimal:
    """Validate that a decimal field is zero or positive."""

    if value < Decimal("0"):
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    return value


def _require_probability(value: Decimal, field_name: str) -> Decimal:
    """Validate a normalized score or confidence value in the inclusive 0..1 range."""

    if value < Decimal("0") or value > Decimal("1"):
        raise ValueError(f"{field_name} must be between 0 and 1 inclusive.")
    return value


@dataclass(slots=True)
class CompetitionUpsert:
    """Validated input payload for competition upserts."""

    name: str
    country: str | None
    sport: SportName
    league_code: str
    api_football_id: int | None = None
    football_data_id: int | None = None
    is_active: bool = True
    priority: int = 0

    def __post_init__(self) -> None:
        self.name = _require_non_blank(self.name, "name")
        self.country = _normalize_optional_text(self.country)
        self.league_code = _require_non_blank(self.league_code, "league_code")
        self.priority = _require_non_negative_int(self.priority, "priority")


@dataclass(slots=True)
class FixtureUpsert:
    """Validated input payload for canonical fixture upserts."""

    competition_id: UUID | None
    sportradar_id: str
    home_team: str
    away_team: str
    match_date: date
    kickoff_time: datetime | None = None
    venue: str | None = None
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    api_football_id: int | None = None
    sportybet_url: str | None = None

    def __post_init__(self) -> None:
        self.sportradar_id = _require_non_blank(self.sportradar_id, "sportradar_id")
        self.home_team = _require_non_blank(self.home_team, "home_team")
        self.away_team = _require_non_blank(self.away_team, "away_team")
        self.venue = _normalize_optional_text(self.venue)
        self.status = _require_non_blank(self.status, "status")
        self.sportybet_url = _normalize_optional_text(self.sportybet_url)

        if self.home_score is not None:
            self.home_score = _require_non_negative_int(self.home_score, "home_score")
        if self.away_score is not None:
            self.away_score = _require_non_negative_int(self.away_score, "away_score")


@dataclass(slots=True)
class TeamStatsCreate:
    """Validated statistical snapshot payload for a team."""

    team_id: str
    team_name: str
    competition_id: UUID | None
    season: str | None = None
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    clean_sheets: int = 0
    form: str | None = None
    position: int | None = None
    points: int | None = None
    home_wins: int = 0
    away_wins: int = 0
    avg_goals_scored: Decimal | None = None
    avg_goals_conceded: Decimal | None = None
    fetched_at: datetime | None = None

    def __post_init__(self) -> None:
        self.team_id = _require_non_blank(self.team_id, "team_id")
        self.team_name = _require_non_blank(self.team_name, "team_name")
        self.season = _normalize_optional_text(self.season)
        self.form = _normalize_optional_text(self.form)

        for field_name in (
            "matches_played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "clean_sheets",
            "home_wins",
            "away_wins",
        ):
            value = getattr(self, field_name)
            setattr(self, field_name, _require_non_negative_int(value, field_name))

        if self.position is not None:
            self.position = _require_positive_int(self.position, "position")
        if self.points is not None:
            self.points = _require_non_negative_int(self.points, "points")
        if self.avg_goals_scored is not None:
            self.avg_goals_scored = _require_non_negative_decimal(
                self.avg_goals_scored,
                "avg_goals_scored",
            )
        if self.avg_goals_conceded is not None:
            self.avg_goals_conceded = _require_non_negative_decimal(
                self.avg_goals_conceded,
                "avg_goals_conceded",
            )


@dataclass(slots=True)
class InjuryCreate:
    """Validated injury or suspension payload tied to a fixture."""

    fixture_id: UUID
    team_id: str
    player_name: str
    injury_type: str | None = None
    reason: str | None = None
    is_key_player: bool = False
    fetched_at: datetime | None = None

    def __post_init__(self) -> None:
        self.team_id = _require_non_blank(self.team_id, "team_id")
        self.player_name = _require_non_blank(self.player_name, "player_name")
        self.injury_type = _normalize_optional_text(self.injury_type)
        self.reason = _normalize_optional_text(self.reason)


@dataclass(slots=True)
class OddsUpsert:
    """Validated odds payload for upsert operations."""

    fixture_id: UUID
    provider: str
    market_type: str
    selection: str
    odds_value: Decimal
    market_label: str | None = None
    sportybet_market_id: int | None = None
    is_available: bool = True
    fetched_at: datetime | None = None

    def __post_init__(self) -> None:
        self.provider = _require_non_blank(self.provider, "provider")
        self.market_type = _require_non_blank(self.market_type, "market_type")
        self.selection = _require_non_blank(self.selection, "selection")
        self.market_label = _normalize_optional_text(self.market_label)
        self.odds_value = _require_positive_decimal(self.odds_value, "odds_value")


@dataclass(slots=True)
class MatchAnalysisCreate:
    """Validated scoring output payload for a single fixture."""

    run_id: UUID
    fixture_id: UUID | None
    match_date: date
    composite_score: Decimal
    confidence: Decimal
    form_score: Decimal | None = None
    h2h_score: Decimal | None = None
    injury_impact_score: Decimal | None = None
    odds_value_score: Decimal | None = None
    context_score: Decimal | None = None
    venue_score: Decimal | None = None
    global_rank: int | None = None
    recommended_market: str | None = None
    recommended_selection: str | None = None
    recommended_odds: Decimal | None = None
    news_summary: str | None = None
    context_notes: str | None = None
    llm_assessment: str | None = None

    def __post_init__(self) -> None:
        self.composite_score = _require_probability(self.composite_score, "composite_score")
        self.confidence = _require_probability(self.confidence, "confidence")
        self.recommended_market = _normalize_optional_text(self.recommended_market)
        self.recommended_selection = _normalize_optional_text(self.recommended_selection)
        self.news_summary = _normalize_optional_text(self.news_summary)
        self.context_notes = _normalize_optional_text(self.context_notes)
        self.llm_assessment = _normalize_optional_text(self.llm_assessment)

        for field_name in (
            "form_score",
            "h2h_score",
            "injury_impact_score",
            "odds_value_score",
            "context_score",
            "venue_score",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, _require_probability(value, field_name))

        if self.global_rank is not None:
            self.global_rank = _require_positive_int(self.global_rank, "global_rank")
        if self.recommended_odds is not None:
            self.recommended_odds = _require_positive_decimal(
                self.recommended_odds,
                "recommended_odds",
            )


@dataclass(slots=True)
class AccumulatorCreate:
    """Validated accumulator slip payload."""

    run_id: UUID
    slip_date: date
    slip_number: int
    total_odds: Decimal
    leg_count: int
    confidence: Decimal
    rationale: str | None = None
    status: str = "pending"
    outcome: str | None = None
    is_published: bool = False
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        self.slip_number = _require_positive_int(self.slip_number, "slip_number")
        self.total_odds = _require_positive_decimal(self.total_odds, "total_odds")
        self.leg_count = _require_positive_int(self.leg_count, "leg_count")
        self.confidence = _require_probability(self.confidence, "confidence")
        self.rationale = _normalize_optional_text(self.rationale)
        self.status = _require_non_blank(self.status, "status")
        self.outcome = _normalize_optional_text(self.outcome)


@dataclass(slots=True)
class AccumulatorLegCreate:
    """Validated leg payload for an accumulator slip."""

    leg_number: int
    market_type: str
    selection: str
    odds_value: Decimal
    provider: str
    fixture_id: UUID | None = None
    analysis_id: UUID | None = None
    rationale: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        self.leg_number = _require_positive_int(self.leg_number, "leg_number")
        self.market_type = _require_non_blank(self.market_type, "market_type")
        self.selection = _require_non_blank(self.selection, "selection")
        self.provider = _require_non_blank(self.provider, "provider")
        self.odds_value = _require_positive_decimal(self.odds_value, "odds_value")
        self.rationale = _normalize_optional_text(self.rationale)
        self.outcome = _normalize_optional_text(self.outcome)


@dataclass(slots=True)
class DeliveryLogCreate:
    """Validated delivery log payload for a published accumulator."""

    accumulator_id: UUID | None
    user_id: UUID | None
    channel: str
    status: str
    error_message: str | None = None
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        self.channel = _require_non_blank(self.channel, "channel")
        self.status = _require_non_blank(self.status, "status")
        self.error_message = _normalize_optional_text(self.error_message)


@dataclass(slots=True)
class PipelineRunCreate:
    """Validated creation payload for a pipeline execution log."""

    run_date: date
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    trigger: str = "scheduled"
    fixtures_analyzed: int = 0
    accumulators_generated: int = 0
    accumulators_published: int = 0
    errors: list[JsonValue] = field(default_factory=list)
    stage_timings: dict[str, JsonValue] = field(default_factory=dict)
    llm_tokens_used: dict[str, JsonValue] = field(default_factory=dict)
    llm_cost_usd: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        self.status = _require_non_blank(self.status, "status")
        self.trigger = _require_non_blank(self.trigger, "trigger")
        self.fixtures_analyzed = _require_non_negative_int(
            self.fixtures_analyzed,
            "fixtures_analyzed",
        )
        self.accumulators_generated = _require_non_negative_int(
            self.accumulators_generated,
            "accumulators_generated",
        )
        self.accumulators_published = _require_non_negative_int(
            self.accumulators_published,
            "accumulators_published",
        )
        self.llm_cost_usd = _require_non_negative_decimal(
            self.llm_cost_usd,
            "llm_cost_usd",
        )


@dataclass(slots=True)
class PipelineRunUpdate:
    """Patch payload for pipeline run updates.

    Any field left as `None` is not modified.
    """

    completed_at: datetime | None = None
    status: str | None = None
    trigger: str | None = None
    fixtures_analyzed: int | None = None
    accumulators_generated: int | None = None
    accumulators_published: int | None = None
    errors: list[JsonValue] | None = None
    stage_timings: dict[str, JsonValue] | None = None
    llm_tokens_used: dict[str, JsonValue] | None = None
    llm_cost_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if self.status is not None:
            self.status = _require_non_blank(self.status, "status")
        if self.trigger is not None:
            self.trigger = _require_non_blank(self.trigger, "trigger")
        if self.fixtures_analyzed is not None:
            self.fixtures_analyzed = _require_non_negative_int(
                self.fixtures_analyzed,
                "fixtures_analyzed",
            )
        if self.accumulators_generated is not None:
            self.accumulators_generated = _require_non_negative_int(
                self.accumulators_generated,
                "accumulators_generated",
            )
        if self.accumulators_published is not None:
            self.accumulators_published = _require_non_negative_int(
                self.accumulators_published,
                "accumulators_published",
            )
        if self.llm_cost_usd is not None:
            self.llm_cost_usd = _require_non_negative_decimal(
                self.llm_cost_usd,
                "llm_cost_usd",
            )


@dataclass(slots=True)
class PaymentCreate:
    """Validated payment transaction payload."""

    amount_ngn: Decimal
    plan: str
    duration_days: int
    status: str
    user_id: UUID | None = None
    provider: str = "paystack"
    provider_reference: str | None = None

    def __post_init__(self) -> None:
        self.amount_ngn = _require_positive_decimal(self.amount_ngn, "amount_ngn")
        self.plan = _require_non_blank(self.plan, "plan")
        self.duration_days = _require_positive_int(self.duration_days, "duration_days")
        self.status = _require_non_blank(self.status, "status")
        self.provider = _require_non_blank(self.provider, "provider")
        self.provider_reference = _normalize_optional_text(self.provider_reference)


async def get_competition_by_league_code(
    session: AsyncSession,
    league_code: str,
) -> Competition | None:
    """Fetch a competition by its canonical league code.

    Args:
        session: Open async database session.
        league_code: Canonical competition code such as `PL` or `NBA`.

    Returns:
        The matching competition row, if present.
    """

    statement = select(Competition).where(
        Competition.league_code == _require_non_blank(league_code, "league_code")
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def list_active_competitions(
    session: AsyncSession,
    *,
    sport: SportName | None = None,
) -> list[Competition]:
    """List active competitions, optionally filtered by sport."""

    statement = select(Competition).where(Competition.is_active.is_(True))
    if sport is not None:
        statement = statement.where(Competition.sport == sport)
    statement = statement.order_by(Competition.priority.asc(), Competition.name.asc())
    result = await session.execute(statement)
    return list(result.scalars().all())


async def upsert_competition(
    session: AsyncSession,
    payload: CompetitionUpsert,
) -> Competition:
    """Insert or update a competition using `league_code` as the canonical key."""

    insert_statement = pg_insert(Competition).values(
        name=payload.name,
        country=payload.country,
        sport=payload.sport,
        league_code=payload.league_code,
        api_football_id=payload.api_football_id,
        football_data_id=payload.football_data_id,
        is_active=payload.is_active,
        priority=payload.priority,
    )
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=[Competition.league_code],
        set_={
            "name": payload.name,
            "country": payload.country,
            "sport": payload.sport,
            "api_football_id": payload.api_football_id,
            "football_data_id": payload.football_data_id,
            "is_active": payload.is_active,
            "priority": payload.priority,
        },
    ).returning(Competition)
    result = await session.execute(upsert_statement)
    return result.scalar_one()


async def upsert_fixture(
    session: AsyncSession,
    payload: FixtureUpsert,
) -> Fixture:
    """Insert or update a fixture using `sportradar_id` as the canonical key."""

    insert_statement = pg_insert(Fixture).values(
        competition_id=payload.competition_id,
        sportradar_id=payload.sportradar_id,
        home_team=payload.home_team,
        away_team=payload.away_team,
        match_date=payload.match_date,
        kickoff_time=payload.kickoff_time,
        venue=payload.venue,
        status=payload.status,
        home_score=payload.home_score,
        away_score=payload.away_score,
        api_football_id=payload.api_football_id,
        sportybet_url=payload.sportybet_url,
    )
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=[Fixture.sportradar_id],
        set_={
            "competition_id": payload.competition_id,
            "home_team": payload.home_team,
            "away_team": payload.away_team,
            "match_date": payload.match_date,
            "kickoff_time": payload.kickoff_time,
            "venue": payload.venue,
            "status": payload.status,
            "home_score": payload.home_score,
            "away_score": payload.away_score,
            "api_football_id": payload.api_football_id,
            "sportybet_url": payload.sportybet_url,
            "updated_at": _utc_now(),
        },
    ).returning(Fixture)
    result = await session.execute(upsert_statement)
    return result.scalar_one()


async def upsert_fixtures(
    session: AsyncSession,
    fixtures: list[FixtureUpsert],
) -> list[Fixture]:
    """Bulk-upsert fixtures while preserving the canonical conflict rule."""

    if not fixtures:
        return []

    rows = [
        {
            "competition_id": payload.competition_id,
            "sportradar_id": payload.sportradar_id,
            "home_team": payload.home_team,
            "away_team": payload.away_team,
            "match_date": payload.match_date,
            "kickoff_time": payload.kickoff_time,
            "venue": payload.venue,
            "status": payload.status,
            "home_score": payload.home_score,
            "away_score": payload.away_score,
            "api_football_id": payload.api_football_id,
            "sportybet_url": payload.sportybet_url,
        }
        for payload in fixtures
    ]
    insert_statement = pg_insert(Fixture).values(rows)
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=[Fixture.sportradar_id],
        set_={
            "competition_id": insert_statement.excluded.competition_id,
            "home_team": insert_statement.excluded.home_team,
            "away_team": insert_statement.excluded.away_team,
            "match_date": insert_statement.excluded.match_date,
            "kickoff_time": insert_statement.excluded.kickoff_time,
            "venue": insert_statement.excluded.venue,
            "status": insert_statement.excluded.status,
            "home_score": insert_statement.excluded.home_score,
            "away_score": insert_statement.excluded.away_score,
            "api_football_id": insert_statement.excluded.api_football_id,
            "sportybet_url": insert_statement.excluded.sportybet_url,
            "updated_at": _utc_now(),
        },
    ).returning(Fixture)
    result = await session.execute(upsert_statement)
    return list(result.scalars().all())


async def list_fixtures_for_date(
    session: AsyncSession,
    match_date: date,
) -> list[Fixture]:
    """List fixtures scheduled for a given day in kickoff order."""

    statement = (
        select(Fixture)
        .where(Fixture.match_date == match_date)
        .order_by(Fixture.kickoff_time.asc().nulls_last(), Fixture.home_team.asc())
    )
    result = await session.execute(statement)
    return list(result.scalars().all())


async def insert_team_stats_snapshots(
    session: AsyncSession,
    stats_rows: list[TeamStatsCreate],
) -> list[TeamStats]:
    """Insert one or more team statistic snapshots."""

    if not stats_rows:
        return []

    models = [
        TeamStats(
            team_id=payload.team_id,
            team_name=payload.team_name,
            competition_id=payload.competition_id,
            season=payload.season,
            matches_played=payload.matches_played,
            wins=payload.wins,
            draws=payload.draws,
            losses=payload.losses,
            goals_for=payload.goals_for,
            goals_against=payload.goals_against,
            clean_sheets=payload.clean_sheets,
            form=payload.form,
            position=payload.position,
            points=payload.points,
            home_wins=payload.home_wins,
            away_wins=payload.away_wins,
            avg_goals_scored=payload.avg_goals_scored,
            avg_goals_conceded=payload.avg_goals_conceded,
            fetched_at=payload.fetched_at,
        )
        for payload in stats_rows
    ]
    session.add_all(models)
    await session.flush()
    return models


async def replace_fixture_injuries(
    session: AsyncSession,
    fixture_id: UUID,
    injuries: list[InjuryCreate],
) -> list[Injury]:
    """Replace a fixture's injury snapshot with the supplied rows.

    The pipeline treats injuries as a point-in-time snapshot, so the canonical
    behavior is to delete the prior snapshot first and then insert the fresh one.
    """

    for payload in injuries:
        if payload.fixture_id != fixture_id:
            raise ValueError("All injury payloads must reference the provided fixture_id.")

    await session.execute(delete(Injury).where(Injury.fixture_id == fixture_id))

    if not injuries:
        return []

    models = [
        Injury(
            fixture_id=payload.fixture_id,
            team_id=payload.team_id,
            player_name=payload.player_name,
            injury_type=payload.injury_type,
            reason=payload.reason,
            is_key_player=payload.is_key_player,
            fetched_at=payload.fetched_at,
        )
        for payload in injuries
    ]
    session.add_all(models)
    await session.flush()
    return models


async def upsert_odds_rows(
    session: AsyncSession,
    odds_rows: list[OddsUpsert],
) -> list[Odds]:
    """Upsert normalized odds rows using the canonical uniqueness constraint."""

    if not odds_rows:
        return []

    rows = [
        {
            "fixture_id": payload.fixture_id,
            "provider": payload.provider,
            "market_type": payload.market_type,
            "market_label": payload.market_label,
            "selection": payload.selection,
            "odds_value": payload.odds_value,
            "sportybet_market_id": payload.sportybet_market_id,
            "is_available": payload.is_available,
            "fetched_at": payload.fetched_at or _utc_now(),
        }
        for payload in odds_rows
    ]
    insert_statement = pg_insert(Odds).values(rows)
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=[
            Odds.fixture_id,
            Odds.provider,
            Odds.market_type,
            Odds.selection,
        ],
        set_={
            "market_label": insert_statement.excluded.market_label,
            "odds_value": insert_statement.excluded.odds_value,
            "sportybet_market_id": insert_statement.excluded.sportybet_market_id,
            "is_available": insert_statement.excluded.is_available,
            "fetched_at": insert_statement.excluded.fetched_at,
        },
    ).returning(Odds)
    result = await session.execute(upsert_statement)
    return list(result.scalars().all())


async def insert_match_analyses(
    session: AsyncSession,
    analyses: list[MatchAnalysisCreate],
) -> list[MatchAnalysis]:
    """Insert scoring outputs for a pipeline run."""

    if not analyses:
        return []

    models = [
        MatchAnalysis(
            run_id=payload.run_id,
            fixture_id=payload.fixture_id,
            match_date=payload.match_date,
            form_score=payload.form_score,
            h2h_score=payload.h2h_score,
            injury_impact_score=payload.injury_impact_score,
            odds_value_score=payload.odds_value_score,
            context_score=payload.context_score,
            venue_score=payload.venue_score,
            composite_score=payload.composite_score,
            confidence=payload.confidence,
            global_rank=payload.global_rank,
            recommended_market=payload.recommended_market,
            recommended_selection=payload.recommended_selection,
            recommended_odds=payload.recommended_odds,
            news_summary=payload.news_summary,
            context_notes=payload.context_notes,
            llm_assessment=payload.llm_assessment,
        )
        for payload in analyses
    ]
    session.add_all(models)
    await session.flush()
    return models


async def create_accumulator_with_legs(
    session: AsyncSession,
    accumulator_payload: AccumulatorCreate,
    leg_payloads: list[AccumulatorLegCreate],
) -> Accumulator:
    """Create an accumulator and its ordered legs in one flush.

    Args:
        session: Open async database session.
        accumulator_payload: Slip-level metadata.
        leg_payloads: Ordered leg definitions for the slip.

    Returns:
        The accumulator ORM instance with attached leg ORM instances.

    Raises:
        ValueError: If leg numbering is inconsistent with the declared slip shape.
    """

    if not leg_payloads:
        raise ValueError("Accumulator creation requires at least one leg.")

    if accumulator_payload.leg_count != len(leg_payloads):
        raise ValueError("Accumulator leg_count must match the number of supplied legs.")

    sorted_leg_payloads = sorted(leg_payloads, key=lambda payload: payload.leg_number)
    expected_leg_numbers = list(range(1, len(sorted_leg_payloads) + 1))
    actual_leg_numbers = [payload.leg_number for payload in sorted_leg_payloads]
    if actual_leg_numbers != expected_leg_numbers:
        raise ValueError("Accumulator legs must be consecutively numbered starting at 1.")

    accumulator = Accumulator(
        run_id=accumulator_payload.run_id,
        slip_date=accumulator_payload.slip_date,
        slip_number=accumulator_payload.slip_number,
        total_odds=accumulator_payload.total_odds,
        leg_count=accumulator_payload.leg_count,
        confidence=accumulator_payload.confidence,
        rationale=accumulator_payload.rationale,
        status=accumulator_payload.status,
        outcome=accumulator_payload.outcome,
        is_published=accumulator_payload.is_published,
        published_at=accumulator_payload.published_at,
    )

    # Assigning legs through the relationship keeps the object graph coherent
    # for callers before the transaction is committed.
    accumulator.legs = [
        AccumulatorLeg(
            fixture_id=payload.fixture_id,
            analysis_id=payload.analysis_id,
            leg_number=payload.leg_number,
            market_type=payload.market_type,
            selection=payload.selection,
            odds_value=payload.odds_value,
            provider=payload.provider,
            rationale=payload.rationale,
            outcome=payload.outcome,
        )
        for payload in sorted_leg_payloads
    ]
    session.add(accumulator)
    await session.flush()
    return accumulator


async def get_users_by_tier(
    session: AsyncSession,
    subscription_tier: SubscriptionTier,
    *,
    active_only: bool = True,
    as_of: datetime | None = None,
) -> list[User]:
    """Fetch users for a subscription tier, with optional active-subscription filtering."""

    statement = select(User).where(User.subscription_tier == subscription_tier)

    if active_only:
        effective_timestamp = as_of or _utc_now()
        statement = statement.where(User.subscription_status == "active").where(
            or_(
                User.subscription_expires_at.is_(None),
                User.subscription_expires_at >= effective_timestamp,
            )
        )

    statement = statement.order_by(User.created_at.asc(), User.id.asc())
    result = await session.execute(statement)
    return list(result.scalars().all())


async def create_delivery_logs(
    session: AsyncSession,
    delivery_logs: list[DeliveryLogCreate],
) -> list[DeliveryLog]:
    """Insert one or more delivery log rows."""

    if not delivery_logs:
        return []

    models = [
        DeliveryLog(
            accumulator_id=payload.accumulator_id,
            user_id=payload.user_id,
            channel=payload.channel,
            status=payload.status,
            error_message=payload.error_message,
            delivered_at=payload.delivered_at or _utc_now(),
        )
        for payload in delivery_logs
    ]
    session.add_all(models)
    await session.flush()
    return models


async def create_pipeline_run(
    session: AsyncSession,
    payload: PipelineRunCreate,
) -> PipelineRun:
    """Insert a new pipeline run record."""

    pipeline_run = PipelineRun(
        run_date=payload.run_date,
        started_at=payload.started_at,
        completed_at=payload.completed_at,
        status=payload.status,
        trigger=payload.trigger,
        fixtures_analyzed=payload.fixtures_analyzed,
        accumulators_generated=payload.accumulators_generated,
        accumulators_published=payload.accumulators_published,
        errors=payload.errors,
        stage_timings=payload.stage_timings,
        llm_tokens_used=payload.llm_tokens_used,
        llm_cost_usd=payload.llm_cost_usd,
    )
    session.add(pipeline_run)
    await session.flush()
    return pipeline_run


async def get_pipeline_run(session: AsyncSession, run_id: UUID) -> PipelineRun | None:
    """Fetch a pipeline run by ID."""

    result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
    return result.scalar_one_or_none()


async def update_pipeline_run(
    session: AsyncSession,
    run_id: UUID,
    patch: PipelineRunUpdate,
) -> PipelineRun | None:
    """Update mutable pipeline run fields and return the updated row.

    Raises:
        ValueError: If the patch does not include any fields to update.
    """

    updates: dict[str, object] = {}
    for field_name in (
        "completed_at",
        "status",
        "trigger",
        "fixtures_analyzed",
        "accumulators_generated",
        "accumulators_published",
        "errors",
        "stage_timings",
        "llm_tokens_used",
        "llm_cost_usd",
    ):
        value = getattr(patch, field_name)
        if value is not None:
            updates[field_name] = value

    if not updates:
        raise ValueError("PipelineRunUpdate must include at least one field to update.")

    statement = (
        update(PipelineRun).where(PipelineRun.id == run_id).values(**updates).returning(PipelineRun)
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def create_payment(session: AsyncSession, payload: PaymentCreate) -> Payment:
    """Insert a payment transaction row."""

    payment = Payment(
        user_id=payload.user_id,
        provider=payload.provider,
        provider_reference=payload.provider_reference,
        amount_ngn=payload.amount_ngn,
        plan=payload.plan,
        duration_days=payload.duration_days,
        status=payload.status,
    )
    session.add(payment)
    await session.flush()
    return payment


__all__ = [
    "AccumulatorCreate",
    "AccumulatorLegCreate",
    "CompetitionUpsert",
    "DeliveryLogCreate",
    "FixtureUpsert",
    "InjuryCreate",
    "JsonValue",
    "MatchAnalysisCreate",
    "OddsUpsert",
    "PaymentCreate",
    "PipelineRunCreate",
    "PipelineRunUpdate",
    "SportName",
    "SubscriptionTier",
    "TeamStatsCreate",
    "create_accumulator_with_legs",
    "create_delivery_logs",
    "create_payment",
    "create_pipeline_run",
    "get_competition_by_league_code",
    "get_pipeline_run",
    "get_users_by_tier",
    "insert_match_analyses",
    "insert_team_stats_snapshots",
    "list_active_competitions",
    "list_fixtures_for_date",
    "replace_fixture_injuries",
    "update_pipeline_run",
    "upsert_competition",
    "upsert_fixture",
    "upsert_fixtures",
    "upsert_odds_rows",
]
