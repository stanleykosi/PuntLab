"""Tests for PuntLab's async database query helpers.

Purpose: verify that the repository layer builds the expected SQLAlchemy
statements, validates critical inputs, and constructs ORM object graphs
correctly before live database integration tests are added later.
Scope: pure unit tests with lightweight session/result doubles.
Dependencies: pytest, SQLAlchemy PostgreSQL dialect compilation, and the
query helpers in `src.db.queries`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects.postgresql import dialect as postgresql_dialect
from src.db.models import Fixture, Odds, PipelineRun
from src.db.queries import (
    AccumulatorCreate,
    AccumulatorLegCreate,
    FixtureUpsert,
    InjuryCreate,
    OddsUpsert,
    PipelineRunCreate,
    PipelineRunUpdate,
    create_accumulator_with_legs,
    create_pipeline_run,
    get_users_by_tier,
    replace_fixture_injuries,
    update_pipeline_run,
    upsert_fixture,
    upsert_odds_rows,
)


class FakeExecutionResult:
    """Minimal SQLAlchemy result double for repository unit tests."""

    def __init__(self, values: list[object]) -> None:
        self._values = values

    def scalars(self) -> FakeExecutionResult:
        """Return a scalar-oriented view for `all()` access."""

        return self

    def all(self) -> list[object]:
        """Return all queued scalar values."""

        return list(self._values)

    def scalar_one(self) -> object:
        """Return the single queued value, mirroring SQLAlchemy's contract."""

        if len(self._values) != 1:
            raise AssertionError("Expected exactly one scalar value.")
        return self._values[0]

    def scalar_one_or_none(self) -> object | None:
        """Return the single queued value or `None`."""

        if not self._values:
            return None
        if len(self._values) != 1:
            raise AssertionError("Expected at most one scalar value.")
        return self._values[0]


class RecordingSession:
    """Small async session double that records writes and statements."""

    def __init__(self) -> None:
        self.statements: list[object] = []
        self.added: list[object] = []
        self.added_batches: list[list[object]] = []
        self.flush_count = 0
        self._queued_results: list[FakeExecutionResult] = []

    def queue_execute_result(self, *values: object) -> None:
        """Queue the next `execute()` return value."""

        self._queued_results.append(FakeExecutionResult(list(values)))

    async def execute(self, statement: object) -> FakeExecutionResult:
        """Record a SQLAlchemy statement and return a queued fake result."""

        self.statements.append(statement)
        if self._queued_results:
            return self._queued_results.pop(0)
        return FakeExecutionResult([])

    def add(self, model: object) -> None:
        """Record a single ORM instance passed to `add()`."""

        self.added.append(model)

    def add_all(self, models: list[object]) -> None:
        """Record a batch of ORM instances passed to `add_all()`."""

        self.added_batches.append(list(models))

    async def flush(self) -> None:
        """Record that a flush occurred."""

        self.flush_count += 1


def compile_sql(statement: object) -> str:
    """Compile a SQLAlchemy statement to PostgreSQL SQL for assertions."""

    return str(statement.compile(dialect=postgresql_dialect()))


@pytest.mark.asyncio
async def test_upsert_fixture_uses_sportradar_conflict_key() -> None:
    """Fixture upserts should target the canonical `sportradar_id` uniqueness."""

    session = RecordingSession()
    returned_fixture = Fixture(
        sportradar_id="sr:match:1001",
        home_team="Arsenal",
        away_team="Chelsea",
        match_date=date(2026, 4, 3),
    )
    session.queue_execute_result(returned_fixture)

    result = await upsert_fixture(
        session,
        FixtureUpsert(
            competition_id=uuid4(),
            sportradar_id="sr:match:1001",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=date(2026, 4, 3),
            status="scheduled",
        ),
    )

    sql = compile_sql(session.statements[0])

    assert result is returned_fixture
    assert "ON CONFLICT (sportradar_id) DO UPDATE" in sql
    assert "updated_at" in sql


@pytest.mark.asyncio
async def test_upsert_odds_rows_uses_composite_uniqueness_constraint() -> None:
    """Odds upserts should align with the migration's composite unique key."""

    fixture_id = uuid4()
    session = RecordingSession()
    returned_odds = Odds(
        fixture_id=fixture_id,
        provider="sportybet",
        market_type="1x2",
        selection="home",
        odds_value=Decimal("1.850"),
    )
    session.queue_execute_result(returned_odds)

    result = await upsert_odds_rows(
        session,
        [
            OddsUpsert(
                fixture_id=fixture_id,
                provider="sportybet",
                market_type="1x2",
                selection="home",
                odds_value=Decimal("1.850"),
            )
        ],
    )

    sql = compile_sql(session.statements[0])

    assert result == [returned_odds]
    assert "ON CONFLICT (fixture_id, provider, market_type, selection) DO UPDATE" in sql
    assert "sportybet_market_id = excluded.sportybet_market_id" in sql


@pytest.mark.asyncio
async def test_replace_fixture_injuries_deletes_previous_snapshot_before_insert() -> None:
    """Replacing injuries should clear the prior snapshot first."""

    fixture_id = uuid4()
    session = RecordingSession()

    injuries = await replace_fixture_injuries(
        session,
        fixture_id,
        [
            InjuryCreate(
                fixture_id=fixture_id,
                team_id="arsenal",
                player_name="Bukayo Saka",
                injury_type="muscle",
            )
        ],
    )

    first_statement_sql = compile_sql(session.statements[0])

    assert "DELETE FROM injuries" in first_statement_sql
    assert session.flush_count == 1
    assert len(session.added_batches) == 1
    assert injuries[0].player_name == "Bukayo Saka"


@pytest.mark.asyncio
async def test_create_accumulator_with_legs_sorts_and_attaches_legs() -> None:
    """Accumulator creation should preserve a coherent, ordered leg graph."""

    session = RecordingSession()

    accumulator = await create_accumulator_with_legs(
        session,
        AccumulatorCreate(
            run_id=uuid4(),
            slip_date=date(2026, 4, 3),
            slip_number=1,
            total_odds=Decimal("4.200"),
            leg_count=2,
            confidence=Decimal("0.840"),
        ),
        [
            AccumulatorLegCreate(
                leg_number=2,
                market_type="totals",
                selection="Over 2.5",
                odds_value=Decimal("1.900"),
                provider="sportybet",
            ),
            AccumulatorLegCreate(
                leg_number=1,
                market_type="1x2",
                selection="Home Win",
                odds_value=Decimal("2.210"),
                provider="sportybet",
            ),
        ],
    )

    assert session.flush_count == 1
    assert session.added == [accumulator]
    assert [leg.leg_number for leg in accumulator.legs] == [1, 2]
    assert [leg.selection for leg in accumulator.legs] == ["Home Win", "Over 2.5"]


@pytest.mark.asyncio
async def test_create_accumulator_with_legs_rejects_leg_count_mismatch() -> None:
    """Accumulator validation should fail fast on inconsistent slip metadata."""

    session = RecordingSession()

    with pytest.raises(ValueError, match="leg_count must match"):
        await create_accumulator_with_legs(
            session,
            AccumulatorCreate(
                run_id=uuid4(),
                slip_date=date(2026, 4, 3),
                slip_number=1,
                total_odds=Decimal("2.300"),
                leg_count=2,
                confidence=Decimal("0.700"),
            ),
            [
                AccumulatorLegCreate(
                    leg_number=1,
                    market_type="1x2",
                    selection="Away Win",
                    odds_value=Decimal("2.300"),
                    provider="sportybet",
                )
            ],
        )


@pytest.mark.asyncio
async def test_get_users_by_tier_applies_active_subscription_filters() -> None:
    """Tier lookups should narrow to active, non-expired subscribers by default."""

    session = RecordingSession()
    await get_users_by_tier(
        session,
        "plus",
        as_of=datetime(2026, 4, 3, 7, 0, tzinfo=UTC),
    )

    sql = compile_sql(session.statements[0])

    assert "FROM users" in sql
    assert "users.subscription_tier = " in sql
    assert "users.subscription_status = " in sql
    assert "users.subscription_expires_at IS NULL" in sql


@pytest.mark.asyncio
async def test_create_pipeline_run_adds_model_and_flushes() -> None:
    """Pipeline run inserts should create a tracked ORM instance."""

    session = RecordingSession()

    pipeline_run = await create_pipeline_run(
        session,
        PipelineRunCreate(
            run_date=date(2026, 4, 3),
            started_at=datetime(2026, 4, 3, 6, 55, tzinfo=UTC),
        ),
    )

    assert session.flush_count == 1
    assert session.added == [pipeline_run]
    assert isinstance(pipeline_run, PipelineRun)


@pytest.mark.asyncio
async def test_update_pipeline_run_requires_non_empty_patch() -> None:
    """Pipeline run patches should fail fast when nothing is being changed."""

    session = RecordingSession()

    with pytest.raises(ValueError, match="at least one field"):
        await update_pipeline_run(session, uuid4(), PipelineRunUpdate())


def test_fixture_upsert_rejects_blank_sportradar_id() -> None:
    """Fixture payload validation should enforce the canonical fixture identity."""

    with pytest.raises(ValueError, match="sportradar_id must not be blank"):
        FixtureUpsert(
            competition_id=UUID(int=0),
            sportradar_id="   ",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=date(2026, 4, 3),
        )
