"""Tests for PuntLab's SQLAlchemy ORM metadata.

Purpose: catch schema drift between the Step 5 migration and the Step 6 ORM
models before query code starts depending on them.
Scope: metadata-level assertions only; no live database required.
Dependencies: SQLAlchemy declarative metadata from `src.db.models`.
"""

from __future__ import annotations

from src.db.models import Base


def test_expected_tables_are_registered() -> None:
    """Ensure all schema tables are represented in ORM metadata."""

    expected_tables = {
        "competitions",
        "fixtures",
        "odds",
        "team_stats",
        "injuries",
        "match_analyses",
        "accumulators",
        "accumulator_legs",
        "users",
        "pipeline_runs",
        "delivery_log",
        "payments",
    }

    assert expected_tables.issubset(Base.metadata.tables.keys())


def test_fixtures_table_retains_named_indexes() -> None:
    """Ensure the fixture indexes match the canonical migration names."""

    fixture_table = Base.metadata.tables["fixtures"]
    index_names = {index.name for index in fixture_table.indexes}

    assert {"idx_fixtures_date", "idx_fixtures_sportradar"} <= index_names


def test_odds_table_retains_composite_uniqueness() -> None:
    """Ensure odds rows remain unique per fixture/provider/market/selection."""

    odds_table = Base.metadata.tables["odds"]
    unique_constraints = list(odds_table.constraints)

    assert any(
        getattr(constraint, "columns", None) is not None
        and tuple(column.name for column in constraint.columns)
        == ("fixture_id", "provider", "market_type", "selection")
        for constraint in unique_constraints
    )
