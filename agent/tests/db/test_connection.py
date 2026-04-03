"""Tests for PuntLab's async SQLAlchemy connection helpers.

Purpose: validate URL normalization and session factory construction so later
database code inherits predictable async behavior.
Scope: pure-unit tests that do not require a live database server.
Dependencies: pytest and SQLAlchemy async primitives.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from src.db.connection import (
    create_engine,
    create_session_factory,
    normalize_database_url,
)


def test_normalize_database_url_upgrades_plain_postgres_scheme() -> None:
    """Ensure canonical asyncpg URLs are derived from plain PostgreSQL URLs."""

    normalized = normalize_database_url("postgresql://puntlab:secret@localhost:5432/puntlab")

    assert normalized == "postgresql+asyncpg://puntlab:secret@localhost:5432/puntlab"


def test_normalize_database_url_rejects_non_async_postgres_drivers() -> None:
    """Ensure unsupported drivers fail fast with a clear configuration error."""

    message = "DATABASE_URL must target PostgreSQL using the asyncpg driver."

    try:
        normalize_database_url("postgresql+psycopg://puntlab:secret@localhost:5432/puntlab")
    except ValueError as exc:
        assert message in str(exc)
    else:  # pragma: no cover - defensive branch for assertion clarity.
        raise AssertionError("normalize_database_url unexpectedly accepted psycopg.")


def test_create_session_factory_produces_async_sessions() -> None:
    """Ensure the session factory is configured for `AsyncSession` usage."""

    engine = create_engine("postgresql://puntlab:secret@localhost:5432/puntlab")
    session_factory = create_session_factory(engine)

    assert session_factory.class_ is AsyncSession
    assert session_factory.kw["expire_on_commit"] is False
    assert session_factory.kw["autoflush"] is False
