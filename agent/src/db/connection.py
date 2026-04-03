"""Async SQLAlchemy connection helpers for PuntLab.

Purpose: centralize the canonical async PostgreSQL engine and session
configuration used by the Python agent.
Scope: database URL normalization, engine/session factory creation, and
managed async session lifecycles for repository code.
Dependencies: relies on `src.config.get_settings` for environment-backed
database configuration and `sqlalchemy.ext.asyncio` for async persistence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings


def normalize_database_url(database_url: str) -> str:
    """Normalize PostgreSQL URLs for SQLAlchemy's asyncpg dialect.

    Args:
        database_url: Raw database URL from environment configuration.

    Returns:
        A PostgreSQL URL that explicitly uses the `asyncpg` driver.

    Raises:
        ValueError: If the URL is blank, malformed, or points to an
            unsupported driver/backend.
    """

    candidate = database_url.strip()
    if not candidate:
        raise ValueError("DATABASE_URL must not be empty.")

    try:
        url = make_url(candidate)
    except Exception as exc:  # pragma: no cover - SQLAlchemy message is enough context.
        raise ValueError(f"Invalid DATABASE_URL: {exc}") from exc

    if url.drivername in {"postgresql", "postgres"}:
        url = url.set(drivername="postgresql+asyncpg")
    elif url.drivername != "postgresql+asyncpg":
        raise ValueError(
            "DATABASE_URL must target PostgreSQL using the asyncpg driver. "
            "Use a URL like 'postgresql+asyncpg://user:pass@host:5432/dbname'."
        )

    return url.render_as_string(hide_password=False)


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Create a configured async SQLAlchemy engine.

    Args:
        database_url: Raw or normalized PostgreSQL connection string.
        echo: Whether SQLAlchemy should log emitted SQL statements.

    Returns:
        A ready-to-use async engine bound to PostgreSQL via `asyncpg`.
    """

    return create_async_engine(
        normalize_database_url(database_url),
        echo=echo,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a reusable async session factory for the supplied engine.

    Args:
        engine: The async engine sessions should bind to.

    Returns:
        A configured `async_sessionmaker` for `AsyncSession` instances.
    """

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide async database engine.

    Returns:
        The cached async SQLAlchemy engine for the configured database.
    """

    settings = get_settings()
    return create_engine(settings.database_url)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory.

    Returns:
        A cached async session factory bound to the shared engine.
    """

    return create_session_factory(get_engine())


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session and ensure cleanup after use.

    Yields:
        An `AsyncSession` ready for transactional work.
    """

    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the cached async engine and clear connection caches.

    This helper keeps tests and future shutdown hooks deterministic by
    releasing open pools before process exit or cache resets.
    """

    engine = get_engine()
    await engine.dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
