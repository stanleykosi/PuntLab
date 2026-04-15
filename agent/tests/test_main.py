"""Tests for PuntLab's agent runtime entry point.

Purpose: verify scheduler registration, WAT-to-UTC conversion, manual trigger
queueing, and runtime lifecycle wiring introduced by the main runtime module.
Scope: unit tests for `src.main` with fake scheduler/Redis dependencies and no
external database, Redis, or Telegram calls.
Dependencies: pytest, APScheduler trigger objects, and `src.main.AgentRuntime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from src.config import Settings
from src.main import AgentRuntime, _wat_hour_to_utc_hour


@dataclass(slots=True)
class FakeRedisClient:
    """Minimal Redis client double for runtime lifecycle tests."""

    ping_result: bool = True
    ping_calls: int = 0
    close_calls: int = 0

    async def ping(self) -> bool:
        """Record ping calls and return the configured probe result."""

        self.ping_calls += 1
        return self.ping_result

    async def close(self) -> None:
        """Record runtime close calls for lifecycle assertions."""

        self.close_calls += 1


class FakeScheduler:
    """In-memory APScheduler double used by runtime tests."""

    def __init__(self, *, running: bool = False) -> None:
        """Initialize stateful scheduler call tracking."""

        self.running = running
        self.add_job_calls: list[dict[str, Any]] = []
        self.start_calls = 0
        self.shutdown_calls: list[bool] = []

    def add_job(self, func: object, **kwargs: object) -> None:
        """Record scheduler job registrations for assertions."""

        self.add_job_calls.append({"func": func, **kwargs})

    def start(self) -> None:
        """Switch scheduler state to running."""

        self.running = True
        self.start_calls += 1

    def shutdown(self, *, wait: bool = True) -> None:
        """Record shutdown behavior and mark the scheduler as stopped."""

        self.running = False
        self.shutdown_calls.append(wait)


class _FakePipeline:
    """Small pipeline stub for runtime construction in tests."""

    async def ainvoke(self, payload: object) -> object:
        """Echo payloads to satisfy runtime invocation contracts in tests."""

        return payload


def _build_settings(*, telegram_bot_token: str | None = "123456:token") -> Settings:
    """Create deterministic settings for runtime unit tests."""

    return Settings(
        _env_file=None,
        database_url="postgresql://puntlab:puntlab@localhost:5432/puntlab",
        redis_url="redis://localhost:6379/0",
        telegram_bot_token=telegram_bot_token,
    )


def _get_trigger_field(trigger: object, field_name: str) -> str:
    """Extract one APScheduler trigger-field string value by name."""

    fields = getattr(trigger, "fields", ())
    for field in fields:
        if getattr(field, "name", None) == field_name:
            return str(field)
    raise AssertionError(f"Trigger field '{field_name}' was not present.")


@pytest.mark.parametrize(
    ("hour_wat", "expected_utc_hour"),
    [(0, 23), (1, 0), (7, 6), (10, 9), (23, 22)],
)
def test_wat_hour_to_utc_hour_applies_fixed_offset(
    hour_wat: int,
    expected_utc_hour: int,
) -> None:
    """WAT scheduler hours should map to UTC using a fixed one-hour offset."""

    assert _wat_hour_to_utc_hour(hour_wat) == expected_utc_hour


def test_wat_hour_to_utc_hour_rejects_invalid_hours() -> None:
    """Hour conversion should fail fast for non-24-hour values."""

    with pytest.raises(ValueError, match="inclusive range"):
        _wat_hour_to_utc_hour(-1)

    with pytest.raises(ValueError, match="inclusive range"):
        _wat_hour_to_utc_hour(24)


@pytest.mark.asyncio
async def test_runtime_startup_registers_daily_pipeline_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should register one daily cron job at 06:00 UTC by default."""

    settings = _build_settings()
    fake_scheduler = FakeScheduler()
    fake_redis = FakeRedisClient()
    register_calls: list[object] = []

    monkeypatch.setattr("src.main.build_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "src.main.register_manual_pipeline_trigger",
        lambda callback: register_calls.append(callback),
    )

    runtime = AgentRuntime(  # type: ignore[arg-type]
        settings,
        redis_client=fake_redis,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
    )

    async def fake_db_check() -> None:
        """Skip live database probes in this unit test."""

    monkeypatch.setattr(runtime, "_verify_database_connectivity", fake_db_check)

    await runtime.startup()

    assert fake_scheduler.start_calls == 1
    assert len(fake_scheduler.add_job_calls) == 1
    job_call = fake_scheduler.add_job_calls[0]
    assert job_call["id"] == "daily-pipeline-run"
    assert _get_trigger_field(job_call["trigger"], "minute") == "0"
    assert _get_trigger_field(job_call["trigger"], "hour") == "6"
    assert register_calls == [runtime.queue_manual_pipeline_run]


@pytest.mark.asyncio
async def test_runtime_shutdown_unregisters_manual_trigger_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown should unregister manual trigger callback and close dependencies."""

    settings = _build_settings()
    fake_scheduler = FakeScheduler(running=True)
    fake_redis = FakeRedisClient()
    register_calls: list[object] = []

    monkeypatch.setattr("src.main.build_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "src.main.register_manual_pipeline_trigger",
        lambda callback: register_calls.append(callback),
    )

    runtime = AgentRuntime(  # type: ignore[arg-type]
        settings,
        redis_client=fake_redis,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
    )

    await runtime.shutdown()

    assert register_calls == [None]
    assert fake_scheduler.shutdown_calls == [False]
    assert fake_redis.close_calls == 1


@pytest.mark.asyncio
async def test_manual_trigger_requires_running_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual trigger requests should fail before scheduler startup."""

    settings = _build_settings()
    fake_scheduler = FakeScheduler(running=False)

    monkeypatch.setattr("src.main.build_pipeline", lambda: _FakePipeline())

    runtime = AgentRuntime(  # type: ignore[arg-type]
        settings,
        redis_client=FakeRedisClient(),  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="Scheduler is not running"):
        await runtime.queue_manual_pipeline_run(123)


@pytest.mark.asyncio
async def test_manual_trigger_queues_one_off_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual trigger requests should enqueue a date-triggered scheduler job."""

    settings = _build_settings()
    fake_scheduler = FakeScheduler(running=True)

    monkeypatch.setattr("src.main.build_pipeline", lambda: _FakePipeline())

    runtime = AgentRuntime(  # type: ignore[arg-type]
        settings,
        redis_client=FakeRedisClient(),  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
    )

    job_id = await runtime.queue_manual_pipeline_run(999001)

    assert job_id.startswith("manual-pipeline-run-999001-")
    assert len(fake_scheduler.add_job_calls) == 1
    job_call = fake_scheduler.add_job_calls[0]
    assert job_call["id"] == job_id
    assert job_call["kwargs"] == {"admin_telegram_id": 999001}
    assert job_call["replace_existing"] is False


@pytest.mark.asyncio
async def test_check_readiness_requires_telegram_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness checks should fail fast when Telegram bot token is missing."""

    settings = _build_settings(telegram_bot_token=None)
    monkeypatch.setattr("src.main.build_pipeline", lambda: _FakePipeline())

    runtime = AgentRuntime(  # type: ignore[arg-type]
        settings,
        redis_client=FakeRedisClient(),  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        await runtime.check_readiness()
