"""Agent runtime entry point for PuntLab.

Purpose: run the production agent process by wiring readiness checks,
APScheduler daily execution, LangGraph pipeline runs, Telegram polling, and
clean shutdown behavior in one canonical startup path.
Scope: process bootstrap, dependency verification (database + Redis), daily and
manual pipeline triggering, pipeline-run persistence, and lifecycle management.
Dependencies: APScheduler, LangGraph pipeline graph, SQLAlchemy async session
helpers, Redis cache client, and Telegram bot runtime modules.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import text

from src.cache.client import RedisClient
from src.config import WAT_UTC_OFFSET_HOURS, Settings, get_settings
from src.db.connection import dispose_engine, get_session
from src.db.queries import (
    PipelineRunCreate,
    PipelineRunUpdate,
    create_pipeline_run,
    update_pipeline_run,
)
from src.pipeline import PipelineState, build_pipeline
from src.telegram import (
    BotRunMode,
    create_telegram_application,
    register_manual_pipeline_trigger,
    run_telegram_application,
)

LOGGER = logging.getLogger("puntlab.agent")

_DAILY_PIPELINE_JOB_ID: Final[str] = "daily-pipeline-run"
_MANUAL_PIPELINE_JOB_PREFIX: Final[str] = "manual-pipeline-run"
_MANUAL_PIPELINE_DELAY_SECONDS: Final[int] = 1


def configure_logging(log_level: str) -> None:
    """Configure root logging for the agent process.

    Inputs:
        log_level: Desired log level from runtime settings.

    Behavior:
        Installs one process-wide log formatter so scheduler, pipeline, and
        Telegram runtime messages share a consistent structure.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the agent runtime entry point.

    Outputs:
        A configured argument parser supporting startup health checks.
    """

    parser = argparse.ArgumentParser(description="Run the PuntLab agent runtime.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run dependency readiness checks and exit.",
    )
    return parser


def _wat_hour_to_utc_hour(hour_wat: int) -> int:
    """Convert a WAT hour into the equivalent UTC hour.

    Inputs:
        hour_wat: Hour-of-day in West Africa Time (0-23).

    Outputs:
        The matching UTC hour for scheduler registration.

    Raises:
        ValueError: If `hour_wat` is outside the valid 24-hour range.
    """

    if hour_wat < 0 or hour_wat > 23:
        raise ValueError("hour_wat must be in the inclusive range 0..23.")
    return (hour_wat - WAT_UTC_OFFSET_HOURS) % 24


def _build_runtime_run_id(run_date: date) -> str:
    """Build the canonical pipeline runtime run identifier.

    Inputs:
        run_date: The run date in WAT.

    Outputs:
        A deterministic-date prefixed ID with UUID entropy.
    """

    return f"run-{run_date.isoformat()}-{uuid4()}"


class AgentRuntime:
    """Orchestrate PuntLab runtime startup, scheduling, and shutdown.

    Inputs:
        settings: Process-wide validated configuration.
        redis_client: Optional Redis wrapper override used by tests.
        scheduler: Optional APScheduler instance override used by tests.

    Outputs:
        Runtime service that can run readiness checks, start recurring jobs,
        queue manual runs, and execute the Telegram polling loop.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        redis_client: RedisClient | None = None,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        """Initialize runtime dependencies and durable in-memory state."""

        self._settings = settings
        self._redis_client = redis_client or RedisClient(redis_url=settings.redis.url)
        self._scheduler = scheduler or AsyncIOScheduler(
            timezone=UTC,
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 60 * 60,
            },
        )
        self._pipeline = build_pipeline()
        self._pipeline_lock = asyncio.Lock()
        self._started = False

    @property
    def scheduler(self) -> AsyncIOScheduler:
        """Expose the scheduler for runtime introspection and unit tests."""

        return self._scheduler

    async def check_readiness(self) -> None:
        """Run strict dependency checks required by the runtime.

        Behavior:
            Validates Telegram token presence plus live database and Redis
            connectivity so startup fails fast before polling begins.
        """

        self._require_telegram_token()
        await self._verify_database_connectivity()
        await self._verify_redis_connectivity()
        LOGGER.info("Agent readiness checks completed successfully.")

    async def startup(self) -> None:
        """Start scheduler services and register admin manual trigger hooks."""

        if self._started:
            return

        await self.check_readiness()
        self._register_scheduler_jobs()
        self._scheduler.start()
        register_manual_pipeline_trigger(self.queue_manual_pipeline_run)
        self._started = True

        LOGGER.info(
            "Agent runtime started",
            extra={
                "environment": self._settings.environment,
                "pipeline_start_hour_wat": self._settings.pipeline_start_hour,
                "pipeline_start_hour_utc": _wat_hour_to_utc_hour(
                    self._settings.pipeline_start_hour
                ),
                "publish_hour_wat": self._settings.publish_hour,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown runtime resources in deterministic order."""

        register_manual_pipeline_trigger(None)

        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

        await self._redis_client.close()
        await dispose_engine()
        self._started = False

    async def run(self) -> None:
        """Run startup, Telegram polling, and final shutdown cleanup."""

        try:
            await self.startup()
            application = create_telegram_application(token=self._require_telegram_token())
            await run_telegram_application(application, mode=BotRunMode.POLLING)
        finally:
            await self.shutdown()

    async def queue_manual_pipeline_run(self, admin_telegram_id: int) -> str:
        """Queue a one-off manual pipeline run requested from Telegram admin UI.

        Inputs:
            admin_telegram_id: Telegram user ID of the requesting admin.

        Outputs:
            APScheduler job ID returned to the admin command handler.

        Raises:
            RuntimeError: If the scheduler is not running yet.
            ValueError: If `admin_telegram_id` is invalid.
        """

        if admin_telegram_id <= 0:
            raise ValueError("admin_telegram_id must be a positive integer.")
        if not self._scheduler.running:
            raise RuntimeError("Scheduler is not running; cannot queue manual pipeline jobs.")

        run_at = datetime.now(UTC) + timedelta(seconds=_MANUAL_PIPELINE_DELAY_SECONDS)
        job_id = (
            f"{_MANUAL_PIPELINE_JOB_PREFIX}-{admin_telegram_id}-"
            f"{int(run_at.timestamp())}"
        )

        self._scheduler.add_job(
            self._run_manual_pipeline_job,
            trigger=DateTrigger(run_date=run_at, timezone=UTC),
            id=job_id,
            replace_existing=False,
            kwargs={"admin_telegram_id": admin_telegram_id},
        )
        return job_id

    async def _run_scheduled_pipeline_job(self) -> None:
        """Execute one scheduled pipeline run from the daily cron job."""

        await self._run_pipeline(trigger="scheduled")

    async def _run_manual_pipeline_job(self, *, admin_telegram_id: int) -> None:
        """Execute one queued manual pipeline run from an admin request."""

        await self._run_pipeline(trigger="manual", admin_telegram_id=admin_telegram_id)

    async def _run_pipeline(
        self,
        *,
        trigger: str,
        admin_telegram_id: int | None = None,
    ) -> UUID | None:
        """Run one full LangGraph pipeline execution and persist status.

        Inputs:
            trigger: Pipeline trigger source (`scheduled` or `manual`).
            admin_telegram_id: Optional admin actor ID for manual runs.

        Outputs:
            The persisted pipeline-run UUID when a run record was created,
            otherwise `None` when the run was skipped due to concurrency guard.
        """

        if self._pipeline_lock.locked():
            LOGGER.warning(
                "Skipping %s pipeline trigger because another run is already active.",
                trigger,
            )
            return None

        async with self._pipeline_lock:
            run_started_at = datetime.now(UTC)
            run_started_perf = time.perf_counter()
            run_date = datetime.now(self._settings.timezone).date()
            run_id = _build_runtime_run_id(run_date)

            pipeline_run_db_id = await self._persist_run_start(
                trigger=trigger,
                run_date=run_date,
                started_at=run_started_at,
            )

            try:
                initial_state = PipelineState(
                    run_id=run_id,
                    run_date=run_date,
                    started_at=run_started_at,
                )
                pipeline_output = await self._pipeline.ainvoke(
                    initial_state.model_dump(mode="python")
                )
                final_state = PipelineState.model_validate(pipeline_output)
            except Exception as exc:
                elapsed_seconds = round(time.perf_counter() - run_started_perf, 3)
                await self._persist_run_failure(
                    run_id=pipeline_run_db_id,
                    completed_at=datetime.now(UTC),
                    elapsed_seconds=elapsed_seconds,
                    error_message=(
                        f"Pipeline execution failed ({trigger})"
                        + (
                            f" for admin {admin_telegram_id}"
                            if admin_telegram_id is not None
                            else ""
                        )
                        + f": {exc}"
                    ),
                )
                LOGGER.exception(
                    "Pipeline execution failed",
                    extra={
                        "pipeline_run_id": str(pipeline_run_db_id),
                        "trigger": trigger,
                        "admin_telegram_id": admin_telegram_id,
                    },
                )
                return pipeline_run_db_id

            elapsed_seconds = round(time.perf_counter() - run_started_perf, 3)
            await self._persist_run_success(
                run_id=pipeline_run_db_id,
                completed_at=datetime.now(UTC),
                elapsed_seconds=elapsed_seconds,
                state=final_state,
            )
            LOGGER.info(
                "Pipeline execution completed",
                extra={
                    "pipeline_run_id": str(pipeline_run_db_id),
                    "trigger": trigger,
                    "admin_telegram_id": admin_telegram_id,
                    "fixtures": len(final_state.fixtures),
                    "accumulators": len(final_state.accumulators),
                    "published": sum(1 for slip in final_state.accumulators if slip.is_published),
                    "errors": len(final_state.errors),
                    "elapsed_seconds": elapsed_seconds,
                },
            )
            return pipeline_run_db_id

    def _register_scheduler_jobs(self) -> None:
        """Register the canonical daily pipeline cron job."""

        hour_utc = _wat_hour_to_utc_hour(self._settings.pipeline_start_hour)
        self._scheduler.add_job(
            self._run_scheduled_pipeline_job,
            trigger=CronTrigger(hour=hour_utc, minute=0, timezone=UTC),
            id=_DAILY_PIPELINE_JOB_ID,
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=60 * 60,
            coalesce=True,
        )

    def _require_telegram_token(self) -> str:
        """Return the configured Telegram token or fail fast."""

        token = self._settings.telegram.bot_token
        if token is None:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is required for the PuntLab runtime entry point."
            )
        return token

    async def _verify_database_connectivity(self) -> None:
        """Run a fail-fast database connectivity probe."""

        async with get_session() as session:
            await session.execute(text("SELECT 1"))

    async def _verify_redis_connectivity(self) -> None:
        """Run a fail-fast Redis connectivity probe."""

        redis_ok = await self._redis_client.ping()
        if not redis_ok:
            raise RuntimeError("Redis ping returned a non-success response.")

    async def _persist_run_start(
        self,
        *,
        trigger: str,
        run_date: date,
        started_at: datetime,
    ) -> UUID:
        """Insert the initial running pipeline record in the database."""

        async with get_session() as session:
            pipeline_run = await create_pipeline_run(
                session,
                PipelineRunCreate(
                    run_date=run_date,
                    started_at=started_at,
                    status="running",
                    trigger=trigger,
                ),
            )
            await session.commit()
            return pipeline_run.id

    async def _persist_run_success(
        self,
        *,
        run_id: UUID,
        completed_at: datetime,
        elapsed_seconds: float,
        state: PipelineState,
    ) -> None:
        """Update a pipeline-run record after successful graph execution."""

        async with get_session() as session:
            await update_pipeline_run(
                session,
                run_id,
                PipelineRunUpdate(
                    completed_at=completed_at,
                    status="completed",
                    fixtures_analyzed=len(state.fixtures),
                    accumulators_generated=len(state.accumulators),
                    accumulators_published=sum(
                        1 for slip in state.accumulators if slip.is_published
                    ),
                    errors=list(state.errors),
                    stage_timings={
                        "total_seconds": elapsed_seconds,
                        "final_stage": state.current_stage.value,
                    },
                ),
            )
            await session.commit()

    async def _persist_run_failure(
        self,
        *,
        run_id: UUID,
        completed_at: datetime,
        elapsed_seconds: float,
        error_message: str,
    ) -> None:
        """Update a pipeline-run record for failed execution paths."""

        async with get_session() as session:
            await update_pipeline_run(
                session,
                run_id,
                PipelineRunUpdate(
                    completed_at=completed_at,
                    status="failed",
                    errors=[error_message],
                    stage_timings={
                        "total_seconds": elapsed_seconds,
                        "failure": "pipeline_execution",
                    },
                ),
            )
            await session.commit()


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Run the asynchronous PuntLab runtime workflow.

    Inputs:
        argv: Optional argument vector used by tests.

    Outputs:
        Process exit code (`0` on success).
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    configure_logging(settings.log_level)

    runtime = AgentRuntime(settings)
    if args.check:
        try:
            await runtime.check_readiness()
        finally:
            await runtime.shutdown()
        return 0

    await runtime.run()
    return 0


def run() -> int:
    """Execute the PuntLab runtime from synchronous contexts.

    Outputs:
        Process exit code from the async runtime.
    """

    return asyncio.run(async_main())


__all__ = [
    "AgentRuntime",
    "async_main",
    "build_parser",
    "configure_logging",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(run())
