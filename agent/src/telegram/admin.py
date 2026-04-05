"""Telegram admin command handlers and inline-control workflows for PuntLab.

Purpose: expose admin-only Telegram controls for reviewing pipeline status,
updating accumulator approval states, requesting manual pipeline execution,
and checking SportyBet scraper coverage health.
Scope: `/admin` command handling, inline keyboard builders, callback parsing,
admin authorization checks, and database-backed admin read/write operations.
Dependencies: aiogram routing primitives, PuntLab configuration settings, and
SQLAlchemy async models for pipeline-run, accumulator, fixture, and odds data.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Final
from uuid import UUID

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import func, select, update

from src.config import Settings, get_settings
from src.db.connection import get_session
from src.db.models import Accumulator, Fixture, Odds, PipelineRun
from src.schemas.accumulators import AccumulatorStatus

_ADMIN_MENU_CALLBACK: Final[str] = "admin:menu"
_VIEW_PIPELINE_RUNS_CALLBACK: Final[str] = "admin:view_runs"
_VIEW_ACCUMULATORS_CALLBACK: Final[str] = "admin:view_accumulators"
_MANUAL_TRIGGER_CALLBACK: Final[str] = "admin:trigger"
_VIEW_SCRAPER_HEALTH_CALLBACK: Final[str] = "admin:view_scraper_health"
_APPROVE_ACCUMULATOR_PREFIX: Final[str] = "admin:approve:"
_BLOCK_ACCUMULATOR_PREFIX: Final[str] = "admin:block:"

_MAX_ACTIONABLE_ACCUMULATORS: Final[int] = 10
_DEFAULT_PIPELINE_RUN_LIMIT: Final[int] = 5

_ADMIN_UNAUTHORIZED_MESSAGE: Final[str] = (
    "Admin access denied. Add your Telegram ID to ADMIN_TELEGRAM_IDS to use this command."
)
_MANUAL_TRIGGER_NOT_CONFIGURED_MESSAGE: Final[str] = (
    "Manual pipeline trigger is not configured. "
    "Register one via register_manual_pipeline_trigger() during runtime startup."
)

type ManualPipelineTrigger = Callable[[int], Awaitable[str | None]]
_manual_pipeline_trigger: ManualPipelineTrigger | None = None


@dataclass(slots=True, frozen=True)
class PipelineRunSummary:
    """Compact pipeline-run details used by admin readouts."""

    run_id: UUID
    run_date: date
    status: str
    trigger: str
    started_at: datetime
    completed_at: datetime | None
    accumulators_generated: int
    error_count: int


@dataclass(slots=True, frozen=True)
class AccumulatorSummary:
    """Compact accumulator details shown in admin approval controls."""

    accumulator_id: UUID
    slip_number: int
    status: AccumulatorStatus
    confidence: Decimal


@dataclass(slots=True, frozen=True)
class ScraperHealthSnapshot:
    """SportyBet scraper-coverage snapshot for one calendar day."""

    evaluated_on: date
    fixtures_total: int
    fixtures_with_sportybet_odds: int
    sportybet_odds_rows: int
    latest_sportybet_fetch_at: datetime | None

    @property
    def coverage_ratio(self) -> float:
        """Return fixture-level SportyBet coverage ratio in the inclusive 0..1 range."""

        if self.fixtures_total <= 0:
            return 0.0
        return self.fixtures_with_sportybet_odds / self.fixtures_total


def register_manual_pipeline_trigger(trigger: ManualPipelineTrigger | None) -> None:
    """Register the runtime callback used by `/admin` manual-trigger actions.

    Inputs:
        trigger: Async callback receiving the admin Telegram ID that requested
            the run. The callback returns an optional run-reference string.
    """

    global _manual_pipeline_trigger
    _manual_pipeline_trigger = trigger


def is_admin_telegram_id(telegram_id: int, *, settings: Settings | None = None) -> bool:
    """Return whether the Telegram user ID is configured as an admin."""

    active_settings = settings or get_settings()
    return telegram_id in active_settings.telegram.admin_telegram_ids


def build_admin_keyboard() -> InlineKeyboardMarkup:
    """Build the root inline keyboard for the `/admin` command."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="View Pipeline Runs",
                    callback_data=_VIEW_PIPELINE_RUNS_CALLBACK,
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Review Accumulators",
                    callback_data=_VIEW_ACCUMULATORS_CALLBACK,
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Trigger Manual Run",
                    callback_data=_MANUAL_TRIGGER_CALLBACK,
                ),
            ],
            [
                InlineKeyboardButton(
                    text="View Scraper Health",
                    callback_data=_VIEW_SCRAPER_HEALTH_CALLBACK,
                ),
            ],
        ]
    )


def build_accumulator_review_keyboard(
    accumulators: Sequence[AccumulatorSummary],
) -> InlineKeyboardMarkup:
    """Build inline controls for approving or blocking today's accumulators."""

    inline_keyboard: list[list[InlineKeyboardButton]] = []
    for summary in accumulators[:_MAX_ACTIONABLE_ACCUMULATORS]:
        inline_keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"Approve #{summary.slip_number}",
                    callback_data=f"{_APPROVE_ACCUMULATOR_PREFIX}{summary.accumulator_id}",
                ),
                InlineKeyboardButton(
                    text=f"Block #{summary.slip_number}",
                    callback_data=f"{_BLOCK_ACCUMULATOR_PREFIX}{summary.accumulator_id}",
                ),
            ]
        )

    inline_keyboard.append(
        [
            InlineKeyboardButton(
                text="Refresh",
                callback_data=_VIEW_ACCUMULATORS_CALLBACK,
            ),
            InlineKeyboardButton(text="Back", callback_data=_ADMIN_MENU_CALLBACK),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)


def parse_accumulator_action(callback_data: str) -> tuple[AccumulatorStatus, UUID]:
    """Parse callback payloads for accumulator approve/block actions.

    Inputs:
        callback_data: Raw callback data string from Telegram button presses.

    Outputs:
        Tuple of target accumulator status and accumulator UUID.

    Raises:
        ValueError: If the payload shape or UUID segment is invalid.
    """

    normalized_data = callback_data.strip()
    if normalized_data.startswith(_APPROVE_ACCUMULATOR_PREFIX):
        accumulator_id = normalized_data.removeprefix(_APPROVE_ACCUMULATOR_PREFIX)
        return AccumulatorStatus.APPROVED, UUID(accumulator_id)
    if normalized_data.startswith(_BLOCK_ACCUMULATOR_PREFIX):
        accumulator_id = normalized_data.removeprefix(_BLOCK_ACCUMULATOR_PREFIX)
        return AccumulatorStatus.BLOCKED, UUID(accumulator_id)
    raise ValueError(f"Unsupported accumulator action callback: {callback_data!r}")


async def handle_admin_command(message: Message) -> None:
    """Handle `/admin` command requests with strict admin-only access control."""

    sender = message.from_user
    if sender is None:
        await message.answer("Unable to identify your Telegram account for admin verification.")
        return

    if not is_admin_telegram_id(sender.id):
        await message.answer(_ADMIN_UNAUTHORIZED_MESSAGE)
        return

    await message.answer(
        "PuntLab Admin Panel\n"
        "Choose an action below.",
        reply_markup=build_admin_keyboard(),
    )


async def handle_admin_callback(callback: CallbackQuery) -> None:
    """Handle all admin inline-button actions from the admin panel keyboard."""

    if not is_admin_telegram_id(callback.from_user.id):
        await callback.answer(_ADMIN_UNAUTHORIZED_MESSAGE, show_alert=True)
        return

    callback_data = callback.data or ""
    target_date = _current_wat_date()

    if callback_data == _ADMIN_MENU_CALLBACK:
        await _edit_callback_message(
            callback,
            "PuntLab Admin Panel\nChoose an action below.",
            build_admin_keyboard(),
        )
        return

    if callback_data == _VIEW_PIPELINE_RUNS_CALLBACK:
        runs = await _list_recent_pipeline_runs(limit=_DEFAULT_PIPELINE_RUN_LIMIT)
        await _edit_callback_message(
            callback,
            format_pipeline_runs_summary(runs),
            build_admin_keyboard(),
        )
        return

    if callback_data == _VIEW_ACCUMULATORS_CALLBACK:
        accumulators = await _list_accumulators_for_date(target_date)
        await _edit_callback_message(
            callback,
            format_accumulator_review_summary(accumulators, target_date),
            build_accumulator_review_keyboard(accumulators),
        )
        return

    if callback_data == _VIEW_SCRAPER_HEALTH_CALLBACK:
        health_snapshot = await _collect_scraper_health_snapshot(target_date)
        await _edit_callback_message(
            callback,
            format_scraper_health_summary(health_snapshot),
            build_admin_keyboard(),
        )
        return

    if callback_data == _MANUAL_TRIGGER_CALLBACK:
        message = await _handle_manual_trigger_request(callback.from_user.id)
        await _edit_callback_message(callback, message, build_admin_keyboard())
        return

    if callback_data.startswith(_APPROVE_ACCUMULATOR_PREFIX) or callback_data.startswith(
        _BLOCK_ACCUMULATOR_PREFIX
    ):
        await _handle_accumulator_action(callback, callback_data, target_date)
        return

    await callback.answer("Unknown admin action.", show_alert=True)


def format_pipeline_runs_summary(runs: Sequence[PipelineRunSummary]) -> str:
    """Render a compact pipeline-runs overview for Telegram delivery."""

    if not runs:
        return "No pipeline runs have been recorded yet."

    lines = ["Recent Pipeline Runs:"]
    for run in runs:
        completed_label = (
            run.completed_at.astimezone(get_settings().timezone).strftime("%H:%M")
            if run.completed_at is not None
            else "in-progress"
        )
        lines.append(
            f"#{str(run.run_id)[:8]} | {run.run_date.isoformat()} | "
            f"{run.status.upper()} ({run.trigger})"
        )
        lines.append(
            f"Generated: {run.accumulators_generated} | "
            f"Errors: {run.error_count} | Completed: {completed_label}"
        )
    return "\n".join(lines)


def format_accumulator_review_summary(
    accumulators: Sequence[AccumulatorSummary],
    target_date: date,
) -> str:
    """Render today's accumulator approval status summary."""

    if not accumulators:
        return f"No accumulators found for {target_date.isoformat()}."

    lines = [f"Accumulator Review ({target_date.isoformat()}):"]
    for summary in accumulators:
        lines.append(
            f"Slip #{summary.slip_number} | {summary.status.value.upper()} | "
            f"Confidence {summary.confidence:.3f}"
        )

    if len(accumulators) > _MAX_ACTIONABLE_ACCUMULATORS:
        lines.append(
            f"Only the first {_MAX_ACTIONABLE_ACCUMULATORS} slips have inline action buttons."
        )
    return "\n".join(lines)


def format_scraper_health_summary(snapshot: ScraperHealthSnapshot) -> str:
    """Render SportyBet scraper-health metrics for Telegram admins."""

    latest_fetch_label = (
        snapshot.latest_sportybet_fetch_at.astimezone(get_settings().timezone).isoformat()
        if snapshot.latest_sportybet_fetch_at is not None
        else "no SportyBet odds fetched"
    )
    coverage_percent = round(snapshot.coverage_ratio * 100, 1)
    return (
        f"Scraper Health ({snapshot.evaluated_on.isoformat()}):\n"
        f"Fixtures tracked: {snapshot.fixtures_total}\n"
        f"Fixtures with SportyBet odds: {snapshot.fixtures_with_sportybet_odds}\n"
        f"SportyBet odds rows: {snapshot.sportybet_odds_rows}\n"
        f"Fixture coverage: {coverage_percent}%\n"
        f"Latest fetch: {latest_fetch_label}"
    )


async def _handle_accumulator_action(
    callback: CallbackQuery,
    callback_data: str,
    target_date: date,
) -> None:
    """Apply an approve/block action and re-render the accumulator review panel."""

    try:
        new_status, accumulator_id = parse_accumulator_action(callback_data)
    except ValueError:
        await callback.answer("Invalid accumulator action payload.", show_alert=True)
        return

    updated = await _set_accumulator_status(accumulator_id, new_status)
    accumulators = await _list_accumulators_for_date(target_date)
    summary = format_accumulator_review_summary(accumulators, target_date)
    if not updated:
        summary = f"{summary}\n\nAccumulator update failed: record not found."
    else:
        summary = f"{summary}\n\nUpdated {str(accumulator_id)[:8]} to {new_status.value.upper()}."

    await _edit_callback_message(
        callback,
        summary,
        build_accumulator_review_keyboard(accumulators),
    )


async def _handle_manual_trigger_request(admin_telegram_id: int) -> str:
    """Execute the registered manual-trigger callback and format the user result."""

    if _manual_pipeline_trigger is None:
        return _MANUAL_TRIGGER_NOT_CONFIGURED_MESSAGE

    try:
        run_reference = await _manual_pipeline_trigger(admin_telegram_id)
    except Exception as exc:
        return f"Manual pipeline trigger failed: {exc}"

    if run_reference is None:
        return "Manual pipeline run accepted."
    return f"Manual pipeline run accepted: {run_reference}"


async def _list_recent_pipeline_runs(*, limit: int) -> tuple[PipelineRunSummary, ...]:
    """Fetch recent pipeline runs ordered by newest start time first."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    async with get_session() as session:
        statement = (
            select(PipelineRun)
            .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
            .limit(limit)
        )
        result = await session.execute(statement)
        rows = list(result.scalars().all())

    return tuple(
        PipelineRunSummary(
            run_id=row.id,
            run_date=row.run_date,
            status=row.status,
            trigger=row.trigger,
            started_at=row.started_at,
            completed_at=row.completed_at,
            accumulators_generated=row.accumulators_generated,
            error_count=len(row.errors),
        )
        for row in rows
    )


async def _list_accumulators_for_date(target_date: date) -> tuple[AccumulatorSummary, ...]:
    """Fetch all accumulators for one date sorted by slip number."""

    async with get_session() as session:
        statement = (
            select(Accumulator)
            .where(Accumulator.slip_date == target_date)
            .order_by(Accumulator.slip_number.asc())
        )
        result = await session.execute(statement)
        rows = list(result.scalars().all())

    return tuple(
        AccumulatorSummary(
            accumulator_id=row.id,
            slip_number=row.slip_number,
            status=AccumulatorStatus(row.status),
            confidence=row.confidence,
        )
        for row in rows
    )


async def _set_accumulator_status(accumulator_id: UUID, status: AccumulatorStatus) -> bool:
    """Update an accumulator lifecycle status and return whether a row changed."""

    async with get_session() as session:
        statement = (
            update(Accumulator)
            .where(Accumulator.id == accumulator_id)
            .values(
                status=status.value,
                is_published=False,
                published_at=None,
            )
            .returning(Accumulator.id)
        )
        result = await session.execute(statement)
        updated_id = result.scalar_one_or_none()
        if updated_id is None:
            await session.rollback()
            return False

        await session.commit()
        return True


async def _collect_scraper_health_snapshot(target_date: date) -> ScraperHealthSnapshot:
    """Collect SportyBet fixture coverage metrics for one calendar date."""

    async with get_session() as session:
        total_fixtures_result = await session.execute(
            select(func.count(Fixture.id)).where(Fixture.match_date == target_date)
        )
        total_fixtures = int(total_fixtures_result.scalar_one() or 0)

        sportybet_rows_result = await session.execute(
            select(func.count(Odds.id))
            .join(Fixture, Fixture.id == Odds.fixture_id)
            .where(
                Fixture.match_date == target_date,
                Odds.provider == "sportybet",
            )
        )
        sportybet_odds_rows = int(sportybet_rows_result.scalar_one() or 0)

        covered_fixtures_result = await session.execute(
            select(func.count(func.distinct(Odds.fixture_id)))
            .join(Fixture, Fixture.id == Odds.fixture_id)
            .where(
                Fixture.match_date == target_date,
                Odds.provider == "sportybet",
            )
        )
        covered_fixtures = int(covered_fixtures_result.scalar_one() or 0)

        latest_fetch_result = await session.execute(
            select(func.max(Odds.fetched_at))
            .join(Fixture, Fixture.id == Odds.fixture_id)
            .where(
                Fixture.match_date == target_date,
                Odds.provider == "sportybet",
            )
        )
        latest_fetch = latest_fetch_result.scalar_one_or_none()

    return ScraperHealthSnapshot(
        evaluated_on=target_date,
        fixtures_total=total_fixtures,
        fixtures_with_sportybet_odds=covered_fixtures,
        sportybet_odds_rows=sportybet_odds_rows,
        latest_sportybet_fetch_at=latest_fetch,
    )


def _current_wat_date() -> date:
    """Return today's date in the canonical West Africa Time timezone."""

    return datetime.now(get_settings().timezone).date()


async def _edit_callback_message(
    callback: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Acknowledge callback queries and update the source message safely."""

    await callback.answer()
    callback_message = callback.message
    if callback_message is None:
        return

    try:
        await callback_message.edit_text(text, reply_markup=reply_markup)
    except Exception:
        # Falling back to a new message avoids silent admin UX failures if
        # Telegram rejects an edit (for example when the content is unchanged).
        await callback_message.answer(text, reply_markup=reply_markup)


def build_admin_router() -> Router:
    """Create a fresh admin router with canonical handlers registered."""

    router = Router(name="telegram-admin")
    router.message.register(handle_admin_command, Command("admin"))
    router.callback_query.register(handle_admin_callback, F.data.startswith("admin:"))
    return router


telegram_admin_router = build_admin_router()


__all__ = [
    "AccumulatorSummary",
    "ManualPipelineTrigger",
    "PipelineRunSummary",
    "ScraperHealthSnapshot",
    "build_accumulator_review_keyboard",
    "build_admin_keyboard",
    "build_admin_router",
    "format_accumulator_review_summary",
    "format_pipeline_runs_summary",
    "format_scraper_health_summary",
    "handle_admin_callback",
    "handle_admin_command",
    "is_admin_telegram_id",
    "parse_accumulator_action",
    "register_manual_pipeline_trigger",
    "telegram_admin_router",
]
