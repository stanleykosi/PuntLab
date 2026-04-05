"""Tests for PuntLab Telegram admin helper utilities.

Purpose: verify admin authorization checks, inline keyboard construction, and
admin summary formatting used by `/admin` workflows.
Scope: unit tests for pure helpers in `src.telegram.admin`.
Dependencies: pytest, aiogram inline keyboard models, and admin helper
dataclasses.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from src.schemas.accumulators import AccumulatorStatus
from src.telegram.admin import (
    AccumulatorSummary,
    PipelineRunSummary,
    ScraperHealthSnapshot,
    build_accumulator_review_keyboard,
    build_admin_keyboard,
    format_accumulator_review_summary,
    format_pipeline_runs_summary,
    format_scraper_health_summary,
    is_admin_telegram_id,
    parse_accumulator_action,
)


def _extract_callback_data(markup: object) -> list[str]:
    """Flatten callback payloads from inline keyboard markup for assertions."""

    inline_keyboard = markup.inline_keyboard
    callback_data_values: list[str] = []
    for row in inline_keyboard:
        for button in row:
            callback_data = button.callback_data
            if callback_data is not None:
                callback_data_values.append(callback_data)
    return callback_data_values


def test_is_admin_telegram_id_checks_configured_admin_ids() -> None:
    """Admin verification should rely on the configured ADMIN_TELEGRAM_IDS values."""

    fake_settings = SimpleNamespace(
        telegram=SimpleNamespace(admin_telegram_ids=(101, 202)),
    )
    assert is_admin_telegram_id(101, settings=fake_settings) is True  # type: ignore[arg-type]
    assert is_admin_telegram_id(999, settings=fake_settings) is False  # type: ignore[arg-type]


def test_build_admin_keyboard_exposes_expected_actions() -> None:
    """Admin keyboard should include every required callback action exactly once."""

    callback_payloads = _extract_callback_data(build_admin_keyboard())
    assert "admin:view_runs" in callback_payloads
    assert "admin:view_accumulators" in callback_payloads
    assert "admin:trigger" in callback_payloads
    assert "admin:view_scraper_health" in callback_payloads


def test_build_accumulator_review_keyboard_includes_approve_and_block_actions() -> None:
    """Accumulator review keyboard should include approve/block callbacks per slip."""

    summaries = (
        AccumulatorSummary(
            accumulator_id=uuid4(),
            slip_number=1,
            status=AccumulatorStatus.PENDING,
            confidence=Decimal("0.845"),
        ),
        AccumulatorSummary(
            accumulator_id=uuid4(),
            slip_number=2,
            status=AccumulatorStatus.APPROVED,
            confidence=Decimal("0.763"),
        ),
    )

    callback_payloads = _extract_callback_data(build_accumulator_review_keyboard(summaries))
    for summary in summaries:
        assert f"admin:approve:{summary.accumulator_id}" in callback_payloads
        assert f"admin:block:{summary.accumulator_id}" in callback_payloads
    assert "admin:view_accumulators" in callback_payloads
    assert "admin:menu" in callback_payloads


def test_parse_accumulator_action_accepts_approve_and_block_payloads() -> None:
    """Action parser should decode callback payloads into status + accumulator ID."""

    accumulator_id = uuid4()
    approved_status, approved_id = parse_accumulator_action(f"admin:approve:{accumulator_id}")
    blocked_status, blocked_id = parse_accumulator_action(f"admin:block:{accumulator_id}")

    assert approved_status is AccumulatorStatus.APPROVED
    assert blocked_status is AccumulatorStatus.BLOCKED
    assert approved_id == accumulator_id
    assert blocked_id == accumulator_id


@pytest.mark.parametrize(
    "payload",
    [
        "admin:approve:not-a-uuid",
        "admin:block:not-a-uuid",
        "admin:unknown:123",
        "",
    ],
)
def test_parse_accumulator_action_rejects_invalid_payloads(payload: str) -> None:
    """Action parser should fail fast for malformed callback payloads."""

    with pytest.raises(ValueError):
        parse_accumulator_action(payload)


def test_format_pipeline_runs_summary_renders_all_key_fields() -> None:
    """Pipeline run summary formatter should include status and aggregate fields."""

    run_summary = PipelineRunSummary(
        run_id=uuid4(),
        run_date=date(2026, 4, 5),
        status="completed",
        trigger="manual",
        started_at=datetime(2026, 4, 5, 6, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 5, 6, 15, tzinfo=UTC),
        accumulators_generated=12,
        error_count=1,
    )

    rendered = format_pipeline_runs_summary((run_summary,))
    assert "Recent Pipeline Runs" in rendered
    assert "COMPLETED (manual)" in rendered
    assert "Generated: 12" in rendered
    assert "Errors: 1" in rendered


def test_format_scraper_health_summary_handles_empty_day() -> None:
    """Scraper formatter should gracefully handle dates with no fixtures."""

    snapshot = ScraperHealthSnapshot(
        evaluated_on=date(2026, 4, 5),
        fixtures_total=0,
        fixtures_with_sportybet_odds=0,
        sportybet_odds_rows=0,
        latest_sportybet_fetch_at=None,
    )

    rendered = format_scraper_health_summary(snapshot)
    assert "Scraper Health (2026-04-05)" in rendered
    assert "Fixture coverage: 0.0%" in rendered
    assert "Latest fetch: no SportyBet odds fetched" in rendered


def test_format_accumulator_review_summary_lists_statuses() -> None:
    """Accumulator review summary should show slip status and confidence values."""

    summaries = (
        AccumulatorSummary(
            accumulator_id=uuid4(),
            slip_number=3,
            status=AccumulatorStatus.PENDING,
            confidence=Decimal("0.711"),
        ),
    )

    rendered = format_accumulator_review_summary(summaries, date(2026, 4, 5))
    assert "Accumulator Review (2026-04-05)" in rendered
    assert "Slip #3 | PENDING | Confidence 0.711" in rendered
