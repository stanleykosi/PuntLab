"""Tests for the quick-run pipeline progress CLI.

Purpose: verify that quick-run execution reports stage progress, produces a
final summary, and handles optional delivery failures without losing analysis
output.
Scope: unit tests for `src.pipeline.quick_run` with stub node functions and no
external provider, database, or Telegram dependencies.
Dependencies: pytest, canonical pipeline enums, and `run_quick_pipeline`.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest
from src.config import MarketType, SportName
from src.pipeline.quick_run import StageNode, run_quick_pipeline
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorStatus,
    ExplainedAccumulator,
    ResolutionSource,
)
from src.schemas.users import DeliveryChannel, DeliveryStatus


def _capture_output(buffer: list[str]) -> callable:
    """Return a print-compatible callable that appends log lines to a buffer."""

    def emit(line: str) -> None:
        """Append one progress line emitted by quick-run."""

        buffer.append(line)

    return emit


def _build_approved_explained_slip(*, slip_number: int) -> ExplainedAccumulator:
    """Create one approved explained accumulator for quick-run delivery tests."""

    leg = AccumulatorLeg(
        leg_number=1,
        fixture_ref=f"sr:match:{93000 + slip_number}",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {slip_number}",
        away_team=f"Away {slip_number}",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.85,
        provider="sportybet",
        confidence=0.78,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale="Home side carries the stronger recent form and chance creation profile.",
    )
    return ExplainedAccumulator(
        slip_date=date(2026, 4, 5),
        slip_number=slip_number,
        legs=(leg,),
        total_odds=1.85,
        leg_count=1,
        confidence=0.78,
        rationale="Single-leg anchor with stable home edge and manageable variance.",
        status=AccumulatorStatus.APPROVED,
    )


@pytest.mark.asyncio
async def test_quick_run_reports_stage_progress_and_final_summary() -> None:
    """Quick-run should emit progress lines and a success summary for happy paths."""

    logs: list[str] = []

    async def ingestion_stub(_: PipelineState) -> dict[str, object]:
        """Simulate a successful ingestion stage update."""

        return {"current_stage": PipelineStage.RESEARCH}

    async def research_stub(_: PipelineState) -> dict[str, object]:
        """Simulate a successful research stage update."""

        return {"current_stage": PipelineStage.SCORING}

    async def approval_stub(_: PipelineState) -> dict[str, object]:
        """Block approval so delivery is skipped in this test."""

        return {
            "current_stage": PipelineStage.APPROVAL,
            "approval_status": ApprovalStatus.BLOCKED,
        }

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=False,
        approval_wait_seconds=0,
        ignore_delivery_errors=True,
        node_timeout_seconds=5.0,
        show_traceback=False,
        max_error_lines=5,
        print_fn=_capture_output(logs),
        pre_approval_nodes=(
            StageNode(name="ingestion", node=ingestion_stub),
            StageNode(name="research", node=research_stub),
        ),
        approval_stage=StageNode(name="approval", node=approval_stub),
    )

    assert report.success is True
    assert len(report.stage_records) == 3
    assert all(record.status == "ok" for record in report.stage_records)
    assert any("start ingestion" in line for line in logs)
    assert any("done research" in line for line in logs)
    assert any("final report" in line for line in logs)


@pytest.mark.asyncio
async def test_quick_run_can_ignore_delivery_failures_and_keep_summary() -> None:
    """Delivery failures should be reportable without aborting quick-run output."""

    logs: list[str] = []

    async def approval_stub(_: PipelineState) -> dict[str, object]:
        """Approve the run so quick-run attempts delivery."""

        return {
            "current_stage": PipelineStage.APPROVAL,
            "approval_status": ApprovalStatus.APPROVED,
        }

    async def failing_delivery(_: PipelineState) -> dict[str, object]:
        """Simulate a network delivery outage."""

        raise OSError("Network is unreachable")

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=True,
        approval_wait_seconds=0,
        ignore_delivery_errors=True,
        node_timeout_seconds=5.0,
        show_traceback=False,
        max_error_lines=5,
        print_fn=_capture_output(logs),
        pre_approval_nodes=(),
        approval_stage=StageNode(name="approval", node=approval_stub),
        delivery_stage=StageNode(name="delivery", node=failing_delivery),
        approval_route_resolver=lambda _state: "delivery",
    )

    assert report.success is True
    assert report.stage_records[-1].name == "delivery"
    assert report.stage_records[-1].status == "failed_ignored"
    assert any("Stage 'delivery' failed" in message for message in report.state.errors)
    assert any("delivery failure was ignored" in line for line in logs)


@pytest.mark.asyncio
async def test_quick_run_fails_when_delivery_is_strict() -> None:
    """Strict delivery mode should fail the run when delivery raises errors."""

    async def approval_stub(_: PipelineState) -> dict[str, object]:
        """Approve the run so delivery is attempted."""

        return {
            "current_stage": PipelineStage.APPROVAL,
            "approval_status": ApprovalStatus.APPROVED,
        }

    async def failing_delivery(_: PipelineState) -> dict[str, object]:
        """Raise a deterministic error for strict failure validation."""

        raise RuntimeError("delivery unavailable")

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=True,
        approval_wait_seconds=0,
        ignore_delivery_errors=False,
        node_timeout_seconds=5.0,
        show_traceback=False,
        max_error_lines=5,
        print_fn=lambda _line: None,
        pre_approval_nodes=(),
        approval_stage=StageNode(name="approval", node=approval_stub),
        delivery_stage=StageNode(name="delivery", node=failing_delivery),
        approval_route_resolver=lambda _state: "delivery",
    )

    assert report.success is False
    assert report.stage_records[-1].status == "failed"


@pytest.mark.asyncio
async def test_quick_run_accepts_zero_timeout_for_long_stages() -> None:
    """A zero timeout should disable timeout enforcement for stage execution."""

    async def slow_stage(_: PipelineState) -> dict[str, object]:
        """Simulate a stage that takes longer than a small timeout value."""

        await asyncio.sleep(0.05)
        return {"current_stage": PipelineStage.RESEARCH}

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=False,
        approval_wait_seconds=0,
        ignore_delivery_errors=True,
        node_timeout_seconds=0.0,
        show_traceback=False,
        max_error_lines=5,
        print_fn=lambda _line: None,
        pre_approval_nodes=(StageNode(name="slow", node=slow_stage),),
        approval_stage=StageNode(
            name="approval",
            node=lambda _state: asyncio.sleep(
                0,
                result={
                    "current_stage": PipelineStage.APPROVAL,
                    "approval_status": ApprovalStatus.BLOCKED,
                },
            ),
        ),
    )

    assert report.success is True
    assert report.stage_records[0].name == "slow"
    assert report.stage_records[0].status == "ok"


@pytest.mark.asyncio
async def test_quick_run_emits_live_heartbeat_lines_for_long_stages() -> None:
    """Live dashboard should print heartbeat lines while a stage is running."""

    logs: list[str] = []

    async def slow_stage(_: PipelineState) -> dict[str, object]:
        """Simulate a long-running stage to trigger heartbeat output."""

        await asyncio.sleep(0.75)
        return {"current_stage": PipelineStage.RESEARCH}

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=False,
        approval_wait_seconds=0,
        ignore_delivery_errors=True,
        node_timeout_seconds=0.0,
        show_traceback=False,
        max_error_lines=5,
        heartbeat_seconds=0.5,
        live_dashboard_enabled=True,
        print_fn=_capture_output(logs),
        pre_approval_nodes=(StageNode(name="slow", node=slow_stage),),
        approval_stage=StageNode(
            name="approval",
            node=lambda _state: asyncio.sleep(
                0,
                result={
                    "current_stage": PipelineStage.APPROVAL,
                    "approval_status": ApprovalStatus.BLOCKED,
                },
            ),
        ),
    )

    assert report.success is True
    assert any("live slow" in line for line in logs)


@pytest.mark.asyncio
async def test_quick_run_default_delivery_publishes_to_cli_without_external_dependencies() -> None:
    """Default quick-run delivery should publish approved slips in CLI output."""

    logs: list[str] = []
    approved_slip = _build_approved_explained_slip(slip_number=1)

    async def pre_stage(_: PipelineState) -> dict[str, object]:
        """Inject one approved explained slip before approval executes."""

        return {
            "current_stage": PipelineStage.EXPLANATION,
            "accumulators": [approved_slip],
            "explained_accumulators": [approved_slip],
        }

    async def approval_stub(_: PipelineState) -> dict[str, object]:
        """Approve the run so quick-run delivery executes."""

        return {
            "current_stage": PipelineStage.APPROVAL,
            "approval_status": ApprovalStatus.APPROVED,
        }

    report = await run_quick_pipeline(
        run_date=date(2026, 4, 5),
        with_delivery=True,
        approval_wait_seconds=0,
        ignore_delivery_errors=False,
        node_timeout_seconds=5.0,
        show_traceback=False,
        max_error_lines=5,
        print_fn=_capture_output(logs),
        pre_approval_nodes=(StageNode(name="pre_stage", node=pre_stage),),
        approval_stage=StageNode(name="approval", node=approval_stub),
        approval_route_resolver=lambda _state: "delivery",
    )

    assert report.success is True
    assert report.stage_records[-1].name == "delivery"
    assert report.stage_records[-1].status == "ok"
    assert len(report.state.delivery_results) == 1
    assert report.state.delivery_results[0].channel is DeliveryChannel.API
    assert report.state.delivery_results[0].status is DeliveryStatus.SENT
    assert report.state.delivery_results[0].recipient == "cli"
    assert report.state.explained_accumulators[0].is_published is True
    assert any("slips (showing up to" in line for line in logs)
    assert any("leg 1: Home 1 vs Away 1" in line for line in logs)
