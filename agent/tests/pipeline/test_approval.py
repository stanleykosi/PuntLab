"""Tests for PuntLab's approval pipeline node and conditional router.

Purpose: verify soft-gate approval behavior, blocked-ID handling, and router
branch selection before the delivery stage is wired into the full graph.
Scope: unit tests for `src.pipeline.nodes.approval` and `src.pipeline.router`.
Dependencies: pytest plus canonical pipeline/accumulator schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.approval import approval_node
from src.pipeline.router import approval_router
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorStatus,
    ExplainedAccumulator,
    ResolutionSource,
)


def build_explained_accumulator(
    *,
    slip_number: int,
    fixture_ref: str,
    home_team: str,
    away_team: str,
) -> ExplainedAccumulator:
    """Create a canonical explained accumulator for approval-node tests."""

    leg = AccumulatorLeg(
        leg_number=1,
        fixture_ref=fixture_ref,
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=home_team,
        away_team=away_team,
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.8,
        provider="sportybet",
        confidence=0.75,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale=f"{home_team} enters with a stronger form profile.",
    )
    return ExplainedAccumulator(
        slip_date=date(2026, 4, 5),
        slip_number=slip_number,
        legs=(leg,),
        total_odds=1.8,
        leg_count=1,
        confidence=0.75,
        rationale=f"Slip {slip_number} leans on the stronger home side edge.",
    )


@pytest.mark.asyncio
async def test_approval_node_auto_approves_unblocked_slips_and_advances_to_delivery() -> None:
    """Unblocked slips should be approved after the soft-gate wait window."""

    sleep_calls: list[float] = []
    notifier_messages: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        """Capture requested wait durations instead of sleeping in tests."""

        sleep_calls.append(seconds)

    async def notifier(message: str, _: PipelineState) -> None:
        """Capture approval notification messages for assertions."""

        notifier_messages.append(message)

    first_slip = build_explained_accumulator(
        slip_number=1,
        fixture_ref="sr:match:9001",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    second_slip = build_explained_accumulator(
        slip_number=2,
        fixture_ref="sr:match:9002",
        home_team="Liverpool",
        away_team="Brighton",
    )

    result = await approval_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.APPROVAL,
            accumulators=[first_slip, second_slip],
            explained_accumulators=[first_slip, second_slip],
            errors=["Explanation stage completed."],
        ),
        wait_seconds=5,
        sleep=fake_sleep,
        admin_notifier=notifier,
    )

    assert sleep_calls == [5.0]
    assert len(notifier_messages) == 1
    assert "2 accumulators are pending review" in notifier_messages[0]
    assert result["current_stage"] == PipelineStage.DELIVERY
    assert result["approval_status"] == ApprovalStatus.APPROVED
    assert result["blocked_ids"] == []
    assert all(
        slip.status is AccumulatorStatus.APPROVED
        for slip in result["explained_accumulators"]
    )
    assert all(
        slip.status is AccumulatorStatus.APPROVED for slip in result["accumulators"]
    )
    assert result["errors"][0] == "Explanation stage completed."
    assert result["errors"][-1] == (
        "Approval stage completed for run run-2026-04-05-main: approved=2, blocked=0."
    )


@pytest.mark.asyncio
async def test_approval_node_blocks_matching_slips_and_records_invalid_block_ids() -> None:
    """Fixture and slip references in blocked IDs should force blocked status."""

    first_slip = build_explained_accumulator(
        slip_number=1,
        fixture_ref="sr:match:9101",
        home_team="Real Madrid",
        away_team="Valencia",
    )
    second_slip = build_explained_accumulator(
        slip_number=2,
        fixture_ref="sr:match:9102",
        home_team="Barcelona",
        away_team="Getafe",
    )

    async def blocked_ids_resolver(_: PipelineState) -> list[str]:
        """Simulate admin review controls returning extra blocked IDs."""

        return ["slip:2", "   "]

    async def notifier_noop(_message: str, _state: PipelineState) -> None:
        """Skip outbound notification work while exercising block logic."""

        return None

    result = await approval_node(
        PipelineState(
            run_id="run-2026-04-06-main",
            run_date=date(2026, 4, 6),
            started_at=datetime(2026, 4, 6, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.APPROVAL,
            blocked_ids=[" sr:match:9101 "],
            accumulators=[first_slip, second_slip],
            explained_accumulators=[first_slip, second_slip],
        ),
        wait_seconds=0,
        admin_notifier=notifier_noop,
        blocked_ids_resolver=blocked_ids_resolver,
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["approval_status"] == ApprovalStatus.BLOCKED
    assert result["blocked_ids"] == ["sr:match:9101", "slip:2"]
    assert [slip.status for slip in result["explained_accumulators"]] == [
        AccumulatorStatus.BLOCKED,
        AccumulatorStatus.BLOCKED,
    ]
    assert any(
        "Ignored blocked identifier at position 2 because it is blank."
        in message
        for message in result["errors"]
    )
    assert result["errors"][-1] == (
        "Approval stage completed for run run-2026-04-06-main: approved=0, blocked=2."
    )


@pytest.mark.asyncio
async def test_approval_node_marks_run_blocked_when_no_explained_accumulators_exist() -> None:
    """Missing explained accumulators should block delivery and record diagnostics."""

    result = await approval_node(
        PipelineState(
            run_id="run-2026-04-07-main",
            run_date=date(2026, 4, 7),
            started_at=datetime(2026, 4, 7, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.APPROVAL,
            accumulators=[],
            explained_accumulators=[],
        ),
        wait_seconds=0,
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["approval_status"] == ApprovalStatus.BLOCKED
    assert result["accumulators"] == []
    assert result["explained_accumulators"] == []
    assert result["errors"] == [
        (
            "Approval skipped for run run-2026-04-07-main: "
            "no explained accumulators were available."
        )
    ]


def test_approval_router_routes_to_delivery_for_approved_runs() -> None:
    """Approved status should map to the delivery branch."""

    route = approval_router(
        PipelineState(
            run_id="run-2026-04-08-main",
            run_date=date(2026, 4, 8),
            started_at=datetime(2026, 4, 8, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.APPROVAL,
            approval_status=ApprovalStatus.APPROVED,
        )
    )

    assert route == "delivery"


def test_approval_router_routes_to_blocked_for_blocked_runs() -> None:
    """Blocked status should map to the blocked terminal branch."""

    route = approval_router(
        PipelineState(
            run_id="run-2026-04-09-main",
            run_date=date(2026, 4, 9),
            started_at=datetime(2026, 4, 9, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.APPROVAL,
            approval_status=ApprovalStatus.BLOCKED,
        )
    )

    assert route == "blocked"


def test_approval_router_rejects_pending_status() -> None:
    """Pending status should fail fast because approval is not resolved yet."""

    with pytest.raises(ValueError, match="requires a resolved approval status"):
        approval_router(
            PipelineState(
                run_id="run-2026-04-10-main",
                run_date=date(2026, 4, 10),
                started_at=datetime(2026, 4, 10, 8, 0, tzinfo=UTC),
                current_stage=PipelineStage.APPROVAL,
                approval_status=ApprovalStatus.PENDING,
            )
        )
