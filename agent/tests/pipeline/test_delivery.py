"""Tests for PuntLab's delivery pipeline node.

Purpose: verify tier-based Telegram delivery, delivery-log persistence, and
failure-path behavior for the final delivery stage.
Scope: unit tests for `src.pipeline.nodes.delivery`.
Dependencies: pytest plus canonical pipeline, accumulator, and user schemas.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import UUID, uuid4

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.delivery import delivery_node
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorStatus,
    ExplainedAccumulator,
    ResolutionSource,
)
from src.schemas.users import DeliveryStatus, SubscriptionTier


@dataclass(slots=True)
class FakeUser:
    """Lightweight user record used by delivery-node tests."""

    id: UUID
    telegram_id: int | None


class RecordingSession:
    """Track commit and rollback calls from delivery-node tests."""

    def __init__(self) -> None:
        """Initialize commit/rollback counters."""

        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        """Record successful transaction commits."""

        self.commit_count += 1

    async def rollback(self) -> None:
        """Record transaction rollbacks for failing paths."""

        self.rollback_count += 1


def build_explained_accumulator(
    *,
    slip_number: int,
    confidence: float,
    status: AccumulatorStatus = AccumulatorStatus.APPROVED,
) -> ExplainedAccumulator:
    """Create a canonical explained accumulator for delivery tests."""

    leg = AccumulatorLeg(
        leg_number=1,
        fixture_ref=f"sr:match:{9300 + slip_number}",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {slip_number}",
        away_team=f"Away {slip_number}",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.80 + (slip_number * 0.05),
        provider="sportybet",
        confidence=confidence,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale="Stronger form and venue profile support the selection.",
    )
    return ExplainedAccumulator(
        slip_date=date(2026, 4, 8),
        slip_number=slip_number,
        legs=(leg,),
        total_odds=leg.odds,
        leg_count=1,
        confidence=confidence,
        rationale=f"Slip {slip_number} leans on the higher-confidence home angle.",
        status=status,
    )


@pytest.mark.asyncio
async def test_delivery_node_distributes_by_tier_and_persists_delivery_logs() -> None:
    """Delivery should send tier-specific slips and persist all attempts."""

    first_slip = build_explained_accumulator(slip_number=1, confidence=0.88)
    second_slip = build_explained_accumulator(slip_number=2, confidence=0.73)

    free_user = FakeUser(id=uuid4(), telegram_id=10101)
    plus_user = FakeUser(id=uuid4(), telegram_id=20202)
    elite_user_without_telegram = FakeUser(id=uuid4(), telegram_id=None)
    sent_messages: list[tuple[int, str]] = []
    persisted_logs_count: list[int] = []
    session = RecordingSession()

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Yield the recording session for node execution."""

        yield session

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return deterministic user sets by subscription tier."""

        if tier is SubscriptionTier.FREE:
            return (free_user,)
        if tier is SubscriptionTier.PLUS:
            return (plus_user,)
        if tier is SubscriptionTier.ELITE:
            return (elite_user_without_telegram,)
        raise AssertionError(f"Unexpected tier: {tier}")

    async def sender(chat_id: int, message: str) -> None:
        """Capture outbound Telegram sends for assertions."""

        sent_messages.append((chat_id, message))

    async def log_writer(_session: RecordingSession, logs: list[object]) -> None:
        """Capture how many delivery logs were persisted."""

        persisted_logs_count.append(len(logs))

    result = await delivery_node(
        PipelineState(
            run_id="run-2026-04-08-main",
            run_date=date(2026, 4, 8),
            started_at=datetime(2026, 4, 8, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.DELIVERY,
            approval_status=ApprovalStatus.APPROVED,
            accumulators=[first_slip, second_slip],
            explained_accumulators=[first_slip, second_slip],
            errors=["Approval stage completed."],
        ),
        session_provider=session_provider,
        user_fetcher=user_fetcher,  # type: ignore[arg-type]
        log_writer=log_writer,  # type: ignore[arg-type]
        telegram_sender=sender,
    )

    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert persisted_logs_count == [5]
    assert len(sent_messages) == 3
    assert all("PUNTLAB — Slip #" in message for _, message in sent_messages)
    assert len(result["delivery_results"]) == 5
    assert sum(
        1 for row in result["delivery_results"] if row.status is DeliveryStatus.SENT
    ) == 3
    assert sum(
        1 for row in result["delivery_results"] if row.status is DeliveryStatus.FAILED
    ) == 2
    assert result["errors"] == ["Approval stage completed."]
    assert all(slip.is_published for slip in result["explained_accumulators"])
    assert all(slip.is_published for slip in result["accumulators"])


@pytest.mark.asyncio
async def test_delivery_node_delivers_only_approved_slips() -> None:
    """Blocked slips should be excluded from delivery attempts."""

    approved = build_explained_accumulator(slip_number=1, confidence=0.85)
    blocked = build_explained_accumulator(
        slip_number=2,
        confidence=0.70,
        status=AccumulatorStatus.BLOCKED,
    )

    user = FakeUser(id=uuid4(), telegram_id=30303)
    sent_messages: list[str] = []

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Provide a disposable recording session."""

        yield RecordingSession()

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return one free user and no paid-tier users."""

        if tier is SubscriptionTier.FREE:
            return (user,)
        return ()

    async def sender(_chat_id: int, message: str) -> None:
        """Capture message bodies to verify delivered slip selection."""

        sent_messages.append(message)

    result = await delivery_node(
        PipelineState(
            run_id="run-2026-04-09-main",
            run_date=date(2026, 4, 9),
            started_at=datetime(2026, 4, 9, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.DELIVERY,
            approval_status=ApprovalStatus.APPROVED,
            accumulators=[approved, blocked],
            explained_accumulators=[approved, blocked],
        ),
        session_provider=session_provider,
        user_fetcher=user_fetcher,  # type: ignore[arg-type]
        log_writer=lambda _session, _logs: _async_noop(),  # type: ignore[arg-type]
        telegram_sender=sender,
    )

    assert len(sent_messages) == 1
    assert "Slip #1" in sent_messages[0]
    assert all("Slip #2" not in message for message in sent_messages)
    assert len(result["delivery_results"]) == 1
    assert result["delivery_results"][0].status is DeliveryStatus.SENT


@pytest.mark.asyncio
async def test_delivery_node_records_send_failures_and_continues() -> None:
    """Telegram send failures should produce failed delivery rows."""

    slip = build_explained_accumulator(slip_number=3, confidence=0.82)
    user = FakeUser(id=uuid4(), telegram_id=40404)

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Provide a disposable recording session."""

        yield RecordingSession()

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return one free-tier user and no users for other tiers."""

        if tier is SubscriptionTier.FREE:
            return (user,)
        return ()

    async def sender(_chat_id: int, _message: str) -> None:
        """Simulate a Telegram API failure for this delivery attempt."""

        raise RuntimeError("chat not found")

    result = await delivery_node(
        PipelineState(
            run_id="run-2026-04-10-main",
            run_date=date(2026, 4, 10),
            started_at=datetime(2026, 4, 10, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.DELIVERY,
            approval_status=ApprovalStatus.APPROVED,
            accumulators=[slip],
            explained_accumulators=[slip],
        ),
        session_provider=session_provider,
        user_fetcher=user_fetcher,  # type: ignore[arg-type]
        log_writer=lambda _session, _logs: _async_noop(),  # type: ignore[arg-type]
        telegram_sender=sender,
    )

    assert len(result["delivery_results"]) == 1
    assert result["delivery_results"][0].status is DeliveryStatus.FAILED
    assert "Telegram send failed: chat not found" in (
        result["delivery_results"][0].error_message or ""
    )
    assert any(
        "All delivery attempts failed" in message for message in result["errors"]
    )


@pytest.mark.asyncio
async def test_delivery_node_fails_fast_when_approval_not_approved() -> None:
    """Delivery should reject execution unless approval has completed."""

    with pytest.raises(ValueError, match="requires approval_status='approved'"):
        await delivery_node(
            PipelineState(
                run_id="run-2026-04-11-main",
                run_date=date(2026, 4, 11),
                started_at=datetime(2026, 4, 11, 8, 0, tzinfo=UTC),
                current_stage=PipelineStage.DELIVERY,
                approval_status=ApprovalStatus.BLOCKED,
            ),
            telegram_sender=lambda _chat_id, _message: _async_noop(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_delivery_node_rolls_back_when_log_write_fails() -> None:
    """Database write failures should trigger rollback before propagating."""

    slip = build_explained_accumulator(slip_number=4, confidence=0.90)
    user = FakeUser(id=uuid4(), telegram_id=50505)
    session = RecordingSession()

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Yield the recording session for rollback assertions."""

        yield session

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return one free-tier user only."""

        if tier is SubscriptionTier.FREE:
            return (user,)
        return ()

    async def log_writer(_session: RecordingSession, _logs: list[object]) -> None:
        """Simulate a database-layer failure while persisting logs."""

        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await delivery_node(
            PipelineState(
                run_id="run-2026-04-12-main",
                run_date=date(2026, 4, 12),
                started_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
                current_stage=PipelineStage.DELIVERY,
                approval_status=ApprovalStatus.APPROVED,
                accumulators=[slip],
                explained_accumulators=[slip],
            ),
            session_provider=session_provider,
            user_fetcher=user_fetcher,  # type: ignore[arg-type]
            log_writer=log_writer,  # type: ignore[arg-type]
            telegram_sender=lambda _chat_id, _message: _async_noop(),  # type: ignore[arg-type]
        )

    assert session.commit_count == 0
    assert session.rollback_count == 1


async def _async_noop() -> None:
    """Return immediately for async callback injection in tests."""

    return None
