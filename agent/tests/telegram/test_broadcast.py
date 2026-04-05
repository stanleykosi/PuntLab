"""Tests for PuntLab Telegram broadcast delivery helpers.

Purpose: verify retry behavior, tier fan-out, and delivery-log persistence for
Telegram broadcast workflows.
Scope: unit tests for `src.telegram.broadcast`.
Dependencies: pytest plus lightweight fake bot/session/runtime objects.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

import pytest
from src.config import MarketType, SportName
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorSlip,
    ResolutionSource,
)
from src.schemas.users import DeliveryStatus, SubscriptionTier
from src.telegram.broadcast import broadcast_daily, send_to_user


@dataclass(slots=True)
class FakeUser:
    """Minimal user shape required by broadcast helpers."""

    id: UUID
    telegram_id: int | None


@dataclass(slots=True)
class FakeBot:
    """Fake bot with programmable send outcomes for retry tests."""

    outcomes: list[Exception | None]
    calls: list[dict[str, object]]

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str,
    ) -> None:
        """Record calls and replay queued success/failure outcomes."""

        self.calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )
        if not self.outcomes:
            return
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome


class RecordingSession:
    """Track commit and rollback calls during broadcast tests."""

    def __init__(self) -> None:
        """Initialize commit/rollback counters."""

        self.commit_count = 0
        self.rollback_count = 0

    async def commit(self) -> None:
        """Record successful commits."""

        self.commit_count += 1

    async def rollback(self) -> None:
        """Record rollback calls on failures."""

        self.rollback_count += 1


def _build_slip(*, slip_number: int, confidence: float) -> AccumulatorSlip:
    """Create a canonical accumulator slip for broadcast tests."""

    leg = AccumulatorLeg(
        leg_number=1,
        fixture_ref=f"sr:match:{9800 + slip_number}",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {slip_number}",
        away_team=f"Away {slip_number}",
        market=MarketType.MATCH_RESULT,
        selection="home",
        odds=1.82,
        provider="sportybet",
        confidence=confidence,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale="Stronger form profile supports the home selection.",
    )
    return AccumulatorSlip(
        slip_date=date(2026, 4, 14),
        slip_number=slip_number,
        legs=(leg,),
        total_odds=1.82,
        leg_count=1,
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_send_to_user_retries_with_exponential_backoff_and_succeeds() -> None:
    """Send helper should retry failed sends and succeed on final attempt."""

    bot = FakeBot(
        outcomes=[RuntimeError("blocked"), RuntimeError("timeout"), None],
        calls=[],
    )
    user = FakeUser(id=uuid4(), telegram_id=12345)
    slip = _build_slip(slip_number=1, confidence=0.8)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        """Record backoff delays rather than sleeping."""

        sleep_calls.append(delay)

    result, logs = await send_to_user(
        bot=bot,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        slip=slip,
        tier=SubscriptionTier.FREE,
        retries=3,
        backoff_base_seconds=0.5,
        sleep=fake_sleep,
        formatter=lambda current_slip, current_tier: (
            f"Slip {current_slip.slip_number} for {current_tier.value}"
        ),
    )

    assert result.status is DeliveryStatus.SENT
    assert len(logs) == 3
    assert logs[0].status == DeliveryStatus.FAILED.value
    assert logs[1].status == DeliveryStatus.FAILED.value
    assert logs[2].status == DeliveryStatus.SENT.value
    assert sleep_calls == [0.5, 1.0]
    assert len(bot.calls) == 3


@pytest.mark.asyncio
async def test_send_to_user_returns_failed_result_for_missing_telegram_id() -> None:
    """Users without Telegram IDs should fail fast with one logged attempt."""

    bot = FakeBot(outcomes=[], calls=[])
    user = FakeUser(id=uuid4(), telegram_id=None)
    slip = _build_slip(slip_number=2, confidence=0.77)

    result, logs = await send_to_user(
        bot=bot,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        slip=slip,
        tier=SubscriptionTier.PLUS,
    )

    assert result.status is DeliveryStatus.FAILED
    assert "no telegram_id" in (result.error_message or "").lower()
    assert len(logs) == 1
    assert logs[0].status == DeliveryStatus.FAILED.value
    assert bot.calls == []


@pytest.mark.asyncio
async def test_broadcast_daily_distributes_by_tier_and_logs_attempts() -> None:
    """Broadcast helper should fan out slips by tier and persist all attempts."""

    free_slip = _build_slip(slip_number=1, confidence=0.85)
    plus_slip_a = _build_slip(slip_number=2, confidence=0.8)
    plus_slip_b = _build_slip(slip_number=3, confidence=0.75)

    bot = FakeBot(outcomes=[], calls=[])
    session = RecordingSession()
    persisted_log_counts: list[int] = []

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Yield the recording session to broadcast helper."""

        yield session

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return one user per tier for deterministic fan-out."""

        if tier is SubscriptionTier.FREE:
            return (FakeUser(id=uuid4(), telegram_id=101),)
        if tier is SubscriptionTier.PLUS:
            return (FakeUser(id=uuid4(), telegram_id=202),)
        if tier is SubscriptionTier.ELITE:
            return (FakeUser(id=uuid4(), telegram_id=303),)
        raise AssertionError(f"Unexpected tier {tier}")

    async def log_writer(_session: RecordingSession, logs: list[object]) -> None:
        """Capture persisted log volume for assertions."""

        persisted_log_counts.append(len(logs))

    results = await broadcast_daily(
        {
            SubscriptionTier.FREE: (free_slip,),
            SubscriptionTier.PLUS: (plus_slip_a, plus_slip_b),
            # Elite intentionally omitted so fallback-to-free behavior is used.
        },
        bot=bot,  # type: ignore[arg-type]
        session_provider=session_provider,
        user_fetcher=user_fetcher,  # type: ignore[arg-type]
        log_writer=log_writer,  # type: ignore[arg-type]
        formatter=lambda slip, tier: f"{tier.value}:{slip.slip_number}",
    )

    # Free user gets 1 slip, plus user gets 2 slips, elite falls back to free (1 slip)
    assert len(results) == 4
    assert all(result.status is DeliveryStatus.SENT for result in results)
    assert len(bot.calls) == 4
    assert persisted_log_counts == [4]
    assert session.commit_count == 1
    assert session.rollback_count == 0


@pytest.mark.asyncio
async def test_broadcast_daily_rolls_back_when_log_writer_fails() -> None:
    """Broadcast helper should rollback DB work when log writes fail."""

    free_slip = _build_slip(slip_number=4, confidence=0.9)
    bot = FakeBot(outcomes=[], calls=[])
    session = RecordingSession()

    @asynccontextmanager
    async def session_provider() -> RecordingSession:
        """Yield the recording session used for rollback assertions."""

        yield session

    async def user_fetcher(
        _session: RecordingSession,
        tier: SubscriptionTier,
    ) -> tuple[FakeUser, ...]:
        """Return one free user and no paid-tier users."""

        if tier is SubscriptionTier.FREE:
            return (FakeUser(id=uuid4(), telegram_id=404),)
        return ()

    async def log_writer(_session: RecordingSession, _logs: list[object]) -> None:
        """Simulate delivery-log persistence failure."""

        raise RuntimeError("delivery log write failed")

    with pytest.raises(RuntimeError, match="delivery log write failed"):
        await broadcast_daily(
            {SubscriptionTier.FREE: (free_slip,)},
            bot=bot,  # type: ignore[arg-type]
            session_provider=session_provider,
            user_fetcher=user_fetcher,  # type: ignore[arg-type]
            log_writer=log_writer,  # type: ignore[arg-type]
            formatter=lambda slip, tier: f"{tier.value}:{slip.slip_number}",
        )

    assert session.commit_count == 0
    assert session.rollback_count == 1
