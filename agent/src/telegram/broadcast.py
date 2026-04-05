"""Telegram broadcast delivery helpers for PuntLab.

Purpose: deliver daily accumulator recommendations to active users by
subscription tier, with retry handling and per-attempt delivery logging.
Scope: tier distribution normalization, user fan-out, 3-attempt exponential
backoff sending, and database delivery-log persistence.
Dependencies: aiogram bot client, async database query helpers, delivery
schemas, and Telegram message formatters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

from aiogram import Bot
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.connection import get_session
from src.db.models import User
from src.db.queries import DeliveryLogCreate, create_delivery_logs, get_users_by_tier
from src.schemas.accumulators import AccumulatorSlip
from src.schemas.users import DeliveryChannel, DeliveryResult, DeliveryStatus, SubscriptionTier
from src.telegram.formatters import format_accumulator_message

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0

type SessionProvider = Callable[[], AbstractAsyncContextManager[AsyncSession]]
type UserFetcher = Callable[[AsyncSession, SubscriptionTier], Awaitable[Sequence[User]]]
type DeliveryLogWriter = Callable[[AsyncSession, list[DeliveryLogCreate]], Awaitable[None]]
type SleepCallable = Callable[[float], Awaitable[None]]
type MessageFormatter = Callable[[AccumulatorSlip, SubscriptionTier], str]
type TierAccumulatorMap = Mapping[
    SubscriptionTier | str,
    Sequence[AccumulatorSlip],
]


async def broadcast_daily(
    accumulators: TierAccumulatorMap,
    bot: Bot,
    *,
    retries: int = DEFAULT_RETRY_ATTEMPTS,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    session_provider: SessionProvider = get_session,
    user_fetcher: UserFetcher | None = None,
    log_writer: DeliveryLogWriter | None = None,
    sleep: SleepCallable = asyncio.sleep,
    formatter: MessageFormatter | None = None,
) -> list[DeliveryResult]:
    """Broadcast daily recommendations to active users across all tiers.

    Inputs:
        accumulators: Mapping of tier to daily accumulator slips. Missing
            paid tiers fall back to free-tier slips.
        bot: Active aiogram bot used for message delivery.
        retries: Maximum attempts per slip delivery before failing.
        backoff_base_seconds: Base delay used for exponential retry backoff.
        session_provider: Async DB-session provider.
        user_fetcher: Optional user lookup override for tests/runtime wiring.
        log_writer: Optional log persistence override for tests/runtime wiring.
        sleep: Async sleeper used between retry attempts.
        formatter: Slip-to-message formatter.

    Outputs:
        Ordered delivery outcomes for every user/slip send attempt group.

    Raises:
        ValueError: If retry/backoff configuration is invalid or free-tier
            slip distribution is missing.
        TypeError: If tier mapping keys/slips are invalid.
    """

    _validate_retry_config(retries=retries, backoff_base_seconds=backoff_base_seconds)
    tier_distribution = _normalize_tier_accumulators(accumulators)

    resolved_user_fetcher = user_fetcher or _default_user_fetcher
    resolved_log_writer = log_writer or _default_log_writer
    resolved_formatter = formatter or _default_message_formatter

    delivery_results: list[DeliveryResult] = []
    delivery_logs: list[DeliveryLogCreate] = []

    async with session_provider() as session:
        try:
            for tier in (
                SubscriptionTier.FREE,
                SubscriptionTier.PLUS,
                SubscriptionTier.ELITE,
            ):
                tier_users = await resolved_user_fetcher(session, tier)
                tier_slips = tier_distribution[tier]

                for user in tier_users:
                    for slip in tier_slips:
                        result, attempt_logs = await send_to_user(
                            bot=bot,
                            user=user,
                            slip=slip,
                            tier=tier,
                            retries=retries,
                            backoff_base_seconds=backoff_base_seconds,
                            sleep=sleep,
                            formatter=resolved_formatter,
                        )
                        delivery_results.append(result)
                        delivery_logs.extend(attempt_logs)

            if delivery_logs:
                await resolved_log_writer(session, delivery_logs)

            await session.commit()
        except Exception:
            await session.rollback()
            raise

    return delivery_results


async def send_to_user(
    *,
    bot: Bot,
    user: User,
    slip: AccumulatorSlip,
    tier: SubscriptionTier,
    retries: int = DEFAULT_RETRY_ATTEMPTS,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep: SleepCallable = asyncio.sleep,
    formatter: MessageFormatter | None = None,
) -> tuple[DeliveryResult, list[DeliveryLogCreate]]:
    """Send one slip to one user with exponential-backoff retries.

    Inputs:
        bot: Active aiogram bot used for message delivery.
        user: Delivery target user row.
        slip: Accumulator slip to deliver.
        tier: Subscription tier context for formatter output.
        retries: Maximum attempts before final failure.
        backoff_base_seconds: Base backoff delay in seconds.
        sleep: Async sleeper used between retries.
        formatter: Slip-to-message formatter.

    Outputs:
        A tuple of final `DeliveryResult` and per-attempt delivery logs.
    """

    _validate_retry_config(retries=retries, backoff_base_seconds=backoff_base_seconds)
    resolved_formatter = formatter or _default_message_formatter

    if user.telegram_id is None:
        failed_at = datetime.now(UTC)
        error_message = "User has no telegram_id configured."
        return (
            DeliveryResult(
                accumulator_id=slip.slip_id,
                user_id=user.id,
                channel=DeliveryChannel.TELEGRAM,
                status=DeliveryStatus.FAILED,
                subscription_tier=tier,
                recipient=None,
                error_message=error_message,
                delivered_at=failed_at,
            ),
            [
                DeliveryLogCreate(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM.value,
                    status=DeliveryStatus.FAILED.value,
                    error_message=error_message,
                    delivered_at=failed_at,
                )
            ],
        )

    message = resolved_formatter(slip, tier)
    attempt_logs: list[DeliveryLogCreate] = []
    final_error = "Unknown delivery failure."

    for attempt in range(1, retries + 1):
        attempted_at = datetime.now(UTC)
        try:
            await bot.send_message(
                chat_id=user.telegram_id,
                text=message,
                parse_mode="HTML",
            )
        except Exception as exc:
            final_error = str(exc)
            attempt_logs.append(
                DeliveryLogCreate(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM.value,
                    status=DeliveryStatus.FAILED.value,
                    error_message=f"Attempt {attempt}/{retries} failed: {exc}",
                    delivered_at=attempted_at,
                )
            )
            if attempt < retries:
                await sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                continue
            break

        attempt_logs.append(
            DeliveryLogCreate(
                accumulator_id=slip.slip_id,
                user_id=user.id,
                channel=DeliveryChannel.TELEGRAM.value,
                status=DeliveryStatus.SENT.value,
                delivered_at=attempted_at,
            )
        )
        return (
            DeliveryResult(
                accumulator_id=slip.slip_id,
                user_id=user.id,
                channel=DeliveryChannel.TELEGRAM,
                status=DeliveryStatus.SENT,
                subscription_tier=tier,
                recipient=str(user.telegram_id),
                delivered_at=attempted_at,
            ),
            attempt_logs,
        )

    failed_at = datetime.now(UTC)
    return (
        DeliveryResult(
            accumulator_id=slip.slip_id,
            user_id=user.id,
            channel=DeliveryChannel.TELEGRAM,
            status=DeliveryStatus.FAILED,
            subscription_tier=tier,
            recipient=str(user.telegram_id),
            error_message=final_error,
            delivered_at=failed_at,
        ),
        attempt_logs,
    )


def _validate_retry_config(*, retries: int, backoff_base_seconds: float) -> None:
    """Validate retry and backoff configuration values."""

    if retries <= 0:
        raise ValueError("retries must be greater than zero.")
    if backoff_base_seconds <= 0:
        raise ValueError("backoff_base_seconds must be greater than zero.")


def _normalize_tier_accumulators(
    accumulators: TierAccumulatorMap,
) -> dict[SubscriptionTier, tuple[AccumulatorSlip, ...]]:
    """Normalize and validate tier slip mapping for broadcast fan-out.

    Behavior:
        - requires non-empty free-tier slips
        - coerces string tier keys to `SubscriptionTier`
        - falls back missing plus/elite tiers to free-tier slips
    """

    normalized: dict[SubscriptionTier, tuple[AccumulatorSlip, ...]] = {}
    for raw_tier, raw_slips in accumulators.items():
        tier = _coerce_tier(raw_tier)
        slips = tuple(raw_slips)
        for slip in slips:
            if not isinstance(slip, AccumulatorSlip):
                raise TypeError(
                    "broadcast_daily expects AccumulatorSlip instances in tier mappings."
                )
        normalized[tier] = slips

    free_slips = normalized.get(SubscriptionTier.FREE)
    if not free_slips:
        raise ValueError(
            "broadcast_daily requires at least one free-tier accumulator slip."
        )

    normalized.setdefault(SubscriptionTier.PLUS, free_slips)
    normalized.setdefault(SubscriptionTier.ELITE, free_slips)
    return normalized


def _coerce_tier(raw_tier: SubscriptionTier | str) -> SubscriptionTier:
    """Convert tier keys into canonical `SubscriptionTier` enum values."""

    if isinstance(raw_tier, SubscriptionTier):
        return raw_tier
    try:
        return SubscriptionTier(raw_tier.strip().lower())
    except Exception as exc:
        raise TypeError(f"Unsupported subscription tier key: {raw_tier!r}") from exc


async def _default_user_fetcher(
    session: AsyncSession,
    tier: SubscriptionTier,
) -> Sequence[User]:
    """Fetch active users for the specified subscription tier."""

    return await get_users_by_tier(
        session,
        tier.value,
        active_only=True,
    )


async def _default_log_writer(
    session: AsyncSession,
    logs: list[DeliveryLogCreate],
) -> None:
    """Persist delivery logs to the canonical delivery log table."""

    await create_delivery_logs(session, logs)


def _default_message_formatter(
    slip: AccumulatorSlip,
    tier: SubscriptionTier,
) -> str:
    """Render one slip message for delivery using canonical formatters."""

    return format_accumulator_message(slip, tier=tier)


__all__ = ["broadcast_daily", "send_to_user"]
