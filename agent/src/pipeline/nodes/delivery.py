"""Delivery node for PuntLab's LangGraph pipeline.

Purpose: distribute approved accumulator slips to subscribed users via
Telegram, persist delivery attempts, and expose per-attempt outcomes for
pipeline observability.
Scope: tier-based slip distribution, Telegram send orchestration, delivery-log
persistence, and publication metadata updates for delivered accumulators.
Dependencies: `src.accumulators.distributor` for entitlement slicing,
`src.db.queries` for user lookup and delivery-log writes, and `aiogram` for
Telegram message delivery.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from aiogram import Bot
from sqlalchemy.ext.asyncio import AsyncSession

from src.accumulators import distribute_to_tiers
from src.config import get_settings
from src.db.connection import get_session
from src.db.models import User
from src.db.queries import DeliveryLogCreate, create_delivery_logs, get_users_by_tier
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorSlip, AccumulatorStatus, ExplainedAccumulator
from src.schemas.users import DeliveryChannel, DeliveryResult, DeliveryStatus, SubscriptionTier

type SessionProvider = Callable[[], AbstractAsyncContextManager[AsyncSession]]
type TierUserFetcher = Callable[
    [AsyncSession, SubscriptionTier], Awaitable[Sequence[User]]
]
type DeliveryLogWriter = Callable[
    [AsyncSession, list[DeliveryLogCreate]], Awaitable[None]
]
type TelegramSender = Callable[[int, str], Awaitable[None]]
_DELIVERY_SEND_MAX_ATTEMPTS = 3
_DELIVERY_SEND_RETRY_BASE_SECONDS = 1.0


async def delivery_node(
    state: PipelineState | Mapping[str, Any],
    *,
    session_provider: SessionProvider = get_session,
    user_fetcher: TierUserFetcher | None = None,
    log_writer: DeliveryLogWriter | None = None,
    telegram_sender: TelegramSender | None = None,
) -> dict[str, object]:
    """Execute the delivery stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        session_provider: Async session-context provider used for database
            reads/writes.
        user_fetcher: Optional user-fetch callback override used by tests.
        log_writer: Optional delivery-log persistence override used by tests.
        telegram_sender: Optional Telegram send callback override used by tests
            or custom runtime integrations.

    Outputs:
        A partial LangGraph update containing delivery results, published
        accumulator updates, and merged diagnostics.

    Raises:
        ValueError: If approval has not resolved to approved before delivery.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    if validated_state.approval_status is not ApprovalStatus.APPROVED:
        raise ValueError(
            "delivery_node requires approval_status='approved' before delivery."
        )

    approved_explained = _approved_explained_accumulators(
        validated_state.explained_accumulators
    )
    if not approved_explained:
        raise ValueError(
            "delivery_node requires at least one approved explained accumulator."
        )

    tier_distribution = distribute_to_tiers(approved_explained)
    resolved_user_fetcher = user_fetcher or _default_user_fetcher
    resolved_log_writer = log_writer or _default_log_writer

    diagnostics: list[str] = []
    delivery_results: list[DeliveryResult] = []
    delivery_logs: list[DeliveryLogCreate] = []
    users_processed = 0

    async with _telegram_sender_context(telegram_sender) as send_message:
        try:
            async with session_provider() as session:
                try:
                    for tier in (
                        SubscriptionTier.FREE,
                        SubscriptionTier.PLUS,
                        SubscriptionTier.ELITE,
                    ):
                        tier_users = await resolved_user_fetcher(session, tier)
                        if not tier_users:
                            diagnostics.append(
                                f"No active users found for tier `{tier.value}` during delivery."
                            )
                            continue

                        tier_slips = tier_distribution[tier]
                        for user in tier_users:
                            users_processed += 1
                            user_results, user_logs = await _deliver_to_user(
                                user=user,
                                tier=tier,
                                slips=tier_slips,
                                send_message=send_message,
                            )
                            delivery_results.extend(user_results)
                            delivery_logs.extend(user_logs)

                    if delivery_logs:
                        await resolved_log_writer(session, delivery_logs)

                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
        except Exception as exc:
            admin_fallback = await _deliver_to_admin_fallback(
                slips=approved_explained,
                send_message=send_message,
                cause=exc,
            )
            if admin_fallback is None:
                raise
            fallback_results, fallback_diagnostics = admin_fallback
            delivery_results.extend(fallback_results)
            diagnostics.extend(fallback_diagnostics)

    if users_processed == 0:
        diagnostics.append("No active users were processed during delivery.")
    if delivery_results and not any(
        result.status is DeliveryStatus.SENT for result in delivery_results
    ):
        diagnostics.append("All delivery attempts failed; no Telegram messages were sent.")

    published_at = datetime.now(UTC)
    published_explained = _mark_published_explained(
        validated_state.explained_accumulators,
        approved_explained,
        published_at=published_at,
    )
    published_accumulators = _mark_published_accumulators(
        validated_state.accumulators,
        published_explained,
    )

    return {
        "current_stage": PipelineStage.DELIVERY,
        "explained_accumulators": published_explained,
        "accumulators": published_accumulators,
        "delivery_results": delivery_results,
        "errors": _merge_diagnostics(validated_state.errors, diagnostics),
    }


def _approved_explained_accumulators(
    explained_accumulators: Sequence[ExplainedAccumulator],
) -> tuple[ExplainedAccumulator, ...]:
    """Return approved explained accumulators in source order."""

    return tuple(
        slip
        for slip in explained_accumulators
        if slip.status is AccumulatorStatus.APPROVED
    )


async def _deliver_to_user(
    *,
    user: User,
    tier: SubscriptionTier,
    slips: Sequence[AccumulatorSlip],
    send_message: TelegramSender,
) -> tuple[list[DeliveryResult], list[DeliveryLogCreate]]:
    """Deliver tier-eligible slips to one user and collect attempt outputs.

    Inputs:
        user: Persisted user row targeted for Telegram delivery.
        tier: Subscription tier used to select this user's slip slice.
        slips: Tier-filtered slip slate to deliver to this user.
        send_message: Async Telegram send callable.

    Outputs:
        A tuple of `(delivery_results, delivery_logs)` with one record per
        attempted slip delivery.
    """

    user_results: list[DeliveryResult] = []
    user_logs: list[DeliveryLogCreate] = []

    for slip in slips:
        if user.telegram_id is None:
            error_message = "User has no telegram_id configured."
            attempt_time = datetime.now(UTC)
            user_results.append(
                DeliveryResult(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM,
                    status=DeliveryStatus.FAILED,
                    subscription_tier=tier,
                    recipient=None,
                    error_message=error_message,
                    delivered_at=attempt_time,
                )
            )
            user_logs.append(
                DeliveryLogCreate(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM.value,
                    status=DeliveryStatus.FAILED.value,
                    error_message=error_message,
                    delivered_at=attempt_time,
                )
            )
            continue

        try:
            await _send_with_retries(
                send_message=send_message,
                chat_id=user.telegram_id,
                message=_format_accumulator_message(slip),
            )
        except Exception as exc:
            error_message = f"Telegram send failed: {exc}"
            attempt_time = datetime.now(UTC)
            user_results.append(
                DeliveryResult(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM,
                    status=DeliveryStatus.FAILED,
                    subscription_tier=tier,
                    recipient=str(user.telegram_id),
                    error_message=error_message,
                    delivered_at=attempt_time,
                )
            )
            user_logs.append(
                DeliveryLogCreate(
                    accumulator_id=slip.slip_id,
                    user_id=user.id,
                    channel=DeliveryChannel.TELEGRAM.value,
                    status=DeliveryStatus.FAILED.value,
                    error_message=error_message,
                    delivered_at=attempt_time,
                )
            )
            continue

        delivered_at = datetime.now(UTC)
        user_results.append(
            DeliveryResult(
                accumulator_id=slip.slip_id,
                user_id=user.id,
                channel=DeliveryChannel.TELEGRAM,
                status=DeliveryStatus.SENT,
                subscription_tier=tier,
                recipient=str(user.telegram_id),
                delivered_at=delivered_at,
            )
        )
        user_logs.append(
            DeliveryLogCreate(
                accumulator_id=slip.slip_id,
                user_id=user.id,
                channel=DeliveryChannel.TELEGRAM.value,
                status=DeliveryStatus.SENT.value,
                delivered_at=delivered_at,
            )
        )

    return user_results, user_logs


async def _deliver_to_admin_fallback(
    *,
    slips: Sequence[AccumulatorSlip],
    send_message: TelegramSender,
    cause: Exception,
) -> tuple[list[DeliveryResult], tuple[str, ...]] | None:
    """Deliver directly to configured admin IDs when DB delivery cannot run.

    Inputs:
        slips: Approved slips ready for delivery.
        send_message: Telegram sender callback used for message delivery.
        cause: Original exception that prevented DB-backed delivery.

    Outputs:
        `None` when no admin fallback recipients are configured, otherwise a
        tuple of fallback delivery results and diagnostics.
    """

    admin_ids = get_settings().telegram.admin_telegram_ids
    if not admin_ids:
        return None

    diagnostics: list[str] = [
        "Database-backed delivery failed; using TELEGRAM_ADMIN_IDS fallback recipients. "
        f"Root cause: {cause}"
    ]
    fallback_results: list[DeliveryResult] = []

    for admin_id in admin_ids:
        for slip in slips:
            try:
                await _send_with_retries(
                    send_message=send_message,
                    chat_id=admin_id,
                    message=_format_accumulator_message(slip),
                )
            except Exception as exc:
                fallback_results.append(
                    DeliveryResult(
                        accumulator_id=slip.slip_id,
                        user_id=None,
                        channel=DeliveryChannel.TELEGRAM,
                        status=DeliveryStatus.FAILED,
                        subscription_tier=SubscriptionTier.ELITE,
                        recipient=str(admin_id),
                        error_message=f"Telegram send failed: {exc}",
                        delivered_at=datetime.now(UTC),
                    )
                )
                continue

            fallback_results.append(
                DeliveryResult(
                    accumulator_id=slip.slip_id,
                    user_id=None,
                    channel=DeliveryChannel.TELEGRAM,
                    status=DeliveryStatus.SENT,
                    subscription_tier=SubscriptionTier.ELITE,
                    recipient=str(admin_id),
                    delivered_at=datetime.now(UTC),
                )
            )

    if not any(result.status is DeliveryStatus.SENT for result in fallback_results):
        diagnostics.append(
            "Admin fallback delivery attempted but all Telegram sends failed."
        )

    return fallback_results, tuple(diagnostics)


async def _send_with_retries(
    *,
    send_message: TelegramSender,
    chat_id: int,
    message: str,
) -> None:
    """Send one Telegram message with retry handling for transient errors."""

    for attempt in range(_DELIVERY_SEND_MAX_ATTEMPTS):
        try:
            await send_message(chat_id, message)
            return
        except Exception as exc:
            is_last_attempt = attempt == (_DELIVERY_SEND_MAX_ATTEMPTS - 1)
            if is_last_attempt or not _is_transient_delivery_error(exc):
                raise
            await asyncio.sleep(_DELIVERY_SEND_RETRY_BASE_SECONDS * (2**attempt))


def _is_transient_delivery_error(error: Exception) -> bool:
    """Return whether one delivery error appears transient and retryable."""

    if isinstance(error, (TimeoutError, OSError)):
        return True
    message = " ".join(str(error).split()).casefold()
    return any(
        token in message
        for token in (
            "network is unreachable",
            "temporarily unavailable",
            "timeout",
            "connection reset",
            "connection aborted",
            "connection error",
        )
    )


def _format_accumulator_message(slip: AccumulatorSlip) -> str:
    """Render a compact Telegram message for one accumulator slip."""

    header = (
        f"🎯 PUNTLAB — Slip #{slip.slip_number}\n"
        f"📅 {slip.slip_date.isoformat()} | Total Odds: {slip.total_odds:.2f} | "
        f"{slip.leg_count} Legs"
    )

    leg_lines = []
    for leg in slip.legs:
        leg_lines.append(
            f"{leg.leg_number}. {leg.home_team} vs {leg.away_team}\n"
            f"   {leg.market_label or leg.market.value}: {leg.selection} @ {leg.odds:.2f}"
        )

    rationale = ""
    if isinstance(slip, ExplainedAccumulator):
        rationale = f"\n\n🧠 Overall: {slip.rationale}"

    return (
        f"{header}\n\n"
        + "\n\n".join(leg_lines)
        + f"{rationale}\n\n⚠️ Play responsibly. You decide the final bet."
    )


@asynccontextmanager
async def _telegram_sender_context(
    sender: TelegramSender | None,
) -> AsyncIterator[TelegramSender]:
    """Yield a Telegram sender function and close owned bot resources."""

    if sender is not None:
        yield sender
        return

    bot_token = get_settings().telegram.bot_token
    if not bot_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is required for delivery_node when no telegram_sender "
            "override is supplied."
        )

    bot = Bot(token=bot_token)

    async def send(chat_id: int, message: str) -> None:
        """Send one text message to the target Telegram chat ID."""

        await bot.send_message(
            chat_id=chat_id,
            text=message,
        )

    try:
        yield send
    finally:
        await bot.session.close()


async def _default_user_fetcher(
    session: AsyncSession,
    tier: SubscriptionTier,
) -> Sequence[User]:
    """Load active users for one subscription tier from the database."""

    return await get_users_by_tier(
        session,
        tier.value,
    )


async def _default_log_writer(
    session: AsyncSession,
    logs: list[DeliveryLogCreate],
) -> None:
    """Persist delivery logs in one canonical batch write."""

    await create_delivery_logs(session, logs)


def _mark_published_explained(
    original_slips: Sequence[ExplainedAccumulator],
    approved_slips: Sequence[ExplainedAccumulator],
    *,
    published_at: datetime,
) -> list[ExplainedAccumulator]:
    """Mark approved explained slips as published while preserving others."""

    approved_by_number = {slip.slip_number for slip in approved_slips}
    return [
        slip.model_copy(
            update={
                "is_published": slip.slip_number in approved_by_number,
                "published_at": published_at
                if slip.slip_number in approved_by_number
                else None,
            }
        )
        for slip in original_slips
    ]


def _mark_published_accumulators(
    original_slips: Sequence[AccumulatorSlip],
    explained_slips: Sequence[ExplainedAccumulator],
) -> list[AccumulatorSlip]:
    """Mirror publication flags from explained slips onto stage-6 slips."""

    explained_publication = {
        slip.slip_number: (slip.is_published, slip.published_at)
        for slip in explained_slips
    }
    published: list[AccumulatorSlip] = []
    for slip in original_slips:
        is_published, published_at = explained_publication.get(
            slip.slip_number,
            (False, None),
        )
        published.append(
            slip.model_copy(
                update={
                    "is_published": is_published,
                    "published_at": published_at,
                }
            )
        )
    return published


def _merge_diagnostics(existing_errors: Sequence[str], diagnostics: Sequence[str]) -> list[str]:
    """Append new diagnostics to the pipeline error list without duplicates."""

    merged_errors = list(existing_errors)
    seen_messages = {message.casefold() for message in merged_errors}

    for diagnostic in diagnostics:
        lookup_key = diagnostic.casefold()
        if lookup_key in seen_messages:
            continue
        seen_messages.add(lookup_key)
        merged_errors.append(diagnostic)

    return merged_errors


__all__ = ["delivery_node"]
