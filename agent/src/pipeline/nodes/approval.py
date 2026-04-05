"""Approval node for PuntLab's LangGraph pipeline.

Purpose: execute the soft-gate approval stage that marks slips pending,
notifies admins, waits for review time, and auto-approves non-blocked slips.
Scope: deterministic approval state transitions for explained accumulators and
synchronized accumulator status updates for downstream delivery.
Dependencies: `src.pipeline.state` for stage enums/state validation,
`src.schemas.accumulators` for accumulator lifecycle statuses, and optional
aiogram-based Telegram notification when bot settings are configured.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from aiogram import Bot

from src.config import get_settings
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorSlip, AccumulatorStatus, ExplainedAccumulator

DEFAULT_APPROVAL_WAIT_SECONDS = 30 * 60
type SleepCallable = Callable[[float], Awaitable[None]]
type AdminNotifier = Callable[[str, PipelineState], Awaitable[None]]
type BlockedIdsResolver = Callable[[PipelineState], Awaitable[Sequence[str]]]


async def approval_node(
    state: PipelineState | Mapping[str, Any],
    *,
    wait_seconds: int = DEFAULT_APPROVAL_WAIT_SECONDS,
    sleep: SleepCallable = asyncio.sleep,
    admin_notifier: AdminNotifier | None = None,
    blocked_ids_resolver: BlockedIdsResolver | None = None,
) -> dict[str, object]:
    """Execute the approval stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state or a raw mapping that can be validated
            into `PipelineState`.
        wait_seconds: Soft-gate review window in seconds before auto-approval.
        sleep: Async sleeper override used by tests to avoid real delays.
        admin_notifier: Optional notifier callback for admin alerts. When not
            provided, this node tries a Telegram notifier from settings.
        blocked_ids_resolver: Optional callback used after the wait window to
            pull the latest blocked IDs from an external control surface.

    Outputs:
        A partial LangGraph update containing approval status, blocked IDs,
        accumulator status updates, merged diagnostics, and the next stage
        marker.
    """

    if wait_seconds < 0:
        raise ValueError("wait_seconds must be greater than or equal to zero.")

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    diagnostics: list[str] = []

    pending_explained = _mark_pending(validated_state.explained_accumulators)
    pending_accumulators = _mark_pending(validated_state.accumulators)

    if not pending_explained:
        diagnostics.append(
            "Approval skipped for run "
            f"{validated_state.run_id}: no explained accumulators were available."
        )
        return {
            "current_stage": PipelineStage.APPROVAL,
            "approval_status": ApprovalStatus.BLOCKED,
            "blocked_ids": list(validated_state.blocked_ids),
            "accumulators": pending_accumulators,
            "explained_accumulators": pending_explained,
            "errors": _merge_diagnostics(validated_state.errors, diagnostics),
        }

    resolved_admin_notifier, notifier_diagnostic = _resolve_admin_notifier(admin_notifier)
    if notifier_diagnostic is not None:
        diagnostics.append(notifier_diagnostic)

    if resolved_admin_notifier is not None:
        notification_message = _build_admin_notification_message(
            run_id=validated_state.run_id,
            run_date=validated_state.run_date.isoformat(),
            slip_count=len(pending_explained),
            wait_seconds=wait_seconds,
        )
        try:
            await resolved_admin_notifier(notification_message, validated_state)
        except Exception as exc:
            diagnostics.append(
                f"Approval admin notification failed for run {validated_state.run_id}: {exc}"
            )

    if wait_seconds > 0:
        await sleep(float(wait_seconds))

    merged_blocked_ids = list(validated_state.blocked_ids)
    if blocked_ids_resolver is not None:
        try:
            merged_blocked_ids.extend(await blocked_ids_resolver(validated_state))
        except Exception as exc:
            diagnostics.append(
                f"Blocked-ID refresh failed for run {validated_state.run_id}: {exc}"
            )

    normalized_blocked_ids, blocked_id_diagnostics = _normalize_blocked_ids(merged_blocked_ids)
    diagnostics.extend(blocked_id_diagnostics)
    blocked_lookup = {blocked_id.casefold() for blocked_id in normalized_blocked_ids}

    approved_count = 0
    blocked_count = 0
    explained_status_index: dict[int, AccumulatorStatus] = {}
    resolved_explained: list[ExplainedAccumulator] = []

    for explained_slip in pending_explained:
        slip_status = _resolve_slip_status(explained_slip, blocked_lookup)
        resolved_explained.append(_apply_status(explained_slip, slip_status))
        explained_status_index[explained_slip.slip_number] = slip_status
        if slip_status is AccumulatorStatus.BLOCKED:
            blocked_count += 1
        else:
            approved_count += 1

    resolved_accumulators = _sync_accumulator_statuses(
        pending_accumulators,
        explained_status_index,
    )

    if approved_count > 0:
        approval_status = ApprovalStatus.APPROVED
        current_stage = PipelineStage.DELIVERY
    else:
        approval_status = ApprovalStatus.BLOCKED
        current_stage = PipelineStage.APPROVAL

    diagnostics.append(
        "Approval stage completed for run "
        f"{validated_state.run_id}: approved={approved_count}, blocked={blocked_count}."
    )

    return {
        "current_stage": current_stage,
        "approval_status": approval_status,
        "blocked_ids": normalized_blocked_ids,
        "accumulators": resolved_accumulators,
        "explained_accumulators": resolved_explained,
        "errors": _merge_diagnostics(validated_state.errors, diagnostics),
    }


def _build_admin_notification_message(
    *,
    run_id: str,
    run_date: str,
    slip_count: int,
    wait_seconds: int,
) -> str:
    """Build the canonical approval-notification message for admins."""

    wait_minutes = max(1, round(wait_seconds / 60))
    return (
        "PuntLab approval gate: "
        f"{slip_count} accumulators are pending review for {run_date} (run {run_id}). "
        f"Auto-approval will run in {wait_minutes} minute(s) for any unblocked slips."
    )


def _resolve_admin_notifier(
    notifier: AdminNotifier | None,
) -> tuple[AdminNotifier | None, str | None]:
    """Resolve the admin notifier callback and return a setup diagnostic."""

    if notifier is not None:
        return notifier, None

    settings = get_settings()
    telegram_config = settings.telegram
    if not telegram_config.bot_token:
        return (
            None,
            "Approval admin notification skipped because TELEGRAM_BOT_TOKEN is not configured.",
        )
    if not telegram_config.admin_telegram_ids:
        return (
            None,
            "Approval admin notification skipped because ADMIN_TELEGRAM_IDS is empty.",
        )

    token = telegram_config.bot_token
    admin_ids = telegram_config.admin_telegram_ids

    async def notify(message: str, _: PipelineState) -> None:
        """Send one approval message to every configured admin Telegram ID."""

        bot = Bot(token=token)
        try:
            send_failures: list[str] = []
            for admin_id in admin_ids:
                try:
                    await bot.send_message(chat_id=admin_id, text=message)
                except Exception as exc:
                    send_failures.append(f"{admin_id}: {exc}")

            if send_failures:
                raise RuntimeError("; ".join(send_failures))
        finally:
            await bot.session.close()

    return notify, None


def _normalize_blocked_ids(blocked_ids: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """Trim, deduplicate, and validate blocked identifiers for matching.

    Inputs:
        blocked_ids: Raw blocked IDs gathered from state and optional resolver.

    Outputs:
        A tuple of `(normalized_ids, diagnostics)` where invalid entries are
        dropped and recorded in diagnostics.
    """

    normalized_ids: list[str] = []
    diagnostics: list[str] = []
    seen_lookup: set[str] = set()

    for index, blocked_id in enumerate(blocked_ids):
        if not isinstance(blocked_id, str):
            diagnostics.append(
                f"Ignored blocked identifier at position {index} because it is not a string."
            )
            continue

        normalized = blocked_id.strip()
        if not normalized:
            diagnostics.append(
                f"Ignored blocked identifier at position {index} because it is blank."
            )
            continue

        lookup = normalized.casefold()
        if lookup in seen_lookup:
            continue
        seen_lookup.add(lookup)
        normalized_ids.append(normalized)

    return normalized_ids, tuple(diagnostics)


def _mark_pending[T: AccumulatorSlip](slips: Sequence[T]) -> list[T]:
    """Reset slips to pending status before the soft-gate wait window."""

    return [slip.model_copy(update={"status": AccumulatorStatus.PENDING}) for slip in slips]


def _resolve_slip_status(
    slip: AccumulatorSlip,
    blocked_lookup: set[str],
) -> AccumulatorStatus:
    """Return the final approval status for one slip based on blocked IDs."""

    slip_keys = _build_slip_block_keys(slip)
    if slip_keys.intersection(blocked_lookup):
        return AccumulatorStatus.BLOCKED
    return AccumulatorStatus.APPROVED


def _build_slip_block_keys(slip: AccumulatorSlip) -> set[str]:
    """Build the canonical blocked-ID lookup keys for one accumulator."""

    keys = {str(slip.slip_number), f"slip:{slip.slip_number}", f"slip-{slip.slip_number}"}
    if slip.slip_id is not None:
        keys.add(str(slip.slip_id))
    keys.update(leg.fixture_ref for leg in slip.legs)
    return {key.casefold() for key in keys}


def _apply_status[T: AccumulatorSlip](slip: T, status: AccumulatorStatus) -> T:
    """Apply approval status while enforcing pre-delivery publication invariants."""

    return slip.model_copy(
        update={
            "status": status,
            "is_published": False,
            "published_at": None,
        }
    )


def _sync_accumulator_statuses(
    accumulators: Sequence[AccumulatorSlip],
    explained_status_index: Mapping[int, AccumulatorStatus],
) -> list[AccumulatorSlip]:
    """Mirror explained-slip statuses onto the stage-6 accumulator collection."""

    synchronized: list[AccumulatorSlip] = []
    for accumulator in accumulators:
        status = explained_status_index.get(accumulator.slip_number, accumulator.status)
        synchronized.append(_apply_status(accumulator, status))
    return synchronized


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


__all__ = ["approval_node"]
