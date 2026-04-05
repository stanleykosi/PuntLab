"""Telegram message formatters for PuntLab delivery and command responses.

Purpose: render canonical HTML-safe Telegram messages for accumulator slips,
history summaries, stats summaries, and welcome prompts.
Scope: deterministic formatting utilities that support tier-aware copy and
emoji-rich layouts aligned with the product specification.
Dependencies: accumulator/user schemas and standard-library HTML escaping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from html import escape

from src.config import MarketType, SportName
from src.schemas.accumulators import (
    AccumulatorOutcome,
    AccumulatorSlip,
    AccumulatorStatus,
    ExplainedAccumulator,
)
from src.schemas.users import SubscriptionTier

_TIER_BADGES: dict[SubscriptionTier, str] = {
    SubscriptionTier.FREE: "🟢 Free",
    SubscriptionTier.PLUS: "🔵 Plus",
    SubscriptionTier.ELITE: "🟡 Elite",
}

_TIER_CAPABILITIES: dict[SubscriptionTier, str] = {
    SubscriptionTier.FREE: "1 daily accumulator",
    SubscriptionTier.PLUS: "up to 10 daily accumulators",
    SubscriptionTier.ELITE: "full analyzed slate access",
}

_SPORT_EMOJIS: dict[SportName, str] = {
    SportName.SOCCER: "⚽",
    SportName.BASKETBALL: "🏀",
}

_OUTCOME_BADGES: dict[AccumulatorOutcome, str] = {
    AccumulatorOutcome.WON: "✅ WON",
    AccumulatorOutcome.LOST: "❌ LOST",
    AccumulatorOutcome.PARTIAL: "🟨 PARTIAL",
    AccumulatorOutcome.VOID: "⚪ VOID",
}

_STATUS_BADGES: dict[AccumulatorStatus, str] = {
    AccumulatorStatus.PENDING: "⏳ PENDING",
    AccumulatorStatus.APPROVED: "✅ APPROVED",
    AccumulatorStatus.BLOCKED: "🚫 BLOCKED",
    AccumulatorStatus.SETTLED: "📌 SETTLED",
}


def format_accumulator_message(
    slip: AccumulatorSlip,
    *,
    tier: SubscriptionTier | None = None,
    include_disclaimer: bool = True,
) -> str:
    """Render a full accumulator-slip Telegram message in HTML-safe format.

    Inputs:
        slip: Canonical accumulator slip to render.
        tier: Optional subscription tier badge to display in the header.
        include_disclaimer: Whether to append responsible-play guidance.

    Outputs:
        Formatted message string ready for Telegram `parse_mode="HTML"`.
    """

    lines = [
        f"🎯 <b>PUNTLAB — Slip #{slip.slip_number}</b>",
        (
            f"📅 {slip.slip_date.isoformat()} | "
            f"Total Odds: {slip.total_odds:.2f} | {slip.leg_count} Legs"
        ),
    ]
    if tier is not None:
        lines.append(f"🔐 Tier: <b>{_escape_text(_TIER_BADGES[tier])}</b>")

    for leg in slip.legs:
        sport_emoji = _SPORT_EMOJIS.get(leg.sport, "🎯")
        market_label = _format_market_label(leg.market, leg.market_label, leg.line)
        lines.extend(
            (
                "",
                f"{_leg_prefix(leg.leg_number)} <b>{_escape_text(leg.fixture_label())}</b>",
                f"   {sport_emoji} {_escape_text(leg.competition)}",
                (
                    "   📊 "
                    f"{_escape_text(market_label)} @ {leg.odds:.2f}"
                ),
            )
        )
        if leg.rationale:
            lines.append(f"   💡 {_escape_text(leg.rationale)}")

    if isinstance(slip, ExplainedAccumulator):
        lines.extend(
            (
                "",
                f"🧠 <b>Overall:</b> {_escape_text(slip.rationale)}",
            )
        )

    if include_disclaimer:
        lines.extend(
            (
                "",
                "⚠️ Play responsibly. You decide the final bet.",
            )
        )

    return "\n".join(lines)


def format_history_message(
    slips: Sequence[AccumulatorSlip],
    *,
    tier: SubscriptionTier | None = None,
    limit: int = 10,
) -> str:
    """Render recent slip-history summary for Telegram delivery.

    Inputs:
        slips: Slip history rows, usually settled or previously published.
        tier: Optional subscription tier badge for entitlement context.
        limit: Maximum number of history rows to render.

    Outputs:
        Multi-line HTML-safe history summary.

    Raises:
        ValueError: If `limit` is not a positive integer.
    """

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    lines = ["🗂️ <b>PUNTLAB History</b>"]
    if tier is not None:
        lines.append(f"🔐 Tier: <b>{_escape_text(_TIER_BADGES[tier])}</b>")

    if not slips:
        lines.append("No recommendation history is available yet.")
        return "\n".join(lines)

    ordered_slips = sorted(
        slips,
        key=lambda slip: (slip.slip_date, slip.slip_number),
        reverse=True,
    )
    rendered_slips = ordered_slips[:limit]

    for slip in rendered_slips:
        badge = _outcome_or_status_badge(slip)
        lines.append(
            f"• {slip.slip_date.isoformat()} | Slip #{slip.slip_number} | "
            f"{badge} | Odds {slip.total_odds:.2f} | {slip.leg_count} legs"
        )

    if len(ordered_slips) > limit:
        lines.append(f"… showing {limit} of {len(ordered_slips)} entries.")

    return "\n".join(lines)


def format_stats_message(
    metrics: Mapping[str, int | float | str],
    *,
    tier: SubscriptionTier | None = None,
    as_of: date | None = None,
) -> str:
    """Render a Telegram-friendly stats summary in HTML-safe format.

    Inputs:
        metrics: Arbitrary metric mapping from the stats pipeline/API layer.
        tier: Optional subscription tier badge for entitlement context.
        as_of: Optional metrics-as-of date annotation.

    Outputs:
        Multi-line HTML-safe statistics summary.
    """

    lines = ["📈 <b>PUNTLAB Stats</b>"]
    if as_of is not None:
        lines.append(f"As of: {as_of.isoformat()}")
    if tier is not None:
        lines.append(f"🔐 Tier: <b>{_escape_text(_TIER_BADGES[tier])}</b>")

    if not metrics:
        lines.append("Metrics are not available yet.")
        return "\n".join(lines)

    preferred_order = (
        "leg_hit_rate",
        "accumulator_hit_rate",
        "roi_percent",
        "resolver_success_rate",
        "total_legs",
        "total_accumulators",
    )

    rendered_keys: set[str] = set()
    for key in preferred_order:
        if key not in metrics:
            continue
        rendered_keys.add(key)
        lines.append(_format_metric_line(key, metrics[key]))

    for key in sorted(metrics.keys()):
        if key in rendered_keys:
            continue
        lines.append(_format_metric_line(key, metrics[key]))

    return "\n".join(lines)


def format_welcome_message(
    *,
    display_name: str | None,
    subscription_tier: SubscriptionTier,
    is_registered: bool = True,
) -> str:
    """Render a tier-aware Telegram welcome message.

    Inputs:
        display_name: Optional user display name.
        subscription_tier: User's active subscription tier.
        is_registered: Whether this is a returning registered user.

    Outputs:
        HTML-safe welcome message that guides next commands.
    """

    normalized_name = (display_name or "").strip() or "there"
    greeting = "Welcome back" if is_registered else "Welcome"
    tier_badge = _TIER_BADGES[subscription_tier]
    tier_capability = _TIER_CAPABILITIES[subscription_tier]

    return (
        f"👋 <b>{greeting}, {_escape_text(normalized_name)}.</b>\n"
        f"🔐 Tier: <b>{_escape_text(tier_badge)}</b> ({_escape_text(tier_capability)})\n\n"
        "Use /today for live recommendations, /history for past slips, "
        "and /help for the full command guide."
    )


def _format_market_label(
    market: MarketType,
    market_label: str | None,
    line: float | None,
) -> str:
    """Build a concise market display label for one accumulator leg."""

    base_label = market_label or _normalize_market_name(market)
    selection_suffix = ""
    if line is not None:
        selection_suffix = f" (line {line:+.1f})"
    return f"{base_label}{selection_suffix}"


def _normalize_market_name(market: MarketType) -> str:
    """Convert canonical market enum values into readable labels."""

    normalized = market.value.replace("_", " ")
    if market is MarketType.MATCH_RESULT:
        return "Match Result (1X2)"
    if market is MarketType.BTTS:
        return "Both Teams To Score"
    if market.value.startswith("over_under_"):
        return "Over/Under Goals"
    return normalized.title()


def _format_metric_line(metric_key: str, metric_value: int | float | str) -> str:
    """Render one metric key/value pair into Telegram-friendly copy."""

    display_key = metric_key.replace("_", " ").title()
    if isinstance(metric_value, str):
        return f"• <b>{_escape_text(display_key)}:</b> {_escape_text(metric_value)}"

    if isinstance(metric_value, int):
        return f"• <b>{_escape_text(display_key)}:</b> {metric_value}"

    numeric_value = float(metric_value)
    if "rate" in metric_key or metric_key.endswith("_percent"):
        if 0.0 <= numeric_value <= 1.0:
            return f"• <b>{_escape_text(display_key)}:</b> {numeric_value * 100:.1f}%"
        return f"• <b>{_escape_text(display_key)}:</b> {numeric_value:.1f}%"
    return f"• <b>{_escape_text(display_key)}:</b> {numeric_value:.2f}"


def _outcome_or_status_badge(slip: AccumulatorSlip) -> str:
    """Resolve the best available badge label for a history row."""

    if slip.outcome is not None:
        return _OUTCOME_BADGES[slip.outcome]
    return _STATUS_BADGES[slip.status]


def _leg_prefix(leg_number: int) -> str:
    """Return keycap prefixes for legs 1-9 and numeric prefixes otherwise."""

    keycaps = {
        1: "1️⃣",
        2: "2️⃣",
        3: "3️⃣",
        4: "4️⃣",
        5: "5️⃣",
        6: "6️⃣",
        7: "7️⃣",
        8: "8️⃣",
        9: "9️⃣",
    }
    return keycaps.get(leg_number, f"{leg_number}.")


def _escape_text(value: str) -> str:
    """Escape dynamic text for Telegram HTML parse mode safety."""

    return escape(value, quote=True)


__all__ = [
    "format_accumulator_message",
    "format_history_message",
    "format_stats_message",
    "format_welcome_message",
]
