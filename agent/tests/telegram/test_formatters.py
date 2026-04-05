"""Tests for PuntLab Telegram message formatter utilities.

Purpose: verify HTML-safe formatting, tier-aware messaging, and structured
output for accumulator, history, stats, and welcome message templates.
Scope: unit tests for `src.telegram.formatters`.
Dependencies: pytest plus canonical accumulator/user schema models.
"""

from __future__ import annotations

from datetime import date

import pytest
from src.config import MarketType, SportName
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorOutcome,
    AccumulatorSlip,
    AccumulatorStatus,
    ExplainedAccumulator,
    ResolutionSource,
)
from src.schemas.users import SubscriptionTier
from src.telegram.formatters import (
    format_accumulator_message,
    format_history_message,
    format_stats_message,
    format_welcome_message,
)


def _build_leg(
    *,
    leg_number: int,
    home_team: str,
    away_team: str,
    market: MarketType = MarketType.OVER_UNDER_25,
    rationale: str | None = None,
) -> AccumulatorLeg:
    """Create one canonical accumulator leg for formatter tests."""

    return AccumulatorLeg(
        leg_number=leg_number,
        fixture_ref=f"sr:match:{9600 + leg_number}",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=home_team,
        away_team=away_team,
        market=market,
        selection="Over 2.5",
        odds=1.85,
        provider="sportybet",
        confidence=0.72,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale=rationale,
    )


def _build_explained_slip() -> ExplainedAccumulator:
    """Create a representative explained accumulator for format tests."""

    legs = (
        _build_leg(
            leg_number=1,
            home_team="Arsenal",
            away_team="Chelsea",
            rationale="Both teams have scored in four straight league matches.",
        ),
        _build_leg(
            leg_number=2,
            home_team="Barcelona <B>",
            away_team="Valencia & Co",
            market=MarketType.MATCH_RESULT,
        ),
    )
    return ExplainedAccumulator(
        slip_date=date(2026, 4, 13),
        slip_number=1,
        legs=legs,
        total_odds=3.423,
        leg_count=2,
        confidence=0.74,
        rationale="Strong cross-league form alignment supports this two-leg blend.",
        status=AccumulatorStatus.APPROVED,
    )


def test_format_accumulator_message_renders_structured_html_output() -> None:
    """Accumulator formatting should include all required structural blocks."""

    slip = _build_explained_slip()
    message = format_accumulator_message(
        slip,
        tier=SubscriptionTier.PLUS,
    )

    assert "🎯 <b>PUNTLAB — Slip #1</b>" in message
    assert "🔐 Tier: <b>🔵 Plus</b>" in message
    assert "1️⃣ <b>Arsenal vs Chelsea</b>" in message
    assert "📊 Over/Under Goals @ 1.85" in message
    assert "💡 Both teams have scored in four straight league matches." in message
    assert "🧠 <b>Overall:</b>" in message
    assert "⚠️ Play responsibly." in message
    assert "Barcelona &lt;B&gt; vs Valencia &amp; Co" in message


def test_format_accumulator_message_can_skip_disclaimer() -> None:
    """Formatter should omit the disclaimer when explicitly disabled."""

    message = format_accumulator_message(
        _build_explained_slip(),
        include_disclaimer=False,
    )
    assert "⚠️ Play responsibly." not in message


def test_format_history_message_handles_empty_and_limited_lists() -> None:
    """History formatter should support empty state and output truncation."""

    empty_message = format_history_message([], tier=SubscriptionTier.FREE)
    assert "🗂️ <b>PUNTLAB History</b>" in empty_message
    assert "No recommendation history is available yet." in empty_message

    slip_one = AccumulatorSlip(
        slip_date=date(2026, 4, 10),
        slip_number=1,
        legs=(_build_leg(leg_number=1, home_team="A", away_team="B"),),
        total_odds=1.85,
        leg_count=1,
        confidence=0.7,
        status=AccumulatorStatus.SETTLED,
        outcome=AccumulatorOutcome.WON,
    )
    slip_two = AccumulatorSlip(
        slip_date=date(2026, 4, 11),
        slip_number=2,
        legs=(_build_leg(leg_number=1, home_team="C", away_team="D"),),
        total_odds=1.85,
        leg_count=1,
        confidence=0.68,
        status=AccumulatorStatus.SETTLED,
        outcome=AccumulatorOutcome.LOST,
    )

    message = format_history_message([slip_one, slip_two], limit=1)
    assert "Slip #2" in message
    assert "Slip #1" not in message
    assert "… showing 1 of 2 entries." in message


def test_format_history_message_rejects_non_positive_limits() -> None:
    """History formatter should fail fast for invalid limits."""

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        format_history_message([], limit=0)


def test_format_stats_message_formats_known_unknown_and_empty_metrics() -> None:
    """Stats formatter should handle common numeric and custom metric keys."""

    stats_message = format_stats_message(
        {
            "leg_hit_rate": 0.63,
            "roi_percent": 12.4,
            "resolver_success_rate": 0.91,
            "custom_signal": "healthy",
        },
        tier=SubscriptionTier.ELITE,
        as_of=date(2026, 4, 13),
    )

    assert "📈 <b>PUNTLAB Stats</b>" in stats_message
    assert "As of: 2026-04-13" in stats_message
    assert "🔐 Tier: <b>🟡 Elite</b>" in stats_message
    assert "Leg Hit Rate:</b> 63.0%" in stats_message
    assert "Roi Percent:</b> 12.4%" in stats_message
    assert "Custom Signal:</b> healthy" in stats_message

    empty_message = format_stats_message({})
    assert "Metrics are not available yet." in empty_message


def test_format_welcome_message_is_tier_aware_and_html_safe() -> None:
    """Welcome formatter should render tier context and escape user names."""

    welcome_message = format_welcome_message(
        display_name="Stanley <Admin>",
        subscription_tier=SubscriptionTier.FREE,
        is_registered=True,
    )
    assert "👋 <b>Welcome back, Stanley &lt;Admin&gt;.</b>" in welcome_message
    assert "🔐 Tier: <b>🟢 Free</b> (1 daily accumulator)" in welcome_message
    assert "/today" in welcome_message


def test_format_welcome_message_supports_new_user_copy() -> None:
    """Welcome formatter should render first-time copy when requested."""

    welcome_message = format_welcome_message(
        display_name=None,
        subscription_tier=SubscriptionTier.PLUS,
        is_registered=False,
    )
    assert "👋 <b>Welcome, there.</b>" in welcome_message
    assert "🔐 Tier: <b>🔵 Plus</b>" in welcome_message
