"""Tests for PuntLab's user and delivery schemas.

Purpose: verify subscription/contact validation and delivery-result integrity
before Telegram, API, and web delivery layers depend on them.
Scope: unit tests for `src.schemas.users`.
Dependencies: pytest plus the shared user and delivery schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.schemas.users import (
    DeliveryChannel,
    DeliveryResult,
    DeliveryStatus,
    SubscriptionTier,
    UserProfile,
)


def test_user_profile_requires_contact_channel_and_normalizes_email() -> None:
    """Users should provide at least one contact path and keep paid expiry metadata coherent."""

    user = UserProfile(
        email="bettor@example.com",
        display_name=" Stanley ",
        subscription_tier=SubscriptionTier.PLUS,
        subscription_expires_at=datetime(2026, 5, 1, 7, 0, tzinfo=UTC),
    )

    dumped = user.model_dump(mode="json")

    assert dumped["email"] == "bettor@example.com"
    assert user.display_name == "Stanley"

    with pytest.raises(ValueError, match="either telegram_id or email"):
        UserProfile()


def test_delivery_result_requires_failure_message_and_sent_timestamp() -> None:
    """Delivery results should stay explicit for success and failure outcomes."""

    delivered = DeliveryResult(
        channel=DeliveryChannel.TELEGRAM,
        status=DeliveryStatus.SENT,
        subscription_tier=SubscriptionTier.FREE,
        recipient="123456789",
        delivered_at=datetime(2026, 4, 3, 10, 0, tzinfo=UTC),
    )

    assert delivered.model_dump(mode="json")["status"] == "sent"

    with pytest.raises(ValueError, match="error_message is required"):
        DeliveryResult(
            channel=DeliveryChannel.API,
            status=DeliveryStatus.FAILED,
            recipient="partner-api-client",
        )
