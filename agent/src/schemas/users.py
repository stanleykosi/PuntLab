"""User and delivery schemas shared across PuntLab's client surfaces.

Purpose: define canonical subscription, user-profile, and delivery-result
contracts used by Telegram, the REST API, and downstream delivery workflows.
Scope: user identity, subscription state, and per-channel delivery outcomes.
Dependencies: common validation helpers from `src.schemas.common` and Pydantic
for structured serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
)


class SubscriptionTier(StrEnum):
    """Supported subscription tiers exposed to PuntLab users."""

    FREE = "free"
    PLUS = "plus"
    ELITE = "elite"


class SubscriptionStatus(StrEnum):
    """Canonical subscription lifecycle states for a user."""

    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DeliveryChannel(StrEnum):
    """Delivery surfaces supported by the canonical V1 implementation."""

    TELEGRAM = "telegram"
    WEB = "web"
    API = "api"


class DeliveryStatus(StrEnum):
    """Per-channel delivery outcomes recorded by the pipeline."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


def _validate_email(value: str | None) -> str | None:
    """Validate a lightweight email shape without introducing extra packages.

    Args:
        value: Raw email string or `None`.

    Returns:
        A normalized email string or `None`.

    Raises:
        ValueError: If the normalized value does not resemble an email address.
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    local_part, separator, domain = normalized.partition("@")
    if not separator or not local_part or "." not in domain:
        raise ValueError("email must be a valid address.")
    return normalized


class UserProfile(BaseModel):
    """Canonical user record shared across Telegram, web, and API surfaces.

    Inputs:
        Database or auth-layer user state, including subscription metadata.

    Outputs:
        A validated user profile that delivery and access-control layers can
        consume without re-validating contact channels or subscription state.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: UUID | None = Field(
        default=None,
        description="Database identifier for the user when already persisted.",
    )
    telegram_id: int | None = Field(
        default=None,
        gt=0,
        description="Telegram user identifier when the user is linked to Telegram.",
    )
    telegram_username: str | None = Field(
        default=None,
        description="Telegram username without validation of the leading `@`.",
    )
    email: str | None = Field(
        default=None,
        description="Primary email address when the user has web access.",
    )
    display_name: str | None = Field(
        default=None,
        description="Preferred user-facing display name.",
    )
    subscription_tier: SubscriptionTier = Field(
        default=SubscriptionTier.FREE,
        description="Current plan tier used for delivery and entitlement checks.",
    )
    subscription_status: SubscriptionStatus = Field(
        default=SubscriptionStatus.ACTIVE,
        description="Current lifecycle status of the user's subscription.",
    )
    subscription_expires_at: datetime | None = Field(
        default=None,
        description="Timezone-aware expiry timestamp for paid access when present.",
    )
    paystack_customer_id: str | None = Field(
        default=None,
        description="Paystack customer identifier when billing is linked.",
    )
    is_admin: bool = Field(
        default=False,
        description="Whether the user can access admin-only controls.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Timezone-aware creation timestamp when loaded from persistence.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timezone-aware update timestamp when loaded from persistence.",
    )

    @field_validator("telegram_username", "display_name", "paystack_customer_id")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional user text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        """Apply lightweight email-shape validation without extra dependencies."""

        return _validate_email(value)

    @field_validator("subscription_expires_at", "created_at", "updated_at")
    @classmethod
    def validate_datetimes(cls, value: datetime | None, info: object) -> datetime | None:
        """Require timezone-aware timestamps for persisted user metadata."""

        if value is None:
            return None
        field_name = getattr(info, "field_name", "value")
        return ensure_timezone_aware(value, field_name)

    @model_validator(mode="after")
    def validate_contact_and_subscription_state(self) -> Self:
        """Require at least one contact channel and coherent subscription metadata."""

        if self.telegram_id is None and self.email is None:
            raise ValueError("A user must include either telegram_id or email.")
        if (
            self.subscription_status is SubscriptionStatus.ACTIVE
            and self.subscription_tier is SubscriptionTier.FREE
            and self.subscription_expires_at is not None
        ):
            raise ValueError("Free-tier users must not carry a subscription_expires_at timestamp.")
        return self


class DeliveryResult(BaseModel):
    """Per-user delivery outcome for one accumulator recommendation.

    Inputs:
        Delivery attempts emitted by Telegram, web, or API broadcasting layers.

    Outputs:
        A normalized status object suitable for pipeline-state tracking and
        persistence into the delivery log table.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    accumulator_id: UUID | None = Field(
        default=None,
        description="Accumulator identifier when the delivery target was persisted.",
    )
    user_id: UUID | None = Field(
        default=None,
        description="User identifier when the delivery target maps to a known user.",
    )
    channel: DeliveryChannel = Field(description="Surface used for the delivery attempt.")
    status: DeliveryStatus = Field(description="Outcome of the delivery attempt.")
    subscription_tier: SubscriptionTier | None = Field(
        default=None,
        description="Entitlement tier used when selecting the delivered content.",
    )
    recipient: str | None = Field(
        default=None,
        description="Human-readable delivery target such as a Telegram ID or email.",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure detail captured when delivery does not succeed.",
    )
    delivered_at: datetime | None = Field(
        default=None,
        description="Timezone-aware timestamp for when the delivery attempt completed.",
    )

    @field_validator("recipient", "error_message")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional delivery text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("delivered_at")
    @classmethod
    def validate_delivered_at(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware delivery timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "delivered_at")

    @model_validator(mode="after")
    def validate_delivery_shape(self) -> Self:
        """Keep success and failure records explicit and fail-fast."""

        if self.user_id is None and self.recipient is None:
            raise ValueError("Delivery results must include either user_id or recipient.")
        if self.status is DeliveryStatus.FAILED and self.error_message is None:
            raise ValueError("error_message is required when delivery status is failed.")
        if self.status is DeliveryStatus.SENT and self.delivered_at is None:
            raise ValueError("delivered_at is required when delivery status is sent.")
        return self


__all__ = [
    "DeliveryChannel",
    "DeliveryResult",
    "DeliveryStatus",
    "SubscriptionStatus",
    "SubscriptionTier",
    "UserProfile",
]
