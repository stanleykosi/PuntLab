"""Normalized odds schemas shared across PuntLab's provider integrations.

Purpose: define the canonical contract for normalized bookmaker markets and
selections before scoring and market resolution consume them.
Scope: market identity, selection labeling, odds values, and SportyBet
availability metadata.
Dependencies: shared market taxonomy from `src.config` and Pydantic validation.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import MarketType
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_finite_number,
    require_non_blank_text,
)

_LINE_BASED_MARKETS = {
    MarketType.ASIAN_HANDICAP,
    MarketType.POINT_SPREAD,
    MarketType.TOTAL_POINTS,
}


class NormalizedOdds(BaseModel):
    """Canonical normalized market row for one fixture selection.

    Inputs:
        Raw bookmaker market data normalized by provider adapters.

    Outputs:
        A validated odds record with canonical market taxonomy and consistent
        selection-level metadata used by scoring and resolution.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(
        description="Fixture reference, usually a Sportradar ID or provider fallback key."
    )
    market: MarketType = Field(description="Canonical internal market taxonomy.")
    selection: str = Field(description="Human-readable selection or side label.")
    odds: float = Field(gt=1.0, description="Decimal odds for the selection.")
    provider: str = Field(description="Bookmaker or odds provider label.")
    sportybet_available: bool = Field(
        default=False,
        description="Whether the same market is known to exist on SportyBet.",
    )
    market_label: str | None = Field(
        default=None,
        description="Provider-facing market label when supplied.",
    )
    line: float | None = Field(
        default=None,
        description="Numeric line for spread and totals markets when applicable.",
    )
    provider_market_id: int | str | None = Field(
        default=None,
        description="Provider-native market identifier for traceability.",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Timezone-aware provider update timestamp when available.",
    )

    @field_validator("fixture_ref", "selection", "provider")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("market_label")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional display text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("odds")
    @classmethod
    def validate_odds_value(cls, value: float) -> float:
        """Reject NaN and infinite decimal odds values."""

        return require_finite_number(value, "odds")

    @field_validator("line")
    @classmethod
    def validate_line_value(cls, value: float | None) -> float | None:
        """Require finite line values when a provider supplies one."""

        if value is None:
            return None
        return require_finite_number(value, "line")

    @field_validator("last_updated")
    @classmethod
    def validate_last_updated(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware provider timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "last_updated")

    @model_validator(mode="after")
    def validate_market_shape(self) -> NormalizedOdds:
        """Enforce line requirements for handicap and totals markets."""

        if self.market in _LINE_BASED_MARKETS and self.line is None:
            raise ValueError(f"line is required when market is `{self.market.value}`.")

        return self


__all__ = ["MarketType", "NormalizedOdds"]
