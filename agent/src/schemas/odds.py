"""Normalized odds schemas shared across PuntLab's provider integrations.

Purpose: define the canonical ingestion contract for bookmaker markets and
selections before scoring and market resolution choose the subset PuntLab can
evaluate today.
Scope: preserve all provider markets losslessly while still exposing optional
canonical market mappings, odds values, provider labels, and market metadata.
Dependencies: shared market taxonomy from `src.config` and Pydantic validation.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import MarketType
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_finite_number,
    require_non_blank_text,
)

JSONPrimitive = str | int | float | bool | None
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
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
        A validated odds record that preserves the provider's original market
        and selection labels while exposing canonical PuntLab mappings when
        available.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(
        description="Fixture reference, usually a Sportradar ID or provider fallback key."
    )
    market: MarketType | None = Field(
        default=None,
        description="Canonical internal market taxonomy when PuntLab can map the provider market.",
    )
    selection: str = Field(
        description="Canonical selection label when mapped, otherwise the provider selection label."
    )
    odds: float = Field(gt=1.0, description="Decimal odds for the selection.")
    provider: str = Field(description="Bookmaker or odds provider label.")
    provider_market_name: str = Field(
        description="Original provider market name preserved for unmapped ingestion rows."
    )
    provider_selection_name: str = Field(
        description="Original provider selection label preserved for unmapped ingestion rows."
    )
    provider_market_key: str | None = Field(
        default=None,
        description="Stable normalized key derived from the provider market name.",
    )
    provider_selection_key: str | None = Field(
        default=None,
        description="Stable normalized key derived from the provider selection label.",
    )
    sportybet_available: bool = Field(
        default=False,
        description="Whether the same market is known to exist on SportyBet.",
    )
    market_label: str | None = Field(
        default=None,
        description="Display-oriented market label, defaulting to the provider market name.",
    )
    line: float | None = Field(
        default=None,
        description="Numeric line for spread and totals markets when applicable.",
    )
    period: str | None = Field(
        default=None,
        description="Provider-normalized period scope such as `match` or `first_half`.",
    )
    participant_scope: str | None = Field(
        default=None,
        description="Target scope such as `match`, `team`, or `player` when detectable.",
    )
    provider_market_id: int | str | None = Field(
        default=None,
        description="Provider-native market identifier for traceability.",
    )
    raw_metadata: dict[str, JSONPrimitive] = Field(
        default_factory=dict,
        description="Extra provider metadata preserved without forcing canonical mappings.",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Timezone-aware provider update timestamp when available.",
    )

    @field_validator(
        "fixture_ref",
        "selection",
        "provider",
        "provider_market_name",
        "provider_selection_name",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator(
        "market_label",
        "provider_market_key",
        "provider_selection_key",
        "period",
        "participant_scope",
    )
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

    @field_validator("raw_metadata")
    @classmethod
    def validate_raw_metadata(
        cls,
        value: dict[str, JSONPrimitive],
    ) -> dict[str, JSONPrimitive]:
        """Normalize metadata keys and reject non-finite numeric payloads."""

        normalized_metadata: dict[str, JSONPrimitive] = {}
        for raw_key, raw_value in value.items():
            key = require_non_blank_text(raw_key, "raw_metadata_key")
            if isinstance(raw_value, float):
                require_finite_number(raw_value, f"raw_metadata[{key}]")
            normalized_metadata[key] = raw_value
        return normalized_metadata

    @field_validator("last_updated")
    @classmethod
    def validate_last_updated(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware provider timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "last_updated")

    @model_validator(mode="after")
    def validate_market_shape(self) -> NormalizedOdds:
        """Enforce line requirements and derive normalized provider keys."""

        if self.market in _LINE_BASED_MARKETS and self.line is None:
            raise ValueError(f"line is required when market is `{self.market.value}`.")
        if self.provider_market_key is None:
            self.provider_market_key = self._normalize_key(self.provider_market_name)
        if self.provider_selection_key is None:
            self.provider_selection_key = self._normalize_key(self.provider_selection_name)
        if self.market_label is None:
            self.market_label = self.provider_market_name

        return self

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Convert provider labels into deterministic machine-friendly keys."""

        normalized = require_non_blank_text(value, "provider_key_source").lower()
        compacted = _NON_ALNUM_PATTERN.sub("_", normalized).strip("_")
        if not compacted:
            raise ValueError("provider-derived keys must contain at least one alphanumeric.")
        return compacted


__all__ = ["MarketType", "NormalizedOdds"]
