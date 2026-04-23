"""Fixture-scoped market snapshot schemas for full betting-menu visibility.

Purpose: expose one compact, serializable view of a fixture's entire fetched
market universe without replacing the canonical odds catalog.
Scope: per-selection, per-market, per-group, and per-fixture market snapshots
derived from `OddsMarketCatalog` for prompt grounding and local inspection.
Dependencies: shared market/sport enums plus common text normalization helpers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MarketType, SportName
from src.schemas.common import normalize_optional_text, require_non_blank_text


class FixtureMarketSelection(BaseModel):
    """One selection inside a fixture market snapshot."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider_selection_name: str = Field(
        description="Original provider-facing selection label."
    )
    odds: float = Field(gt=1.0, description="Decimal odds for the selection.")
    line: float | None = Field(
        default=None,
        description="Optional numeric line carried by the selection.",
    )
    canonical_market: MarketType | None = Field(
        default=None,
        description="Current PuntLab canonical market mapping when scoreable.",
    )
    canonical_selection: str | None = Field(
        default=None,
        description="Current PuntLab canonical selection label when scoreable.",
    )
    provider_selection_id: int | str | None = Field(
        default=None,
        description="Provider-native selection identifier when supplied.",
    )
    is_scoreable: bool = Field(
        default=False,
        description="Whether PuntLab can currently score this selection.",
    )

    @field_validator("provider_selection_name", "canonical_selection")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: object) -> str | None:
        """Trim optional text fields and reject blank required selection labels."""

        field_name = getattr(info, "field_name", "value")
        if field_name == "provider_selection_name":
            if value is None:
                raise ValueError("provider_selection_name must not be blank.")
            return require_non_blank_text(value, field_name)
        return normalize_optional_text(value)


class FixtureMarketSnapshotEntry(BaseModel):
    """One grouped provider market with every fetched selection preserved."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider_market_name: str = Field(description="Original provider market name.")
    provider_market_key: str = Field(description="Stable provider-native market key.")
    market_label: str = Field(description="Display-ready market label.")
    provider_market_id: int | str | None = Field(
        default=None,
        description="Provider-native market identifier when available.",
    )
    period: str | None = Field(
        default=None,
        description="Provider-normalized period such as `match` or `first_half`.",
    )
    participant_scope: str | None = Field(
        default=None,
        description="Provider-normalized participant scope such as `match` or `team`.",
    )
    canonical_markets: tuple[MarketType, ...] = Field(
        default_factory=tuple,
        description="Unique canonical market types represented inside this provider market.",
    )
    selections: tuple[FixtureMarketSelection, ...] = Field(
        default_factory=tuple,
        description="All fetched selections attached to the market.",
    )

    @field_validator(
        "provider_market_name",
        "provider_market_key",
        "market_label",
        "period",
        "participant_scope",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info: object) -> str | None:
        """Reject blank required text and normalize optional market metadata."""

        field_name = getattr(info, "field_name", "value")
        if field_name in {"provider_market_name", "provider_market_key", "market_label"}:
            if value is None:
                raise ValueError(f"{field_name} must not be blank.")
            return require_non_blank_text(value, field_name)
        return normalize_optional_text(value)


class FixtureMarketGroupSnapshot(BaseModel):
    """One SportyBet market-group section inside a fixture snapshot."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    group_id: str = Field(description="Stable SportyBet market-group identifier.")
    group_name: str = Field(description="SportyBet market-group display label.")
    markets: tuple[FixtureMarketSnapshotEntry, ...] = Field(
        default_factory=tuple,
        description="Ordered provider markets preserved within the group.",
    )

    @field_validator("group_id", "group_name")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank market-group fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)


class FixtureMarketSnapshot(BaseModel):
    """Compact fixture-scoped view of the fetched market universe."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(description="Canonical fixture reference.")
    sport: SportName = Field(description="Sport associated with the fixture.")
    competition: str = Field(description="Competition display label.")
    home_team: str = Field(description="Home-team display name.")
    away_team: str = Field(description="Away-team display name.")
    provider: str = Field(description="Provider label that supplied the snapshot.")
    event_id: str | None = Field(
        default=None,
        description="SportyBet event identifier when available.",
    )
    game_id: str | None = Field(
        default=None,
        description="SportyBet game identifier when available.",
    )
    fetch_source: str | None = Field(
        default=None,
        description="Source path that supplied the snapshot, such as `api` or `browser`.",
    )
    total_market_size: int = Field(
        ge=0,
        description=(
            "Best available total market count for the fixture, preferring SportyBet's "
            "reported total when present and otherwise falling back to fetched markets."
        ),
    )
    reported_total_market_size: int | None = Field(
        default=None,
        ge=0,
        description="SportyBet-reported total market size when the payload exposes it.",
    )
    fetched_market_count: int = Field(
        ge=0,
        description="Number of provider markets fetched into the snapshot.",
    )
    fetched_selection_count: int = Field(
        ge=0,
        description="Number of fetched selections preserved across all markets.",
    )
    scoreable_market_count: int = Field(
        ge=0,
        description="Number of markets containing at least one currently scoreable selection.",
    )
    scoreable_selection_count: int = Field(
        ge=0,
        description="Number of currently scoreable selections in the snapshot.",
    )
    unmapped_market_count: int = Field(
        ge=0,
        description="Number of fetched markets that remain outside PuntLab's scoring taxonomy.",
    )
    market_groups: tuple[FixtureMarketGroupSnapshot, ...] = Field(
        default_factory=tuple,
        description="Ordered SportyBet market groups preserved for the fixture.",
    )

    @field_validator(
        "fixture_ref",
        "competition",
        "home_team",
        "away_team",
        "provider",
        "event_id",
        "game_id",
        "fetch_source",
    )
    @classmethod
    def validate_text_fields(cls, value: str | None, info: object) -> str | None:
        """Reject blank required text and normalize optional identifiers."""

        field_name = getattr(info, "field_name", "value")
        if field_name in {"fixture_ref", "competition", "home_team", "away_team", "provider"}:
            if value is None:
                raise ValueError(f"{field_name} must not be blank.")
            return require_non_blank_text(value, field_name)
        return normalize_optional_text(value)


__all__ = [
    "FixtureMarketGroupSnapshot",
    "FixtureMarketSelection",
    "FixtureMarketSnapshot",
    "FixtureMarketSnapshotEntry",
]
