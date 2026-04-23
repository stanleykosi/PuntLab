"""Accumulator and market-resolution schemas shared across PuntLab stages.

Purpose: define the canonical resolved-market, accumulator-leg, and slip
contracts used between market resolution, accumulator building, explanation,
and delivery layers.
Scope: selected bookmaker markets, accumulator integrity validation, and
public-facing slip metadata.
Dependencies: shared market and sport enums from `src.config`, normalized odds
contracts from `src.schemas.odds`, and common validation helpers.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import date, datetime
from enum import StrEnum
from math import isclose, prod
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from src.config import MarketType, SportName
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_finite_number,
    require_non_blank_text,
)
from src.schemas.odds import NormalizedOdds

_LINE_BASED_MARKETS = {
    MarketType.ASIAN_HANDICAP,
    MarketType.POINT_SPREAD,
    MarketType.TOTAL_POINTS,
}


class ResolutionSource(StrEnum):
    """Canonical sources that can supply a resolved market."""

    SPORTYBET_API = "sportybet_api"
    SPORTYBET_BROWSER = "sportybet_browser"


class AccumulatorStrategy(StrEnum):
    """Accumulator-construction strategies described in the technical spec."""

    CONFIDENT = "confident"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class AccumulatorStatus(StrEnum):
    """Lifecycle states for an accumulator slip."""

    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"
    SETTLED = "settled"


class AccumulatorOutcome(StrEnum):
    """Supported accumulator outcomes once settlement completes."""

    WON = "won"
    LOST = "lost"
    PARTIAL = "partial"
    VOID = "void"


class LegOutcome(StrEnum):
    """Supported settlement outcomes for one accumulator leg."""

    WON = "won"
    LOST = "lost"
    VOID = "void"
    PENDING = "pending"


class ResolvedMarket(NormalizedOdds):
    """Best available market selected for a fixture after resolution.

    Inputs:
        A normalized odds row chosen from the canonical SportyBet sources.

    Outputs:
        A resolved market enriched with resolution metadata for downstream leg
        assembly and observability.
    """

    market: str = Field(
        description="Provider-native market key selected by the resolver for downstream use."
    )
    canonical_market: MarketType | None = Field(
        default=None,
        description="Canonical market hint when the resolved market maps into PuntLab taxonomy.",
    )
    resolution_source: ResolutionSource = Field(
        description="Resolver path that produced this market."
    )
    sport: SportName = Field(description="Sport associated with the resolved fixture.")
    competition: str = Field(description="Competition display name for the fixture.")
    home_team: str = Field(description="Home team display name.")
    away_team: str = Field(description="Away team display name.")
    sportybet_market_id: int | None = Field(
        default=None,
        description="SportyBet market identifier when the result came from SportyBet.",
    )
    sportybet_url: HttpUrl | None = Field(
        default=None,
        description="Public SportyBet fixture URL when available.",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="Timezone-aware timestamp for when resolution completed.",
    )

    @field_validator("market", "competition", "home_team", "away_team")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank fixture display fields after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("resolved_at")
    @classmethod
    def validate_resolved_at(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware resolution timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "resolved_at")

    @model_validator(mode="after")
    def validate_fixture_identity(self) -> Self:
        """Reject impossible resolved-market fixture metadata."""

        if self.canonical_market is None:
            with suppress(ValueError):
                self.canonical_market = MarketType(self.market)
        if self.home_team.casefold() == self.away_team.casefold():
            raise ValueError("home_team and away_team must describe different teams.")
        return self


class AccumulatorLeg(BaseModel):
    """One resolved market selection included in an accumulator slip.

    Inputs:
        A ranked match and resolved market chosen by the builder.

    Outputs:
        A validated leg contract suitable for explanation, persistence, and
        delivery formatting.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    leg_number: int = Field(gt=0, description="1-based position of the leg in the slip.")
    fixture_ref: str = Field(description="Canonical reference of the selected fixture.")
    sport: SportName = Field(description="Sport associated with this leg.")
    competition: str = Field(description="Competition display name.")
    home_team: str = Field(description="Home team display name.")
    away_team: str = Field(description="Away team display name.")
    market: str = Field(description="Provider-native market key for the leg.")
    canonical_market: MarketType | None = Field(
        default=None,
        description="Canonical market hint when the leg maps into PuntLab taxonomy.",
    )
    selection: str = Field(description="Human-readable selection within the chosen market.")
    odds: float = Field(gt=1.0, description="Decimal odds for the chosen leg.")
    provider: str = Field(description="Provider label backing the chosen odds.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence contribution of this leg within the slip.",
    )
    resolution_source: ResolutionSource = Field(
        description="Resolver path that supplied the chosen market."
    )
    market_label: str | None = Field(
        default=None,
        description="Provider-facing market label when useful for delivery.",
    )
    line: float | None = Field(
        default=None,
        description="Numeric line for handicap or totals legs when applicable.",
    )
    rationale: str | None = Field(
        default=None,
        description="Short explanation for why the leg was selected.",
    )
    outcome: LegOutcome = Field(
        default=LegOutcome.PENDING,
        description="Settlement status for the leg.",
    )
    sportybet_url: HttpUrl | None = Field(
        default=None,
        description="Direct SportyBet URL when available for manual lookup.",
    )

    @field_validator(
        "fixture_ref",
        "competition",
        "home_team",
        "away_team",
        "market",
        "selection",
        "provider",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank identifying and provider text fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("market_label", "rationale")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional display text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("odds", "confidence", "line")
    @classmethod
    def validate_finite_numbers(cls, value: float | None, info: object) -> float | None:
        """Reject non-finite leg numeric values before bound checks are trusted."""

        if value is None:
            return None
        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @model_validator(mode="after")
    def validate_leg_shape(self) -> Self:
        """Enforce canonical line-market handling and distinct fixture teams."""

        if self.canonical_market is None:
            with suppress(ValueError):
                self.canonical_market = MarketType(self.market)
        if self.home_team.casefold() == self.away_team.casefold():
            raise ValueError("home_team and away_team must describe different teams.")
        if self.canonical_market in _LINE_BASED_MARKETS and self.line is None:
            raise ValueError(
                f"line is required when market is `{self.canonical_market.value}`."
            )
        return self

    def fixture_label(self) -> str:
        """Return a consistent `Home vs Away` label for delivery surfaces."""

        return f"{self.home_team} vs {self.away_team}"


class AccumulatorSlip(BaseModel):
    """Canonical accumulator slip output from the builder stage.

    Inputs:
        Ordered accumulator legs plus computed slip-level metadata.

    Outputs:
        A validated accumulator contract with deterministic leg ordering and
        coherent odds, confidence, publication, and settlement metadata.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    slip_id: UUID | None = Field(
        default=None,
        description="Database identifier once the slip has been persisted.",
    )
    run_id: UUID | None = Field(
        default=None,
        description="Pipeline run identifier that produced the slip.",
    )
    slip_date: date = Field(description="Date the slip belongs to.")
    slip_number: int = Field(gt=0, description="1-based slip number for the day.")
    legs: tuple[AccumulatorLeg, ...] = Field(
        default=(),
        description="Ordered immutable tuple of accumulator legs.",
    )
    total_odds: float = Field(gt=1.0, description="Combined decimal odds for the full slip.")
    leg_count: int = Field(gt=0, description="Total number of legs in the slip.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Slip-level confidence after builder penalties are applied.",
    )
    strategy: AccumulatorStrategy | None = Field(
        default=None,
        description="Generation strategy used by the builder when known.",
    )
    status: AccumulatorStatus = Field(
        default=AccumulatorStatus.PENDING,
        description="Current approval or settlement lifecycle state.",
    )
    outcome: AccumulatorOutcome | None = Field(
        default=None,
        description="Slip outcome once every leg has settled.",
    )
    is_published: bool = Field(
        default=False,
        description="Whether this slip has been published to users.",
    )
    published_at: datetime | None = Field(
        default=None,
        description="Timezone-aware publication timestamp when the slip is published.",
    )

    @field_validator("total_odds", "confidence")
    @classmethod
    def validate_finite_scores(cls, value: float, info: object) -> float:
        """Reject non-finite slip-level numeric values."""

        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware publication timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "published_at")

    @model_validator(mode="after")
    def validate_slip_integrity(self) -> Self:
        """Ensure leg numbering, odds totals, and publication state stay coherent."""

        if not self.legs:
            raise ValueError("legs must include at least one accumulator leg.")
        if self.leg_count != len(self.legs):
            raise ValueError("leg_count must match the number of supplied legs.")

        leg_numbers = [leg.leg_number for leg in self.legs]
        expected_leg_numbers = list(range(1, len(self.legs) + 1))
        if leg_numbers != expected_leg_numbers:
            raise ValueError("Accumulator legs must be consecutively numbered starting at 1.")

        fixture_refs = [leg.fixture_ref for leg in self.legs]
        if len(set(fixture_refs)) != len(fixture_refs):
            raise ValueError("A slip must not include the same fixture more than once.")

        computed_total = prod(leg.odds for leg in self.legs)
        # Allow a tiny tolerance for upstream rounding while still catching
        # broken odds aggregation early.
        if not isclose(computed_total, self.total_odds, rel_tol=0.0, abs_tol=0.02):
            raise ValueError("total_odds must match the product of the leg odds.")

        if self.is_published and self.status is AccumulatorStatus.BLOCKED:
            raise ValueError("Blocked accumulators cannot be published.")
        if self.is_published and self.published_at is None:
            raise ValueError("published_at is required when is_published is true.")
        if not self.is_published and self.published_at is not None:
            raise ValueError("published_at must be omitted until the slip is published.")

        if self.outcome is not None and self.status is not AccumulatorStatus.SETTLED:
            raise ValueError("status must be `settled` when an accumulator outcome is present.")
        return self


class ExplainedAccumulator(AccumulatorSlip):
    """Accumulator slip enriched with human-readable explanation text.

    Inputs:
        A validated accumulator slip plus generated explanation content.

    Outputs:
        A delivery-ready accumulator object containing overall rationale and
        optional per-leg rationale fields on each leg.
    """

    rationale: str = Field(
        min_length=1,
        max_length=320,
        description="Short human-readable rationale for the full accumulator.",
    )

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        """Reject blank rationales after whitespace normalization."""

        return require_non_blank_text(value, "rationale")


__all__ = [
    "AccumulatorLeg",
    "AccumulatorOutcome",
    "AccumulatorSlip",
    "AccumulatorStatus",
    "AccumulatorStrategy",
    "ExplainedAccumulator",
    "LegOutcome",
    "ResolutionSource",
    "ResolvedMarket",
]
