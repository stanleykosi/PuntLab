"""Normalized team, player, and injury schemas for PuntLab providers.

Purpose: define the canonical statistical and availability data contracts
shared between ingestion, scoring, and research stages.
Scope: team form snapshots, player-level stat bundles, and injury signals.
Dependencies: Pydantic validation helpers and shared sport enums from `src.config`.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import SportName
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_non_blank_text,
)


def _normalize_metric_map(values: dict[str, float], field_name: str) -> dict[str, float]:
    """Normalize metric-map keys and reject non-finite numeric values.

    Args:
        values: Provider-supplied metric mapping.
        field_name: Field name used in validation errors.

    Returns:
        A normalized metric map with trimmed keys.

    Raises:
        ValueError: If a key is blank or a value is not finite.
    """

    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = require_non_blank_text(raw_key, f"{field_name}_key")
        if not isfinite(raw_value):
            raise ValueError(f"{field_name}[{key}] must be a finite number.")
        normalized[key] = raw_value
    return normalized


def _normalize_form(value: str | None) -> str | None:
    """Normalize compact form strings such as `WWDLW` or `W-L-W`.

    Args:
        value: Provider-supplied form string.

    Returns:
        A compact uppercase form string or `None`.

    Raises:
        ValueError: If unsupported characters remain after normalization.
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    compact = normalized.replace(" ", "").replace("-", "").upper()
    if not compact:
        return None
    if any(character not in {"W", "D", "L"} for character in compact):
        raise ValueError("form must contain only W, D, and L markers.")
    return compact


class InjuryType(StrEnum):
    """Canonical injury and availability categories used by ingestion."""

    INJURY = "injury"
    SUSPENSION = "suspension"
    DOUBTFUL = "doubtful"
    QUESTIONABLE = "questionable"
    ILLNESS = "illness"
    REST = "rest"
    OTHER = "other"


class TeamStats(BaseModel):
    """Canonical statistical snapshot for one team in a competition.

    Inputs:
        Provider-normalized team form and table metrics.

    Outputs:
        A validated team summary used by scoring factors and later persistence.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    team_id: str = Field(description="Provider-stable team identifier.")
    team_name: str = Field(description="Display name for the team.")
    sport: SportName = Field(description="Sport this snapshot belongs to.")
    source_provider: str = Field(description="Provider that supplied the stats snapshot.")
    fetched_at: datetime = Field(description="Timezone-aware snapshot timestamp.")
    competition: str | None = Field(
        default=None,
        description="Competition or league display name.",
    )
    season: str | None = Field(default=None, description="Season label such as `2025-26`.")
    matches_played: int = Field(default=0, ge=0, description="Matches included in the sample.")
    wins: int = Field(default=0, ge=0, description="Matches won in the sample.")
    draws: int = Field(default=0, ge=0, description="Matches drawn in the sample.")
    losses: int = Field(default=0, ge=0, description="Matches lost in the sample.")
    goals_for: int = Field(default=0, ge=0, description="Goals or points scored.")
    goals_against: int = Field(default=0, ge=0, description="Goals or points conceded.")
    clean_sheets: int = Field(default=0, ge=0, description="Soccer clean-sheet count.")
    form: str | None = Field(
        default=None,
        description="Compact recent-results string such as `WWDLW`.",
    )
    position: int | None = Field(default=None, gt=0, description="Table or conference position.")
    points: int | None = Field(default=None, ge=0, description="Table points total.")
    home_wins: int = Field(default=0, ge=0, description="Home wins in the sample.")
    away_wins: int = Field(default=0, ge=0, description="Away wins in the sample.")
    avg_goals_scored: float | None = Field(
        default=None,
        ge=0.0,
        description="Average goals or points scored per game.",
    )
    avg_goals_conceded: float | None = Field(
        default=None,
        ge=0.0,
        description="Average goals or points conceded per game.",
    )
    advanced_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional provider-specific metrics such as xG, pace, or ELO deltas.",
    )

    @field_validator("team_id", "team_name", "source_provider")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("competition", "season")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional provider text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("form")
    @classmethod
    def validate_form(cls, value: str | None) -> str | None:
        """Normalize recent-results shorthand into a compact uppercase string."""

        return _normalize_form(value)

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware snapshot timestamps."""

        return ensure_timezone_aware(value, "fetched_at")

    @field_validator("avg_goals_scored", "avg_goals_conceded")
    @classmethod
    def validate_average_fields(cls, value: float | None, info: object) -> float | None:
        """Reject non-finite averages while preserving optionality."""

        if value is None:
            return None
        field_name = getattr(info, "field_name", "value")
        if not isfinite(value):
            raise ValueError(f"{field_name} must be a finite number.")
        return value

    @field_validator("advanced_metrics")
    @classmethod
    def validate_advanced_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        """Normalize and validate optional advanced metric maps."""

        return _normalize_metric_map(value, "advanced_metrics")

    @model_validator(mode="after")
    def validate_record_totals(self) -> TeamStats:
        """Validate cross-field consistency for wins, losses, and sample size."""

        record_total = self.wins + self.draws + self.losses
        if record_total > self.matches_played:
            raise ValueError("wins + draws + losses must not exceed matches_played.")
        if self.home_wins + self.away_wins > self.wins:
            raise ValueError("home_wins + away_wins must not exceed wins.")
        if self.clean_sheets > self.matches_played:
            raise ValueError("clean_sheets must not exceed matches_played.")
        return self


class PlayerStats(BaseModel):
    """Canonical player-level statistical snapshot for provider integrations.

    Inputs:
        Provider-normalized player totals, rates, and contextual metadata.

    Outputs:
        A validated player stat bundle with a typed metadata shell and a
        flexible finite metric map for sport-specific measures.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    player_id: str = Field(description="Provider-stable player identifier.")
    player_name: str = Field(description="Display name for the player.")
    team_id: str = Field(description="Provider-stable team identifier.")
    sport: SportName = Field(description="Sport this player snapshot belongs to.")
    source_provider: str = Field(description="Provider that supplied the stats snapshot.")
    fetched_at: datetime = Field(description="Timezone-aware snapshot timestamp.")
    team_name: str | None = Field(default=None, description="Display name for the team.")
    competition: str | None = Field(default=None, description="Competition display name.")
    season: str | None = Field(default=None, description="Season label such as `2025-26`.")
    position: str | None = Field(default=None, description="Player position when available.")
    appearances: int = Field(default=0, ge=0, description="Games played.")
    starts: int = Field(default=0, ge=0, description="Games started.")
    minutes_played: int = Field(default=0, ge=0, description="Minutes played in total.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Finite provider-specific metrics such as goals, rebounds, or assists.",
    )

    @field_validator(
        "player_id",
        "player_name",
        "team_id",
        "source_provider",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("team_name", "competition", "season", "position")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional display text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware snapshot timestamps."""

        return ensure_timezone_aware(value, "fetched_at")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        """Normalize and validate provider-specific metric maps."""

        return _normalize_metric_map(value, "metrics")

    @model_validator(mode="after")
    def validate_player_totals(self) -> PlayerStats:
        """Ensure start counts do not exceed total appearances."""

        if self.starts > self.appearances:
            raise ValueError("starts must not exceed appearances.")
        return self


class InjuryData(BaseModel):
    """Canonical injury or suspension signal tied to a specific fixture.

    Inputs:
        Provider-normalized availability updates and suspension reports.

    Outputs:
        A validated injury object suitable for deterministic scoring inputs and
        later fixture-scoped persistence.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(description="Fixture reference linked to the availability update.")
    team_id: str = Field(description="Provider-stable team identifier.")
    player_name: str = Field(description="Display name of the affected player.")
    source_provider: str = Field(description="Provider that supplied the injury signal.")
    injury_type: InjuryType = Field(description="Canonical injury or availability category.")
    team_name: str | None = Field(default=None, description="Display name of the affected team.")
    player_id: str | None = Field(default=None, description="Provider-stable player identifier.")
    reason: str | None = Field(default=None, description="Human-readable injury explanation.")
    is_key_player: bool = Field(
        default=False,
        description="Whether the player is treated as materially important.",
    )
    expected_return: date | None = Field(
        default=None,
        description="Expected return date when the provider supplies one.",
    )
    reported_at: datetime | None = Field(
        default=None,
        description="Timezone-aware report timestamp when available.",
    )

    @field_validator("fixture_ref", "team_id", "player_name", "source_provider")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("team_name", "player_id", "reason")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional injury text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("reported_at")
    @classmethod
    def validate_reported_at(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware injury timestamps when present."""

        if value is None:
            return None
        return ensure_timezone_aware(value, "reported_at")


__all__ = ["InjuryData", "InjuryType", "PlayerStats", "TeamStats"]
