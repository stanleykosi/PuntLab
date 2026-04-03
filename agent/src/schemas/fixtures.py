"""Normalized fixture schemas shared across PuntLab's ingestion pipeline.

Purpose: define the canonical fixture contract used by providers, research,
scoring, and market resolution layers.
Scope: pre-match fixture identity, scheduling, and SportyBet lookup metadata.
Dependencies: Pydantic for validation plus shared sport enums from `src.config`.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import SportName
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_non_blank_text,
    slugify_segment,
)

_SPORTRADAR_MATCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sr:match:\d+$")


class FixtureStatus(StrEnum):
    """Lifecycle states supported by the canonical normalized fixture model."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class NormalizedFixture(BaseModel):
    """Canonical fixture model used throughout the agent pipeline.

    Inputs:
        Provider-normalized fixture data from ingestion adapters.

    Outputs:
        A validated, deterministic fixture contract with stable identity fields,
        timezone-aware kickoff timing, and SportyBet lookup metadata.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sportradar_id: str | None = Field(
        default=None,
        description="Canonical Sportradar match identifier when available.",
    )
    home_team: str = Field(description="Home team display name.")
    away_team: str = Field(description="Away team display name.")
    competition: str = Field(description="Competition display name.")
    sport: SportName = Field(description="Supported sport for this fixture.")
    kickoff: datetime = Field(description="Timezone-aware kickoff timestamp.")
    source_provider: str = Field(description="Provider that supplied the fixture.")
    source_id: str = Field(description="Provider-specific fixture identifier.")
    country: str | None = Field(
        default=None,
        description="Country or region label for the competition.",
    )
    league: str | None = Field(
        default=None,
        description="URL-safe league segment used for SportyBet resolution.",
    )
    home_team_id: str | None = Field(
        default=None,
        description="Provider-specific home team identifier.",
    )
    away_team_id: str | None = Field(
        default=None,
        description="Provider-specific away team identifier.",
    )
    venue: str | None = Field(default=None, description="Match venue when known.")
    status: FixtureStatus = Field(
        default=FixtureStatus.SCHEDULED,
        description="Current fixture lifecycle state.",
    )

    @field_validator(
        "home_team",
        "away_team",
        "competition",
        "source_provider",
        "source_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after whitespace normalization."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("country", "league", "home_team_id", "away_team_id", "venue")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("sportradar_id")
    @classmethod
    def validate_sportradar_id(cls, value: str | None) -> str | None:
        """Ensure Sportradar identifiers follow the canonical `sr:match:*` shape."""

        normalized = normalize_optional_text(value)
        if normalized is None:
            return None
        if not _SPORTRADAR_MATCH_PATTERN.fullmatch(normalized):
            raise ValueError("sportradar_id must match the pattern `sr:match:<id>`.")
        return normalized

    @field_validator("kickoff")
    @classmethod
    def validate_kickoff(cls, value: datetime) -> datetime:
        """Require timezone-aware kickoff timestamps."""

        return ensure_timezone_aware(value, "kickoff")

    @model_validator(mode="after")
    def validate_fixture_consistency(self) -> NormalizedFixture:
        """Apply cross-field validation and derived league normalization."""

        if self.home_team.casefold() == self.away_team.casefold():
            raise ValueError("home_team and away_team must be different teams.")

        if self.league is None:
            self.league = slugify_segment(self.competition)

        return self

    def get_fixture_ref(self) -> str:
        """Return the canonical cross-provider reference for this fixture.

        Returns:
            The Sportradar ID when present, otherwise a provider-scoped fallback
            identifier built from the source provider and source fixture ID.
        """

        return self.sportradar_id or f"{self.source_provider}:{self.source_id}"

    def get_sportybet_country_slug(self) -> str | None:
        """Return a URL-safe SportyBet country segment when country is known."""

        if self.country is None:
            return None
        return slugify_segment(self.country)

    def get_sportybet_league_slug(self) -> str:
        """Return the normalized SportyBet league segment for this fixture."""

        return self.league or slugify_segment(self.competition)
