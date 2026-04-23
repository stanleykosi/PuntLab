"""Fixture-page detail schemas for SportyBet-powered research context.

Purpose: carry compact per-fixture SportyBet/Sportradar page details through
the pipeline without making markdown exports part of runtime analysis.
Scope: lineups, statistics, team info, previews, standings, H2H, and related
fixture-page widget sections.
Dependencies: Pydantic validation helpers and shared text normalization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_non_blank_text,
)


class FixtureDetailSection(BaseModel):
    """One compact SportyBet fixture-page section.

    Inputs:
        A normalized widget key, rendered lines, and supporting response URLs.

    Outputs:
        A prompt-ready section that keeps data structured until rendering.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    widget_key: str = Field(description="Stable internal widget key.")
    widget_type: str = Field(description="Sportradar widget type string.")
    status: Literal["mounted", "timeout", "error", "unavailable"] = Field(
        description="Widget fetch or render status.",
    )
    headings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Widget headings found during capture.",
    )
    content_lines: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Deduplicated, prompt-ready lines for this section.",
    )
    response_urls: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Supporting Sportradar response URLs for traceability.",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure detail when the section could not be populated.",
    )

    @field_validator("widget_key", "widget_type")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("headings", "content_lines", "response_urls")
    @classmethod
    def validate_text_tuples(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        """Normalize text tuple fields while preserving order."""

        field_name = getattr(info, "field_name", "value")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized_item = require_non_blank_text(item, field_name)
            lookup_key = normalized_item.casefold()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            normalized.append(normalized_item)
        return tuple(normalized)

    @field_validator("error_message")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text and collapse empties."""

        return normalize_optional_text(value)


class FixtureDetails(BaseModel):
    """Compact SportyBet fixture-page details for one pipeline fixture."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(description="Canonical pipeline fixture reference.")
    source_provider: str = Field(default="sportybet", description="Detail source label.")
    fixture_url: str = Field(description="Public SportyBet fixture URL used for capture.")
    event_id: str = Field(description="Canonical `sr:match:*` event id.")
    match_id: str = Field(description="Numeric Sportradar match id.")
    fetched_at: datetime = Field(description="Timezone-aware capture timestamp.")
    widget_loader_status: Literal["loaded", "failed"] = Field(
        description="Whether the Sportradar widget loader executed.",
    )
    sections: tuple[FixtureDetailSection, ...] = Field(
        default_factory=tuple,
        description="Captured fixture-page sections.",
    )

    @field_validator("fixture_ref", "source_provider", "fixture_url", "event_id", "match_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware timestamps."""

        return ensure_timezone_aware(value, "fetched_at")


__all__ = ["FixtureDetailSection", "FixtureDetails"]
