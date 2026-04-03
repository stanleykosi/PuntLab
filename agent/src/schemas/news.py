"""Normalized news schemas shared across PuntLab's research pipeline.

Purpose: define the canonical article contract used by RSS ingestion, Tavily
search, and the qualitative research stage.
Scope: article identity, publication metadata, team matching, and relevance.
Dependencies: Pydantic validation plus shared sport enums from `src.config`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from src.config import SportName
from src.schemas.common import (
    ensure_timezone_aware,
    normalize_optional_text,
    require_non_blank_text,
)


class NewsArticle(BaseModel):
    """Canonical normalized article model used by the research stage.

    Inputs:
        News items normalized from RSS feeds or web-search providers.

    Outputs:
        A validated article contract with publication metadata, matched teams,
        and optional relevance scoring for fixture-level research.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: str = Field(description="Article headline or title.")
    url: HttpUrl = Field(description="Canonical article URL.")
    published_at: datetime = Field(description="Timezone-aware publication timestamp.")
    source: str = Field(description="Publisher or publication name.")
    source_provider: str = Field(description="Pipeline provider that supplied the article.")
    summary: str | None = Field(default=None, description="Short article summary or abstract.")
    content_snippet: str | None = Field(
        default=None,
        description="Relevant snippet or excerpt extracted by the provider.",
    )
    sport: SportName | None = Field(
        default=None,
        description="Detected sport when the source is fixture-specific.",
    )
    competition: str | None = Field(
        default=None,
        description="Competition or league referenced by the article.",
    )
    teams: tuple[str, ...] = Field(
        default=(),
        description="Ordered unique team names referenced in the article.",
    )
    fixture_ref: str | None = Field(
        default=None,
        description="Linked fixture reference when the article is matched to a game.",
    )
    author: str | None = Field(default=None, description="Article author when provided.")
    source_id: str | None = Field(
        default=None,
        description="Provider-native article identifier when available.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional normalized relevance score for ranking research inputs.",
    )

    @field_validator("headline", "source", "source_provider")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings after trimming."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator(
        "summary",
        "content_snippet",
        "competition",
        "fixture_ref",
        "author",
        "source_id",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional article text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        """Require timezone-aware publication timestamps."""

        return ensure_timezone_aware(value, "published_at")

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance_score(cls, value: float | None) -> float | None:
        """Reject NaN and infinite relevance scores while preserving optionality."""

        if value is None:
            return None
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("relevance_score must be a finite number.")
        return value

    @field_validator("teams")
    @classmethod
    def normalize_team_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Trim, deduplicate, and preserve the order of matched team names."""

        normalized: list[str] = []
        seen: set[str] = set()

        for raw_name in value:
            team_name = require_non_blank_text(raw_name, "teams")
            lookup_key = team_name.casefold()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            normalized.append(team_name)

        return tuple(normalized)


__all__ = ["NewsArticle"]
