"""Analysis-stage schemas shared across PuntLab's scoring pipeline.

Purpose: define canonical qualitative context, scoring-factor, and ranked
match contracts used between research, scoring, and ranking stages.
Scope: LLM-assisted context, deterministic score breakdowns, and scoring
weight validation for one analyzed fixture at a time.
Dependencies: shared sport and market enums from `src.config`, plus common
validation helpers from `src.schemas.common`.
"""

from __future__ import annotations

from contextlib import suppress
from math import isclose
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import MarketType, SportName
from src.schemas.common import (
    normalize_optional_text,
    require_finite_number,
    require_non_blank_text,
)


def _normalize_unique_sources(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize ordered source labels and reject blank or duplicate entries.

    Args:
        values: Source names supplied by research or scoring stages.

    Returns:
        A trimmed, ordered tuple with case-insensitive deduplication applied.

    Raises:
        ValueError: If no usable source labels remain after normalization.
    """

    normalized: list[str] = []
    seen: set[str] = set()

    for raw_value in values:
        source = require_non_blank_text(raw_value, "data_sources")
        lookup_key = source.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        normalized.append(source)

    if not normalized:
        raise ValueError("data_sources must include at least one source label.")

    return tuple(normalized)


class MatchContext(BaseModel):
    """LLM-generated SportyBet fixture context for a single fixture.

    Inputs:
        Fixture-aware research prompts grounded first in SportyBet pre-match
        fixture-page widgets, with RSS/Tavily news only as supplemental context.

    Outputs:
        A normalized context object with compact SportyBet-derived summaries and
        an aggregate qualitative score for downstream LLM market scoring.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str | None = Field(
        default=None,
        description="Fixture reference tied to this context when already known.",
    )
    fixture_detail_summary: str = Field(
        min_length=1,
        max_length=600,
        description="Primary summary derived from SportyBet fixture-page widgets.",
    )
    tactical_context: str | None = Field(
        default=None,
        max_length=600,
        description="Lineup, formation, style, or matchup context from SportyBet widgets.",
    )
    statistical_context: str | None = Field(
        default=None,
        max_length=600,
        description="Statistics, comparison, H2H, table, or probability context.",
    )
    availability_context: str | None = Field(
        default=None,
        max_length=600,
        description="Absence, lineup, suspension, or team-info context when available.",
    )
    market_context: str | None = Field(
        default=None,
        max_length=600,
        description="Concise read of relevant SportyBet market shape without a final pick.",
    )
    supplemental_news_context: str | None = Field(
        default=None,
        max_length=500,
        description="RSS/Tavily context used only to enrich SportyBet fixture detail analysis.",
    )
    qualitative_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the fixture context quality and betting signal.",
    )
    data_sources: tuple[str, ...] = Field(
        default=(),
        description="Ordered list of source labels used to form the context.",
    )

    @field_validator(
        "fixture_ref",
        "tactical_context",
        "statistical_context",
        "availability_context",
        "market_context",
        "supplemental_news_context",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional context text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("fixture_detail_summary")
    @classmethod
    def validate_fixture_detail_summary(cls, value: str) -> str:
        """Reject blank fixture summaries after whitespace normalization."""

        return require_non_blank_text(value, "fixture_detail_summary")

    @field_validator("qualitative_score")
    @classmethod
    def validate_bounded_scores(cls, value: float, info: object) -> float:
        """Reject non-finite scoring inputs before bound checks are trusted."""

        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @field_validator("data_sources", mode="before")
    @classmethod
    def validate_data_sources(cls, value: object) -> tuple[str, ...]:
        """Require at least one unique source for qualitative context."""

        if isinstance(value, str):
            return _normalize_unique_sources((value,))
        if isinstance(value, list | tuple):
            return _normalize_unique_sources(tuple(str(item) for item in value))
        return _normalize_unique_sources(())


class QualitativeScore(BaseModel):
    """Structured qualitative assessment generated from stats, news, and context.

    Inputs:
        Aggregated stats, research context, and qualitative prompts for one
        fixture.

    Outputs:
        A bounded scoring bundle with a single overall qualitative score plus
        decomposed sub-signals and a terse explanation for tracing.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str | None = Field(
        default=None,
        description="Fixture reference tied to this assessment when available.",
    )
    momentum_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized signal for recent momentum and form trajectory.",
    )
    lineup_stability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized signal for likely lineup continuity and availability.",
    )
    motivation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized signal for fixture importance and team motivation.",
    )
    narrative_alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized signal for how strongly the narrative supports the edge.",
    )
    qualitative_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall bounded qualitative contribution for downstream scoring.",
    )
    summary: str = Field(
        min_length=1,
        max_length=240,
        description="Short explanation of the qualitative edge or caution.",
    )
    data_sources: tuple[str, ...] = Field(
        default=(),
        description="Ordered unique labels for supporting qualitative sources.",
    )

    @field_validator("fixture_ref")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional fixture references and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        """Reject blank summaries after normalization."""

        return require_non_blank_text(value, "summary")

    @field_validator(
        "momentum_score",
        "lineup_stability_score",
        "motivation_score",
        "narrative_alignment_score",
        "qualitative_score",
    )
    @classmethod
    def validate_finite_scores(cls, value: float, info: object) -> float:
        """Reject non-finite qualitative signals before bound checks are applied."""

        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @field_validator("data_sources", mode="before")
    @classmethod
    def validate_data_sources(cls, value: object) -> tuple[str, ...]:
        """Require at least one unique source for qualitative scoring."""

        if isinstance(value, str):
            return _normalize_unique_sources((value,))
        if isinstance(value, list | tuple):
            return _normalize_unique_sources(tuple(str(item) for item in value))
        return _normalize_unique_sources(())


class ScoreFactorBreakdown(BaseModel):
    """Deterministic and qualitative factor breakdown for one match score.

    Inputs:
        Individual factor outputs from the scoring engine.

    Outputs:
        A bounded, serializable factor bundle suitable for persistence and
        ranking diagnostics.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    form: float = Field(ge=0.0, le=1.0, description="Recent-form factor score.")
    h2h: float = Field(ge=0.0, le=1.0, description="Head-to-head factor score.")
    injury_impact: float = Field(
        ge=0.0,
        le=1.0,
        description="Availability and injury impact factor score.",
    )
    odds_value: float = Field(
        ge=0.0,
        le=1.0,
        description="Value-versus-market factor score.",
    )
    context: float = Field(
        ge=0.0,
        le=1.0,
        description="Qualitative context factor score.",
    )
    venue: float = Field(ge=0.0, le=1.0, description="Venue and home/away factor score.")
    statistical: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Raw statistical edge factor score.",
    )

    @field_validator(
        "form",
        "h2h",
        "injury_impact",
        "odds_value",
        "context",
        "venue",
        "statistical",
    )
    @classmethod
    def validate_finite_scores(cls, value: float, info: object) -> float:
        """Reject NaN and infinite factor scores."""

        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)


class ScoringWeights(BaseModel):
    """Configurable weights used by the composite scoring engine.

    Inputs:
        Runtime or environment-supplied factor weights.

    Outputs:
        A validated set of bounded weights whose total is effectively `1.0`.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    form: float = Field(default=0.25, ge=0.0, le=1.0, description="Form-factor weight.")
    h2h: float = Field(default=0.10, ge=0.0, le=1.0, description="Head-to-head weight.")
    injury_impact: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Availability and injury-impact weight.",
    )
    odds_value: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Market-value weight.",
    )
    context: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Qualitative context weight.",
    )
    venue: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Venue and home/away advantage weight.",
    )
    statistical: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Statistical edge weight.",
    )

    @field_validator(
        "form",
        "h2h",
        "injury_impact",
        "odds_value",
        "context",
        "venue",
        "statistical",
    )
    @classmethod
    def validate_finite_weights(cls, value: float, info: object) -> float:
        """Reject non-finite weight values before sum validation runs."""

        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @model_validator(mode="after")
    def validate_weight_sum(self) -> Self:
        """Ensure the factor weights resolve to one canonical composite scale."""

        total = (
            self.form
            + self.h2h
            + self.injury_impact
            + self.odds_value
            + self.context
            + self.venue
            + self.statistical
        )
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=0.001):
            raise ValueError("Scoring weights must sum to 1.0 within a tolerance of 0.001.")
        return self


class MatchScore(BaseModel):
    """Composite score output for one analyzed fixture.

    Inputs:
        Fixture metadata plus deterministic and qualitative factor outputs from
        the scoring engine.

    Outputs:
        A validated scoring record that downstream ranking and market
        resolution stages can sort, persist, and explain.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(description="Canonical fixture reference for the score.")
    sport: SportName = Field(description="Sport associated with the scored fixture.")
    competition: str = Field(description="Competition display name for the fixture.")
    home_team: str = Field(description="Home team display name.")
    away_team: str = Field(description="Away team display name.")
    composite_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Final composite score used for ranking.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall recommendation confidence after completeness checks.",
    )
    factors: ScoreFactorBreakdown = Field(
        description="Factor-level breakdown that produced the composite score.",
    )
    recommended_market: str | None = Field(
        default=None,
        description="Provider-native market key selected by the scoring engine.",
    )
    recommended_market_label: str | None = Field(
        default=None,
        description="Display label for the selected provider market.",
    )
    recommended_canonical_market: MarketType | None = Field(
        default=None,
        description="Canonical market hint when the selected market fits PuntLab taxonomy.",
    )
    recommended_selection: str | None = Field(
        default=None,
        description="Best-fit provider-facing selection within the recommended market.",
    )
    recommended_odds: float | None = Field(
        default=None,
        gt=1.0,
        description="Decimal odds tied to the recommended selection when known.",
    )
    recommended_line: float | None = Field(
        default=None,
        description="Numeric line tied to the recommended market when known.",
    )
    qualitative_summary: str | None = Field(
        default=None,
        description="Short human-readable context note for tracing and explanation.",
    )

    @field_validator("sport", mode="before")
    @classmethod
    def validate_sport_label(cls, value: object) -> object:
        """Normalize common LLM-facing sport labels into canonical enum values."""

        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"football", "soccer"}:
                return SportName.SOCCER
            if normalized in {"basketball", "nba"}:
                return SportName.BASKETBALL
        return value

    @field_validator("recommended_canonical_market", mode="before")
    @classmethod
    def validate_recommended_canonical_market(cls, value: object) -> object:
        """Normalize common LLM-facing market labels into canonical enum values."""

        if value is None or isinstance(value, MarketType):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
            label_map = {
                "1x2": MarketType.MATCH_RESULT,
                "match_result": MarketType.MATCH_RESULT,
                "full_time_result": MarketType.MATCH_RESULT,
                "fulltime_result": MarketType.MATCH_RESULT,
                "btts": MarketType.BTTS,
                "both_teams_to_score": MarketType.BTTS,
                "double_chance": MarketType.DOUBLE_CHANCE,
                "draw_no_bet": MarketType.DRAW_NO_BET,
                "dnb": MarketType.DRAW_NO_BET,
                "over_under_0.5": MarketType.OVER_UNDER_05,
                "over_under_1.5": MarketType.OVER_UNDER_15,
                "over_under_2.5": MarketType.OVER_UNDER_25,
                "over_under_3.5": MarketType.OVER_UNDER_35,
            }
            return label_map.get(normalized, value)
        return value

    @field_validator(
        "fixture_ref",
        "competition",
        "home_team",
        "away_team",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank fixture-identifying text fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator(
        "recommended_market",
        "recommended_market_label",
        "recommended_selection",
        "qualitative_summary",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional explanatory text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator(
        "composite_score",
        "confidence",
        "recommended_odds",
        "recommended_line",
    )
    @classmethod
    def validate_finite_score_fields(cls, value: float | None, info: object) -> float | None:
        """Reject non-finite composite, confidence, and odds values."""

        if value is None:
            return None
        field_name = getattr(info, "field_name", "value")
        return require_finite_number(value, field_name)

    @model_validator(mode="after")
    def validate_recommendation_shape(self) -> Self:
        """Ensure optional recommendation fields appear in coherent combinations."""

        if self.home_team.casefold() == self.away_team.casefold():
            raise ValueError("home_team and away_team must describe different teams.")
        if self.recommended_canonical_market is None and self.recommended_market is not None:
            with suppress(ValueError):
                self.recommended_canonical_market = MarketType(self.recommended_market)
        if self.recommended_market is not None and self.recommended_selection is None:
            raise ValueError(
                "recommended_selection is required when recommended_market is provided."
            )
        if self.recommended_odds is not None and self.recommended_market is None:
            raise ValueError(
                "recommended_market is required when recommended_odds is provided."
            )
        return self


class RankedMatch(MatchScore):
    """Ranked match output produced by the ranking stage.

    Inputs:
        A validated `MatchScore` plus its global ranking position.

    Outputs:
        A score record that can be consumed directly by market resolution and
        accumulator-building logic.
    """

    rank: int = Field(gt=0, description="1-based global ranking position for the day.")


__all__ = [
    "MatchContext",
    "MatchScore",
    "QualitativeScore",
    "RankedMatch",
    "ScoreFactorBreakdown",
    "ScoringWeights",
]
