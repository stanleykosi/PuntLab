"""Canonical LangGraph pipeline state schema for PuntLab.

Purpose: define the master state object that flows through every pipeline
stage, keeping stage outputs, approval metadata, and delivery results in one
validated contract.
Scope: run metadata, stage-by-stage normalized data collections, approval
status, and execution error tracking for a single daily pipeline run.
Dependencies: shared schemas from `src.schemas` plus common validation helpers
for timezone-aware timestamps and fail-fast text normalization.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.providers.odds_mapping import OddsMarketCatalog
from src.schemas.accumulators import AccumulatorSlip, ExplainedAccumulator, ResolvedMarket
from src.schemas.analysis import MatchContext, MatchScore, RankedMatch
from src.schemas.common import ensure_timezone_aware, require_non_blank_text
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, PlayerStats, TeamStats
from src.schemas.users import DeliveryResult


class PipelineStage(StrEnum):
    """Canonical stage names used by the LangGraph pipeline runtime."""

    INGESTION = "ingestion"
    RESEARCH = "research"
    SCORING = "scoring"
    RANKING = "ranking"
    MARKET_RESOLUTION = "market_resolution"
    ACCUMULATOR_BUILDING = "accumulator_building"
    EXPLANATION = "explanation"
    APPROVAL = "approval"
    DELIVERY = "delivery"


class ApprovalStatus(StrEnum):
    """Supported approval outcomes for a pipeline run."""

    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"


def _normalize_string_items(values: list[str], field_name: str) -> list[str]:
    """Trim list items and reject blank pipeline identifiers or error messages.

    Args:
        values: Ordered string values supplied for a pipeline list field.
        field_name: Field name used in validation errors.

    Returns:
        A normalized list preserving the original order.

    Raises:
        ValueError: If any list item is blank after whitespace normalization.
    """

    return [require_non_blank_text(value, f"{field_name}_item") for value in values]


class PipelineState(BaseModel):
    """Master state object passed between all PuntLab LangGraph nodes.

    Inputs:
        Run metadata plus the normalized outputs produced by each completed
        pipeline stage.

    Outputs:
        A validated state snapshot that downstream nodes can consume without
        re-validating fixtures, scores, accumulators, approval state, or
        delivery outcomes.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    run_id: str = Field(description="Unique identifier for this pipeline execution.")
    run_date: date = Field(description="Calendar date the pipeline is analyzing.")
    started_at: datetime = Field(
        description="Timezone-aware timestamp for when the run started."
    )
    current_stage: PipelineStage = Field(
        default=PipelineStage.INGESTION,
        description="Current stage the pipeline is executing or last completed.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Ordered error messages captured during execution.",
    )

    fixtures: list[NormalizedFixture] = Field(
        default_factory=list,
        description="Stage 1 fixture outputs gathered during ingestion.",
    )
    odds_market_catalog: OddsMarketCatalog = Field(
        default_factory=OddsMarketCatalog,
        description=(
            "Stage 1 full provider-market odds catalog, including both scoreable and "
            "currently unmapped selections."
        ),
    )
    odds_data: list[NormalizedOdds] = Field(
        default_factory=list,
        description=(
            "Stage 1 canonically mapped, scoreable odds rows derived from the full "
            "ingested provider market catalog."
        ),
    )
    team_stats: list[TeamStats] = Field(
        default_factory=list,
        description="Stage 1 team-stat outputs gathered during ingestion.",
    )
    player_stats: list[PlayerStats] = Field(
        default_factory=list,
        description="Stage 1 player-stat outputs gathered during ingestion.",
    )
    injuries: list[InjuryData] = Field(
        default_factory=list,
        description="Stage 1 injury and suspension outputs gathered during ingestion.",
    )
    news_articles: list[NewsArticle] = Field(
        default_factory=list,
        description="Stage 1 article outputs gathered during ingestion.",
    )

    match_contexts: list[MatchContext] = Field(
        default_factory=list,
        description="Stage 2 qualitative research outputs for analyzed fixtures.",
    )

    match_scores: list[MatchScore] = Field(
        default_factory=list,
        description="Stage 3 composite scoring outputs for analyzed fixtures.",
    )

    ranked_matches: list[RankedMatch] = Field(
        default_factory=list,
        description="Stage 4 globally ranked match outputs.",
    )

    resolved_markets: list[ResolvedMarket] = Field(
        default_factory=list,
        description="Stage 5 resolved bookmaker markets for ranked fixtures.",
    )

    accumulators: list[AccumulatorSlip] = Field(
        default_factory=list,
        description="Stage 6 accumulator slips assembled from resolved markets.",
    )

    explained_accumulators: list[ExplainedAccumulator] = Field(
        default_factory=list,
        description="Stage 7 delivery-ready accumulators enriched with rationale text.",
    )

    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="Stage 8 approval status used by the publication gate.",
    )
    blocked_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers for slips or fixtures explicitly blocked during approval.",
    )

    delivery_results: list[DeliveryResult] = Field(
        default_factory=list,
        description="Stage 9 delivery outcomes recorded across client channels.",
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """Reject blank run identifiers after whitespace normalization."""

        return require_non_blank_text(value, "run_id")

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: datetime) -> datetime:
        """Require timezone-aware pipeline start timestamps."""

        return ensure_timezone_aware(value, "started_at")

    @field_validator("errors", "blocked_ids")
    @classmethod
    def validate_string_lists(cls, value: list[str], info: object) -> list[str]:
        """Normalize and validate list-based error and blocked-reference fields."""

        field_name = getattr(info, "field_name", "value")
        return _normalize_string_items(value, field_name)


__all__ = ["ApprovalStatus", "PipelineStage", "PipelineState"]
