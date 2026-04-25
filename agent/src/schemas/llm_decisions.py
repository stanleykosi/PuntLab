"""LLM decision schemas for canonical recommendation stages.

Purpose: define validated JSON contracts for model-led ranking, market
resolution, and accumulator construction.
Scope: decision outputs only; existing domain schemas still represent final
pipeline state.
Dependencies: accumulator enums plus common validation helpers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.schemas.accumulators import AccumulatorStrategy
from src.schemas.common import normalize_optional_text, require_non_blank_text


class LLMRankingDecision(BaseModel):
    """Fixture order selected by the ranking LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ranked_fixture_refs: tuple[str, ...] = Field(
        min_length=1,
        description="Fixture refs ordered from strongest to weakest recommendation.",
    )

    @field_validator("ranked_fixture_refs")
    @classmethod
    def validate_ranked_fixture_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank or duplicate fixture refs."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            fixture_ref = require_non_blank_text(item, "ranked_fixture_refs")
            lookup_key = fixture_ref.casefold()
            if lookup_key in seen:
                raise ValueError("ranked_fixture_refs must not contain duplicates.")
            seen.add(lookup_key)
            normalized.append(fixture_ref)
        return tuple(normalized)


class LLMResolvedMarketChoice(BaseModel):
    """Exact SportyBet row selected by the market-resolution LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_ref: str = Field(description="Fixture ref being resolved.")
    row_id: str = Field(description="Exact row_id from the supplied SportyBet market menu.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in the row.")
    rationale: str = Field(
        min_length=1,
        max_length=1000,
        description="Short evidence-backed reason for choosing the market row.",
    )

    @field_validator("fixture_ref", "row_id", "rationale")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required text fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)


class LLMAccumulatorSlipChoice(BaseModel):
    """One accumulator slip selected by the accumulator-builder LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    slip_number: int = Field(gt=0, description="1-based slip number.")
    leg_fixture_refs: tuple[str, ...] = Field(
        min_length=2,
        description="Ordered fixture refs to include as legs in this slip.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Slip-level model confidence.")
    strategy: AccumulatorStrategy = Field(description="Risk style for the slip.")
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional model summary for the slip.",
    )

    @field_validator("leg_fixture_refs")
    @classmethod
    def validate_leg_fixture_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank or duplicate leg fixture refs."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            fixture_ref = require_non_blank_text(item, "leg_fixture_refs")
            lookup_key = fixture_ref.casefold()
            if lookup_key in seen:
                raise ValueError("leg_fixture_refs must not contain duplicates.")
            seen.add(lookup_key)
            normalized.append(fixture_ref)
        return tuple(normalized)

    @field_validator("rationale")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional model rationale text."""

        return normalize_optional_text(value)


class LLMAccumulatorPortfolio(BaseModel):
    """Complete accumulator portfolio selected by the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    slips: tuple[LLMAccumulatorSlipChoice, ...] = Field(
        min_length=1,
        description="Accumulator slips selected for the day.",
    )

    @model_validator(mode="after")
    def validate_slip_numbers(self) -> LLMAccumulatorPortfolio:
        """Require consecutive slip numbering from the model output."""

        expected = tuple(range(1, len(self.slips) + 1))
        observed = tuple(slip.slip_number for slip in self.slips)
        if observed != expected:
            raise ValueError("slip_number values must be consecutive starting at 1.")
        return self


__all__ = [
    "LLMAccumulatorPortfolio",
    "LLMAccumulatorSlipChoice",
    "LLMRankingDecision",
    "LLMResolvedMarketChoice",
]
