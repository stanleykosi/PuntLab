"""Structured LLM output schemas for PuntLab's LangChain integrations.

Purpose: provide the single current-state schema surface used with
`with_structured_output()` across research, qualitative scoring, and
explanation stages.
Scope: re-export the canonical shared `MatchContext` and `QualitativeScore`
contracts plus define the slip-level explanation payload.
Dependencies: `src.schemas.analysis` for the shared qualitative models and
`src.schemas.common` for text normalization helpers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.analysis import MatchContext, QualitativeScore
from src.schemas.common import normalize_optional_text, require_non_blank_text


class AccumulatorRationale(BaseModel):
    """Structured explanation output for a full accumulator slip.

    Inputs:
        A prompt grounded in the slip's legs, confidence posture, and shared
        portfolio-style risks.

    Outputs:
        A delivery-ready accumulator summary and an optional biggest-risk note
        that downstream formatting can surface separately when useful.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    slip_number: int | None = Field(
        default=None,
        gt=0,
        description="Optional 1-based slip number echoed back for tracing.",
    )
    rationale: str = Field(
        max_length=1200,
        description="Compact 2-3 sentence explanation for the accumulator.",
    )
    shared_risk: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional concise note on the slip's biggest shared risk.",
    )

    @field_validator("shared_risk")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional shared-risk text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        """Reject blank accumulator rationale text after normalization."""

        return require_non_blank_text(value, "rationale")


__all__ = [
    "AccumulatorRationale",
    "MatchContext",
    "QualitativeScore",
]
