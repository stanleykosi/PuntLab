"""Tests for PuntLab's structured LLM output schemas.

Purpose: verify the LLM namespace reuses the canonical shared qualitative
schemas and validate the new explanation-specific output contracts.
Scope: unit tests for `src.llm.schemas`.
Dependencies: the shared analysis and LLM schema modules.
"""

from __future__ import annotations

from src.llm.schemas import (
    AccumulatorRationale,
)
from src.llm.schemas import (
    MatchContext as LLMMatchContext,
)
from src.llm.schemas import (
    QualitativeScore as LLMQualitativeScore,
)
from src.schemas.analysis import (
    MatchContext as AnalysisMatchContext,
)
from src.schemas.analysis import (
    QualitativeScore as AnalysisQualitativeScore,
)


def test_llm_schema_namespace_reuses_canonical_analysis_models() -> None:
    """The LLM package should not fork duplicate qualitative schema classes."""

    assert LLMMatchContext is AnalysisMatchContext
    assert LLMQualitativeScore is AnalysisQualitativeScore


def test_accumulator_rationale_supports_optional_shared_risk() -> None:
    """Accumulator rationales should keep the main explanation while trimming risk text."""

    rationale = AccumulatorRationale(
        slip_number=3,
        rationale=(
            "This slip leans on two strong home-form edges and one totals angle in a fast-paced "
            "matchup."
        ),
        shared_risk=" ",
    )

    dumped = rationale.model_dump(mode="json")

    assert dumped["slip_number"] == 3
    assert dumped["shared_risk"] is None
    assert "home-form edges" in dumped["rationale"]
