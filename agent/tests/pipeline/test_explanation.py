"""Tests for PuntLab's explanation pipeline node.

Purpose: verify that accumulator slips are enriched with delivery-ready leg
and slip rationales while explanation failures degrade gracefully.
Scope: unit tests for `src.pipeline.nodes.explanation`.
Dependencies: pytest plus lightweight LLM stubs and the canonical
accumulator, pipeline-state, and explanation schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.accumulators import AccumulatorBuilder
from src.config import MarketType, SportName
from src.llm import AccumulatorRationale, AllProvidersFailedError, LegRationale
from src.pipeline.nodes.explanation import explanation_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorSlip,
    ResolutionSource,
)


class QueueStructuredLLM:
    """Async structured-output runner returning queued explanation payloads."""

    def __init__(self, results: list[object]) -> None:
        """Persist queued structured payloads returned by `ainvoke()`."""

        self._results = list(results)
        self.invocations: list[object] = []
        self.error: Exception | None = None

    async def ainvoke(self, prompt_value: object) -> object:
        """Return the next queued payload or raise the configured error."""

        self.invocations.append(prompt_value)
        if self.error is not None:
            raise self.error
        if not self._results:
            raise AssertionError("QueueStructuredLLM ran out of explanation payloads.")
        return self._results.pop(0)


class FakeLLM:
    """Minimal chat-model stub exposing `with_structured_output()`."""

    def __init__(self, runner: QueueStructuredLLM) -> None:
        """Persist the structured-output runner used by this fake LLM."""

        self.runner = runner
        self.requested_schema: type[object] | None = None

    def with_structured_output(self, schema: type[object]) -> QueueStructuredLLM:
        """Record the requested schema and return the queued runner."""

        self.requested_schema = schema
        return self.runner


def build_leg(
    *,
    leg_number: int,
    fixture_ref: str,
    home_team: str,
    away_team: str,
    selection: str,
    confidence: float,
    odds: float,
) -> AccumulatorLeg:
    """Create a canonical accumulator leg used by explanation-node tests."""

    return AccumulatorLeg(
        leg_number=leg_number,
        fixture_ref=fixture_ref,
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=home_team,
        away_team=away_team,
        market=MarketType.MATCH_RESULT,
        selection=selection,
        odds=odds,
        provider="sportybet",
        confidence=confidence,
        resolution_source=ResolutionSource.SPORTYBET_API,
        market_label="Full Time Result",
        rationale=f"{home_team} enters with the cleaner form profile.",
    )


def build_accumulator() -> AccumulatorSlip:
    """Create a canonical accumulator slip ready for explanation-node tests."""

    legs = (
        build_leg(
            leg_number=1,
            fixture_ref="sr:match:8101",
            home_team="Arsenal",
            away_team="Chelsea",
            selection="home",
            confidence=0.78,
            odds=1.68,
        ),
        build_leg(
            leg_number=2,
            fixture_ref="sr:match:8102",
            home_team="Liverpool",
            away_team="Brighton",
            selection="home",
            confidence=0.73,
            odds=1.74,
        ),
    )
    builder = AccumulatorBuilder(target_count=1)
    confidence = builder.calculate_acca_confidence(legs)
    return AccumulatorSlip(
        slip_date=date(2026, 4, 5),
        slip_number=1,
        legs=legs,
        total_odds=round(legs[0].odds * legs[1].odds, 3),
        leg_count=2,
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_explanation_node_generates_leg_and_slip_rationales() -> None:
    """The explanation node should produce delivery-ready explained accumulators."""

    leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale=(
                    "Arsenal's recent control and shot quality make the home win "
                    "look justified."
                ),
                key_risk="Chelsea can still punish transitions.",
            ),
            LegRationale(
                fixture_ref="sr:match:8102",
                rationale=(
                    "Liverpool's home edge and tempo profile still point toward "
                    "the stronger result."
                ),
                key_risk=None,
            ),
        ]
    )
    accumulator_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale=(
                    "This slip leans on two stronger home-form angles with "
                    "manageable prices."
                ),
                shared_risk="Both legs still depend on favorites converting pressure into results.",
            )
        ]
    )

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 30, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
            errors=["Accumulator building completed."],
        ),
        leg_llm=FakeLLM(leg_runner),  # type: ignore[arg-type]
        accumulator_llm=FakeLLM(accumulator_runner),  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["errors"] == ["Accumulator building completed."]
    assert len(result["explained_accumulators"]) == 1
    explained = result["explained_accumulators"][0]
    assert explained.legs[0].rationale == (
        "Arsenal's recent control and shot quality make the home win look justified. "
        "Risk: Chelsea can still punish transitions."
    )
    assert explained.legs[1].rationale == (
        "Liverpool's home edge and tempo profile still point toward the stronger result."
    )
    assert explained.rationale == (
        "This slip leans on two stronger home-form angles with manageable prices. "
        "Shared risk: Both legs still depend on favorites converting pressure into results."
    )


@pytest.mark.asyncio
async def test_explanation_node_falls_back_when_llm_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider setup failures should not block deterministic explanation output."""

    async def fail_get_llm(task: str) -> object:
        """Raise the canonical provider failure for the requested explanation task."""

        raise AllProvidersFailedError(
            task=task,
            attempted_providers=("openai",),
            reasons=("openai skipped because its API key is not configured.",),
        )

    monkeypatch.setattr("src.pipeline.nodes.explanation.get_llm", fail_get_llm)

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 45, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        )
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert len(result["explained_accumulators"]) == 1
    explained = result["explained_accumulators"][0]
    assert explained.legs[0].rationale.startswith(
        "Arsenal enters with the cleaner form profile."
    )
    assert explained.rationale.startswith("This slip leans on 2 steady legs")
    assert len(result["errors"]) == 2
    assert "Unable to construct an LLM for task 'leg_rationale'" in result["errors"][0]
    assert "Unable to construct an LLM for task 'accumulator_rationale'" in result["errors"][1]
