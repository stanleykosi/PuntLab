"""Tests for PuntLab's slip-only LLM explanation pipeline node."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.accumulators import AccumulatorBuilder
from src.config import MarketType, SportName
from src.llm import AllProvidersFailedError
from src.pipeline.nodes.explanation import explanation_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import (
    AccumulatorLeg,
    AccumulatorSlip,
    ResolutionSource,
)


class QueueJsonLLM:
    """Minimal async chat-model stub returning queued JSON text responses."""

    def __init__(self, responses: list[str]) -> None:
        """Persist queued plain-text responses for `ainvoke()`."""

        self._responses = list(responses)
        self.invocations: list[object] = []
        self.error: Exception | None = None

    async def ainvoke(self, prompt_value: object) -> str:
        """Return the next queued response or raise the configured error."""

        self.invocations.append(prompt_value)
        if self.error is not None:
            raise self.error
        if not self._responses:
            raise AssertionError("QueueJsonLLM ran out of JSON responses.")
        return self._responses.pop(0)


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
async def test_explanation_node_generates_only_slip_rationale_from_json() -> None:
    """The explanation node should call the LLM once per accumulator slip."""

    accumulator_llm = QueueJsonLLM(
        [
            (
                '{"slip_number":1,"rationale":"This slip leans on two stronger '
                'home-form angles at manageable prices.","shared_risk":"Both legs '
                'still need favorites to convert pressure."}'
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
        accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["errors"] == ["Accumulator building completed."]
    explained = result["explained_accumulators"][0]
    assert explained.legs[0].rationale == "Arsenal enters with the cleaner form profile."
    assert explained.legs[1].rationale == "Liverpool enters with the cleaner form profile."
    assert explained.rationale == (
        "This slip leans on two stronger home-form angles at manageable prices. "
        "Shared risk: Both legs still need favorites to convert pressure."
    )
    assert len(accumulator_llm.invocations) == 1


@pytest.mark.asyncio
async def test_explanation_node_fails_fast_when_llm_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider setup failures should stop the slip-only explanation stage."""

    async def fail_get_llm(task: str) -> object:
        """Raise the canonical provider failure for the requested task."""

        raise AllProvidersFailedError(
            task=task,
            attempted_providers=("openai",),
            reasons=("openai skipped because its API key is not configured.",),
        )

    monkeypatch.setattr("src.pipeline.nodes.explanation.get_llm", fail_get_llm)

    with pytest.raises(AllProvidersFailedError, match="accumulator_rationale"):
        await explanation_node(
            PipelineState(
                run_id="run-2026-04-05-main",
                run_date=date(2026, 4, 5),
                started_at=datetime(2026, 4, 5, 7, 45, tzinfo=UTC),
                current_stage=PipelineStage.EXPLANATION,
                accumulators=[build_accumulator()],
            )
        )


@pytest.mark.asyncio
async def test_explanation_node_emits_slip_progress_updates() -> None:
    """Explanation progress should track slip-level LLM calls only."""

    accumulator_llm = QueueJsonLLM(
        [
            (
                '{"slip_number":1,"rationale":"Both legs align with stable home-side '
                'confidence.","shared_risk":null}'
            )
        ]
    )
    updates: list[dict[str, object]] = []

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-progress",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 55, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        ),
        accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
        progress_callback=lambda payload: updates.append(dict(payload)),
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert updates[0]["event"] == "stage_started"
    assert updates[-1]["event"] == "stage_completed"
    assert any(update["event"] == "slip_started" for update in updates)
    assert any(update["event"] == "slip_rationale_started" for update in updates)
    assert any(update["event"] == "slip_completed" for update in updates)
    assert not any(str(update["event"]).startswith("leg_") for update in updates)


@pytest.mark.asyncio
async def test_explanation_node_fails_fast_on_invalid_llm_json() -> None:
    """Malformed slip explanation output should not fall back to canned text."""

    accumulator_llm = QueueJsonLLM(["not json"])

    with pytest.raises(ValueError, match="valid JSON object"):
        await explanation_node(
            PipelineState(
                run_id="run-2026-04-05-invalid-json",
                run_date=date(2026, 4, 5),
                started_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
                current_stage=PipelineStage.EXPLANATION,
                accumulators=[build_accumulator()],
            ),
            accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
        )

    assert len(accumulator_llm.invocations) == 1


@pytest.mark.asyncio
async def test_explanation_node_fails_fast_on_rate_limit() -> None:
    """Upstream accumulator-rationale errors should bubble."""

    accumulator_llm = QueueJsonLLM([])
    accumulator_llm.error = RuntimeError("Error code: 429 - provider rate-limited")

    with pytest.raises(RuntimeError, match="429"):
        await explanation_node(
            PipelineState(
                run_id="run-2026-04-05-rate-limit",
                run_date=date(2026, 4, 5),
                started_at=datetime(2026, 4, 5, 8, 5, tzinfo=UTC),
                current_stage=PipelineStage.EXPLANATION,
                accumulators=[build_accumulator()],
            ),
            accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
        )

    assert len(accumulator_llm.invocations) == 1
