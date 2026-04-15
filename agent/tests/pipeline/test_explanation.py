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

    def __init__(
        self,
        runner: QueueStructuredLLM,
        *,
        json_mode_runner: QueueStructuredLLM | None = None,
        openai_api_base: str | None = None,
        plain_result: object | None = None,
    ) -> None:
        """Persist the structured-output runner used by this fake LLM."""

        self.runner = runner
        self.json_mode_runner = json_mode_runner
        self.requested_schema: type[object] | None = None
        self.requested_methods: list[str] = []
        self.openai_api_base = openai_api_base
        self.plain_result = plain_result

    def with_structured_output(
        self,
        schema: type[object],
        **kwargs: object,
    ) -> QueueStructuredLLM:
        """Record the requested schema and return the queued runner."""

        method = kwargs.get("method")
        self.requested_schema = schema
        self.requested_methods.append(str(method) if method is not None else "default")
        if method == "json_mode" and self.json_mode_runner is not None:
            return self.json_mode_runner
        return self.runner

    async def ainvoke(self, _prompt_value: object) -> object:
        """Return plain output for OpenRouter text-mode parsing tests."""

        if self.plain_result is None:
            raise RuntimeError("FakeLLM.plain_result must be configured before ainvoke().")
        return self.plain_result


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


@pytest.mark.asyncio
async def test_explanation_node_emits_progress_updates_for_slips_and_legs() -> None:
    """Explanation should emit structured progress callbacks throughout execution."""

    leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale="Arsenal's control profile remains stronger at home.",
                key_risk=None,
            ),
            LegRationale(
                fixture_ref="sr:match:8102",
                rationale="Liverpool's tempo and chance volume stay favorable.",
                key_risk=None,
            ),
        ]
    )
    accumulator_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale="Both legs align with stable home-side confidence.",
                shared_risk=None,
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
        leg_llm=FakeLLM(leg_runner),  # type: ignore[arg-type]
        accumulator_llm=FakeLLM(accumulator_runner),  # type: ignore[arg-type]
        progress_callback=lambda payload: updates.append(dict(payload)),
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert updates[0]["event"] == "stage_started"
    assert updates[-1]["event"] == "stage_completed"
    assert any(update["event"] == "slip_started" for update in updates)
    assert any(update["event"] == "leg_started" for update in updates)
    assert any(update["event"] == "leg_completed" for update in updates)
    assert any(update["event"] == "slip_rationale_started" for update in updates)
    assert any(update["event"] == "slip_completed" for update in updates)


@pytest.mark.asyncio
async def test_explanation_node_normalizes_text_field_payloads() -> None:
    """Non-canonical dict payloads with `text` should still validate cleanly."""

    leg_runner = QueueStructuredLLM(
        [
            {"fixture_ref": "sr:match:8101", "text": "Arsenal are favored at home."},
            {"fixture_ref": "sr:match:8102", "text": "Liverpool remain the stronger side."},
        ]
    )
    accumulator_runner = QueueStructuredLLM(
        [
            {"slip_number": 1, "text": "Both legs align with stronger home profiles."}
        ]
    )

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-normalized-payload",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        ),
        leg_llm=FakeLLM(leg_runner),  # type: ignore[arg-type]
        accumulator_llm=FakeLLM(accumulator_runner),  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["errors"] == []
    assert len(result["explained_accumulators"]) == 1
    assert result["explained_accumulators"][0].legs[0].rationale.startswith(
        "Arsenal are favored at home."
    )


@pytest.mark.asyncio
async def test_explanation_node_disables_leg_llm_after_rate_limit() -> None:
    """A 429 leg failure should disable remaining leg LLM calls in the stage."""

    leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale="Should not be consumed because first call rate limits.",
                key_risk=None,
            )
        ]
    )
    leg_runner.error = RuntimeError("Error code: 429 - provider rate-limited")
    accumulator_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale="Fallback-friendly slip summary.",
                shared_risk=None,
            )
        ]
    )

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-rate-limit",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 5, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        ),
        leg_llm=FakeLLM(leg_runner),  # type: ignore[arg-type]
        accumulator_llm=FakeLLM(accumulator_runner),  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert any(
        "Leg rationale LLM was disabled for the remaining slips due to upstream rate limiting."
        in message
        for message in result["errors"]
    )
    assert any(
        "Leg explanation failed for" in message and "LLM provider rate limit reached."
        in message
        for message in result["errors"]
    )
    # First leg invoked the LLM, second leg fell back with the LLM disabled.
    assert len(leg_runner.invocations) == 1


@pytest.mark.asyncio
async def test_explanation_node_uses_json_schema_mode_for_openrouter_models() -> None:
    """OpenRouter explanation calls should prefer JSON-schema structured output."""

    leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale="Arsenal remain favorites.",
                key_risk=None,
            ),
            LegRationale(
                fixture_ref="sr:match:8102",
                rationale="Liverpool hold the stronger profile.",
                key_risk=None,
            ),
        ]
    )
    accumulator_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale="Combined profile remains coherent.",
                shared_risk=None,
            )
        ]
    )
    leg_llm = FakeLLM(
        leg_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    accumulator_llm = FakeLLM(
        accumulator_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-openrouter-text",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 10, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        ),
        leg_llm=leg_llm,  # type: ignore[arg-type]
        accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["errors"] == []
    assert leg_llm.requested_methods == ["json_schema", "json_schema"]
    assert accumulator_llm.requested_methods == ["json_schema"]
    assert len(leg_runner.invocations) == 2
    assert len(accumulator_runner.invocations) == 1


@pytest.mark.asyncio
async def test_explanation_node_falls_back_to_json_mode_when_openrouter_json_schema_fails() -> None:
    """OpenRouter explanation should retry with JSON mode when schema mode is unsupported."""

    json_schema_leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale="unused",
                key_risk=None,
            )
        ]
    )
    json_schema_leg_runner.error = RuntimeError(
        "Model does not support response_format json_schema for this route."
    )
    json_mode_leg_runner = QueueStructuredLLM(
        [
            LegRationale(
                fixture_ref="sr:match:8101",
                rationale="Arsenal hold the stronger form edge at home.",
                key_risk=None,
            ),
            LegRationale(
                fixture_ref="sr:match:8102",
                rationale="Liverpool still carry superior chance quality trends.",
                key_risk=None,
            ),
        ]
    )

    json_schema_acc_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale="unused",
                shared_risk=None,
            )
        ]
    )
    json_schema_acc_runner.error = RuntimeError(
        "Model does not support response_format json_schema for this route."
    )
    json_mode_acc_runner = QueueStructuredLLM(
        [
            AccumulatorRationale(
                slip_number=1,
                rationale="Both legs align around stronger home-side confidence signals.",
                shared_risk=None,
            )
        ]
    )

    leg_llm = FakeLLM(
        json_schema_leg_runner,
        json_mode_runner=json_mode_leg_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    accumulator_llm = FakeLLM(
        json_schema_acc_runner,
        json_mode_runner=json_mode_acc_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    result = await explanation_node(
        PipelineState(
            run_id="run-2026-04-05-openrouter-json-mode-fallback",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 15, tzinfo=UTC),
            current_stage=PipelineStage.EXPLANATION,
            accumulators=[build_accumulator()],
        ),
        leg_llm=leg_llm,  # type: ignore[arg-type]
        accumulator_llm=accumulator_llm,  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.APPROVAL
    assert result["errors"] == []
    assert leg_llm.requested_methods == [
        "json_schema",
        "json_mode",
        "json_schema",
        "json_mode",
    ]
    assert accumulator_llm.requested_methods == ["json_schema", "json_mode"]
