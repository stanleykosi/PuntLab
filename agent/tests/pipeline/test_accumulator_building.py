"""Tests for the LLM-led accumulator-building pipeline node."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from src.accumulators import AccumulatorBuilder
from src.config import MarketType, SportName
from src.pipeline.nodes import accumulator_building as accumulator_module
from src.pipeline.nodes.accumulator_building import accumulator_building_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import AccumulatorStrategy, ResolutionSource, ResolvedMarket
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown


class FakeLLM:
    """Minimal async chat model returning accumulator JSON."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def ainvoke(self, messages: object) -> object:
        del messages
        return type("FakeMessage", (), {"content": json.dumps(self.payload)})()


def build_ranked_match(index: int) -> RankedMatch:
    """Create one ranked match for accumulator tests."""

    return RankedMatch(
        fixture_ref=f"sr:match:{7100 + index}",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {index}",
        away_team=f"Away {index}",
        composite_score=0.90 - (index * 0.03),
        confidence=0.86 - (index * 0.03),
        factors=ScoreFactorBreakdown(
            form=0.78,
            h2h=0.56,
            injury_impact=0.67,
            odds_value=0.72,
            context=0.64,
            venue=0.69,
            statistical=0.62,
        ),
        recommended_market="full_time_result",
        recommended_canonical_market=MarketType.MATCH_RESULT,
        recommended_selection=f"Home {index}",
        recommended_odds=1.72,
        qualitative_summary=f"Home {index} profiles stronger.",
        rank=index,
    )


def build_resolved_market(ranked_match: RankedMatch, odds: float) -> ResolvedMarket:
    """Create one resolved market aligned with a ranked match."""

    return ResolvedMarket(
        fixture_ref=ranked_match.fixture_ref,
        sport=ranked_match.sport,
        competition=ranked_match.competition,
        home_team=ranked_match.home_team,
        away_team=ranked_match.away_team,
        market="full_time_result",
        canonical_market=MarketType.MATCH_RESULT,
        selection=ranked_match.recommended_selection or ranked_match.home_team,
        odds=odds,
        provider="sportybet",
        provider_market_name="Full Time Result",
        provider_selection_name=ranked_match.recommended_selection or ranked_match.home_team,
        provider_market_id=1,
        market_label="Full Time Result",
        resolution_source=ResolutionSource.SPORTYBET_API,
        resolved_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
    )


def build_candidate_pool() -> tuple[list[RankedMatch], list[ResolvedMarket]]:
    """Build a small resolved leg pool."""

    ranked = [build_ranked_match(index) for index in range(1, 5)]
    resolved = [
        build_resolved_market(ranked_match, odds=1.55 + (ranked_match.rank * 0.08))
        for ranked_match in ranked
    ]
    return ranked, resolved


@pytest.mark.asyncio
async def test_accumulator_building_node_uses_llm_slips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node should build slips from the exact LLM-selected fixture refs."""

    run_uuid = uuid4()
    ranked_matches, resolved_markets = build_candidate_pool()
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM(
            {
                "slips": [
                    {
                        "slip_number": 1,
                        "leg_fixture_refs": ["sr:match:7101", "sr:match:7102"],
                        "confidence": 0.74,
                        "strategy": "confident",
                        "rationale": "Two strongest resolved legs.",
                    },
                    {
                        "slip_number": 2,
                        "leg_fixture_refs": ["sr:match:7103", "sr:match:7104"],
                        "confidence": 0.61,
                        "strategy": "balanced",
                        "rationale": "A wider second angle.",
                    },
                ]
            }
        )

    monkeypatch.setattr(accumulator_module, "get_llm", fake_get_llm)

    result = await accumulator_building_node(
        PipelineState(
            run_id=str(run_uuid),
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.ACCUMULATOR_BUILDING,
            ranked_matches=ranked_matches,
            resolved_markets=resolved_markets,
            errors=["Market resolution completed."],
        ),
        builder=AccumulatorBuilder(target_count=2),
    )

    assert result["current_stage"] == PipelineStage.EXPLANATION
    assert result["errors"] == ["Market resolution completed."]
    assert len(result["accumulators"]) == 2
    first_slip = result["accumulators"][0]
    assert first_slip.run_id == run_uuid
    assert first_slip.strategy is AccumulatorStrategy.CONFIDENT
    assert [leg.fixture_ref for leg in first_slip.legs] == [
        "sr:match:7101",
        "sr:match:7102",
    ]
    assert first_slip.total_odds == pytest.approx(round(1.63 * 1.71, 3))


@pytest.mark.asyncio
async def test_accumulator_building_node_rejects_unknown_llm_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accumulator LLM cannot select fixtures outside the resolved pool."""

    ranked_matches, resolved_markets = build_candidate_pool()
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM(
            {
                "slips": [
                    {
                        "slip_number": 1,
                        "leg_fixture_refs": ["sr:match:7101", "sr:match:9999"],
                        "confidence": 0.74,
                        "strategy": "confident",
                        "rationale": "Includes unknown fixture.",
                    }
                ]
            }
        )

    monkeypatch.setattr(accumulator_module, "get_llm", fake_get_llm)

    with pytest.raises(ValueError, match="unknown or unresolved fixture"):
        await accumulator_building_node(
            PipelineState(
                run_id="run-2026-04-05-main",
                run_date=date(2026, 4, 5),
                started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
                current_stage=PipelineStage.ACCUMULATOR_BUILDING,
                ranked_matches=ranked_matches,
                resolved_markets=resolved_markets,
            )
        )
