"""Tests for the LLM-led ranking pipeline node."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes import ranking as ranking_module
from src.pipeline.nodes.ranking import ranking_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchScore, ScoreFactorBreakdown


class FakeLLM:
    """Minimal async chat model returning ranking JSON."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def ainvoke(self, messages: object) -> object:
        del messages
        return type("FakeMessage", (), {"content": json.dumps(self.payload)})()


def build_match_score(fixture_ref: str, score: float) -> MatchScore:
    """Create one scored fixture for ranking tests."""

    suffix = fixture_ref.rsplit(":", maxsplit=1)[-1]
    return MatchScore(
        fixture_ref=fixture_ref,
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=f"Home {suffix}",
        away_team=f"Away {suffix}",
        composite_score=score,
        confidence=0.74,
        factors=ScoreFactorBreakdown(
            form=0.74,
            h2h=0.58,
            injury_impact=0.62,
            odds_value=0.66,
            context=0.71,
            venue=0.64,
            statistical=0.69,
        ),
        recommended_market="full_time_result",
        recommended_canonical_market=MarketType.MATCH_RESULT,
        recommended_selection=f"Home {suffix}",
        recommended_odds=1.82,
        qualitative_summary="The stronger home side profiles better.",
    )


@pytest.mark.asyncio
async def test_ranking_node_uses_llm_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ranking node should preserve the exact fixture order returned by the LLM."""

    scores = [
        build_match_score("sr:match:7001", 0.78),
        build_match_score("sr:match:7002", 0.72),
        build_match_score("sr:match:7003", 0.84),
    ]
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM(
            {"ranked_fixture_refs": ["sr:match:7003", "sr:match:7001", "sr:match:7002"]}
        )

    monkeypatch.setattr(
        ranking_module,
        "get_llm",
        fake_get_llm,
    )

    result = await ranking_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.RANKING,
            match_scores=scores,
            errors=["Upstream scoring warning."],
        )
    )

    assert result["current_stage"] == PipelineStage.MARKET_RESOLUTION
    assert result["errors"] == ["Upstream scoring warning."]
    ranked_matches = result["ranked_matches"]
    assert [match.fixture_ref for match in ranked_matches] == [
        "sr:match:7003",
        "sr:match:7001",
        "sr:match:7002",
    ]
    assert [match.rank for match in ranked_matches] == [1, 2, 3]


@pytest.mark.asyncio
async def test_ranking_node_rejects_missing_llm_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ranking LLM must return exactly the actionable fixture set."""

    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM({"ranked_fixture_refs": ["sr:match:7001"]})

    monkeypatch.setattr(ranking_module, "get_llm", fake_get_llm)

    with pytest.raises(ValueError, match="did not match scored fixture set"):
        await ranking_node(
            PipelineState(
                run_id="run-2026-04-04-main",
                run_date=date(2026, 4, 4),
                started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
                current_stage=PipelineStage.RANKING,
                match_scores=[
                    build_match_score("sr:match:7001", 0.78),
                    build_match_score("sr:match:7002", 0.72),
                ],
            )
        )
