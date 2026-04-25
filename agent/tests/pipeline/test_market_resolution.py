"""Tests for the LLM-led market-resolution pipeline node."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes import market_resolution as market_resolution_module
from src.pipeline.nodes.market_resolution import market_resolution_node
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import build_odds_market_catalog
from src.schemas.accumulators import ResolutionSource
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds


class FakeLLM:
    """Minimal async chat model returning resolution JSON."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def ainvoke(self, messages: object) -> object:
        del messages
        return type("FakeMessage", (), {"content": json.dumps(self.payload)})()


def build_fixture() -> NormalizedFixture:
    """Create one fixture for market-resolution tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 5, 19, 0, tzinfo=UTC),
        source_provider="sportybet",
        source_id="7001",
        country="England",
    )


def build_ranked_match(fixture: NormalizedFixture) -> RankedMatch:
    """Create one ranked recommendation."""

    return RankedMatch(
        fixture_ref=fixture.get_fixture_ref(),
        sport=fixture.sport,
        competition=fixture.competition,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        composite_score=0.76,
        confidence=0.71,
        factors=ScoreFactorBreakdown(
            form=0.8,
            h2h=0.58,
            injury_impact=0.63,
            odds_value=0.68,
            context=0.7,
            venue=0.65,
            statistical=0.61,
        ),
        recommended_market="full_time_result",
        recommended_market_label="Full Time Result",
        recommended_canonical_market=MarketType.MATCH_RESULT,
        recommended_selection="Arsenal",
        recommended_odds=1.84,
        qualitative_summary="Arsenal profile as the stronger side.",
        rank=1,
    )


def build_catalog_rows(fixture: NormalizedFixture) -> tuple[NormalizedOdds, ...]:
    """Create two concrete SportyBet rows for the LLM to choose from."""

    return (
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=MarketType.MATCH_RESULT,
            selection="Home",
            odds=1.84,
            provider="sportybet",
            provider_market_name="Full Time Result",
            provider_selection_name="Arsenal",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={"market_group_name": "Main"},
            last_updated=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=MarketType.OVER_UNDER_25,
            selection="Over",
            odds=1.94,
            provider="sportybet",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Over 2.5",
            provider_market_id=18,
            line=2.5,
            period="match",
            participant_scope="match",
            raw_metadata={"market_group_name": "Goals"},
            last_updated=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
        ),
    )


@pytest.mark.asyncio
async def test_market_resolution_node_uses_llm_selected_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node should convert the exact LLM-selected SportyBet row into ResolvedMarket."""

    fixture = build_fixture()
    ranked_match = build_ranked_match(fixture)
    catalog_rows = build_catalog_rows(fixture)
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM(
            {
                "fixture_ref": fixture.get_fixture_ref(),
                "row_id": "row_1",
                "confidence": 0.72,
                "rationale": "Best matches the scored recommendation.",
            }
        )

    monkeypatch.setattr(market_resolution_module, "get_llm", fake_get_llm)

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[fixture],
            ranked_matches=[ranked_match],
            odds_market_catalog=build_odds_market_catalog(
                catalog_rows,
                sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
            ),
            errors=["Ranking completed."],
        )
    )

    assert result["current_stage"] == PipelineStage.ACCUMULATOR_BUILDING
    assert result["errors"] == ["Ranking completed."]
    assert len(result["resolved_markets"]) == 1
    resolved = result["resolved_markets"][0]
    assert resolved.fixture_ref == fixture.get_fixture_ref()
    assert resolved.market == "full_time_result"
    assert resolved.canonical_market is MarketType.MATCH_RESULT
    assert resolved.provider_selection_name == "Arsenal"
    assert resolved.resolution_source is ResolutionSource.SPORTYBET_API


@pytest.mark.asyncio
async def test_market_resolution_node_rejects_unknown_llm_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolution LLM cannot invent row IDs."""

    fixture = build_fixture()
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return FakeLLM(
            {
                "fixture_ref": fixture.get_fixture_ref(),
                "row_id": "row_99",
                "confidence": 0.72,
                "rationale": "Invented row.",
            }
        )

    monkeypatch.setattr(market_resolution_module, "get_llm", fake_get_llm)

    with pytest.raises(ValueError, match="unknown row_id"):
        await market_resolution_node(
            PipelineState(
                run_id="run-2026-04-05-main",
                run_date=date(2026, 4, 5),
                started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
                current_stage=PipelineStage.MARKET_RESOLUTION,
                fixtures=[fixture],
                ranked_matches=[build_ranked_match(fixture)],
                odds_market_catalog=build_odds_market_catalog(
                    build_catalog_rows(fixture),
                    sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
                ),
            )
        )
