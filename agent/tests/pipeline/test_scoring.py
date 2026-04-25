"""Tests for the LLM-led scoring pipeline node."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes import scoring as scoring_module
from src.pipeline.nodes.scoring import scoring_node
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import build_odds_market_catalog
from src.schemas.analysis import MatchContext
from src.schemas.fixture_details import FixtureDetails, FixtureDetailSection
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds


class FakeLLM:
    """Minimal async chat model returning a JSON object as message content."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[object] = []

    async def ainvoke(self, messages: object) -> object:
        self.calls.append(messages)
        return type("FakeMessage", (), {"content": json.dumps(self.payload)})()


def build_fixture() -> NormalizedFixture:
    """Create one fixture for scoring-node tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 0, tzinfo=UTC),
        source_provider="sportybet",
        source_id="61301159",
        country="England",
        venue="Emirates Stadium",
    )


def build_odds_row(fixture_ref: str) -> NormalizedOdds:
    """Create one SportyBet odds row available to the LLM prompt."""

    return NormalizedOdds(
        fixture_ref=fixture_ref,
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
        last_updated=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
    )


def build_context(fixture_ref: str) -> MatchContext:
    """Create one LLM research context required by scoring."""

    return MatchContext(
        fixture_ref=fixture_ref,
        fixture_detail_summary="SportyBet widgets show Arsenal with the cleaner setup.",
        tactical_context="Arsenal probable XI is stable.",
        statistical_context="Arsenal carry the stronger shot-volume indicators.",
        availability_context="No major home absence captured.",
        market_context="Home win is the shortest main-market price.",
        supplemental_news_context="RSS supports Arsenal's home form.",
        qualitative_score=0.74,
        data_sources=("SportyBet fixture-page widgets",),
    )


def build_fixture_details(fixture: NormalizedFixture) -> FixtureDetails:
    """Create raw SportyBet fixture details supplied to market scoring."""

    return FixtureDetails(
        fixture_ref=fixture.get_fixture_ref(),
        fixture_url="https://www.sportybet.com/ng/sport/football/england/premier-league/Arsenal_vs_Chelsea/sr:match:61301159",
        event_id=fixture.sportradar_id or "sr:match:61301159",
        match_id="61301159",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        widget_loader_status="loaded",
        sections=(
            FixtureDetailSection(
                widget_key="statistics",
                widget_type="match.statistics",
                status="mounted",
                content_lines=("Shots: Arsenal 14.2 | Chelsea 10.1",),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_scoring_node_uses_llm_json_for_market_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoring node should validate the LLM's JSON MatchScore response."""

    fixture = build_fixture()
    odds_row = build_odds_row(fixture.get_fixture_ref())
    llm = FakeLLM(
        {
            "fixture_ref": fixture.get_fixture_ref(),
            "sport": "soccer",
            "competition": "Premier League",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "composite_score": 0.81,
            "confidence": 0.76,
            "factors": {
                "form": 0.8,
                "h2h": 0.55,
                "injury_impact": 0.7,
                "odds_value": 0.72,
                "context": 0.78,
                "venue": 0.74,
                "statistical": 0.68,
            },
            "recommended_market": "full_time_result",
            "recommended_market_label": "Full Time Result",
            "recommended_canonical_market": "1x2",
            "recommended_selection": "Arsenal",
            "recommended_odds": 1.84,
            "recommended_line": None,
            "qualitative_summary": "Arsenal have the clearer pre-match edge.",
        }
    )
    async def fake_get_llm(task: str) -> FakeLLM:
        del task
        return llm

    monkeypatch.setattr(scoring_module, "get_llm", fake_get_llm)

    result = await scoring_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.SCORING,
            fixtures=[fixture],
            odds_market_catalog=build_odds_market_catalog(
                (odds_row,),
                sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
            ),
            match_contexts=[build_context(fixture.get_fixture_ref())],
            fixture_details=[build_fixture_details(fixture)],
            errors=["Earlier-stage warning."],
        )
    )

    assert result["current_stage"] == PipelineStage.RANKING
    assert result["errors"] == ["Earlier-stage warning."]
    assert len(result["match_scores"]) == 1
    score = result["match_scores"][0]
    assert score.fixture_ref == fixture.get_fixture_ref()
    assert score.recommended_market == "full_time_result"
    assert score.recommended_canonical_market is MarketType.MATCH_RESULT
    assert score.recommended_selection == "Arsenal"
    assert score.recommended_odds == pytest.approx(1.84)
    assert llm.calls


@pytest.mark.asyncio
async def test_scoring_node_fails_fast_without_research_context() -> None:
    """Scoring should not fabricate a fallback context."""

    fixture = build_fixture()
    with pytest.raises(RuntimeError, match="requires research context"):
        await scoring_node(
            PipelineState(
                run_id="run-2026-04-04-main",
                run_date=date(2026, 4, 4),
                started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
                current_stage=PipelineStage.SCORING,
                fixtures=[fixture],
                match_contexts=[],
            )
        )
