"""Tests for the LLM-led research pipeline node."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from src.config import SportName
from src.llm import AllProvidersFailedError
from src.pipeline.nodes import research as research_module
from src.pipeline.nodes.research import research_node
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.odds_mapping import build_odds_market_catalog
from src.schemas.fixture_details import FixtureDetails, FixtureDetailSection
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds


class FakeLLM:
    """Minimal async chat model returning research JSON."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[object] = []

    async def ainvoke(self, messages: object) -> object:
        self.calls.append(messages)
        return type("FakeMessage", (), {"content": json.dumps(self.payload)})()


class EmptyTavilyProvider:
    """Test double that performs no extra research enrichment."""

    async def search_match_news(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int,
    ) -> tuple[NewsArticle, ...]:
        del fixture, max_results
        return ()


def build_fixture() -> NormalizedFixture:
    """Create one fixture for research tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 0, tzinfo=UTC),
        source_provider="sportybet",
        source_id="7001",
        country="England",
        venue="Emirates Stadium",
    )


def build_article(fixture_ref: str) -> NewsArticle:
    """Create one fixture-bound article."""

    return NewsArticle(
        headline="Arsenal carry strong home form into London clash",
        url="https://example.com/arsenal-home-form",
        published_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        source="BBC Sport",
        source_provider="rss",
        summary="Arsenal arrive in better recent form.",
        sport=SportName.SOCCER,
        competition="Premier League",
        teams=("Arsenal", "Chelsea"),
        fixture_ref=fixture_ref,
        relevance_score=0.92,
    )


def build_fixture_details(fixture: NormalizedFixture) -> FixtureDetails:
    """Create SportyBet widget details for research prompt grounding."""

    return FixtureDetails(
        fixture_ref=fixture.get_fixture_ref(),
        fixture_url="https://www.sportybet.com/ng/sport/football/england/premier-league/Arsenal_vs_Chelsea/sr:match:7001",
        event_id=fixture.sportradar_id or "sr:match:7001",
        match_id="7001",
        fetched_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        widget_loader_status="loaded",
        sections=(
            FixtureDetailSection(
                widget_key="lineups",
                widget_type="match.lineups",
                status="mounted",
                content_lines=("Arsenal probable XI unchanged",),
            ),
            FixtureDetailSection(
                widget_key="statistics",
                widget_type="match.statistics",
                status="mounted",
                content_lines=("Arsenal average shots 14.2 | Chelsea 10.1",),
            ),
        ),
    )


def build_odds_row(fixture_ref: str) -> NormalizedOdds:
    """Create one SportyBet market row for prompt grounding."""

    return NormalizedOdds(
        fixture_ref=fixture_ref,
        market=None,
        selection="Home",
        odds=1.83,
        provider="sportybet",
        provider_market_name="1X2",
        provider_selection_name="Home",
        provider_market_id=1,
        period="match",
        participant_scope="match",
        raw_metadata={
            "market_group_id": "1001",
            "market_group_name": "Main",
            "event_total_market_size": 42,
        },
    )


@pytest.mark.asyncio
async def test_research_node_uses_prompt_level_json() -> None:
    """Research should validate JSON returned directly by the supplied LLM."""

    fixture = build_fixture()
    llm = FakeLLM(
        {
            "fixture_detail_summary": "SportyBet widgets point to Arsenal continuity.",
            "tactical_context": "Arsenal probable XI is unchanged.",
            "statistical_context": "Arsenal show the stronger shot profile.",
            "availability_context": None,
            "market_context": None,
            "supplemental_news_context": "RSS also notes Arsenal home form.",
            "qualitative_score": 0.69,
            "data_sources": ["ignored-by-node"],
        }
    )

    result = await research_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.RESEARCH,
            fixtures=[fixture],
            fixture_details=[build_fixture_details(fixture)],
            news_articles=[build_article(fixture.get_fixture_ref())],
            odds_market_catalog=build_odds_market_catalog(
                (build_odds_row(fixture.get_fixture_ref()),),
                sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
            ),
        ),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=EmptyTavilyProvider(),  # type: ignore[arg-type]
        batch_size=1,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert result["errors"] == []
    assert len(result["match_contexts"]) == 1
    context = result["match_contexts"][0]
    assert context.fixture_ref == fixture.get_fixture_ref()
    assert context.data_sources == ("SportyBet fixture-page widgets", "BBC Sport")
    assert context.qualitative_score == pytest.approx(0.69)
    assert context.fixture_detail_summary == "SportyBet widgets point to Arsenal continuity."
    assert llm.calls


@pytest.mark.asyncio
async def test_research_node_fails_fast_when_provider_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing LLM providers should raise instead of creating neutral fallback context."""

    async def fake_get_llm(task: str = "default") -> FakeLLM:
        raise AllProvidersFailedError(
            task,
            attempted_providers=("openrouter",),
            reasons=("primary provider missing key",),
        )

    fixture = build_fixture()
    monkeypatch.setattr(research_module, "get_llm", fake_get_llm)
    monkeypatch.setattr(
        research_module,
        "_build_optional_tavily_provider",
        lambda: (None, None, None),
    )

    with pytest.raises(AllProvidersFailedError):
        await research_node(
            PipelineState(
                run_id="run-2026-04-04-main",
                run_date=date(2026, 4, 4),
                started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
                current_stage=PipelineStage.RESEARCH,
                fixtures=[fixture],
            ),
            batch_size=1,
        )
