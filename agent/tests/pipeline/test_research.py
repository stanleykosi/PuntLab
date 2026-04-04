"""Tests for PuntLab's research pipeline node.

Purpose: verify that the research stage batches fixture analysis, enriches
evidence with Tavily when available, and falls back to neutral contexts when
LLM access is unavailable.
Scope: unit tests for `src.pipeline.nodes.research`.
Dependencies: pytest plus lightweight LLM and Tavily stubs and the canonical
pipeline, fixture, news, and injury schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.llm import AllProvidersFailedError, MatchContext
from src.pipeline.nodes.research import research_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.stats import InjuryData, InjuryType


class FakeStructuredLLM:
    """Async structured-output runner used by research-node tests."""

    def __init__(self, result: MatchContext | dict[str, object]) -> None:
        """Store the result payload returned by the fake LLM."""

        self.result = result
        self.error: Exception | None = None
        self.invocations: list[object] = []

    async def ainvoke(self, prompt_value: object) -> MatchContext | dict[str, object]:
        """Record prompt invocations and return the configured payload."""

        self.invocations.append(prompt_value)
        if self.error is not None:
            raise self.error
        return self.result


class FakeLLM:
    """Minimal chat-model stand-in that exposes `with_structured_output()`."""

    def __init__(self, structured_runner: FakeStructuredLLM) -> None:
        """Persist the runner handling structured invocations."""

        self.structured_runner = structured_runner
        self.requested_schema: type[object] | None = None

    def with_structured_output(self, schema: type[object]) -> FakeStructuredLLM:
        """Record the requested schema and return the structured runner."""

        self.requested_schema = schema
        return self.structured_runner


@dataclass(slots=True)
class StubTavilyProvider:
    """Fixture-aware Tavily stub returning deterministic article sets."""

    results_by_fixture: dict[str, tuple[NewsArticle, ...]]
    requested_fixture_refs: list[str]

    async def search_match_news(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int | None = None,
        search_depth: str | None = None,
        lookback_days: int = 7,
        include_domains: tuple[str, ...] | None = None,
        exclude_domains: tuple[str, ...] | None = None,
    ) -> list[NewsArticle]:
        """Return configured Tavily articles for one fixture."""

        del max_results, search_depth, lookback_days, include_domains, exclude_domains
        fixture_ref = fixture.get_fixture_ref()
        self.requested_fixture_refs.append(fixture_ref)
        return list(self.results_by_fixture.get(fixture_ref, ()))


def build_fixture(
    *,
    sportradar_id: str,
    home_team: str,
    away_team: str,
) -> NormalizedFixture:
    """Create a canonical fixture used in research-node tests."""

    return NormalizedFixture(
        sportradar_id=sportradar_id,
        home_team=home_team,
        away_team=away_team,
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 0, tzinfo=UTC),
        source_provider="api-football",
        source_id=sportradar_id.split(":")[-1],
        country="England",
        home_team_id=f"{home_team.lower()}-id",
        away_team_id=f"{away_team.lower()}-id",
    )


def build_article(
    *,
    headline: str,
    source: str,
    url: str,
    fixture_ref: str,
    teams: tuple[str, ...],
) -> NewsArticle:
    """Create a normalized article used across research tests."""

    return NewsArticle(
        headline=headline,
        url=url,
        published_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
        source=source,
        source_provider="test-suite",
        summary="A concise article summary used for prompt grounding.",
        sport=SportName.SOCCER,
        competition="Premier League",
        teams=teams,
        fixture_ref=fixture_ref,
    )


def build_state(
    *,
    fixtures: tuple[NormalizedFixture, ...],
    news_articles: tuple[NewsArticle, ...],
) -> PipelineState:
    """Create a pipeline state containing fixture and news evidence."""

    return PipelineState(
        run_id="run-2026-04-04-main",
        run_date=datetime(2026, 4, 4, tzinfo=UTC).date(),
        started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        current_stage=PipelineStage.RESEARCH,
        fixtures=list(fixtures),
        news_articles=list(news_articles),
        injuries=[
            InjuryData(
                fixture_ref=fixtures[0].get_fixture_ref(),
                team_id=fixtures[0].away_team_id or "away-id",
                team_name=fixtures[0].away_team,
                player_name="Reece James",
                source_provider="api-football",
                injury_type=InjuryType.INJURY,
                is_key_player=True,
                reported_at=datetime(2026, 4, 4, 6, 30, tzinfo=UTC),
            )
        ],
        errors=["Earlier-stage warning."],
    )


@pytest.mark.asyncio
async def test_research_node_generates_match_contexts_with_tavily_enrichment() -> None:
    """The research node should merge ingestion news with Tavily and advance the stage."""

    fixture = build_fixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    state_article = build_article(
        headline="Arsenal carry strong home form into London clash",
        source="BBC Sport",
        url="https://example.com/arsenal-home-form",
        fixture_ref=fixture.get_fixture_ref(),
        teams=("Arsenal", "Chelsea"),
    )
    tavily_article = build_article(
        headline="Chelsea prepare for derby pressure after midweek loss",
        source="ESPN",
        url="https://example.com/chelsea-derby-pressure",
        fixture_ref=fixture.get_fixture_ref(),
        teams=("Chelsea",),
    )
    llm_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.72,
            morale_away=0.44,
            rivalry_factor=0.63,
            pressure_home=0.41,
            pressure_away=0.71,
            key_narrative="Arsenal look steadier while Chelsea arrive under more pressure.",
            qualitative_score=0.69,
            data_sources=("ignored-by-node",),
            news_summary=None,
        )
    )
    llm = FakeLLM(llm_runner)
    tavily_provider = StubTavilyProvider(
        results_by_fixture={fixture.get_fixture_ref(): (tavily_article,)},
        requested_fixture_refs=[],
    )

    result = await research_node(
        build_state(fixtures=(fixture,), news_articles=(state_article,)),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=tavily_provider,  # type: ignore[arg-type]
        batch_size=1,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert llm.requested_schema is MatchContext
    assert tavily_provider.requested_fixture_refs == [fixture.get_fixture_ref()]
    assert len(result["match_contexts"]) == 1
    context = result["match_contexts"][0]
    assert context.fixture_ref == fixture.get_fixture_ref()
    assert context.data_sources == ("BBC Sport", "ESPN")
    assert context.news_summary == (
        "Arsenal carry strong home form into London clash; "
        "Chelsea prepare for derby pressure after midweek loss"
    )
    assert result["errors"] == ["Earlier-stage warning."]


@pytest.mark.asyncio
async def test_research_node_falls_back_to_neutral_context_when_llm_provider_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing LLM providers should not block conservative research output."""

    fixture = build_fixture(
        sportradar_id="sr:match:7002",
        home_team="Manchester City",
        away_team="Liverpool",
    )
    article = build_article(
        headline="City and Liverpool brace for another title-race showdown",
        source="Sky Sports",
        url="https://example.com/title-race-showdown",
        fixture_ref=fixture.get_fixture_ref(),
        teams=("Manchester City", "Liverpool"),
    )

    async def fake_get_llm(task: str = "default") -> FakeLLM:
        """Raise the canonical provider-construction error for the research task."""

        raise AllProvidersFailedError(
            task,
            attempted_providers=("openai", "anthropic"),
            reasons=("primary provider missing key", "secondary provider missing key"),
        )

    monkeypatch.setattr("src.pipeline.nodes.research.get_llm", fake_get_llm)
    monkeypatch.setattr(
        "src.pipeline.nodes.research._build_optional_tavily_provider",
        lambda: (None, None, None),
    )

    result = await research_node(
        build_state(fixtures=(fixture,), news_articles=(article,)),
        batch_size=2,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert len(result["match_contexts"]) == 1
    context = result["match_contexts"][0]
    assert context.fixture_ref == fixture.get_fixture_ref()
    assert context.qualitative_score == pytest.approx(0.5)
    assert context.data_sources == ("Sky Sports",)
    assert context.news_summary == "City and Liverpool brace for another title-race showdown"
    assert "Unable to construct an LLM for task 'research'." in result["errors"][1]
