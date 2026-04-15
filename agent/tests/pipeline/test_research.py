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
        self.errors_by_call: list[Exception] = []
        self.invocations: list[object] = []

    async def ainvoke(self, prompt_value: object) -> MatchContext | dict[str, object]:
        """Record prompt invocations and return the configured payload."""

        self.invocations.append(prompt_value)
        if self.errors_by_call:
            raise self.errors_by_call.pop(0)
        if self.error is not None:
            raise self.error
        return self.result


class FakeLLM:
    """Minimal chat-model stand-in that exposes `with_structured_output()`."""

    def __init__(
        self,
        structured_runner: FakeStructuredLLM,
        *,
        json_mode_runner: FakeStructuredLLM | None = None,
        openai_api_base: str | None = None,
        plain_result: object | None = None,
    ) -> None:
        """Persist the runner handling structured invocations."""

        self.structured_runner = structured_runner
        self.json_mode_runner = json_mode_runner
        self.openai_api_base = openai_api_base
        self.plain_result = plain_result
        self.requested_schema: type[object] | None = None
        self.requested_methods: list[str] = []

    def with_structured_output(
        self,
        schema: type[object],
        **kwargs: object,
    ) -> FakeStructuredLLM:
        """Record the requested schema and return the structured runner."""

        method = kwargs.get("method")
        self.requested_schema = schema
        self.requested_methods.append(str(method) if method is not None else "default")
        if method == "json_mode" and self.json_mode_runner is not None:
            return self.json_mode_runner
        return self.structured_runner

    async def ainvoke(self, prompt_value: object) -> object:
        """Return plain text content for OpenRouter text-mode parsing tests."""

        del prompt_value
        if self.plain_result is None:
            raise RuntimeError("FakeLLM.plain_result must be configured for ainvoke().")
        return self.plain_result


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


@pytest.mark.asyncio
async def test_research_node_groups_repeated_llm_failures_by_root_cause() -> None:
    """Repeated fixture-level LLM failures should be summarized once."""

    fixture_one = build_fixture(
        sportradar_id="sr:match:8101",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    fixture_two = build_fixture(
        sportradar_id="sr:match:8102",
        home_team="Liverpool",
        away_team="Brighton",
    )
    article_one = build_article(
        headline="Arsenal keep momentum ahead of kickoff",
        source="BBC Sport",
        url="https://example.com/arsenal-momentum",
        fixture_ref=fixture_one.get_fixture_ref(),
        teams=(fixture_one.home_team, fixture_one.away_team),
    )
    article_two = build_article(
        headline="Liverpool expected to rotate in congested week",
        source="Sky Sports",
        url="https://example.com/liverpool-rotation",
        fixture_ref=fixture_two.get_fixture_ref(),
        teams=(fixture_two.home_team, fixture_two.away_team),
    )
    llm_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.5,
            morale_away=0.5,
            rivalry_factor=0.5,
            pressure_home=0.5,
            pressure_away=0.5,
            key_narrative="placeholder",
            qualitative_score=0.5,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm_runner.error = RuntimeError("schema validation failed")
    llm = FakeLLM(llm_runner)

    result = await research_node(
        build_state(
            fixtures=(fixture_one, fixture_two),
            news_articles=(article_one, article_two),
        ),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=2,
    )

    grouped_message = (
        "Research LLM analysis failed for 2 fixtures due to: "
        "LLM output schema validation failed. "
        "Fallback context was used. Sample fixtures: "
        f"{fixture_one.get_fixture_ref()}, {fixture_two.get_fixture_ref()}."
    )
    assert grouped_message in result["errors"]


@pytest.mark.asyncio
async def test_research_node_groups_rate_limit_failures_with_dynamic_request_ids() -> None:
    """Volatile request IDs should not fragment repeated rate-limit diagnostics."""

    fixture_one = build_fixture(
        sportradar_id="sr:match:8201",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    fixture_two = build_fixture(
        sportradar_id="sr:match:8202",
        home_team="Liverpool",
        away_team="Brighton",
    )
    article_one = build_article(
        headline="Arsenal await derby test",
        source="BBC Sport",
        url="https://example.com/arsenal-derby",
        fixture_ref=fixture_one.get_fixture_ref(),
        teams=(fixture_one.home_team, fixture_one.away_team),
    )
    article_two = build_article(
        headline="Liverpool continue title push",
        source="Sky Sports",
        url="https://example.com/liverpool-title-push",
        fixture_ref=fixture_two.get_fixture_ref(),
        teams=(fixture_two.home_team, fixture_two.away_team),
    )
    llm_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.5,
            morale_away=0.5,
            rivalry_factor=0.5,
            pressure_home=0.5,
            pressure_away=0.5,
            key_narrative="placeholder",
            qualitative_score=0.5,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm_runner.errors_by_call = [
        RuntimeError(
            "HTTP 429 from upstream provider. request_id=req-aaa111 and payload rejected."
        ),
        RuntimeError(
            "HTTP 429 from upstream provider. request_id=req-bbb222 and payload rejected."
        ),
    ]
    llm = FakeLLM(llm_runner)

    result = await research_node(
        build_state(
            fixtures=(fixture_one, fixture_two),
            news_articles=(article_one, article_two),
        ),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=2,
    )

    grouped_message = (
        "Research LLM analysis failed for 2 fixtures due to: LLM provider rate limit reached. "
        "Fallback context was used. Sample fixtures: "
        f"{fixture_one.get_fixture_ref()}, {fixture_two.get_fixture_ref()}."
    )
    assert grouped_message in result["errors"]


@pytest.mark.asyncio
async def test_research_node_retries_structured_output_in_json_mode() -> None:
    """Research should retry in JSON mode when default structured parsing fails."""

    fixture = build_fixture(
        sportradar_id="sr:match:8301",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    article = build_article(
        headline="Arsenal prepare for another derby night",
        source="BBC Sport",
        url="https://example.com/arsenal-derby-night",
        fixture_ref=fixture.get_fixture_ref(),
        teams=(fixture.home_team, fixture.away_team),
    )
    primary_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.5,
            morale_away=0.5,
            rivalry_factor=0.5,
            pressure_home=0.5,
            pressure_away=0.5,
            key_narrative="placeholder",
            qualitative_score=0.5,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    primary_runner.error = RuntimeError("'NoneType' object is not iterable")
    json_mode_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.74,
            morale_away=0.41,
            rivalry_factor=0.66,
            pressure_home=0.36,
            pressure_away=0.69,
            key_narrative="Arsenal enter with stronger momentum.",
            qualitative_score=0.71,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm = FakeLLM(primary_runner, json_mode_runner=json_mode_runner)

    result = await research_node(
        build_state(fixtures=(fixture,), news_articles=(article,)),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=1,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert len(result["match_contexts"]) == 1
    assert "Earlier-stage warning." in result["errors"]
    assert not any(
        message.startswith("Research LLM analysis failed for")
        for message in result["errors"]
    )
    assert llm.requested_methods == ["default", "json_mode"]


@pytest.mark.asyncio
async def test_research_node_uses_json_schema_mode_for_openrouter_models() -> None:
    """OpenRouter models should prefer JSON-schema structured output."""

    fixture = build_fixture(
        sportradar_id="sr:match:8302",
        home_team="Liverpool",
        away_team="Brighton",
    )
    article = build_article(
        headline="Liverpool host Brighton under title pressure",
        source="Sky Sports",
        url="https://example.com/liverpool-brighton-preview",
        fixture_ref=fixture.get_fixture_ref(),
        teams=(fixture.home_team, fixture.away_team),
    )
    llm_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.5,
            morale_away=0.5,
            rivalry_factor=0.5,
            pressure_home=0.5,
            pressure_away=0.5,
            key_narrative="placeholder",
            qualitative_score=0.5,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm = FakeLLM(
        llm_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    result = await research_node(
        build_state(fixtures=(fixture,), news_articles=(article,)),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=1,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert len(result["match_contexts"]) == 1
    assert result["match_contexts"][0].fixture_ref == fixture.get_fixture_ref()
    assert llm.requested_methods == ["json_schema"]
    assert not any(
        message.startswith("Research LLM analysis failed for")
        for message in result["errors"]
    )


@pytest.mark.asyncio
async def test_research_node_falls_back_to_json_mode_when_openrouter_json_schema_fails() -> None:
    """OpenRouter should retry with JSON mode if JSON schema mode is unsupported."""

    fixture = build_fixture(
        sportradar_id="sr:match:8303",
        home_team="Newcastle",
        away_team="Tottenham",
    )
    article = build_article(
        headline="Newcastle host Tottenham with both sides chasing Europe",
        source="BBC Sport",
        url="https://example.com/newcastle-tottenham-europe-race",
        fixture_ref=fixture.get_fixture_ref(),
        teams=(fixture.home_team, fixture.away_team),
    )
    json_schema_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.5,
            morale_away=0.5,
            rivalry_factor=0.5,
            pressure_home=0.5,
            pressure_away=0.5,
            key_narrative="placeholder",
            qualitative_score=0.5,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    json_schema_runner.error = RuntimeError(
        "Model does not support response_format json_schema for this route."
    )
    json_mode_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.59,
            morale_away=0.57,
            rivalry_factor=0.62,
            pressure_home=0.54,
            pressure_away=0.56,
            key_narrative="Both teams arrive in tight form with little separation.",
            qualitative_score=0.58,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm = FakeLLM(
        json_schema_runner,
        json_mode_runner=json_mode_runner,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    result = await research_node(
        build_state(fixtures=(fixture,), news_articles=(article,)),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=1,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert len(result["match_contexts"]) == 1
    assert result["match_contexts"][0].fixture_ref == fixture.get_fixture_ref()
    assert llm.requested_methods == ["json_schema", "json_mode"]
    assert not any(
        message.startswith("Research LLM analysis failed for")
        for message in result["errors"]
    )


@pytest.mark.asyncio
async def test_research_node_does_not_leak_fixture_bound_articles_into_other_fixtures() -> None:
    """Fixture-bound article relevance must not bleed into unrelated fixtures."""

    fixture_one = build_fixture(
        sportradar_id="sr:match:8401",
        home_team="Inter Milan",
        away_team="Roma",
    )
    fixture_two = build_fixture(
        sportradar_id="sr:match:8402",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    # This article is explicitly bound to fixture one and carries a high
    # provider relevance score. It must never be reused for fixture two.
    fixture_one_article = NewsArticle(
        headline="Serie A: Inter Milan rout Roma, go nine points clear",
        url="https://example.com/inter-roma",
        published_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        source="ESPN",
        source_provider="rss",
        summary="Inter dominated Roma in a statement title-race performance.",
        sport=SportName.SOCCER,
        competition="Premier League",
        teams=("Inter Milan", "Roma"),
        fixture_ref=fixture_one.get_fixture_ref(),
        relevance_score=0.95,
    )
    fixture_two_article = build_article(
        headline="Arsenal prepare for derby pressure against Chelsea",
        source="BBC Sport",
        url="https://example.com/arsenal-chelsea-derby",
        fixture_ref=fixture_two.get_fixture_ref(),
        teams=(fixture_two.home_team, fixture_two.away_team),
    )
    llm_runner = FakeStructuredLLM(
        MatchContext(
            fixture_ref=None,
            morale_home=0.61,
            morale_away=0.52,
            rivalry_factor=0.64,
            pressure_home=0.49,
            pressure_away=0.58,
            key_narrative="Arsenal remain slight favorites ahead of kickoff.",
            qualitative_score=0.62,
            data_sources=("ignored",),
            news_summary=None,
        )
    )
    llm = FakeLLM(llm_runner)

    result = await research_node(
        build_state(
            fixtures=(fixture_one, fixture_two),
            news_articles=(fixture_one_article, fixture_two_article),
        ),
        llm=llm,  # type: ignore[arg-type]
        tavily_provider=None,
        batch_size=2,
    )

    assert result["current_stage"] == PipelineStage.SCORING
    assert len(result["match_contexts"]) == 2
    context_by_fixture = {
        context.fixture_ref: context for context in result["match_contexts"]
    }
    fixture_two_context = context_by_fixture[fixture_two.get_fixture_ref()]
    assert fixture_two_context.news_summary is not None
    assert "Inter Milan rout Roma" not in fixture_two_context.news_summary
