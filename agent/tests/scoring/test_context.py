"""Tests for PuntLab's LLM-assisted qualitative context scoring factor.

Purpose: verify that context scoring stays conservative, fixture-aware, and
neutral when the qualitative layer lacks usable evidence or LLM access.
Scope: unit tests for `src.scoring.factors.context`.
Dependencies: pytest plus canonical fixture/news schemas and lightweight LLM
test doubles that mimic `with_structured_output(...).ainvoke(...)`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.llm import MatchContext
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.scoring.factors.context import NEUTRAL_CONTEXT_SCORE, analyze_context


class FakeStructuredLLM:
    """Simple async structured-output runner used by the context-factor tests."""

    def __init__(self, *, result: MatchContext | dict[str, object] | None = None) -> None:
        """Store the result payload returned by the fake LLM."""

        self.result = result
        self.error: Exception | None = None
        self.invocations: list[object] = []

    async def ainvoke(self, prompt_value: object) -> MatchContext | dict[str, object]:
        """Record prompt invocations and return the configured payload."""

        self.invocations.append(prompt_value)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("FakeStructuredLLM requires a configured result.")
        return self.result


class FakeLLM:
    """Minimal chat-model stand-in that exposes `with_structured_output()`."""

    def __init__(self, structured_runner: FakeStructuredLLM) -> None:
        """Persist the runner that handles future `ainvoke()` calls."""

        self.structured_runner = structured_runner
        self.requested_schema: type[object] | None = None

    def with_structured_output(self, schema: type[object]) -> FakeStructuredLLM:
        """Record the requested schema and return the structured runner."""

        self.requested_schema = schema
        return self.structured_runner


def build_fixture() -> NormalizedFixture:
    """Build a canonical fixture for context-factor tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 17, 30, tzinfo=UTC),
        source_provider="test-suite",
        source_id="fixture-61301159",
        country="England",
        venue="Emirates Stadium",
    )


def build_article(
    *,
    headline: str,
    published_at: datetime,
    teams: tuple[str, ...],
    source: str = "BBC Sport",
    fixture_ref: str | None = "sr:match:61301159",
    competition: str | None = "Premier League",
    relevance_score: float | None = None,
) -> NewsArticle:
    """Build a normalized article for one fixture-aware context test."""

    return NewsArticle(
        headline=headline,
        url=f"https://example.com/{headline.lower().replace(' ', '-')}",
        published_at=published_at,
        source=source,
        source_provider="test-suite",
        summary="A short article summary for testing the qualitative prompt payload.",
        sport=SportName.SOCCER,
        competition=competition,
        teams=teams,
        fixture_ref=fixture_ref,
        relevance_score=relevance_score,
    )


@pytest.mark.asyncio
async def test_analyze_context_temperes_high_llm_score_with_conservative_damping() -> None:
    """Strong qualitative output should still be pulled back toward neutral."""

    fixture = build_fixture()
    articles = (
        build_article(
            headline="Arsenal buoyed by unbeaten stretch before Chelsea visit",
            published_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
            teams=("Arsenal", "Chelsea"),
            source="BBC Sport",
            relevance_score=0.90,
        ),
        build_article(
            headline="Chelsea face pressure after derby loss ahead of London clash",
            published_at=datetime(2026, 4, 3, 19, 0, tzinfo=UTC),
            teams=("Chelsea",),
            source="ESPN",
            relevance_score=0.80,
        ),
    )
    structured_runner = FakeStructuredLLM(
        result=MatchContext(
            fixture_ref=fixture.get_fixture_ref(),
            morale_home=0.82,
            morale_away=0.34,
            rivalry_factor=0.75,
            pressure_home=0.46,
            pressure_away=0.78,
            key_narrative="Arsenal arrive steadier while Chelsea carry heavier pressure.",
            qualitative_score=0.92,
            data_sources=("BBC Sport", "ESPN"),
            news_summary="Home morale is stronger and the away side looks more unsettled.",
        )
    )
    llm = FakeLLM(structured_runner)

    score = await analyze_context(fixture, articles, llm)

    assert llm.requested_schema is MatchContext
    assert len(structured_runner.invocations) == 1
    prompt_messages = structured_runner.invocations[0]
    assert isinstance(prompt_messages, list)
    assert "Arsenal vs Chelsea" in prompt_messages[1].content
    assert "BBC Sport, ESPN" in prompt_messages[1].content
    assert score == pytest.approx(0.756, abs=0.01)
    assert NEUTRAL_CONTEXT_SCORE < score < 0.92


@pytest.mark.asyncio
async def test_analyze_context_returns_neutral_when_no_relevant_news_exists() -> None:
    """Irrelevant articles should skip the LLM and leave the factor neutral."""

    fixture = build_fixture()
    unrelated_articles = (
        build_article(
            headline="Lakers prepare for marquee Western Conference showdown",
            published_at=datetime(2026, 4, 4, 8, 0, tzinfo=UTC),
            teams=("Los Angeles Lakers", "Denver Nuggets"),
            source="ESPN",
            fixture_ref="sr:match:7001",
            competition="NBA",
        ),
    )
    structured_runner = FakeStructuredLLM(
        result={
            "fixture_ref": fixture.get_fixture_ref(),
            "morale_home": 0.5,
            "morale_away": 0.5,
            "rivalry_factor": 0.2,
            "pressure_home": 0.5,
            "pressure_away": 0.5,
            "key_narrative": "Unused because the LLM should not be called.",
            "qualitative_score": 0.8,
            "data_sources": ["ESPN"],
        }
    )
    llm = FakeLLM(structured_runner)

    score = await analyze_context(fixture, unrelated_articles, llm)

    assert score == pytest.approx(NEUTRAL_CONTEXT_SCORE)
    assert structured_runner.invocations == []
    assert llm.requested_schema is None


@pytest.mark.asyncio
async def test_analyze_context_returns_neutral_when_llm_analysis_fails() -> None:
    """LLM failures should degrade gracefully instead of blocking scoring."""

    fixture = build_fixture()
    articles = (
        build_article(
            headline="Arsenal monitor late fitness concern before derby-style clash",
            published_at=datetime(2026, 4, 4, 10, 30, tzinfo=UTC),
            teams=("Arsenal",),
            source="Sky Sports",
            relevance_score=0.88,
        ),
    )
    structured_runner = FakeStructuredLLM(
        result={
            "fixture_ref": fixture.get_fixture_ref(),
            "morale_home": 0.7,
            "morale_away": 0.4,
            "rivalry_factor": 0.6,
            "pressure_home": 0.4,
            "pressure_away": 0.7,
            "key_narrative": "Unused fallback payload.",
            "qualitative_score": 0.75,
            "data_sources": ["Sky Sports"],
        }
    )
    structured_runner.error = RuntimeError("provider timeout")
    llm = FakeLLM(structured_runner)

    score = await analyze_context(fixture, articles, llm)

    assert llm.requested_schema is MatchContext
    assert len(structured_runner.invocations) == 1
    assert score == pytest.approx(NEUTRAL_CONTEXT_SCORE)


@pytest.mark.asyncio
async def test_analyze_context_rejects_non_canonical_news_inputs() -> None:
    """The factor should fail fast when callers pass non-news records."""

    fixture = build_fixture()

    with pytest.raises(TypeError, match="NewsArticle instances only"):
        await analyze_context(fixture, ["not-a-news-article"], llm=None)  # type: ignore[list-item]
