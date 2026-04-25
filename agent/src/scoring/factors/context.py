"""LLM-assisted qualitative context factor for PuntLab's scoring engine.

Purpose: translate fixture-specific news and narrative signals into one
bounded context score that can complement the deterministic scoring factors.
Scope: filter relevant articles, invoke the canonical structured research
prompt, and temper qualitative influence back toward neutral when evidence is
thin or unavailable.
Dependencies: canonical LLM prompt/schema exports from `src.llm`, fixture and
news schemas from `src.schemas`, and the shared WAT timezone configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import WAT_TIMEZONE
from src.llm import MatchContext, get_prompt
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle

logger = logging.getLogger(__name__)

NEUTRAL_CONTEXT_SCORE: Final[float] = 0.5
MAX_NEWS_ARTICLES: Final[int] = 6
MAX_NEWS_AGE_DAYS: Final[int] = 5
MIN_RELEVANCE_THRESHOLD: Final[float] = 0.25
MIN_DAMPING_MULTIPLIER: Final[float] = 0.35
MAX_DAMPING_MULTIPLIER: Final[float] = 0.70


async def analyze_context(
    fixture: NormalizedFixture,
    news: Sequence[NewsArticle],
    llm: BaseChatModel | None,
) -> float:
    """Return a conservative qualitative context score for one fixture.

    Inputs:
        fixture: Canonical fixture under analysis.
        news: Fixture-adjacent news candidates from RSS or search providers.
        llm: Preconfigured LangChain chat model for the `research` task. A
            missing model is treated as a signal to skip the qualitative layer.

    Outputs:
        A bounded score in the `0.0-1.0` range. When the LLM or evidence is
        unavailable, the function returns a neutral `0.5` score so the match is
        neither rewarded nor penalized for missing qualitative data.

    Raises:
        TypeError: If `fixture` or any news item is not canonical.
    """

    if not isinstance(fixture, NormalizedFixture):
        raise TypeError("analyze_context expects a NormalizedFixture instance.")

    relevant_articles = _select_relevant_articles(fixture, news)
    if llm is None or not relevant_articles:
        return NEUTRAL_CONTEXT_SCORE

    prompt = get_prompt("research")
    prompt_messages = prompt.format_messages(
        run_date=fixture.kickoff.astimezone(WAT_TIMEZONE).date().isoformat(),
        fixture_summary=_build_fixture_summary(fixture),
        competition_context=_build_competition_context(fixture),
        kickoff_context=_build_kickoff_context(fixture),
        known_absences=(
            "No structured absences feed was provided to this factor. Use only"
            " article-backed availability context."
        ),
        fixture_details=(
            "No SportyBet fixture-page widget details were supplied to this legacy "
            "context factor."
        ),
        recent_news_bullets=_render_news_bullets(relevant_articles),
        source_labels=", ".join(_ordered_unique_sources(relevant_articles)),
    )

    try:
        structured_llm = llm.with_structured_output(MatchContext)
        raw_result = await structured_llm.ainvoke(prompt_messages)
        context = (
            raw_result
            if isinstance(raw_result, MatchContext)
            else MatchContext.model_validate(raw_result)
        )
    except Exception as exc:
        logger.warning(
            "Context analysis failed for fixture '%s'; returning neutral score. Error: %s",
            fixture.get_fixture_ref(),
            exc,
        )
        return NEUTRAL_CONTEXT_SCORE

    evidence_strength = _calculate_evidence_strength(fixture, relevant_articles, context)
    return _temper_context_score(context.qualitative_score, evidence_strength)


def _select_relevant_articles(
    fixture: NormalizedFixture,
    news: Sequence[NewsArticle],
) -> tuple[NewsArticle, ...]:
    """Filter and rank the news set down to the most useful fixture articles."""

    normalized_articles: list[tuple[float, NewsArticle]] = []
    for article in news:
        if not isinstance(article, NewsArticle):
            raise TypeError("analyze_context expects NewsArticle instances only.")

        relevance = _score_article_relevance(fixture, article)
        if relevance < MIN_RELEVANCE_THRESHOLD:
            continue
        normalized_articles.append((relevance, article))

    normalized_articles.sort(
        key=lambda item: (
            item[0],
            item[1].relevance_score or 0.0,
            item[1].published_at.astimezone(UTC),
        ),
        reverse=True,
    )
    return tuple(article for _, article in normalized_articles[:MAX_NEWS_ARTICLES])


def _score_article_relevance(fixture: NormalizedFixture, article: NewsArticle) -> float:
    """Estimate how closely one article maps to the target fixture."""

    fixture_ref = fixture.get_fixture_ref()
    score = 0.0

    if article.fixture_ref == fixture_ref:
        score += 0.70
    if article.sport == fixture.sport:
        score += 0.10
    if article.competition and article.competition.casefold() == fixture.competition.casefold():
        score += 0.20

    matched_teams = {
        team.casefold()
        for team in article.teams
        if team.casefold() in {fixture.home_team.casefold(), fixture.away_team.casefold()}
    }
    score += min(0.50, len(matched_teams) * 0.25)

    if article.relevance_score is not None:
        score = max(score, article.relevance_score)

    return _clamp(score)


def _build_fixture_summary(fixture: NormalizedFixture) -> str:
    """Build the compact fixture label used in research prompts."""

    return f"{fixture.home_team} vs {fixture.away_team}"


def _build_competition_context(fixture: NormalizedFixture) -> str:
    """Build a competition and country summary for the research prompt."""

    if fixture.country:
        return f"{fixture.competition} ({fixture.country})"
    return fixture.competition


def _build_kickoff_context(fixture: NormalizedFixture) -> str:
    """Describe kickoff timing in WAT plus optional venue context."""

    kickoff_wat = fixture.kickoff.astimezone(WAT_TIMEZONE)
    kickoff_label = kickoff_wat.strftime("%Y-%m-%d %H:%M WAT")
    if fixture.venue:
        return f"{kickoff_label} at {fixture.venue}"
    return kickoff_label


def _render_news_bullets(articles: Sequence[NewsArticle]) -> str:
    """Render the prompt-safe article summary list for the research model."""

    bullets: list[str] = []
    for article in articles:
        source = article.source
        headline = _truncate(article.headline, 120)
        snippet = _truncate(article.summary or article.content_snippet or "", 140)
        published_label = article.published_at.astimezone(WAT_TIMEZONE).strftime("%Y-%m-%d %H:%M")

        if snippet:
            bullets.append(f"- [{source} | {published_label} WAT] {headline}: {snippet}")
        else:
            bullets.append(f"- [{source} | {published_label} WAT] {headline}")

    return "\n".join(bullets)


def _ordered_unique_sources(articles: Sequence[NewsArticle]) -> tuple[str, ...]:
    """Return ordered unique source labels for prompt grounding."""

    ordered_sources: list[str] = []
    seen: set[str] = set()

    for article in articles:
        lookup_key = article.source.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        ordered_sources.append(article.source)

    return tuple(ordered_sources)


def _calculate_evidence_strength(
    fixture: NormalizedFixture,
    articles: Sequence[NewsArticle],
    context: MatchContext,
) -> float:
    """Estimate how much the qualitative score should be allowed to move."""

    now = fixture.kickoff.astimezone(UTC)
    source_diversity = _clamp(len(_ordered_unique_sources(articles)) / 3.0)
    article_coverage = _clamp(len(articles) / 4.0)
    freshness = _average(
        _news_freshness_score(article.published_at.astimezone(UTC), now) for article in articles
    )
    provider_relevance = _average(
        article.relevance_score
        if article.relevance_score is not None
        else _score_article_relevance(fixture, article)
        for article in articles
    )
    context_source_coverage = _clamp(
        len(context.data_sources) / max(len(_ordered_unique_sources(articles)), 1)
    )

    evidence_strength = (
        (source_diversity * 0.25)
        + (article_coverage * 0.25)
        + (freshness * 0.20)
        + (provider_relevance * 0.20)
        + (context_source_coverage * 0.10)
    )
    return _clamp(evidence_strength)


def _news_freshness_score(published_at: datetime, reference_time: datetime) -> float:
    """Score article freshness relative to the upcoming fixture kickoff."""

    age = max(timedelta(), reference_time - published_at)
    max_age = timedelta(days=MAX_NEWS_AGE_DAYS)
    if age >= max_age:
        return 0.0
    return _clamp(1.0 - (age / max_age))


def _temper_context_score(raw_score: float, evidence_strength: float) -> float:
    """Shrink raw qualitative scores toward neutral so the factor stays bounded."""

    dampening_multiplier = MIN_DAMPING_MULTIPLIER + (
        (MAX_DAMPING_MULTIPLIER - MIN_DAMPING_MULTIPLIER) * _clamp(evidence_strength)
    )
    tempered_score = NEUTRAL_CONTEXT_SCORE + (
        (raw_score - NEUTRAL_CONTEXT_SCORE) * dampening_multiplier
    )
    return _clamp(tempered_score)


def _average(values: Iterable[float]) -> float:
    """Return a safe average for bounded evidence signals."""

    normalized_values = tuple(values)
    if not normalized_values:
        return 0.0
    return sum(normalized_values) / len(normalized_values)


def _truncate(value: str, limit: int) -> str:
    """Clip long prompt fragments to keep the LLM context compact."""

    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


__all__ = [
    "MAX_NEWS_ARTICLES",
    "NEUTRAL_CONTEXT_SCORE",
    "analyze_context",
]
