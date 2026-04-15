"""Research node for PuntLab's LangGraph pipeline.

Purpose: generate fixture-level qualitative context using the canonical
research prompt, structured `MatchContext` output, and optional Tavily search
enrichment on top of ingestion-stage news.
Scope: batch-oriented fixture research, evidence selection, conservative
fallback contexts, and diagnostic propagation for LLM or search failures.
Dependencies: `src.llm` for provider/prompt access, `src.providers` for
optional Tavily enrichment, and `src.pipeline.state.PipelineState` for
validated state exchange between LangGraph nodes.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, get_settings
from src.llm import AllProvidersFailedError, MatchContext, get_llm, get_prompt
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.base import ProviderError, RateLimitedClient
from src.providers.tavily_search import TavilySearchProvider
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.stats import InjuryData

DEFAULT_RESEARCH_BATCH_SIZE = 4
MAX_CONTEXT_ARTICLES = 6
MAX_TAVILY_RESULTS = 3
MAX_ARTICLE_AGE_DAYS = 7
MIN_RELEVANCE_THRESHOLD = 0.20
NEUTRAL_CONTEXT_SCORE = 0.5
FALLBACK_SOURCE_LABEL = "PuntLab fallback"
_RESEARCH_LLM_FAILURE_PATTERN = re.compile(
    r"^Research LLM analysis failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_TAVILY_RESEARCH_FAILURE_PATTERN = re.compile(
    r"^Tavily research fetch failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_FIXTURE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:sr:match:\d+|football-data:\d+|api-football:\d+|balldontlie:\d+|the-odds-api:[a-z0-9_-]+)\b",
    flags=re.IGNORECASE,
)
_REQUEST_ID_PATTERN = re.compile(
    r"\brequest(?:[-_ ]?id)?\s*[:=]\s*[a-z0-9._-]+\b",
    flags=re.IGNORECASE,
)
_UUID_PATTERN = re.compile(
    r"\b[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FixtureResearchResult:
    """Research output bundle for one fixture.

    Inputs:
        A fixture, its supporting evidence set, and the configured research
        dependencies.

    Outputs:
        A validated `MatchContext` plus any recoverable diagnostics generated
        while building the context.
    """

    context: MatchContext
    diagnostics: tuple[str, ...] = ()


async def research_node(
    state: PipelineState | Mapping[str, Any],
    *,
    llm: BaseChatModel | None = None,
    tavily_provider: TavilySearchProvider | None = None,
    batch_size: int = DEFAULT_RESEARCH_BATCH_SIZE,
) -> dict[str, object]:
    """Execute the qualitative research stage for every ingested fixture.

    Inputs:
        state: Current pipeline state or a raw LangGraph-compatible mapping.
        llm: Optional injected research-model instance for tests or explicit
            runtime wiring.
        tavily_provider: Optional Tavily provider override used to enrich
            fixture-level evidence during the research pass.
        batch_size: Maximum number of fixtures to process concurrently in one
            batch. This keeps Tavily and LLM traffic conservative.

    Outputs:
        A partial LangGraph state update containing ordered `MatchContext`
        records, merged diagnostics, and the next stage marker.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    diagnostics: list[str] = []

    resolved_llm = llm
    if resolved_llm is None:
        try:
            resolved_llm = await get_llm("research")
        except AllProvidersFailedError as exc:
            diagnostics.append(str(exc))
            resolved_llm = None

    owned_cleanup: Callable[[], Awaitable[None]] | None = None
    resolved_tavily = tavily_provider
    if resolved_tavily is None:
        resolved_tavily, owned_cleanup, provider_diagnostic = _build_optional_tavily_provider()
        if provider_diagnostic is not None:
            diagnostics.append(provider_diagnostic)

    try:
        match_contexts: list[MatchContext] = []
        for fixture_batch in _iter_batches(validated_state.fixtures, batch_size):
            batch_results = await asyncio.gather(
                *[
                    _research_fixture(
                        fixture=fixture,
                        state_news=validated_state.news_articles,
                        injuries=validated_state.injuries,
                        llm=resolved_llm,
                        tavily_provider=resolved_tavily,
                    )
                    for fixture in fixture_batch
                ]
            )
            for research_result in batch_results:
                match_contexts.append(research_result.context)
                diagnostics.extend(research_result.diagnostics)
    finally:
        if owned_cleanup is not None:
            await owned_cleanup()

    return {
        "current_stage": PipelineStage.SCORING,
        "match_contexts": match_contexts,
        "errors": _merge_diagnostics(
            validated_state.errors,
            _summarize_research_diagnostics(diagnostics),
        ),
    }


async def _research_fixture(
    *,
    fixture: NormalizedFixture,
    state_news: Sequence[NewsArticle],
    injuries: Sequence[InjuryData],
    llm: BaseChatModel | None,
    tavily_provider: TavilySearchProvider | None,
) -> FixtureResearchResult:
    """Build one conservative `MatchContext` for a fixture.

    Inputs:
        fixture: Canonical fixture being researched.
        state_news: Ingestion-stage news articles already collected for the run.
        injuries: Structured injury rows available for the full run.
        llm: Optional research LLM. Missing models trigger the deterministic
            fallback context path.
        tavily_provider: Optional Tavily provider used for fresh fixture-level
            match-news enrichment.

    Outputs:
        A `FixtureResearchResult` containing a validated context plus any
        diagnostics produced while enriching or analyzing the fixture.
    """

    diagnostics: list[str] = []
    fixture_injuries = _fixture_injuries(fixture, injuries)
    evidence_articles = _select_relevant_articles(fixture, state_news)

    if tavily_provider is not None:
        try:
            tavily_articles = await tavily_provider.search_match_news(
                fixture=fixture,
                max_results=MAX_TAVILY_RESULTS,
            )
        except ProviderError as exc:
            diagnostics.append(
                f"Tavily research fetch failed for {fixture.get_fixture_ref()}: {exc}"
            )
        else:
            evidence_articles = _deduplicate_articles((*evidence_articles, *tavily_articles))

    if llm is None:
        return FixtureResearchResult(
            context=_build_fallback_context(
                fixture=fixture,
                articles=evidence_articles,
                injuries=fixture_injuries,
            ),
            diagnostics=tuple(diagnostics),
        )

    prompt = get_prompt("research")
    source_labels = _source_labels(evidence_articles, fixture_injuries)
    prompt_messages = prompt.format_messages(
        run_date=fixture.kickoff.astimezone(WAT_TIMEZONE).date().isoformat(),
        fixture_summary=_build_fixture_summary(fixture),
        competition_context=_build_competition_context(fixture),
        kickoff_context=_build_kickoff_context(fixture),
        known_absences=_render_known_absences(fixture, fixture_injuries),
        recent_news_bullets=_render_news_bullets(evidence_articles),
        source_labels=", ".join(source_labels),
    )

    try:
        context = await _invoke_structured_match_context(
            llm=llm,
            prompt_messages=prompt_messages,
        )
    except Exception as exc:
        diagnostics.append(
            f"Research LLM analysis failed for {fixture.get_fixture_ref()}: {exc}"
        )
        context = _build_fallback_context(
            fixture=fixture,
            articles=evidence_articles,
            injuries=fixture_injuries,
        )
    else:
        context = context.model_copy(
            update={
                "fixture_ref": fixture.get_fixture_ref(),
                "data_sources": source_labels,
                "news_summary": context.news_summary
                or _build_news_summary(evidence_articles, fixture_injuries),
            }
        )

    return FixtureResearchResult(context=context, diagnostics=tuple(diagnostics))


async def _invoke_structured_match_context(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
) -> MatchContext:
    """Invoke the research LLM with schema enforcement and a targeted retry.

    Inputs:
        llm: Configured chat model resolved for the research task.
        prompt_messages: Prompt message sequence produced by the canonical
            research prompt template.

    Outputs:
        A validated `MatchContext` object.

    Raises:
        Exception: Propagates provider or parsing errors when both the default
            and JSON-mode structured output paths fail.
    """

    if _is_openrouter_model(llm):
        return await _invoke_openrouter_structured_match_context(
            llm=llm,
            prompt_messages=prompt_messages,
        )

    primary_runner = llm.with_structured_output(MatchContext)
    try:
        raw_result = await primary_runner.ainvoke(prompt_messages)
    except Exception as exc:
        if not _should_retry_structured_output_with_json_mode(exc):
            raise
        json_mode_runner = _build_json_mode_structured_runner(llm)
        if json_mode_runner is None:
            raise
        raw_result = await json_mode_runner.ainvoke(prompt_messages)

    return (
        raw_result
        if isinstance(raw_result, MatchContext)
        else MatchContext.model_validate(raw_result)
    )


def _build_json_mode_structured_runner(llm: BaseChatModel) -> Any | None:
    """Attempt to build a JSON-mode structured-output runner.

    Inputs:
        llm: Configured chat model resolved for the research task.

    Outputs:
        A structured-output runner configured for JSON mode when the provider
        implementation supports that strategy; otherwise `None`.
    """

    try:
        return llm.with_structured_output(MatchContext, method="json_mode")
    except TypeError:
        return None


def _build_json_schema_structured_runner(llm: BaseChatModel) -> Any | None:
    """Attempt to build a JSON-schema structured-output runner.

    Inputs:
        llm: Configured chat model resolved for the research task.

    Outputs:
        A structured-output runner configured for JSON schema mode when the
        provider implementation supports that strategy; otherwise `None`.
    """

    try:
        return llm.with_structured_output(MatchContext, method="json_schema")
    except TypeError:
        return None


def _should_retry_structured_output_with_json_mode(error: Exception) -> bool:
    """Return whether structured output should retry using JSON mode.

    Inputs:
        error: Exception raised by the default structured-output call path.

    Outputs:
        `True` when the error matches known tool-calling parser instability
        signatures seen with some OpenRouter models.
    """

    message = " ".join(str(error).split()).casefold()
    return (
        "nonetype" in message and "iterable" in message
    ) or "tool_calls" in message


def _should_retry_openrouter_structured_output(error: Exception) -> bool:
    """Return whether OpenRouter structured output should fallback to JSON mode.

    Inputs:
        error: Exception raised while invoking JSON-schema structured output.

    Outputs:
        `True` when the error indicates either unsupported strict-schema mode
        or known structured-output parser instability.
    """

    message = " ".join(str(error).split()).casefold()
    return any(
        token in message
        for token in (
            "response_format",
            "json_schema",
            "structured output",
            "does not support",
            "doesn't support",
            "unsupported",
            "nonetype",
            "tool_calls",
            "schema validation",
        )
    )


def _is_openrouter_model(llm: BaseChatModel) -> bool:
    """Return whether the resolved chat model is configured for OpenRouter."""

    base_url = str(
        getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", "") or ""
    )
    return "openrouter.ai" in base_url.casefold()


async def _invoke_openrouter_structured_match_context(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
) -> MatchContext:
    """Invoke OpenRouter using documented structured-output request modes.

    Inputs:
        llm: Configured chat model resolved for the research task.
        prompt_messages: Prompt message sequence produced by the canonical
            research prompt template.

    Outputs:
        A validated `MatchContext` object.

    Raises:
        Exception: Propagates provider or parsing errors when both JSON schema
            and JSON mode structured output paths fail.
    """

    json_schema_runner = _build_json_schema_structured_runner(llm)
    json_schema_error: Exception | None = None
    if json_schema_runner is not None:
        try:
            json_schema_result = await json_schema_runner.ainvoke(prompt_messages)
            return (
                json_schema_result
                if isinstance(json_schema_result, MatchContext)
                else MatchContext.model_validate(json_schema_result)
            )
        except Exception as exc:
            json_schema_error = exc
            if not _should_retry_openrouter_structured_output(exc):
                raise

    json_mode_runner = _build_json_mode_structured_runner(llm)
    if json_mode_runner is None:
        if json_schema_error is not None:
            raise json_schema_error
        raise RuntimeError(
            "OpenRouter structured-output runner is unavailable: "
            "neither json_schema nor json_mode is supported by this model adapter."
        )

    json_mode_result = await json_mode_runner.ainvoke(prompt_messages)
    return (
        json_mode_result
        if isinstance(json_mode_result, MatchContext)
        else MatchContext.model_validate(json_mode_result)
    )


def _build_optional_tavily_provider() -> tuple[
    TavilySearchProvider | None,
    Callable[[], Awaitable[None]] | None,
    str | None,
]:
    """Build an owned Tavily provider only when configuration is present.

    Outputs:
        A provider instance, an async cleanup callback for owned resources, and
        an optional diagnostic when the provider could not be constructed.
    """

    tavily_api_key = get_settings().data_providers.tavily_api_key
    if not tavily_api_key:
        return None, None, "Tavily research enrichment skipped because no API key is configured."

    cache = RedisClient()
    client = RateLimitedClient(cache)
    try:
        provider = TavilySearchProvider(client, api_key=tavily_api_key)
    except ValueError as exc:
        return None, None, f"Tavily research enrichment could not start: {exc}"

    async def cleanup() -> None:
        """Close resources created specifically for the research node."""

        await client.aclose()
        await cache.close()

    return provider, cleanup, None


def _iter_batches[T](values: Sequence[T], batch_size: int) -> tuple[tuple[T, ...], ...]:
    """Split ordered values into fixed-size batches for rate-aware processing."""

    return tuple(
        tuple(values[index : index + batch_size])
        for index in range(0, len(values), batch_size)
    )


def _fixture_injuries(
    fixture: NormalizedFixture,
    injuries: Sequence[InjuryData],
) -> tuple[InjuryData, ...]:
    """Return injuries tied to the target fixture in source order."""

    fixture_ref = fixture.get_fixture_ref()
    return tuple(injury for injury in injuries if injury.fixture_ref == fixture_ref)


def _select_relevant_articles(
    fixture: NormalizedFixture,
    articles: Sequence[NewsArticle],
) -> tuple[NewsArticle, ...]:
    """Filter and rank articles down to the strongest fixture evidence set."""

    scored_articles: list[tuple[float, NewsArticle]] = []
    reference_time = fixture.kickoff.astimezone(UTC)
    for article in articles:
        if not isinstance(article, NewsArticle):
            raise TypeError("research_node expects NewsArticle instances only.")

        if reference_time - article.published_at.astimezone(UTC) > timedelta(
            days=MAX_ARTICLE_AGE_DAYS
        ):
            continue

        relevance = _score_article_relevance(fixture, article)
        if relevance < MIN_RELEVANCE_THRESHOLD:
            continue
        scored_articles.append((relevance, article))

    scored_articles.sort(
        key=lambda item: (
            item[0],
            item[1].relevance_score or 0.0,
            item[1].published_at.astimezone(UTC),
        ),
        reverse=True,
    )
    return tuple(article for _, article in scored_articles[:MAX_CONTEXT_ARTICLES])


def _score_article_relevance(fixture: NormalizedFixture, article: NewsArticle) -> float:
    """Estimate how closely one article maps to the target fixture."""

    fixture_ref = fixture.get_fixture_ref()
    if article.fixture_ref is not None and article.fixture_ref != fixture_ref:
        # Articles that are already fixture-linked must never bleed into other
        # fixtures, even if provider-level relevance metadata is high.
        return 0.0

    score = 0.0
    if article.fixture_ref == fixture_ref:
        score += 0.60

    sport_matches = article.sport == fixture.sport
    if sport_matches:
        score += 0.10

    competition_matches = bool(
        article.competition
        and article.competition.casefold() == fixture.competition.casefold()
    )
    if competition_matches:
        score += 0.15

    matched_teams = {
        team.casefold()
        for team in article.teams
        if team.casefold() in {fixture.home_team.casefold(), fixture.away_team.casefold()}
    }
    score += min(0.30, len(matched_teams) * 0.15)

    if article.relevance_score is not None:
        # Only trust provider-level relevance when the article is not already
        # fixture-linked and has at least one local anchor signal.
        has_local_anchor = bool(matched_teams) or competition_matches or sport_matches
        if article.fixture_ref is None and has_local_anchor:
            score = max(score, article.relevance_score)

    return max(0.0, min(1.0, score))


def _deduplicate_articles(articles: Sequence[NewsArticle]) -> tuple[NewsArticle, ...]:
    """Keep the first occurrence of each article by source ID or URL."""

    deduplicated: list[NewsArticle] = []
    seen_keys: set[tuple[str, str]] = set()
    for article in articles:
        source_id = article.source_id or ""
        dedupe_key = ("source_id", source_id.casefold()) if source_id else (
            "url",
            str(article.url).casefold(),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduplicated.append(article)
    return tuple(deduplicated)


def _source_labels(
    articles: Sequence[NewsArticle],
    injuries: Sequence[InjuryData],
) -> tuple[str, ...]:
    """Build ordered unique source labels grounded in evidence, not model output."""

    labels: list[str] = []
    seen: set[str] = set()

    for article in articles:
        lookup_key = article.source.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        labels.append(article.source)

    if labels:
        return tuple(labels)

    for injury in injuries:
        lookup_key = injury.source_provider.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        labels.append(injury.source_provider)

    return tuple(labels) if labels else (FALLBACK_SOURCE_LABEL,)


def _render_known_absences(
    fixture: NormalizedFixture,
    injuries: Sequence[InjuryData],
) -> str:
    """Summarize structured injury signals for the research prompt."""

    if not injuries:
        return "No structured absences were recorded for this fixture."

    team_labels = _fixture_team_labels(fixture)
    grouped: dict[str, list[str]] = {"home": [], "away": [], "unknown": []}
    for injury in injuries:
        team_side = _resolve_injury_team_side(fixture, injury)
        player_label = injury.player_name
        if injury.is_key_player:
            player_label += " (key)"
        grouped[team_side].append(player_label)

    summaries: list[str] = []
    if grouped["home"]:
        summaries.append(
            f"{team_labels['home']}: {', '.join(grouped['home'][:4])}"
        )
    if grouped["away"]:
        summaries.append(
            f"{team_labels['away']}: {', '.join(grouped['away'][:4])}"
        )
    if grouped["unknown"]:
        summaries.append(f"Unassigned: {', '.join(grouped['unknown'][:4])}")

    return " | ".join(summaries)


def _fixture_team_labels(fixture: NormalizedFixture) -> dict[str, str]:
    """Return stable team labels for home and away fixture participants."""

    return {"home": fixture.home_team, "away": fixture.away_team}


def _resolve_injury_team_side(fixture: NormalizedFixture, injury: InjuryData) -> str:
    """Map one injury row to the home, away, or unknown fixture side."""

    if injury.team_name:
        normalized_team_name = injury.team_name.casefold()
        if normalized_team_name == fixture.home_team.casefold():
            return "home"
        if normalized_team_name == fixture.away_team.casefold():
            return "away"

    if fixture.home_team_id and injury.team_id == fixture.home_team_id:
        return "home"
    if fixture.away_team_id and injury.team_id == fixture.away_team_id:
        return "away"
    return "unknown"


def _render_news_bullets(articles: Sequence[NewsArticle]) -> str:
    """Render compact bullet points for the research prompt payload."""

    if not articles:
        return "- No fixture-specific RSS or Tavily coverage was available."

    bullets: list[str] = []
    for article in articles:
        published_label = article.published_at.astimezone(WAT_TIMEZONE).strftime("%Y-%m-%d %H:%M")
        headline = _truncate(article.headline, 120)
        summary = _truncate(article.summary or article.content_snippet or "", 140)
        if summary:
            bullets.append(f"- [{article.source} | {published_label} WAT] {headline}: {summary}")
        else:
            bullets.append(f"- [{article.source} | {published_label} WAT] {headline}")

    return "\n".join(bullets)


def _build_fixture_summary(fixture: NormalizedFixture) -> str:
    """Return the compact fixture summary supplied to the research prompt."""

    return f"{fixture.home_team} vs {fixture.away_team}"


def _build_competition_context(fixture: NormalizedFixture) -> str:
    """Return the competition and country summary used in prompt grounding."""

    if fixture.country:
        return f"{fixture.competition} ({fixture.country})"
    return fixture.competition


def _build_kickoff_context(fixture: NormalizedFixture) -> str:
    """Describe the fixture kickoff in WAT, including venue when known."""

    kickoff_label = fixture.kickoff.astimezone(WAT_TIMEZONE).strftime("%Y-%m-%d %H:%M WAT")
    if fixture.venue:
        return f"{kickoff_label} at {fixture.venue}"
    return kickoff_label


def _build_news_summary(
    articles: Sequence[NewsArticle],
    injuries: Sequence[InjuryData],
) -> str:
    """Build a conservative news-summary field for fallback or incomplete LLM output."""

    if articles:
        summary = "; ".join(article.headline for article in articles[:2])
        return _truncate(summary, 160)
    if injuries:
        return "Availability updates were noted, but broader fixture news remained limited."
    return "Limited fixture-specific evidence keeps the qualitative view neutral."


def _build_fallback_context(
    *,
    fixture: NormalizedFixture,
    articles: Sequence[NewsArticle],
    injuries: Sequence[InjuryData],
) -> MatchContext:
    """Return a deterministic neutral context when research cannot use the LLM."""

    summary = _build_news_summary(articles, injuries)
    return MatchContext(
        fixture_ref=fixture.get_fixture_ref(),
        morale_home=NEUTRAL_CONTEXT_SCORE,
        morale_away=NEUTRAL_CONTEXT_SCORE,
        rivalry_factor=NEUTRAL_CONTEXT_SCORE,
        pressure_home=NEUTRAL_CONTEXT_SCORE,
        pressure_away=NEUTRAL_CONTEXT_SCORE,
        key_narrative=_truncate(summary, 200),
        qualitative_score=NEUTRAL_CONTEXT_SCORE,
        data_sources=_source_labels(articles, injuries),
        news_summary=summary,
    )


def _merge_diagnostics(existing_errors: Sequence[str], diagnostics: Sequence[str]) -> list[str]:
    """Append new diagnostics to the pipeline error list without duplicates."""

    merged = list(existing_errors)
    seen = {message.casefold() for message in merged}
    for diagnostic in diagnostics:
        lookup_key = diagnostic.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        merged.append(diagnostic)
    return merged


def _summarize_research_diagnostics(diagnostics: Sequence[str]) -> tuple[str, ...]:
    """Compress repetitive fixture-level research diagnostics.

    Inputs:
        diagnostics: Ordered diagnostic rows produced while researching a slate.

    Outputs:
        A tuple where repeated fixture-specific failures are grouped into one
        concise message per root cause while one-off diagnostics are preserved.
    """

    summarized: list[str] = []
    llm_failures_by_reason: dict[str, list[str]] = {}
    llm_reason_examples: dict[str, str] = {}
    tavily_failures_by_reason: dict[str, list[str]] = {}
    tavily_reason_examples: dict[str, str] = {}

    for diagnostic in diagnostics:
        llm_match = _RESEARCH_LLM_FAILURE_PATTERN.match(diagnostic)
        if llm_match is not None:
            raw_reason = llm_match.group("reason").strip()
            reason = _normalize_research_failure_reason(raw_reason)
            fixture_ref = llm_match.group("fixture").strip()
            llm_failures_by_reason.setdefault(reason, []).append(fixture_ref)
            llm_reason_examples.setdefault(reason, raw_reason)
            continue

        tavily_match = _TAVILY_RESEARCH_FAILURE_PATTERN.match(diagnostic)
        if tavily_match is not None:
            raw_reason = tavily_match.group("reason").strip()
            reason = _normalize_research_failure_reason(raw_reason)
            fixture_ref = tavily_match.group("fixture").strip()
            tavily_failures_by_reason.setdefault(reason, []).append(fixture_ref)
            tavily_reason_examples.setdefault(reason, raw_reason)
            continue

        summarized.append(diagnostic)

    for reason, fixture_refs in llm_failures_by_reason.items():
        if len(fixture_refs) == 1:
            reason_for_single = llm_reason_examples.get(reason, reason)
            summarized.append(
                f"Research LLM analysis failed for {fixture_refs[0]}: {reason_for_single}"
            )
            continue
        normalized_reason = reason.rstrip(".")
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Research LLM analysis failed for "
            f"{len(fixture_refs)} fixtures due to: {normalized_reason}. "
            f"Fallback context was used. Sample fixtures: {sample_refs}."
        )

    for reason, fixture_refs in tavily_failures_by_reason.items():
        if len(fixture_refs) == 1:
            reason_for_single = tavily_reason_examples.get(reason, reason)
            summarized.append(
                f"Tavily research fetch failed for {fixture_refs[0]}: {reason_for_single}"
            )
            continue
        normalized_reason = reason.rstrip(".")
        sample_refs = ", ".join(fixture_refs[:3])
        summarized.append(
            "Tavily research fetch failed for "
            f"{len(fixture_refs)} fixtures due to: {normalized_reason}. "
            f"Sample fixtures: {sample_refs}."
        )

    return tuple(summarized)


def _truncate(value: str, limit: int) -> str:
    """Clamp long strings to keep prompt and fallback text compact."""

    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _normalize_research_failure_reason(reason: str) -> str:
    """Normalize volatile provider error text into a stable root-cause string.

    Inputs:
        reason: Raw exception text captured during fixture-level research.

    Outputs:
        A compact reason string with request-specific identifiers removed so
        repeated failures aggregate into one operator-visible root cause.
    """

    normalized = " ".join(reason.split())
    lowered = normalized.casefold()

    if "rate limit" in lowered or "http 429" in lowered or "status code: 429" in lowered:
        return "LLM provider rate limit reached."
    if "timed out" in lowered or "timeout" in lowered:
        return "LLM request timed out."
    if (
        "network is unreachable" in lowered
        or "name resolution" in lowered
        or "connection error" in lowered
        or "connection refused" in lowered
    ):
        return "LLM provider connection failed."
    if (
        "authentication" in lowered
        or "unauthorized" in lowered
        or "http 401" in lowered
        or "invalid api key" in lowered
    ):
        return "LLM provider authentication failed."
    if (
        "json" in lowered
        or "validation" in lowered
        or "field required" in lowered
    ):
        return "LLM output schema validation failed."

    normalized = _REQUEST_ID_PATTERN.sub("request_id=<redacted>", normalized)
    normalized = _UUID_PATTERN.sub("<uuid>", normalized)
    normalized = _FIXTURE_IDENTIFIER_PATTERN.sub("<fixture>", normalized)
    return normalized


__all__ = ["DEFAULT_RESEARCH_BATCH_SIZE", "research_node"]
