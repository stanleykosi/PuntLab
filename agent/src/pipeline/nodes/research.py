"""Research node for PuntLab's LangGraph pipeline.

Purpose: generate fixture-level qualitative context using the canonical
research prompt, structured `MatchContext` output, and optional Tavily search
enrichment on top of ingestion-stage news.
Scope: batch-oriented fixture research, evidence selection, fail-fast LLM
analysis, and diagnostic propagation for search failures.
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
from src.llm import MatchContext, get_llm, get_prompt
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.base import ProviderError, RateLimitedClient
from src.providers.tavily_search import TavilySearchProvider
from src.schemas.fixture_details import FixtureDetails
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.stats import InjuryData

DEFAULT_RESEARCH_BATCH_SIZE = 1
MAX_CONTEXT_ARTICLES = 6
MAX_TAVILY_RESULTS = 3
MAX_ARTICLE_AGE_DAYS = 7
MAX_FIXTURE_DETAIL_SECTION_LINES = 32
MAX_FIXTURE_DETAIL_TOTAL_LINES = 180
MIN_RELEVANCE_THRESHOLD = 0.20
LOCAL_SOURCE_LABEL = "PuntLab local evidence"
_RESEARCH_LLM_FAILURE_PATTERN = re.compile(
    r"^Research LLM analysis failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_TAVILY_RESEARCH_FAILURE_PATTERN = re.compile(
    r"^Tavily research fetch failed for (?P<fixture>.+?): (?P<reason>.+)$"
)
_FIXTURE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:sr:match:\d+|api-football:\d+|balldontlie:\d+)\b",
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
    resolved_llm = llm
    if resolved_llm is None:
        resolved_llm = await get_llm("research")

    owned_cleanup: Callable[[], Awaitable[None]] | None = None
    resolved_tavily = tavily_provider
    diagnostics: list[str] = []
    if resolved_tavily is None:
        resolved_tavily, owned_cleanup, provider_diagnostic = _build_optional_tavily_provider()
        if provider_diagnostic is not None:
            diagnostics.append(provider_diagnostic)

    try:
        fixture_details_by_fixture = {
            details.fixture_ref: details for details in validated_state.fixture_details
        }
        match_contexts: list[MatchContext] = []
        for fixture_batch in _iter_batches(validated_state.fixtures, batch_size):
            batch_results = await asyncio.gather(
                *[
                    _research_fixture(
                        fixture=fixture,
                        state_news=validated_state.news_articles,
                        injuries=validated_state.injuries,
                        fixture_details=fixture_details_by_fixture.get(
                            fixture.get_fixture_ref()
                        ),
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
    fixture_details: FixtureDetails | None,
    llm: BaseChatModel | None,
    tavily_provider: TavilySearchProvider | None,
) -> FixtureResearchResult:
    """Build one conservative `MatchContext` for a fixture.

    Inputs:
        fixture: Canonical fixture being researched.
        state_news: Ingestion-stage news articles already collected for the run.
        injuries: Structured injury rows available for the full run.
        llm: Research LLM. Missing models are not allowed in the canonical path.
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
        raise RuntimeError("research_node requires an LLM; deterministic fallback is disabled.")
    if fixture_details is None:
        raise RuntimeError(
            f"research_node requires SportyBet fixture details for {fixture.get_fixture_ref()}."
        )

    prompt = get_prompt("research")
    rendered_fixture_details = _render_fixture_details(fixture_details)
    source_labels = _source_labels(evidence_articles, fixture_injuries, fixture_details)
    prompt_messages = prompt.format_messages(
        run_date=fixture.kickoff.astimezone(WAT_TIMEZONE).date().isoformat(),
        fixture_summary=_build_fixture_summary(fixture),
        competition_context=_build_competition_context(fixture),
        kickoff_context=_build_kickoff_context(fixture),
        known_absences=_render_known_absences(fixture, fixture_injuries),
        fixture_details=rendered_fixture_details,
        recent_news_bullets=_render_news_bullets(evidence_articles),
        source_labels=", ".join(source_labels),
    )

    context = await _invoke_structured_match_context(
        llm=llm,
        prompt_messages=prompt_messages,
    )
    context = context.model_copy(
        update={
            "fixture_ref": fixture.get_fixture_ref(),
            "data_sources": source_labels,
        }
    )

    return FixtureResearchResult(context=context, diagnostics=tuple(diagnostics))


async def _invoke_structured_match_context(
    *,
    llm: BaseChatModel,
    prompt_messages: Sequence[Any],
) -> MatchContext:
    """Invoke the research LLM with prompt-level JSON enforcement.

    Inputs:
        llm: Configured chat model resolved for the research task.
        prompt_messages: Prompt message sequence produced by the canonical
            research prompt template.

    Outputs:
        A validated `MatchContext` object.

    Raises:
        Exception: Propagates provider or parsing errors; no fallback is used.
    """

    return await invoke_json_schema(
        llm=llm,
        prompt_messages=prompt_messages,
        schema=MatchContext,
        instruction=_match_context_json_instruction(),
    )


def _match_context_json_instruction() -> str:
    """Return the JSON-only instruction used for OpenRouter research models."""

    return (
        "Return only one valid JSON object. Do not use markdown. "
        "Required keys: fixture_detail_summary, tactical_context, statistical_context, "
        "availability_context, market_context, supplemental_news_context, qualitative_score, "
        "data_sources. Use null for optional context fields when the supplied widgets do "
        "not support them. Do not include any keys outside the schema. Use a number from "
        "0 to 1 for qualitative_score."
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
    fixture_details: FixtureDetails | None = None,
) -> tuple[str, ...]:
    """Build ordered unique source labels grounded in evidence, not model output."""

    labels: list[str] = []
    seen: set[str] = set()

    if fixture_details is not None:
        labels.append("SportyBet fixture-page widgets")
        seen.add("sportybet fixture-page widgets")

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

    return tuple(labels) if labels else (LOCAL_SOURCE_LABEL,)


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


def _render_fixture_details(details: FixtureDetails | None) -> str:
    """Render SportyBet fixture-page details compactly for one LLM prompt."""

    if details is None or not details.sections:
        return "No SportyBet fixture-page details were captured for this fixture."

    lines = [
        (
            "SportyBet fixture-page coverage: "
            f"{len(details.sections)} sections; widget loader {details.widget_loader_status}; "
            f"fetched {details.fetched_at.isoformat()}."
        )
    ]
    emitted_detail_lines = 0
    for section in details.sections:
        section_lines = section.content_lines[:MAX_FIXTURE_DETAIL_SECTION_LINES]
        if not section_lines:
            error = section.error_message or "no readable content"
            lines.append(f"{section.widget_key}: {section.status} ({error}).")
            continue

        lines.append(f"{section.widget_key}:")
        for content_line in section_lines:
            if emitted_detail_lines >= MAX_FIXTURE_DETAIL_TOTAL_LINES:
                lines.append("- Additional fixture-page lines omitted to keep prompt size bounded.")
                return "\n".join(lines)
            lines.append(f"- {content_line}")
            emitted_detail_lines += 1

        omitted_count = len(section.content_lines) - len(section_lines)
        if omitted_count > 0:
            lines.append(
                f"- {omitted_count} additional {section.widget_key} lines omitted."
            )

    return "\n".join(lines)


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
