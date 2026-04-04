"""RSS feed provider for PuntLab's fixture-aware sports news ingestion.

Purpose: collect recent sports articles from a configurable RSS catalog and
normalize them into fixture-linked `NewsArticle` objects for the research
stage.
Scope: feed fetching, RSS/Atom parsing, article normalization, fixture
relevance filtering, and duplicate suppression across publishers.
Dependencies: `feedparser` for XML parsing, `src.providers.base` for shared
HTTP caching and retries, and PuntLab's shared fixture/news schemas.
"""

from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final, cast

import feedparser  # type: ignore[import-untyped]
from pydantic import HttpUrl, TypeAdapter

from src.config import SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS: Final[int] = 3
_MAX_LOOKBACK_DAYS: Final[int] = 30
_DEFAULT_MAX_ENTRIES_PER_FEED: Final[int] = 25
_MAX_ENTRIES_PER_FEED: Final[int] = 100
_RSS_CACHE_TTL_SECONDS: Final[int] = 15 * 60
_RSS_ACCEPT_HEADER: Final[str] = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8"
)
_HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
_NON_ALNUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_TEAM_SUFFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(fc|cf|sc|afc|bc|bk|basketball club|football club)\b",
)
_GENERIC_FINAL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "afc",
        "athletic",
        "basketball",
        "bc",
        "cf",
        "city",
        "club",
        "county",
        "fc",
        "football",
        "inter",
        "real",
        "rovers",
        "sc",
        "sporting",
        "town",
        "united",
        "wanderers",
    }
)
_HTTP_URL_ADAPTER: Final[TypeAdapter[HttpUrl]] = TypeAdapter(HttpUrl)


@dataclass(frozen=True, slots=True)
class RSSFeedDefinition:
    """Static metadata describing one configured RSS or Atom feed.

    Inputs:
        Canonical source label plus the fetch URL and optional sport metadata.

    Outputs:
        An immutable feed definition that `RSSFeedProvider` can iterate over
        deterministically during one news-ingestion pass.
    """

    source: str
    url: str
    sport: SportName | None = None
    competition: str | None = None

    def __post_init__(self) -> None:
        """Reject blank feed labels and URLs early during provider setup."""

        if not self.source.strip():
            raise ValueError("RSS feed `source` must not be blank.")
        if not self.url.strip():
            raise ValueError("RSS feed `url` must not be blank.")


DEFAULT_RSS_FEEDS: Final[tuple[RSSFeedDefinition, ...]] = (
    RSSFeedDefinition(
        source="BBC Sport",
        url="https://feeds.bbci.co.uk/sport/football/rss.xml",
        sport=SportName.SOCCER,
    ),
    RSSFeedDefinition(
        source="ESPN",
        url="https://www.espn.com/espn/rss/soccer/news",
        sport=SportName.SOCCER,
    ),
    RSSFeedDefinition(
        source="ESPN",
        url="https://www.espn.com/espn/rss/nba/news",
        sport=SportName.BASKETBALL,
        competition="NBA",
    ),
    RSSFeedDefinition(
        source="Sky Sports",
        url="https://www.skysports.com/rss/12040",
        sport=SportName.SOCCER,
    ),
    # Goal.com's public site does not currently expose a stable RSS endpoint in
    # our validation checks, so we use a site-filtered Google News RSS feed to
    # keep Goal coverage inside the canonical feedparser-based ingestion path.
    RSSFeedDefinition(
        source="Goal.com",
        url=(
            "https://news.google.com/rss/search?q=site:goal.com+football"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        sport=SportName.SOCCER,
    ),
)


class RSSFeedProvider(DataProvider):
    """Fixture-aware RSS news provider used ahead of Tavily fallback search.

    Inputs:
        A shared `RateLimitedClient`, an optional configured feed catalog, and
        an optional clock used to bound recency filtering in tests.

    Outputs:
        `fetch_news()` returns normalized `NewsArticle` objects linked to the
        day's fixtures when the article text mentions relevant teams or
        competitions.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        feeds: Sequence[RSSFeedDefinition] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the provider with a validated feed catalog.

        Args:
            client: Shared `RateLimitedClient` used for cached RSS fetches.
            feeds: Optional explicit feed definitions. When omitted, the
                provider uses PuntLab's confirmed default RSS catalog.
            clock: Optional clock injector used for deterministic recency
                filtering in tests.

        Raises:
            ValueError: If the feed catalog is empty.
        """

        super().__init__(client)
        resolved_feeds = tuple(feeds or DEFAULT_RSS_FEEDS)
        if not resolved_feeds:
            raise ValueError("RSSFeedProvider requires at least one configured feed.")

        self._feeds = resolved_feeds
        self._clock = clock or (lambda: datetime.now(get_settings().timezone))

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in logs and cache keys."""

        return "rss-feeds"

    @property
    def base_url(self) -> str:
        """Return a placeholder base URL because this provider uses absolute URLs."""

        return "https://feeds.bbci.co.uk"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Return a conservative local politeness limit for RSS polling."""

        return RateLimitPolicy(limit=120, window_seconds=60)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Request XML-friendly content types from upstream RSS publishers."""

        return {"Accept": _RSS_ACCEPT_HEADER}

    @property
    def default_cache_ttl_seconds(self) -> int:
        """Cache RSS documents briefly to avoid repeated feed fetches."""

        return _RSS_CACHE_TTL_SECONDS

    async def fetch_news(
        self,
        *,
        fixtures: Sequence[NormalizedFixture],
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
        max_entries_per_feed: int = _DEFAULT_MAX_ENTRIES_PER_FEED,
    ) -> list[NewsArticle]:
        """Fetch and normalize relevant RSS articles for the supplied fixtures.

        Args:
            fixtures: Canonical fixtures being analyzed for the current run.
            lookback_days: Trailing recency window used to ignore stale news.
            max_entries_per_feed: Maximum number of parsed entries evaluated
                from each configured feed.

        Returns:
            A deduplicated list of fixture-relevant articles sorted by
            descending relevance and recency.

        Raises:
            ProviderError: If every configured feed fails to download or parse.
        """

        normalized_fixtures = tuple(fixtures)
        if not normalized_fixtures:
            return []

        resolved_lookback_days = self._validate_lookback_days(lookback_days)
        resolved_max_entries = self._validate_max_entries(max_entries_per_feed)
        cutoff = self._clock().astimezone(UTC) - timedelta(days=resolved_lookback_days)
        fixture_contexts = self._build_fixture_contexts(normalized_fixtures)

        collected_articles: list[NewsArticle] = []
        successful_feeds = 0
        feed_failures: list[str] = []

        for feed_definition in self._feeds:
            try:
                feed_articles = await self._fetch_feed_articles(
                    feed_definition=feed_definition,
                    fixture_contexts=fixture_contexts,
                    cutoff=cutoff,
                    max_entries_per_feed=resolved_max_entries,
                )
            except ProviderError as exc:
                logger.warning(
                    "RSS feed fetch failed for source=%s url=%s: %s",
                    feed_definition.source,
                    feed_definition.url,
                    exc,
                )
                feed_failures.append(f"{feed_definition.source}: {exc}")
                continue

            successful_feeds += 1
            collected_articles.extend(feed_articles)

        if successful_feeds == 0:
            detail = "; ".join(feed_failures) or "No RSS feeds could be fetched."
            raise ProviderError(self.provider_name, f"All configured RSS feeds failed. {detail}")

        deduplicated_articles = self._deduplicate_articles(collected_articles)
        return sorted(
            deduplicated_articles,
            key=lambda article: (
                article.relevance_score or 0.0,
                article.published_at,
            ),
            reverse=True,
        )

    async def _fetch_feed_articles(
        self,
        *,
        feed_definition: RSSFeedDefinition,
        fixture_contexts: Sequence[_FixtureContext],
        cutoff: datetime,
        max_entries_per_feed: int,
    ) -> list[NewsArticle]:
        """Fetch one feed and normalize its relevant entries into articles."""

        response = await self.fetch(
            "GET",
            feed_definition.url,
            cache_ttl_seconds=_RSS_CACHE_TTL_SECONDS,
        )

        parsed_feed = feedparser.parse(response.content)
        entries = cast(list[dict[str, Any]], list(parsed_feed.entries))

        if getattr(parsed_feed, "bozo", False) and not entries:
            bozo_exception = getattr(parsed_feed, "bozo_exception", None)
            detail = str(bozo_exception) if bozo_exception is not None else "malformed feed content"
            raise ProviderError(
                self.provider_name,
                f"Feed '{feed_definition.source}' could not be parsed: {detail}",
            )

        normalized_articles: list[NewsArticle] = []
        for raw_entry in entries[:max_entries_per_feed]:
            article = self._normalize_entry(
                raw_entry=raw_entry,
                feed_definition=feed_definition,
                fixture_contexts=fixture_contexts,
                cutoff=cutoff,
            )
            if article is not None:
                normalized_articles.append(article)

        return normalized_articles

    def _normalize_entry(
        self,
        *,
        raw_entry: Mapping[str, Any],
        feed_definition: RSSFeedDefinition,
        fixture_contexts: Sequence[_FixtureContext],
        cutoff: datetime,
    ) -> NewsArticle | None:
        """Normalize one RSS entry and discard it when it is stale or irrelevant."""

        headline = self._clean_text(raw_entry.get("title"))
        url = self._clean_text(raw_entry.get("link"))
        if headline is None or url is None:
            return None

        published_at = self._extract_published_at(raw_entry)
        if published_at is None or published_at < cutoff:
            return None

        summary = self._clean_text(
            raw_entry.get("summary") or raw_entry.get("description"),
        )
        content_snippet = self._extract_content_snippet(raw_entry)
        article_text = self._compose_article_text(headline, summary, content_snippet)
        match_result = self._match_article_to_fixture(article_text, fixture_contexts)
        if match_result is None:
            return None

        source = self._resolve_source(raw_entry, feed_definition)
        author = self._clean_text(raw_entry.get("author"))
        source_id = self._clean_text(raw_entry.get("id")) or url

        return NewsArticle(
            headline=headline,
            url=_HTTP_URL_ADAPTER.validate_python(url),
            published_at=published_at,
            source=source,
            source_provider="rss",
            summary=summary,
            content_snippet=content_snippet,
            sport=(
                match_result.fixture.sport
                if match_result.fixture is not None
                else feed_definition.sport
            ),
            competition=(
                match_result.fixture.competition
                if match_result.fixture is not None
                else feed_definition.competition
            ),
            teams=match_result.teams,
            fixture_ref=(
                match_result.fixture.get_fixture_ref()
                if match_result.fixture is not None
                else None
            ),
            author=author,
            source_id=source_id,
            relevance_score=match_result.relevance_score,
        )

    def _extract_published_at(self, raw_entry: Mapping[str, Any]) -> datetime | None:
        """Extract a timezone-aware publication timestamp from one RSS entry."""

        for field_name in ("published_parsed", "updated_parsed", "created_parsed"):
            parsed_struct = raw_entry.get(field_name)
            if isinstance(parsed_struct, time.struct_time):
                return datetime(*parsed_struct[:6], tzinfo=UTC)

        for field_name in ("published", "updated", "created"):
            parsed_value = self._clean_text(raw_entry.get(field_name))
            if parsed_value is None:
                continue
            try:
                parsed_datetime = parsedate_to_datetime(parsed_value)
            except Exception:
                continue
            if parsed_datetime.tzinfo is None:
                return parsed_datetime.replace(tzinfo=UTC)
            return parsed_datetime.astimezone(UTC)

        return None

    def _extract_content_snippet(self, raw_entry: Mapping[str, Any]) -> str | None:
        """Extract and clean the richest content snippet available on an RSS entry."""

        content_items = raw_entry.get("content")
        if isinstance(content_items, list):
            for content_item in content_items:
                if not isinstance(content_item, Mapping):
                    continue
                cleaned_value = self._clean_text(content_item.get("value"))
                if cleaned_value is not None:
                    return cleaned_value

        return self._clean_text(raw_entry.get("subtitle"))

    def _resolve_source(
        self,
        raw_entry: Mapping[str, Any],
        feed_definition: RSSFeedDefinition,
    ) -> str:
        """Resolve the human-readable publisher name for one normalized article."""

        source_data = raw_entry.get("source")
        if isinstance(source_data, Mapping):
            source_title = self._clean_text(source_data.get("title"))
            if source_title is not None:
                return source_title

        return feed_definition.source

    def _match_article_to_fixture(
        self,
        article_text: str,
        fixture_contexts: Sequence[_FixtureContext],
    ) -> _ArticleMatchResult | None:
        """Return the best fixture match for an article or `None` if ambiguous."""

        best_match: _ArticleMatchResult | None = None
        second_best_score = 0.0

        for fixture_context in fixture_contexts:
            match_score, matched_teams = self._score_fixture_match(article_text, fixture_context)
            if match_score <= 0:
                continue

            candidate = _ArticleMatchResult(
                fixture=fixture_context.fixture,
                teams=matched_teams,
                relevance_score=match_score,
            )

            if best_match is None or candidate.relevance_score > best_match.relevance_score:
                if best_match is not None:
                    second_best_score = best_match.relevance_score
                best_match = candidate
                continue

            if candidate.relevance_score > second_best_score:
                second_best_score = candidate.relevance_score

        if best_match is None:
            return None

        if (
            best_match.relevance_score < 0.65
            and abs(best_match.relevance_score - second_best_score) < 1e-9
        ):
            # Single-team articles can easily collide across the daily slate, so
            # ambiguous low-confidence matches are dropped instead of guessed.
            return None

        return best_match

    def _score_fixture_match(
        self,
        article_text: str,
        fixture_context: _FixtureContext,
    ) -> tuple[float, tuple[str, ...]]:
        """Score how strongly one article appears related to one fixture."""

        home_matched = self._matches_any_alias(article_text, fixture_context.home_aliases)
        away_matched = self._matches_any_alias(article_text, fixture_context.away_aliases)
        competition_matched = self._matches_any_alias(
            article_text,
            fixture_context.competition_aliases,
        )

        relevance_score = 0.0
        matched_teams: list[str] = []

        if home_matched and away_matched:
            relevance_score += 0.75
            matched_teams.extend(
                [fixture_context.fixture.home_team, fixture_context.fixture.away_team]
            )
        elif home_matched:
            relevance_score += 0.4
            matched_teams.append(fixture_context.fixture.home_team)
        elif away_matched:
            relevance_score += 0.4
            matched_teams.append(fixture_context.fixture.away_team)
        else:
            return 0.0, ()

        if competition_matched:
            relevance_score += 0.2

        if matched_teams and competition_matched:
            relevance_score += 0.05

        return min(relevance_score, 1.0), tuple(matched_teams)

    def _compose_article_text(
        self,
        headline: str,
        summary: str | None,
        content_snippet: str | None,
    ) -> str:
        """Build one normalized search corpus used for fixture matching."""

        combined_text = " ".join(
            part for part in (headline, summary, content_snippet) if part
        )
        return self._normalize_match_text(combined_text)

    def _matches_any_alias(self, article_text: str, aliases: Sequence[str]) -> bool:
        """Return whether any normalized alias appears in the article text."""

        padded_text = f" {article_text} "
        return any(f" {alias} " in padded_text for alias in aliases)

    def _build_fixture_contexts(
        self,
        fixtures: Sequence[NormalizedFixture],
    ) -> tuple[_FixtureContext, ...]:
        """Pre-compute normalized aliases for the fixture slate."""

        return tuple(
            _FixtureContext(
                fixture=fixture,
                home_aliases=self._build_aliases(fixture.home_team, allow_final_token_alias=True),
                away_aliases=self._build_aliases(fixture.away_team, allow_final_token_alias=True),
                competition_aliases=self._build_aliases(fixture.competition),
            )
            for fixture in fixtures
        )

    def _build_aliases(
        self,
        value: str,
        *,
        allow_final_token_alias: bool = False,
    ) -> tuple[str, ...]:
        """Build normalized aliases for team and competition matching."""

        primary_alias = self._normalize_match_text(value)
        aliases = [primary_alias]

        without_suffix = self._normalize_match_text(_TEAM_SUFFIX_PATTERN.sub(" ", value))
        if without_suffix and without_suffix != primary_alias:
            aliases.append(without_suffix)

        final_token = without_suffix.split()[-1] if without_suffix else ""
        if (
            allow_final_token_alias
            and final_token
            and final_token not in _GENERIC_FINAL_TOKENS
            and len(final_token) >= 5
        ):
            aliases.append(final_token)

        deduplicated_aliases: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            if not alias or alias in seen:
                continue
            seen.add(alias)
            deduplicated_aliases.append(alias)

        return tuple(deduplicated_aliases)

    def _normalize_match_text(self, value: str) -> str:
        """Normalize free text into a whitespace-padded matching representation."""

        normalized = html.unescape(value).casefold().replace("&", " and ")
        normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized

    def _clean_text(self, value: object) -> str | None:
        """Strip HTML and collapse empty article text values to `None`."""

        if not isinstance(value, str):
            return None

        without_tags = _HTML_TAG_PATTERN.sub(" ", html.unescape(value))
        normalized = _WHITESPACE_PATTERN.sub(" ", without_tags).strip()
        return normalized or None

    def _deduplicate_articles(self, articles: Sequence[NewsArticle]) -> list[NewsArticle]:
        """Keep the strongest copy of each article URL across overlapping feeds."""

        best_by_url: dict[str, NewsArticle] = {}

        for article in articles:
            existing_article = best_by_url.get(str(article.url))
            if existing_article is None:
                best_by_url[str(article.url)] = article
                continue

            existing_score = existing_article.relevance_score or 0.0
            candidate_score = article.relevance_score or 0.0
            if candidate_score > existing_score or (
                candidate_score == existing_score
                and article.published_at > existing_article.published_at
            ):
                best_by_url[str(article.url)] = article

        return list(best_by_url.values())

    def _validate_lookback_days(self, lookback_days: int) -> int:
        """Validate the recency window used to filter RSS articles."""

        if lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer.")
        if lookback_days > _MAX_LOOKBACK_DAYS:
            raise ValueError(
                f"lookback_days must be less than or equal to {_MAX_LOOKBACK_DAYS}."
            )
        return lookback_days

    def _validate_max_entries(self, max_entries_per_feed: int) -> int:
        """Validate the per-feed entry limit used during parsing."""

        if max_entries_per_feed <= 0:
            raise ValueError("max_entries_per_feed must be a positive integer.")
        if max_entries_per_feed > _MAX_ENTRIES_PER_FEED:
            raise ValueError(
                "max_entries_per_feed must be less than or equal to "
                f"{_MAX_ENTRIES_PER_FEED}."
            )
        return max_entries_per_feed


@dataclass(frozen=True, slots=True)
class _FixtureContext:
    """Precomputed normalized aliases used for fixture-article matching."""

    fixture: NormalizedFixture
    home_aliases: tuple[str, ...]
    away_aliases: tuple[str, ...]
    competition_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArticleMatchResult:
    """Internal match result used before constructing the public schema."""

    fixture: NormalizedFixture | None
    teams: tuple[str, ...]
    relevance_score: float


__all__ = ["DEFAULT_RSS_FEEDS", "RSSFeedDefinition", "RSSFeedProvider"]
