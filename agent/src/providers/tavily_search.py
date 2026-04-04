"""Tavily search provider for PuntLab's qualitative news enrichment layer.

Purpose: connect Tavily's web-search API to PuntLab's shared provider
infrastructure so match previews, injury updates, and breaking sports context
can be fetched with the same caching and rate-limit controls as other
providers.
Scope: authenticated Tavily search requests, sports-focused query helpers, and
normalization of search results into `NewsArticle` schemas.
Dependencies: `src.providers.base` for shared HTTP behavior, `src.config` for
credentials and sport metadata, and `src.schemas.news.NewsArticle` for the
canonical research-stage output contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from typing import Final, Literal, cast
from urllib.parse import urlparse

from pydantic import HttpUrl, TypeAdapter

from src.config import SportName, get_settings
from src.providers.base import DataProvider, ProviderError, RateLimitedClient, RateLimitPolicy
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle

logger = logging.getLogger(__name__)

SearchDepth = Literal["basic", "advanced"]

_MONTH_WINDOW_SECONDS: Final[int] = 30 * 24 * 60 * 60
_SEARCH_CACHE_TTL_SECONDS: Final[int] = 15 * 60
_DEFAULT_MAX_RESULTS: Final[int] = 5
_MAX_RESULTS_LIMIT: Final[int] = 20
_DEFAULT_MATCH_LOOKBACK_DAYS: Final[int] = 7
_DEFAULT_BREAKING_LOOKBACK_DAYS: Final[int] = 2
_MAX_INCLUDE_DOMAINS: Final[int] = 300
_MAX_EXCLUDE_DOMAINS: Final[int] = 150
_HTTP_URL_ADAPTER: Final[TypeAdapter[HttpUrl]] = TypeAdapter(HttpUrl)
_SOURCE_NAME_OVERRIDES: Final[dict[str, str]] = {
    "bbc.co.uk": "BBC Sport",
    "bbc.com": "BBC Sport",
    "espn.com": "ESPN",
    "goal.com": "Goal.com",
    "skysports.com": "Sky Sports",
    "nba.com": "NBA",
    "premierleague.com": "Premier League",
    "uefa.com": "UEFA",
}


class TavilySearchProvider(DataProvider):
    """Concrete Tavily integration for real-time sports news search.

    Inputs:
        A shared `RateLimitedClient`, a Tavily API key, and optional defaults
        controlling result count, search depth, and clock behavior.

    Outputs:
        Search helpers that return normalized `NewsArticle` objects for
        match-specific coverage, injury updates, and broader breaking news.
    """

    def __init__(
        self,
        client: RateLimitedClient,
        *,
        api_key: str | None = None,
        default_max_results: int = _DEFAULT_MAX_RESULTS,
        default_search_depth: SearchDepth = "basic",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the provider with validated credentials and defaults.

        Args:
            client: Shared `RateLimitedClient` used for cached and retried HTTP
                requests.
            api_key: Optional explicit Tavily API key. When omitted, the
                provider falls back to `TAVILY_API_KEY` from settings.
            default_max_results: Default result count used by search helpers.
            default_search_depth: Default Tavily search depth.
            clock: Optional clock injection used for deterministic date-window
                calculations in tests.

        Raises:
            ValueError: If no usable API key is available or defaults are
                outside Tavily's documented request bounds.
        """

        super().__init__(client)
        resolved_api_key = (api_key or get_settings().data_providers.tavily_api_key or "").strip()
        if not resolved_api_key:
            raise ValueError("Tavily requires `TAVILY_API_KEY` or an explicit `api_key`.")

        self._api_key = resolved_api_key
        self._default_max_results = self._validate_max_results(default_max_results)
        self._default_search_depth = self._validate_search_depth(default_search_depth)
        self._clock = clock or (lambda: datetime.now(get_settings().timezone))

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used in cache keys and logs."""

        return "tavily"

    @property
    def base_url(self) -> str:
        """Return the canonical Tavily API base URL."""

        return "https://api.tavily.com"

    @property
    def rate_limit_policy(self) -> RateLimitPolicy:
        """Approximate Tavily's free monthly quota with a rolling monthly bucket."""

        return RateLimitPolicy(limit=1000, window_seconds=_MONTH_WINDOW_SECONDS)

    @property
    def default_headers(self) -> Mapping[str, str]:
        """Return the canonical Tavily authentication and JSON headers."""

        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @property
    def default_cache_ttl_seconds(self) -> int:
        """Cache Tavily search responses briefly to conserve monthly quota."""

        return _SEARCH_CACHE_TTL_SECONDS

    async def search_news(
        self,
        *,
        query: str,
        sport: SportName | None = None,
        competition: str | None = None,
        teams: Sequence[str] = (),
        fixture_ref: str | None = None,
        max_results: int | None = None,
        search_depth: SearchDepth | None = None,
        lookback_days: int = _DEFAULT_MATCH_LOOKBACK_DAYS,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        exact_match: bool = False,
    ) -> list[NewsArticle]:
        """Search Tavily news results and normalize them to PuntLab articles.

        Args:
            query: Search query sent to Tavily.
            sport: Optional sport tag attached to normalized articles.
            competition: Optional competition tag attached to articles.
            teams: Optional ordered team names linked to the search context.
            fixture_ref: Optional canonical fixture reference for article
                association.
            max_results: Optional Tavily max-results override.
            search_depth: Optional Tavily search-depth override.
            lookback_days: Number of trailing days to search, inclusive of the
                provider call date.
            include_domains: Optional domain allow-list forwarded to Tavily.
            exclude_domains: Optional domain deny-list forwarded to Tavily.
            exact_match: Whether Tavily should enforce exact-match semantics.

        Returns:
            A list of normalized `NewsArticle` instances sorted in Tavily's
            relevance order.
        """

        normalized_query = self._require_text(query, "query")
        resolved_max_results = self._resolve_max_results(max_results)
        if resolved_max_results == 0:
            return []

        payload = self._build_search_payload(
            query=normalized_query,
            max_results=resolved_max_results,
            search_depth=search_depth or self._default_search_depth,
            lookback_days=lookback_days,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            exact_match=exact_match,
        )
        response_payload = await self._fetch_search_payload(payload)
        return self._normalize_articles(
            response_payload,
            sport=sport,
            competition=competition,
            teams=teams,
            fixture_ref=fixture_ref,
        )

    async def search_match_news(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int | None = None,
        search_depth: SearchDepth | None = None,
        lookback_days: int = _DEFAULT_MATCH_LOOKBACK_DAYS,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[NewsArticle]:
        """Search recent preview and context coverage for one fixture.

        Args:
            fixture: Canonical fixture driving the search query and article
                tagging.
            max_results: Optional Tavily max-results override.
            search_depth: Optional Tavily search-depth override.
            lookback_days: Trailing lookback window used for published-date
                filtering.
            include_domains: Optional domain allow-list.
            exclude_domains: Optional domain deny-list.

        Returns:
            A list of fixture-linked articles relevant to the match.
        """

        query = (
            f"{fixture.home_team} {fixture.away_team} {fixture.competition} "
            "match preview team news"
        )
        return await self.search_news(
            query=query,
            sport=fixture.sport,
            competition=fixture.competition,
            teams=(fixture.home_team, fixture.away_team),
            fixture_ref=fixture.get_fixture_ref(),
            max_results=max_results,
            search_depth=search_depth,
            lookback_days=lookback_days,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    async def search_injury_updates(
        self,
        *,
        fixture: NormalizedFixture,
        max_results: int | None = None,
        search_depth: SearchDepth | None = None,
        lookback_days: int = _DEFAULT_MATCH_LOOKBACK_DAYS,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[NewsArticle]:
        """Search injury, suspension, and lineup context for one fixture.

        Args:
            fixture: Canonical fixture driving the search query and article
                tagging.
            max_results: Optional Tavily max-results override.
            search_depth: Optional Tavily search-depth override.
            lookback_days: Trailing lookback window used for published-date
                filtering.
            include_domains: Optional domain allow-list.
            exclude_domains: Optional domain deny-list.

        Returns:
            A list of fixture-linked injury or availability related articles.
        """

        query = (
            f"{fixture.home_team} {fixture.away_team} {fixture.competition} "
            "injuries suspensions lineup team news"
        )
        return await self.search_news(
            query=query,
            sport=fixture.sport,
            competition=fixture.competition,
            teams=(fixture.home_team, fixture.away_team),
            fixture_ref=fixture.get_fixture_ref(),
            max_results=max_results,
            search_depth=search_depth,
            lookback_days=lookback_days,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    async def search_breaking_news(
        self,
        *,
        sport: SportName,
        competition: str | None = None,
        teams: Sequence[str] = (),
        max_results: int | None = None,
        search_depth: SearchDepth | None = None,
        lookback_days: int = _DEFAULT_BREAKING_LOOKBACK_DAYS,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[NewsArticle]:
        """Search recent breaking news for one sport or competition context.

        Args:
            sport: Supported PuntLab sport attached to normalized articles.
            competition: Optional competition to focus the breaking-news query.
            teams: Optional team names to attach to the results.
            max_results: Optional Tavily max-results override.
            search_depth: Optional Tavily search-depth override.
            lookback_days: Trailing lookback window used for published-date
                filtering.
            include_domains: Optional domain allow-list.
            exclude_domains: Optional domain deny-list.

        Returns:
            A list of normalized breaking-news articles.
        """

        subject = competition or ("NBA" if sport == SportName.BASKETBALL else "soccer")
        query = f"{subject} breaking news"
        return await self.search_news(
            query=query,
            sport=sport,
            competition=competition,
            teams=teams,
            max_results=max_results,
            search_depth=search_depth,
            lookback_days=lookback_days,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    async def _fetch_search_payload(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Execute one Tavily search request and validate the JSON envelope."""

        response = await self.fetch("POST", "/search", json=dict(payload))
        try:
            decoded_payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider_name,
                "Tavily returned a non-JSON search response.",
                cause=exc,
            ) from exc

        if not isinstance(decoded_payload, Mapping):
            raise ProviderError(
                self.provider_name,
                "Tavily search responses must decode to a JSON object.",
            )

        return cast(Mapping[str, object], decoded_payload)

    def _build_search_payload(
        self,
        *,
        query: str,
        max_results: int,
        search_depth: SearchDepth,
        lookback_days: int,
        include_domains: Sequence[str] | None,
        exclude_domains: Sequence[str] | None,
        exact_match: bool,
    ) -> dict[str, object]:
        """Build a validated Tavily search request payload."""

        start_date, end_date = self._resolve_date_window(lookback_days)
        payload: dict[str, object] = {
            "query": query,
            "topic": "news",
            "search_depth": self._validate_search_depth(search_depth),
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "exact_match": exact_match,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        normalized_include_domains = self._normalize_domain_filters(
            include_domains,
            field_name="include_domains",
            max_count=_MAX_INCLUDE_DOMAINS,
        )
        if normalized_include_domains:
            payload["include_domains"] = normalized_include_domains

        normalized_exclude_domains = self._normalize_domain_filters(
            exclude_domains,
            field_name="exclude_domains",
            max_count=_MAX_EXCLUDE_DOMAINS,
        )
        if normalized_exclude_domains:
            payload["exclude_domains"] = normalized_exclude_domains

        return payload

    def _normalize_articles(
        self,
        payload: Mapping[str, object],
        *,
        sport: SportName | None,
        competition: str | None,
        teams: Sequence[str],
        fixture_ref: str | None,
    ) -> list[NewsArticle]:
        """Convert Tavily result objects into canonical `NewsArticle` models."""

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ProviderError(
                self.provider_name,
                "Tavily search response did not include a `results` list.",
            )

        normalized_teams = self._normalize_team_list(teams)
        request_id = self._normalize_optional_text(payload.get("request_id"))
        articles: list[NewsArticle] = []

        for index, raw_result in enumerate(raw_results):
            if not isinstance(raw_result, Mapping):
                raise ProviderError(
                    self.provider_name,
                    f"Tavily search result at index {index} must be an object.",
                )

            article = self._normalize_article(
                raw_result,
                index=index,
                request_id=request_id,
                sport=sport,
                competition=competition,
                teams=normalized_teams,
                fixture_ref=fixture_ref,
            )
            if article is not None:
                articles.append(article)

        return articles

    def _normalize_article(
        self,
        payload: Mapping[str, object],
        *,
        index: int,
        request_id: str | None,
        sport: SportName | None,
        competition: str | None,
        teams: tuple[str, ...],
        fixture_ref: str | None,
    ) -> NewsArticle | None:
        """Normalize one Tavily result object to a `NewsArticle`.

        Invalid result rows are skipped with a warning so one malformed source
        does not discard the rest of a useful search response.
        """

        result_label = f"results[{index}]"
        try:
            url = self._require_text(payload.get("url"), f"{result_label}.url")
            headline = self._require_text(payload.get("title"), f"{result_label}.title")
            published_at = self._parse_published_at(payload.get("published_date"), result_label)
        except ValueError as exc:
            logger.warning(
                "Skipping malformed Tavily result for %s at index %s: %s",
                request_id or "request",
                index,
                exc,
            )
            return None

        content_snippet = self._normalize_optional_text(payload.get("content"))
        relevance_score = self._normalize_relevance_score(payload.get("score"), result_label)
        validated_url = _HTTP_URL_ADAPTER.validate_python(url)

        return NewsArticle(
            headline=headline,
            url=validated_url,
            published_at=published_at,
            source=self._derive_source_name(url),
            source_provider=self.provider_name,
            summary=self._build_summary(content_snippet),
            content_snippet=content_snippet,
            sport=sport,
            competition=competition,
            teams=teams,
            fixture_ref=fixture_ref,
            source_id=f"{request_id}:{index}" if request_id else url,
            relevance_score=relevance_score,
        )

    def _resolve_date_window(self, lookback_days: int) -> tuple[date, date]:
        """Build an inclusive published-date window for Tavily news search."""

        if lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer.")

        end_date = self._clock().astimezone(get_settings().timezone).date()
        start_date = end_date - timedelta(days=lookback_days - 1)
        return start_date, end_date

    def _resolve_max_results(self, max_results: int | None) -> int:
        """Resolve a search result limit from call-time or provider defaults."""

        if max_results is None:
            return self._default_max_results
        return self._validate_max_results(max_results)

    @staticmethod
    def _validate_max_results(max_results: int) -> int:
        """Validate Tavily's documented max-results bounds."""

        if max_results < 0 or max_results > _MAX_RESULTS_LIMIT:
            raise ValueError(
                f"max_results must be between 0 and {_MAX_RESULTS_LIMIT}, inclusive."
            )
        return max_results

    @staticmethod
    def _validate_search_depth(search_depth: SearchDepth) -> SearchDepth:
        """Validate supported Tavily search-depth options."""

        if search_depth not in {"basic", "advanced"}:
            raise ValueError("search_depth must be either 'basic' or 'advanced'.")
        return search_depth

    def _normalize_domain_filters(
        self,
        domains: Sequence[str] | None,
        *,
        field_name: str,
        max_count: int,
    ) -> tuple[str, ...]:
        """Normalize domain allow/deny lists into Tavily-compatible hostnames."""

        if domains is None:
            return ()

        normalized_domains: list[str] = []
        seen: set[str] = set()
        for raw_domain in domains:
            domain = self._normalize_domain(raw_domain)
            if domain in seen:
                continue
            seen.add(domain)
            normalized_domains.append(domain)

        if len(normalized_domains) > max_count:
            raise ValueError(f"{field_name} supports at most {max_count} domains.")
        return tuple(normalized_domains)

    @staticmethod
    def _normalize_domain(value: str) -> str:
        """Normalize one domain or URL into the bare hostname Tavily expects."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("domain filters must not be blank.")

        parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
        host = parsed.netloc or parsed.path
        normalized_host = host.removeprefix("www.").strip(".")
        if not normalized_host:
            raise ValueError("domain filters must include a hostname.")
        return normalized_host

    @staticmethod
    def _normalize_team_list(teams: Sequence[str]) -> tuple[str, ...]:
        """Normalize, deduplicate, and preserve the order of team labels."""

        normalized_teams: list[str] = []
        seen: set[str] = set()
        for raw_team in teams:
            team_name = raw_team.strip()
            if not team_name:
                raise ValueError("team names must not be blank.")
            lookup_key = team_name.casefold()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            normalized_teams.append(team_name)
        return tuple(normalized_teams)

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        """Trim optional string-like values and collapse blanks to `None`."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Expected text or None, received {type(value).__name__}.")

        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _require_text(value: object, field_name: str) -> str:
        """Trim and validate required provider text fields."""

        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank.")
        return normalized

    @staticmethod
    def _parse_published_at(value: object, result_label: str) -> datetime:
        """Parse Tavily's optional `published_date` into a timezone-aware datetime."""

        raw_value = TavilySearchProvider._require_text(
            value,
            f"{result_label}.published_date",
        )
        normalized_value = raw_value.replace("Z", "+00:00")

        try:
            parsed_value = datetime.fromisoformat(normalized_value)
        except ValueError:
            try:
                parsed_date = date.fromisoformat(normalized_value)
            except ValueError as exc:
                raise ValueError(
                    f"{result_label}.published_date must be an ISO date or datetime."
                ) from exc
            return datetime.combine(parsed_date, datetime.min.time(), tzinfo=UTC)

        if parsed_value.tzinfo is None:
            return parsed_value.replace(tzinfo=UTC)
        return parsed_value

    @staticmethod
    def _normalize_relevance_score(value: object, result_label: str) -> float | None:
        """Normalize Tavily's optional relevance score into the schema bounds."""

        if value is None:
            return None

        numeric_score: float
        if isinstance(value, bool):
            logger.warning(
                "Ignoring boolean score for Tavily result %s because scores must be numeric.",
                result_label,
            )
            return None

        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                return None
            try:
                numeric_score = float(stripped_value)
            except ValueError:
                logger.warning(
                    "Ignoring non-numeric Tavily score for %s: %r",
                    result_label,
                    value,
                )
                return None
        elif isinstance(value, (int, float)):
            numeric_score = float(value)
        else:
            logger.warning(
                "Ignoring unsupported Tavily score type for %s: %s",
                result_label,
                type(value).__name__,
            )
            return None

        if not isfinite(numeric_score):
            logger.warning("Ignoring non-finite Tavily score for %s.", result_label)
            return None
        if not 0.0 <= numeric_score <= 1.0:
            logger.warning(
                "Ignoring out-of-range Tavily score for %s: %s",
                result_label,
                numeric_score,
            )
            return None
        return numeric_score

    @staticmethod
    def _build_summary(content_snippet: str | None) -> str | None:
        """Generate a concise article summary from Tavily's content snippet."""

        if content_snippet is None:
            return None
        if len(content_snippet) <= 280:
            return content_snippet
        truncated = content_snippet[:280].rsplit(" ", 1)[0].strip()
        return truncated or content_snippet[:280]

    @staticmethod
    def _derive_source_name(url: str) -> str:
        """Derive a stable publisher label from one result URL."""

        host = urlparse(url).netloc.lower().removeprefix("www.")
        if host in _SOURCE_NAME_OVERRIDES:
            return _SOURCE_NAME_OVERRIDES[host]

        segments = [segment for segment in host.split(".") if segment and segment != "com"]
        if not segments:
            return "Unknown Source"

        primary_segment = segments[0]
        words = primary_segment.replace("-", " ").split()
        if not words:
            return primary_segment.upper()
        return " ".join(word.upper() if len(word) <= 4 else word.capitalize() for word in words)


__all__ = ["TavilySearchProvider"]
