"""Playwright-based SportyBet fallback scraper for PuntLab market resolution.

Purpose: provide the slower but more resilient browser fallback used when the
primary HTTP interceptor cannot resolve SportyBet fixture markets directly.
Scope: open public SportyBet match pages, wait for market content, extract
embedded event payloads or visible DOM markets, and normalize them into the
shared `NormalizedOdds` contract.
Dependencies: Playwright for browser automation, Redis-backed cache helpers,
the canonical SportyBet API parser for market normalization, and fixture/odds
schemas shared across the agent pipeline.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal, Protocol, cast
from urllib.parse import unquote, urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, SportName
from src.providers.base import ProviderError
from src.schemas.common import normalize_optional_text, require_non_blank_text
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SportyBetAPIClient,
    SportyBetMarketCacheEntry,
)

logger = logging.getLogger(__name__)
_SPORTRADAR_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"sr:match:\d+")
_TEAM_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?i)(?:_vs_|-vs-| vs | v )")
DEFAULT_MARKET_WAIT_SELECTORS: Final[tuple[str, ...]] = (
    "[data-market-id]",
    "[data-testid*='market']",
    "[class*='market']",
    "[class*='coupon']",
)
DOM_MARKET_EXTRACTOR_SCRIPT: Final[str] = """
() => {
  const compact = (value) => typeof value === "string"
    ? value.replace(/\\s+/g, " ").trim()
    : "";

  const attrText = (element, names) => {
    if (!element) {
      return "";
    }
    for (const name of names) {
      const rawValue = element.getAttribute(name);
      if (typeof rawValue === "string") {
        const normalized = compact(rawValue);
        if (normalized) {
          return normalized;
        }
      }
    }
    return "";
  };

  const textContent = (element) => compact(element?.textContent || "");

  const looksLikeEvent = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return false;
    }
    const markets = value.markets;
    if (!Array.isArray(markets) || markets.length === 0) {
      return false;
    }
    const homeTeam = value.homeTeamName || value.homeName || value.homeTeam;
    const awayTeam = value.awayTeamName || value.awayName || value.awayTeam;
    return typeof homeTeam === "string" && typeof awayTeam === "string";
  };

  const eventPayloads = [];
  const seenEventKeys = new Set();

  const walkJson = (value, depth = 0) => {
    if (depth > 7 || eventPayloads.length >= 25) {
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        walkJson(item, depth + 1);
      }
      return;
    }
    if (!value || typeof value !== "object") {
      return;
    }

    if (looksLikeEvent(value)) {
      const eventKey = JSON.stringify([
        value.eventId ?? value.matchId ?? value.id ?? "",
        value.homeTeamName ?? value.homeName ?? value.homeTeam ?? "",
        value.awayTeamName ?? value.awayName ?? value.awayTeam ?? "",
      ]);
      if (!seenEventKeys.has(eventKey)) {
        seenEventKeys.add(eventKey);
        eventPayloads.push(value);
      }
    }

    for (const nestedValue of Object.values(value).slice(0, 60)) {
      walkJson(nestedValue, depth + 1);
    }
  };

  for (const script of document.querySelectorAll("script:not([src])")) {
    const rawContent = compact(script.textContent || "");
    if (!rawContent || rawContent.length > 500000) {
      continue;
    }
    const firstCharacter = rawContent[0];
    if (firstCharacter !== "{" && firstCharacter !== "[") {
      continue;
    }
    try {
      walkJson(JSON.parse(rawContent));
    } catch {
      continue;
    }
  }

  const findHeading = () => {
    for (const selector of [
      "h1",
      "[data-testid='match-title']",
      "[class*='match'][class*='title']",
      "[class*='event'][class*='title']",
    ]) {
      const value = textContent(document.querySelector(selector));
      if (value) {
        return value;
      }
    }
    return "";
  };

  const competition = (() => {
    for (const selector of [
      "[data-testid='breadcrumb']",
      "[class*='breadcrumb']",
      "[class*='league']",
      "[class*='competition']",
      "[class*='tournament']",
    ]) {
      const value = textContent(document.querySelector(selector));
      if (value) {
        return value;
      }
    }
    return "";
  })();

  const outcomeSelectors = [
    "button",
    "[role='button']",
    "[data-testid*='outcome']",
    "[class*='selection']",
    "[class*='outcome']",
    "[class*='option']",
  ].join(",");

  const parseOutcomeNode = (element) => {
    const rawText = textContent(element);
    if (!rawText) {
      return null;
    }

    const oddsFromAttributes = attrText(element, ["data-odds", "data-price", "data-odd"]);
    const oddsSource = oddsFromAttributes || rawText;
    const matches = [...oddsSource.matchAll(/\\d+(?:\\.\\d+)?/g)];
    const odds = matches.length > 0 ? matches[matches.length - 1][0] : "";
    if (!odds) {
      return null;
    }

    const selectionName = compact(rawText.replace(odds, " "));
    if (!selectionName) {
      return null;
    }

    const className = typeof element.className === "string" ? element.className.toLowerCase() : "";
    const ariaDisabled = element.getAttribute("aria-disabled");
    const isInactive = element.hasAttribute("disabled")
      || ariaDisabled === "true"
      || /disabled|suspended|inactive|closed/.test(className);

    return {
      selection_id: attrText(element, ["data-selection-id", "data-outcome-id", "data-id"]) || null,
      selection_name: selectionName,
      odds,
      status: isInactive ? "inactive" : "active",
    };
  };

  const marketNodes = [];
  for (const selector of [
    "[data-market-id]",
    "[data-testid*='market']",
    "section[class*='market']",
    "div[class*='market-group']",
    "div[class*='market']",
    "section[class*='coupon']",
    "div[class*='coupon']",
  ]) {
    for (const node of document.querySelectorAll(selector)) {
      marketNodes.push(node);
    }
  }

  const domMarkets = [];
  const seenMarketKeys = new Set();

  for (const node of marketNodes) {
    const marketName = attrText(node, ["data-market-name", "aria-label"])
      || textContent(
        node.querySelector(
          "h2, h3, h4, [role='heading'], [data-testid*='market-name'], "
          + "[class*='market'][class*='title'], [class*='market'][class*='name']",
        ),
      );
    if (!marketName) {
      continue;
    }

    const outcomes = [];
    const seenOutcomeKeys = new Set();
    for (const outcomeNode of node.querySelectorAll(outcomeSelectors)) {
      const outcome = parseOutcomeNode(outcomeNode);
      if (!outcome) {
        continue;
      }
      const outcomeKey = `${outcome.selection_name}|${outcome.odds}`;
      if (seenOutcomeKeys.has(outcomeKey)) {
        continue;
      }
      seenOutcomeKeys.add(outcomeKey);
      outcomes.push(outcome);
    }

    if (outcomes.length < 2) {
      continue;
    }

    const marketId = attrText(node, ["data-market-id", "data-id"]) || null;
    const specifier = attrText(node, ["data-specifier", "data-market-specifier"]) || null;
    const outcomeNames = outcomes.map((outcome) => outcome.selection_name).join("|");
    const marketKey = `${marketId || marketName}|${outcomeNames}`;
    if (seenMarketKeys.has(marketKey)) {
      continue;
    }
    seenMarketKeys.add(marketKey);

    domMarkets.push({
      market_id: marketId,
      market_name: marketName,
      specifier,
      outcomes,
    });
  }

  const sport = window.location.pathname.includes("/basketball/") ? "basketball"
    : window.location.pathname.includes("/football/") ? "football"
    : "";

  return {
    title: compact(document.title || ""),
    heading: findHeading(),
    competition,
    sport,
    home_team: "",
    away_team: "",
    event_payloads: eventPayloads,
    dom_markets: domMarkets,
  };
}
"""


class SupportsScrapePage(Protocol):
    """Protocol describing the page APIs the browser scraper requires."""

    async def goto(
        self,
        url: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] | None = None,
        referer: str | None = None,
    ) -> object:
        """Navigate to a page and wait for the requested lifecycle event."""

    async def wait_for_load_state(
        self,
        state: Literal["domcontentloaded", "load", "networkidle"] | None = None,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> None:
        """Wait for the page to reach a Playwright load state."""

    async def wait_for_selector(
        self,
        selector: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> object | None:
        """Wait for one selector to appear in the current DOM."""

    async def evaluate(self, expression: str) -> object:
        """Evaluate JavaScript in the browser page and return JSON-safe data."""


type PageSessionFactory = Callable[
    [str],
    AbstractAsyncContextManager[SupportsScrapePage],
]


class BrowserOutcomeSnapshot(BaseModel):
    """Validated DOM outcome snapshot before canonical normalization.

    Inputs:
        One raw outcome object returned by the browser evaluation script.

    Outputs:
        A typed, sanitized snapshot that can be transformed into the same
        SportyBet market shape used by the HTTP interceptor parser.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    selection_name: str = Field(description="Visible outcome label from the page.")
    odds: str | int | float = Field(description="Raw odds text or numeric value.")
    selection_id: str | int | None = Field(
        default=None,
        description="Optional DOM-level outcome identifier.",
    )
    status: str | int | bool | None = Field(
        default=None,
        description="Optional DOM-level outcome status flag.",
    )

    @field_validator("selection_name")
    @classmethod
    def validate_selection_name(cls, value: str) -> str:
        """Reject blank selection names before normalization begins."""

        return require_non_blank_text(value, "selection_name")

    @field_validator("selection_id", mode="before")
    @classmethod
    def normalize_selection_id(cls, value: object) -> object:
        """Collapse blank string identifiers to `None`."""

        if isinstance(value, str):
            return normalize_optional_text(value)
        return value

    @field_validator("odds", mode="before")
    @classmethod
    def validate_odds(cls, value: object) -> object:
        """Reject blank odds values so parser failures stay explicit."""

        if isinstance(value, str):
            return require_non_blank_text(value, "odds")
        return value

    def to_market_outcome(self) -> dict[str, str | int | float | bool | None]:
        """Convert one DOM snapshot into a SportyBet-like outcome mapping."""

        return {
            "id": self.selection_id,
            "desc": self.selection_name,
            "odds": self.odds,
            "status": self.status,
        }


class BrowserMarketSnapshot(BaseModel):
    """Validated DOM market snapshot before canonical normalization.

    Inputs:
        One raw market object returned by the browser evaluation script.

    Outputs:
        A typed market snapshot with validated child outcomes ready to be
        wrapped inside a SportyBet-like event payload.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    market_name: str = Field(description="Visible market label from the page.")
    outcomes: tuple[BrowserOutcomeSnapshot, ...] = Field(
        default_factory=tuple,
        description="All visible outcomes extracted from the DOM.",
    )
    market_id: str | int | None = Field(
        default=None,
        description="Optional market identifier from DOM data attributes.",
    )
    specifier: str | None = Field(
        default=None,
        description="Optional market specifier such as `total=2.5`.",
    )

    @field_validator("market_name")
    @classmethod
    def validate_market_name(cls, value: str) -> str:
        """Reject blank market labels extracted from the DOM."""

        return require_non_blank_text(value, "market_name")

    @field_validator("specifier")
    @classmethod
    def normalize_specifier(cls, value: str | None) -> str | None:
        """Trim optional specifier text and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("market_id", mode="before")
    @classmethod
    def normalize_market_id(cls, value: object) -> object:
        """Collapse blank string identifiers to `None`."""

        if isinstance(value, str):
            return normalize_optional_text(value)
        return value

    @field_validator("outcomes")
    @classmethod
    def validate_outcomes(
        cls,
        value: tuple[BrowserOutcomeSnapshot, ...],
    ) -> tuple[BrowserOutcomeSnapshot, ...]:
        """Require at least two outcomes for one market snapshot."""

        if len(value) < 2:
            raise ValueError("DOM market snapshots must include at least two outcomes.")
        return value

    def to_market_mapping(self) -> dict[str, object]:
        """Convert one DOM market into a SportyBet-like market mapping."""

        return {
            "id": self.market_id,
            "name": self.market_name,
            "specifier": self.specifier,
            "outcomes": [outcome.to_market_outcome() for outcome in self.outcomes],
        }


class PageExtractionSnapshot(BaseModel):
    """Validated browser extraction payload returned by page evaluation.

    Inputs:
        The raw JSON-safe object returned by `DOM_MARKET_EXTRACTOR_SCRIPT`.

    Outputs:
        A typed page snapshot that preserves embedded event payloads alongside
        DOM-derived markets and lightweight page metadata.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_payloads: tuple[dict[str, object], ...] = Field(
        default_factory=tuple,
        description="Embedded JSON objects that already resemble SportyBet events.",
    )
    dom_markets: tuple[BrowserMarketSnapshot, ...] = Field(
        default_factory=tuple,
        description="Visible DOM market snapshots found on the page.",
    )
    title: str | None = Field(default=None, description="Document title.")
    heading: str | None = Field(default=None, description="Primary match heading.")
    competition: str | None = Field(default=None, description="Visible competition label.")
    sport: str | None = Field(default=None, description="Sport inferred in the browser.")
    home_team: str | None = Field(default=None, description="Optional home team label.")
    away_team: str | None = Field(default=None, description="Optional away team label.")

    @field_validator("title", "heading", "competition", "sport", "home_team", "away_team")
    @classmethod
    def normalize_optional_text_field(cls, value: str | None) -> str | None:
        """Trim optional text fields collected from the page snapshot."""

        return normalize_optional_text(value)


@dataclass(frozen=True, slots=True)
class ParsedURLContext:
    """Fixture metadata inferred directly from a canonical SportyBet URL."""

    sportradar_id: str | None
    sport: SportName | None
    country: str | None
    competition: str | None
    home_team: str | None
    away_team: str | None


class SportyBetBrowserScraper:
    """Playwright-powered SportyBet browser fallback scraper.

    Args:
        cache: Optional Redis cache wrapper reused for SportyBet market snapshots.
            When omitted, the canonical cache client is created from settings.
        page_session_factory: Optional injected page-session factory for tests.
            The factory receives the rotated user-agent chosen for the request.
        user_agents: Ordered user-agent pool rotated across browser sessions.
        clock: Optional clock injector used for cache timestamps.
        navigation_timeout_ms: Maximum time allowed for initial page navigation.
        network_idle_timeout_ms: Additional wait budget for SPA hydration after
            the first document load event.
        market_wait_timeout_ms: Selector wait budget used before extraction.
        market_wait_selectors: DOM selectors that indicate market content may
            already be visible.
    """

    def __init__(
        self,
        cache: RedisClient | None = None,
        *,
        page_session_factory: PageSessionFactory | None = None,
        user_agents: Sequence[str] = DEFAULT_USER_AGENTS,
        clock: Callable[[], datetime] | None = None,
        navigation_timeout_ms: int = 15_000,
        network_idle_timeout_ms: int = 5_000,
        market_wait_timeout_ms: int = 6_000,
        market_wait_selectors: Sequence[str] = DEFAULT_MARKET_WAIT_SELECTORS,
    ) -> None:
        """Initialize the browser fallback with canonical PuntLab defaults."""

        if not user_agents:
            raise ValueError("user_agents must contain at least one value.")
        if navigation_timeout_ms <= 0:
            raise ValueError("navigation_timeout_ms must be positive.")
        if network_idle_timeout_ms <= 0:
            raise ValueError("network_idle_timeout_ms must be positive.")
        if market_wait_timeout_ms <= 0:
            raise ValueError("market_wait_timeout_ms must be positive.")
        if not market_wait_selectors:
            raise ValueError("market_wait_selectors must contain at least one selector.")

        self._owns_cache = cache is None
        self._cache = cache or RedisClient()
        self._clock = clock or (lambda: datetime.now(WAT_TIMEZONE))
        self._user_agents = itertools.cycle(
            tuple(self._normalize_user_agent(user_agent) for user_agent in user_agents)
        )
        self._navigation_timeout_ms = navigation_timeout_ms
        self._network_idle_timeout_ms = network_idle_timeout_ms
        self._market_wait_timeout_ms = market_wait_timeout_ms
        self._market_wait_selectors = tuple(
            self._normalize_selector(selector) for selector in market_wait_selectors
        )
        self._page_session_factory = page_session_factory or self._default_page_session
        self._normalizer = SportyBetAPIClient(self._cache, clock=self._clock)

    async def scrape_markets(
        self,
        url: str,
        *,
        fixture: NormalizedFixture | None = None,
        use_cache: bool = True,
    ) -> tuple[NormalizedOdds, ...]:
        """Scrape and normalize SportyBet markets from one public match page.

        Inputs:
            url: Canonical public SportyBet match URL.
            fixture: Optional normalized fixture metadata used to fill any
                display fields missing from the browser page.
            use_cache: Whether cached SportyBet market snapshots may satisfy
                the request before opening a browser session.

        Outputs:
            A tuple of normalized SportyBet odds rows ready for resolver use.

        Raises:
            ProviderError: If the page cannot be loaded or no active markets
                can be extracted from the hydrated DOM.
        """

        normalized_url = self._normalize_url(url)
        url_context = self._parse_url_context(normalized_url)
        requested_sportradar_id = (
            fixture.sportradar_id if fixture is not None else url_context.sportradar_id
        )

        if use_cache and requested_sportradar_id is not None:
            cached_markets = await self._get_cached_markets(requested_sportradar_id)
            if cached_markets is not None:
                return cached_markets

        raw_snapshot = await self._load_page_snapshot(normalized_url)
        snapshot = self._validate_snapshot(raw_snapshot)
        resolved_sportradar_id = requested_sportradar_id or self._resolve_sportradar_id(
            snapshot,
            fallback_url_context=url_context,
        )
        if resolved_sportradar_id is None:
            raise ProviderError(
                "sportybet",
                (
                    "SportyBet browser scrape could not determine the fixture "
                    "Sportradar ID from the URL or page payload."
                ),
            )

        markets = self._normalize_snapshot(
            snapshot,
            url=normalized_url,
            requested_sportradar_id=resolved_sportradar_id,
            fixture=fixture,
            url_context=url_context,
        )
        if use_cache:
            await self._cache_markets(resolved_sportradar_id, markets)
        return markets

    def build_sportybet_url(self, fixture: NormalizedFixture) -> str:
        """Delegate SportyBet URL construction to the canonical API helper."""

        return self._normalizer.build_sportybet_url(fixture)

    async def aclose(self) -> None:
        """Close owned helper resources created by the browser scraper."""

        await self._normalizer.aclose()
        if self._owns_cache:
            await self._cache.close()

    async def _load_page_snapshot(self, url: str) -> object:
        """Open one SportyBet page and return the extracted DOM snapshot."""

        user_agent = next(self._user_agents)
        async with self._page_session_factory(user_agent) as page:
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=float(self._navigation_timeout_ms),
                )
                with suppress(PlaywrightTimeoutError):
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=float(self._network_idle_timeout_ms),
                    )

                # Market selectors are treated as a hint, not a hard requirement,
                # because embedded page state may still contain parseable markets.
                await self._wait_for_market_signal(page)
                return await page.evaluate(DOM_MARKET_EXTRACTOR_SCRIPT)
            except PlaywrightTimeoutError as exc:
                raise ProviderError(
                    "sportybet",
                    f"Timed out while waiting for SportyBet page '{url}' to load.",
                ) from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "sportybet",
                    f"SportyBet browser scrape failed for '{url}': {exc!s}",
                ) from exc

    async def _wait_for_market_signal(self, page: SupportsScrapePage) -> bool:
        """Wait briefly for any likely market selector without failing hard."""

        for selector in self._market_wait_selectors:
            try:
                await page.wait_for_selector(
                    selector,
                    timeout=float(self._market_wait_timeout_ms),
                )
                return True
            except PlaywrightTimeoutError:
                continue
        return False

    def _validate_snapshot(self, raw_snapshot: object) -> PageExtractionSnapshot:
        """Validate one raw browser-evaluation payload into a typed snapshot."""

        try:
            return PageExtractionSnapshot.model_validate(raw_snapshot)
        except ValidationError as exc:
            raise ProviderError(
                "sportybet",
                f"SportyBet browser scrape returned an invalid extraction payload: {exc!s}",
            ) from exc

    def _resolve_sportradar_id(
        self,
        snapshot: PageExtractionSnapshot,
        *,
        fallback_url_context: ParsedURLContext,
    ) -> str | None:
        """Resolve the fixture Sportradar ID from page payloads when needed."""

        if fallback_url_context.sportradar_id is not None:
            return fallback_url_context.sportradar_id

        for payload in snapshot.event_payloads:
            for event in self._normalizer._iter_event_candidates(payload):
                for identifier in self._normalizer._extract_event_identifiers(event):
                    if _SPORTRADAR_ID_PATTERN.fullmatch(identifier):
                        return identifier
        return None

    def _normalize_snapshot(
        self,
        snapshot: PageExtractionSnapshot,
        *,
        url: str,
        requested_sportradar_id: str,
        fixture: NormalizedFixture | None,
        url_context: ParsedURLContext,
    ) -> tuple[NormalizedOdds, ...]:
        """Normalize embedded event payloads and DOM markets into odds rows."""

        parsed_rows: list[NormalizedOdds] = []
        parse_errors: list[str] = []

        # Prefer embedded event payloads when present because they preserve the
        # richest market metadata, then merge DOM-derived rows as a fallback.
        for payload in snapshot.event_payloads:
            try:
                event = self._normalizer._extract_matching_event(
                    payload,
                    requested_sportradar_id,
                )
                parsed_rows.extend(
                    self._normalizer._tag_fetch_source(
                        self._normalizer._parse_markets(
                            event,
                            sportradar_id=requested_sportradar_id,
                            fixture=fixture,
                        ),
                        fetch_source="browser",
                    )
                )
            except ProviderError as exc:
                parse_errors.append(f"embedded payload -> {exc!s}")

        if snapshot.dom_markets:
            try:
                dom_event = self._build_dom_event(
                    snapshot,
                    requested_sportradar_id=requested_sportradar_id,
                    fixture=fixture,
                    url_context=url_context,
                )
                parsed_rows.extend(
                    self._normalizer._tag_fetch_source(
                        self._normalizer._parse_markets(
                            dom_event,
                            sportradar_id=requested_sportradar_id,
                            fixture=fixture,
                        ),
                        fetch_source="browser",
                    )
                )
            except ProviderError as exc:
                parse_errors.append(f"dom markets -> {exc!s}")

        deduped_rows = self._dedupe_rows(parsed_rows)
        if deduped_rows:
            return deduped_rows

        detail = " | ".join(parse_errors) if parse_errors else "no markets were discovered"
        raise ProviderError(
            "sportybet",
            (
                "SportyBet browser scrape loaded the page but could not normalize "
                f"active markets for '{url}'. Details: {detail}"
            ),
        )

    def _build_dom_event(
        self,
        snapshot: PageExtractionSnapshot,
        *,
        requested_sportradar_id: str,
        fixture: NormalizedFixture | None,
        url_context: ParsedURLContext,
    ) -> dict[str, object]:
        """Synthesize a SportyBet-like event object from DOM market snapshots."""

        teams_from_heading = self._parse_teams_from_text(snapshot.heading or snapshot.title)
        home_team = (
            fixture.home_team
            if fixture is not None
            else snapshot.home_team
            or teams_from_heading[0]
            or url_context.home_team
        )
        away_team = (
            fixture.away_team
            if fixture is not None
            else snapshot.away_team
            or teams_from_heading[1]
            or url_context.away_team
        )
        competition = (
            fixture.competition
            if fixture is not None
            else snapshot.competition or url_context.competition
        )
        sport = (
            fixture.sport
            if fixture is not None
            else self._infer_sport(snapshot.sport, url_context.sport)
        )
        category_name = (
            fixture.country
            if fixture is not None
            else url_context.country or "International"
        )

        if home_team is None or away_team is None:
            raise ProviderError(
                "sportybet",
                "DOM scrape extracted markets but could not determine both team names.",
            )
        if competition is None:
            raise ProviderError(
                "sportybet",
                "DOM scrape extracted markets but could not determine the competition name.",
            )
        if sport is None:
            raise ProviderError(
                "sportybet",
                "DOM scrape extracted markets but could not determine the sport.",
            )

        return {
            "eventId": requested_sportradar_id,
            "homeTeamName": home_team,
            "awayTeamName": away_team,
            "sport": {"name": "Football" if sport == SportName.SOCCER else "Basketball"},
            "category": {
                "name": category_name,
                "tournament": {"name": competition},
            },
            "markets": [market.to_market_mapping() for market in snapshot.dom_markets],
        }

    def _infer_sport(
        self,
        snapshot_sport: str | None,
        url_sport: SportName | None,
    ) -> SportName | None:
        """Resolve sport identity from the page snapshot or URL segments."""

        if snapshot_sport is not None:
            try:
                return self._normalizer._normalize_sport_name(snapshot_sport)
            except ProviderError:
                pass
        return url_sport

    async def _get_cached_markets(self, sportradar_id: str) -> tuple[NormalizedOdds, ...] | None:
        """Load one cached SportyBet market snapshot from Redis when present."""

        cache_key = RedisClient.build_sportybet_markets_key(sportradar_id)
        try:
            cached_snapshot = await self._cache.get(
                cache_key,
                model=SportyBetMarketCacheEntry,
            )
        except Exception as exc:
            logger.warning(
                "SportyBet browser market cache read skipped for %s: %s",
                sportradar_id,
                exc,
            )
            return None
        if not isinstance(cached_snapshot, SportyBetMarketCacheEntry):
            return None
        return cached_snapshot.markets

    async def _cache_markets(
        self,
        sportradar_id: str,
        markets: Sequence[NormalizedOdds],
    ) -> None:
        """Persist one normalized SportyBet page scrape under the canonical cache key."""

        cache_key = RedisClient.build_sportybet_markets_key(sportradar_id)
        snapshot = SportyBetMarketCacheEntry(
            sportradar_id=sportradar_id,
            fetched_at=self._clock().astimezone(UTC),
            markets=tuple(markets),
        )
        try:
            await self._cache.set(cache_key, snapshot)
        except Exception as exc:
            logger.warning(
                "SportyBet browser market cache write skipped for %s: %s",
                sportradar_id,
                exc,
            )

    @staticmethod
    def _dedupe_rows(markets: Sequence[NormalizedOdds]) -> tuple[NormalizedOdds, ...]:
        """Deduplicate normalized rows while preserving their discovery order."""

        deduped: dict[tuple[object, ...], NormalizedOdds] = {}
        for market in markets:
            key = (
                market.fixture_ref,
                market.provider_market_id,
                market.provider_market_name.casefold(),
                market.provider_selection_name.casefold(),
                market.line,
                market.odds,
            )
            deduped.setdefault(key, market)
        return tuple(deduped.values())

    @staticmethod
    def _parse_url_context(url: str) -> ParsedURLContext:
        """Infer fixture metadata from a canonical SportyBet page URL."""

        parsed_url = urlparse(url)
        segments = [unquote(segment) for segment in parsed_url.path.split("/") if segment]
        sportradar_match = _SPORTRADAR_ID_PATTERN.search(parsed_url.path)
        sportradar_id = sportradar_match.group(0) if sportradar_match is not None else None

        try:
            sport_index = segments.index("sport")
        except ValueError:
            sport_index = -1

        sport: SportName | None = None
        country: str | None = None
        competition: str | None = None
        home_team: str | None = None
        away_team: str | None = None

        if sport_index >= 0 and len(segments) > sport_index + 4:
            sport = SportyBetBrowserScraper._normalize_url_sport(segments[sport_index + 1])
            country = SportyBetBrowserScraper._decode_url_segment(segments[sport_index + 2])
            competition = SportyBetBrowserScraper._decode_url_segment(segments[sport_index + 3])
            home_team, away_team = SportyBetBrowserScraper._split_team_slug(
                segments[sport_index + 4]
            )

        return ParsedURLContext(
            sportradar_id=sportradar_id,
            sport=sport,
            country=country,
            competition=competition,
            home_team=home_team,
            away_team=away_team,
        )

    @staticmethod
    def _parse_teams_from_text(value: str | None) -> tuple[str | None, str | None]:
        """Extract home and away team names from a title-like text fragment."""

        normalized = normalize_optional_text(value)
        if normalized is None:
            return (None, None)

        parts = [part.strip() for part in _TEAM_SEPARATOR_PATTERN.split(normalized) if part.strip()]
        if len(parts) < 2:
            return (None, None)
        return (parts[0], parts[1])

    @staticmethod
    def _split_team_slug(value: str) -> tuple[str | None, str | None]:
        """Split one canonical SportyBet team slug into home and away names."""

        normalized = normalize_optional_text(value)
        if normalized is None:
            return (None, None)

        parts = [part for part in _TEAM_SEPARATOR_PATTERN.split(normalized) if part]
        if len(parts) < 2:
            return (None, None)
        return (
            SportyBetBrowserScraper._decode_url_segment(parts[0]),
            SportyBetBrowserScraper._decode_url_segment(parts[1]),
        )

    @staticmethod
    def _decode_url_segment(value: str) -> str:
        """Convert one URL segment into a readable label without altering case."""

        return require_non_blank_text(value.replace("_", " ").replace("-", " "), "url_segment")

    @staticmethod
    def _normalize_url_sport(value: str) -> SportName | None:
        """Map a SportyBet URL sport segment onto PuntLab's sport enum."""

        normalized = value.strip().casefold()
        if normalized == "football":
            return SportName.SOCCER
        if normalized == "basketball":
            return SportName.BASKETBALL
        return None

    @staticmethod
    def _normalize_user_agent(value: str) -> str:
        """Reject blank user-agent strings before browser sessions begin."""

        return require_non_blank_text(value, "user_agent")

    @staticmethod
    def _normalize_selector(value: str) -> str:
        """Reject blank DOM selectors configured for market waiting."""

        return require_non_blank_text(value, "market_wait_selector")

    @staticmethod
    def _normalize_url(value: str) -> str:
        """Require a non-blank fully-qualified URL before navigation."""

        normalized = require_non_blank_text(value, "url")
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("url must be an absolute SportyBet page URL.")
        return normalized

    @asynccontextmanager
    async def _default_page_session(self, user_agent: str) -> AsyncIterator[SupportsScrapePage]:
        """Create the canonical Playwright browser session for one scrape."""

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                locale="en-NG",
                timezone_id="Africa/Lagos",
                user_agent=user_agent,
            )
            page = await context.new_page()
            try:
                yield cast(SupportsScrapePage, page)
            finally:
                await context.close()
                await browser.close()


__all__ = [
    "DEFAULT_MARKET_WAIT_SELECTORS",
    "DOM_MARKET_EXTRACTOR_SCRIPT",
    "SportyBetBrowserScraper",
]
