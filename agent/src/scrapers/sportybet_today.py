"""CLI collector for SportyBet's today pre-match slate.

Purpose: provide one runnable command that discovers SportyBet's current
"Today Games" pre-match feed through the live browser flow, paginates the
results inside the same browser session, and emits normalized event records.
Scope: sport-route discovery, today-feed capture, pagination, event
normalization, and JSON/JSONL CLI output for local operational use.
Dependencies: Playwright for browser execution, canonical PuntLab timezone and
shared SportyBet constants, plus lightweight Pydantic models for validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, date, datetime
from math import ceil
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from playwright.async_api import Browser, BrowserContext
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, Playwright, Response
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, WAT_TIMEZONE_NAME, SportName
from src.providers.base import ProviderError, RateLimitExhausted
from src.providers.odds_mapping import (
    build_fixture_market_snapshots,
    build_odds_market_catalog,
)
from src.schemas.common import normalize_optional_text, require_non_blank_text
from src.schemas.fixtures import NormalizedFixture
from src.schemas.market_snapshot import (
    FixtureMarketSelection,
    FixtureMarketSnapshot,
    FixtureMarketSnapshotEntry,
)
from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SPORTYBET_BASE_URL,
    SPORTYBET_COUNTRY_CODE,
    SPORTYBET_RATE_LIMIT_POLICY,
    SportyBetAPIClient,
)

SPORTYBET_BOOTSTRAP_PAGE_URL: Final[str] = (
    f"{SPORTYBET_BASE_URL}/{SPORTYBET_COUNTRY_CODE}/sport/football/"
)
SPORTYBET_ORDERED_SPORT_LIST_PATH: Final[str] = (
    f"/api/{SPORTYBET_COUNTRY_CODE}/factsCenter/orderedSportList"
)
SPORTYBET_TODAY_FEED_PATH: Final[str] = (
    f"/api/{SPORTYBET_COUNTRY_CODE}/factsCenter/pcUpcomingEvents"
)
SPORTYBET_TODAY_TAB_LABEL: Final[str] = "Today Games"
DEFAULT_PAGE_SIZE: Final[int] = 100
DEFAULT_NAVIGATION_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_RESPONSE_TIMEOUT_MS: Final[int] = 60_000
DEFAULT_SETTLE_DELAY_MS: Final[int] = 1_500
DEFAULT_OUTPUT_FORMAT: Final[str] = "json"
FULL_MARKET_EXPORT_SUPPORTED_SPORTS: Final[dict[str, SportName]] = {
    "basketball": SportName.BASKETBALL,
    "football": SportName.SOCCER,
}

SPORTYBET_SPORT_LINKS_SCRIPT: Final[str] = """
() => Array.from(document.querySelectorAll("a"))
  .map((element) => ({
    text: (element.textContent || "").replace(/\\s+/g, " ").trim(),
    href: element.href,
  }))
  .filter((item) => item.text && item.href && item.href.includes("/ng/sport/"))
"""


class SportyBetTodaySport(BaseModel):
    """One SportyBet sport route available to the today-games collector."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sport_id: str = Field(description="SportyBet sport identifier.")
    sport_name: str = Field(description="Sport display name.")
    route_slug: str = Field(description="Canonical SportyBet route segment.")
    page_url: str = Field(description="Absolute SportyBet sport page URL.")
    event_size_hint: int | None = Field(
        default=None,
        ge=0,
        description="Optional SportyBet count hint returned by the sport list.",
    )
    fetched_event_count: int = Field(
        default=0,
        ge=0,
        description="Number of today events collected for this sport.",
    )
    today_request_url: str | None = Field(
        default=None,
        description="Exact today-feed request URL captured from the live page.",
    )

    @field_validator("sport_id", "sport_name", "route_slug", "page_url")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required text fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("today_request_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        """Trim optional today-request URLs."""

        return normalize_optional_text(value)


class SportyBetTodayEvent(BaseModel):
    """Normalized SportyBet pre-match event extracted from the today feed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str = Field(description="SportyBet event identifier.")
    game_id: str | None = Field(
        default=None,
        description="Optional SportyBet numeric game identifier.",
    )
    sport_id: str = Field(description="SportyBet sport identifier.")
    sport_name: str = Field(description="Sport display name.")
    category_id: str = Field(description="SportyBet category identifier.")
    category_name: str = Field(description="SportyBet category display name.")
    tournament_id: str = Field(description="SportyBet tournament identifier.")
    tournament_name: str = Field(description="SportyBet tournament display name.")
    kickoff: datetime = Field(description="Kickoff timestamp converted into WAT.")
    home_team_name: str = Field(description="Home team display name.")
    away_team_name: str = Field(description="Away team display name.")
    total_market_size: int = Field(
        ge=0,
        description="SportyBet-reported total available market count for the event.",
    )
    market_count: int = Field(
        ge=0,
        description="Number of market objects present in the fetched page payload.",
    )
    status: int | None = Field(
        default=None,
        description="Optional numeric SportyBet event status.",
    )
    match_status: str | None = Field(
        default=None,
        description="Optional SportyBet match-status label.",
    )

    @field_validator(
        "event_id",
        "sport_id",
        "sport_name",
        "category_id",
        "category_name",
        "tournament_id",
        "tournament_name",
        "home_team_name",
        "away_team_name",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required text fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("game_id", "match_status")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text fields and collapse empties to `None`."""

        return normalize_optional_text(value)

    @field_validator("kickoff")
    @classmethod
    def validate_kickoff(cls, value: datetime) -> datetime:
        """Require timezone-aware kickoffs."""

        if value.utcoffset() is None:
            raise ValueError("kickoff must include timezone information.")
        return value


class SportyBetTodaySlate(BaseModel):
    """Top-level today-games output bundle."""

    model_config = ConfigDict(extra="forbid")

    run_date: date = Field(description="Run date in WAT.")
    fetched_at: datetime = Field(description="Timestamp when collection finished.")
    timezone: str = Field(description="Timezone used for kickoff normalization.")
    sports: tuple[SportyBetTodaySport, ...] = Field(
        default_factory=tuple,
        description="Sports included in the collection run.",
    )
    events: tuple[SportyBetTodayEvent, ...] = Field(
        default_factory=tuple,
        description="Normalized today pre-match events returned by SportyBet.",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Reject blank timezone names."""

        return require_non_blank_text(value, "timezone")

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware collection timestamps."""

        if value.utcoffset() is None:
            raise ValueError("fetched_at must include timezone information.")
        return value


class _InMemoryAsyncRedis:
    """Minimal in-memory Redis stub used by the markdown exporter."""

    def __init__(self) -> None:
        """Initialize value and expiration stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return one stored value by key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Persist one value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment one numeric counter."""

        next_value = int(self.values.get(name, "0")) + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Attach a TTL to an existing stored key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the in-memory Redis stub is always healthy."""

        return True

    async def aclose(self) -> None:
        """Match the async Redis protocol expected by `RedisClient`."""


class SportyBetTodayGamesCollector:
    """Collect SportyBet's today-only pre-match events through the live page flow."""

    def __init__(
        self,
        *,
        user_agents: Sequence[str] = DEFAULT_USER_AGENTS,
        clock: Callable[[], datetime] | None = None,
        headless: bool = True,
        page_size: int = DEFAULT_PAGE_SIZE,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        response_timeout_ms: int = DEFAULT_RESPONSE_TIMEOUT_MS,
        settle_delay_ms: int = DEFAULT_SETTLE_DELAY_MS,
    ) -> None:
        """Store deterministic runtime settings for the collector."""

        if not user_agents:
            raise ValueError("user_agents must contain at least one value.")
        if page_size <= 0:
            raise ValueError("page_size must be positive.")
        if navigation_timeout_ms <= 0:
            raise ValueError("navigation_timeout_ms must be positive.")
        if response_timeout_ms <= 0:
            raise ValueError("response_timeout_ms must be positive.")
        if settle_delay_ms < 0:
            raise ValueError("settle_delay_ms must be zero or positive.")

        self._user_agent = require_non_blank_text(user_agents[0], "user_agent")
        self._clock = clock or (lambda: datetime.now(WAT_TIMEZONE))
        self._headless = headless
        self._page_size = page_size
        self._navigation_timeout_ms = navigation_timeout_ms
        self._response_timeout_ms = response_timeout_ms
        self._settle_delay_ms = settle_delay_ms

    async def collect(
        self,
        *,
        sports: Sequence[str] = (),
    ) -> SportyBetTodaySlate:
        """Collect the selected SportyBet today pre-match slate."""

        started_at = self._clock()

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            context = await browser.new_context(
                locale="en-NG",
                timezone_id=WAT_TIMEZONE_NAME,
                user_agent=self._user_agent,
            )
            try:
                bootstrap_page = await context.new_page()
                try:
                    all_routes = await self._discover_sport_routes(bootstrap_page)
                finally:
                    await bootstrap_page.close()

                selected_routes = self._select_sports(all_routes, sports=sports)
                collected_sports: list[SportyBetTodaySport] = []
                collected_events: list[SportyBetTodayEvent] = []

                for route in selected_routes:
                    events, today_request_url = await self._collect_sport_today_events(
                        context,
                        route,
                    )
                    collected_sports.append(
                        route.model_copy(
                            update={
                                "fetched_event_count": len(events),
                                "today_request_url": today_request_url,
                            }
                        )
                    )
                    collected_events.extend(events)
            finally:
                await context.close()
                await browser.close()

        deduped_events = _dedupe_today_events(collected_events)
        sorted_events = tuple(
            sorted(
                deduped_events,
                key=lambda event: (
                    event.kickoff,
                    event.sport_name.casefold(),
                    event.tournament_name.casefold(),
                    event.home_team_name.casefold(),
                    event.away_team_name.casefold(),
                ),
            )
        )
        return SportyBetTodaySlate(
            run_date=started_at.date(),
            fetched_at=self._clock(),
            timezone=WAT_TIMEZONE_NAME,
            sports=tuple(collected_sports),
            events=sorted_events,
        )

    async def _launch_browser(self, playwright: Playwright) -> Browser:
        """Launch Chromium and surface installation errors with clear recovery."""

        try:
            return await playwright.chromium.launch(headless=self._headless)
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise RuntimeError(
                    "Playwright Chromium is not installed. Run "
                    "`playwright install chromium` and retry."
                ) from exc
            raise RuntimeError(f"Could not launch Playwright Chromium: {message}") from exc

    async def _discover_sport_routes(
        self,
        page: Page,
    ) -> tuple[SportyBetTodaySport, ...]:
        """Load one bootstrap page and build the available SportyBet sport routes."""

        async with page.expect_response(
            lambda response: (
                SPORTYBET_ORDERED_SPORT_LIST_PATH in response.url and response.status == 200
            ),
            timeout=float(self._response_timeout_ms),
        ) as ordered_sport_list_info:
            await page.goto(
                SPORTYBET_BOOTSTRAP_PAGE_URL,
                wait_until="domcontentloaded",
                timeout=float(self._navigation_timeout_ms),
            )
        ordered_sport_list_response = await ordered_sport_list_info.value
        ordered_sport_list_payload = await ordered_sport_list_response.json()
        await page.wait_for_selector(
            "a[href*='/ng/sport/']",
            timeout=float(self._response_timeout_ms),
        )
        await page.wait_for_timeout(float(self._settle_delay_ms))
        anchor_rows = await page.evaluate(SPORTYBET_SPORT_LINKS_SCRIPT)
        if not isinstance(anchor_rows, list):
            raise RuntimeError("SportyBet sport-link extraction returned an invalid payload.")
        sport_rows = ordered_sport_list_payload.get("data")
        if not isinstance(sport_rows, list):
            raise RuntimeError("SportyBet orderedSportList returned an invalid data payload.")
        return _build_today_sport_routes(sport_rows, anchor_rows)

    async def _collect_sport_today_events(
        self,
        context: BrowserContext,
        route: SportyBetTodaySport,
    ) -> tuple[tuple[SportyBetTodayEvent, ...], str]:
        """Collect and paginate one sport's today-only pre-match feed."""

        page = await context.new_page()

        try:
            async with page.expect_response(
                lambda response: self._is_today_feed_response(response),
                timeout=float(self._response_timeout_ms),
            ) as today_response_info:
                await page.goto(
                    route.page_url,
                    wait_until="domcontentloaded",
                    timeout=float(self._navigation_timeout_ms),
                )
                with suppress(PlaywrightTimeoutError):
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=float(self._settle_delay_ms),
                    )
                await page.wait_for_timeout(float(self._settle_delay_ms))

                today_tab = page.get_by_text(SPORTYBET_TODAY_TAB_LABEL, exact=True)
                with suppress(PlaywrightTimeoutError):
                    await today_tab.click(timeout=float(self._response_timeout_ms))

            today_response = await today_response_info.value
            first_payload = await today_response.json()
            _validate_today_feed_payload(first_payload)

            raw_event_count = _count_payload_events(first_payload)
            collected_events = list(_flatten_today_feed(first_payload, route))
            total_num = _extract_total_num(first_payload)
            total_pages = ceil(total_num / self._page_size) if total_num > 0 else 0

            for page_num in range(2, total_pages + 1):
                paged_url = _build_paged_feed_url(
                    today_response.url,
                    page_num,
                    timestamp_ms=int(self._clock().timestamp() * 1000),
                )
                payload = await self._fetch_json_in_context(context, paged_url)
                _validate_today_feed_payload(payload)
                raw_event_count += _count_payload_events(payload)
                collected_events.extend(_flatten_today_feed(payload, route))

            if raw_event_count < total_num:
                raise RuntimeError(
                    "SportyBet today feed pagination ended early for "
                    f"'{route.sport_name}': collected {raw_event_count} of {total_num} events."
                )

            deduped = _dedupe_today_events(collected_events)
            return (deduped, today_response.url)
        finally:
            await page.close()

    async def _fetch_json_in_context(
        self,
        context: BrowserContext,
        url: str,
    ) -> Mapping[str, object]:
        """Fetch one SportyBet JSON URL through the live browser context session."""

        response = await context.request.get(
            url,
            headers={
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        if not response.ok:
            status = response.status
            body = await response.text()
            raise RuntimeError(
                f"SportyBet context fetch failed for '{url}' with status {status}: {body!s}"
            )
        payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"SportyBet context fetch returned non-JSON content for '{url}'.")
        return payload

    @staticmethod
    def _is_today_feed_response(response: Response) -> bool:
        """Return whether the response is the today-only pre-match feed."""

        return (
            SPORTYBET_TODAY_FEED_PATH in response.url
            and "todayGames=true" in response.url
            and response.status == 200
        )

    @staticmethod
    def _select_sports(
        all_routes: Sequence[SportyBetTodaySport],
        *,
        sports: Sequence[str],
    ) -> tuple[SportyBetTodaySport, ...]:
        """Filter discovered sports by user-requested selectors when provided."""

        if not sports:
            return tuple(all_routes)

        normalized_requested = {_normalize_sport_selector(selector) for selector in sports}
        selected: list[SportyBetTodaySport] = []
        available_names = sorted(route.sport_name for route in all_routes)

        for route in all_routes:
            route_keys = {
                _normalize_sport_selector(route.sport_name),
                _normalize_sport_selector(route.route_slug),
            }
            if route_keys & normalized_requested:
                selected.append(route)

        if not selected:
            available = ", ".join(available_names)
            requested = ", ".join(sports)
            raise ValueError(
                f"No SportyBet sports matched '{requested}'. Available sports: {available}."
            )

        return tuple(selected)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the today-games collector."""

    parser = argparse.ArgumentParser(
        description=(
            "Fetch all available SportyBet pre-match games playing today through "
            "the live browser feed."
        )
    )
    parser.add_argument(
        "--sport",
        action="append",
        default=[],
        help=(
            "Optional sport filter. Repeat the flag or pass a comma-separated list. "
            "Examples: --sport football --sport basketball, or --sport football,basketball."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl", "markdown"),
        default=DEFAULT_OUTPUT_FORMAT,
        help=(
            "Output format. `json` emits the full slate bundle; `jsonl` emits one event "
            "per line; `markdown` fetches the full SportyBet market book for each event "
            "and renders an ordered readable report."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. When omitted, the result is written to stdout.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium in headed mode for local debugging.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> str:
    """Execute the collector and return the rendered output string."""

    requested_sports = _parse_requested_sports(args.sport)
    collector = SportyBetTodayGamesCollector(headless=not args.headful)
    slate = await collector.collect(sports=requested_sports)
    if args.format == "markdown":
        return await _render_markdown_slate(slate)
    return _render_slate(slate, output_format=args.format)


def run(argv: Sequence[str] | None = None) -> None:
    """CLI entry point used by `puntlab-agent-sportybet-today`."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    rendered_output = asyncio.run(_run_async(args))
    if args.output is not None:
        args.output.write_text(rendered_output, encoding="utf-8")
        return
    print(rendered_output)


def _parse_requested_sports(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize repeated or comma-separated CLI sport selectors."""

    requested_sports: list[str] = []
    for value in values:
        for part in value.split(","):
            normalized = normalize_optional_text(part)
            if normalized is not None:
                requested_sports.append(normalized)
    return tuple(requested_sports)


def _build_today_sport_routes(
    sport_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
) -> tuple[SportyBetTodaySport, ...]:
    """Build SportyBet sport routes from orderedSportList plus nav anchors."""

    anchors_by_name: dict[str, tuple[str, str]] = {}
    for row in anchor_rows:
        sport_name = normalize_optional_text(_coerce_optional_text(row.get("text")))
        href = normalize_optional_text(_coerce_optional_text(row.get("href")))
        if sport_name is None or href is None:
            continue
        if "/sport/live/" in href:
            continue
        anchors_by_name[_normalize_sport_selector(sport_name)] = (_extract_route_slug(href), href)

    routes: list[SportyBetTodaySport] = []
    for row in sport_rows:
        sport_id = _require_text(row.get("id"), field_name="sport_id")
        sport_name = _require_text(row.get("name"), field_name="sport_name")
        event_size_hint = _coerce_non_negative_int(row.get("eventSize"), field_name="eventSize")
        if event_size_hint == 0:
            continue

        route_slug: str
        page_url: str
        anchor = anchors_by_name.get(_normalize_sport_selector(sport_name))
        if anchor is not None:
            route_slug, page_url = anchor
        else:
            route_slug = _fallback_route_slug(sport_name)
            page_url = _build_sport_page_url(route_slug)

        routes.append(
            SportyBetTodaySport(
                sport_id=sport_id,
                sport_name=sport_name,
                route_slug=route_slug,
                page_url=page_url,
                event_size_hint=event_size_hint,
            )
        )
    return tuple(routes)


def _flatten_today_feed(
    payload: Mapping[str, object],
    route: SportyBetTodaySport,
) -> tuple[SportyBetTodayEvent, ...]:
    """Flatten a SportyBet today-feed payload into normalized event rows."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("SportyBet today feed payload is missing a valid `data` mapping.")

    tournaments = data.get("tournaments")
    if not isinstance(tournaments, Sequence):
        raise RuntimeError("SportyBet today feed payload is missing `tournaments`.")

    flattened_events: list[SportyBetTodayEvent] = []
    for tournament_row in tournaments:
        if not isinstance(tournament_row, Mapping):
            raise RuntimeError("SportyBet tournament rows must be object mappings.")
        top_level_category_id = _normalize_optional_identifier(tournament_row.get("categoryId"))
        top_level_category_name = normalize_optional_text(
            _coerce_optional_text(tournament_row.get("categoryName"))
        )
        top_level_tournament_id = _normalize_optional_identifier(tournament_row.get("id"))
        top_level_tournament_name = normalize_optional_text(
            _coerce_optional_text(tournament_row.get("name"))
        )

        events = tournament_row.get("events")
        if not isinstance(events, Sequence):
            raise RuntimeError("SportyBet tournament rows must contain an `events` sequence.")

        for event_row in events:
            if not isinstance(event_row, Mapping):
                raise RuntimeError("SportyBet event rows must be object mappings.")

            sport_row = _require_mapping(event_row.get("sport"), field_name="sport")
            category_row = _require_mapping(sport_row.get("category"), field_name="sport.category")
            tournament_meta = _require_mapping(
                category_row.get("tournament"),
                field_name="sport.category.tournament",
            )

            event_id = _require_identifier(event_row.get("eventId"), field_name="eventId")
            sport_id = _require_identifier(
                sport_row.get("id") or route.sport_id,
                field_name="sport.id",
            )
            sport_name = require_non_blank_text(
                (_coerce_optional_text(sport_row.get("name")) or route.sport_name),
                "sport.name",
            )
            category_id = _require_identifier(
                top_level_category_id or category_row.get("id"),
                field_name="category.id",
            )
            category_name = _require_text(
                top_level_category_name or category_row.get("name"),
                field_name="category.name",
            )
            tournament_id = _require_identifier(
                top_level_tournament_id or tournament_meta.get("id"),
                field_name="tournament.id",
            )
            tournament_name = _require_text(
                top_level_tournament_name or tournament_meta.get("name"),
                field_name="tournament.name",
            )
            kickoff = _coerce_kickoff(
                event_row.get("estimateStartTime"),
                field_name="estimateStartTime",
            )
            markets = event_row.get("markets")
            market_count = len(markets) if isinstance(markets, Sequence) else 0

            flattened_events.append(
                SportyBetTodayEvent(
                    event_id=event_id,
                    game_id=_normalize_optional_identifier(event_row.get("gameId")),
                    sport_id=sport_id,
                    sport_name=sport_name,
                    category_id=category_id,
                    category_name=category_name,
                    tournament_id=tournament_id,
                    tournament_name=tournament_name,
                    kickoff=kickoff,
                    home_team_name=_require_text(
                        event_row.get("homeTeamName"),
                        field_name="homeTeamName",
                    ),
                    away_team_name=_require_text(
                        event_row.get("awayTeamName"),
                        field_name="awayTeamName",
                    ),
                    total_market_size=_coerce_non_negative_int(
                        event_row.get("totalMarketSize"),
                        field_name="totalMarketSize",
                    ),
                    market_count=market_count,
                    status=_coerce_optional_int(event_row.get("status")),
                    match_status=normalize_optional_text(
                        _coerce_optional_text(event_row.get("matchStatus"))
                    ),
                )
            )

    return tuple(flattened_events)


def _count_payload_events(payload: Mapping[str, object]) -> int:
    """Count the number of event rows inside one today-feed payload."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("SportyBet today feed payload is missing a valid `data` mapping.")
    tournaments = data.get("tournaments")
    if not isinstance(tournaments, Sequence):
        raise RuntimeError("SportyBet today feed payload is missing `tournaments`.")

    count = 0
    for tournament_row in tournaments:
        if not isinstance(tournament_row, Mapping):
            raise RuntimeError("SportyBet tournament rows must be object mappings.")
        events = tournament_row.get("events")
        if not isinstance(events, Sequence):
            raise RuntimeError("SportyBet tournament rows must contain an `events` sequence.")
        count += len(events)
    return count


def _extract_total_num(payload: Mapping[str, object]) -> int:
    """Extract the SportyBet event total from one today-feed payload."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("SportyBet today feed payload is missing a valid `data` mapping.")
    return _coerce_non_negative_int(data.get("totalNum"), field_name="totalNum")


def _build_paged_feed_url(
    url: str,
    page_num: int,
    *,
    timestamp_ms: int,
) -> str:
    """Replace one today-feed URL's page number while preserving all other params."""

    if page_num <= 0:
        raise ValueError("page_num must be positive.")
    if timestamp_ms <= 0:
        raise ValueError("timestamp_ms must be positive.")

    parsed_url = urlparse(require_non_blank_text(url, "url"))
    query_items = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    query_items["pageNum"] = str(page_num)
    query_items["_t"] = str(timestamp_ms)
    return urlunparse(parsed_url._replace(query=urlencode(query_items)))


def _render_slate(
    slate: SportyBetTodaySlate,
    *,
    output_format: str,
) -> str:
    """Render the today slate as JSON or JSONL."""

    normalized_format = output_format.strip().casefold()
    if normalized_format == "json":
        return json.dumps(slate.model_dump(mode="json"), indent=2)
    if normalized_format == "jsonl":
        return "\n".join(
            json.dumps(event.model_dump(mode="json"), sort_keys=True) for event in slate.events
        )
    raise ValueError(f"Unsupported output_format '{output_format}'.")


async def _render_markdown_slate(slate: SportyBetTodaySlate) -> str:
    """Fetch full markets for the slate and render them as ordered markdown."""

    ordered_events = _sort_today_events(slate.events)
    supported_events: list[SportyBetTodayEvent] = []
    unsupported_events: list[SportyBetTodayEvent] = []
    fixtures: list[NormalizedFixture] = []
    for event in ordered_events:
        fixture = _build_fixture_from_today_event(event)
        if fixture is None:
            unsupported_events.append(event)
            continue
        supported_events.append(event)
        fixtures.append(fixture)

    if not fixtures:
        return _render_markdown_output(
            slate=slate,
            ordered_events=ordered_events,
            snapshots=(),
            failures={},
            unsupported_events=tuple(unsupported_events),
        )

    redis_cache = RedisClient(redis_client=_InMemoryAsyncRedis())
    api_client = SportyBetAPIClient(redis_cache)
    fetched_rows = []
    failures: dict[str, str] = {}
    try:
        for fixture in fixtures:
            while True:
                try:
                    fetched_rows.extend(
                        await api_client.fetch_markets(
                            fixture.get_fixture_ref(),
                            fixture=fixture,
                            use_cache=True,
                        )
                    )
                    break
                except RateLimitExhausted:
                    await asyncio.sleep(_seconds_until_next_rate_limit_window())
                except (ProviderError, ValueError) as exc:
                    failures[fixture.get_fixture_ref()] = str(exc)
                    break
    finally:
        await api_client.aclose()
        await redis_cache.close()

    catalog = build_odds_market_catalog(
        fetched_rows,
        sport_by_fixture={fixture.get_fixture_ref(): fixture.sport for fixture in fixtures},
    )
    snapshots = build_fixture_market_snapshots(fixtures, catalog)
    return _render_markdown_output(
        slate=slate,
        ordered_events=tuple(supported_events),
        snapshots=snapshots,
        failures=failures,
        unsupported_events=tuple(unsupported_events),
    )


def _render_markdown_output(
    *,
    slate: SportyBetTodaySlate,
    ordered_events: Sequence[SportyBetTodayEvent],
    snapshots: Sequence[FixtureMarketSnapshot],
    failures: Mapping[str, str],
    unsupported_events: Sequence[SportyBetTodayEvent],
) -> str:
    """Render one fully expanded today slate as markdown."""

    snapshot_by_fixture_ref = {snapshot.fixture_ref: snapshot for snapshot in snapshots}
    rendered_lines = [
        "# SportyBet Today Games Full Market Export",
        "",
        f"- Generated: {slate.fetched_at.astimezone(WAT_TIMEZONE).strftime('%Y-%m-%d %H:%M %Z')}",
        f"- Feed run date: {slate.run_date.isoformat()}",
        f"- Sports requested: {', '.join(route.sport_name for route in slate.sports) or 'all'}",
        f"- Today-feed events returned: {len(slate.events)}",
        f"- Full-market export attempted: {len(ordered_events)}",
        f"- Full-market export failures: {len(failures)}",
        f"- Unsupported events skipped: {len(unsupported_events)}",
    ]

    out_of_run_date_events = tuple(
        event
        for event in ordered_events
        if event.kickoff.astimezone(WAT_TIMEZONE).date() != slate.run_date
    )
    if out_of_run_date_events:
        distinct_dates = ", ".join(
            sorted(
                {
                    event.kickoff.astimezone(WAT_TIMEZONE).date().isoformat()
                    for event in out_of_run_date_events
                }
            )
        )
        rendered_lines.extend(
            [
                "",
                (
                    "SportyBet's live Today Games feed included fixtures outside the run-date "
                    f"window. Extra kickoff dates in this export: {distinct_dates}."
                ),
            ]
        )

    if unsupported_events:
        rendered_lines.extend(
            [
                "",
                "Unsupported events were skipped because full-market export currently supports "
                "Football and Basketball only.",
            ]
        )

    current_date: date | None = None
    for event in ordered_events:
        kickoff_wat = event.kickoff.astimezone(WAT_TIMEZONE)
        kickoff_date = kickoff_wat.date()
        if kickoff_date != current_date:
            current_date = kickoff_date
            rendered_lines.extend(["", f"## {kickoff_date.isoformat()}", ""])

        fixture_ref = event.event_id
        snapshot = snapshot_by_fixture_ref.get(fixture_ref)
        rendered_lines.append(
            f"### {kickoff_wat.strftime('%H:%M')} WAT - {event.home_team_name} vs {event.away_team_name}"
        )
        rendered_lines.append(f"{event.tournament_name} / {event.category_name}")
        rendered_lines.append(
            "Event ID: "
            f"`{event.event_id}` | Game ID: `{event.game_id or 'n/a'}` | "
            f"Status: `{event.match_status or event.status or 'unknown'}`"
        )
        if snapshot is None or snapshot.fetched_market_count == 0:
            rendered_lines.append(
                "Today-feed listing markets: "
                f"`{event.market_count}` | Reported total markets: `{event.total_market_size}`"
            )
            failure = failures.get(fixture_ref)
            if failure is not None:
                rendered_lines.append(f"Full-market fetch failed: `{failure}`")
            else:
                rendered_lines.append("Full-market fetch returned no market rows for this fixture.")
            rendered_lines.append("")
            continue

        rendered_lines.append(
            "Today-feed listing markets: "
            f"`{event.market_count}` | Reported total markets: `{event.total_market_size}` | "
            f"Full fetched markets: `{snapshot.fetched_market_count}` | "
            f"Full fetched selections: `{snapshot.fetched_selection_count}` | "
            f"Scoreable markets: `{snapshot.scoreable_market_count}` | "
            f"Unmapped markets: `{snapshot.unmapped_market_count}` | "
            f"Fetch source: `{snapshot.fetch_source or 'unknown'}`"
        )
        for market_group in snapshot.market_groups:
            rendered_lines.extend(["", f"#### {market_group.group_name}"])
            for market in market_group.markets:
                rendered_lines.append(_render_markdown_market_entry(market))
                rendered_lines.extend(
                    _render_markdown_market_selection(selection)
                    for selection in market.selections
                )
        rendered_lines.append("")

    return "\n".join(rendered_lines).rstrip() + "\n"


def _render_markdown_market_entry(market: FixtureMarketSnapshotEntry) -> str:
    """Render one market header line inside the markdown export."""

    mapped_markets = (
        ", ".join(market_type.value for market_type in market.canonical_markets)
        if market.canonical_markets
        else "unmapped"
    )
    return (
        f"- {market.market_label} [key={market.provider_market_key}] | "
        f"market_id={market.provider_market_id or 'n/a'} | "
        f"mapped={mapped_markets} | selections={len(market.selections)}"
    )


def _render_markdown_market_selection(selection: FixtureMarketSelection) -> str:
    """Render one markdown line for a market selection."""

    selection_label = selection.provider_selection_name
    if selection.canonical_selection is not None:
        selection_label += f" [{selection.canonical_selection}]"
    if selection.line is not None and str(selection.line) not in selection_label:
        selection_label += f" ({selection.line:g})"
    return f"  - {selection_label} @ {selection.odds:.2f}"


def build_fixtures_from_today_slate(
    slate: SportyBetTodaySlate,
) -> tuple[NormalizedFixture, ...]:
    """Build canonical normalized fixtures from a collected today slate."""

    fixtures: list[NormalizedFixture] = []
    seen_fixture_refs: set[str] = set()
    for event in _sort_today_events(slate.events):
        fixture = _build_fixture_from_today_event(event)
        if fixture is None:
            continue
        fixture_ref = fixture.get_fixture_ref()
        if fixture_ref in seen_fixture_refs:
            continue
        seen_fixture_refs.add(fixture_ref)
        fixtures.append(fixture)
    return tuple(fixtures)


def _sort_today_events(
    events: Sequence[SportyBetTodayEvent],
) -> tuple[SportyBetTodayEvent, ...]:
    """Sort events by kickoff time and stable display fields."""

    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.kickoff.astimezone(WAT_TIMEZONE),
                event.tournament_name.casefold(),
                event.home_team_name.casefold(),
                event.away_team_name.casefold(),
            ),
        )
    )


def _build_fixture_from_today_event(event: SportyBetTodayEvent) -> NormalizedFixture | None:
    """Build one normalized fixture used for full-market markdown export."""

    sport = FULL_MARKET_EXPORT_SUPPORTED_SPORTS.get(
        _normalize_sport_selector(event.sport_name)
    )
    if sport is None:
        return None

    return NormalizedFixture(
        sportradar_id=event.event_id,
        home_team=event.home_team_name,
        away_team=event.away_team_name,
        competition=event.tournament_name,
        sport=sport,
        kickoff=event.kickoff,
        source_provider="sportybet",
        source_id=event.game_id or event.event_id,
        country=event.category_name,
    )


def _seconds_until_next_rate_limit_window() -> int:
    """Return the number of seconds until SportyBet's next minute bucket opens."""

    current_time = datetime.now(WAT_TIMEZONE)
    window_seconds = SPORTYBET_RATE_LIMIT_POLICY.window_seconds
    seconds_into_window = int(current_time.timestamp()) % window_seconds
    wait_seconds = window_seconds - seconds_into_window
    return max(wait_seconds, 1)


def _validate_today_feed_payload(payload: Mapping[str, object]) -> None:
    """Validate the top-level SportyBet response wrapper before parsing rows."""

    if payload.get("bizCode") not in {None, 10000}:
        message = normalize_optional_text(_coerce_optional_text(payload.get("message"))) or "unknown"
        raise RuntimeError(
            "SportyBet today feed returned a non-success business code: "
            f"{payload.get('bizCode')} ({message})."
        )


def _dedupe_today_events(
    events: Sequence[SportyBetTodayEvent],
) -> tuple[SportyBetTodayEvent, ...]:
    """Deduplicate today events by their stable SportyBet event identifier."""

    deduped: dict[str, SportyBetTodayEvent] = {}
    for event in events:
        deduped.setdefault(event.event_id, event)
    return tuple(deduped.values())


def _normalize_sport_selector(value: str) -> str:
    """Normalize one sport name or route slug for case-insensitive matching."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _fallback_route_slug(sport_name: str) -> str:
    """Derive one route slug when SportyBet's live nav does not expose it."""

    normalized_selector = _normalize_sport_selector(sport_name)
    special_cases = {
        "leagueoflegends": "lol",
    }
    if normalized_selector in special_cases:
        return special_cases[normalized_selector]

    normalized_name = sport_name.strip()
    if normalized_name == "Counter-Strike":
        normalized_name = normalized_name.replace("-", "")
    parts = normalized_name.split()
    if not parts:
        raise ValueError("sport_name must not be blank.")
    return parts[0].casefold() + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _build_sport_page_url(route_slug: str) -> str:
    """Return the canonical SportyBet sport-page URL for one route slug."""

    normalized_slug = require_non_blank_text(route_slug, "route_slug")
    return f"{SPORTYBET_BASE_URL}/{SPORTYBET_COUNTRY_CODE}/sport/{normalized_slug}/"


def _extract_route_slug(url: str) -> str:
    """Extract one SportyBet route slug from a sport-page URL."""

    parsed_url = urlparse(require_non_blank_text(url, "url"))
    path_segments = [segment for segment in parsed_url.path.split("/") if segment]
    try:
        sport_index = path_segments.index("sport")
    except ValueError as exc:
        raise ValueError(f"SportyBet sport URL is missing the '/sport/' segment: {url}") from exc
    if len(path_segments) <= sport_index + 1:
        raise ValueError(f"SportyBet sport URL is missing a route slug: {url}")
    return require_non_blank_text(path_segments[sport_index + 1], "route_slug")


def _coerce_optional_text(value: object) -> str | None:
    """Coerce strings, ints, and floats into optional text."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _normalize_optional_identifier(value: object) -> str | None:
    """Normalize one optional SportyBet identifier into string form."""

    return normalize_optional_text(_coerce_optional_text(value))


def _require_identifier(value: object, *, field_name: str) -> str:
    """Require one non-blank identifier and return it as text."""

    normalized = _normalize_optional_identifier(value)
    if normalized is None:
        raise ValueError(f"{field_name} must not be blank.")
    return require_non_blank_text(normalized, field_name)


def _require_text(value: object, *, field_name: str) -> str:
    """Require one non-blank text value and return it as a normalized string."""

    normalized = normalize_optional_text(_coerce_optional_text(value))
    if normalized is None:
        raise ValueError(f"{field_name} must not be blank.")
    return require_non_blank_text(normalized, field_name)


def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    """Require one non-negative integer-compatible value."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not a boolean.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field_name} must not be negative.")
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be a whole number.")
        if value < 0:
            raise ValueError(f"{field_name} must not be negative.")
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank.")
        parsed = int(normalized)
        if parsed < 0:
            raise ValueError(f"{field_name} must not be negative.")
        return parsed
    raise TypeError(f"{field_name} must be integer-compatible.")


def _coerce_optional_int(value: object) -> int | None:
    """Coerce one optional integer-compatible value."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("status must be an integer, not a boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("status must be a whole number.")
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return int(normalized)
    raise TypeError("status must be integer-compatible.")


def _coerce_kickoff(value: object, *, field_name: str) -> datetime:
    """Convert one epoch-millisecond kickoff into a WAT datetime."""

    timestamp_ms = _coerce_non_negative_int(value, field_name=field_name)
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).astimezone(WAT_TIMEZONE)


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    """Require one object-like mapping."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"SportyBet payload field '{field_name}' must be an object mapping.")
    return value


__all__ = [
    "DEFAULT_OUTPUT_FORMAT",
    "SportyBetTodayEvent",
    "SportyBetTodayGamesCollector",
    "SportyBetTodaySlate",
    "SportyBetTodaySport",
    "run",
]


if __name__ == "__main__":
    run()
