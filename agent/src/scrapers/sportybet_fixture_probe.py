"""Playwright probe for SportyBet fixture-page detail and stats requests.

Purpose: capture the real SportyBet fixture-page API responses that expose
team/player detail widgets, match-analysis summaries, and related statistics.
Scope: open one public SportyBet fixture page, observe `factsCenter` JSON
responses, click likely detail tabs, summarize keyword-rich payloads, and
optionally save the raw payloads for later integration work.
Dependencies: Playwright for browser execution, shared SportyBet constants,
and pure helper functions for URL construction and payload keyword discovery.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Page, Playwright, Response
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import WAT_TIMEZONE_NAME
from src.schemas.common import normalize_optional_text, require_non_blank_text, slugify_segment
from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SPORTYBET_BASE_URL,
    SPORTYBET_COUNTRY_CODE,
)

SPORTYBET_FACTS_CENTER_PATH: Final[str] = f"/api/{SPORTYBET_COUNTRY_CODE}/factsCenter/"
SPORTRADAR_WIDGET_HOST: Final[str] = "widgets.sir.sportradar.com"
DEFAULT_SPORTRADAR_WIDGET_CLIENT_ID: Final[str] = "638846b93b23ecfc94ce1a6d45b1dbe6"
DEFAULT_NAVIGATION_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_RESPONSE_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_SETTLE_DELAY_MS: Final[int] = 1_500
DEFAULT_SPORT: Final[str] = "football"
DEFAULT_BOOTSTRAP_LABELS: Final[tuple[str, ...]] = ("OK",)
DEFAULT_DETAIL_LABELS: Final[tuple[str, ...]] = (
    "Stats",
    "Statistics",
    "Analysis",
    "Info",
    "Lineups",
    "Lineup",
    "Form",
    "H2H",
    "Head to Head",
    "Standings",
)
DEFAULT_DETAIL_KEYWORDS: Final[tuple[str, ...]] = (
    "stat",
    "stats",
    "team",
    "teams",
    "player",
    "players",
    "win",
    "wins",
    "draw",
    "draws",
    "loss",
    "losses",
    "yellow",
    "red",
    "card",
    "cards",
    "lineup",
    "formation",
    "standing",
    "standings",
    "head",
    "h2h",
)
DEFAULT_WIDGET_TYPE_KEYS: Final[tuple[str, ...]] = (
    "statistics",
    "lineups",
    "comparison",
    "standings",
    "probability",
    "preview",
    "teamInfo",
    "h2h",
    "table",
)
SPORTRADAR_WIDGET_TYPES: Final[dict[str, str]] = {
    "commentary": "match.commentary",
    "matchTracker": "match.lmtPlus",
    "h2h": "match.headToHead",
    "h2h_V3": "headToHead.standalone",
    "table": "season.liveTable",
    "lineups": "match.lineups",
    "avgGoals": "match.avgGoalsScoredConceded",
    "statistics": "match.statistics",
    "topLists": "season.topLists",
    "leagueTable": "season.liveTable",
    "cupRoster": "season.cupRoster",
    "standings": "season.standings",
    "comparison": "team.comparison",
    "probability": "match.winProbability",
    "preview": "match.preview",
    "teamInfo": "team.info",
    "BetInsights": "betInsights",
}
SPORTRADAR_EVENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"sr:match:(\d+)")
MAX_MATCHED_PATHS: Final[int] = 40
MAX_MAPPING_ITEMS: Final[int] = 60
MAX_SEQUENCE_ITEMS: Final[int] = 20
MAX_PAYLOAD_DEPTH: Final[int] = 7
MAX_TEXT_PAYLOAD_CHARS: Final[int] = 250_000
VISIBLE_LABELS_SCRIPT: Final[str] = """
() => Array.from(document.querySelectorAll("button, [role='button'], [role='tab'], a"))
  .map((element) => (element.textContent || "").replace(/\\s+/g, " ").trim())
  .filter(Boolean)
"""


class SportyBetProbeResponse(BaseModel):
    """Summary of one captured SportyBet `factsCenter` response."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: Literal["facts_center", "sportradar"] = Field(
        description="Detected response source family.",
    )
    url: str = Field(description="Exact response URL observed in the browser.")
    path: str = Field(description="Path segment parsed from the response URL.")
    status: int = Field(ge=100, le=599, description="HTTP status code.")
    content_type: str | None = Field(
        default=None,
        description="Response content-type header when available.",
    )
    body_kind: Literal["json", "text"] = Field(
        description="Captured payload storage format.",
    )
    top_level_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Top-level object keys when the payload is a mapping.",
    )
    matched_keywords: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Keywords found in the payload or URL.",
    )
    matched_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Representative payload paths that matched the probe keywords.",
    )
    is_candidate: bool = Field(
        default=False,
        description="Whether this response appears to contain detail/stat data.",
    )
    saved_path: str | None = Field(
        default=None,
        description="Optional local file path where the raw payload was written.",
    )

    @field_validator("url", "path")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required URL fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("saved_path")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional saved-path fields."""

        return normalize_optional_text(value)


class SportyBetProbeWidgetInvocation(BaseModel):
    """Summary of one direct Sportradar widget invocation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    widget_key: str = Field(description="Short internal widget key.")
    widget_type: str = Field(description="Sportradar widget type string.")
    match_id: str = Field(description="Numeric Sportradar match identifier.")
    status: Literal["mounted", "timeout", "error", "unavailable"] = Field(
        description="Outcome of the widget invocation attempt.",
    )
    error_message: str | None = Field(
        default=None,
        description="Optional error details for failed widget invocations.",
    )

    @field_validator("widget_key", "widget_type", "match_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required widget fields."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("error_message")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional widget error fields."""

        return normalize_optional_text(value)


class SportyBetFixtureProbeResult(BaseModel):
    """Top-level result returned by the fixture-page probe."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_url: str = Field(description="Requested SportyBet fixture page URL.")
    final_url: str = Field(description="Final URL after page navigation.")
    page_title: str | None = Field(default=None, description="Resolved document title.")
    visible_labels: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Distinct clickable labels discovered on the page.",
    )
    clicked_labels: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Detail-related labels the probe clicked successfully.",
    )
    widget_invocations: tuple[SportyBetProbeWidgetInvocation, ...] = Field(
        default_factory=tuple,
        description="Direct Sportradar widget calls issued during the probe.",
    )
    response_count: int = Field(ge=0, description="Total captured `factsCenter` responses.")
    candidate_response_count: int = Field(
        ge=0,
        description="Captured responses that matched detail/stat heuristics.",
    )
    responses: tuple[SportyBetProbeResponse, ...] = Field(
        default_factory=tuple,
        description="Ordered summaries for all captured `factsCenter` responses.",
    )

    @field_validator("fixture_url", "final_url")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank URLs in the probe result."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("page_title")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional page-title values."""

        return normalize_optional_text(value)


@dataclass(frozen=True, slots=True)
class _CapturedResponse:
    """Internal captured response payload before serialization."""

    summary: SportyBetProbeResponse
    payload: object


class SportyBetFixturePageProbe:
    """Capture SportyBet fixture-page detail requests through Playwright."""

    def __init__(
        self,
        *,
        user_agents: Sequence[str] = DEFAULT_USER_AGENTS,
        headless: bool = True,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        response_timeout_ms: int = DEFAULT_RESPONSE_TIMEOUT_MS,
        settle_delay_ms: int = DEFAULT_SETTLE_DELAY_MS,
        bootstrap_labels: Sequence[str] = DEFAULT_BOOTSTRAP_LABELS,
        detail_labels: Sequence[str] = DEFAULT_DETAIL_LABELS,
        detail_keywords: Sequence[str] = DEFAULT_DETAIL_KEYWORDS,
        widget_type_keys: Sequence[str] = DEFAULT_WIDGET_TYPE_KEYS,
    ) -> None:
        """Store deterministic runtime settings for the probe."""

        if not user_agents:
            raise ValueError("user_agents must contain at least one value.")
        if navigation_timeout_ms <= 0:
            raise ValueError("navigation_timeout_ms must be positive.")
        if response_timeout_ms <= 0:
            raise ValueError("response_timeout_ms must be positive.")
        if settle_delay_ms < 0:
            raise ValueError("settle_delay_ms must be zero or positive.")
        if not bootstrap_labels:
            raise ValueError("bootstrap_labels must contain at least one value.")
        if not detail_labels:
            raise ValueError("detail_labels must contain at least one value.")
        if not detail_keywords:
            raise ValueError("detail_keywords must contain at least one value.")
        if not widget_type_keys:
            raise ValueError("widget_type_keys must contain at least one value.")

        self._user_agent = require_non_blank_text(user_agents[0], "user_agent")
        self._headless = headless
        self._navigation_timeout_ms = navigation_timeout_ms
        self._response_timeout_ms = response_timeout_ms
        self._settle_delay_ms = settle_delay_ms
        self._bootstrap_labels = tuple(
            require_non_blank_text(label, "bootstrap_label") for label in bootstrap_labels
        )
        self._detail_labels = tuple(
            require_non_blank_text(label, "detail_label") for label in detail_labels
        )
        self._detail_keywords = tuple(
            require_non_blank_text(keyword, "detail_keyword").casefold()
            for keyword in detail_keywords
        )
        normalized_widget_type_keys = tuple(
            require_non_blank_text(widget_type_key, "widget_type_key")
            for widget_type_key in widget_type_keys
        )
        unknown_widget_types = [
            widget_type_key
            for widget_type_key in normalized_widget_type_keys
            if widget_type_key not in SPORTRADAR_WIDGET_TYPES
        ]
        if unknown_widget_types:
            unknown_text = ", ".join(sorted(unknown_widget_types))
            raise ValueError(f"Unsupported widget_type_keys: {unknown_text}.")
        self._widget_type_keys = normalized_widget_type_keys

    async def probe(
        self,
        *,
        fixture_url: str,
        output_dir: Path | None = None,
    ) -> SportyBetFixtureProbeResult:
        """Probe one public SportyBet fixture page and summarize detail responses."""

        normalized_url = require_non_blank_text(fixture_url, "fixture_url")

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            context = await browser.new_context(
                locale="en-NG",
                timezone_id=WAT_TIMEZONE_NAME,
                user_agent=self._user_agent,
            )
            try:
                result = await self._probe_in_context(
                    context,
                    fixture_url=normalized_url,
                    output_dir=output_dir,
                )
            finally:
                await context.close()
                await browser.close()

        return result

    async def _launch_browser(self, playwright: Playwright) -> Browser:
        """Launch Chromium with clear recovery messaging for missing binaries."""

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

    async def _probe_in_context(
        self,
        context: BrowserContext,
        *,
        fixture_url: str,
        output_dir: Path | None,
    ) -> SportyBetFixtureProbeResult:
        """Execute one probe using an already-initialized browser context."""

        page = await context.new_page()
        captured_responses: list[_CapturedResponse] = []
        pending_capture_tasks: set[asyncio.Task[None]] = set()

        def schedule_capture(response: Response) -> None:
            """Queue one async response capture task from Playwright's sync callback."""

            task = asyncio.create_task(
                self._capture_response(response, captured_responses)
            )
            pending_capture_tasks.add(task)
            task.add_done_callback(pending_capture_tasks.discard)

        page.on("response", schedule_capture)

        try:
            await page.goto(
                fixture_url,
                wait_until="domcontentloaded",
                timeout=float(self._navigation_timeout_ms),
            )
            with suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=float(self._response_timeout_ms),
                )
            await page.wait_for_timeout(float(self._settle_delay_ms))
            widget_invocations = await self._mount_sportradar_widgets(
                page,
                fixture_url=fixture_url,
            )
            await self._activate_bootstrap_controls(page)
            await self._trigger_lazy_sections(page)

            visible_labels = await _extract_visible_labels(page)
            clicked_labels = await self._click_detail_labels(page, visible_labels)

            await page.wait_for_timeout(float(self._settle_delay_ms))
            if pending_capture_tasks:
                await asyncio.gather(*pending_capture_tasks, return_exceptions=True)

            serialized_responses = self._serialize_captured_responses(
                captured_responses,
                output_dir=output_dir,
            )
            page_title = await page.title()

            candidate_count = sum(1 for response in serialized_responses if response.is_candidate)
            return SportyBetFixtureProbeResult(
                fixture_url=fixture_url,
                final_url=page.url,
                page_title=page_title,
                visible_labels=tuple(visible_labels),
                clicked_labels=tuple(clicked_labels),
                widget_invocations=widget_invocations,
                response_count=len(serialized_responses),
                candidate_response_count=candidate_count,
                responses=tuple(serialized_responses),
            )
        finally:
            await page.close()

    async def _capture_response(
        self,
        response: Response,
        captured_responses: list[_CapturedResponse],
    ) -> None:
        """Capture one candidate SportyBet or Sportradar response."""

        source = _classify_response_source(response.url)
        if source is None:
            return
        if response.status >= 400:
            return

        content_type = normalize_optional_text(response.headers.get("content-type"))
        try:
            if content_type is not None and "json" in content_type.casefold():
                payload = await response.json()
                summary = _summarize_json_response(
                    source=source,
                    response_url=response.url,
                    status=response.status,
                    content_type=content_type,
                    payload=payload,
                    keywords=self._detail_keywords,
                )
            else:
                payload = await response.text()
                if len(payload) > MAX_TEXT_PAYLOAD_CHARS:
                    payload = payload[:MAX_TEXT_PAYLOAD_CHARS]
                summary = _summarize_text_response(
                    source=source,
                    response_url=response.url,
                    status=response.status,
                    content_type=content_type,
                    payload=payload,
                    keywords=self._detail_keywords,
                )
        except PlaywrightError:
            return

        captured_responses.append(_CapturedResponse(summary=summary, payload=payload))

    async def _click_detail_labels(
        self,
        page: Page,
        visible_labels: Sequence[str],
    ) -> tuple[str, ...]:
        """Click visible detail-oriented labels to trigger additional requests."""

        clicked_labels: list[str] = []
        visible_lookup = {label.casefold(): label for label in visible_labels}

        for label in self._detail_labels:
            actual_label = visible_lookup.get(label.casefold())
            if actual_label is None:
                continue
            if actual_label.casefold() in {clicked.casefold() for clicked in clicked_labels}:
                continue
            if await self._click_label(page, actual_label):
                clicked_labels.append(actual_label)

        return tuple(clicked_labels)

    async def _click_label(self, page: Page, label: str) -> bool:
        """Attempt to click one label through common accessible roles."""

        locators = (
            page.get_by_role("tab", name=label, exact=True),
            page.get_by_role("button", name=label, exact=True),
            page.get_by_role("link", name=label, exact=True),
            page.get_by_text(label, exact=True),
        )

        for locator in locators:
            try:
                if await locator.count() == 0:
                    continue
                await locator.first.click(timeout=float(self._response_timeout_ms))
            except (PlaywrightError, PlaywrightTimeoutError):
                continue

            with suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=float(self._response_timeout_ms),
                )
            await page.wait_for_timeout(float(self._settle_delay_ms))
            return True

        return False

    async def _activate_bootstrap_controls(self, page: Page) -> None:
        """Dismiss obvious blocking controls such as consent or lazy-load buttons."""

        for label in self._bootstrap_labels:
            with suppress(PlaywrightError, PlaywrightTimeoutError):
                await self._click_label(page, label)

    async def _trigger_lazy_sections(self, page: Page) -> None:
        """Scroll through the fixture page to trigger lazy-loaded widgets."""

        with suppress(PlaywrightError):
            await page.evaluate(
                """
                () => {
                  window.scrollTo({ top: document.body.scrollHeight, behavior: "instant" });
                }
                """
            )
        await page.wait_for_timeout(float(self._settle_delay_ms))
        with suppress(PlaywrightError):
            await page.evaluate(
                """
                () => {
                  window.scrollTo({ top: 0, behavior: "instant" });
                }
                """
            )
        await page.wait_for_timeout(float(self._settle_delay_ms))

    async def _mount_sportradar_widgets(
        self,
        page: Page,
        *,
        fixture_url: str,
    ) -> tuple[SportyBetProbeWidgetInvocation, ...]:
        """Invoke the known Sportradar fixture widgets for one SportyBet match."""

        match_id = extract_match_id_from_fixture_url(fixture_url)
        if match_id is None:
            return ()

        with suppress(PlaywrightTimeoutError):
            await page.wait_for_function(
                "() => typeof window.SIR === 'function'",
                timeout=float(self._response_timeout_ms),
            )

        widget_definitions = [
            {
                "widget_key": widget_type_key,
                "widget_type": SPORTRADAR_WIDGET_TYPES[widget_type_key],
            }
            for widget_type_key in self._widget_type_keys
        ]
        invocation_payload = await page.evaluate(
            """
            async ({
              matchId,
              widgetDefinitions,
              responseTimeoutMs,
              settleDelayMs,
              widgetLoaderUrl,
            }) => {
              const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
              const ensureWidgetLoader = async () => {
                if (typeof window.SIR === "function") {
                  return null;
                }

                const existingScript = document.querySelector(
                  `script[data-probe-widget-loader="true"][src="${widgetLoaderUrl}"]`
                );
                if (existingScript instanceof HTMLScriptElement) {
                  await new Promise((resolve) => window.setTimeout(resolve, 1000));
                  return typeof window.SIR === "function"
                    ? null
                    : "Sportradar widget loader script was present but did not expose window.SIR.";
                }

                const script = document.createElement("script");
                script.src = widgetLoaderUrl;
                script.async = true;
                script.setAttribute("data-probe-widget-loader", "true");

                const loadError = await new Promise((resolve) => {
                  let settled = false;
                  const finish = (value) => {
                    if (settled) {
                      return;
                    }
                    settled = true;
                    resolve(value);
                  };

                  const timeoutId = window.setTimeout(() => {
                    finish("Timed out while loading the Sportradar widget loader script.");
                  }, responseTimeoutMs);

                  script.addEventListener("load", () => {
                    window.clearTimeout(timeoutId);
                    finish(null);
                  });
                  script.addEventListener("error", () => {
                    window.clearTimeout(timeoutId);
                    finish("The Sportradar widget loader script failed to load.");
                  });
                  (document.body || document.documentElement).appendChild(script);
                });

                if (loadError) {
                  return loadError;
                }

                await wait(250);
                return typeof window.SIR === "function"
                  ? null
                  : "The Sportradar widget loader script loaded but window.SIR is still unavailable.";
              };

              const loaderError = await ensureWidgetLoader();
              if (typeof window.SIR !== "function") {
                return widgetDefinitions.map((definition) => ({
                  widget_key: definition.widget_key,
                  widget_type: definition.widget_type,
                  match_id: matchId,
                  status: "unavailable",
                  error_message: loaderError || "window.SIR is not available on the page.",
                }));
              }

              const root = document.body || document.documentElement;
              const invocations = [];
              for (const definition of widgetDefinitions) {
                const containerId = `fixture-probe-widget-${definition.widget_key}`;
                let container = document.getElementById(containerId);
                if (!container) {
                  container = document.createElement("div");
                  container.id = containerId;
                  container.setAttribute("data-probe-widget-type", definition.widget_type);
                  root.appendChild(container);
                }

                const props = {
                  matchId,
                  pitchCustomBgColor: "#0E8E36",
                  adsFrequency: false,
                  activeSwitcher: "scoreDetails",
                  tabsPosition: "top",
                  goalBannerCustomBgColor: "#E41827",
                  logoLink: "",
                };
                if (definition.widget_key !== "h2h_V3") {
                  props.layout = "double";
                }

                const invocation = await new Promise((resolve) => {
                  let settled = false;
                  const finish = (status, errorMessage) => {
                    if (settled) {
                      return;
                    }
                    settled = true;
                    resolve({
                      widget_key: definition.widget_key,
                      widget_type: definition.widget_type,
                      match_id: matchId,
                      status,
                      error_message: errorMessage || null,
                    });
                  };

                  const timeoutId = window.setTimeout(() => {
                    finish("timeout", "Widget callback did not fire before the timeout.");
                  }, responseTimeoutMs);

                  try {
                    window.SIR(
                      "addWidget",
                      `#${containerId}`,
                      definition.widget_type,
                      props,
                      () => {
                        window.clearTimeout(timeoutId);
                        finish("mounted", null);
                      }
                    );
                  } catch (error) {
                    window.clearTimeout(timeoutId);
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    finish("error", errorMessage);
                  }
                });

                invocations.push(invocation);
                await wait(settleDelayMs);
              }

              return invocations;
            }
            """,
            {
                "matchId": match_id,
                "widgetDefinitions": widget_definitions,
                "responseTimeoutMs": self._response_timeout_ms,
                "settleDelayMs": self._settle_delay_ms,
                "widgetLoaderUrl": (
                    f"https://{SPORTRADAR_WIDGET_HOST}/"
                    f"{DEFAULT_SPORTRADAR_WIDGET_CLIENT_ID}/widgetloader"
                ),
            },
        )
        if not isinstance(invocation_payload, list):
            raise RuntimeError("Sportradar widget invocation payload was invalid.")

        invocations: list[SportyBetProbeWidgetInvocation] = []
        for raw_invocation in invocation_payload:
            if not isinstance(raw_invocation, Mapping):
                continue
            invocations.append(
                SportyBetProbeWidgetInvocation.model_validate(raw_invocation)
            )
        return tuple(invocations)

    def _serialize_captured_responses(
        self,
        captured_responses: Sequence[_CapturedResponse],
        *,
        output_dir: Path | None,
    ) -> list[SportyBetProbeResponse]:
        """Dedupe captured responses and optionally persist raw payloads to disk."""

        deduped_by_url: dict[str, _CapturedResponse] = {}
        for captured in captured_responses:
            deduped_by_url[captured.summary.url] = captured

        ordered_captured = list(deduped_by_url.values())
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        serialized: list[SportyBetProbeResponse] = []
        for index, captured in enumerate(ordered_captured, start=1):
            saved_path: str | None = None
            if output_dir is not None:
                file_path = output_dir / _build_response_file_name(
                    index,
                    captured.summary.url,
                    body_kind=captured.summary.body_kind,
                )
                if captured.summary.body_kind == "json":
                    file_path.write_text(
                        json.dumps(captured.payload, indent=2, sort_keys=True, ensure_ascii=True),
                        encoding="utf-8",
                    )
                else:
                    file_path.write_text(str(captured.payload), encoding="utf-8")
                saved_path = str(file_path)

            serialized.append(
                captured.summary.model_copy(
                    update={"saved_path": saved_path},
                )
            )

        return serialized


def build_fixture_page_url(
    *,
    event_id: str,
    home_team: str,
    away_team: str,
    country: str,
    competition: str,
    sport: str = DEFAULT_SPORT,
) -> str:
    """Build a public SportyBet fixture-page URL from normalized fixture metadata."""

    normalized_event_id = require_non_blank_text(event_id, "event_id")
    normalized_home_team = require_non_blank_text(home_team, "home_team")
    normalized_away_team = require_non_blank_text(away_team, "away_team")
    normalized_country = require_non_blank_text(country, "country")
    normalized_competition = require_non_blank_text(competition, "competition")
    normalized_sport = require_non_blank_text(sport, "sport").casefold()

    if normalized_sport not in {"football", "basketball"}:
        raise ValueError("sport must currently be either 'football' or 'basketball'.")

    country_segment = slugify_segment(normalized_country)
    competition_segment = slugify_segment(normalized_competition)
    home_segment = normalized_home_team.replace(" ", "_")
    away_segment = normalized_away_team.replace(" ", "_")
    match_segment = f"{home_segment}_vs_{away_segment}"
    return (
        f"{SPORTYBET_BASE_URL}/{SPORTYBET_COUNTRY_CODE}/sport/{normalized_sport}/"
        f"{country_segment}/{competition_segment}/{match_segment}/{normalized_event_id}"
    )


def _summarize_json_response(
    *,
    source: Literal["facts_center", "sportradar"],
    response_url: str,
    status: int,
    content_type: str | None,
    payload: object,
    keywords: Sequence[str],
) -> SportyBetProbeResponse:
    """Summarize one captured JSON response for operator review."""

    parsed_url = urlparse(response_url)
    top_level_keys = tuple(payload.keys()) if isinstance(payload, Mapping) else ()
    matched_paths = _collect_keyword_paths(payload, keywords=keywords)
    matched_keywords = _matched_keywords_for_response(
        response_url=response_url,
        matched_paths=matched_paths,
        keywords=keywords,
    )
    return SportyBetProbeResponse(
        source=source,
        url=response_url,
        path=parsed_url.path,
        status=status,
        content_type=content_type,
        body_kind="json",
        top_level_keys=top_level_keys,
        matched_keywords=matched_keywords,
        matched_paths=matched_paths,
        is_candidate=bool(matched_keywords),
    )


def _summarize_text_response(
    *,
    source: Literal["facts_center", "sportradar"],
    response_url: str,
    status: int,
    content_type: str | None,
    payload: str,
    keywords: Sequence[str],
) -> SportyBetProbeResponse:
    """Summarize one captured text response for operator review."""

    parsed_url = urlparse(response_url)
    normalized_text = payload.casefold()
    matched_keywords = tuple(
        keyword
        for keyword in keywords
        if keyword.casefold() in response_url.casefold() or keyword.casefold() in normalized_text
    )
    return SportyBetProbeResponse(
        source=source,
        url=response_url,
        path=parsed_url.path,
        status=status,
        content_type=content_type,
        body_kind="text",
        top_level_keys=(),
        matched_keywords=matched_keywords,
        matched_paths=(),
        is_candidate=bool(matched_keywords),
    )


def _collect_keyword_paths(
    payload: object,
    *,
    keywords: Sequence[str],
) -> tuple[str, ...]:
    """Collect representative payload paths containing the probe keywords."""

    matched_paths: list[str] = []
    lowered_keywords = tuple(keyword.casefold() for keyword in keywords)

    def record_match(path: str) -> None:
        if path in matched_paths:
            return
        matched_paths.append(path)

    def walk(value: object, *, path: str, depth: int) -> None:
        if depth > MAX_PAYLOAD_DEPTH or len(matched_paths) >= MAX_MATCHED_PATHS:
            return

        if isinstance(value, Mapping):
            for key, nested_value in list(value.items())[:MAX_MAPPING_ITEMS]:
                normalized_key = str(key).casefold()
                nested_path = f"{path}.{key}" if path else str(key)
                if any(keyword in normalized_key for keyword in lowered_keywords):
                    record_match(nested_path)
                walk(nested_value, path=nested_path, depth=depth + 1)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value[:MAX_SEQUENCE_ITEMS]):
                walk(item, path=f"{path}[{index}]", depth=depth + 1)
            return

        if isinstance(value, str):
            normalized_value = value.casefold()
            if any(keyword in normalized_value for keyword in lowered_keywords):
                record_match(path)

    walk(payload, path="", depth=0)
    return tuple(matched_paths)


def _matched_keywords_for_response(
    *,
    response_url: str,
    matched_paths: Sequence[str],
    keywords: Sequence[str],
) -> tuple[str, ...]:
    """Return the unique matched keywords found in the response URL or payload paths."""

    normalized_sources = [response_url.casefold(), *[path.casefold() for path in matched_paths]]
    matched_keywords = [
        keyword
        for keyword in keywords
        if any(keyword.casefold() in source for source in normalized_sources)
    ]
    return tuple(dict.fromkeys(matched_keywords))


async def _extract_visible_labels(page: Page) -> tuple[str, ...]:
    """Extract distinct clickable labels from the loaded fixture page."""

    raw_labels = await page.evaluate(VISIBLE_LABELS_SCRIPT)
    if not isinstance(raw_labels, list):
        raise RuntimeError("Fixture-page label extraction returned an invalid payload.")

    visible_labels: list[str] = []
    seen: set[str] = set()
    for value in raw_labels:
        if not isinstance(value, str):
            continue
        label = normalize_optional_text(value)
        if label is None:
            continue
        lookup_key = label.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        visible_labels.append(label)
    return tuple(visible_labels)


def _build_response_file_name(
    index: int,
    response_url: str,
    *,
    body_kind: Literal["json", "text"],
) -> str:
    """Build a stable filename for one captured response payload."""

    parsed_url = urlparse(response_url)
    path_label = parsed_url.path.rsplit("/", maxsplit=1)[-1] or "response"
    digest = hashlib.sha1(response_url.encode("utf-8")).hexdigest()[:10]
    extension = "json" if body_kind == "json" else "txt"
    return f"{index:02d}-{path_label}-{digest}.{extension}"


def _classify_response_source(response_url: str) -> Literal["facts_center", "sportradar"] | None:
    """Classify a captured response URL into the known source families."""

    if SPORTYBET_FACTS_CENTER_PATH in response_url:
        return "facts_center"
    if SPORTRADAR_WIDGET_HOST in response_url:
        return "sportradar"
    return None


def extract_match_id_from_event_id(event_id: str) -> str:
    """Extract the numeric Sportradar match id from an `sr:match:*` identifier."""

    normalized_event_id = require_non_blank_text(event_id, "event_id")
    match = SPORTRADAR_EVENT_ID_PATTERN.search(normalized_event_id)
    if match is None:
        raise ValueError("event_id must include a valid `sr:match:<id>` segment.")
    return match.group(1)


def extract_match_id_from_fixture_url(fixture_url: str) -> str | None:
    """Extract the numeric Sportradar match id from a SportyBet fixture URL."""

    normalized_url = require_non_blank_text(fixture_url, "fixture_url")
    match = SPORTRADAR_EVENT_ID_PATTERN.search(normalized_url)
    if match is None:
        return None
    return match.group(1)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the SportyBet fixture-page probe."""

    parser = argparse.ArgumentParser(
        description=(
            "Open one SportyBet fixture page with Playwright and capture the "
            "underlying factsCenter detail/stat responses."
        )
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Full public SportyBet fixture-page URL to probe.",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Fixture `sr:match:*` identifier used to build the public page URL.",
    )
    parser.add_argument("--home-team", type=str, default=None, help="Home team name.")
    parser.add_argument("--away-team", type=str, default=None, help="Away team name.")
    parser.add_argument("--country", type=str, default=None, help="Competition country or region.")
    parser.add_argument("--competition", type=str, default=None, help="Competition name.")
    parser.add_argument(
        "--sport",
        type=str,
        default=DEFAULT_SPORT,
        help="Sport route segment used for URL construction. Default: football.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where raw captured payloads plus the summary will be written.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium in headed mode for interactive debugging.",
    )
    return parser


def _resolve_fixture_url(args: argparse.Namespace) -> str:
    """Resolve the target fixture URL from direct or constructed CLI inputs."""

    direct_url = normalize_optional_text(args.url)
    if direct_url is not None:
        return direct_url

    required_fields = {
        "event_id": normalize_optional_text(args.event_id),
        "home_team": normalize_optional_text(args.home_team),
        "away_team": normalize_optional_text(args.away_team),
        "country": normalize_optional_text(args.country),
        "competition": normalize_optional_text(args.competition),
    }
    missing_fields = [field_name for field_name, value in required_fields.items() if value is None]
    if missing_fields:
        joined_missing = ", ".join(missing_fields)
        raise ValueError(
            "Provide either `--url` or all of: --event-id, --home-team, "
            f"--away-team, --country, --competition. Missing: {joined_missing}."
        )

    return build_fixture_page_url(
        event_id=required_fields["event_id"] or "",
        home_team=required_fields["home_team"] or "",
        away_team=required_fields["away_team"] or "",
        country=required_fields["country"] or "",
        competition=required_fields["competition"] or "",
        sport=args.sport,
    )


async def _run_async(args: argparse.Namespace) -> str:
    """Execute the probe and return the serialized summary."""

    fixture_url = _resolve_fixture_url(args)
    probe = SportyBetFixturePageProbe(headless=not args.headful)
    result = await probe.probe(fixture_url=fixture_url, output_dir=args.output_dir)
    rendered = json.dumps(
        result.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )
    if args.output_dir is not None:
        summary_path = args.output_dir / "probe-summary.json"
        summary_path.write_text(rendered, encoding="utf-8")
    return rendered


def run(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for `puntlab-agent-sportybet-fixture-probe`."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    rendered = asyncio.run(_run_async(args))
    print(rendered)


__all__ = [
    "DEFAULT_DETAIL_KEYWORDS",
    "DEFAULT_DETAIL_LABELS",
    "DEFAULT_WIDGET_TYPE_KEYS",
    "SportyBetFixturePageProbe",
    "SportyBetFixtureProbeResult",
    "SportyBetProbeResponse",
    "SportyBetProbeWidgetInvocation",
    "build_fixture_page_url",
    "extract_match_id_from_event_id",
    "extract_match_id_from_fixture_url",
    "run",
    "_collect_keyword_paths",
]


if __name__ == "__main__":
    run()
