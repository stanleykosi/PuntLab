"""Playwright-based SportyBet fixture stats fetcher.

Purpose: access the richer SportyBet fixture-page details that are powered by
Sportradar widgets rather than the normal `factsCenter/event` market payload.
Scope: open one public SportyBet fixture page, fetch the widget loader through
the live browser context, mount fixture widgets by numeric match id, capture
their rendered text, and optionally persist supporting network responses.
Dependencies: Playwright for browser execution, shared SportyBet URL helpers,
and Pydantic models for deterministic result serialization.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Response,
    Route,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import WAT_TIMEZONE_NAME, SportName
from src.schemas.common import normalize_optional_text, require_non_blank_text
from src.schemas.fixture_details import FixtureDetails, FixtureDetailSection
from src.schemas.stats import TeamStats
from src.scrapers.sportybet_api import (
    DEFAULT_USER_AGENTS,
    SPORTYBET_BASE_URL,
)
from src.scrapers.sportybet_fixture_probe import build_fixture_page_url

DEFAULT_NAVIGATION_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_POST_MOUNT_WAIT_MS: Final[int] = 8_000
DEFAULT_RESPONSE_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_RESPONSE_BODY_TIMEOUT_MS: Final[int] = 10_000
DEFAULT_SETTLE_DELAY_MS: Final[int] = 1_500
DEFAULT_SPORT: Final[str] = "football"
DEFAULT_WIDGET_TIMEOUT_MS: Final[int] = 20_000
DEFAULT_WIDGET_KEYS: Final[tuple[str, ...]] = (
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
DEFAULT_SPORTRADAR_WIDGET_CLIENT_ID: Final[str] = "638846b93b23ecfc94ce1a6d45b1dbe6"
SPORTRADAR_WIDGET_HOST: Final[str] = "widgets.sir.sportradar.com"
SPORTRADAR_FISHNET_HOST: Final[str] = "widgets.fn.sportradar.com"
SPORTRADAR_WIDGET_LOADER_URL: Final[str] = (
    f"https://{SPORTRADAR_WIDGET_HOST}/{DEFAULT_SPORTRADAR_WIDGET_CLIENT_ID}/widgetloader"
)
SPORTRADAR_WIDGET_ROUTE_GLOB: Final[str] = f"https://{SPORTRADAR_WIDGET_HOST}/**"
SPORTRADAR_FISHNET_ROUTE_GLOB: Final[str] = f"https://{SPORTRADAR_FISHNET_HOST}/**"
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
SPORTRADAR_EVENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"(sr:match:\d+)")
SPORTRADAR_TEAM_INFO_PATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"/stats_team_info/(?P<team_uid>\d+)(?:$|[/?#])"
)
MAX_RESPONSE_PREVIEW_CHARS: Final[int] = 20_000
MAX_WIDGET_TEXT_CHARS: Final[int] = 50_000

INSTALL_WIDGETLOADER_SCRIPT: Final[str] = """
async (widgetLoaderUrl) => {
  if (typeof window.SIR === "function") {
    return typeof window.SIR;
  }

  const existingScript = document.querySelector(
    `script[data-sportyai-widget-loader="true"][src="${widgetLoaderUrl}"]`
  );
  if (existingScript instanceof HTMLScriptElement) {
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    return typeof window.SIR;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = widgetLoaderUrl;
    script.async = true;
    script.setAttribute("data-sportyai-widget-loader", "true");
    script.addEventListener("load", () => resolve(null), { once: true });
    script.addEventListener(
      "error",
      () => reject(new Error("The Sportradar widget loader script failed to load.")),
      { once: true }
    );
    (document.head || document.body || document.documentElement).appendChild(script);
  });
  await new Promise((resolve) => window.setTimeout(resolve, 250));
  return typeof window.SIR;
}
"""

MOUNT_WIDGET_SCRIPT: Final[str] = """
async (params) => {
  const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
  const normalize = (value) => typeof value === "string"
    ? value
        .replace(/\\u00a0/g, " ")
        .replace(/\\r/g, "\\n")
        .replace(/[ \\t]+/g, " ")
        .replace(/\\n{3,}/g, "\\n\\n")
        .trim()
    : "";

  const splitLines = (value) => {
    const normalized = normalize(value);
    if (!normalized) {
      return [];
    }
    const lines = normalized
      .split(/\\n+/)
      .map((line) => normalize(line))
      .filter(Boolean);
    const output = [];
    const seen = new Set();
    for (const line of lines) {
      const key = line.toLowerCase();
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      output.push(line);
    }
    return output;
  };

  const collectRoots = (container) => {
    const roots = [];
    const stack = [container];
    const visited = new Set();

    while (stack.length > 0) {
      const current = stack.pop();
      if (!(current instanceof Element) || visited.has(current)) {
        continue;
      }
      visited.add(current);
      roots.push(current);

      if (current.shadowRoot instanceof ShadowRoot) {
        roots.push(current.shadowRoot);
        for (const child of Array.from(current.shadowRoot.children)) {
          stack.push(child);
        }
      }

      for (const child of Array.from(current.children)) {
        stack.push(child);
      }
    }

    return roots;
  };

  const snapshotContainer = (container) => {
    const headings = [];
    const seenHeading = new Set();
    const textFragments = [];
    const seenText = new Set();
    const htmlFragments = [];
    const seenHtml = new Set();
    const iframeUrls = [];
    const seenIframe = new Set();

    for (const root of collectRoots(container)) {
      if (root instanceof Element || root instanceof ShadowRoot) {
        for (const element of root.querySelectorAll("h1, h2, h3, h4, [role='heading']")) {
          const headingText = normalize(element.innerText || element.textContent || "");
          if (!headingText) {
            continue;
          }
          const key = headingText.toLowerCase();
          if (seenHeading.has(key)) {
            continue;
          }
          seenHeading.add(key);
          headings.push(headingText);
        }

        for (const iframe of root.querySelectorAll("iframe")) {
          const iframeUrl = normalize(iframe.src || "");
          if (!iframeUrl) {
            continue;
          }
          const key = iframeUrl.toLowerCase();
          if (seenIframe.has(key)) {
            continue;
          }
          seenIframe.add(key);
          iframeUrls.push(iframeUrl);
        }
      }

      const rawText = normalize(
        (root instanceof ShadowRoot ? root.textContent : (root.innerText || root.textContent)) || ""
      );
      if (rawText) {
        const key = rawText.toLowerCase();
        if (!seenText.has(key)) {
          seenText.add(key);
          textFragments.push(rawText);
        }
      }

      const rawHtml = normalize(root.innerHTML || "");
      if (rawHtml) {
        const key = rawHtml.toLowerCase();
        if (!seenHtml.has(key)) {
          seenHtml.add(key);
          htmlFragments.push(rawHtml);
        }
      }
    }

    const contentText = textFragments.join("\\n\\n");
    return {
      headings,
      content_text: contentText,
      content_lines: splitLines(contentText),
      iframe_urls: iframeUrls,
      html: htmlFragments.join("\\n\\n"),
    };
  };

  const hasRenderableContent = (snapshot) => {
    if (!snapshot || typeof snapshot !== "object") {
      return false;
    }
    if (Array.isArray(snapshot.content_lines) && snapshot.content_lines.length > 0) {
      return true;
    }
    if (Array.isArray(snapshot.iframe_urls) && snapshot.iframe_urls.length > 0) {
      return true;
    }
    if (typeof snapshot.content_text === "string" && snapshot.content_text.trim().length > 0) {
      return true;
    }
    return typeof snapshot.html === "string" && snapshot.html.trim().length > 0;
  };

  const waitForRenderableContent = async (container) => {
    const deadline = Date.now() + params.postMountWaitMs;
    let snapshot = snapshotContainer(container);
    while (!hasRenderableContent(snapshot) && Date.now() < deadline) {
      await wait(250);
      snapshot = snapshotContainer(container);
    }
    return snapshot;
  };

  if (typeof window.SIR !== "function") {
    return {
      widget_key: params.widgetKey,
      widget_type: params.widgetType,
      status: "unavailable",
      error_message: "window.SIR is not available in the current page.",
      headings: [],
      content_text: "",
      content_lines: [],
      iframe_urls: [],
      html: "",
    };
  }

  const root = document.body || document.documentElement;
  let container = document.getElementById(params.containerId);
  if (!(container instanceof HTMLElement)) {
    container = document.createElement("div");
    container.id = params.containerId;
    root.appendChild(container);
  }
  container.classList.add("m-livetracker", "sportyai-sportradar-widget");
  container.setAttribute("data-sportyai-widget-key", params.widgetKey);
  container.style.display = "block";
  container.style.width = "100%";
  container.style.minHeight = `${params.minHeightPx}px`;
  container.style.margin = "0";
  container.style.padding = "0";
  container.innerHTML = "";

  return await new Promise((resolve) => {
    let settled = false;
    const finish = (status, errorMessage, snapshotOverride) => {
      if (settled) {
        return;
      }
      settled = true;
      const snapshot = snapshotOverride || snapshotContainer(container);
      resolve({
        widget_key: params.widgetKey,
        widget_type: params.widgetType,
        status,
        error_message: errorMessage || null,
        headings: snapshot.headings,
        content_text: snapshot.content_text,
        content_lines: snapshot.content_lines,
        iframe_urls: snapshot.iframe_urls,
        html: snapshot.html,
      });
    };

    const timeoutId = window.setTimeout(() => {
      finish("timeout", "Widget callback did not fire before the timeout.");
    }, params.widgetTimeoutMs);

    try {
      const addWidgetResult = window.SIR(
        "addWidget",
        params.selector,
        params.widgetType,
        params.props,
        () => {
          window.clearTimeout(timeoutId);
          window.setTimeout(async () => {
            const snapshot = await waitForRenderableContent(container);
            finish("mounted", null, snapshot);
          }, params.settleDelayMs);
        }
      );
      if (addWidgetResult === false) {
        window.clearTimeout(timeoutId);
        finish("error", "Sportradar rejected the widget mount request.");
      }
    } catch (error) {
      window.clearTimeout(timeoutId);
      const errorMessage = error instanceof Error ? error.message : String(error);
      finish("error", errorMessage);
    }
  });
}
"""


class SportyBetFixtureStatsResponse(BaseModel):
    """One captured Sportradar network response observed during widget loading."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: str = Field(description="Exact response URL.")
    path: str = Field(description="Parsed path from the response URL.")
    status: int = Field(ge=100, le=599, description="HTTP status code.")
    content_type: str | None = Field(
        default=None,
        description="Response content type when available.",
    )
    body_kind: Literal["json", "text"] = Field(description="Captured response body kind.")
    preview_text: str | None = Field(
        default=None,
        description="Compact response preview used for debugging and inspection.",
    )
    saved_path: str | None = Field(
        default=None,
        description="Optional local file path containing the captured body.",
    )

    @field_validator("url", "path")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("content_type", "preview_text", "saved_path")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text fields and collapse empties to `None`."""

        return normalize_optional_text(value)


class SportyBetFixtureStatsWidget(BaseModel):
    """Rendered SportyBet fixture widget snapshot."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    widget_key: str = Field(description="Short widget key used internally.")
    widget_type: str = Field(description="Sportradar widget type string.")
    status: Literal["mounted", "timeout", "error", "unavailable"] = Field(
        description="Mount outcome for the widget.",
    )
    headings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Rendered headings discovered inside the widget container.",
    )
    content_text: str | None = Field(
        default=None,
        description="Normalized rendered text inside the widget container.",
    )
    content_lines: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Deduplicated rendered text lines.",
    )
    iframe_urls: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Widget iframe URLs discovered in the rendered container tree.",
    )
    response_urls: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Network responses observed while this widget was mounting.",
    )
    error_message: str | None = Field(
        default=None,
        description="Optional failure detail for non-mounted widgets.",
    )
    saved_path: str | None = Field(
        default=None,
        description="Optional local file path containing the widget text dump.",
    )

    @field_validator("widget_key", "widget_type")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("content_text", "error_message", "saved_path")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text values and collapse empties to `None`."""

        return normalize_optional_text(value)


class SportyBetFixtureStatsResult(BaseModel):
    """Top-level SportyBet fixture stats result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fixture_url: str = Field(description="Requested public SportyBet fixture URL.")
    final_url: str = Field(description="Final page URL after navigation.")
    event_id: str = Field(description="Canonical `sr:match:*` event id.")
    match_id: str = Field(description="Numeric match id used by the widgets.")
    page_title: str | None = Field(default=None, description="Resolved page title.")
    fetched_at: datetime = Field(description="Timezone-aware fetch timestamp.")
    home_team_uid: str | None = Field(
        default=None,
        description="Sportradar unique team id for the home side when available.",
    )
    away_team_uid: str | None = Field(
        default=None,
        description="Sportradar unique team id for the away side when available.",
    )
    widget_loader_status: Literal["loaded", "failed"] = Field(
        description="Whether the widget loader executed successfully.",
    )
    team_stats: tuple[TeamStats, ...] = Field(
        default_factory=tuple,
        description="Structured team-stat snapshots derived from captured Sportradar feeds.",
    )
    widgets: tuple[SportyBetFixtureStatsWidget, ...] = Field(
        default_factory=tuple,
        description="Rendered widget snapshots collected for the fixture.",
    )
    responses: tuple[SportyBetFixtureStatsResponse, ...] = Field(
        default_factory=tuple,
        description="Captured Sportradar network responses observed during loading.",
    )

    @field_validator("fixture_url", "final_url", "event_id", "match_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        """Reject blank required strings."""

        field_name = getattr(info, "field_name", "value")
        return require_non_blank_text(value, field_name)

    @field_validator("page_title", "home_team_uid", "away_team_uid")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Trim optional text values."""

        return normalize_optional_text(value)

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at(cls, value: datetime) -> datetime:
        """Require timezone-aware timestamps."""

        if value.utcoffset() is None:
            raise ValueError("fetched_at must include timezone information.")
        return value


@dataclass(frozen=True, slots=True)
class _CapturedResponse:
    """Internal captured response payload before serialization."""

    summary: SportyBetFixtureStatsResponse
    payload: object


class SportyBetFixtureStatsScraper:
    """Fetch SportyBet fixture-page stats through the live browser context."""

    def __init__(
        self,
        *,
        user_agents: Sequence[str] = DEFAULT_USER_AGENTS,
        headless: bool = True,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        post_mount_wait_ms: int = DEFAULT_POST_MOUNT_WAIT_MS,
        response_timeout_ms: int = DEFAULT_RESPONSE_TIMEOUT_MS,
        response_body_timeout_ms: int = DEFAULT_RESPONSE_BODY_TIMEOUT_MS,
        settle_delay_ms: int = DEFAULT_SETTLE_DELAY_MS,
        widget_timeout_ms: int = DEFAULT_WIDGET_TIMEOUT_MS,
        widget_keys: Sequence[str] = DEFAULT_WIDGET_KEYS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Store deterministic runtime settings for the stats scraper."""

        if not user_agents:
            raise ValueError("user_agents must contain at least one value.")
        if navigation_timeout_ms <= 0:
            raise ValueError("navigation_timeout_ms must be positive.")
        if post_mount_wait_ms < 0:
            raise ValueError("post_mount_wait_ms must be zero or positive.")
        if response_timeout_ms <= 0:
            raise ValueError("response_timeout_ms must be positive.")
        if response_body_timeout_ms <= 0:
            raise ValueError("response_body_timeout_ms must be positive.")
        if settle_delay_ms < 0:
            raise ValueError("settle_delay_ms must be zero or positive.")
        if widget_timeout_ms <= 0:
            raise ValueError("widget_timeout_ms must be positive.")
        if not widget_keys:
            raise ValueError("widget_keys must contain at least one value.")

        normalized_widget_keys = tuple(
            require_non_blank_text(widget_key, "widget_key") for widget_key in widget_keys
        )
        unknown_widget_keys = [
            widget_key
            for widget_key in normalized_widget_keys
            if widget_key not in SPORTRADAR_WIDGET_TYPES
        ]
        if unknown_widget_keys:
            unknown_text = ", ".join(sorted(unknown_widget_keys))
            raise ValueError(f"Unsupported widget_keys: {unknown_text}.")

        self._user_agent = require_non_blank_text(user_agents[0], "user_agent")
        self._headless = headless
        self._navigation_timeout_ms = navigation_timeout_ms
        self._post_mount_wait_ms = post_mount_wait_ms
        self._response_timeout_ms = response_timeout_ms
        self._response_body_timeout_ms = response_body_timeout_ms
        self._settle_delay_ms = settle_delay_ms
        self._widget_timeout_ms = widget_timeout_ms
        self._widget_keys = normalized_widget_keys
        self._clock = clock or (lambda: datetime.now(UTC))

    async def fetch_fixture_stats(
        self,
        *,
        fixture_url: str,
        output_dir: Path | None = None,
    ) -> SportyBetFixtureStatsResult:
        """Fetch one SportyBet fixture's rendered widget stats."""

        normalized_url = require_non_blank_text(fixture_url, "fixture_url")
        event_id = extract_event_id_from_fixture_url(normalized_url)
        match_id = extract_match_id_from_event_id(event_id)

        async with async_playwright() as playwright:
            browser = await self._launch_browser(playwright)
            context = await browser.new_context(
                locale="en-NG",
                timezone_id=WAT_TIMEZONE_NAME,
                user_agent=self._user_agent,
            )
            loader_source = await self._fetch_widget_loader_source(
                context=context,
                fixture_url=normalized_url,
            )
            page = await context.new_page()
            captured_responses: list[_CapturedResponse] = []
            pending_capture_tasks: set[asyncio.Task[None]] = set()
            await self._install_sportradar_routes(
                context=context,
                page=page,
                fixture_url=normalized_url,
                loader_source=loader_source,
            )

            def schedule_capture(response: Response) -> None:
                """Queue one async response capture task from Playwright callbacks."""

                task = asyncio.create_task(
                    self._capture_response(response, captured_responses)
                )
                pending_capture_tasks.add(task)
                task.add_done_callback(pending_capture_tasks.discard)

            page.on("response", schedule_capture)

            try:
                await page.goto(
                    normalized_url,
                    wait_until="domcontentloaded",
                    timeout=float(self._navigation_timeout_ms),
                )
                with suppress(PlaywrightTimeoutError):
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=float(self._response_timeout_ms),
                    )
                await page.wait_for_timeout(float(self._settle_delay_ms))

                widget_loader_status = await self._ensure_widget_loader(
                    page=page,
                )
                await self._prepare_widget_runtime(page=page)

                widgets = await self._mount_widgets(
                    page=page,
                    match_id=match_id,
                    captured_responses=captured_responses,
                    pending_capture_tasks=pending_capture_tasks,
                    output_dir=output_dir,
                )

                if pending_capture_tasks:
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            asyncio.gather(*pending_capture_tasks, return_exceptions=True),
                            timeout=(self._response_body_timeout_ms / 1000) + 2,
                        )

                serialized_responses = self._serialize_captured_responses(
                    captured_responses,
                    output_dir=output_dir,
                )
                page_title = await page.title()

                fetched_at = self._clock().astimezone(UTC)
                match_info = _extract_sportradar_doc_data(
                    captured_responses,
                    "/match_info/",
                )
                team_uids = _extract_team_uids(match_info)

                result = SportyBetFixtureStatsResult(
                    fixture_url=normalized_url,
                    final_url=page.url,
                    event_id=event_id,
                    match_id=match_id,
                    page_title=page_title,
                    fetched_at=fetched_at,
                    home_team_uid=team_uids.get("home"),
                    away_team_uid=team_uids.get("away"),
                    widget_loader_status=widget_loader_status,
                    team_stats=_build_team_stats_from_captured_responses(
                        captured_responses,
                        fetched_at=fetched_at,
                    ),
                    widgets=tuple(widgets),
                    responses=tuple(serialized_responses),
                )
                if output_dir is not None:
                    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
                    summary_path = output_dir / "fixture-stats-summary.json"
                    await asyncio.to_thread(
                        summary_path.write_text,
                        json.dumps(
                            result.model_dump(mode="json"),
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=True,
                        ),
                        encoding="utf-8",
                    )
                return result
            finally:
                for task in tuple(pending_capture_tasks):
                    task.cancel()
                if pending_capture_tasks:
                    await asyncio.gather(*pending_capture_tasks, return_exceptions=True)
                await context.close()
                await browser.close()

    async def _launch_browser(self, playwright: Playwright) -> Browser:
        """Launch Chromium and surface installation errors clearly."""

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

    async def _fetch_widget_loader_source(
        self,
        *,
        context: BrowserContext,
        fixture_url: str,
    ) -> str:
        """Fetch the Sportradar widget loader through the live browser context."""

        response = await context.request.get(
            SPORTRADAR_WIDGET_LOADER_URL,
            headers={
                "Accept": "*/*",
                "Referer": fixture_url,
                "Origin": SPORTYBET_BASE_URL,
                "User-Agent": self._user_agent,
            },
        )
        if not response.ok:
            status = response.status
            body = await response.text()
            raise RuntimeError(
                "Could not fetch the Sportradar widget loader through the live browser "
                f"context (status={status}): {body}"
            )
        return await response.text()

    async def _install_sportradar_routes(
        self,
        *,
        context: BrowserContext,
        page: Page,
        fixture_url: str,
        loader_source: str,
    ) -> None:
        """Proxy Sportradar widget and fishnet hosts through the browser context."""

        async def proxy_sportradar_request(route: Route) -> None:
            request_url = route.request.url
            if request_url == SPORTRADAR_WIDGET_LOADER_URL:
                await route.fulfill(
                    status=200,
                    headers={
                        "access-control-allow-origin": SPORTYBET_BASE_URL,
                        "access-control-allow-credentials": "true",
                        "cache-control": "no-store",
                        "content-type": "application/javascript",
                        "vary": "Origin",
                    },
                    body=loader_source,
                )
                return

            try:
                upstream_response = await context.request.get(
                    request_url,
                    headers={
                        "Accept": "*/*",
                        "Referer": fixture_url,
                        "User-Agent": self._user_agent,
                    },
                )
                response_body = await upstream_response.body()
            except PlaywrightError as exc:
                await route.fulfill(
                    status=502,
                    headers={
                        "access-control-allow-origin": SPORTYBET_BASE_URL,
                        "access-control-allow-credentials": "true",
                        "content-type": "text/plain; charset=utf-8",
                        "vary": "Origin",
                    },
                    body=f"Could not proxy Sportradar request: {exc}",
                )
                return

            await route.fulfill(
                status=upstream_response.status,
                headers=_build_sportradar_proxy_response_headers(upstream_response.headers),
                body=response_body,
            )

        await page.route(SPORTRADAR_WIDGET_ROUTE_GLOB, proxy_sportradar_request)
        await page.route(SPORTRADAR_FISHNET_ROUTE_GLOB, proxy_sportradar_request)

    async def _ensure_widget_loader(
        self,
        *,
        page: Page,
    ) -> Literal["loaded", "failed"]:
        """Ensure the routed Sportradar widget loader is installed in the page."""

        loader_type = await page.evaluate(
            INSTALL_WIDGETLOADER_SCRIPT,
            SPORTRADAR_WIDGET_LOADER_URL,
        )
        if loader_type != "function":
            raise RuntimeError(
                "Sportradar widget loader executed, but window.SIR is still unavailable."
            )
        return "loaded"

    async def _prepare_widget_runtime(
        self,
        *,
        page: Page,
    ) -> None:
        """Set the live Sportradar runtime language before mounting widgets."""

        runtime_ready = await page.evaluate(
            """() => {
                if (typeof window.SIR !== "function") {
                    return false;
                }
                window.SIR("changeLanguage", "en");
                return true;
            }"""
        )
        if runtime_ready is not True:
            raise RuntimeError(
                "Sportradar widget runtime is unavailable after loader installation."
            )

    async def _mount_widgets(
        self,
        *,
        page: Page,
        match_id: str,
        captured_responses: list[_CapturedResponse],
        pending_capture_tasks: set[asyncio.Task[None]],
        output_dir: Path | None,
    ) -> list[SportyBetFixtureStatsWidget]:
        """Mount the configured widgets one by one and capture their text snapshots."""

        widgets: list[SportyBetFixtureStatsWidget] = []

        for widget_index, widget_key in enumerate(self._widget_keys, start=1):
            widget_type = SPORTRADAR_WIDGET_TYPES[widget_key]
            starting_response_count = len(captured_responses)
            try:
                widget_payload = await asyncio.wait_for(
                    page.evaluate(
                        MOUNT_WIDGET_SCRIPT,
                        {
                            "widgetKey": widget_key,
                            "widgetType": widget_type,
                            "containerId": f"sportybet_fixture_widget_{widget_index}",
                            "selector": f"#sportybet_fixture_widget_{widget_index}",
                            "widgetTimeoutMs": self._widget_timeout_ms,
                            "settleDelayMs": self._settle_delay_ms,
                            "postMountWaitMs": self._post_mount_wait_ms,
                            "minHeightPx": 640,
                            "props": {
                                "matchId": match_id,
                                "layout": "double",
                                "activeSwitcher": "scoreDetails",
                                "tabsPosition": "top",
                                "adsFrequency": False,
                                "logoLink": "",
                                "pitchCustomBgColor": "#0E8E36",
                                "goalBannerCustomBgColor": "#E41827",
                            },
                        },
                    ),
                    timeout=(
                        (
                            self._widget_timeout_ms
                            + self._settle_delay_ms
                            + self._post_mount_wait_ms
                        )
                        / 1000
                    )
                    + 5,
                )
            except TimeoutError:
                widget_payload = {
                    "widget_key": widget_key,
                    "widget_type": widget_type,
                    "status": "timeout",
                    "error_message": "Widget evaluation exceeded the allowed timeout.",
                    "headings": [],
                    "content_text": "",
                    "content_lines": [],
                    "iframe_urls": [],
                    "html": "",
                }
            await page.wait_for_timeout(float(self._settle_delay_ms))
            if pending_capture_tasks:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(*tuple(pending_capture_tasks), return_exceptions=True),
                        timeout=(self._response_body_timeout_ms / 1000) + 2,
                    )

            widget_responses = tuple(captured_responses[starting_response_count:])
            if isinstance(widget_payload, Mapping):
                widget_payload = self._augment_widget_payload(
                    widget_key=widget_key,
                    widget_payload=widget_payload,
                    widget_responses=widget_responses,
                    all_responses=tuple(captured_responses),
                )

            raw_html = ""
            if isinstance(widget_payload, Mapping):
                raw_html_value = widget_payload.get("html")
                if isinstance(raw_html_value, str):
                    raw_html = raw_html_value
            response_urls = tuple(
                captured.summary.url
                for captured in widget_responses
            )
            if widget_key == "teamInfo":
                response_urls = _merge_text_sequences(
                    response_urls,
                    _extract_team_info_response_urls(captured_responses),
                )
            saved_path: str | None = None
            if output_dir is not None:
                await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
                saved_path = self._write_widget_dump(
                    output_dir=output_dir,
                    widget_index=widget_index,
                    widget_key=widget_key,
                    payload=widget_payload,
                    raw_html=raw_html,
                )

            if not isinstance(widget_payload, Mapping):
                widgets.append(
                    SportyBetFixtureStatsWidget(
                        widget_key=widget_key,
                        widget_type=widget_type,
                        status="error",
                        error_message="Widget mount returned an invalid payload.",
                        response_urls=response_urls,
                        saved_path=saved_path,
                    )
                )
                continue

            headings = _normalize_text_sequence(widget_payload.get("headings"))
            content_text = _normalize_optional_text_value(widget_payload.get("content_text"))
            content_lines = _normalize_text_sequence(widget_payload.get("content_lines"))
            iframe_urls = _normalize_text_sequence(widget_payload.get("iframe_urls"))
            error_message = _normalize_optional_text_value(widget_payload.get("error_message"))
            status = widget_payload.get("status")
            if not isinstance(status, str):
                status = "error"

            widgets.append(
                SportyBetFixtureStatsWidget(
                    widget_key=widget_key,
                    widget_type=widget_type,
                    status=status,
                    headings=headings,
                    content_text=content_text,
                    content_lines=content_lines,
                    iframe_urls=iframe_urls,
                    response_urls=response_urls,
                    error_message=error_message,
                    saved_path=saved_path,
                )
            )

        return widgets

    def _augment_widget_payload(
        self,
        *,
        widget_key: str,
        widget_payload: Mapping[str, object],
        widget_responses: Sequence[_CapturedResponse],
        all_responses: Sequence[_CapturedResponse],
    ) -> dict[str, object]:
        """Enrich widget output with derived stats when the DOM is sparse."""

        payload = dict(widget_payload)
        if widget_key == "teamInfo":
            derived_lines = _build_team_info_lines(all_responses)
            if not derived_lines:
                return payload

            payload["content_lines"] = list(derived_lines)
            payload["content_text"] = "\n".join(derived_lines)
            payload["headings"] = list(
                _merge_text_sequences(
                    payload.get("headings"),
                    ("Team Info",),
                )
            )
            payload["status"] = "mounted"
            payload["error_message"] = None
            return payload

        if widget_key != "statistics":
            return payload

        existing_lines = _normalize_text_sequence(payload.get("content_lines"))
        existing_text = _normalize_optional_text_value(payload.get("content_text"))
        if existing_lines or existing_text:
            return payload

        derived_lines = _build_statistics_fallback_lines(widget_responses)
        if not derived_lines:
            return payload

        payload["content_lines"] = list(derived_lines)
        payload["content_text"] = "\n".join(derived_lines)
        payload["headings"] = list(_merge_text_sequences(payload.get("headings"), ("Statistics",)))
        return payload

    def _write_widget_dump(
        self,
        *,
        output_dir: Path,
        widget_index: int,
        widget_key: str,
        payload: Mapping[str, object],
        raw_html: str,
    ) -> str:
        """Persist one widget snapshot to disk for operator inspection."""

        file_path = output_dir / f"{widget_index:02d}-{widget_key}.md"
        content_text = _normalize_optional_text_value(payload.get("content_text")) or ""
        content_lines = _normalize_text_sequence(payload.get("content_lines"))
        headings = _normalize_text_sequence(payload.get("headings"))
        iframe_urls = _normalize_text_sequence(payload.get("iframe_urls"))
        lines: list[str] = [f"# {widget_key}", ""]
        if headings:
            lines.append("Headings:")
            for heading in headings:
                lines.append(f"- {heading}")
            lines.append("")
        if iframe_urls:
            lines.append("Iframe URLs:")
            for iframe_url in iframe_urls:
                lines.append(f"- {iframe_url}")
            lines.append("")
        lines.append("Content:")
        if content_lines:
            for content_line in content_lines:
                lines.append(f"- {content_line}")
        elif content_text:
            lines.append(content_text)
        else:
            lines.append("- <empty>")
        if raw_html:
            lines.extend(
                [
                    "",
                    "HTML Preview:",
                    "",
                    "```html",
                    raw_html[:MAX_WIDGET_TEXT_CHARS],
                    "```",
                ]
            )
        file_path.write_text("\n".join(lines), encoding="utf-8")
        return str(file_path)

    async def _capture_response(
        self,
        response: Response,
        captured_responses: list[_CapturedResponse],
    ) -> None:
        """Capture one Sportradar response body for later inspection."""

        parsed_url = urlparse(response.url)
        netloc = parsed_url.netloc.casefold()
        if "sportradar" not in netloc:
            return

        content_type = normalize_optional_text(response.headers.get("content-type"))
        content_type_key = content_type.casefold() if content_type is not None else ""
        body_kind: Literal["json", "text"] = "json" if "json" in content_type_key else "text"
        try:
            if body_kind == "json":
                payload = await asyncio.wait_for(
                    response.json(),
                    timeout=self._response_body_timeout_ms / 1000,
                )
                preview_text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
            elif _is_probably_binary_content_type(content_type):
                preview_text = (
                    "Binary response omitted from preview"
                    if content_type is None
                    else f"Binary response omitted from preview ({content_type})"
                )
                payload = preview_text
            else:
                payload = await asyncio.wait_for(
                    response.text(),
                    timeout=self._response_body_timeout_ms / 1000,
                )
                preview_text = str(payload)
        except TimeoutError:
            preview_text = "Response body capture timed out."
            payload = (
                {"capture_error": preview_text}
                if body_kind == "json"
                else preview_text
            )
        except UnicodeDecodeError:
            preview_text = (
                "Binary response omitted from preview"
                if content_type is None
                else f"Binary response omitted from preview ({content_type})"
            )
            payload = preview_text
        except PlaywrightError:
            return

        normalized_preview = normalize_optional_text(
            preview_text[:MAX_RESPONSE_PREVIEW_CHARS]
        )
        captured_responses.append(
            _CapturedResponse(
                summary=SportyBetFixtureStatsResponse(
                    url=response.url,
                    path=parsed_url.path,
                    status=response.status,
                    content_type=content_type,
                    body_kind=body_kind,
                    preview_text=normalized_preview,
                ),
                payload=payload,
            )
        )

    def _serialize_captured_responses(
        self,
        captured_responses: Sequence[_CapturedResponse],
        *,
        output_dir: Path | None,
    ) -> list[SportyBetFixtureStatsResponse]:
        """Deduplicate captured responses and optionally persist them to disk."""

        deduped_by_url: dict[str, _CapturedResponse] = {}
        for captured in captured_responses:
            deduped_by_url[captured.summary.url] = captured

        ordered = list(deduped_by_url.values())
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        serialized: list[SportyBetFixtureStatsResponse] = []
        for index, captured in enumerate(ordered, start=1):
            saved_path: str | None = None
            if output_dir is not None:
                file_path = output_dir / _build_response_file_name(
                    index=index,
                    response_url=captured.summary.url,
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
                captured.summary.model_copy(update={"saved_path": saved_path})
            )
        return serialized


def extract_event_id_from_fixture_url(fixture_url: str) -> str:
    """Extract the `sr:match:*` event id from one SportyBet fixture URL."""

    normalized_url = require_non_blank_text(fixture_url, "fixture_url")
    match = SPORTRADAR_EVENT_ID_PATTERN.search(normalized_url)
    if match is None:
        raise ValueError("fixture_url must contain a valid `sr:match:<id>` segment.")
    return match.group(1)


def extract_match_id_from_event_id(event_id: str) -> str:
    """Extract the numeric match id used by Sportradar widgets."""

    normalized_event_id = require_non_blank_text(event_id, "event_id")
    match = SPORTRADAR_EVENT_ID_PATTERN.search(normalized_event_id)
    if match is None:
        raise ValueError("event_id must contain a valid `sr:match:<id>` segment.")
    return match.group(1).split(":")[-1]


def render_fixture_stats_markdown(result: SportyBetFixtureStatsResult) -> str:
    """Render one fixture stats result as a readable markdown report."""

    lines = [
        "# SportyBet Fixture Stats",
        "",
        f"- Fixture URL: {result.fixture_url}",
        f"- Final URL: {result.final_url}",
        f"- Event ID: {result.event_id}",
        f"- Match ID: {result.match_id}",
        f"- Page Title: {result.page_title or '<none>'}",
        f"- Widget Loader: {result.widget_loader_status}",
        f"- Fetched At: {result.fetched_at.isoformat()}",
        f"- Widget Count: {len(result.widgets)}",
        f"- Response Count: {len(result.responses)}",
    ]

    for widget in result.widgets:
        lines.extend(
            [
                "",
                f"## {widget.widget_key}",
                "",
                f"- Widget Type: {widget.widget_type}",
                f"- Status: {widget.status}",
                f"- Saved Path: {widget.saved_path or '<none>'}",
                f"- Iframe URLs: {len(widget.iframe_urls)}",
                f"- Response URLs: {len(widget.response_urls)}",
            ]
        )
        if widget.error_message is not None:
            lines.append(f"- Error: {widget.error_message}")
        if widget.headings:
            lines.append("- Headings:")
            for heading in widget.headings:
                lines.append(f"  - {heading}")
        lines.append("- Content:")
        if widget.content_lines:
            for content_line in widget.content_lines:
                lines.append(f"  - {content_line}")
        elif widget.content_text is not None:
            lines.append(f"  - {widget.content_text}")
        else:
            lines.append("  - <empty>")
        if widget.iframe_urls:
            lines.append("- Iframe URLs:")
            for iframe_url in widget.iframe_urls:
                lines.append(f"  - {iframe_url}")
        if widget.response_urls:
            lines.append("- Response URLs:")
            for response_url in widget.response_urls:
                lines.append(f"  - {response_url}")

    if result.responses:
        lines.extend(["", "## Network Responses", ""])
        for response in result.responses:
            lines.append(f"- {response.status} {response.url}")
            if response.content_type is not None:
                lines.append(f"  - Content-Type: {response.content_type}")
            if response.saved_path is not None:
                lines.append(f"  - Saved Path: {response.saved_path}")
            if response.preview_text is not None:
                lines.append(f"  - Preview: {response.preview_text[:240]}")

    return "\n".join(lines)


def build_fixture_details_snapshot(
    result: SportyBetFixtureStatsResult,
    *,
    fixture_ref: str | None = None,
) -> FixtureDetails:
    """Convert a raw stats fetch into the compact pipeline detail snapshot."""

    resolved_fixture_ref = normalize_optional_text(fixture_ref) or result.event_id
    return FixtureDetails(
        fixture_ref=resolved_fixture_ref,
        source_provider="sportybet",
        fixture_url=result.fixture_url,
        event_id=result.event_id,
        match_id=result.match_id,
        fetched_at=result.fetched_at,
        widget_loader_status=result.widget_loader_status,
        sections=tuple(
            FixtureDetailSection(
                widget_key=widget.widget_key,
                widget_type=widget.widget_type,
                status=widget.status,
                headings=widget.headings,
                content_lines=widget.content_lines,
                response_urls=widget.response_urls,
                error_message=widget.error_message,
            )
            for widget in result.widgets
        ),
    )


def _normalize_optional_text_value(value: object) -> str | None:
    """Normalize one optional text-like value."""

    if not isinstance(value, str):
        return None
    return normalize_optional_text(value)


def _coerce_int_value(value: object) -> int | None:
    """Convert one scalar payload value into an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        normalized = normalize_optional_text(value)
        if normalized is None:
            return None
        try:
            return int(float(normalized))
        except ValueError:
            return None
    return None


def _coerce_float_value(value: object) -> float | None:
    """Convert one scalar payload value into a float when possible."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = normalize_optional_text(value)
        if normalized is None:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def _normalize_text_sequence(value: object) -> tuple[str, ...]:
    """Normalize one list-like sequence of text values."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized_item = normalize_optional_text(item)
        if normalized_item is None:
            continue
        lookup_key = normalized_item.casefold()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        normalized.append(normalized_item)
    return tuple(normalized)


def _merge_text_sequences(*values: object) -> tuple[str, ...]:
    """Merge multiple text sequences into one normalized tuple without duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _normalize_text_sequence(value):
            lookup_key = item.casefold()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            merged.append(item)
    return tuple(merged)


def _build_team_info_lines(
    captured_responses: Sequence[_CapturedResponse],
) -> tuple[str, ...]:
    """Build readable home/away team info from Sportradar team-scoped feeds."""

    match_info = _extract_sportradar_doc_data(captured_responses, "/match_info/")
    team_names = _extract_team_names(match_info)
    team_uids = _extract_team_uids(match_info)
    team_info_by_uid = _extract_team_info_payloads(captured_responses)

    if not team_uids or not team_info_by_uid:
        return ()

    lines: list[str] = []
    for side, side_label in (("home", "Home"), ("away", "Away")):
        team_uid = team_uids.get(side)
        if team_uid is None:
            continue
        team_info = team_info_by_uid.get(team_uid)
        if team_info is None:
            continue
        lines.extend(
            _build_single_team_info_lines(
                side_label=side_label,
                team_info=team_info,
                fallback_name=team_names[side],
            )
        )

    return _merge_text_sequences(lines)


def _build_team_stats_from_captured_responses(
    captured_responses: Sequence[_CapturedResponse],
    *,
    fetched_at: datetime,
) -> tuple[TeamStats, ...]:
    """Derive canonical team-stat snapshots from SportyBet fixture responses."""

    match_info = _extract_sportradar_doc_data(captured_responses, "/match_info/")
    if match_info is None:
        return ()

    competition = _extract_match_competition_name(match_info)
    season = _extract_match_season_label(match_info)
    team_names = _extract_team_names(match_info)
    team_uids = _extract_team_uids(match_info)

    team_stats: list[TeamStats] = []
    for side in ("home", "away"):
        team_uid = team_uids.get(side)
        if team_uid is None:
            continue
        season_row = _extract_season_table_row(captured_responses, team_uid)
        form_row = _extract_form_table_row(captured_responses, team_uid)
        team_name = _extract_team_name_for_side(
            side=side,
            fallback_name=team_names[side],
            season_row=season_row,
        )
        team_stats.append(
            _build_team_stats_snapshot(
                team_uid=team_uid,
                team_name=team_name,
                competition=competition,
                season=season,
                fetched_at=fetched_at,
                season_row=season_row,
                form_row=form_row,
            )
        )

    return tuple(team_stats)


def _build_team_stats_snapshot(
    *,
    team_uid: str,
    team_name: str,
    competition: str | None,
    season: str | None,
    fetched_at: datetime,
    season_row: Mapping[str, object] | None,
    form_row: Mapping[str, object] | None,
) -> TeamStats:
    """Build one canonical `TeamStats` row from SportyBet table feeds."""

    matches_played = (
        _coerce_int_value(season_row.get("total")) if season_row is not None else None
    )
    wins = _coerce_int_value(season_row.get("winTotal")) if season_row is not None else None
    draws = _coerce_int_value(season_row.get("drawTotal")) if season_row is not None else None
    losses = _coerce_int_value(season_row.get("lossTotal")) if season_row is not None else None
    goals_for = (
        _coerce_int_value(season_row.get("goalsForTotal"))
        if season_row is not None
        else None
    )
    goals_against = (
        _coerce_int_value(season_row.get("goalsAgainstTotal"))
        if season_row is not None
        else None
    )
    points = _coerce_int_value(season_row.get("pointsTotal")) if season_row is not None else None
    position = (
        _coerce_int_value(season_row.get("pos"))
        if season_row is not None
        else None
    )
    home_wins = _coerce_int_value(season_row.get("winHome")) if season_row is not None else None
    away_wins = _coerce_int_value(season_row.get("winAway")) if season_row is not None else None

    if form_row is not None:
        matches_played = matches_played if matches_played is not None else _split_metric_value(
            form_row,
            "played",
            "total",
        )
        wins = wins if wins is not None else _split_metric_value(form_row, "win", "total")
        draws = draws if draws is not None else _split_metric_value(form_row, "draw", "total")
        losses = losses if losses is not None else _split_metric_value(form_row, "loss", "total")
        goals_for = goals_for if goals_for is not None else _split_metric_value(
            form_row,
            "goalsfor",
            "total",
        )
        goals_against = goals_against if goals_against is not None else _split_metric_value(
            form_row,
            "goalsagainst",
            "total",
        )
        points = points if points is not None else _split_metric_value(form_row, "points", "total")
        position = position if position is not None else _split_metric_value(
            form_row,
            "position",
            "total",
        )
        home_wins = home_wins if home_wins is not None else _split_metric_value(
            form_row,
            "win",
            "home",
        )
        away_wins = away_wins if away_wins is not None else _split_metric_value(
            form_row,
            "win",
            "away",
        )

    resolved_matches_played = max(matches_played or 0, 0)
    resolved_goals_for = max(goals_for or 0, 0)
    resolved_goals_against = max(goals_against or 0, 0)
    avg_goals_scored = (
        resolved_goals_for / resolved_matches_played if resolved_matches_played > 0 else None
    )
    avg_goals_conceded = (
        resolved_goals_against / resolved_matches_played if resolved_matches_played > 0 else None
    )

    return TeamStats(
        team_id=team_uid,
        team_name=team_name,
        sport=SportName.SOCCER,
        source_provider="sportybet_fixture_stats",
        fetched_at=fetched_at,
        competition=competition,
        season=season,
        matches_played=resolved_matches_played,
        wins=max(wins or 0, 0),
        draws=max(draws or 0, 0),
        losses=max(losses or 0, 0),
        goals_for=resolved_goals_for,
        goals_against=resolved_goals_against,
        clean_sheets=0,
        form=_extract_form_string(form_row),
        position=position,
        points=points,
        home_wins=max(home_wins or 0, 0),
        away_wins=max(away_wins or 0, 0),
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
    )


def _extract_team_name_for_side(
    *,
    side: str,
    fallback_name: str,
    season_row: Mapping[str, object] | None,
) -> str:
    """Resolve the best available team label for one fixture side."""

    if season_row is None:
        return fallback_name
    team = season_row.get("team")
    if not isinstance(team, Mapping):
        return fallback_name
    return (
        _first_text_value(team, ("name", "mediumname", "abbr"))
        or fallback_name
    )


def _extract_match_competition_name(match_info: Mapping[str, object]) -> str | None:
    """Resolve the competition name from a SportyBet match-info payload."""

    tournament = match_info.get("tournament")
    if isinstance(tournament, Mapping):
        tournament_name = _first_text_value(tournament, ("name",))
        if tournament_name is not None:
            return tournament_name

    season = match_info.get("season")
    if isinstance(season, Mapping):
        return _first_text_value(season, ("name", "abbr"))
    return None


def _extract_match_season_label(match_info: Mapping[str, object]) -> str | None:
    """Resolve the compact season label from a SportyBet match-info payload."""

    season = match_info.get("season")
    if not isinstance(season, Mapping):
        return None
    return _first_text_value(season, ("year", "name", "abbr"))


def _extract_season_table_row(
    captured_responses: Sequence[_CapturedResponse],
    team_uid: str,
) -> Mapping[str, object] | None:
    """Locate one season table row for the requested team uid."""

    for path_fragment in ("/season_dynamictable/", "/stats_season_tables/"):
        data = _extract_sportradar_doc_data(captured_responses, path_fragment)
        if data is None:
            continue
        row = _find_table_row_for_team_uid(data, team_uid)
        if row is not None:
            return row
    return None


def _find_table_row_for_team_uid(
    value: object,
    team_uid: str,
) -> Mapping[str, object] | None:
    """Recursively locate a `tablerow` payload that belongs to one team uid."""

    if isinstance(value, Mapping):
        team = value.get("team")
        if isinstance(team, Mapping):
            row_team_uid = _normalize_scalar_text(team.get("uid"))
            if value.get("_doc") == "tablerow" and row_team_uid == team_uid:
                return value

        for child in value.values():
            found = _find_table_row_for_team_uid(child, team_uid)
            if found is not None:
                return found
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_table_row_for_team_uid(child, team_uid)
            if found is not None:
                return found
    return None


def _extract_form_table_row(
    captured_responses: Sequence[_CapturedResponse],
    team_uid: str,
) -> Mapping[str, object] | None:
    """Locate one `stats_formtable` row for the requested team uid."""

    data = _extract_sportradar_doc_data(captured_responses, "/stats_formtable/")
    if data is None:
        return None
    teams = data.get("teams")
    if not isinstance(teams, Sequence) or isinstance(teams, (str, bytes, bytearray)):
        return None
    for row in teams:
        if not isinstance(row, Mapping):
            continue
        team = row.get("team")
        if not isinstance(team, Mapping):
            continue
        row_team_uid = _normalize_scalar_text(team.get("uid"))
        if row_team_uid == team_uid:
            return row
    return None


def _split_metric_value(
    row: Mapping[str, object],
    metric_key: str,
    split_key: str,
) -> int | None:
    """Return one integer metric from a split-stat row such as `win.total`."""

    metric = row.get(metric_key)
    if not isinstance(metric, Mapping):
        return None
    return _coerce_int_value(metric.get(split_key))


def _extract_form_string(form_row: Mapping[str, object] | None) -> str | None:
    """Convert SportyBet form entries into an oldest-to-newest compact string."""

    if form_row is None:
        return None
    form = form_row.get("form")
    if not isinstance(form, Mapping):
        return None
    total_entries = form.get("total")
    if not isinstance(total_entries, Sequence) or isinstance(
        total_entries,
        (str, bytes, bytearray),
    ):
        return None

    markers: list[str] = []
    for entry in reversed(total_entries):
        if not isinstance(entry, Mapping):
            continue
        marker = _normalize_scalar_text(entry.get("typeid")) or _normalize_scalar_text(
            entry.get("value")
        )
        if marker is None:
            continue
        marker = marker.upper()
        if marker not in {"W", "D", "L"}:
            continue
        markers.append(marker)

    compact = "".join(markers)
    return compact or None


def _extract_team_info_response_urls(
    captured_responses: Sequence[_CapturedResponse],
) -> tuple[str, ...]:
    """Return captured `stats_team_info` response URLs in source order."""

    return tuple(
        captured.summary.url
        for captured in captured_responses
        if SPORTRADAR_TEAM_INFO_PATH_PATTERN.search(captured.summary.path)
    )


def _extract_team_info_payloads(
    captured_responses: Sequence[_CapturedResponse],
) -> dict[str, Mapping[str, object]]:
    """Map Sportradar team uid to its `stats_team_info` payload."""

    team_info_by_uid: dict[str, Mapping[str, object]] = {}
    for captured in captured_responses:
        path_match = SPORTRADAR_TEAM_INFO_PATH_PATTERN.search(captured.summary.path)
        if path_match is None:
            continue
        data = _extract_sportradar_payload_data(captured.payload)
        if data is None:
            continue
        team_info_by_uid[path_match.group("team_uid")] = data
    return team_info_by_uid


def _extract_team_uids(match_info: Mapping[str, object] | None) -> Mapping[str, str]:
    """Resolve home and away unique team ids from match info."""

    if match_info is None:
        return {}

    match = match_info.get("match")
    if not isinstance(match, Mapping):
        return {}

    teams = match.get("teams")
    if not isinstance(teams, Mapping):
        return {}

    team_uids: dict[str, str] = {}
    for side in ("home", "away"):
        team = teams.get(side)
        if not isinstance(team, Mapping):
            continue
        team_uid = _normalize_scalar_text(team.get("uid"))
        if team_uid is not None:
            team_uids[side] = team_uid
    return team_uids


def _build_single_team_info_lines(
    *,
    side_label: str,
    team_info: Mapping[str, object],
    fallback_name: str,
) -> tuple[str, ...]:
    """Format one side's team profile from `stats_team_info` data."""

    team = team_info.get("team")
    team_mapping = team if isinstance(team, Mapping) else {}
    team_name = _first_text_value(team_mapping, ("name", "mediumname", "abbr")) or fallback_name
    team_uid = _normalize_scalar_text(team_mapping.get("_id"))
    team_abbr = _first_text_value(team_mapping, ("abbr",))
    founded = _normalize_scalar_text(team_mapping.get("founded"))
    nickname = _normalize_scalar_text(team_mapping.get("nickname"))
    website = _normalize_scalar_text(team_mapping.get("website"))

    lines = [
        _join_line_parts(
            f"{side_label} team: {team_name}",
            f"abbr {team_abbr}" if team_abbr is not None else None,
            f"uid {team_uid}" if team_uid is not None else None,
            f"founded {founded}" if founded is not None else None,
            f"nickname {nickname}" if nickname is not None else None,
            f"website {website}" if website is not None else None,
        )
    ]

    manager_line = _build_manager_line(side_label=side_label, team_info=team_info)
    if manager_line is not None:
        lines.append(manager_line)

    stadium_line = _build_team_stadium_line(side_label=side_label, team_info=team_info)
    if stadium_line is not None:
        lines.append(stadium_line)

    tournaments_line = _build_team_tournaments_line(
        side_label=side_label,
        team_info=team_info,
    )
    if tournaments_line is not None:
        lines.append(tournaments_line)

    jersey_line = _build_team_jersey_line(side_label=side_label, team_info=team_info)
    if jersey_line is not None:
        lines.append(jersey_line)

    return tuple(line for line in lines if line)


def _build_manager_line(
    *,
    side_label: str,
    team_info: Mapping[str, object],
) -> str | None:
    """Format manager data from team info when present."""

    manager = team_info.get("manager")
    if not isinstance(manager, Mapping):
        return None

    manager_name = _first_text_value(manager, ("name",))
    if manager_name is None:
        return None

    nationality = manager.get("nationality")
    nationality_name = (
        _first_text_value(nationality, ("name", "ioc", "a3"))
        if isinstance(nationality, Mapping)
        else None
    )
    member_since = _extract_sportradar_date_label(manager.get("membersince"))
    return _join_line_parts(
        f"{side_label} manager: {manager_name}",
        nationality_name,
        f"member since {member_since}" if member_since is not None else None,
    )


def _build_team_stadium_line(
    *,
    side_label: str,
    team_info: Mapping[str, object],
) -> str | None:
    """Format stadium data from team info when present."""

    stadium = team_info.get("stadium")
    if not isinstance(stadium, Mapping):
        return None

    stadium_name = _first_text_value(stadium, ("name",))
    if stadium_name is None:
        return None

    city = _first_text_value(stadium, ("city", "state"))
    country = _first_text_value(stadium, ("country",))
    capacity = _normalize_scalar_text(stadium.get("capacity"))
    return _join_line_parts(
        f"{side_label} stadium: {stadium_name}",
        city,
        country,
        f"capacity {capacity}" if capacity is not None else None,
    )


def _build_team_tournaments_line(
    *,
    side_label: str,
    team_info: Mapping[str, object],
) -> str | None:
    """Format the competitions attached to one team profile."""

    tournaments = team_info.get("tournaments")
    if not isinstance(tournaments, Sequence) or isinstance(
        tournaments,
        (str, bytes, bytearray),
    ):
        return None

    labels: list[str] = []
    for tournament in tournaments:
        if not isinstance(tournament, Mapping):
            continue
        tournament_name = _first_text_value(tournament, ("name", "abbr"))
        if tournament_name is None:
            continue
        season = _first_text_value(tournament, ("year",))
        season_type = _first_text_value(tournament, ("seasontypename",))
        label_parts = tuple(
            part
            for part in (tournament_name, season, season_type)
            if part is not None
        )
        labels.append(" ".join(label_parts))

    if not labels:
        return None
    return f"{side_label} competitions: {'; '.join(labels[:4])}"


def _build_team_jersey_line(
    *,
    side_label: str,
    team_info: Mapping[str, object],
) -> str | None:
    """Format kit color hints from team info when present."""

    jersey_labels: list[str] = []
    for source_key, display_label in (
        ("homejersey", "home"),
        ("awayjersey", "away"),
        ("gkjersey", "goalkeeper"),
    ):
        jersey = team_info.get(source_key)
        if not isinstance(jersey, Mapping):
            continue
        base_color = _first_text_value(jersey, ("base",))
        if base_color is None:
            continue
        jersey_labels.append(f"{display_label} #{base_color}")

    if not jersey_labels:
        return None
    return f"{side_label} kits: {', '.join(jersey_labels)}"


def _build_statistics_fallback_lines(
    captured_responses: Sequence[_CapturedResponse],
) -> tuple[str, ...]:
    """Build readable statistics lines from Sportradar JSON responses."""

    match_info = _extract_sportradar_doc_data(captured_responses, "/match_info/")
    match_details = _extract_sportradar_doc_data(captured_responses, "/match_details/")
    match_timeline = _extract_sportradar_doc_data(captured_responses, "/match_timeline/")

    if match_info is None and match_details is None and match_timeline is None:
        return ()

    team_names = _extract_team_names(match_info)
    lines: list[str] = []

    result_line = _build_match_result_line(match_info, team_names)
    if result_line is not None:
        lines.append(result_line)

    for stat_key in (
        "110",
        "1126",
        "1029",
        "1030",
        "goalattempts",
        "125",
        "126",
        "171",
        "124",
        "123",
        "129",
        "120",
        "121",
        "122",
        "127",
        "158",
        "40",
        "60",
    ):
        line = _build_match_stat_line(
            stat_key=stat_key,
            match_details=match_details,
            team_names=team_names,
        )
        if line is not None:
            lines.append(line)

    cards_lines = _build_match_cards_lines(match_info, team_names)
    lines.extend(cards_lines)

    goals_timeline_line = _build_goals_timeline_line(match_timeline, team_names)
    if goals_timeline_line is not None:
        lines.append(goals_timeline_line)

    return _merge_text_sequences(lines)


def _extract_sportradar_doc_data(
    captured_responses: Sequence[_CapturedResponse],
    path_fragment: str,
) -> Mapping[str, object] | None:
    """Return the first Sportradar `doc[0].data` payload matching one path fragment."""

    for captured in captured_responses:
        if path_fragment not in captured.summary.path:
            continue
        payload = captured.payload
        if not isinstance(payload, Mapping):
            continue
        docs = payload.get("doc")
        if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes, bytearray)):
            continue
        if not docs:
            continue
        first_doc = docs[0]
        if not isinstance(first_doc, Mapping):
            continue
        data = first_doc.get("data")
        if isinstance(data, Mapping):
            return data
    return None


def _extract_sportradar_payload_data(payload: object) -> Mapping[str, object] | None:
    """Return `doc[0].data` from one captured Sportradar payload."""

    if not isinstance(payload, Mapping):
        return None
    docs = payload.get("doc")
    if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes, bytearray)):
        return None
    if not docs:
        return None
    first_doc = docs[0]
    if not isinstance(first_doc, Mapping):
        return None
    data = first_doc.get("data")
    return data if isinstance(data, Mapping) else None


def _first_text_value(
    values: Mapping[str, object],
    keys: Sequence[str],
) -> str | None:
    """Return the first non-empty scalar text value from a mapping."""

    for key in keys:
        value = _normalize_scalar_text(values.get(key))
        if value is not None:
            return value
    return None


def _normalize_scalar_text(value: object) -> str | None:
    """Normalize scalar values into compact text for human-readable reports."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return normalize_optional_text(str(value))
    if isinstance(value, str):
        return normalize_optional_text(value)
    return None


def _extract_sportradar_date_label(value: object) -> str | None:
    """Extract the compact date label used in Sportradar date objects."""

    if not isinstance(value, Mapping):
        return None
    return _first_text_value(value, ("date",))


def _join_line_parts(
    lead: str,
    *parts: str | None,
    separator: str = ", ",
) -> str:
    """Join an already required lead with optional details."""

    normalized_parts = [
        part for part in (normalize_optional_text(part) for part in parts) if part is not None
    ]
    if not normalized_parts:
        return lead
    return f"{lead} ({separator.join(normalized_parts)})"


def _extract_team_names(match_info: Mapping[str, object] | None) -> Mapping[str, str]:
    """Resolve stable home and away team labels from a match info payload."""

    fallback_names = {"home": "Home", "away": "Away"}
    if match_info is None:
        return fallback_names

    match = match_info.get("match")
    if not isinstance(match, Mapping):
        return fallback_names

    teams = match.get("teams")
    if not isinstance(teams, Mapping):
        return fallback_names

    resolved_names = dict(fallback_names)
    for side in ("home", "away"):
        team = teams.get(side)
        if not isinstance(team, Mapping):
            continue
        for key in ("name", "mediumname", "abbr"):
            value = normalize_optional_text(team.get(key) if isinstance(team, Mapping) else None)
            if value is not None:
                resolved_names[side] = value
                break
    return resolved_names


def _build_match_result_line(
    match_info: Mapping[str, object] | None,
    team_names: Mapping[str, str],
) -> str | None:
    """Build the full-time result line from match info when available."""

    if match_info is None:
        return None

    match = match_info.get("match")
    if not isinstance(match, Mapping):
        return None

    result = match.get("result")
    if not isinstance(result, Mapping):
        return None

    home_score = result.get("home")
    away_score = result.get("away")
    if not isinstance(home_score, (int, float)) or not isinstance(away_score, (int, float)):
        return None

    return (
        f"Result: {team_names['home']} {int(home_score)} - "
        f"{int(away_score)} {team_names['away']}"
    )


def _build_match_stat_line(
    *,
    stat_key: str,
    match_details: Mapping[str, object] | None,
    team_names: Mapping[str, str],
) -> str | None:
    """Format one human-readable match statistic from the match details payload."""

    if match_details is None:
        return None

    values = match_details.get("values")
    if not isinstance(values, Mapping):
        return None

    stat_entry = values.get(stat_key)
    if not isinstance(stat_entry, Mapping):
        return None

    stat_name = normalize_optional_text(stat_entry.get("name"))
    stat_values = stat_entry.get("value")
    if stat_name is None or not isinstance(stat_values, Mapping):
        return None

    home_value = stat_values.get("home")
    away_value = stat_values.get("away")
    if home_value is None or away_value is None:
        return None

    home_text = _format_stat_value(stat_key, home_value)
    away_text = _format_stat_value(stat_key, away_value)
    return (
        f"{stat_name}: {team_names['home']} {home_text} | "
        f"{team_names['away']} {away_text}"
    )


def _build_match_cards_lines(
    match_info: Mapping[str, object] | None,
    team_names: Mapping[str, str],
) -> tuple[str, ...]:
    """Format explicit yellow and red card lines from match info totals."""

    if match_info is None:
        return ()

    match = match_info.get("match")
    if not isinstance(match, Mapping):
        return ()

    cards = match.get("cards")
    if not isinstance(cards, Mapping):
        return ()

    home_cards = cards.get("home")
    away_cards = cards.get("away")
    if not isinstance(home_cards, Mapping) or not isinstance(away_cards, Mapping):
        return ()

    lines: list[str] = []
    for label, key in (("Yellow cards", "yellow_count"), ("Red cards", "red_count")):
        home_value = home_cards.get(key)
        away_value = away_cards.get(key)
        if not isinstance(home_value, (int, float)) or not isinstance(away_value, (int, float)):
            continue
        lines.append(
            f"{label}: {team_names['home']} {int(home_value)} | "
            f"{team_names['away']} {int(away_value)}"
        )
    return tuple(lines)


def _build_goals_timeline_line(
    match_timeline: Mapping[str, object] | None,
    team_names: Mapping[str, str],
) -> str | None:
    """Format a compact goals timeline from match events."""

    if match_timeline is None:
        return None

    events = match_timeline.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return None

    goal_labels: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if normalize_optional_text(event.get("type")) != "goal":
            continue
        side = normalize_optional_text(event.get("team"))
        if side not in ("home", "away"):
            continue
        minute = event.get("time")
        if not isinstance(minute, (int, float)):
            continue
        injury_time = event.get("injurytime")
        minute_text = str(int(minute))
        if isinstance(injury_time, (int, float)) and int(injury_time) > 0:
            minute_text = f"{minute_text}+{int(injury_time)}"
        goal_labels.append(f"{minute_text}' {team_names[side]}")

    if not goal_labels:
        return None
    return f"Goals timeline: {'; '.join(goal_labels)}"


def _format_stat_value(stat_key: str, value: object) -> str:
    """Render one statistic value with lightweight formatting."""

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    if stat_key == "110" and not text.endswith("%"):
        return f"{text}%"
    return text


def _is_probably_binary_content_type(content_type: str | None) -> bool:
    """Return whether a content type is likely binary and unsafe for `response.text()`."""

    if content_type is None:
        return False

    normalized_content_type = content_type.casefold()
    text_markers = ("javascript", "json", "html", "plain", "svg", "text", "xml")
    if any(marker in normalized_content_type for marker in text_markers):
        return False

    binary_prefixes = ("audio/", "font/", "image/", "video/")
    return normalized_content_type.startswith(binary_prefixes) or (
        "octet-stream" in normalized_content_type
    )


def _build_sportradar_proxy_response_headers(
    source_headers: Mapping[str, str],
) -> dict[str, str]:
    """Normalize proxied Sportradar response headers for in-page browser use."""

    excluded_headers = {
        "access-control-allow-credentials",
        "access-control-allow-origin",
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "vary",
    }
    headers = {
        header_name: header_value
        for header_name, header_value in source_headers.items()
        if header_name.casefold() not in excluded_headers
    }
    headers["access-control-allow-origin"] = SPORTYBET_BASE_URL
    headers["access-control-allow-credentials"] = "true"
    headers["vary"] = "Origin"
    return headers


def _build_response_file_name(
    *,
    index: int,
    response_url: str,
    body_kind: Literal["json", "text"],
) -> str:
    """Build a stable filename for one captured response body."""

    parsed_url = urlparse(response_url)
    path_label = parsed_url.path.rsplit("/", maxsplit=1)[-1] or "response"
    digest = hashlib.sha1(response_url.encode("utf-8")).hexdigest()[:10]
    extension = "json" if body_kind == "json" else "txt"
    return f"{index:02d}-{path_label}-{digest}.{extension}"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the fixture stats scraper."""

    parser = argparse.ArgumentParser(
        description=(
            "Fetch SportyBet fixture-page stats by loading the underlying "
            "Sportradar widgets in a live browser session."
        )
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Full public SportyBet fixture-page URL to fetch.",
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
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. `markdown` is easier to inspect manually.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. When omitted, output is written to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where widget and response dumps will be written.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium in headed mode for local debugging.",
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
    """Execute the fixture stats fetcher and render the chosen output format."""

    fixture_url = _resolve_fixture_url(args)
    scraper = SportyBetFixtureStatsScraper(headless=not args.headful)
    result = await scraper.fetch_fixture_stats(
        fixture_url=fixture_url,
        output_dir=args.output_dir,
    )
    if args.format == "markdown":
        return render_fixture_stats_markdown(result)
    return json.dumps(
        result.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )


def run(argv: Sequence[str] | None = None) -> None:
    """CLI entry point used by `puntlab-agent-sportybet-fixture-stats`."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    rendered = asyncio.run(_run_async(args))
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
        return
    print(rendered)


__all__ = [
    "DEFAULT_WIDGET_KEYS",
    "SPORTRADAR_WIDGET_TYPES",
    "SportyBetFixtureStatsResponse",
    "SportyBetFixtureStatsResult",
    "SportyBetFixtureStatsScraper",
    "SportyBetFixtureStatsWidget",
    "build_fixture_details_snapshot",
    "extract_event_id_from_fixture_url",
    "extract_match_id_from_event_id",
    "render_fixture_stats_markdown",
    "run",
]


if __name__ == "__main__":
    run()
