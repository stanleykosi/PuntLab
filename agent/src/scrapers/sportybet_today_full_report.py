"""Full SportyBet today prematch report exporter.

Purpose: generate one operator-readable markdown report containing today's
SportyBet prematch fixtures, full market books, and fixture-page details.
Scope: collect Today Games, filter prematch rows, fetch all SportyBet markets,
fetch Sportradar-powered fixture details, and render a sorted markdown report.
Dependencies: the canonical SportyBet today collector, market API client,
fixture stats scraper, odds catalog, and fixture-detail schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE
from src.providers.base import ProviderError, RateLimitExhausted
from src.providers.odds_mapping import (
    build_fixture_market_snapshots,
    build_odds_market_catalog,
)
from src.schemas.common import normalize_optional_text
from src.schemas.fixture_details import FixtureDetails
from src.schemas.fixtures import NormalizedFixture
from src.schemas.market_snapshot import (
    FixtureMarketSelection,
    FixtureMarketSnapshot,
    FixtureMarketSnapshotEntry,
)
from src.schemas.odds import NormalizedOdds
from src.scrapers.sportybet_api import SportyBetAPIClient
from src.scrapers.sportybet_fixture_probe import build_fixture_page_url
from src.scrapers.sportybet_fixture_stats import (
    DEFAULT_WIDGET_KEYS,
    SportyBetFixtureStatsScraper,
    build_fixture_details_snapshot,
)
from src.scrapers.sportybet_today import (
    SportyBetTodayEvent,
    SportyBetTodayGamesCollector,
    SportyBetTodaySlate,
    _build_fixture_from_today_event,
    _InMemoryAsyncRedis,
    _seconds_until_next_rate_limit_window,
    _sort_today_events,
)

DEFAULT_OUTPUT_PATH = Path("sportybet-today-full-report.md")
DEFAULT_DETAILS_CONCURRENCY = 2
DEFAULT_DETAIL_TIMEOUT_SECONDS = 300
DEFAULT_DETAIL_NAVIGATION_TIMEOUT_MS = 60_000
DEFAULT_DETAIL_RETRIES = 2
DEFAULT_DETAIL_RETRY_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class FullReportBundle:
    """Data bundle required by the markdown renderer."""

    slate: SportyBetTodaySlate
    ordered_events: tuple[SportyBetTodayEvent, ...]
    fixtures_by_ref: Mapping[str, NormalizedFixture]
    market_snapshots_by_ref: Mapping[str, FixtureMarketSnapshot]
    fixture_details_by_ref: Mapping[str, FixtureDetails]
    market_failures: Mapping[str, str]
    detail_failures: Mapping[str, str]
    skipped_events: tuple[SportyBetTodayEvent, ...]


def _build_parser() -> argparse.ArgumentParser:
    """Build the full-report CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Fetch SportyBet Today Games prematch fixtures, full markets, and "
            "fixture-page details into one sorted markdown report."
        )
    )
    parser.add_argument(
        "--sport",
        action="append",
        default=None,
        help="Sport selector. Defaults to football. May be comma-separated or repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--details-concurrency",
        type=int,
        default=DEFAULT_DETAILS_CONCURRENCY,
        help="Maximum concurrent fixture-page detail fetches.",
    )
    parser.add_argument(
        "--detail-timeout-seconds",
        type=int,
        default=DEFAULT_DETAIL_TIMEOUT_SECONDS,
        help="Per-fixture timeout for fixture-page detail fetches.",
    )
    parser.add_argument(
        "--detail-navigation-timeout-ms",
        type=int,
        default=DEFAULT_DETAIL_NAVIGATION_TIMEOUT_MS,
        help="Per-fixture page navigation timeout used by the detail scraper.",
    )
    parser.add_argument(
        "--detail-retries",
        type=int,
        default=DEFAULT_DETAIL_RETRIES,
        help="Number of retry attempts after a fixture-page detail fetch fails.",
    )
    parser.add_argument(
        "--detail-retry-backoff-seconds",
        type=float,
        default=DEFAULT_DETAIL_RETRY_BACKOFF_SECONDS,
        help="Base delay between fixture-page detail retries.",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
        help="Optional cap for live smoke tests. Omit to fetch every prematch fixture.",
    )
    parser.add_argument(
        "--include-extra-dates",
        action="store_true",
        help="Include SportyBet Today Games rows whose WAT kickoff date is not the feed run date.",
    )
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="Fetch markets only. Useful when isolating a fixture-page outage.",
    )
    parser.add_argument(
        "--widget-keys",
        type=str,
        default=",".join(DEFAULT_WIDGET_KEYS),
        help="Comma-separated fixture-page widget keys to fetch.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium in headed mode for local debugging.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> str:
    """Collect, fetch, render, and write the full report."""

    if args.details_concurrency <= 0:
        raise ValueError("--details-concurrency must be positive.")
    if args.detail_timeout_seconds <= 0:
        raise ValueError("--detail-timeout-seconds must be positive.")
    if args.detail_navigation_timeout_ms <= 0:
        raise ValueError("--detail-navigation-timeout-ms must be positive.")
    if args.detail_retries < 0:
        raise ValueError("--detail-retries cannot be negative.")
    if args.detail_retry_backoff_seconds < 0:
        raise ValueError("--detail-retry-backoff-seconds cannot be negative.")
    if args.max_fixtures is not None and args.max_fixtures <= 0:
        raise ValueError("--max-fixtures must be positive when supplied.")

    progress = _build_progress_writer()
    requested_sports = _parse_requested_sports(args.sport)
    progress(f"Collecting SportyBet Today Games for: {', '.join(requested_sports)}")
    collector = SportyBetTodayGamesCollector(headless=not args.headful)
    slate = await collector.collect(sports=requested_sports)

    ordered_events, skipped_events = _select_report_events(
        slate,
        include_extra_dates=args.include_extra_dates,
        max_fixtures=args.max_fixtures,
    )
    progress(
        f"Selected {len(ordered_events)} prematch report fixtures "
        f"from {len(slate.events)} today-feed rows."
    )

    fixtures_by_ref = _build_fixtures_by_ref(ordered_events)
    market_rows, market_failures = await _fetch_market_rows(
        tuple(fixtures_by_ref.values()),
        progress=progress,
    )
    catalog = build_odds_market_catalog(
        market_rows,
        sport_by_fixture={
            fixture_ref: fixture.sport for fixture_ref, fixture in fixtures_by_ref.items()
        },
    )
    market_snapshots = build_fixture_market_snapshots(
        tuple(fixtures_by_ref.values()),
        catalog,
    )
    market_snapshots_by_ref = {snapshot.fixture_ref: snapshot for snapshot in market_snapshots}

    if args.skip_details:
        fixture_details_by_ref: dict[str, FixtureDetails] = {}
        detail_failures: dict[str, str] = {}
    else:
        fixture_details_by_ref, detail_failures = await _fetch_fixture_details(
            tuple(fixtures_by_ref.values()),
            widget_keys=_parse_widget_keys(args.widget_keys),
            headless=not args.headful,
            concurrency=args.details_concurrency,
            timeout_seconds=args.detail_timeout_seconds,
            navigation_timeout_ms=args.detail_navigation_timeout_ms,
            retries=args.detail_retries,
            retry_backoff_seconds=args.detail_retry_backoff_seconds,
            progress=progress,
        )

    rendered = render_full_report_markdown(
        FullReportBundle(
            slate=slate,
            ordered_events=ordered_events,
            fixtures_by_ref=fixtures_by_ref,
            market_snapshots_by_ref=market_snapshots_by_ref,
            fixture_details_by_ref=fixture_details_by_ref,
            market_failures=market_failures,
            detail_failures=detail_failures,
            skipped_events=skipped_events,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    progress(f"Wrote markdown report to {args.output}")
    return str(args.output)


def _build_progress_writer() -> Callable[[str], None]:
    """Return a timestamped progress writer for long live runs."""

    def write_progress(message: str) -> None:
        timestamp = datetime.now(WAT_TIMEZONE).strftime("%H:%M:%S WAT")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)

    return write_progress


def _parse_requested_sports(values: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize repeated or comma-separated sport selectors."""

    sports: list[str] = []
    for value in values or ():
        for part in value.split(","):
            normalized = normalize_optional_text(part)
            if normalized is not None:
                sports.append(normalized)
    return tuple(sports) if sports else ("football",)


def _parse_widget_keys(value: str) -> tuple[str, ...]:
    """Parse comma-separated fixture detail widget keys."""

    widget_keys = tuple(
        normalized
        for part in value.split(",")
        if (normalized := normalize_optional_text(part)) is not None
    )
    if not widget_keys:
        raise ValueError("--widget-keys must include at least one widget key.")
    return widget_keys


def _select_report_events(
    slate: SportyBetTodaySlate,
    *,
    include_extra_dates: bool,
    max_fixtures: int | None,
) -> tuple[tuple[SportyBetTodayEvent, ...], tuple[SportyBetTodayEvent, ...]]:
    """Select WAT-run-date prematch events and return skipped rows separately."""

    ordered = _sort_today_events(slate.events)
    selected: list[SportyBetTodayEvent] = []
    skipped: list[SportyBetTodayEvent] = []
    for index, event in enumerate(ordered):
        kickoff_date = event.kickoff.astimezone(WAT_TIMEZONE).date()
        if not include_extra_dates and kickoff_date != slate.run_date:
            skipped.append(event)
            continue
        if not _is_prematch_event(event):
            skipped.append(event)
            continue
        if _build_fixture_from_today_event(event) is None:
            skipped.append(event)
            continue
        selected.append(event)
        if max_fixtures is not None and len(selected) >= max_fixtures:
            skipped.extend(ordered[index + 1 :])
            break
    return tuple(selected), tuple(skipped)


def _is_prematch_event(event: SportyBetTodayEvent) -> bool:
    """Return whether a today-feed row still represents a prematch fixture."""

    status_text = (event.match_status or "").casefold()
    live_or_finished_markers = (
        "live",
        "ended",
        "finished",
        "cancel",
        "postpon",
        "abandon",
        "1st half",
        "2nd half",
        "halftime",
    )
    if any(marker in status_text for marker in live_or_finished_markers):
        return False
    if event.status is not None and event.status != 0:
        return False
    return True


def _build_fixtures_by_ref(
    events: Sequence[SportyBetTodayEvent],
) -> dict[str, NormalizedFixture]:
    """Build normalized fixtures keyed by canonical fixture ref."""

    fixtures_by_ref: dict[str, NormalizedFixture] = {}
    for event in events:
        fixture = _build_fixture_from_today_event(event)
        if fixture is None:
            continue
        fixtures_by_ref.setdefault(fixture.get_fixture_ref(), fixture)
    return fixtures_by_ref


async def _fetch_market_rows(
    fixtures: Sequence[NormalizedFixture],
    *,
    progress: Callable[[str], None],
) -> tuple[tuple[NormalizedOdds, ...], dict[str, str]]:
    """Fetch full SportyBet market rows for every selected fixture."""

    redis_cache = RedisClient(redis_client=_InMemoryAsyncRedis())
    api_client = SportyBetAPIClient(redis_cache)
    fetched_rows: list[NormalizedOdds] = []
    failures: dict[str, str] = {}
    try:
        total = len(fixtures)
        for index, fixture in enumerate(fixtures, start=1):
            fixture_ref = fixture.get_fixture_ref()
            progress(f"Markets {index}/{total}: {fixture.home_team} vs {fixture.away_team}")
            while True:
                try:
                    fetched_rows.extend(
                        await api_client.fetch_markets(
                            fixture_ref,
                            fixture=fixture,
                            use_cache=True,
                        )
                    )
                    break
                except RateLimitExhausted:
                    wait_seconds = _seconds_until_next_rate_limit_window()
                    progress(f"SportyBet market rate limit hit; waiting {wait_seconds}s")
                    await asyncio.sleep(wait_seconds)
                except (ProviderError, ValueError) as exc:
                    failures[fixture_ref] = str(exc)
                    break
    finally:
        await api_client.aclose()
        await redis_cache.close()
    return tuple(fetched_rows), failures


async def _fetch_fixture_details(
    fixtures: Sequence[NormalizedFixture],
    *,
    widget_keys: tuple[str, ...],
    headless: bool,
    concurrency: int,
    timeout_seconds: int,
    navigation_timeout_ms: int,
    retries: int,
    retry_backoff_seconds: float,
    progress: Callable[[str], None],
) -> tuple[dict[str, FixtureDetails], dict[str, str]]:
    """Fetch fixture-page details with bounded concurrency."""

    semaphore = asyncio.Semaphore(concurrency)
    details_by_ref: dict[str, FixtureDetails] = {}
    failures: dict[str, str] = {}
    total = len(fixtures)

    async def fetch_one(index: int, fixture: NormalizedFixture) -> None:
        fixture_ref = fixture.get_fixture_ref()
        async with semaphore:
            progress(f"Details {index}/{total}: {fixture.home_team} vs {fixture.away_team}")
            fixture_url = _build_fixture_url(fixture)
            last_error = None
            for attempt in range(1, retries + 2):
                try:
                    scraper = SportyBetFixtureStatsScraper(
                        headless=headless,
                        navigation_timeout_ms=navigation_timeout_ms,
                        widget_keys=widget_keys,
                    )
                    result = await asyncio.wait_for(
                        scraper.fetch_fixture_stats(fixture_url=fixture_url),
                        timeout=timeout_seconds,
                    )
                    break
                except Exception as exc:
                    last_error = _format_exception_message(exc)
                    if attempt > retries:
                        failures[fixture_ref] = last_error
                        progress(f"Details failed for {fixture_ref}: {last_error}")
                        return
                    delay_seconds = retry_backoff_seconds * attempt
                    progress(
                        f"Details retry {attempt}/{retries} for {fixture_ref}: "
                        f"{last_error}; waiting {delay_seconds:g}s"
                    )
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
            details_by_ref[fixture_ref] = build_fixture_details_snapshot(
                result,
                fixture_ref=fixture_ref,
            )

    await asyncio.gather(
        *(fetch_one(index, fixture) for index, fixture in enumerate(fixtures, start=1))
    )
    return details_by_ref, failures


def _format_exception_message(exc: Exception) -> str:
    """Return useful text for exceptions whose default string is blank."""

    message = normalize_optional_text(str(exc))
    if message is not None:
        return message
    return type(exc).__name__


def _build_fixture_url(fixture: NormalizedFixture) -> str:
    """Build a public SportyBet fixture page URL for one normalized fixture."""

    if fixture.sportradar_id is None:
        raise ValueError(f"Fixture {fixture.get_fixture_ref()} is missing a Sportradar ID.")
    if fixture.country is None:
        raise ValueError(f"Fixture {fixture.get_fixture_ref()} is missing a country label.")
    return build_fixture_page_url(
        event_id=fixture.sportradar_id,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        country=fixture.country,
        competition=fixture.competition,
        sport="football" if fixture.sport.value == "soccer" else fixture.sport.value,
    )


def render_full_report_markdown(bundle: FullReportBundle) -> str:
    """Render one full SportyBet today report as markdown."""

    feed_fetched_label = bundle.slate.fetched_at.astimezone(WAT_TIMEZONE).strftime(
        "%Y-%m-%d %H:%M %Z"
    )
    sports_label = ", ".join(route.sport_name for route in bundle.slate.sports) or "all"
    lines = [
        "# SportyBet Today Prematch Full Report",
        "",
        f"- Generated: {datetime.now(WAT_TIMEZONE).strftime('%Y-%m-%d %H:%M %Z')}",
        f"- Feed fetched: {feed_fetched_label}",
        f"- Feed run date: {bundle.slate.run_date.isoformat()}",
        f"- Sports requested: {sports_label}",
        f"- Today-feed rows: {len(bundle.slate.events)}",
        f"- Prematch fixtures exported: {len(bundle.ordered_events)}",
        f"- Market fetch failures: {len(bundle.market_failures)}",
        f"- Detail fetch failures: {len(bundle.detail_failures)}",
        f"- Skipped rows: {len(bundle.skipped_events)}",
        "",
        "## Fixture Index",
        "",
    ]

    for event in bundle.ordered_events:
        kickoff = event.kickoff.astimezone(WAT_TIMEZONE)
        lines.append(
            f"- {kickoff.strftime('%H:%M')} WAT | {event.home_team_name} vs "
            f"{event.away_team_name} | {event.category_name} / {event.tournament_name} | "
            f"`{event.event_id}`"
        )

    current_date = None
    for event in bundle.ordered_events:
        kickoff = event.kickoff.astimezone(WAT_TIMEZONE)
        kickoff_date = kickoff.date()
        if kickoff_date != current_date:
            current_date = kickoff_date
            lines.extend(["", f"## {kickoff_date.isoformat()}", ""])

        fixture_ref = event.event_id
        fixture = bundle.fixtures_by_ref.get(fixture_ref)
        snapshot = bundle.market_snapshots_by_ref.get(fixture_ref)
        details = bundle.fixture_details_by_ref.get(fixture_ref)

        lines.extend(
            [
                (
                    f"### {kickoff.strftime('%H:%M')} WAT - "
                    f"{event.home_team_name} vs {event.away_team_name}"
                ),
                "",
                f"- Competition: {event.category_name} / {event.tournament_name}",
                f"- Event ID: `{event.event_id}`",
                f"- Game ID: `{event.game_id or 'n/a'}`",
                f"- Status: `{event.match_status or event.status or 'unknown'}`",
                f"- Today-feed listing markets: `{event.market_count}`",
                f"- Reported total markets: `{event.total_market_size}`",
            ]
        )
        if fixture is not None:
            lines.append(f"- Fixture ref: `{fixture.get_fixture_ref()}`")

        _append_fixture_details(
            lines,
            fixture_ref=fixture_ref,
            details=details,
            failure=bundle.detail_failures.get(fixture_ref),
        )
        _append_market_snapshot(
            lines,
            event=event,
            snapshot=snapshot,
            failure=bundle.market_failures.get(fixture_ref),
        )

    if bundle.skipped_events:
        lines.extend(["", "## Skipped Rows", ""])
        for event in bundle.skipped_events:
            kickoff = event.kickoff.astimezone(WAT_TIMEZONE)
            lines.append(
                f"- {kickoff.strftime('%Y-%m-%d %H:%M')} WAT | "
                f"{event.home_team_name} vs {event.away_team_name} | "
                f"{event.match_status or event.status or 'unknown'} | `{event.event_id}`"
            )

    return "\n".join(lines).rstrip() + "\n"


def _append_fixture_details(
    lines: list[str],
    *,
    fixture_ref: str,
    details: FixtureDetails | None,
    failure: str | None,
) -> None:
    """Append fixture-page details to markdown lines."""

    lines.extend(["", "#### Fixture Details"])
    if details is None:
        if failure is not None:
            lines.append(f"- Detail fetch failed for `{fixture_ref}`: `{failure}`")
        else:
            lines.append("- No fixture-page details were captured.")
        return

    lines.append(
        f"- Widget loader: `{details.widget_loader_status}` | "
        f"Sections: `{len(details.sections)}` | "
        f"Fetched: `{details.fetched_at.astimezone(WAT_TIMEZONE).strftime('%Y-%m-%d %H:%M %Z')}`"
    )
    for section in details.sections:
        lines.extend(["", f"##### {section.widget_key}"])
        lines.append(f"- Widget type: `{section.widget_type}`")
        lines.append(f"- Status: `{section.status}`")
        lines.append(f"- Response URLs: `{len(section.response_urls)}`")
        if section.error_message is not None:
            lines.append(f"- Error: `{section.error_message}`")
        if not section.content_lines:
            lines.append("- Content: `<empty>`")
            continue
        lines.append("- Content:")
        for content_line in section.content_lines:
            lines.append(f"  - {content_line}")


def _append_market_snapshot(
    lines: list[str],
    *,
    event: SportyBetTodayEvent,
    snapshot: FixtureMarketSnapshot | None,
    failure: str | None,
) -> None:
    """Append full market-book details to markdown lines."""

    lines.extend(["", "#### Markets"])
    if snapshot is None or snapshot.fetched_market_count == 0:
        if failure is not None:
            lines.append(f"- Full-market fetch failed: `{failure}`")
        else:
            lines.append("- Full-market fetch returned no rows.")
        return

    lines.append(
        f"- Today listing markets: `{event.market_count}` | "
        f"Reported total markets: `{event.total_market_size}` | "
        f"Full fetched markets: `{snapshot.fetched_market_count}` | "
        f"Full fetched selections: `{snapshot.fetched_selection_count}` | "
        f"Scoreable: `{snapshot.scoreable_market_count}` | "
        f"Unmapped: `{snapshot.unmapped_market_count}` | "
        f"Fetch source: `{snapshot.fetch_source or 'unknown'}`"
    )
    for market_group in snapshot.market_groups:
        lines.extend(["", f"##### Markets - {market_group.group_name}"])
        for market in market_group.markets:
            lines.append(_render_market_entry(market))
            lines.extend(_render_market_selection(selection) for selection in market.selections)


def _render_market_entry(market: FixtureMarketSnapshotEntry) -> str:
    """Render one market heading line."""

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


def _render_market_selection(selection: FixtureMarketSelection) -> str:
    """Render one market selection line."""

    selection_label = selection.provider_selection_name
    if selection.canonical_selection is not None:
        selection_label += f" [{selection.canonical_selection}]"
    if selection.line is not None and str(selection.line) not in selection_label:
        selection_label += f" ({selection.line:g})"
    return f"  - {selection_label} @ {selection.odds:.2f}"


def run(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for `puntlab-agent-sportybet-today-full-report`."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = asyncio.run(_run_async(args))
    print(output_path)


__all__ = [
    "FullReportBundle",
    "render_full_report_markdown",
    "run",
]


if __name__ == "__main__":
    run()
