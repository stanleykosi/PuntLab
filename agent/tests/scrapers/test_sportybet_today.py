"""Tests for PuntLab's SportyBet today-games collector helpers.

Purpose: verify the pure route-building, pagination, payload-flattening, and
output-rendering logic used by the SportyBet today-games CLI without opening a
real browser session.
Scope: unit coverage for helper functions in `src.scrapers.sportybet_today`.
Dependencies: pytest plus the today-games collector models and helpers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from src.providers.odds_mapping import build_fixture_market_snapshots, build_odds_market_catalog
from src.schemas.fixture_details import FixtureDetails, FixtureDetailSection
from src.schemas.odds import NormalizedOdds
from src.scrapers import sportybet_today_full_report
from src.scrapers.sportybet_today import (
    SportyBetTodayEvent,
    SportyBetTodaySlate,
    SportyBetTodaySport,
    _build_fixture_from_today_event,
    _build_paged_feed_url,
    _build_today_sport_routes,
    _flatten_today_feed,
    _render_markdown_output,
    _render_slate,
    build_fixtures_from_today_slate,
)
from src.scrapers.sportybet_today_full_report import (
    FullReportBundle,
    _fetch_fixture_details,
    _is_prematch_event,
    _select_report_events,
    render_full_report_markdown,
)


def build_route() -> SportyBetTodaySport:
    """Return a canonical football route used across today-feed tests."""

    return SportyBetTodaySport(
        sport_id="sr:sport:1",
        sport_name="Football",
        route_slug="football",
        page_url="https://www.sportybet.com/ng/sport/football/",
        event_size_hint=170,
    )


def build_payload() -> dict[str, object]:
    """Return a representative SportyBet today-feed payload."""

    return {
        "bizCode": 10000,
        "data": {
            "totalNum": 2,
            "tournaments": [
                {
                    "categoryId": "sr:category:393",
                    "categoryName": "International Clubs",
                    "id": "sr:tournament:7",
                    "name": "UEFA Champions League",
                    "events": [
                        {
                            "eventId": "sr:match:69339428",
                            "gameId": "29566",
                            "estimateStartTime": 1776279600000,
                            "status": 0,
                            "matchStatus": "Not start",
                            "homeTeamName": "Arsenal",
                            "awayTeamName": "Sporting",
                            "totalMarketSize": 907,
                            "markets": [{"id": "1"}, {"id": "18"}],
                            "sport": {
                                "id": "sr:sport:1",
                                "name": "Football",
                                "category": {
                                    "id": "sr:category:393",
                                    "name": "International Clubs",
                                    "tournament": {
                                        "id": "sr:tournament:7",
                                        "name": "UEFA Champions League",
                                    },
                                },
                            },
                        }
                    ],
                },
                {
                    "categoryId": "sr:category:30",
                    "categoryName": "England",
                    "id": "sr:tournament:17",
                    "name": "Premier League",
                    "events": [
                        {
                            "eventId": "sr:match:61301159",
                            "gameId": "35710",
                            "estimateStartTime": 1776285000000,
                            "status": 0,
                            "matchStatus": "Not start",
                            "homeTeamName": "Brentford",
                            "awayTeamName": "Fulham",
                            "totalMarketSize": 772,
                            "markets": [{"id": "1"}],
                            "sport": {
                                "id": "sr:sport:1",
                                "name": "Football",
                                "category": {
                                    "id": "sr:category:30",
                                    "name": "England",
                                    "tournament": {
                                        "id": "sr:tournament:17",
                                        "name": "Premier League",
                                    },
                                },
                            },
                        }
                    ],
                },
            ],
        },
    }


def test_build_today_sport_routes_uses_nav_links_and_skips_zero_count_sports() -> None:
    """Route building should prefer SportyBet's live nav URLs when present."""

    sport_rows = [
        {"id": "sr:sport:1", "name": "Football", "eventSize": 170},
        {"id": "sr:sport:31", "name": "Counter-Strike", "eventSize": 3},
        {"id": "sr:sport:109", "name": "League of Legends", "eventSize": 2},
        {"id": "sr:sport:999", "name": "Unused", "eventSize": 0},
    ]
    anchor_rows = [
        {"text": "Live Betting", "href": "https://www.sportybet.com/ng/sport/live/"},
        {"text": "Football", "href": "https://www.sportybet.com/ng/sport/football/"},
        {
            "text": "Counter-Strike",
            "href": "https://www.sportybet.com/ng/sport/counterStrike/",
        },
        {"text": "League of Legends", "href": "https://www.sportybet.com/ng/sport/lol/"},
    ]

    routes = _build_today_sport_routes(sport_rows, anchor_rows)

    assert [route.sport_name for route in routes] == [
        "Football",
        "Counter-Strike",
        "League of Legends",
    ]
    assert [route.route_slug for route in routes] == ["football", "counterStrike", "lol"]
    assert routes[0].page_url == "https://www.sportybet.com/ng/sport/football/"


def test_build_paged_feed_url_replaces_page_number_and_timestamp() -> None:
    """Pagination should only change `pageNum` and `_t`."""

    paged_url = _build_paged_feed_url(
        (
            "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents?"
            "sportId=sr%3Asport%3A1&marketId=1%2C18&pageSize=100&pageNum=1"
            "&todayGames=true&timeline=10.2&_t=1776257392248"
        ),
        2,
        timestamp_ms=1776257399999,
    )

    assert "pageNum=2" in paged_url
    assert "_t=1776257399999" in paged_url
    assert "todayGames=true" in paged_url
    assert "timeline=10.2" in paged_url


def test_flatten_today_feed_extracts_event_rows() -> None:
    """Today-feed flattening should preserve tournament and kickoff metadata."""

    events = _flatten_today_feed(build_payload(), build_route())

    assert len(events) == 2
    assert events[0].event_id == "sr:match:69339428"
    assert events[0].game_id == "29566"
    assert events[0].tournament_name == "UEFA Champions League"
    assert events[0].category_name == "International Clubs"
    assert events[0].market_count == 2
    assert events[0].total_market_size == 907
    assert events[0].kickoff == datetime.fromtimestamp(
        1776279600000 / 1000,
        tz=UTC,
    ).astimezone(events[0].kickoff.tzinfo)
    assert events[1].tournament_name == "Premier League"
    assert events[1].home_team_name == "Brentford"


def test_render_slate_jsonl_serializes_one_event_per_line() -> None:
    """JSONL rendering should emit exactly one line per normalized event."""

    events = _flatten_today_feed(build_payload(), build_route())
    slate = SportyBetTodaySlate(
        run_date=datetime(2026, 4, 15, tzinfo=UTC).date(),
        fetched_at=datetime(2026, 4, 15, 13, 0, tzinfo=UTC),
        timezone="Africa/Lagos",
        sports=(build_route(),),
        events=events,
    )

    rendered = _render_slate(slate, output_format="jsonl")
    lines = rendered.splitlines()

    assert len(lines) == 2
    first_row = json.loads(lines[0])
    assert first_row["event_id"] == events[0].event_id
    assert first_row["tournament_name"] == events[0].tournament_name
    assert isinstance(SportyBetTodayEvent.model_validate(first_row), SportyBetTodayEvent)


def test_build_fixtures_from_today_slate_creates_canonical_fixture_rows() -> None:
    """Today-slate fixture conversion should preserve canonical event identity."""

    events = _flatten_today_feed(build_payload(), build_route())
    slate = SportyBetTodaySlate(
        run_date=datetime(2026, 4, 15, tzinfo=UTC).date(),
        fetched_at=datetime(2026, 4, 15, 13, 0, tzinfo=UTC),
        timezone="Africa/Lagos",
        sports=(build_route(),),
        events=events,
    )

    fixtures = build_fixtures_from_today_slate(slate)

    assert len(fixtures) == 2
    assert fixtures[0].sportradar_id == "sr:match:69339428"
    assert fixtures[0].source_provider == "sportybet"
    assert fixtures[0].source_id == "29566"
    assert fixtures[0].competition == "UEFA Champions League"
    assert fixtures[0].country == "International Clubs"


def test_render_markdown_output_lists_full_market_details_under_each_fixture() -> None:
    """Markdown rendering should order fixtures and show grouped full-market rows."""

    events = _flatten_today_feed(build_payload(), build_route())
    fixture = _build_fixture_from_today_event(events[0])
    assert fixture is not None

    odds_rows = (
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Home",
            odds=1.83,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Home",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "event_id": fixture.get_fixture_ref(),
                "game_id": "29566",
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 907,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Draw",
            odds=3.50,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Draw",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "event_id": fixture.get_fixture_ref(),
                "game_id": "29566",
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 907,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Away",
            odds=4.20,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Away",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "event_id": fixture.get_fixture_ref(),
                "game_id": "29566",
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 907,
                "sportybet_fetch_source": "api",
            },
        ),
    )
    slate = SportyBetTodaySlate(
        run_date=datetime(2026, 4, 15, tzinfo=UTC).date(),
        fetched_at=datetime(2026, 4, 15, 13, 0, tzinfo=UTC),
        timezone="Africa/Lagos",
        sports=(build_route(),),
        events=events,
    )
    catalog = build_odds_market_catalog(odds_rows)
    snapshots = build_fixture_market_snapshots((fixture,), catalog)

    rendered = _render_markdown_output(
        slate=slate,
        ordered_events=(events[0],),
        snapshots=snapshots,
        failures={},
        unsupported_events=(),
    )

    assert "# SportyBet Today Games Full Market Export" in rendered
    assert "Arsenal vs Sporting" in rendered
    assert "Today-feed listing markets: `2` | Reported total markets: `907`" in rendered
    assert "#### Main" in rendered
    assert "- 1X2 [key=1x2] | market_id=1 | mapped=1x2 | selections=3" in rendered
    assert "  - Home [home] @ 1.83" in rendered


def test_full_report_filters_prematch_run_date_rows_and_renders_details() -> None:
    """Full report rendering should combine details and markets in fixture order."""

    events = _flatten_today_feed(build_payload(), build_route())
    slate = SportyBetTodaySlate(
        run_date=datetime(2026, 4, 15, tzinfo=UTC).date(),
        fetched_at=datetime(2026, 4, 15, 13, 0, tzinfo=UTC),
        timezone="Africa/Lagos",
        sports=(build_route(),),
        events=events,
    )
    selected_events, skipped_events = _select_report_events(
        slate,
        include_extra_dates=False,
        max_fixtures=None,
    )
    fixture = _build_fixture_from_today_event(selected_events[0])
    assert fixture is not None
    odds_rows = (
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Home",
            odds=1.83,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Home",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "event_id": fixture.get_fixture_ref(),
                "game_id": "29566",
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 907,
                "sportybet_fetch_source": "api",
            },
        ),
    )
    snapshots = build_fixture_market_snapshots(
        (fixture,),
        build_odds_market_catalog(odds_rows),
    )
    details = FixtureDetails(
        fixture_ref=fixture.get_fixture_ref(),
        fixture_url="https://www.sportybet.com/ng/sport/football/example",
        event_id=fixture.get_fixture_ref(),
        match_id="69339428",
        fetched_at=datetime(2026, 4, 15, 13, 5, tzinfo=UTC),
        widget_loader_status="loaded",
        sections=(
            FixtureDetailSection(
                widget_key="teamInfo",
                widget_type="team.info",
                status="mounted",
                content_lines=("Home manager: Mikel Arteta",),
            ),
            FixtureDetailSection(
                widget_key="h2h",
                widget_type="match.headToHead",
                status="mounted",
                content_lines=("Previous meetings: Arsenal wins 3",),
            ),
        ),
    )

    rendered = render_full_report_markdown(
        FullReportBundle(
            slate=slate,
            ordered_events=(selected_events[0],),
            fixtures_by_ref={fixture.get_fixture_ref(): fixture},
            market_snapshots_by_ref={snapshots[0].fixture_ref: snapshots[0]},
            fixture_details_by_ref={details.fixture_ref: details},
            market_failures={},
            detail_failures={},
            skipped_events=skipped_events,
        )
    )

    assert _is_prematch_event(events[0]) is True
    assert "# SportyBet Today Prematch Full Report" in rendered
    assert "Arsenal vs Sporting" in rendered
    assert "##### teamInfo" in rendered
    assert "Home manager: Mikel Arteta" in rendered
    assert "##### h2h" in rendered
    assert "Previous meetings: Arsenal wins 3" in rendered
    assert "##### Markets - Main" in rendered
    assert "  - Home [home] @ 1.83" in rendered


@pytest.mark.asyncio
async def test_full_report_retries_fixture_detail_failures(monkeypatch) -> None:
    """Transient widget fetch failures should be retried before the report gives up."""

    event = _flatten_today_feed(build_payload(), build_route())[0]
    fixture = _build_fixture_from_today_event(event)
    assert fixture is not None
    attempts = 0

    class FakeScraper:
        def __init__(self, **_: object) -> None:
            pass

        async def fetch_fixture_stats(self, *, fixture_url: str) -> object:
            nonlocal attempts
            attempts += 1
            assert fixture_url
            if attempts == 1:
                raise TimeoutError()
            return object()

    def fake_snapshot(_: object, *, fixture_ref: str) -> FixtureDetails:
        return FixtureDetails(
            fixture_ref=fixture_ref,
            fixture_url="https://www.sportybet.com/ng/sport/football/example",
            event_id=fixture_ref,
            match_id="69339428",
            fetched_at=datetime(2026, 4, 15, 13, 5, tzinfo=UTC),
            widget_loader_status="loaded",
            sections=(),
        )

    monkeypatch.setattr(
        sportybet_today_full_report,
        "SportyBetFixtureStatsScraper",
        FakeScraper,
    )
    monkeypatch.setattr(
        sportybet_today_full_report,
        "build_fixture_details_snapshot",
        fake_snapshot,
    )

    details_by_ref, failures = await _fetch_fixture_details(
        (fixture,),
        widget_keys=("teamInfo",),
        headless=True,
        concurrency=1,
        timeout_seconds=1,
        navigation_timeout_ms=1_000,
        retries=1,
        retry_backoff_seconds=0,
        progress=lambda _: None,
    )

    assert attempts == 2
    assert fixture.get_fixture_ref() in details_by_ref
    assert failures == {}
