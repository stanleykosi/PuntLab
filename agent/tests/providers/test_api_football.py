"""Tests for PuntLab's API-Football provider integration.

Purpose: verify endpoint request shaping, pagination, and normalization from
API-Football payloads into PuntLab's shared schemas without live network calls.
Scope: unit tests for fixtures, odds, standings, player stats, injuries, and
head-to-head responses using `httpx.MockTransport`.
Dependencies: pytest, httpx, the in-memory Redis stub used by the shared
rate-limited client, and the concrete `APIFootballProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, MarketType
from src.providers.api_football import APIFootballProvider
from src.providers.base import RateLimitedClient


class FakeAsyncRedis:
    """Minimal async Redis stub for provider unit tests."""

    def __init__(self) -> None:
        """Initialize in-memory value and TTL stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return a stored string value for one key."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Persist a string value and optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment a stored numeric string value."""

        current_value = int(self.values.get(name, "0"))
        next_value = current_value + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Persist a TTL for an existing key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Pretend the in-memory Redis store is always available."""

        return True

    async def aclose(self) -> None:
        """Close the fake client without side effects."""


def build_provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[APIFootballProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = APIFootballProvider(rate_limited_client, api_key="test-key")
    return provider, rate_limited_client


@pytest.mark.asyncio
async def test_fetch_fixtures_by_date_normalizes_fixture_payload() -> None:
    """Fixture fetches should shape requests and normalize response objects."""

    observed_headers: dict[str, str] = {}
    observed_query: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_headers, observed_query
        observed_headers = dict(request.headers.items())
        observed_query = dict(request.url.params.items())
        return httpx.Response(
            200,
            json={
                "errors": [],
                "response": [
                    {
                        "fixture": {
                            "id": 9001,
                            "date": "2026-04-03T19:45:00+01:00",
                            "status": {"short": "NS"},
                            "venue": {"name": "Emirates Stadium"},
                        },
                        "league": {
                            "id": 39,
                            "name": "Premier League",
                            "country": "England",
                            "season": 2026,
                        },
                        "teams": {
                            "home": {"id": 42, "name": "Arsenal"},
                            "away": {"id": 49, "name": "Chelsea"},
                        },
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    fixtures = await provider.fetch_fixtures_by_date(
        run_date=date(2026, 4, 3),
        league_id=39,
        season=2026,
    )

    assert observed_headers["x-apisports-key"] == "test-key"
    assert observed_query == {
        "league": "39",
        "season": "2026",
        "date": "2026-04-03",
        "timezone": "Africa/Lagos",
    }
    assert len(fixtures) == 1
    assert fixtures[0].home_team == "Arsenal"
    assert fixtures[0].away_team_id == "49"
    assert fixtures[0].source_id == "9001"
    assert fixtures[0].competition == "Premier League"
    assert fixtures[0].kickoff.tzinfo is not None

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_odds_by_fixture_paginates_and_preserves_all_markets() -> None:
    """Odds fetches should keep unsupported markets instead of dropping them."""

    call_pages: list[str] = []
    reference_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reference_calls
        if request.url.path == "/odds/bets":
            reference_calls += 1
            return httpx.Response(
                200,
                json={
                    "errors": [],
                    "response": [
                        {"id": 1, "name": "Match Winner"},
                        {"id": 5, "name": "Goals Over/Under"},
                        {"id": 7, "name": "Both Teams To Score"},
                        {"id": 13, "name": "Asian Handicap"},
                        {"id": 19, "name": "Team To Score First"},
                    ],
                },
                request=request,
            )
        page = request.url.params["page"]
        call_pages.append(page)
        payload = {
            "errors": [],
            "paging": {"current": int(page), "total": 2},
            "response": [
                {
                    "fixture": {"id": 501},
                    "update": "2026-04-03T12:00:00+00:00",
                    "bookmakers": [
                        {
                            "id": 8,
                            "name": "Bet365",
                            "bets": (
                                [
                                    {
                                        "id": 1,
                                        "name": "Match Winner",
                                        "values": [
                                            {"value": "Home", "odd": "1.80"},
                                            {"value": "Draw", "odd": "3.55"},
                                            {"value": "Away", "odd": "4.40"},
                                        ],
                                    },
                                    {
                                        "id": 5,
                                        "name": "Goals Over/Under",
                                        "values": [
                                            {"value": "Over 2.5", "odd": "1.92"},
                                            {"value": "Under 2.5", "odd": "1.86"},
                                        ],
                                    },
                                ]
                                if page == "1"
                                else [
                                    {
                                        "id": 7,
                                        "name": "Both Teams To Score",
                                        "values": [
                                            {"value": "Yes", "odd": "1.70"},
                                            {"value": "No", "odd": "2.10"},
                                        ],
                                    },
                                    {
                                        "id": 13,
                                        "name": "Asian Handicap",
                                        "values": [
                                            {"value": "Home -1.5", "odd": "2.55"},
                                            {"value": "Away +1.5", "odd": "1.54"},
                                        ],
                                    },
                                    {
                                        "id": 19,
                                        "name": "Team To Score First",
                                        "values": [
                                            {"value": "Home", "odd": "1.60"},
                                        ],
                                    },
                                ]
                            ),
                        }
                    ],
                }
            ],
        }
        return httpx.Response(200, json=payload, request=request)

    provider, client = build_provider(handler)

    odds_rows = await provider.fetch_odds_by_fixture(fixture_id=501)

    assert call_pages == ["1", "2"]
    assert reference_calls == 1
    assert len(odds_rows) == 10
    assert {row.market for row in odds_rows} == {
        MarketType.MATCH_RESULT,
        MarketType.OVER_UNDER_25,
        MarketType.BTTS,
        MarketType.ASIAN_HANDICAP,
        None,
    }
    over_market = next(row for row in odds_rows if row.selection == "over")
    handicap_market = next(
        row
        for row in odds_rows
        if row.market == MarketType.ASIAN_HANDICAP and row.selection == "home"
    )
    unmapped_market = next(
        row for row in odds_rows if row.provider_market_name == "Team To Score First"
    )
    assert over_market.fixture_ref == "api-football:501"
    assert over_market.provider == "Bet365"
    assert over_market.provider_market_key == "goals_over_under"
    assert handicap_market.line == -1.5
    assert handicap_market.last_updated == datetime(2026, 4, 3, 12, 0, tzinfo=UTC)
    assert unmapped_market.market is None
    assert unmapped_market.selection == "Home"
    assert unmapped_market.provider_selection_name == "Home"
    assert unmapped_market.participant_scope == "team"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_standings_and_player_stats_normalize_nested_payloads() -> None:
    """Standings and player stats should flatten nested provider structures."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/standings":
            return httpx.Response(
                200,
                json={
                    "errors": [],
                    "response": [
                        {
                            "league": {
                                "id": 39,
                                "name": "Premier League",
                                "season": 2026,
                                "standings": [
                                    [
                                        {
                                            "rank": 1,
                                            "team": {"id": 42, "name": "Arsenal"},
                                            "points": 72,
                                            "goalsDiff": 34,
                                            "form": "W-W-D-W-W",
                                            "all": {
                                                "played": 30,
                                                "win": 22,
                                                "draw": 6,
                                                "lose": 2,
                                                "goals": {"for": 68, "against": 34},
                                            },
                                            "home": {"win": 12},
                                            "away": {"win": 10},
                                        }
                                    ]
                                ],
                            }
                        }
                    ],
                },
                request=request,
            )

        return httpx.Response(
            200,
            json={
                "errors": [],
                "paging": {"current": 1, "total": 1},
                "response": [
                    {
                        "player": {"id": 88, "name": "Bukayo Saka"},
                        "statistics": [
                            {
                                "team": {"id": 42, "name": "Arsenal"},
                                "league": {"id": 39, "name": "Premier League", "season": 2026},
                                "games": {
                                    "appearences": 26,
                                    "lineups": 24,
                                    "minutes": 2100,
                                    "position": "F",
                                },
                                "goals": {"total": 12, "assists": 8},
                                "passes": {"accuracy": "82%"},
                                "shots": {"total": 55, "on": 24},
                            }
                        ],
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    standings = await provider.fetch_standings(league_id=39, season=2026)
    player_stats = await provider.fetch_player_stats(season=2026, team_id=42, league_id=39)

    assert len(standings) == 1
    assert standings[0].team_name == "Arsenal"
    assert standings[0].avg_goals_scored == pytest.approx(68 / 30)
    assert standings[0].advanced_metrics["goal_difference"] == 34.0

    assert len(player_stats) == 1
    assert player_stats[0].player_name == "Bukayo Saka"
    assert player_stats[0].appearances == 26
    assert player_stats[0].starts == 24
    assert player_stats[0].metrics["goals_total"] == 12.0
    assert player_stats[0].metrics["passes_accuracy"] == 82.0

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_injuries_and_head_to_head_normalize_provider_payloads() -> None:
    """Injury and H2H fetches should normalize the remaining API-Football domains."""

    injury_query: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/injuries":
            nonlocal injury_query
            injury_query = dict(request.url.params.items())
            return httpx.Response(
                200,
                json={
                    "errors": [],
                    "response": [
                        {
                            "player": {"id": 88, "name": "Bukayo Saka"},
                            "team": {"id": 42, "name": "Arsenal"},
                            "fixture": {"id": 501},
                            "type": "Injury",
                            "reason": "Hamstring strain",
                            "date": "2026-04-03T08:00:00+00:00",
                            "end": "2026-04-10",
                        }
                    ],
                },
                request=request,
            )

        return httpx.Response(
            200,
            json={
                "errors": [],
                "response": [
                    {
                        "fixture": {
                            "id": 7001,
                            "date": "2026-03-01T15:00:00+00:00",
                            "status": {"short": "FT"},
                            "venue": {"name": "Stamford Bridge"},
                        },
                        "league": {
                            "id": 39,
                            "name": "Premier League",
                            "country": "England",
                            "season": 2026,
                        },
                        "teams": {
                            "home": {"id": 49, "name": "Chelsea"},
                            "away": {"id": 42, "name": "Arsenal"},
                        },
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    injuries = await provider.fetch_injuries(fixture_id=501)
    h2h_fixtures = await provider.fetch_head_to_head(home_team_id=42, away_team_id=49, last=5)

    assert len(injuries) == 1
    assert injury_query == {"fixture": "501", "timezone": "Africa/Lagos"}
    assert injuries[0].fixture_ref == "api-football:501"
    assert injuries[0].player_id == "88"
    assert injuries[0].reason == "Hamstring strain"
    assert injuries[0].expected_return == date(2026, 4, 10)

    assert len(h2h_fixtures) == 1
    assert h2h_fixtures[0].status.value == "finished"
    assert h2h_fixtures[0].venue == "Stamford Bridge"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_injuries_requires_a_real_selector() -> None:
    """Injury queries should fail fast when no meaningful selector is supplied."""

    provider, client = build_provider(
        lambda request: httpx.Response(200, json={"errors": [], "response": []}, request=request)
    )

    with pytest.raises(ValueError, match="requires at least one selector"):
        await provider.fetch_injuries()

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_player_stats_requires_three_character_search() -> None:
    """Player search should enforce the provider's documented minimum length."""

    provider, client = build_provider(
        lambda request: httpx.Response(200, json={"errors": [], "response": []}, request=request)
    )

    with pytest.raises(ValueError, match="at least 3 characters"):
        await provider.fetch_player_stats(season=2026, search="ab")

    await client.aclose()
