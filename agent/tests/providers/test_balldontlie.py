"""Tests for PuntLab's BALLDONTLIE provider integration.

Purpose: verify request shaping, cursor pagination, normalization, and
fail-fast diagnostics for the NBA provider without depending on live upstream
traffic.
Scope: unit tests for NBA games, players, season averages, team season
averages, and box-score tier errors using `httpx.MockTransport`.
Dependencies: pytest, httpx, the shared `RedisClient`, and the concrete
`BallDontLieProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE
from src.providers.balldontlie import BallDontLieProvider
from src.providers.base import ProviderError, RateLimitedClient


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
) -> tuple[BallDontLieProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 4, 7, 0, tzinfo=WAT_TIMEZONE),
    )
    provider = BallDontLieProvider(rate_limited_client, api_key="test-key")
    return provider, rate_limited_client


@pytest.mark.asyncio
async def test_fetch_games_by_date_normalizes_fixture_payload() -> None:
    """Game fetches should normalize NBA games into canonical fixtures."""

    observed_headers: dict[str, str] = {}
    observed_path = ""
    observed_query: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_headers, observed_path, observed_query
        observed_headers = dict(request.headers.items())
        observed_path = request.url.path
        observed_query = list(request.url.params.multi_items())
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 15907925,
                        "date": "2026-04-04",
                        "season": 2025,
                        "status": "Final",
                        "period": 4,
                        "time": "Final",
                        "postseason": False,
                        "postponed": False,
                        "home_team_score": 115,
                        "visitor_team_score": 105,
                        "datetime": "2026-04-04T23:00:00Z",
                        "home_team": {
                            "id": 6,
                            "conference": "East",
                            "division": "Central",
                            "city": "Cleveland",
                            "name": "Cavaliers",
                            "full_name": "Cleveland Cavaliers",
                            "abbreviation": "CLE",
                        },
                        "visitor_team": {
                            "id": 4,
                            "conference": "East",
                            "division": "Southeast",
                            "city": "Charlotte",
                            "name": "Hornets",
                            "full_name": "Charlotte Hornets",
                            "abbreviation": "CHA",
                        },
                    }
                ],
                "meta": {"per_page": 100},
            },
            request=request,
        )

    provider, client = build_provider(handler)

    fixtures = await provider.fetch_games_by_date(
        run_date=date(2026, 4, 4),
        team_ids=[6],
        seasons=[2025],
        postseason=False,
    )

    assert observed_headers["authorization"] == "test-key"
    assert observed_path == "/nba/v1/games"
    assert observed_query == [
        ("dates[]", "2026-04-04"),
        ("team_ids[]", "6"),
        ("seasons[]", "2025"),
        ("postseason", "false"),
        ("per_page", "100"),
    ]
    assert len(fixtures) == 1
    assert fixtures[0].home_team == "Cleveland Cavaliers"
    assert fixtures[0].away_team == "Charlotte Hornets"
    assert fixtures[0].competition == "NBA"
    assert fixtures[0].sport.value == "basketball"
    assert fixtures[0].status.value == "finished"
    assert fixtures[0].source_id == "15907925"
    assert fixtures[0].kickoff == datetime(2026, 4, 4, 23, 0, tzinfo=UTC)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_players_paginates_and_normalizes_roster_rows() -> None:
    """Player fetches should follow cursors and return canonical player rows."""

    observed_requests: list[list[tuple[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(list(request.url.params.multi_items()))
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": 115,
                            "first_name": "Stephen",
                            "last_name": "Curry",
                            "position": "G",
                            "team": {
                                "id": 10,
                                "conference": "West",
                                "division": "Pacific",
                                "city": "Golden State",
                                "name": "Warriors",
                                "full_name": "Golden State Warriors",
                                "abbreviation": "GSW",
                            },
                        }
                    ],
                    "meta": {"next_cursor": 25, "per_page": 100},
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 237,
                        "first_name": "LeBron",
                        "last_name": "James",
                        "position": "F",
                        "team": {
                            "id": 14,
                            "conference": "West",
                            "division": "Pacific",
                            "city": "Los Angeles",
                            "name": "Lakers",
                            "full_name": "Los Angeles Lakers",
                            "abbreviation": "LAL",
                        },
                    }
                ],
                "meta": {"per_page": 100},
            },
            request=request,
        )

    provider, client = build_provider(handler)

    players = await provider.fetch_players(search="James", max_pages=2)

    assert len(observed_requests) == 2
    assert observed_requests[0] == [("search", "James"), ("per_page", "100")]
    assert observed_requests[1] == [
        ("search", "James"),
        ("per_page", "100"),
        ("cursor", "25"),
    ]
    assert [player.player_name for player in players] == ["Stephen Curry", "LeBron James"]
    assert players[0].team_id == "10"
    assert players[0].team_name == "Golden State Warriors"
    assert players[1].position == "F"
    assert players[1].competition == "NBA"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_season_averages_normalizes_player_metrics() -> None:
    """Season averages should preserve provider metrics in canonical player rows."""

    observed_path = ""
    observed_query: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_path, observed_query
        observed_path = request.url.path
        observed_query = list(request.url.params.multi_items())
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "player": {
                            "id": 246,
                            "first_name": "Nikola",
                            "last_name": "Jokic",
                            "position": "C",
                            "team_id": 8,
                        },
                        "season": 2025,
                        "season_type": "regular",
                        "stats": {
                            "gp": 70,
                            "min": 36.2,
                            "pts": 28.7,
                            "reb": 12.4,
                            "ast": 10.1,
                            "5-9_ft._fg_pct": 0.571,
                        },
                    }
                ],
                "meta": {"per_page": 100},
            },
            request=request,
        )

    provider, client = build_provider(handler)

    averages = await provider.fetch_season_averages(
        season=2025,
        average_type="shooting",
        stats_type="5ft_range",
        player_ids=[246],
    )

    assert observed_path == "/nba/v1/season_averages/shooting"
    assert observed_query == [
        ("player_ids[]", "246"),
        ("season", "2025"),
        ("season_type", "regular"),
        ("type", "5ft_range"),
        ("per_page", "100"),
    ]
    assert len(averages) == 1
    assert averages[0].player_name == "Nikola Jokic"
    assert averages[0].team_id == "8"
    assert averages[0].season == "2025-26"
    assert averages[0].appearances == 70
    assert averages[0].minutes_played == 2534
    assert averages[0].metrics["pts"] == pytest.approx(28.7)
    assert averages[0].metrics["5-9_ft._fg_pct"] == pytest.approx(0.571)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_team_season_averages_normalizes_team_stats() -> None:
    """Team season averages should map NBA team metrics into `TeamStats`."""

    observed_path = ""
    observed_query: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_path, observed_query
        observed_path = request.url.path
        observed_query = list(request.url.params.multi_items())
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "team": {
                            "id": 2,
                            "conference": "East",
                            "division": "Atlantic",
                            "city": "Boston",
                            "name": "Celtics",
                            "full_name": "Boston Celtics",
                            "abbreviation": "BOS",
                        },
                        "season": 2025,
                        "season_type": "regular",
                        "stats": {
                            "gp": 82,
                            "w": 61,
                            "l": 21,
                            "pts": 116.3,
                            "plus_minus": 9.1,
                            "fg_pct": 0.462,
                        },
                    }
                ],
                "meta": {"per_page": 100},
            },
            request=request,
        )

    provider, client = build_provider(handler)

    teams = await provider.fetch_team_season_averages(
        season=2025,
        category="general",
        stats_type="base",
        team_ids=[2],
    )

    assert observed_path == "/nba/v1/team_season_averages/general"
    assert observed_query == [
        ("team_ids[]", "2"),
        ("season", "2025"),
        ("season_type", "regular"),
        ("type", "base"),
        ("per_page", "100"),
    ]
    assert len(teams) == 1
    assert teams[0].team_name == "Boston Celtics"
    assert teams[0].season == "2025-26"
    assert teams[0].matches_played == 82
    assert teams[0].wins == 61
    assert teams[0].losses == 21
    assert teams[0].avg_goals_scored == pytest.approx(116.3)
    assert teams[0].goals_for == 9537
    assert teams[0].advanced_metrics["plus_minus"] == pytest.approx(9.1)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_box_scores_surfaces_tier_guidance_for_unauthorized_keys() -> None:
    """Tier-gated endpoints should return explicit recovery guidance on `401`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized", request=request)

    provider, client = build_provider(handler)

    with pytest.raises(ProviderError, match="BOX SCORES access"):
        await provider.fetch_box_scores_by_date(run_date=date(2026, 4, 4))

    await client.aclose()
