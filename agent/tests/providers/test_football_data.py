"""Tests for PuntLab's Football-Data.org provider integration.

Purpose: verify request shaping and normalization for PuntLab's fallback
soccer provider without depending on live Football-Data.org traffic.
Scope: unit tests for fixtures, standings, and competition team metadata using
`httpx.MockTransport` and an in-memory Redis stub.
Dependencies: pytest, httpx, the shared `RedisClient`, and the concrete
`FootballDataProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import pytest
from src.cache.client import RedisClient
from src.providers.base import RateLimitedClient
from src.providers.football_data import FootballDataProvider


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
) -> tuple[FootballDataProvider, RateLimitedClient]:
    """Create a provider backed by a mock transport and in-memory Redis."""

    cache = RedisClient(redis_client=FakeAsyncRedis())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rate_limited_client = RateLimitedClient(
        cache,
        http_client=http_client,
        clock=lambda: datetime(2026, 4, 3, 7, 0, tzinfo=UTC),
    )
    provider = FootballDataProvider(rate_limited_client, api_token="test-token")
    return provider, rate_limited_client


@pytest.mark.asyncio
async def test_fetch_fixtures_by_date_normalizes_payload_and_request_shape() -> None:
    """Fixture fetches should normalize matches and use an exclusive upper bound."""

    observed_headers: dict[str, str] = {}
    observed_query: dict[str, str] = {}
    observed_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_headers, observed_query, observed_path
        observed_headers = dict(request.headers.items())
        observed_query = dict(request.url.params.items())
        observed_path = request.url.path
        return httpx.Response(
            200,
            json={
                "area": {"id": 2072, "name": "England"},
                "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
                "matches": [
                    {
                        "id": 440001,
                        "utcDate": "2026-04-03T19:00:00Z",
                        "status": "TIMED",
                        "venue": "Anfield",
                        "homeTeam": {"id": 64, "name": "Liverpool FC"},
                        "awayTeam": {"id": 57, "name": "Arsenal FC"},
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    fixtures = await provider.fetch_fixtures_by_date(
        run_date=date(2026, 4, 3),
        competition_code="pl",
        season=2026,
    )

    assert observed_headers["x-auth-token"] == "test-token"
    assert observed_path == "/v4/competitions/PL/matches"
    assert observed_query == {
        "dateFrom": "2026-04-03",
        "dateTo": "2026-04-04",
        "season": "2026",
    }
    assert len(fixtures) == 1
    assert fixtures[0].competition == "Premier League"
    assert fixtures[0].home_team == "Liverpool FC"
    assert fixtures[0].away_team_id == "57"
    assert fixtures[0].source_id == "440001"
    assert fixtures[0].status.value == "scheduled"
    assert fixtures[0].country == "England"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_standings_merges_total_home_and_away_tables() -> None:
    """Standings fetches should preserve home and away win splits in TeamStats."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
                "season": {
                    "id": 900,
                    "startDate": "2025-08-15",
                    "endDate": "2026-05-24",
                },
                "standings": [
                    {
                        "type": "TOTAL",
                        "table": [
                            {
                                "position": 1,
                                "team": {"id": 57, "name": "Arsenal FC"},
                                "playedGames": 30,
                                "form": "W,W,D,W,W",
                                "won": 21,
                                "draw": 5,
                                "lost": 4,
                                "points": 68,
                                "goalsFor": 62,
                                "goalsAgainst": 28,
                                "goalDifference": 34,
                                "lastUpdated": "2026-04-03T08:30:00Z",
                            }
                        ],
                    },
                    {
                        "type": "HOME",
                        "table": [
                            {
                                "team": {"id": 57, "name": "Arsenal FC"},
                                "won": 12,
                            }
                        ],
                    },
                    {
                        "type": "AWAY",
                        "table": [
                            {
                                "team": {"id": 57, "name": "Arsenal FC"},
                                "won": 9,
                            }
                        ],
                    },
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    standings = await provider.fetch_standings(
        competition_code="PL",
        season=2025,
        matchday=30,
    )

    assert len(standings) == 1
    assert standings[0].team_name == "Arsenal FC"
    assert standings[0].season == "2025-26"
    assert standings[0].form == "WWDWW"
    assert standings[0].home_wins == 12
    assert standings[0].away_wins == 9
    assert standings[0].avg_goals_scored == pytest.approx(62 / 30)
    assert standings[0].avg_goals_conceded == pytest.approx(28 / 30)
    assert standings[0].advanced_metrics["goal_difference"] == 34.0
    assert standings[0].fetched_at == datetime(2026, 4, 3, 8, 30, tzinfo=UTC)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_teams_normalizes_competition_team_metadata() -> None:
    """Competition team fetches should return minimal canonical team snapshots."""

    observed_query: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_query
        observed_query = dict(request.url.params.items())
        return httpx.Response(
            200,
            json={
                "competition": {"id": 2014, "name": "La Liga", "code": "PD"},
                "season": {
                    "id": 901,
                    "startDate": "2025-08-17",
                    "endDate": "2026-05-25",
                },
                "teams": [
                    {
                        "id": 86,
                        "name": "Real Madrid CF",
                        "lastUpdated": "2026-04-03T10:15:00Z",
                    }
                ],
            },
            request=request,
        )

    provider, client = build_provider(handler)

    teams = await provider.fetch_teams(competition_code="PD", season=2025)

    assert observed_query == {"season": "2025"}
    assert len(teams) == 1
    assert teams[0].team_id == "86"
    assert teams[0].team_name == "Real Madrid CF"
    assert teams[0].competition == "La Liga"
    assert teams[0].season == "2025-26"
    assert teams[0].matches_played == 0
    assert teams[0].fetched_at == datetime(2026, 4, 3, 10, 15, tzinfo=UTC)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_fixtures_rejects_blank_competition_codes() -> None:
    """Competition code validation should fail fast before any network request."""

    provider, client = build_provider(
        lambda request: httpx.Response(200, json={"matches": []}, request=request)
    )

    with pytest.raises(ValueError, match="competition_code"):
        await provider.fetch_fixtures_by_date(
            run_date=date(2026, 4, 3),
            competition_code=" ",
        )

    await client.aclose()
