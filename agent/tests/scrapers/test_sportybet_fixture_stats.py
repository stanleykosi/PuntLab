"""Tests for the SportyBet fixture stats fetcher helpers.

Purpose: verify the stable URL/id extraction and markdown rendering helpers
used by the live Playwright fixture stats scraper.
Scope: pure helper coverage for `src.scrapers.sportybet_fixture_stats`.
Dependencies: pytest plus the public helper functions under test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.scrapers.sportybet_fixture_stats import (
    SportyBetFixtureStatsResponse,
    SportyBetFixtureStatsResult,
    SportyBetFixtureStatsScraper,
    SportyBetFixtureStatsWidget,
    _build_sportradar_proxy_response_headers,
    _build_statistics_fallback_lines,
    _build_team_info_lines,
    _build_team_stats_from_captured_responses,
    _CapturedResponse,
    _is_probably_binary_content_type,
    extract_event_id_from_fixture_url,
    extract_match_id_from_event_id,
    render_fixture_stats_markdown,
)


def test_extract_event_id_from_fixture_url_returns_canonical_segment() -> None:
    """Fixture URLs should yield the `sr:match:*` segment unchanged."""

    fixture_url = (
        "https://www.sportybet.com/ng/sport/football/slovenia/prvaliga/"
        "Bravo_Ljubljana_vs_Primorje_Ajdovscina/sr:match:61454137"
    )

    assert extract_event_id_from_fixture_url(fixture_url) == "sr:match:61454137"


def test_extract_match_id_from_event_id_returns_numeric_suffix() -> None:
    """Widget access should use the numeric suffix from the event id."""

    assert extract_match_id_from_event_id("sr:match:61454137") == "61454137"


def test_scraper_rejects_unknown_widget_key() -> None:
    """The scraper should fail fast for unsupported widget names."""

    with pytest.raises(ValueError, match="Unsupported widget_keys"):
        SportyBetFixtureStatsScraper(widget_keys=("statistics", "unknown"))


def test_render_fixture_stats_markdown_includes_widget_and_response_sections() -> None:
    """Markdown rendering should surface the mounted widget text and response URLs."""

    result = SportyBetFixtureStatsResult(
        fixture_url=(
            "https://www.sportybet.com/ng/sport/football/slovenia/prvaliga/"
            "Bravo_Ljubljana_vs_Primorje_Ajdovscina/sr:match:61454137"
        ),
        final_url=(
            "https://www.sportybet.com/ng/sport/football/slovenia/prvaliga/"
            "Bravo_Ljubljana_vs_Primorje_Ajdovscina/sr:match:61454137"
        ),
        event_id="sr:match:61454137",
        match_id="61454137",
        page_title="Bravo Ljubljana vs Primorje Ajdovscina",
        fetched_at=datetime(2026, 4, 15, 19, 0, tzinfo=UTC),
        widget_loader_status="loaded",
        widgets=(
            SportyBetFixtureStatsWidget(
                widget_key="statistics",
                widget_type="match.statistics",
                status="mounted",
                headings=("Statistics",),
                content_text="Yellow cards 2 1",
                content_lines=("Yellow cards 2 1", "Red cards 0 1"),
                response_urls=("https://widgets.sir.sportradar.com/example/statistics",),
            ),
        ),
        responses=(
            SportyBetFixtureStatsResponse(
                url="https://widgets.sir.sportradar.com/example/statistics",
                path="/example/statistics",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text='{"cards":{"yellow":[2,1]}}',
            ),
        ),
    )

    rendered = render_fixture_stats_markdown(result)

    assert "## statistics" in rendered
    assert "Yellow cards 2 1" in rendered
    assert "Red cards 0 1" in rendered
    assert "## Network Responses" in rendered
    assert "https://widgets.sir.sportradar.com/example/statistics" in rendered


def test_build_sportradar_proxy_response_headers_sets_browser_safe_cors() -> None:
    """Proxied Sportradar responses should be readable from the SportyBet page."""

    headers = _build_sportradar_proxy_response_headers(
        {
            "content-type": "application/json",
            "content-length": "123",
            "vary": "Accept-Encoding",
        }
    )

    assert headers["content-type"] == "application/json"
    assert headers["access-control-allow-origin"] == "https://www.sportybet.com"
    assert headers["access-control-allow-credentials"] == "true"
    assert headers["vary"] == "Origin"
    assert "content-length" not in headers


def test_is_probably_binary_content_type_distinguishes_images_from_text() -> None:
    """Binary asset content types should bypass UTF-8 body decoding."""

    assert _is_probably_binary_content_type("image/png") is True
    assert _is_probably_binary_content_type("font/woff2") is True
    assert _is_probably_binary_content_type("application/javascript") is False


def test_build_statistics_fallback_lines_uses_sportradar_json_payloads() -> None:
    """Statistics fallback should surface readable match stats from captured JSON."""

    captured = (
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/match_info/61454137",
                path="/common/en/Etc:UTC/gismo/match_info/61454137",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "match": {
                                "result": {"home": 2, "away": 1},
                                "cards": {
                                    "home": {"yellow_count": 2, "red_count": 0},
                                    "away": {"yellow_count": 3, "red_count": 0},
                                },
                                "teams": {
                                    "home": {"name": "Bravo Ljubljana"},
                                    "away": {"name": "Primorje Ajdovscina"},
                                },
                            }
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/match_details/61454137",
                path="/common/en/Etc:UTC/gismo/match_details/61454137",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "values": {
                                "110": {
                                    "name": "Ball possession",
                                    "value": {"home": 60, "away": 40},
                                },
                                "124": {
                                    "name": "Corner kicks",
                                    "value": {"home": 7, "away": 3},
                                },
                            }
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/match_timeline/61454137",
                path="/common/en/Etc:UTC/gismo/match_timeline/61454137",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "events": [
                                {"type": "goal", "team": "away", "time": 12, "injurytime": 0},
                                {"type": "goal", "team": "home", "time": 72, "injurytime": 0},
                            ]
                        }
                    }
                ]
            },
        ),
    )

    lines = _build_statistics_fallback_lines(captured)

    assert "Result: Bravo Ljubljana 2 - 1 Primorje Ajdovscina" in lines
    assert "Ball possession: Bravo Ljubljana 60% | Primorje Ajdovscina 40%" in lines
    assert "Corner kicks: Bravo Ljubljana 7 | Primorje Ajdovscina 3" in lines
    assert "Yellow cards: Bravo Ljubljana 2 | Primorje Ajdovscina 3" in lines
    assert "Goals timeline: 12' Primorje Ajdovscina; 72' Bravo Ljubljana" in lines


def test_build_team_info_lines_uses_team_scoped_sportradar_payloads() -> None:
    """Team info fallback should replace the broken generic widget text."""

    captured = (
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/match_info/61454137",
                path="/common/en/Etc:UTC/gismo/match_info/61454137",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "match": {
                                "teams": {
                                    "home": {
                                        "name": "Bravo Ljubljana",
                                        "uid": 362744,
                                    },
                                    "away": {
                                        "name": "Primorje Ajdovscina",
                                        "uid": 2414,
                                    },
                                }
                            }
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/stats_team_info/362744",
                path="/common/en/Etc:UTC/gismo/stats_team_info/362744",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "team": {
                                "_id": 362744,
                                "abbr": "BRA",
                                "name": "Bravo Ljubljana",
                            },
                            "manager": {
                                "name": "Arnol, Ales",
                                "membersince": {"date": "01/03/23"},
                                "nationality": {"name": "Slovenia"},
                            },
                            "stadium": {
                                "name": "ZSD Ljubljana Stadium",
                                "city": "Ljubljana",
                                "country": "Slovenia",
                                "capacity": "2308",
                            },
                            "homejersey": {"base": "ffea00"},
                            "awayjersey": {"base": "21416e"},
                            "gkjersey": {"base": "8a1e1e"},
                            "tournaments": [
                                {
                                    "name": "PrvaLiga",
                                    "year": "25/26",
                                    "seasontypename": "Regular Season",
                                }
                            ],
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/stats_team_info/2414",
                path="/common/en/Etc:UTC/gismo/stats_team_info/2414",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "team": {
                                "_id": 2414,
                                "abbr": "NDP",
                                "name": "Primorje Ajdovscina",
                            },
                            "manager": {"name": "Zlogar, Anton"},
                            "stadium": {"name": "Ajdovscina Stadium"},
                        }
                    }
                ]
            },
        ),
    )

    lines = _build_team_info_lines(captured)

    assert "Home team: Bravo Ljubljana (abbr BRA, uid 362744)" in lines
    assert "Home manager: Arnol, Ales (Slovenia, member since 01/03/23)" in lines
    assert (
        "Home stadium: ZSD Ljubljana Stadium "
        "(Ljubljana, Slovenia, capacity 2308)"
    ) in lines
    assert "Home competitions: PrvaLiga 25/26 Regular Season" in lines
    assert "Home kits: home #ffea00, away #21416e, goalkeeper #8a1e1e" in lines
    assert "Away team: Primorje Ajdovscina (abbr NDP, uid 2414)" in lines
    assert "Away manager: Zlogar, Anton" in lines


def test_build_team_stats_from_captured_responses_derives_canonical_snapshots() -> None:
    """Structured team stats should be derived from SportyBet table and form feeds."""

    fetched_at = datetime(2026, 4, 23, 12, 0, tzinfo=UTC)
    captured = (
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/match_info/62384184",
                path="/common/en/Etc:UTC/gismo/match_info/62384184",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "match": {
                                "teams": {
                                    "home": {"name": "Aswan", "uid": 230486},
                                    "away": {"name": "Masar", "uid": 1085298},
                                }
                            },
                            "season": {"year": "25/26"},
                            "tournament": {"name": "2. Division A"},
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/season_dynamictable/132916",
                path="/common/en/Etc:UTC/gismo/season_dynamictable/132916",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "season": {
                                "tables": [
                                    {
                                        "rows": [
                                            {
                                                "_doc": "tablerow",
                                                "team": {"name": "Aswan", "uid": 230486},
                                                "total": 30,
                                                "winTotal": 5,
                                                "drawTotal": 10,
                                                "lossTotal": 15,
                                                "goalsForTotal": 14,
                                                "goalsAgainstTotal": 32,
                                                "pointsTotal": 25,
                                                "pos": 17,
                                                "winHome": 3,
                                                "winAway": 2,
                                            },
                                            {
                                                "_doc": "tablerow",
                                                "team": {"name": "Masar", "uid": 1085298},
                                                "total": 30,
                                                "winTotal": 13,
                                                "drawTotal": 9,
                                                "lossTotal": 8,
                                                "goalsForTotal": 40,
                                                "goalsAgainstTotal": 25,
                                                "pointsTotal": 48,
                                                "pos": 4,
                                                "winHome": 4,
                                                "winAway": 9,
                                            },
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                ]
            },
        ),
        _CapturedResponse(
            summary=SportyBetFixtureStatsResponse(
                url="https://widgets.fn.sportradar.com/common/en/Etc:UTC/gismo/stats_formtable/132916",
                path="/common/en/Etc:UTC/gismo/stats_formtable/132916",
                status=200,
                content_type="application/json",
                body_kind="json",
                preview_text=None,
            ),
            payload={
                "doc": [
                    {
                        "data": {
                            "teams": [
                                {
                                    "team": {"name": "Aswan", "uid": 230486},
                                    "form": {
                                        "total": [
                                            {"typeid": "L"},
                                            {"typeid": "D"},
                                            {"typeid": "L"},
                                            {"typeid": "D"},
                                            {"typeid": "L"},
                                            {"typeid": "W"},
                                        ]
                                    },
                                },
                                {
                                    "team": {"name": "Masar", "uid": 1085298},
                                    "form": {
                                        "total": [
                                            {"typeid": "L"},
                                            {"typeid": "W"},
                                            {"typeid": "L"},
                                            {"typeid": "W"},
                                            {"typeid": "W"},
                                            {"typeid": "L"},
                                        ]
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
        ),
    )

    team_stats = _build_team_stats_from_captured_responses(
        captured,
        fetched_at=fetched_at,
    )

    assert len(team_stats) == 2
    home_stats = next(stats for stats in team_stats if stats.team_id == "230486")
    away_stats = next(stats for stats in team_stats if stats.team_id == "1085298")

    assert home_stats.team_name == "Aswan"
    assert home_stats.competition == "2. Division A"
    assert home_stats.season == "25/26"
    assert home_stats.matches_played == 30
    assert home_stats.wins == 5
    assert home_stats.home_wins == 3
    assert home_stats.away_wins == 2
    assert home_stats.form == "WLDLDL"
    assert home_stats.avg_goals_scored == pytest.approx(14 / 30)
    assert home_stats.avg_goals_conceded == pytest.approx(32 / 30)

    assert away_stats.team_name == "Masar"
    assert away_stats.position == 4
    assert away_stats.points == 48
    assert away_stats.form == "LWWLWL"
