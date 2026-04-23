"""Tests for SportyBet fixture-page probe helpers.

Purpose: verify the pure URL-building and keyword-discovery helpers used by
the Playwright fixture probe without opening a real browser session.
Scope: unit coverage for `src.scrapers.sportybet_fixture_probe`.
Dependencies: pytest plus the pure helper functions under test.
"""

from __future__ import annotations

from src.scrapers.sportybet_fixture_probe import (
    _collect_keyword_paths,
    build_fixture_page_url,
    extract_match_id_from_event_id,
    extract_match_id_from_fixture_url,
)


def test_build_fixture_page_url_normalizes_fixture_metadata() -> None:
    """URL construction should match SportyBet's public fixture-page shape."""

    url = build_fixture_page_url(
        event_id="sr:match:61454137",
        home_team="Bravo Ljubljana",
        away_team="Primorje Ajdovscina",
        country="Slovenia",
        competition="PrvaLiga",
    )

    assert url == (
        "https://www.sportybet.com/ng/sport/football/slovenia/prvaliga/"
        "Bravo_Ljubljana_vs_Primorje_Ajdovscina/sr:match:61454137"
    )


def test_collect_keyword_paths_finds_nested_detail_and_card_fields() -> None:
    """Keyword discovery should surface nested stat, card, and result fields."""

    payload = {
        "bizCode": 10000,
        "data": {
            "teamStats": {
                "yellowCardsAverage": 2.1,
                "redCardsAverage": 0.2,
            },
            "historicalResults": {
                "winRate": 0.6,
                "drawRate": 0.25,
            },
            "playerBreakdown": [
                {"playerName": "Jane Doe", "yellowCardCount": 4},
            ],
        },
    }

    matched_paths = _collect_keyword_paths(
        payload,
        keywords=("yellow", "red", "win", "draw", "player", "stat"),
    )

    assert "data.teamStats" in matched_paths
    assert "data.teamStats.yellowCardsAverage" in matched_paths
    assert "data.teamStats.redCardsAverage" in matched_paths
    assert "data.historicalResults.winRate" in matched_paths
    assert "data.historicalResults.drawRate" in matched_paths
    assert "data.playerBreakdown" in matched_paths
    assert "data.playerBreakdown[0].playerName" in matched_paths


def test_extract_match_id_from_event_id_returns_numeric_suffix() -> None:
    """Fixture widget mounting should use the numeric segment from `sr:match:*`."""

    assert extract_match_id_from_event_id("sr:match:61454137") == "61454137"


def test_extract_match_id_from_fixture_url_reads_event_segment() -> None:
    """Fixture-page URLs should yield the same numeric match id for widget access."""

    fixture_url = (
        "https://www.sportybet.com/ng/sport/football/slovenia/prvaliga/"
        "Bravo_Ljubljana_vs_Primorje_Ajdovscina/sr:match:61454137"
    )

    assert extract_match_id_from_fixture_url(fixture_url) == "61454137"
