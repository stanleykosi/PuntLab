"""Tests for PuntLab's odds-value scoring factor.

Purpose: verify that the odds-value factor detects strong price edges from
cross-book consensus while staying conservative on aligned or unsupported
markets.
Scope: unit tests for `src.scoring.factors.odds_value`.
Dependencies: pytest, the canonical odds schema, and shared market enums.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.schemas.odds import NormalizedOdds
from src.scoring.factors.odds_value import analyze_odds_value


def build_match_result_row(
    *,
    provider: str,
    selection_name: str,
    odds: float,
) -> NormalizedOdds:
    """Build one lossless match-result odds row for factor tests."""

    return NormalizedOdds(
        fixture_ref="fixture-odds-1",
        market=None,
        selection=selection_name,
        odds=odds,
        provider=provider,
        provider_market_name="Full Time Result",
        provider_selection_name=selection_name,
        provider_market_id="h2h",
        period="match",
        participant_scope="match",
        raw_metadata={
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        },
        last_updated=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
    )


def test_analyze_odds_value_detects_a_strong_consensus_edge() -> None:
    """A market outlier with a much better home price should score as strong value."""

    score = analyze_odds_value(
        (
            build_match_result_row(provider="Pinnacle", selection_name="Arsenal", odds=1.45),
            build_match_result_row(provider="Pinnacle", selection_name="Draw", odds=4.50),
            build_match_result_row(provider="Pinnacle", selection_name="Chelsea", odds=8.00),
            build_match_result_row(provider="Bet365", selection_name="Arsenal", odds=1.50),
            build_match_result_row(provider="Bet365", selection_name="Draw", odds=4.40),
            build_match_result_row(provider="Bet365", selection_name="Chelsea", odds=7.20),
            build_match_result_row(provider="SportyBet", selection_name="Arsenal", odds=2.20),
            build_match_result_row(provider="SportyBet", selection_name="Draw", odds=5.50),
            build_match_result_row(provider="SportyBet", selection_name="Chelsea", odds=9.00),
        )
    )

    assert score == pytest.approx(0.90)


def test_analyze_odds_value_stays_conservative_when_books_are_aligned() -> None:
    """Closely aligned market prices should not be mistaken for strong value."""

    score = analyze_odds_value(
        (
            build_match_result_row(provider="Pinnacle", selection_name="Arsenal", odds=1.79),
            build_match_result_row(provider="Pinnacle", selection_name="Draw", odds=3.65),
            build_match_result_row(provider="Pinnacle", selection_name="Chelsea", odds=4.70),
            build_match_result_row(provider="Bet365", selection_name="Arsenal", odds=1.81),
            build_match_result_row(provider="Bet365", selection_name="Draw", odds=3.60),
            build_match_result_row(provider="Bet365", selection_name="Chelsea", odds=4.65),
            build_match_result_row(provider="SportyBet", selection_name="Arsenal", odds=1.82),
            build_match_result_row(provider="SportyBet", selection_name="Draw", odds=3.55),
            build_match_result_row(provider="SportyBet", selection_name="Chelsea", odds=4.60),
        )
    )

    assert score == pytest.approx(0.10)


def test_analyze_odds_value_rejects_empty_input_and_ignores_unscoreable_rows() -> None:
    """The factor should fail fast on missing odds and stay conservative on unsupported rows."""

    with pytest.raises(ValueError, match="at least one NormalizedOdds"):
        analyze_odds_value(())

    unsupported_score = analyze_odds_value(
        (
            NormalizedOdds(
                fixture_ref="fixture-odds-1",
                market=None,
                selection="Arsenal Over 1.5",
                odds=2.10,
                provider="Pinnacle",
                provider_market_name="Team Totals",
                provider_selection_name="Arsenal Over 1.5",
                provider_market_id="team_totals",
                line=1.5,
                period="match",
                participant_scope="team",
                raw_metadata={
                    "sport_key": "soccer_epl",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                },
                last_updated=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            ),
        )
    )

    assert unsupported_score == pytest.approx(0.10)
