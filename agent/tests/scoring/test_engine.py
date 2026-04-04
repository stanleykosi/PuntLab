"""Tests for PuntLab's composite scoring engine and weight helpers.

Purpose: verify that the scoring engine combines factor outputs into a stable
composite score, chooses coherent markets from scoreable odds, and falls back
cleanly when recommendation odds are unavailable.
Scope: unit tests for `src.scoring.engine` and `src.scoring.weights`.
Dependencies: pytest plus canonical fixture, odds, stats, and context schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import MarketType, SportName
from src.schemas.analysis import MatchContext
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import TeamStats
from src.scoring.engine import ScoringEngine
from src.scoring.weights import DEFAULT_SCORING_WEIGHTS, ScoringWeights, get_default_scoring_weights


def build_fixture(
    *,
    sport: SportName = SportName.SOCCER,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
) -> NormalizedFixture:
    """Build a canonical fixture for scoring-engine tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team=home_team,
        away_team=away_team,
        competition="Premier League" if sport == SportName.SOCCER else "NBA",
        sport=sport,
        kickoff=datetime(2026, 4, 4, 17, 30, tzinfo=UTC),
        source_provider="test-suite",
        source_id="fixture-61301159",
        country="England" if sport == SportName.SOCCER else "USA",
        home_team_id=home_team.lower().replace(" ", "-"),
        away_team_id=away_team.lower().replace(" ", "-"),
        venue="Emirates Stadium" if sport == SportName.SOCCER else "Staples Center",
    )


def build_team_stats(
    *,
    team_id: str,
    team_name: str,
    sport: SportName = SportName.SOCCER,
    matches_played: int = 10,
    wins: int = 5,
    draws: int = 2,
    losses: int = 3,
    goals_for: int = 16,
    goals_against: int = 12,
    home_wins: int = 3,
    away_wins: int = 2,
    form: str | None = "WWDDL",
    avg_goals_scored: float | None = 1.6,
    avg_goals_conceded: float | None = 1.2,
    advanced_metrics: dict[str, float] | None = None,
) -> TeamStats:
    """Build one canonical team-stat snapshot for the engine tests."""

    return TeamStats(
        team_id=team_id,
        team_name=team_name,
        sport=sport,
        source_provider="test-suite",
        fetched_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        competition="Premier League" if sport == SportName.SOCCER else "NBA",
        season="2025-26",
        matches_played=matches_played,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        clean_sheets=3 if sport == SportName.SOCCER else 0,
        form=form,
        points=(wins * 3) + draws if sport == SportName.SOCCER else wins * 2,
        home_wins=home_wins,
        away_wins=away_wins,
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
        advanced_metrics=advanced_metrics or {},
    )


def build_odds_row(
    *,
    fixture_ref: str,
    provider: str,
    provider_market_name: str,
    provider_selection_name: str,
    odds: float,
    market: MarketType | None = None,
    selection: str | None = None,
    line: float | None = None,
    sport_key: str = "soccer_epl",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    participant_scope: str = "match",
    period: str = "match",
) -> NormalizedOdds:
    """Build one normalized odds row for the scoring engine tests."""

    return NormalizedOdds(
        fixture_ref=fixture_ref,
        market=market,
        selection=selection or provider_selection_name,
        odds=odds,
        provider=provider,
        provider_market_name=provider_market_name,
        provider_selection_name=provider_selection_name,
        provider_market_id=provider_market_name.lower().replace(" ", "_"),
        line=line,
        period=period,
        participant_scope=participant_scope,
        raw_metadata={
            "sport_key": sport_key,
            "home_team": home_team,
            "away_team": away_team,
        },
        last_updated=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
    )


def build_context(
    *,
    qualitative_score: float = 0.74,
    morale_home: float = 0.82,
    morale_away: float = 0.46,
    pressure_home: float = 0.40,
    pressure_away: float = 0.68,
    rivalry_factor: float = 0.72,
) -> MatchContext:
    """Build a canonical qualitative context object for engine tests."""

    return MatchContext(
        fixture_ref="sr:match:61301159",
        morale_home=morale_home,
        morale_away=morale_away,
        rivalry_factor=rivalry_factor,
        pressure_home=pressure_home,
        pressure_away=pressure_away,
        key_narrative="The home side arrives with stronger momentum and less pressure.",
        qualitative_score=qualitative_score,
        data_sources=("BBC Sport", "Tavily"),
        news_summary="Home morale is materially stronger heading into the fixture.",
    )


def test_get_default_scoring_weights_returns_independent_copies() -> None:
    """Default-weight helpers should expose fresh copies of the canonical model."""

    first = get_default_scoring_weights()
    second = get_default_scoring_weights()

    assert isinstance(first, ScoringWeights)
    assert first == DEFAULT_SCORING_WEIGHTS
    assert first is not second


def test_scoring_engine_selects_home_match_result_for_strong_home_edge() -> None:
    """A strong home edge with canonical 1X2 odds should recommend the home side."""

    fixture = build_fixture()
    stats = (
        build_team_stats(
            team_id="arsenal",
            team_name="Arsenal",
            wins=8,
            draws=1,
            losses=1,
            goals_for=25,
            goals_against=9,
            home_wins=5,
            away_wins=3,
            form="WWWWDWWLWW",
            avg_goals_scored=2.5,
            avg_goals_conceded=0.9,
            advanced_metrics={"xg_diff": 0.9, "elo": 1640},
        ),
        build_team_stats(
            team_id="chelsea",
            team_name="Chelsea",
            wins=3,
            draws=2,
            losses=5,
            goals_for=12,
            goals_against=18,
            home_wins=2,
            away_wins=1,
            form="LDLLWDLLDL",
            avg_goals_scored=1.2,
            avg_goals_conceded=1.8,
            advanced_metrics={"xg_diff": -0.4, "elo": 1485},
        ),
    )
    odds = (
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Arsenal",
            odds=1.62,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Draw",
            odds=3.90,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Chelsea",
            odds=5.20,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Full Time Result",
            provider_selection_name="Arsenal",
            odds=1.64,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Full Time Result",
            provider_selection_name="Draw",
            odds=3.80,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Full Time Result",
            provider_selection_name="Chelsea",
            odds=5.10,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Pinnacle",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Over 2.5",
            odds=1.75,
            line=2.5,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Pinnacle",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Under 2.5",
            odds=2.05,
            line=2.5,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Over 2.5",
            odds=1.78,
            line=2.5,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Under 2.5",
            odds=2.02,
            line=2.5,
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Team Totals",
            provider_selection_name="Arsenal Over 1.5",
            odds=2.05,
            line=1.5,
            participant_scope="team",
        ),
    )

    engine = ScoringEngine()
    score = engine.calculate_match_score(
        fixture,
        stats,
        odds,
        context=build_context(),
        injuries=(),
        h2h_data=(),
    )

    assert score.recommended_market == MarketType.MATCH_RESULT
    assert score.recommended_selection == "home"
    assert score.recommended_odds == pytest.approx(1.64)
    assert score.composite_score > 0.55
    assert score.confidence > 0.60
    assert (
        score.qualitative_summary
        == "Home morale is materially stronger heading into the fixture."
    )


def test_scoring_engine_prefers_totals_market_when_only_totals_are_scoreable() -> None:
    """High-total fixtures should recommend an over when totals are the scoreable options."""

    fixture = build_fixture(
        sport=SportName.BASKETBALL,
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
    )
    stats = (
        build_team_stats(
            team_id="los-angeles-lakers",
            team_name="Los Angeles Lakers",
            sport=SportName.BASKETBALL,
            wins=7,
            draws=0,
            losses=3,
            goals_for=1185,
            goals_against=1124,
            home_wins=4,
            away_wins=3,
            form="WWLWWWWLWW",
            avg_goals_scored=118.5,
            avg_goals_conceded=112.4,
            advanced_metrics={"net_rating": 6.8, "pace": 101.4},
        ),
        build_team_stats(
            team_id="boston-celtics",
            team_name="Boston Celtics",
            sport=SportName.BASKETBALL,
            wins=8,
            draws=0,
            losses=2,
            goals_for=1196,
            goals_against=1111,
            home_wins=5,
            away_wins=3,
            form="WWWWLWWWLW",
            avg_goals_scored=119.6,
            avg_goals_conceded=111.1,
            advanced_metrics={"net_rating": 8.2, "pace": 100.8},
        ),
    )
    odds = (
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="DraftKings",
            provider_market_name="Totals",
            provider_selection_name="Over",
            odds=1.94,
            line=228.5,
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="DraftKings",
            provider_market_name="Totals",
            provider_selection_name="Under",
            odds=1.88,
            line=228.5,
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="FanDuel",
            provider_market_name="Totals",
            provider_selection_name="Over",
            odds=1.96,
            line=228.5,
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
        ),
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="FanDuel",
            provider_market_name="Totals",
            provider_selection_name="Under",
            odds=1.86,
            line=228.5,
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
        ),
    )

    engine = ScoringEngine()
    score = engine.calculate_match_score(fixture, stats, odds, injuries=(), h2h_data=None)

    assert score.recommended_market == MarketType.TOTAL_POINTS
    assert score.recommended_selection == "over"
    assert score.recommended_odds == pytest.approx(1.96)
    assert score.confidence > 0.45


def test_scoring_engine_handles_unscoreable_or_missing_odds_without_recommendation() -> None:
    """Unsupported odds should lower confidence but not prevent score creation."""

    fixture = build_fixture()
    stats = (
        build_team_stats(team_id="arsenal", team_name="Arsenal", wins=6, draws=2, losses=2),
        build_team_stats(team_id="chelsea", team_name="Chelsea", wins=5, draws=2, losses=3),
    )
    unsupported_odds = (
        build_odds_row(
            fixture_ref=fixture.get_fixture_ref(),
            provider="Bet365",
            provider_market_name="Team Totals",
            provider_selection_name="Arsenal Over 1.5",
            odds=2.05,
            line=1.5,
            participant_scope="team",
        ),
    )

    engine = ScoringEngine()
    score = engine.calculate_match_score(
        fixture,
        stats,
        unsupported_odds,
        context=None,
        injuries=None,
        h2h_data=None,
    )

    assert score.recommended_market is None
    assert score.recommended_selection is None
    assert score.recommended_odds is None
    assert score.factors.odds_value == pytest.approx(0.10)
    assert score.confidence < score.composite_score
