"""Tests for PuntLab's scoring pipeline node.

Purpose: verify that the scoring stage turns researched fixtures into ordered
`MatchScore` outputs and degrades gracefully when individual fixtures cannot
be matched to supporting stats.
Scope: unit tests for `src.pipeline.nodes.scoring`.
Dependencies: pytest plus the canonical pipeline, fixture, context, odds, and
team-stat schemas and the concrete scoring engine.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.scoring import scoring_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchContext
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, InjuryType, TeamStats
from src.scoring import ScoringEngine


def build_fixture(
    *,
    sportradar_id: str,
    home_team: str,
    away_team: str,
) -> NormalizedFixture:
    """Create a canonical soccer fixture used by scoring-node tests."""

    return NormalizedFixture(
        sportradar_id=sportradar_id,
        home_team=home_team,
        away_team=away_team,
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 4, 19, 0, tzinfo=UTC),
        source_provider="api-football",
        source_id=sportradar_id.split(":")[-1],
        country="England",
        home_team_id=home_team.lower().replace(" ", "-"),
        away_team_id=away_team.lower().replace(" ", "-"),
        venue="Emirates Stadium",
    )


def build_team_stats(
    *,
    team_id: str,
    team_name: str,
    wins: int,
    draws: int,
    losses: int,
    goals_for: int,
    goals_against: int,
    home_wins: int,
    away_wins: int,
    form: str,
    avg_goals_scored: float,
    avg_goals_conceded: float,
    xg_diff: float,
) -> TeamStats:
    """Create one canonical team-stat snapshot for node-level scoring tests."""

    return TeamStats(
        team_id=team_id,
        team_name=team_name,
        sport=SportName.SOCCER,
        source_provider="test-suite",
        fetched_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
        competition="Premier League",
        season="2025-26",
        matches_played=10,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        clean_sheets=3,
        form=form,
        points=(wins * 3) + draws,
        home_wins=home_wins,
        away_wins=away_wins,
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
        advanced_metrics={"xg_diff": xg_diff},
    )


def build_odds_row(
    *,
    fixture_ref: str,
    provider: str,
    provider_selection_name: str,
    odds: float,
    line: float | None = None,
) -> NormalizedOdds:
    """Create one normalized odds row for scoring-node tests."""

    return NormalizedOdds(
        fixture_ref=fixture_ref,
        market=None,
        selection=provider_selection_name,
        odds=odds,
        provider=provider,
        provider_market_name="Full Time Result" if line is None else "Goals Over/Under",
        provider_selection_name=provider_selection_name,
        provider_market_id=provider_selection_name.lower().replace(" ", "_"),
        line=line,
        period="match",
        participant_scope="match",
        raw_metadata={
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        },
        last_updated=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
    )


def build_context(fixture_ref: str) -> MatchContext:
    """Create one qualitative context object for a scored fixture."""

    return MatchContext(
        fixture_ref=fixture_ref,
        morale_home=0.82,
        morale_away=0.46,
        rivalry_factor=0.72,
        pressure_home=0.40,
        pressure_away=0.68,
        key_narrative="The home side arrives with stronger momentum and less pressure.",
        qualitative_score=0.74,
        data_sources=("BBC Sport", "Tavily"),
        news_summary="Home morale is materially stronger heading into the fixture.",
    )


@pytest.mark.asyncio
async def test_scoring_node_generates_match_scores_and_advances_stage() -> None:
    """The scoring node should produce `MatchScore` rows for scoreable fixtures."""

    fixture = build_fixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    result = await scoring_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.SCORING,
            fixtures=[fixture],
            odds_data=[
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Arsenal",
                    odds=1.62,
                ),
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Draw",
                    odds=3.90,
                ),
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Chelsea",
                    odds=5.20,
                ),
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Bet365",
                    provider_selection_name="Arsenal",
                    odds=1.64,
                ),
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Bet365",
                    provider_selection_name="Draw",
                    odds=3.80,
                ),
                build_odds_row(
                    fixture_ref=fixture.get_fixture_ref(),
                    provider="Bet365",
                    provider_selection_name="Chelsea",
                    odds=5.10,
                ),
            ],
            team_stats=[
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
                    xg_diff=0.9,
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
                    xg_diff=-0.4,
                ),
            ],
            injuries=[
                InjuryData(
                    fixture_ref=fixture.get_fixture_ref(),
                    team_id="chelsea",
                    team_name="Chelsea",
                    player_name="Reece James",
                    source_provider="api-football",
                    injury_type=InjuryType.INJURY,
                    reported_at=datetime(2026, 4, 4, 6, 30, tzinfo=UTC),
                )
            ],
            match_contexts=[build_context(fixture.get_fixture_ref())],
            errors=["Earlier-stage warning."],
        ),
        engine=ScoringEngine(),
    )

    assert result["current_stage"] == PipelineStage.RANKING
    assert result["errors"] == ["Earlier-stage warning."]
    assert len(result["match_scores"]) == 1
    score = result["match_scores"][0]
    assert score.fixture_ref == fixture.get_fixture_ref()
    assert score.recommended_market == MarketType.MATCH_RESULT
    assert score.recommended_selection == "home"
    assert score.recommended_odds == pytest.approx(1.64)
    assert score.confidence > 0.60


@pytest.mark.asyncio
async def test_scoring_node_records_fixture_failures_and_continues() -> None:
    """Fixtures that cannot be matched to stats should not block the full slate."""

    scored_fixture = build_fixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    failed_fixture = build_fixture(
        sportradar_id="sr:match:7002",
        home_team="Tottenham",
        away_team="Liverpool",
    )

    result = await scoring_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.SCORING,
            fixtures=[scored_fixture, failed_fixture],
            odds_data=[
                build_odds_row(
                    fixture_ref=scored_fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Arsenal",
                    odds=1.70,
                ),
                build_odds_row(
                    fixture_ref=scored_fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Draw",
                    odds=3.70,
                ),
                build_odds_row(
                    fixture_ref=scored_fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Chelsea",
                    odds=4.80,
                ),
                build_odds_row(
                    fixture_ref=failed_fixture.get_fixture_ref(),
                    provider="Pinnacle",
                    provider_selection_name="Tottenham",
                    odds=2.20,
                ),
            ],
            team_stats=[
                build_team_stats(
                    team_id="arsenal",
                    team_name="Arsenal",
                    wins=7,
                    draws=2,
                    losses=1,
                    goals_for=20,
                    goals_against=10,
                    home_wins=4,
                    away_wins=3,
                    form="WWWDDWWLWW",
                    avg_goals_scored=2.0,
                    avg_goals_conceded=1.0,
                    xg_diff=0.6,
                ),
                build_team_stats(
                    team_id="chelsea",
                    team_name="Chelsea",
                    wins=4,
                    draws=2,
                    losses=4,
                    goals_for=14,
                    goals_against=15,
                    home_wins=2,
                    away_wins=2,
                    form="LDWLWDLLDW",
                    avg_goals_scored=1.4,
                    avg_goals_conceded=1.5,
                    xg_diff=-0.1,
                ),
            ],
            match_contexts=[build_context(scored_fixture.get_fixture_ref())],
            errors=[],
        ),
        engine=ScoringEngine(),
    )

    assert result["current_stage"] == PipelineStage.RANKING
    assert len(result["match_scores"]) == 1
    assert result["match_scores"][0].fixture_ref == scored_fixture.get_fixture_ref()
    assert result["errors"] == [
        (
            "Scoring failed for sr:match:7002: calculate_match_score could not match any "
            "TeamStats records to the fixture."
        )
    ]
