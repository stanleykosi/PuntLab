"""Tests for PuntLab's ranking pipeline node.

Purpose: verify that the ranking stage globally orders scored fixtures and
assigns deterministic 1-based ranks before market resolution begins.
Scope: unit tests for `src.pipeline.nodes.ranking`.
Dependencies: pytest plus the canonical pipeline-state and analysis schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.ranking import ranking_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchScore, ScoreFactorBreakdown


def build_match_score(
    *,
    fixture_ref: str,
    home_team: str,
    away_team: str,
    composite_score: float,
    confidence: float,
) -> MatchScore:
    """Create one canonical scored fixture for ranking-node tests."""

    return MatchScore(
        fixture_ref=fixture_ref,
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team=home_team,
        away_team=away_team,
        composite_score=composite_score,
        confidence=confidence,
        factors=ScoreFactorBreakdown(
            form=0.74,
            h2h=0.58,
            injury_impact=0.62,
            odds_value=0.66,
            context=0.71,
            venue=0.64,
            statistical=0.69,
        ),
        recommended_market=MarketType.MATCH_RESULT,
        recommended_selection="home",
        recommended_odds=1.82,
        qualitative_summary="The stronger home side profiles clearly better.",
    )


@pytest.mark.asyncio
async def test_ranking_node_sorts_by_score_and_assigns_global_ranks() -> None:
    """Higher composite scores should receive better global ranks."""

    top_score = build_match_score(
        fixture_ref="sr:match:7003",
        home_team="Liverpool",
        away_team="Brighton",
        composite_score=0.84,
        confidence=0.76,
    )
    middle_score = build_match_score(
        fixture_ref="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
        composite_score=0.78,
        confidence=0.73,
    )
    lower_score = build_match_score(
        fixture_ref="sr:match:7002",
        home_team="Tottenham",
        away_team="Newcastle",
        composite_score=0.72,
        confidence=0.81,
    )

    result = await ranking_node(
        PipelineState(
            run_id="run-2026-04-04-main",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.RANKING,
            match_scores=[middle_score, lower_score, top_score],
            errors=["Upstream scoring warning."],
        )
    )

    assert result["current_stage"] == PipelineStage.MARKET_RESOLUTION
    assert result["errors"] == ["Upstream scoring warning."]
    ranked_matches = result["ranked_matches"]
    assert [match.fixture_ref for match in ranked_matches] == [
        "sr:match:7003",
        "sr:match:7001",
        "sr:match:7002",
    ]
    assert [match.rank for match in ranked_matches] == [1, 2, 3]


@pytest.mark.asyncio
async def test_ranking_node_breaks_ties_by_confidence_then_fixture_ref() -> None:
    """Equal composite scores should still rank deterministically."""

    lower_confidence = build_match_score(
        fixture_ref="sr:match:7009",
        home_team="Milan",
        away_team="Roma",
        composite_score=0.79,
        confidence=0.70,
    )
    higher_confidence = build_match_score(
        fixture_ref="sr:match:7008",
        home_team="Napoli",
        away_team="Lazio",
        composite_score=0.79,
        confidence=0.75,
    )
    same_confidence_lower_ref = build_match_score(
        fixture_ref="sr:match:7006",
        home_team="Juventus",
        away_team="Atalanta",
        composite_score=0.79,
        confidence=0.75,
    )

    result = await ranking_node(
        PipelineState(
            run_id="run-2026-04-04-ties",
            run_date=date(2026, 4, 4),
            started_at=datetime(2026, 4, 4, 7, 30, tzinfo=UTC),
            current_stage=PipelineStage.RANKING,
            match_scores=[
                lower_confidence,
                higher_confidence,
                same_confidence_lower_ref,
            ],
        )
    )

    ranked_matches = result["ranked_matches"]
    assert [match.fixture_ref for match in ranked_matches] == [
        "sr:match:7006",
        "sr:match:7008",
        "sr:match:7009",
    ]
    assert [match.rank for match in ranked_matches] == [1, 2, 3]
