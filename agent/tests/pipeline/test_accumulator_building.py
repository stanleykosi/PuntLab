"""Tests for PuntLab's accumulator-building pipeline node.

Purpose: verify that the builder stage converts resolved ranked matches into
accumulator slips and degrades gracefully when no valid slips can be built.
Scope: unit tests for `src.pipeline.nodes.accumulator_building`.
Dependencies: pytest plus the canonical builder, pipeline-state, ranking, and
resolved-market schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from src.accumulators import AccumulatorBuilder
from src.config import MarketType, SportName
from src.pipeline.nodes.accumulator_building import accumulator_building_node
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.accumulators import ResolutionSource, ResolvedMarket
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown


def build_ranked_match(
    *,
    fixture_ref: str,
    rank: int,
    confidence: float,
    composite_score: float,
    sport: SportName = SportName.SOCCER,
    competition: str = "Premier League",
) -> RankedMatch:
    """Build one canonical ranked match for node-level builder tests."""

    return RankedMatch(
        fixture_ref=fixture_ref,
        sport=sport,
        competition=competition,
        home_team=f"Home {rank}",
        away_team=f"Away {rank}",
        composite_score=composite_score,
        confidence=confidence,
        factors=ScoreFactorBreakdown(
            form=0.78,
            h2h=0.56,
            injury_impact=0.67,
            odds_value=0.72,
            context=0.64,
            venue=0.69,
            statistical=0.62,
        ),
        recommended_market=MarketType.MATCH_RESULT,
        recommended_selection="home",
        recommended_odds=1.72,
        qualitative_summary="The home side profiles clearly stronger.",
        rank=rank,
    )


def build_resolved_market(
    *,
    ranked_match: RankedMatch,
    odds: float,
) -> ResolvedMarket:
    """Build one resolved market aligned with a ranked match recommendation."""

    return ResolvedMarket(
        fixture_ref=ranked_match.fixture_ref,
        sport=ranked_match.sport,
        competition=ranked_match.competition,
        home_team=ranked_match.home_team,
        away_team=ranked_match.away_team,
        market=ranked_match.recommended_market or MarketType.MATCH_RESULT,
        selection=ranked_match.recommended_selection or "home",
        odds=odds,
        provider="sportybet",
        provider_market_name="Full Time Result",
        provider_selection_name=ranked_match.recommended_selection or "home",
        provider_market_id="match_result",
        market_label="Full Time Result",
        resolution_source=ResolutionSource.SPORTYBET_API,
        resolved_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
    )


def build_candidate_pool(
    *,
    count: int = 6,
) -> tuple[tuple[RankedMatch, ...], tuple[ResolvedMarket, ...]]:
    """Build enough ranked matches and resolved markets to create slips."""

    ranked_matches = tuple(
        build_ranked_match(
            fixture_ref=f"sr:match:{7100 + index}",
            rank=index,
            confidence=0.95 - (index * 0.04),
            composite_score=0.90 - (index * 0.03),
            competition="Premier League" if index <= 3 else "La Liga",
        )
        for index in range(1, count + 1)
    )
    resolved_markets = tuple(
        build_resolved_market(
            ranked_match=ranked_match,
            odds=1.55 + (ranked_match.rank * 0.08),
        )
        for ranked_match in ranked_matches
    )
    return ranked_matches, resolved_markets


@pytest.mark.asyncio
async def test_accumulator_building_node_generates_slips_and_advances_stage() -> None:
    """The node should build accumulator slips using the shared builder."""

    run_uuid = uuid4()
    ranked_matches, resolved_markets = build_candidate_pool(count=8)

    result = await accumulator_building_node(
        PipelineState(
            run_id=str(run_uuid),
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.ACCUMULATOR_BUILDING,
            ranked_matches=list(ranked_matches),
            resolved_markets=list(resolved_markets),
            errors=["Market resolution completed."],
        ),
        builder=AccumulatorBuilder(target_count=3),
    )

    assert result["current_stage"] == PipelineStage.EXPLANATION
    assert result["errors"] == ["Market resolution completed."]
    assert len(result["accumulators"]) == 3
    assert [slip.slip_number for slip in result["accumulators"]] == [1, 2, 3]
    assert all(slip.slip_date == date(2026, 4, 5) for slip in result["accumulators"])
    assert all(slip.run_id == run_uuid for slip in result["accumulators"])
    assert [slip.confidence for slip in result["accumulators"]] == sorted(
        (slip.confidence for slip in result["accumulators"]),
        reverse=True,
    )


@pytest.mark.asyncio
async def test_accumulator_building_node_records_builder_failures_and_returns_empty_slips() -> None:
    """Builder failures should surface as diagnostics without crashing the stage."""

    result = await accumulator_building_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 15, tzinfo=UTC),
            current_stage=PipelineStage.ACCUMULATOR_BUILDING,
            ranked_matches=[],
            resolved_markets=[],
        ),
        builder=AccumulatorBuilder(target_count=2),
    )

    assert result["current_stage"] == PipelineStage.EXPLANATION
    assert result["accumulators"] == []
    assert result["errors"] == [
        (
            "Accumulator building failed for run run-2026-04-05-main: "
            "build_accumulators could not produce any unique accumulator slips."
        )
    ]


@pytest.mark.asyncio
async def test_accumulator_building_node_allows_descriptive_run_ids_without_errors() -> None:
    """Non-UUID run IDs should still allow slip generation with `run_id=None`."""

    ranked_matches, resolved_markets = build_candidate_pool()

    result = await accumulator_building_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 20, tzinfo=UTC),
            current_stage=PipelineStage.ACCUMULATOR_BUILDING,
            ranked_matches=list(ranked_matches),
            resolved_markets=list(resolved_markets),
        ),
        builder=AccumulatorBuilder(target_count=1),
    )

    assert result["errors"] == []
    assert len(result["accumulators"]) == 1
    assert result["accumulators"][0].run_id is None
