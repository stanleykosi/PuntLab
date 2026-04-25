"""Tests for PuntLab's canonical pipeline state schema.

Purpose: verify the master LangGraph state contract before graph assembly and
stage nodes begin depending on it.
Scope: unit tests for `src.pipeline.state`.
Dependencies: pytest plus the shared schemas used by the pipeline state.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.providers.odds_mapping import build_odds_market_catalog
from src.schemas.accumulators import (
    AccumulatorLeg,
    ExplainedAccumulator,
    ResolutionSource,
    ResolvedMarket,
)
from src.schemas.analysis import MatchContext, MatchScore, RankedMatch, ScoreFactorBreakdown
from src.schemas.fixtures import NormalizedFixture
from src.schemas.news import NewsArticle
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import (
    InjuryData,
    InjuryType,
    PlayerStats,
    TeamStats,
)
from src.schemas.users import (
    DeliveryChannel,
    DeliveryResult,
    DeliveryStatus,
    SubscriptionTier,
)


def build_fixture() -> NormalizedFixture:
    """Create a representative fixture for pipeline-state tests."""

    return NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 3, 19, 0, tzinfo=UTC),
        source_provider="api-football",
        source_id="fixture-1001",
        country="England",
    )


def build_factors() -> ScoreFactorBreakdown:
    """Create a deterministic score breakdown used across related test models."""

    return ScoreFactorBreakdown(
        form=0.82,
        h2h=0.54,
        injury_impact=0.71,
        odds_value=0.67,
        context=0.76,
        venue=0.63,
        statistical=0.59,
    )


def build_explained_accumulator() -> ExplainedAccumulator:
    """Create a delivery-ready accumulator used by multiple assertions."""

    leg = AccumulatorLeg(
        leg_number=1,
        fixture_ref="sr:match:61301159",
        sport=SportName.SOCCER,
        competition="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        market=MarketType.OVER_UNDER_25,
        selection="Over",
        odds=1.85,
        provider="sportybet",
        confidence=0.72,
        resolution_source=ResolutionSource.SPORTYBET_API,
        rationale="Both sides have sustained attacking output entering the tie.",
    )

    return ExplainedAccumulator(
        slip_date=date(2026, 4, 3),
        slip_number=1,
        legs=(leg,),
        total_odds=1.85,
        leg_count=1,
        confidence=0.72,
        rationale="The matchup profile supports a goals-first angle.",
    )


def test_pipeline_state_serializes_all_stage_outputs() -> None:
    """Pipeline state should preserve validated nested outputs across all stages."""

    fixture = build_fixture()
    factors = build_factors()
    explained_accumulator = build_explained_accumulator()

    state = PipelineState(
        run_id="run-2026-04-03-main",
        run_date=date(2026, 4, 3),
        started_at=datetime(2026, 4, 3, 7, 0, tzinfo=UTC),
        current_stage=PipelineStage.APPROVAL,
        errors=["Provider timeout handled via fallback."],
        fixtures=[fixture],
        odds_market_catalog=build_odds_market_catalog(
            (
                NormalizedOdds(
                    fixture_ref=fixture.get_fixture_ref(),
                    market=None,
                    selection="Over 2.5",
                    odds=1.85,
                    provider="sportybet",
                    provider_market_name="Goals Over/Under",
                    provider_selection_name="Over 2.5",
                    line=2.5,
                    period="match",
                    participant_scope="match",
                    raw_metadata={"sportybet_fetch_source": "api"},
                ),
            )
        ),
        odds_data=[
            NormalizedOdds(
                fixture_ref=fixture.get_fixture_ref(),
                market=MarketType.OVER_UNDER_25,
                selection="Over",
                odds=1.85,
                provider="sportybet",
                provider_market_name="Goals Over/Under",
                provider_selection_name="Over 2.5",
            )
        ],
        team_stats=[
            TeamStats(
                team_id="arsenal",
                team_name="Arsenal",
                sport=SportName.SOCCER,
                source_provider="api-football",
                fetched_at=datetime(2026, 4, 3, 6, 30, tzinfo=UTC),
                competition="Premier League",
                matches_played=30,
                wins=20,
                draws=5,
                losses=5,
            )
        ],
        player_stats=[
            PlayerStats(
                player_id="saka",
                player_name="Bukayo Saka",
                team_id="arsenal",
                sport=SportName.SOCCER,
                source_provider="api-football",
                fetched_at=datetime(2026, 4, 3, 6, 35, tzinfo=UTC),
                appearances=24,
                starts=22,
                metrics={"goals": 12.0, "assists": 9.0},
            )
        ],
        injuries=[
            InjuryData(
                fixture_ref=fixture.get_fixture_ref(),
                team_id="chelsea",
                player_name="Reece James",
                source_provider="api-football",
                injury_type=InjuryType.INJURY,
                reported_at=datetime(2026, 4, 2, 20, 0, tzinfo=UTC),
            )
        ],
        news_articles=[
            NewsArticle(
                headline="Arsenal prepare for decisive London derby",
                url="https://www.example.com/arsenal-derby-preview",
                published_at=datetime(2026, 4, 3, 5, 45, tzinfo=UTC),
                source="BBC Sport",
                source_provider="rss",
                teams=("Arsenal", "Chelsea"),
                fixture_ref=fixture.get_fixture_ref(),
            )
        ],
        match_contexts=[
            MatchContext(
                fixture_ref=fixture.get_fixture_ref(),
                fixture_detail_summary=(
                    "SportyBet widgets show Arsenal with the cleaner derby setup."
                ),
                tactical_context="Arsenal lineup context is stable.",
                statistical_context="SportyBet comparison favors Arsenal.",
                supplemental_news_context="BBC Sport adds derby context.",
                qualitative_score=0.74,
                data_sources=("SportyBet fixture-page widgets", "BBC Sport"),
            )
        ],
        match_scores=[
            MatchScore(
                fixture_ref=fixture.get_fixture_ref(),
                sport=SportName.SOCCER,
                competition="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
                composite_score=0.73,
                confidence=0.7,
                factors=factors,
                recommended_market=MarketType.OVER_UNDER_25,
                recommended_selection="Over",
                recommended_odds=1.85,
                qualitative_summary="Both teams continue to create quality chances.",
            )
        ],
        ranked_matches=[
            RankedMatch(
                fixture_ref=fixture.get_fixture_ref(),
                sport=SportName.SOCCER,
                competition="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
                composite_score=0.73,
                confidence=0.7,
                factors=factors,
                recommended_market=MarketType.OVER_UNDER_25,
                recommended_selection="Over",
                recommended_odds=1.85,
                qualitative_summary="Both teams continue to create quality chances.",
                rank=1,
            )
        ],
        resolved_markets=[
            ResolvedMarket(
                fixture_ref=fixture.get_fixture_ref(),
                sport=SportName.SOCCER,
                competition="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
                market=MarketType.OVER_UNDER_25,
                selection="Over",
                odds=1.85,
                provider="sportybet",
                provider_market_name="Goals Over/Under",
                provider_selection_name="Over 2.5",
                sportybet_available=True,
                resolution_source=ResolutionSource.SPORTYBET_API,
                resolved_at=datetime(2026, 4, 3, 6, 50, tzinfo=UTC),
            )
        ],
        accumulators=[explained_accumulator],
        explained_accumulators=[explained_accumulator],
        approval_status=ApprovalStatus.APPROVED,
        blocked_ids=["fixture-review-2"],
        delivery_results=[
            DeliveryResult(
                channel=DeliveryChannel.TELEGRAM,
                status=DeliveryStatus.SENT,
                subscription_tier=SubscriptionTier.FREE,
                recipient="telegram:123456",
                delivered_at=datetime(2026, 4, 3, 10, 0, tzinfo=UTC),
            )
        ],
    )

    dumped = state.model_dump(mode="json")

    assert dumped["current_stage"] == "approval"
    assert dumped["approval_status"] == "approved"
    assert dumped["fixtures"][0]["sportradar_id"] == "sr:match:61301159"
    assert dumped["odds_market_catalog"]["markets"][0]["provider_market_name"] == "Goals Over/Under"
    assert dumped["delivery_results"][0]["status"] == "sent"
    assert state.explained_accumulators[0].rationale == (
        "The matchup profile supports a goals-first angle."
    )


def test_pipeline_state_rejects_naive_start_time_and_blank_error_entries() -> None:
    """Pipeline state should fail fast on ambiguous timestamps and blank strings."""

    with pytest.raises(ValueError, match="started_at must include timezone"):
        PipelineState(
            run_id="run-2026-04-03-main",
            run_date=date(2026, 4, 3),
            started_at=datetime(2026, 4, 3, 7, 0),
        )

    with pytest.raises(ValueError, match="errors_item must not be blank"):
        PipelineState(
            run_id="run-2026-04-03-main",
            run_date=date(2026, 4, 3),
            started_at=datetime(2026, 4, 3, 7, 0, tzinfo=UTC),
            errors=["   "],
        )
