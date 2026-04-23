"""Tests for PuntLab's market-resolution pipeline node.

Purpose: verify that ranked matches are resolved into sportsbook-ready market
rows using the prefetched SportyBet catalog while fixture-level failures
remain recoverable.
Scope: unit tests for `src.pipeline.nodes.market_resolution`.
Dependencies: pytest plus lightweight resolver stubs and the canonical
pipeline-state, fixture, ranking, and resolved-market schemas.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from src.config import MarketType, SportName
from src.pipeline.nodes.market_resolution import market_resolution_node
from src.pipeline.state import PipelineStage, PipelineState
from src.providers.base import ProviderError
from src.providers.odds_mapping import build_odds_market_catalog
from src.schemas.accumulators import ResolutionSource, ResolvedMarket
from src.schemas.analysis import RankedMatch, ScoreFactorBreakdown
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds


def build_fixture(
    *,
    sportradar_id: str,
    home_team: str,
    away_team: str,
    sport: SportName = SportName.SOCCER,
    competition: str = "Premier League",
) -> NormalizedFixture:
    """Create a canonical fixture used by market-resolution node tests."""

    return NormalizedFixture(
        sportradar_id=sportradar_id,
        home_team=home_team,
        away_team=away_team,
        competition=competition,
        sport=sport,
        kickoff=datetime(2026, 4, 5, 19, 0, tzinfo=UTC),
        source_provider="test-suite",
        source_id=sportradar_id.split(":")[-1],
        country="England" if sport == SportName.SOCCER else "United States",
    )


def build_ranked_match(
    *,
    fixture: NormalizedFixture,
    rank: int,
    market: MarketType = MarketType.MATCH_RESULT,
    selection: str = "home",
) -> RankedMatch:
    """Create one ranked recommendation for market-resolution tests."""

    return RankedMatch(
        fixture_ref=fixture.get_fixture_ref(),
        sport=fixture.sport,
        competition=fixture.competition,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        composite_score=0.76,
        confidence=0.71,
        factors=ScoreFactorBreakdown(
            form=0.8,
            h2h=0.58,
            injury_impact=0.63,
            odds_value=0.68,
            context=0.7,
            venue=0.65,
            statistical=0.61,
        ),
        recommended_market=market,
        recommended_selection=selection,
        recommended_odds=1.84,
        qualitative_summary="The ranked fixture profiles as the stronger side.",
        rank=rank,
    )


def build_catalog_row(
    *,
    fixture: NormalizedFixture,
    market: MarketType,
    selection: str,
    provider: str = "sportybet",
    provider_market_name: str = "Full Time Result",
    line: float | None = None,
) -> NormalizedOdds:
    """Create one preserved SportyBet row used to verify node inputs."""

    return NormalizedOdds(
        fixture_ref=fixture.get_fixture_ref(),
        market=market,
        selection=selection,
        odds=1.84,
        provider=provider,
        provider_market_name=provider_market_name,
        provider_selection_name=selection,
        line=line,
        period="match",
        participant_scope="match",
        raw_metadata={
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "sportradar_id": fixture.sportradar_id,
            "sportybet_fetch_source": "api",
        },
        last_updated=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
    )


def build_resolved_market(
    *,
    fixture: NormalizedFixture,
    selection: str,
    resolution_source: ResolutionSource,
) -> ResolvedMarket:
    """Create one resolved market row returned by the resolver stub."""

    return ResolvedMarket(
        fixture_ref=fixture.get_fixture_ref(),
        sport=fixture.sport,
        competition=fixture.competition,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        market=MarketType.MATCH_RESULT,
        selection=selection,
        odds=1.84,
        provider="sportybet",
        provider_market_name="Full Time Result",
        provider_selection_name=selection,
        sportybet_available=True,
        resolution_source=resolution_source,
        resolved_at=datetime(2026, 4, 5, 8, 5, tzinfo=UTC),
    )


class StubMarketResolver:
    """Async resolver stub used to drive market-resolution node tests."""

    def __init__(
        self,
        *,
        resolved_by_fixture: dict[str, ResolvedMarket] | None = None,
        errors_by_fixture: dict[str, Exception] | None = None,
    ) -> None:
        """Capture deterministic resolver outcomes keyed by fixture reference."""

        self._resolved_by_fixture = resolved_by_fixture or {}
        self._errors_by_fixture = errors_by_fixture or {}
        self.resolve_calls: list[tuple[str, tuple[NormalizedOdds, ...]]] = []
        self.closed = False

    async def resolve(
        self,
        fixture: NormalizedFixture,
        analysis: RankedMatch,
        *,
        prefetched_rows: tuple[NormalizedOdds, ...] = (),
        use_cache: bool = True,
    ) -> ResolvedMarket:
        """Return the configured resolution result or raise a configured error."""

        del use_cache
        self.resolve_calls.append((analysis.fixture_ref, prefetched_rows))
        error = self._errors_by_fixture.get(analysis.fixture_ref)
        if error is not None:
            raise error
        return self._resolved_by_fixture[analysis.fixture_ref]

    async def aclose(self) -> None:
        """Record cleanup calls made by the market-resolution node."""

        self.closed = True


@pytest.mark.asyncio
async def test_market_resolution_node_resolves_ranked_matches_and_uses_full_catalog() -> None:
    """The node should resolve ranked fixtures and pass the preserved odds universe through."""

    fixture = build_fixture(
        sportradar_id="sr:match:7001",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    ranked_match = build_ranked_match(fixture=fixture, rank=1)
    catalog_rows = (
        build_catalog_row(
            fixture=fixture,
            market=MarketType.MATCH_RESULT,
            selection="home",
        ),
        build_catalog_row(
            fixture=fixture,
            market=MarketType.OVER_UNDER_25,
            selection="Over",
            provider_market_name="Goals Over/Under",
            line=2.5,
        ),
    )
    resolver = StubMarketResolver(
        resolved_by_fixture={
            fixture.get_fixture_ref(): build_resolved_market(
                fixture=fixture,
                selection="home",
                resolution_source=ResolutionSource.SPORTYBET_API,
            )
        }
    )

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-main",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 0, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[fixture],
            ranked_matches=[ranked_match],
            odds_market_catalog=build_odds_market_catalog(
                catalog_rows,
                sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
            ),
            errors=["Ranking completed."],
        ),
        resolver=resolver,  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.ACCUMULATOR_BUILDING
    assert result["errors"] == ["Ranking completed."]
    assert resolver.closed is True
    assert len(result["resolved_markets"]) == 1
    assert result["resolved_markets"][0].fixture_ref == fixture.get_fixture_ref()
    assert len(resolver.resolve_calls) == 1
    assert resolver.resolve_calls[0][0] == fixture.get_fixture_ref()
    assert resolver.resolve_calls[0][1] == catalog_rows


@pytest.mark.asyncio
async def test_market_resolution_node_records_failures_and_missing_fixtures() -> None:
    """Fixture-level resolution failures should not block successful resolutions."""

    resolved_fixture = build_fixture(
        sportradar_id="sr:match:7002",
        home_team="Liverpool",
        away_team="Brighton",
    )
    missing_fixture_ranked_match = build_ranked_match(
        fixture=build_fixture(
            sportradar_id="sr:match:9999",
            home_team="Napoli",
            away_team="Roma",
            competition="Serie A",
        ),
        rank=3,
    )
    failed_fixture = build_fixture(
        sportradar_id="sr:match:7003",
        home_team="Milan",
        away_team="Inter",
        competition="Serie A",
    )
    resolver = StubMarketResolver(
        resolved_by_fixture={
            resolved_fixture.get_fixture_ref(): build_resolved_market(
                fixture=resolved_fixture,
                selection="home",
                resolution_source=ResolutionSource.SPORTYBET_BROWSER,
            )
        },
        errors_by_fixture={
            failed_fixture.get_fixture_ref(): ProviderError(
                "market-resolver",
                "all resolver sources failed",
            )
        },
    )

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-errors",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 7, 10, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[resolved_fixture, failed_fixture],
            ranked_matches=[
                build_ranked_match(fixture=resolved_fixture, rank=1),
                build_ranked_match(fixture=failed_fixture, rank=2),
                missing_fixture_ranked_match,
            ],
        ),
        resolver=resolver,  # type: ignore[arg-type]
    )

    assert result["current_stage"] == PipelineStage.ACCUMULATOR_BUILDING
    assert len(result["resolved_markets"]) == 1
    assert result["resolved_markets"][0].fixture_ref == resolved_fixture.get_fixture_ref()
    assert result["errors"] == [
        (
            "Market resolution failed for sr:match:7003: "
            "[market-resolver] all resolver sources failed"
        ),
        (
            "Market resolution skipped for sr:match:9999: "
            "no matching fixture exists in pipeline state."
        ),
    ]
    assert [call[0] for call in resolver.resolve_calls] == [
        resolved_fixture.get_fixture_ref(),
        failed_fixture.get_fixture_ref(),
    ]


@pytest.mark.asyncio
async def test_market_resolution_node_groups_repeated_failures_by_reason() -> None:
    """Repeated fixture-level resolver failures should collapse into one summary."""

    failed_fixture_one = build_fixture(
        sportradar_id="sr:match:7201",
        home_team="Milan",
        away_team="Inter",
        competition="Serie A",
    )
    failed_fixture_two = build_fixture(
        sportradar_id="sr:match:7202",
        home_team="Napoli",
        away_team="Roma",
        competition="Serie A",
    )
    resolver = StubMarketResolver(
        resolved_by_fixture={},
        errors_by_fixture={
            failed_fixture_one.get_fixture_ref(): ProviderError(
                "market-resolver",
                "all resolver sources failed",
            ),
            failed_fixture_two.get_fixture_ref(): ProviderError(
                "market-resolver",
                "all resolver sources failed",
            ),
        },
    )

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-grouped-resolution",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 8, 0, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[failed_fixture_one, failed_fixture_two],
            ranked_matches=[
                build_ranked_match(fixture=failed_fixture_one, rank=1),
                build_ranked_match(fixture=failed_fixture_two, rank=2),
            ],
        ),
        resolver=resolver,  # type: ignore[arg-type]
    )

    assert result["resolved_markets"] == []
    assert result["errors"] == [
        (
            "Market resolution failed for 2 fixtures due to: "
            "[market-resolver] all resolver sources failed. "
            "Sample fixtures: sr:match:7201, sr:match:7202."
        )
    ]


@pytest.mark.asyncio
async def test_market_resolution_node_skips_ranked_matches_without_recommendation() -> None:
    """Ranked matches without a recommendation should be skipped early."""

    fixture = build_fixture(
        sportradar_id="sr:match:7301",
        home_team="Arsenal",
        away_team="Chelsea",
    )
    resolver = StubMarketResolver(resolved_by_fixture={})
    missing_recommendation = build_ranked_match(fixture=fixture, rank=1).model_copy(
        update={
            "recommended_market": None,
            "recommended_selection": None,
            "recommended_odds": None,
        }
    )

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-missing-recommendation",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 9, 0, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[fixture],
            ranked_matches=[missing_recommendation],
        ),
        resolver=resolver,  # type: ignore[arg-type]
    )

    assert result["resolved_markets"] == []
    assert result["errors"] == [
        (
            "Market resolution skipped for sr:match:7301: "
            "ranked match lacks a recommended market and selection."
        )
    ]
    assert resolver.resolve_calls == []


@pytest.mark.asyncio
async def test_market_resolution_node_normalizes_verbose_sportybet_failures() -> None:
    """Fixture-specific resolver detail should collapse into one grouped root cause."""

    failed_fixture_one = build_fixture(
        sportradar_id="sr:match:7401",
        home_team="Milan",
        away_team="Inter",
        competition="Serie A",
    )
    failed_fixture_two = build_fixture(
        sportradar_id="sr:match:7402",
        home_team="Napoli",
        away_team="Roma",
        competition="Serie A",
    )
    resolver_error_detail = (
        "Could not resolve a bookmaker market for fixture sr:match:placeholder "
        "(Team A vs Team B). Recommended market: `match_result`. "
        "Recommended selection: `home`. Diagnostics: SportyBet resolution was skipped because "
        "fixture.sportradar_id is missing. | No SportyBet source produced a compatible "
        "market for the fixture."
    )
    resolver = StubMarketResolver(
        resolved_by_fixture={},
        errors_by_fixture={
            failed_fixture_one.get_fixture_ref(): ProviderError(
                "market-resolver",
                resolver_error_detail,
            ),
            failed_fixture_two.get_fixture_ref(): ProviderError(
                "market-resolver",
                resolver_error_detail.replace("sr:match:placeholder", "sr:match:other"),
            ),
        },
    )

    result = await market_resolution_node(
        PipelineState(
            run_id="run-2026-04-05-grouped-sportybet",
            run_date=date(2026, 4, 5),
            started_at=datetime(2026, 4, 5, 9, 30, tzinfo=UTC),
            current_stage=PipelineStage.MARKET_RESOLUTION,
            fixtures=[failed_fixture_one, failed_fixture_two],
            ranked_matches=[
                build_ranked_match(fixture=failed_fixture_one, rank=1),
                build_ranked_match(fixture=failed_fixture_two, rank=2),
            ],
        ),
        resolver=resolver,  # type: ignore[arg-type]
    )

    assert result["resolved_markets"] == []
    assert result["errors"] == [
        (
            "Market resolution failed for 2 fixtures due to: "
            "[market-resolver] no compatible SportyBet odds were found for the recommended "
            "market and selection. "
            "Sample fixtures: sr:match:7401, sr:match:7402."
        )
    ]
