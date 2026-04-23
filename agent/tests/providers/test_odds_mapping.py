"""Tests for the odds market catalog and canonical mapping layer.

Purpose: verify that PuntLab can preserve the full provider-native odds market
universe while projecting only the currently scoreable subset into canonical
market types.
Scope: unit tests for catalog grouping, canonical market inference, scoreable
filters, and fixture/canonical grouping helpers.
Dependencies: pytest, shared runtime market enums, and the odds mapping module.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.config import MarketType, SportName
from src.providers.odds_mapping import (
    build_fixture_market_snapshots,
    build_odds_market_catalog,
    filter_scoreable_odds,
    filter_unmapped_odds,
    group_markets_by_canonical_market,
    group_markets_by_fixture,
)
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds


def test_build_odds_market_catalog_preserves_full_market_universe() -> None:
    """The catalog should keep every provider market while exposing scoreable rows."""

    odds_rows = (
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Arsenal",
            odds=1.88,
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Arsenal",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "provider_selection_id": "sel-home",
            },
            last_updated=datetime(2026, 4, 3, 11, 30, tzinfo=UTC),
        ),
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Draw",
            odds=3.40,
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Draw",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "provider_selection_id": "sel-draw",
            },
            last_updated=datetime(2026, 4, 3, 11, 30, tzinfo=UTC),
        ),
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Chelsea",
            odds=4.30,
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Chelsea",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "provider_selection_id": "sel-away",
            },
            last_updated=datetime(2026, 4, 3, 11, 30, tzinfo=UTC),
        ),
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Over 2.5",
            odds=1.91,
            provider="Pinnacle",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Over 2.5",
            provider_market_id="totals",
            line=2.5,
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "provider_selection_id": "sel-over-25",
            },
        ),
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Under 2.5",
            odds=1.89,
            provider="Pinnacle",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Under 2.5",
            provider_market_id="totals",
            line=2.5,
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "provider_selection_id": "sel-under-25",
            },
        ),
        NormalizedOdds(
            fixture_ref="fixture-1",
            market=None,
            selection="Arsenal Over 1.5",
            odds=2.12,
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
                "provider_selection_id": "sel-team-total",
            },
        ),
    )

    catalog = build_odds_market_catalog(odds_rows)

    assert len(catalog.markets) == 3
    assert len(catalog.all_rows()) == 6
    assert len(catalog.scoreable_rows()) == 5
    assert len(catalog.unmapped_rows()) == 1
    assert catalog.all_rows()[0].market is None

    match_result_market = catalog.markets[0]
    assert match_result_market.provider_market_key == "full_time_result"
    assert match_result_market.raw_metadata["sport_key"] == "soccer_epl"
    assert match_result_market.selections[0].provider_selection_id == "sel-home"
    assert match_result_market.scoreable_market_types() == (MarketType.MATCH_RESULT,)
    assert match_result_market.scoreable_rows()[0].selection == "home"
    assert match_result_market.scoreable_rows()[1].selection == "draw"
    assert match_result_market.scoreable_rows()[2].selection == "away"

    totals_market = catalog.markets[1]
    assert totals_market.scoreable_market_types() == (MarketType.OVER_UNDER_25,)
    assert totals_market.scoreable_rows()[0].selection == "over"
    assert totals_market.scoreable_rows()[1].selection == "under"

    team_totals_market = catalog.markets[2]
    expected_reason = (
        "participant scope `team` is preserved but not scoreable in the current canonical "
        "`over_under_1.5` taxonomy."
    )
    assert team_totals_market.scoreable_rows() == ()
    assert team_totals_market.selections[0].mapping_reason == expected_reason


def test_filter_scoreable_odds_respects_scope_and_sport_rules() -> None:
    """Scoreable filtering should exclude preserved markets outside the current taxonomy."""

    odds_rows = (
        NormalizedOdds(
            fixture_ref="fixture-2",
            market=None,
            selection="Lakers",
            odds=1.74,
            provider="DraftKings",
            provider_market_name="Head to Head",
            provider_selection_name="Los Angeles Lakers",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
            },
        ),
        NormalizedOdds(
            fixture_ref="fixture-2",
            market=None,
            selection="Over 228.5",
            odds=1.95,
            provider="DraftKings",
            provider_market_name="Totals",
            provider_selection_name="Over",
            provider_market_id="totals",
            line=228.5,
            period="match",
            participant_scope="match",
            raw_metadata={"sport_key": "basketball_nba"},
        ),
        NormalizedOdds(
            fixture_ref="fixture-2",
            market=None,
            selection="Arsenal",
            odds=2.80,
            provider="Bet365",
            provider_market_name="Match Winner",
            provider_selection_name="Arsenal",
            provider_market_id=1,
            period="first_half",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
            },
        ),
        NormalizedOdds(
            fixture_ref="fixture-2",
            market=None,
            selection="Arsenal Over 1.5",
            odds=2.05,
            provider="Bet365",
            provider_market_name="Goals Over/Under",
            provider_selection_name="Arsenal Over 1.5",
            provider_market_id=5,
            line=1.5,
            period="match",
            participant_scope="team",
            raw_metadata={"sport_key": "soccer_epl"},
        ),
    )

    scoreable_rows = filter_scoreable_odds(odds_rows)
    unmapped_rows = filter_unmapped_odds(odds_rows)

    assert [(row.market, row.selection) for row in scoreable_rows] == [
        (MarketType.MONEYLINE, "home"),
        (MarketType.TOTAL_POINTS, "over"),
    ]
    assert [row.provider_market_name for row in unmapped_rows] == [
        "Match Winner",
        "Goals Over/Under",
    ]


def test_grouping_helpers_bucket_catalog_markets_by_fixture_and_canonical_market() -> None:
    """Grouping helpers should preserve catalog order across fixtures and market types."""

    odds_rows = (
        NormalizedOdds(
            fixture_ref="fixture-a",
            market=None,
            selection="Arsenal",
            odds=1.88,
            provider="Pinnacle",
            provider_market_name="Full Time Result",
            provider_selection_name="Arsenal",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
            },
        ),
        NormalizedOdds(
            fixture_ref="fixture-a",
            market=None,
            selection="Yes",
            odds=1.73,
            provider="Pinnacle",
            provider_market_name="Both Teams To Score",
            provider_selection_name="Yes",
            provider_market_id="btts",
            period="match",
            participant_scope="match",
            raw_metadata={"sport_key": "soccer_epl"},
        ),
        NormalizedOdds(
            fixture_ref="fixture-b",
            market=None,
            selection="Lakers",
            odds=1.74,
            provider="DraftKings",
            provider_market_name="Head to Head",
            provider_selection_name="Los Angeles Lakers",
            provider_market_id="h2h",
            period="match",
            participant_scope="match",
            raw_metadata={
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
            },
        ),
    )

    catalog = build_odds_market_catalog(odds_rows)

    grouped_by_fixture = group_markets_by_fixture(catalog.markets)
    grouped_by_market = group_markets_by_canonical_market(catalog.markets)

    assert list(grouped_by_fixture) == ["fixture-a", "fixture-b"]
    assert [market.provider_market_name for market in grouped_by_fixture["fixture-a"]] == [
        "Full Time Result",
        "Both Teams To Score",
    ]
    assert [market.fixture_ref for market in grouped_by_market[MarketType.MATCH_RESULT]] == [
        "fixture-a"
    ]
    assert [market.fixture_ref for market in grouped_by_market[MarketType.BTTS]] == [
        "fixture-a"
    ]
    assert [market.fixture_ref for market in grouped_by_market[MarketType.MONEYLINE]] == [
        "fixture-b"
    ]

    scoreable_with_explicit_sport = filter_scoreable_odds(
        odds_rows,
        sport_by_fixture={"fixture-b": SportName.BASKETBALL},
    )
    assert scoreable_with_explicit_sport[-1].market == MarketType.MONEYLINE


def test_build_fixture_market_snapshots_preserves_market_groups_and_total_size() -> None:
    """Fixture snapshots should expose grouped full-market menus for prompt use."""

    fixture = NormalizedFixture(
        sportradar_id="sr:match:61301159",
        home_team="Arsenal",
        away_team="Chelsea",
        competition="Premier League",
        sport=SportName.SOCCER,
        kickoff=datetime(2026, 4, 3, 19, 0, tzinfo=UTC),
        source_provider="sportybet",
        source_id="61301159",
        country="England",
    )
    odds_rows = (
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Home",
            odds=1.88,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Home",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "event_id": fixture.get_fixture_ref(),
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 42,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Draw",
            odds=3.40,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Draw",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "event_id": fixture.get_fixture_ref(),
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 42,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Away",
            odds=4.20,
            provider="sportybet",
            provider_market_name="1X2",
            provider_selection_name="Away",
            provider_market_id=1,
            period="match",
            participant_scope="match",
            raw_metadata={
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "event_id": fixture.get_fixture_ref(),
                "market_group_id": "1001",
                "market_group_name": "Main",
                "event_total_market_size": 42,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Over 2.5",
            odds=1.91,
            provider="sportybet",
            provider_market_name="Over/Under",
            provider_selection_name="Over 2.5",
            provider_market_id=18,
            line=2.5,
            period="match",
            participant_scope="match",
            raw_metadata={
                "event_id": fixture.get_fixture_ref(),
                "market_group_id": "1002",
                "market_group_name": "Goals",
                "event_total_market_size": 42,
                "sportybet_fetch_source": "api",
            },
        ),
        NormalizedOdds(
            fixture_ref=fixture.get_fixture_ref(),
            market=None,
            selection="Under 2.5",
            odds=1.89,
            provider="sportybet",
            provider_market_name="Over/Under",
            provider_selection_name="Under 2.5",
            provider_market_id=18,
            line=2.5,
            period="match",
            participant_scope="match",
            raw_metadata={
                "event_id": fixture.get_fixture_ref(),
                "market_group_id": "1002",
                "market_group_name": "Goals",
                "event_total_market_size": 42,
                "sportybet_fetch_source": "api",
            },
        ),
    )

    catalog = build_odds_market_catalog(
        odds_rows,
        sport_by_fixture={fixture.get_fixture_ref(): fixture.sport},
    )
    snapshots = build_fixture_market_snapshots((fixture,), catalog)

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.fixture_ref == fixture.get_fixture_ref()
    assert snapshot.total_market_size == 42
    assert snapshot.reported_total_market_size == 42
    assert snapshot.fetched_market_count == 2
    assert snapshot.fetched_selection_count == 5
    assert snapshot.scoreable_market_count == 2
    assert snapshot.fetch_source == "api"
    assert [group.group_name for group in snapshot.market_groups] == ["Main", "Goals"]
    assert snapshot.market_groups[0].markets[0].canonical_markets == (MarketType.MATCH_RESULT,)
    assert (
        snapshot.market_groups[1].markets[0].selections[0].canonical_selection == "over"
    )
