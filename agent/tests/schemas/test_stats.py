"""Tests for PuntLab's normalized team, player, and injury schemas.

Purpose: lock down statistical consistency checks and shared datetime
validation before provider implementations depend on these contracts.
Scope: unit tests for `src.schemas.stats`.
Dependencies: pytest plus the shared sport enum and stats schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.stats import InjuryData, InjuryType, PlayerStats, TeamStats


def test_team_stats_normalize_form_and_validate_record_totals() -> None:
    """Team snapshots should compact form strings and reject impossible records."""

    stats = TeamStats(
        team_id="arsenal",
        team_name="Arsenal",
        sport=SportName.SOCCER,
        source_provider="api-football",
        fetched_at=datetime(2026, 4, 3, 6, 0, tzinfo=UTC),
        matches_played=5,
        wins=3,
        draws=1,
        losses=1,
        home_wins=2,
        away_wins=1,
        clean_sheets=2,
        form="W-W-D-L-W",
        advanced_metrics={"xg_diff": 1.42},
    )

    assert stats.form == "WWDLW"

    with pytest.raises(ValueError, match="must not exceed matches_played"):
        TeamStats(
            team_id="arsenal",
            team_name="Arsenal",
            sport=SportName.SOCCER,
            source_provider="api-football",
            fetched_at=datetime(2026, 4, 3, 6, 0, tzinfo=UTC),
            matches_played=3,
            wins=3,
            draws=1,
            losses=0,
        )


def test_player_stats_reject_invalid_metric_maps_and_start_counts() -> None:
    """Player snapshots should validate finite metrics and appearance totals."""

    with pytest.raises(ValueError, match="must not exceed appearances"):
        PlayerStats(
            player_id="saka",
            player_name="Bukayo Saka",
            team_id="arsenal",
            sport=SportName.SOCCER,
            source_provider="api-football",
            fetched_at=datetime(2026, 4, 3, 6, 0, tzinfo=UTC),
            appearances=4,
            starts=5,
        )

    with pytest.raises(ValueError, match="finite number"):
        PlayerStats(
            player_id="jokic",
            player_name="Nikola Jokic",
            team_id="denver",
            sport=SportName.BASKETBALL,
            source_provider="balldontlie",
            fetched_at=datetime(2026, 4, 3, 6, 0, tzinfo=UTC),
            metrics={"assist_rate": float("nan")},
        )


def test_injury_data_requires_timezone_aware_reported_at() -> None:
    """Injury snapshots should reject naive timestamps."""

    with pytest.raises(ValueError, match="timezone information"):
        InjuryData(
            fixture_ref="sr:match:61301159",
            team_id="arsenal",
            player_name="Bukayo Saka",
            source_provider="api-football",
            injury_type=InjuryType.INJURY,
            reported_at=datetime(2026, 4, 3, 6, 0),
        )
