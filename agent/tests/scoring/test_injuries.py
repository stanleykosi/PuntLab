"""Tests for PuntLab's injury-impact scoring factor.

Purpose: verify that the injury factor rewards stable or clearly exploitable
availability pictures while rejecting invalid fixture injury shapes.
Scope: unit tests for `src.scoring.factors.injuries`.
Dependencies: pytest plus the shared injury schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.schemas.stats import InjuryData, InjuryType
from src.scoring.factors.injuries import analyze_injuries


def build_injury(
    *,
    team_id: str,
    player_name: str,
    injury_type: InjuryType = InjuryType.INJURY,
    is_key_player: bool = False,
    player_id: str | None = None,
) -> InjuryData:
    """Build a canonical injury row for factor tests."""

    return InjuryData(
        fixture_ref="sr:match:7001",
        team_id=team_id,
        player_name=player_name,
        player_id=player_id,
        source_provider="test-suite",
        injury_type=injury_type,
        is_key_player=is_key_player,
        reported_at=datetime(2026, 4, 4, 7, 0, tzinfo=UTC),
    )


def test_analyze_injuries_rewards_stable_or_asymmetric_availability() -> None:
    """No injuries or one-sided key absences should beat broad bilateral disruption."""

    no_injuries_score = analyze_injuries(())
    one_sided_key_absence_score = analyze_injuries(
        (
            build_injury(
                team_id="chelsea",
                player_name="Cole Palmer",
                player_id="10",
                injury_type=InjuryType.SUSPENSION,
                is_key_player=True,
            ),
        )
    )
    widespread_disruption_score = analyze_injuries(
        (
            build_injury(team_id="arsenal", player_name="Bukayo Saka", is_key_player=True),
            build_injury(team_id="arsenal", player_name="William Saliba", is_key_player=True),
            build_injury(team_id="chelsea", player_name="Cole Palmer", is_key_player=True),
            build_injury(team_id="chelsea", player_name="Reece James", is_key_player=True),
        )
    )

    assert no_injuries_score > 0.7
    assert one_sided_key_absence_score > widespread_disruption_score
    assert 0.0 <= widespread_disruption_score <= 1.0


def test_analyze_injuries_deduplicates_the_same_player_across_provider_rows() -> None:
    """Duplicate reports for one player should not double-count the burden."""

    score_with_duplicates = analyze_injuries(
        (
            build_injury(
                team_id="chelsea",
                player_name="Cole Palmer",
                player_id="10",
                injury_type=InjuryType.INJURY,
                is_key_player=True,
            ),
            build_injury(
                team_id="chelsea",
                player_name="Cole Palmer",
                player_id="10",
                injury_type=InjuryType.SUSPENSION,
                is_key_player=True,
            ),
        )
    )
    score_without_duplicates = analyze_injuries(
        (
            build_injury(
                team_id="chelsea",
                player_name="Cole Palmer",
                player_id="10",
                injury_type=InjuryType.SUSPENSION,
                is_key_player=True,
            ),
        )
    )

    assert score_with_duplicates == pytest.approx(score_without_duplicates)


def test_analyze_injuries_rejects_more_than_two_fixture_teams() -> None:
    """Fixture-scoped injury sets should fail fast when they contain extra teams."""

    with pytest.raises(ValueError, match="at most two teams"):
        analyze_injuries(
            (
                build_injury(team_id="arsenal", player_name="Player A"),
                build_injury(team_id="chelsea", player_name="Player B"),
                build_injury(team_id="tottenham", player_name="Player C"),
            )
        )
