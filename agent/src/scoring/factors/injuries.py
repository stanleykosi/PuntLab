"""Injury-impact scoring factor for PuntLab's deterministic scoring engine.

Purpose: translate fixture-scoped injury and suspension reports into a bounded
availability score that rewards stable lineups or clear exploitable absences.
Scope: deduplicate provider overlap, weight key-player absences more heavily,
and penalize broad uncertainty across both teams.
Dependencies: relies on the canonical `InjuryData` schema from `src.schemas`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from src.schemas.stats import InjuryData, InjuryType

_EMPTY_INJURY_SCORE: Final[float] = 0.78
_TEAM_BURDEN_TARGET: Final[float] = 8.0
_KEY_PLAYER_MULTIPLIER: Final[float] = 1.75
_TYPE_SEVERITY: Final[dict[InjuryType, float]] = {
    InjuryType.SUSPENSION: 1.00,
    InjuryType.INJURY: 0.90,
    InjuryType.ILLNESS: 0.65,
    InjuryType.REST: 0.55,
    InjuryType.OTHER: 0.55,
    InjuryType.DOUBTFUL: 0.45,
    InjuryType.QUESTIONABLE: 0.35,
}
_UNCERTAIN_TYPES: Final[frozenset[InjuryType]] = frozenset(
    {InjuryType.DOUBTFUL, InjuryType.QUESTIONABLE}
)


def analyze_injuries(injuries: Sequence[InjuryData]) -> float:
    """Score the fixture's injury and suspension landscape.

    Inputs:
        `injuries`: fixture-scoped injury rows for the two teams involved in a
        candidate match.

    Outputs:
        A bounded score from `0.0` to `1.0`, where higher values indicate a
        cleaner or more decisively exploitable availability picture.

    Raises:
        TypeError: If any item is not an `InjuryData` record.
        ValueError: If the input mixes more than two team identifiers.
    """

    normalized_injuries = _normalize_injuries(injuries)
    if not normalized_injuries:
        return _EMPTY_INJURY_SCORE

    team_burdens: dict[str, float] = {}
    uncertain_burden = 0.0
    for injury in normalized_injuries:
        burden = _injury_burden(injury)
        team_burdens[injury.team_id] = team_burdens.get(injury.team_id, 0.0) + burden
        if injury.injury_type in _UNCERTAIN_TYPES:
            uncertain_burden += burden

    if len(team_burdens) > 2:
        raise ValueError("analyze_injuries expects fixture-scoped injuries from at most two teams.")

    burden_values = sorted(team_burdens.values(), reverse=True)
    primary_burden = burden_values[0]
    secondary_burden = burden_values[1] if len(burden_values) > 1 else 0.0
    total_burden = sum(burden_values)

    stability_score = _clamp(1.0 - (total_burden / _TEAM_BURDEN_TARGET))
    certainty_score = _clamp(1.0 - ((uncertain_burden / max(total_burden, 1.0)) * 0.65))
    imbalance_score = _clamp((primary_burden - secondary_burden) / max(primary_burden, 1.0))
    exploitable_edge_score = _clamp(primary_burden / 4.0) * imbalance_score

    combined_score = (
        (stability_score * 0.40)
        + (certainty_score * 0.20)
        + (exploitable_edge_score * 0.40)
    )
    return _clamp(0.20 + (combined_score * 0.80))


def _normalize_injuries(injuries: Sequence[InjuryData]) -> tuple[InjuryData, ...]:
    """Validate and deduplicate overlapping provider injury rows.

    Args:
        injuries: Raw fixture-scoped injury rows passed to the public factor.

    Returns:
        A tuple of canonical injuries with duplicate player reports collapsed
        to the most severe current record.

    Raises:
        TypeError: If any input item is not an `InjuryData`.
    """

    deduplicated: dict[tuple[str, str], InjuryData] = {}
    for injury in injuries:
        if not isinstance(injury, InjuryData):
            raise TypeError("analyze_injuries expects InjuryData instances only.")

        player_key = (injury.player_id or injury.player_name).casefold()
        record_key = (injury.team_id, player_key)
        current = deduplicated.get(record_key)

        # Multiple providers can report the same absence. We keep the single
        # strongest version so the factor reflects the player once.
        if current is None or _injury_burden(injury) > _injury_burden(current):
            deduplicated[record_key] = injury

    return tuple(deduplicated.values())


def _injury_burden(injury: InjuryData) -> float:
    """Return the weighted burden contributed by one injury row."""

    base_severity = _TYPE_SEVERITY[injury.injury_type]
    multiplier = _KEY_PLAYER_MULTIPLIER if injury.is_key_player else 1.0
    return base_severity * multiplier


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


__all__ = ["analyze_injuries"]
