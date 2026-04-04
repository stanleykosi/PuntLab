"""Head-to-head scoring factor for PuntLab's deterministic scoring engine.

Purpose: quantify how relevant and informative recent matchup history is for a
fixture using the current canonical historical fixture schema.
Scope: assess recency, sample density, competition overlap, and home/away
orientation across prior meetings without depending on provider-specific data.
Dependencies: relies on the canonical `NormalizedFixture` schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from statistics import fmean
from typing import Final

from src.schemas.fixtures import FixtureStatus, NormalizedFixture

_MAX_H2H_SAMPLE: Final[int] = 10
_TARGET_SAMPLE_SIZE: Final[int] = 5


def analyze_h2h(
    fixture: NormalizedFixture,
    h2h_data: Sequence[NormalizedFixture],
) -> float:
    """Score the strength of recent head-to-head history for one fixture.

    Inputs:
        `fixture`: the upcoming canonical fixture being scored.
        `h2h_data`: historical fixtures returned for the same team pair.

    Outputs:
        A bounded score from `0.0` to `1.0`, where higher values indicate a
        larger, more recent, and more contextually relevant matchup history.

    Notes:
        The current canonical fixture schema does not persist historical
        scorelines, so this factor measures matchup-history relevance instead
        of win-loss dominance. That keeps the implementation aligned with the
        data contract already established in earlier steps.

    Raises:
        TypeError: If any historical record is not a `NormalizedFixture`.
        ValueError: If no valid historical meetings remain after validation.
    """

    meetings = _normalize_meetings(fixture, h2h_data)
    if not meetings:
        raise ValueError(
            "No finished historical meetings are available for fixture "
            f"{fixture.get_fixture_ref()}."
        )

    sample_score = _clamp(len(meetings) / _TARGET_SAMPLE_SIZE)
    relevance_scores = tuple(_meeting_relevance(fixture, meeting) for meeting in meetings)
    average_relevance = fmean(relevance_scores)
    orientation_consistency = fmean(
        1.0 if _same_orientation(fixture, meeting) else 0.75 for meeting in meetings
    )
    competition_consistency = fmean(
        1.0 if fixture.competition.casefold() == meeting.competition.casefold() else 0.6
        for meeting in meetings
    )

    score = (
        (sample_score * 0.55)
        + (average_relevance * 0.30)
        + (((orientation_consistency + competition_consistency) / 2) * 0.15)
    )
    return _clamp(score)


def _normalize_meetings(
    fixture: NormalizedFixture,
    h2h_data: Sequence[NormalizedFixture],
) -> tuple[NormalizedFixture, ...]:
    """Filter and order usable historical meetings for one fixture.

    Args:
        fixture: The upcoming fixture being scored.
        h2h_data: Raw historical fixtures returned by the provider layer.

    Returns:
        A tuple of finished, prior meetings sorted from most recent backward.

    Raises:
        TypeError: If a historical row is not a `NormalizedFixture`.
    """

    normalized: list[NormalizedFixture] = []
    for meeting in h2h_data:
        if not isinstance(meeting, NormalizedFixture):
            raise TypeError("analyze_h2h expects NormalizedFixture history rows only.")
        if meeting.sport != fixture.sport:
            continue
        if meeting.status != FixtureStatus.FINISHED:
            continue
        if meeting.kickoff >= fixture.kickoff:
            continue
        if not _is_same_pair(fixture, meeting):
            continue
        normalized.append(meeting)

    normalized.sort(key=lambda meeting: meeting.kickoff, reverse=True)
    return tuple(normalized[:_MAX_H2H_SAMPLE])


def _meeting_relevance(
    fixture: NormalizedFixture,
    meeting: NormalizedFixture,
) -> float:
    """Compute one historical meeting's relevance to the upcoming fixture.

    Args:
        fixture: The upcoming fixture being scored.
        meeting: One validated historical meeting between the same teams.

    Returns:
        A bounded per-meeting relevance score.
    """

    age_in_days = max(_days_between(fixture.kickoff, meeting.kickoff), 0)
    recency_score = 1.0 / (1.0 + (age_in_days / 365.0))
    competition_score = (
        1.0 if fixture.competition.casefold() == meeting.competition.casefold() else 0.6
    )
    orientation_score = 1.0 if _same_orientation(fixture, meeting) else 0.75

    return _clamp(
        (recency_score * 0.55) + (competition_score * 0.25) + (orientation_score * 0.20)
    )


def _is_same_pair(fixture: NormalizedFixture, meeting: NormalizedFixture) -> bool:
    """Check whether two fixtures involve the same team pair in any order."""

    fixture_pair = {fixture.home_team.casefold(), fixture.away_team.casefold()}
    meeting_pair = {meeting.home_team.casefold(), meeting.away_team.casefold()}
    return fixture_pair == meeting_pair


def _same_orientation(fixture: NormalizedFixture, meeting: NormalizedFixture) -> bool:
    """Check whether both fixtures share the same home/away arrangement."""

    return (
        fixture.home_team.casefold() == meeting.home_team.casefold()
        and fixture.away_team.casefold() == meeting.away_team.casefold()
    )


def _days_between(later: datetime, earlier: datetime) -> int:
    """Return the whole-day delta between two timezone-aware datetimes."""

    return int((later - earlier).total_seconds() // 86400)


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


__all__ = ["analyze_h2h"]
