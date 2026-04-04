"""Canonical scoring-weight accessors for PuntLab's composite scorer.

Purpose: expose the single current-state scoring-weight model used by the
composite scoring engine without duplicating schema definitions.
Scope: re-export the validated `ScoringWeights` schema and provide helpers for
obtaining fresh default weight sets at runtime.
Dependencies: `src.schemas.analysis` for the canonical weight contract.
"""

from __future__ import annotations

from typing import Final

from src.schemas.analysis import ScoringWeights

DEFAULT_SCORING_WEIGHTS: Final[ScoringWeights] = ScoringWeights()


def get_default_scoring_weights() -> ScoringWeights:
    """Return a fresh copy of PuntLab's canonical default factor weights.

    Inputs:
        None.

    Outputs:
        A new `ScoringWeights` instance initialized from the validated default
        weight set so callers can customize a copy without mutating shared
        module state.
    """

    return DEFAULT_SCORING_WEIGHTS.model_copy(deep=True)


__all__ = [
    "DEFAULT_SCORING_WEIGHTS",
    "ScoringWeights",
    "get_default_scoring_weights",
]
