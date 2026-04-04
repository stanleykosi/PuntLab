"""Scoring engine package for PuntLab.

Purpose: groups deterministic and qualitative scoring logic for fixture analysis.
Scope: factor calculators, weights, and composite scoring orchestration.
Dependencies: consumed by the scoring LangGraph node.
"""

from src.scoring.engine import ScoringEngine
from src.scoring.weights import (
    DEFAULT_SCORING_WEIGHTS,
    ScoringWeights,
    get_default_scoring_weights,
)

__all__ = [
    "DEFAULT_SCORING_WEIGHTS",
    "ScoringEngine",
    "ScoringWeights",
    "get_default_scoring_weights",
]
