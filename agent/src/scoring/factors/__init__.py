"""Scoring factor namespace for reusable match-analysis components.

Purpose: expose the deterministic factor calculators consumed by the composite
scoring engine.
Scope: leaf calculators for form, head-to-head, injuries, venue, and
odds-value analysis, with later steps adding qualitative context modules.
Dependencies: imported by `src.scoring.engine` and pipeline scoring nodes.
"""

from src.scoring.factors.form import analyze_form
from src.scoring.factors.h2h import analyze_h2h
from src.scoring.factors.injuries import analyze_injuries
from src.scoring.factors.odds_value import analyze_odds_value
from src.scoring.factors.venue import analyze_venue

__all__ = [
    "analyze_form",
    "analyze_h2h",
    "analyze_injuries",
    "analyze_odds_value",
    "analyze_venue",
]
