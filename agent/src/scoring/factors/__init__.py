"""Scoring factor namespace for reusable match-analysis components.

Purpose: expose the deterministic factor calculators consumed by the composite
scoring engine.
Scope: leaf calculators for form and head-to-head analysis, with later steps
adding injuries, venue, context, and odds-value modules.
Dependencies: imported by `src.scoring.engine` and pipeline scoring nodes.
"""

from src.scoring.factors.form import analyze_form
from src.scoring.factors.h2h import analyze_h2h

__all__ = ["analyze_form", "analyze_h2h"]
