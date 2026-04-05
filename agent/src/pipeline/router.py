"""Pipeline routing helpers for LangGraph conditional edges.

Purpose: expose deterministic router functions that map validated pipeline
state to named branch labels consumed by graph conditional edges.
Scope: approval-stage routing from soft-gate outcomes to either delivery or
terminal blocked exit.
Dependencies: `src.pipeline.state` for canonical approval status validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from src.pipeline.state import ApprovalStatus, PipelineState

type ApprovalRoute = Literal["delivery", "blocked"]


def approval_router(state: PipelineState | Mapping[str, Any]) -> ApprovalRoute:
    """Route approval outcomes to the correct downstream graph branch.

    Inputs:
        state: Pipeline state from LangGraph, either as the validated model or
            a raw mapping that can be validated into one.

    Outputs:
        `"delivery"` when at least one accumulator is approved, otherwise
        `"blocked"` when no slips can be published.

    Raises:
        ValueError: If approval status is still pending at routing time.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    if validated_state.approval_status is ApprovalStatus.APPROVED:
        return "delivery"
    if validated_state.approval_status is ApprovalStatus.BLOCKED:
        return "blocked"

    raise ValueError(
        "approval_router requires a resolved approval status, "
        f"but received `{validated_state.approval_status.value}`."
    )


__all__ = ["ApprovalRoute", "approval_router"]
