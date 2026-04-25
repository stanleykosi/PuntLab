"""LLM ranking node for PuntLab's LangGraph pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.llm import get_llm, get_prompt
from src.pipeline.llm_json import invoke_json_schema
from src.pipeline.state import PipelineStage, PipelineState
from src.schemas.analysis import MatchScore, RankedMatch
from src.schemas.llm_decisions import LLMRankingDecision


async def ranking_node(state: PipelineState | Mapping[str, Any]) -> dict[str, object]:
    """Execute the ranking stage and return LangGraph state updates.

    Inputs:
        state: Current pipeline state, either as a validated `PipelineState`
            instance or a raw mapping that can be validated into one.

    Outputs:
        A partial LangGraph update containing globally ordered
        `RankedMatch` rows plus the next-stage marker.
    """

    validated_state = (
        state if isinstance(state, PipelineState) else PipelineState.model_validate(state)
    )
    actionable_scores = _actionable_match_scores(validated_state.match_scores)
    if not actionable_scores:
        raise RuntimeError("ranking_node requires at least one LLM-scored market recommendation.")

    llm = await get_llm("ranking")
    prompt = get_prompt("ranking")
    decision = await invoke_json_schema(
        llm=llm,
        prompt_messages=prompt.format_messages(
            run_date=validated_state.run_date.isoformat(),
            score_menu=_render_score_menu(actionable_scores),
        ),
        schema=LLMRankingDecision,
        instruction=(
            "Return only this JSON shape: {\"ranked_fixture_refs\": [\"fixture_ref\", ...]}. "
            "Include every supplied fixture_ref exactly once and no unknown fixture refs. "
            "Do not include analysis, scores, objects, markdown, or extra keys."
        ),
    )
    ordered_scores = _order_scores_from_llm(actionable_scores, decision)
    ranked_matches = [
        RankedMatch.model_validate(
            {
                **score.model_dump(),
                "rank": rank,
            }
        )
        for rank, score in enumerate(ordered_scores, start=1)
    ]

    return {
        "current_stage": PipelineStage.MARKET_RESOLUTION,
        "ranked_matches": ranked_matches,
        "errors": list(validated_state.errors),
    }


def _actionable_match_scores(match_scores: Sequence[MatchScore]) -> list[MatchScore]:
    """Return only scored fixtures that contain a concrete market recommendation."""

    actionable_scores: list[MatchScore] = []
    for score in match_scores:
        if not isinstance(score, MatchScore):
            raise TypeError("ranking_node expects MatchScore instances only.")
        if score.recommended_market is None or score.recommended_selection is None:
            continue
        actionable_scores.append(score)
    return actionable_scores


def _order_scores_from_llm(
    match_scores: Sequence[MatchScore],
    decision: LLMRankingDecision,
) -> list[MatchScore]:
    """Return scores in the exact order selected by the LLM."""

    score_by_ref = {score.fixture_ref: score for score in match_scores}
    expected_refs = set(score_by_ref)
    observed_refs = set(decision.ranked_fixture_refs)
    if observed_refs != expected_refs:
        missing = sorted(expected_refs - observed_refs)
        unknown = sorted(observed_refs - expected_refs)
        raise ValueError(
            "LLM ranking output did not match scored fixture set. "
            f"missing={missing}; unknown={unknown}."
        )
    return [score_by_ref[fixture_ref] for fixture_ref in decision.ranked_fixture_refs]


def _render_score_menu(match_scores: Sequence[MatchScore]) -> str:
    """Render LLM-scored recommendations for ranking."""

    lines: list[str] = []
    for score in match_scores:
        lines.append(
            f"- fixture_ref={score.fixture_ref}; {score.home_team} vs {score.away_team}; "
            f"score={score.composite_score:.2f}; confidence={score.confidence:.2f}; "
            f"market={score.recommended_market_label}; selection={score.recommended_selection}; "
            f"odds={score.recommended_odds}; line={score.recommended_line}"
        )
    return "\n".join(lines)


__all__ = ["ranking_node"]
