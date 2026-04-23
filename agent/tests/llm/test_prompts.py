"""Tests for PuntLab's canonical LangChain prompt templates.

Purpose: verify prompt-task resolution, required input variables, and rendered
message structure for each LLM-backed pipeline task.
Scope: unit tests for `src.llm.prompts` without calling external services.
Dependencies: pytest and the shared prompt registry exported by `src.llm`.
"""

from __future__ import annotations

import pytest
from langchain_core.prompts import ChatPromptTemplate
from src.llm.prompts import (
    ACCUMULATOR_RATIONALE_PROMPT,
    LEG_RATIONALE_PROMPT,
    NEWS_CONTEXT_ANALYSIS_PROMPT,
    PROMPT_REGISTRY,
    QUALITATIVE_ASSESSMENT_PROMPT,
    get_prompt,
    resolve_prompt_task,
)


def test_get_prompt_accepts_shared_task_aliases() -> None:
    """Prompt lookup should mirror the alias behavior used by LLM providers."""

    assert get_prompt("news_context_analysis") is NEWS_CONTEXT_ANALYSIS_PROMPT
    assert get_prompt("qualitative_score") is QUALITATIVE_ASSESSMENT_PROMPT
    assert get_prompt("leg_explanation") is LEG_RATIONALE_PROMPT
    assert get_prompt("accumulator_explanation") is ACCUMULATOR_RATIONALE_PROMPT


def test_resolve_prompt_task_rejects_unknown_values() -> None:
    """Unexpected prompt task labels should fail fast with a clear error."""

    with pytest.raises(ValueError, match="Unknown prompt task"):
        resolve_prompt_task("imaginary_prompt")


@pytest.mark.parametrize(
    ("prompt", "expected_variables"),
    [
        (
            NEWS_CONTEXT_ANALYSIS_PROMPT,
            {
                "competition_context",
                "fixture_summary",
                "fixture_details",
                "kickoff_context",
                "known_absences",
                "market_menu",
                "recent_news_bullets",
                "run_date",
                "source_labels",
            },
        ),
        (
            QUALITATIVE_ASSESSMENT_PROMPT,
            {
                "fixture_summary",
                "market_signal_summary",
                "match_context_summary",
                "recent_news_bullets",
                "source_labels",
                "statistical_snapshot",
            },
        ),
        (
            LEG_RATIONALE_PROMPT,
            {"fixture_summary", "risk_notes", "score_summary", "selection_summary"},
        ),
        (
            ACCUMULATOR_RATIONALE_PROMPT,
            {"confidence_summary", "legs_summary", "portfolio_note", "slip_summary"},
        ),
    ],
)
def test_prompt_templates_expose_expected_variables(
    prompt: ChatPromptTemplate,
    expected_variables: set[str],
) -> None:
    """Each prompt should require the exact variables its future node will supply."""

    assert set(prompt.input_variables) == expected_variables


def test_research_prompt_renders_conservative_instruction_set() -> None:
    """The research prompt should render system and user guidance together."""

    messages = NEWS_CONTEXT_ANALYSIS_PROMPT.format_messages(
        run_date="2026-04-04",
        fixture_summary="Arsenal vs Real Madrid",
        competition_context="UEFA Champions League quarter-final",
        kickoff_context="20:00 WAT at Emirates Stadium",
        known_absences="Home: none confirmed. Away: one doubtful defender.",
        fixture_details="lineups: Arsenal unchanged; Madrid rotate one defender.",
        market_menu="1X2: Home 2.10 | Draw 3.40 | Away 3.20",
        recent_news_bullets="- Arsenal unbeaten in five\n- Madrid rotating after a derby",
        source_labels="BBC Sport, ESPN",
    )

    assert len(messages) == 2
    assert "Never invent injuries" in messages[0].content
    assert "Arsenal vs Real Madrid" in messages[1].content
    assert "score conservatively" in messages[1].content.lower()


def test_accumulator_prompt_mentions_shared_risk_and_plain_text_output() -> None:
    """The accumulator prompt should steer the model toward compact plain text."""

    messages = ACCUMULATOR_RATIONALE_PROMPT.format_messages(
        slip_summary="Slip #2 with 4 legs and combined odds of 12.8",
        legs_summary=(
            "1. Arsenal win\n"
            "2. Under 3.5 goals in Inter vs Juventus\n"
            "3. Celtics moneyline\n"
            "4. Over 2.5 goals in PSV vs Ajax"
        ),
        confidence_summary="Overall confidence 0.71 with one medium-risk totals leg.",
        portfolio_note="Two legs rely on strong home form; totals legs carry variance.",
    )

    assert len(messages) == 2
    assert "Return plain text only" in messages[0].content
    assert "biggest shared risk" in messages[1].content


def test_prompt_registry_only_contains_supported_canonical_tasks() -> None:
    """The registry should expose one prompt per canonical LLM usage point."""

    assert set(PROMPT_REGISTRY) == {
        "research",
        "qualitative_assessment",
        "leg_rationale",
        "accumulator_rationale",
    }
