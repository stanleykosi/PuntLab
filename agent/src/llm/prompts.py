"""Canonical LangChain prompt templates for PuntLab's LLM-backed stages.

Purpose: define the single current-state prompt set used by research, scoring,
and explanation nodes in the LangGraph pipeline.
Scope: concise, task-specific `ChatPromptTemplate` instances for fixture news
context analysis, qualitative scoring, leg rationale generation, and
accumulator rationale generation.
Dependencies: `langchain_core.prompts` for prompt construction and
`src.llm.providers` for shared task-alias normalization.
"""

from __future__ import annotations

from typing import Final

from langchain_core.prompts import ChatPromptTemplate

from src.llm.providers import TASK_ALIASES

PROMPT_TASK_KEYS: Final[tuple[str, ...]] = (
    "research",
    "qualitative_assessment",
    "leg_rationale",
    "accumulator_rationale",
)

NEWS_CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's fixture news analyst. Use only the supplied evidence. "
                "Never invent injuries, quotes, lineup news, or standings context. Score "
                "conservatively on a 0-1 scale when evidence is thin. Keep key_narrative "
                "under 200 characters and news_summary under 160 characters."
            ),
        ),
        (
            "human",
            (
                "Run date: {run_date}\n"
                "Fixture: {fixture_summary}\n"
                "Competition context: {competition_context}\n"
                "Kickoff context: {kickoff_context}\n"
                "Known absences: {known_absences}\n"
                "Relevant news bullets:\n{recent_news_bullets}\n"
                "Source labels: {source_labels}\n\n"
                "Assess morale, rivalry intensity, and pressure for both teams. Score "
                "conservatively, ground every score in the supplied material, and lower "
                "confidence if the news is weak or mixed."
            ),
        ),
    ]
)

QUALITATIVE_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's qualitative scoring analyst. Combine the supplied stats, "
                "market clues, and news context into cautious 0-1 sub-scores. Do not invent "
                "facts or overstate edges. Keep the summary under 240 characters."
            ),
        ),
        (
            "human",
            (
                "Fixture: {fixture_summary}\n"
                "Statistical snapshot:\n{statistical_snapshot}\n"
                "Market signal summary:\n{market_signal_summary}\n"
                "Match context summary:\n{match_context_summary}\n"
                "Supporting news bullets:\n{recent_news_bullets}\n"
                "Source labels: {source_labels}\n\n"
                "Score momentum, lineup stability, motivation, and narrative alignment. The "
                "overall qualitative_score should reflect the balance of upside versus risk."
            ),
        ),
    ]
)

LEG_RATIONALE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You write betting-selection rationales for PuntLab. Keep the rationale to "
                "1-2 sentences, under 45 words, and focus on the strongest evidence-backed "
                "edge. Mention a caveat only if it materially changes confidence."
            ),
        ),
        (
            "human",
            (
                "Fixture: {fixture_summary}\n"
                "Selection: {selection_summary}\n"
                "Score summary: {score_summary}\n"
                "Risk notes: {risk_notes}\n\n"
                "Write a concise rationale a Nigerian bettor can scan quickly."
            ),
        ),
    ]
)

ACCUMULATOR_RATIONALE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You write accumulator summaries for PuntLab. Keep the summary to 2-3 "
                "sentences, under 70 words, and explain why the legs fit together without "
                "hype or guaranteed language."
            ),
        ),
        (
            "human",
            (
                "Slip summary: {slip_summary}\n"
                "Legs:\n{legs_summary}\n"
                "Confidence summary: {confidence_summary}\n"
                "Portfolio note: {portfolio_note}\n\n"
                "Summarize the slip's overall logic, confidence posture, and biggest shared "
                "risk."
            ),
        ),
    ]
)

PROMPT_REGISTRY: Final[dict[str, ChatPromptTemplate]] = {
    "research": NEWS_CONTEXT_ANALYSIS_PROMPT,
    "qualitative_assessment": QUALITATIVE_ASSESSMENT_PROMPT,
    "leg_rationale": LEG_RATIONALE_PROMPT,
    "accumulator_rationale": ACCUMULATOR_RATIONALE_PROMPT,
}


def resolve_prompt_task(task: str) -> str:
    """Resolve a prompt request to one canonical prompt key.

    Inputs:
        task: Task label used by pipeline nodes or helper code. Aliases from
            `src.llm.providers` are accepted to keep prompt and model routing in
            sync.

    Outputs:
        The canonical prompt key that exists in `PROMPT_REGISTRY`.
    """

    normalized_task = task.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized_task:
        raise ValueError("Prompt task keys must not be blank.")

    resolved_task = TASK_ALIASES.get(normalized_task, normalized_task)
    if resolved_task not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt task '{task}'. Expected one of: {sorted(PROMPT_REGISTRY)}."
        )

    return resolved_task


def get_prompt(task: str) -> ChatPromptTemplate:
    """Return the canonical prompt template for one pipeline task.

    Inputs:
        task: Stable task name such as `research`, `qualitative_assessment`,
            `leg_rationale`, or `accumulator_rationale`.

    Outputs:
        The reusable `ChatPromptTemplate` bound to the resolved task key.
    """

    return PROMPT_REGISTRY[resolve_prompt_task(task)]


__all__ = [
    "ACCUMULATOR_RATIONALE_PROMPT",
    "LEG_RATIONALE_PROMPT",
    "NEWS_CONTEXT_ANALYSIS_PROMPT",
    "PROMPT_REGISTRY",
    "PROMPT_TASK_KEYS",
    "QUALITATIVE_ASSESSMENT_PROMPT",
    "get_prompt",
    "resolve_prompt_task",
]
