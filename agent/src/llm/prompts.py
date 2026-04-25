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
    "market_scoring",
    "ranking",
    "market_resolution",
    "accumulator_builder",
    "accumulator_rationale",
)

NEWS_CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's SportyBet fixture-context researcher. Ground the analysis "
                "first in the supplied SportyBet pre-match fixture-page widgets. RSS/Tavily "
                "news is supplemental only. Never invent injuries, lineups, table position, "
                "H2H, probability, team info, or widget data. The research stage does not "
                "select betting markets; later stages handle market scoring and accumulator "
                "construction."
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
                "SportyBet fixture-page details:\n{fixture_details}\n"
                "Supplemental RSS/Tavily news bullets:\n{recent_news_bullets}\n"
                "Source labels: {source_labels}\n\n"
                "Parse the SportyBet fixture-page details into fixture_detail_summary, "
                "tactical_context, statistical_context, availability_context, market_context, "
                "supplemental_news_context, qualitative_score, and data_sources. Keep all "
                "summaries concise and grounded in the supplied widget lines."
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

MARKET_SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's lead betting analyst. You must choose from the exact "
                "SportyBet markets supplied by the user, using fixture details, research, "
                "injuries, market price, and risk. Do not invent unavailable markets. If "
                "the evidence is weak, still choose the least-bad available market but give "
                "lower confidence. Your output is validated as JSON, so every field must be "
                "present and correctly typed."
            ),
        ),
        (
            "human",
            (
                "Fixture:\n{fixture_summary}\n\n"
                "Research context:\n{match_context_summary}\n\n"
                "Raw SportyBet fixture-page details:\n{fixture_details}\n\n"
                "Known absences:\n{known_absences}\n\n"
                "Available SportyBet market menu:\n{market_menu}\n\n"
                "Return a full MatchScore JSON object. Pick exactly one available market "
                "selection from the menu. Use provider market key in recommended_market, "
                "display label in recommended_market_label, provider selection in "
                "recommended_selection, decimal price in recommended_odds, numeric line or "
                "null in recommended_line, and a concise qualitative_summary explaining the "
                "betting edge."
            ),
        ),
    ]
)

RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's slate-ranking analyst. Rank the supplied fixture "
                "recommendations by expected betting quality, balancing model confidence, "
                "price sanity, evidence strength, and market risk. Do not explain your "
                "reasoning. Keep every fixture ref exactly once."
            ),
        ),
        (
            "human",
            (
                "Run date: {run_date}\n"
                "Scored recommendations:\n{score_menu}\n\n"
                "Return JSON with ranked_fixture_refs as an array of every fixture_ref, "
                "ordered from strongest to weakest recommendation. No other keys."
            ),
        ),
    ]
)

MARKET_RESOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's market-resolution analyst. Select exactly one concrete "
                "SportyBet odds row from the supplied row menu. The row must match the "
                "recommendation and be suitable for a real accumulator leg. Ambiguous labels "
                "or missing interval details should be avoided unless they are explicitly "
                "shown in the row."
            ),
        ),
        (
            "human",
            (
                "Ranked recommendation:\n{ranked_match_summary}\n\n"
                "SportyBet rows for this fixture:\n{row_menu}\n\n"
                "Return JSON with fixture_ref, row_id, confidence, and rationale. row_id "
                "must be copied exactly from the menu."
            ),
        ),
    ]
)

ACCUMULATOR_BUILDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are PuntLab's accumulator architect. Choose fixture-ref combinations "
                "from the resolved SportyBet legs. Return only the requested JSON; do not "
                "write analysis outside the JSON. Avoid duplicate fixtures inside a slip "
                "and avoid obvious same-league correlation when alternatives exist."
            ),
        ),
        (
            "human",
            (
                "Run date: {run_date}\n"
                "Target slip count: {target_count}\n"
                "Resolved legs:\n{resolved_leg_menu}\n\n"
                "Return a compact JSON object with key slips. Produce exactly "
                "{target_count} slips when enough legs exist. Use 2 to 4 legs per slip. "
                "Each slip must include slip_number, leg_fixture_refs, confidence, "
                "strategy, and rationale. Use strategy values confident, balanced, or "
                "aggressive. Keep each rationale under 18 words."
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
                "hype or guaranteed language. Return plain text only."
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
    "market_scoring": MARKET_SCORING_PROMPT,
    "ranking": RANKING_PROMPT,
    "market_resolution": MARKET_RESOLUTION_PROMPT,
    "accumulator_builder": ACCUMULATOR_BUILDER_PROMPT,
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
            `accumulator_builder`, or `accumulator_rationale`.

    Outputs:
        The reusable `ChatPromptTemplate` bound to the resolved task key.
    """

    return PROMPT_REGISTRY[resolve_prompt_task(task)]


__all__ = [
    "ACCUMULATOR_RATIONALE_PROMPT",
    "ACCUMULATOR_BUILDER_PROMPT",
    "MARKET_RESOLUTION_PROMPT",
    "MARKET_SCORING_PROMPT",
    "NEWS_CONTEXT_ANALYSIS_PROMPT",
    "PROMPT_REGISTRY",
    "PROMPT_TASK_KEYS",
    "QUALITATIVE_ASSESSMENT_PROMPT",
    "RANKING_PROMPT",
    "get_prompt",
    "resolve_prompt_task",
]
