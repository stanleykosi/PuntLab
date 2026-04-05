"""LangGraph assembly for PuntLab's canonical daily pipeline.

Purpose: compile the full multi-stage LangGraph DAG that powers PuntLab's
daily run from ingestion through delivery.
Scope: deterministic node registration, linear stage edges, and the approval
conditional branch used to gate delivery publication.
Dependencies: `langgraph` graph primitives, canonical stage nodes under
`src.pipeline.nodes`, and approval routing helpers under `src.pipeline.router`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.pipeline.nodes import (
    accumulator_building_node,
    approval_node,
    delivery_node,
    explanation_node,
    ingestion_node,
    market_resolution_node,
    ranking_node,
    research_node,
    scoring_node,
)
from src.pipeline.router import approval_router
from src.pipeline.state import PipelineState


def build_pipeline() -> CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]:
    """Assemble and compile the canonical PuntLab LangGraph pipeline.

    Inputs:
        None. The graph uses the canonical pipeline state schema and stage
        node implementations shipped in `src.pipeline.nodes`.

    Outputs:
        A compiled LangGraph state graph with deterministic stage ordering and
        an approval conditional branch:
        - `delivery` when approval resolves to approved
        - terminal `END` when approval resolves to blocked
    """

    graph = StateGraph(PipelineState)
    _register_nodes(graph)
    _register_edges(graph)
    return graph.compile()


def _register_nodes(graph: StateGraph[PipelineState, None, PipelineState, PipelineState]) -> None:
    """Register all canonical stage nodes on the pipeline graph."""

    graph.add_node("ingestion", ingestion_node)
    graph.add_node("research", research_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("market_resolution", market_resolution_node)
    graph.add_node("accumulator_building", accumulator_building_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("approval", approval_node)
    graph.add_node("delivery", delivery_node)


def _register_edges(graph: StateGraph[PipelineState, None, PipelineState, PipelineState]) -> None:
    """Register linear and conditional edges for the compiled DAG."""

    graph.add_edge(START, "ingestion")
    graph.add_edge("ingestion", "research")
    graph.add_edge("research", "scoring")
    graph.add_edge("scoring", "ranking")
    graph.add_edge("ranking", "market_resolution")
    graph.add_edge("market_resolution", "accumulator_building")
    graph.add_edge("accumulator_building", "explanation")
    graph.add_edge("explanation", "approval")
    graph.add_conditional_edges(
        "approval",
        approval_router,
        {"delivery": "delivery", "blocked": END},
    )
    graph.add_edge("delivery", END)


__all__ = ["build_pipeline"]
