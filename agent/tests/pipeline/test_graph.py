"""Tests for PuntLab's compiled LangGraph pipeline assembly.

Purpose: verify canonical node registration, stage ordering, and conditional
approval branching for the compiled pipeline graph.
Scope: unit tests for `src.pipeline.graph.build_pipeline`.
Dependencies: langgraph graph constants plus pipeline graph/state exports.
"""

from __future__ import annotations

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from src.pipeline import build_pipeline as build_pipeline_from_package
from src.pipeline.graph import build_pipeline


def test_build_pipeline_registers_all_expected_nodes_and_edges() -> None:
    """Compiled graph should expose the full canonical stage DAG."""

    compiled_graph = build_pipeline()
    builder = compiled_graph.builder

    assert set(builder.nodes.keys()) == {
        "ingestion",
        "research",
        "scoring",
        "ranking",
        "market_resolution",
        "accumulator_building",
        "explanation",
        "approval",
        "delivery",
    }
    assert builder.edges == {
        (START, "ingestion"),
        ("ingestion", "research"),
        ("research", "scoring"),
        ("scoring", "ranking"),
        ("ranking", "market_resolution"),
        ("market_resolution", "accumulator_building"),
        ("accumulator_building", "explanation"),
        ("explanation", "approval"),
        ("delivery", END),
    }

    approval_branches = builder.branches["approval"]
    approval_branch_spec = next(iter(approval_branches.values()))
    assert approval_branch_spec.ends == {"delivery": "delivery", "blocked": END}


def test_build_pipeline_is_exported_from_pipeline_package() -> None:
    """Package-level export should expose the canonical build entry point."""

    compiled_graph = build_pipeline_from_package()
    assert isinstance(compiled_graph, CompiledStateGraph)
