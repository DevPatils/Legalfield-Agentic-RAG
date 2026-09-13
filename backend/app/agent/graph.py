"""The LangGraph wiring of Architecture.md §7.

    Planner ─(needs_retrieval)─▶ Retrieve ─▶ Sufficiency ─(sufficient)─▶ Generate
       │                            ▲            │                          │
       │(no retrieval)              │      (insufficient,                   ▼
       │                            │       iter < max)               Faithfulness
       │                            │            ▼                          │
       ▼                            └───────── Refine ──┐                   ▼
    Generate ◀───────────────────────────────────────┐  │                  END
                                                     │  └─(graph traversal)─▶ Sufficiency
                                          (query rewrite)

Two cycles, which is why this is a graph rather than a chain. The invariant that keeps
it terminating: the iteration cap is checked in the routing function, and hitting it
routes to Generate with a low-confidence flag. The loop never fails closed and never
runs unbounded.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .deps import AgentDeps
from .nodes import (
    make_faithfulness,
    make_generate,
    make_planner,
    make_refine,
    make_retrieve,
    make_sufficiency,
)
from .state import AgentState, new_state


def route_after_planner(state: AgentState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("needs_retrieval", True) else "generate"


def make_route_after_sufficiency(max_iterations: int):
    def route(state: AgentState) -> Literal["refine", "generate"]:
        if state.get("error"):
            return "generate"
        if state.get("sufficient", False):
            return "generate"
        if state.get("iteration", 0) >= max_iterations:
            return "generate"  # cap reached: answer anyway, flagged low-confidence
        # Nothing left to try: no graph edge to follow and no rewrite would differ.
        if not state.get("sections_needed") and not state.get("terms_needed"):
            if state.get("refine_strategy") == "query_rewrite":
                return "generate"
        return "refine"

    return route


def route_after_refine(state: AgentState) -> Literal["retrieve", "sufficiency"]:
    """Graph traversal already fetched its clauses, so it re-checks sufficiency
    directly. A query rewrite has to go back through retrieval."""
    return "sufficiency" if state.get("refine_strategy") == "graph_traversal" else "retrieve"


def make_flag_low_confidence(max_iterations: int):
    """Mark the answer low-confidence when Generate is reached without sufficiency."""

    def flag(state: AgentState) -> dict[str, Any]:
        capped = state.get("iteration", 0) >= max_iterations
        unsatisfied = not state.get("sufficient", True)
        return {"low_confidence": bool(capped and unsatisfied) or unsatisfied}

    return flag


def build_graph(deps: AgentDeps):
    """Compile the agent graph."""
    max_iterations = deps.settings.max_refine_iterations
    builder = StateGraph(AgentState)

    builder.add_node("planner", make_planner(deps))
    builder.add_node("retrieve", make_retrieve(deps))
    builder.add_node("sufficiency", make_sufficiency(deps))
    builder.add_node("refine", make_refine(deps))
    builder.add_node("flag", make_flag_low_confidence(max_iterations))
    builder.add_node("generate", make_generate(deps))
    builder.add_node("faithfulness", make_faithfulness(deps))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner", route_after_planner, {"retrieve": "retrieve", "generate": "flag"}
    )
    builder.add_edge("retrieve", "sufficiency")
    builder.add_conditional_edges(
        "sufficiency",
        make_route_after_sufficiency(max_iterations),
        {"refine": "refine", "generate": "flag"},
    )
    builder.add_conditional_edges(
        "refine", route_after_refine, {"retrieve": "retrieve", "sufficiency": "sufficiency"}
    )
    builder.add_edge("flag", "generate")
    builder.add_edge("generate", "faithfulness")
    builder.add_edge("faithfulness", END)

    # recursion_limit is LangGraph's own backstop; the iteration cap above is the
    # real control. Generous here so the cap is what actually fires.
    return builder.compile()


def run_query(
    deps: AgentDeps,
    query: str,
    query_id: str,
    session_id: str = "",
    doc_id: str | None = None,
) -> AgentState:
    """Run the graph to completion and return the final state."""
    graph = build_graph(deps)
    initial = new_state(query, query_id=query_id, session_id=session_id, doc_id=doc_id)
    return graph.invoke(initial, config={"recursion_limit": 50})


def stream_query(
    deps: AgentDeps,
    query: str,
    query_id: str,
    session_id: str = "",
    doc_id: str | None = None,
):
    """Yield ``(node_name, partial_state)`` as each node completes.

    This is what the SSE endpoint wires to (Architecture.md §8) and what makes the
    reasoning trace visible live rather than only after the answer lands.
    """
    graph = build_graph(deps)
    initial = new_state(query, query_id=query_id, session_id=session_id, doc_id=doc_id)
    for update in graph.stream(initial, config={"recursion_limit": 50}):
        yield from update.items()
