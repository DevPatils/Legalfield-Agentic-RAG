"""The state carried through the LangGraph run.

Shape follows the MongoDB trace schema in Architecture.md §10 closely, because the
trace is written incrementally from this state as each node completes -- keeping them
aligned avoids a translation layer that would drift.

Every node returns a *partial* dict; LangGraph merges it. Lists that grow across
iterations (``context``, ``iterations``) are replaced wholesale by the node that owns
them rather than reduced, so a node always sees the full accumulated value.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..retrieval.pipeline import RetrievedChunk

RefineStrategy = Literal["graph_traversal", "query_rewrite", "none"]


class AgentState(TypedDict, total=False):
    # --- input ---
    query_id: str
    session_id: str
    raw_query: str
    doc_id: str | None  # optional single-document filter

    # --- planner ---
    query_type: str
    needs_retrieval: bool
    sub_queries: list[str]
    planner_reasoning: str

    # --- refinement loop ---
    iteration: int
    active_queries: list[str]  # what Retrieve will run next
    context: list[RetrievedChunk]  # accumulated, deduped by chunk_id
    iterations: list[dict[str, Any]]  # per-iteration trace records

    # --- sufficiency verdict (latest) ---
    sufficient: bool
    missing_info: str
    sections_needed: list[str]
    terms_needed: list[str]
    refine_strategy: RefineStrategy

    # --- output ---
    summary: str
    final_answer: str
    citations: list[dict[str, Any]]
    unverified_citations: list[str]
    faithfulness: dict[str, Any]
    low_confidence: bool
    confidence: str

    # --- meta ---
    latency_ms: dict[str, int]
    token_usage: dict[str, int]
    error: str


def new_state(
    raw_query: str,
    query_id: str,
    session_id: str = "",
    doc_id: str | None = None,
) -> AgentState:
    return AgentState(
        query_id=query_id,
        session_id=session_id,
        raw_query=raw_query,
        doc_id=doc_id,
        iteration=0,
        context=[],
        iterations=[],
        sub_queries=[],
        active_queries=[],
        sections_needed=[],
        terms_needed=[],
        summary="",
        citations=[],
        unverified_citations=[],
        low_confidence=False,
        refine_strategy="none",
    )


def merge_context(
    existing: list[RetrievedChunk], incoming: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Append new chunks, keeping the first occurrence of each chunk_id.

    First-wins matters: a chunk pulled by graph traversal carries
    ``source="graph_traversal"``, and that provenance is what the trace panel shows.
    Re-retrieving it later by search must not overwrite that.
    """
    seen = {c.chunk_id for c in existing}
    out = list(existing)
    for chunk in incoming:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            out.append(chunk)
    return out


def referenced_but_absent(context: list[RetrievedChunk]) -> dict[str, str]:
    """Sections the retrieved clauses point at but which are not in context.

    Computed from the precomputed graph, not inferred by an LLM -- this is the
    candidate set the Sufficiency node chooses from, which keeps the model picking
    from real edges instead of inventing section numbers.

    Returns ``{section_id: chunk_id}``.
    """
    present = {c.chunk_id for c in context}
    out: dict[str, str] = {}
    for chunk in context:
        payload = chunk.payload
        refs = payload.get("cross_references") or []
        ids = payload.get("cross_reference_ids") or []
        for section_id, chunk_id in zip(refs, ids, strict=False):
            if chunk_id not in present and chunk_id not in out.values():
                out[section_id] = chunk_id
    return out


def definitions_absent(context: list[RetrievedChunk]) -> dict[str, str]:
    """Defined terms used in context whose defining clause is not in context.

    Returns ``{term: chunk_id}``.
    """
    present = {c.chunk_id for c in context}
    out: dict[str, str] = {}
    for chunk in context:
        payload = chunk.payload
        terms = payload.get("defined_terms_used") or []
        ids = payload.get("definition_ids") or []
        for term, chunk_id in zip(terms, ids, strict=False):
            if chunk_id not in present and chunk_id not in out.values():
                out[term] = chunk_id
    return out
