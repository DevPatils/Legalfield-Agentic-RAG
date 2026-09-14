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


# A chunk citing a whole article can contribute a dozen candidates, and §5.8.2 of
# doc_001 alone resolves to 22 targets. The list goes into a prompt, so it is capped:
# past a few dozen the model is choosing from noise, not reasoning about a gap.
MAX_CANDIDATES = 40

# Terms are capped harder than sections. Once traversal pulls in a *definitions* chunk
# -- which it does, that being where definitions live -- that chunk uses scores of
# defined terms and floods the list with a long tail of irrelevant ones. The labels are
# also longer, and the model has to echo its choices back within a token budget.
MAX_TERM_CANDIDATES = 20


def doc_of(chunk_id: str) -> str:
    """``doc_001__sec_9.1`` -> ``doc_001``. Structural ids make this a split, not a
    lookup (Architecture.md §5)."""
    return chunk_id.partition("__sec_")[0]


def label_for(chunk_id: str) -> str:
    """``doc_001__sec_9.1`` -> ``doc_001 §9.1``.

    Deliberately the same shape as the citation tags the generator emits, so the
    sufficiency model is picking from a format it already writes fluently.
    """
    doc_id, _, section_id = chunk_id.partition("__sec_")
    return f"{doc_id} §{section_id}" if section_id else chunk_id


def referenced_but_absent(
    context: list[RetrievedChunk], limit: int = MAX_CANDIDATES
) -> dict[str, str]:
    """Sections the retrieved clauses point at but which are not in context.

    Computed from the precomputed graph, not inferred by an LLM -- this is the
    candidate set the Sufficiency node chooses from, which keeps the model picking
    from real edges instead of inventing section numbers.

    Returns ``{"doc_001 §9.1": chunk_id}``.

    Two things this must not do, both of which it used to:

    * **Zip the two payload lists.** ``cross_references`` is deduplicated at index
      time and ``cross_reference_ids`` is not, so a single reference to a container
      ("Article 9") stores one label against eight resolved ids. Pairing them took
      the first and silently dropped the rest -- 41% of the corpus's edges were
      unreachable. The ids alone are authoritative; the label is derived from them.
    * **Key by the bare section number.** All fifteen contracts have an Article 9, so
      a key of "ART-9" meant the last document in context overwrote every earlier
      one, and traversal fetched a different agreement's clause than the one the
      retrieved text actually cited.
    """
    present = {c.chunk_id for c in context}
    seen: set[str] = set()
    out: dict[str, str] = {}
    # Context is in relevance order, so truncating at the limit keeps the edges of
    # the most relevant clauses and drops those of the least.
    for chunk in context:
        for chunk_id in chunk.payload.get("cross_reference_ids") or []:
            if chunk_id in present or chunk_id in seen:
                continue
            seen.add(chunk_id)
            out[label_for(chunk_id)] = chunk_id
            if len(out) >= limit:
                return out
    return out


def definitions_absent(
    context: list[RetrievedChunk], limit: int = MAX_TERM_CANDIDATES
) -> dict[str, str]:
    """Defined terms used in context whose defining clause is not in context.

    Returns ``{'doc_001 "Confidential Information"': chunk_id}``.

    Unlike the cross-reference lists, these two *are* parallel across the whole
    corpus today -- but only because no term happened to be filtered out when
    ``definition_ids`` was built. Nothing enforces that, so a length mismatch is
    treated as unusable rather than paired off: attaching a term to the wrong
    clause would send traversal somewhere confidently wrong, which is worse than
    sending it nowhere.
    """
    present = {c.chunk_id for c in context}
    seen: set[str] = set()
    out: dict[str, str] = {}
    for chunk in context:
        payload = chunk.payload
        terms = payload.get("defined_terms_used") or []
        ids = payload.get("definition_ids") or []
        if len(terms) != len(ids):
            continue
        for term, chunk_id in zip(terms, ids, strict=True):
            if chunk_id in present or chunk_id in seen:
                continue
            seen.add(chunk_id)
            # Doc-qualified for the same reason as sections: every contract in the
            # corpus defines "Confidential Information".
            out[f'{doc_of(chunk_id)} "{term}"'] = chunk_id
            if len(out) >= limit:
                return out
    return out
