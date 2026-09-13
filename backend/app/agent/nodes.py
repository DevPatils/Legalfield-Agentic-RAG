"""The six agent nodes of Architecture.md §7.

Each takes the state and returns a partial update. None of them raise on LLM trouble:
a failure degrades to a defensible default and records it, because a half-finished
trace that says what went wrong is more useful than a 500.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..ingest.models import make_chunk_id
from ..ingest.roster import resolve_doc
from ..retrieval.pipeline import RetrievedChunk
from .deps import AgentDeps
from .prompts import (
    FAITHFULNESS_SYSTEM,
    GENERATE_SYSTEM,
    REWRITE_SYSTEM,
    SUFFICIENCY_SYSTEM,
    planner_system,
)
from .schemas import (
    FaithfulnessOutput,
    GeneratedAnswer,
    PlannerOutput,
    RewrittenQuery,
    SufficiencyOutput,
)
from .state import (
    AgentState,
    definitions_absent,
    merge_context,
    referenced_but_absent,
)

# Matches the citation contract in GENERATE_SYSTEM: [doc_001 §4.2]
RE_CITATION = re.compile(r"\[([A-Za-z0-9_\-]+)\s*§\s*([^\]\s]+)\]")

# A section the user named directly: "section 2.2.2", "§ 4.1", or a bare "2.2.2".
RE_NAMED_SECTION = re.compile(
    r"(?:(?:section|clause|article|§)\s*)?(\d+(?:\.\d+)+(?:\([A-Za-z0-9]{1,4}\))*)",
    re.I,
)

MAX_CONTEXT_CHUNKS = 12


def sections_named_in(query: str) -> list[str]:
    """Section numbers the user asked for by name.

    Naming a section is a lookup, not a search: "2.2.2" and "2.2.3" are adjacent
    strings in embedding space and unrelated clauses in the document, so semantic
    retrieval on a bare number is close to a coin flip. Requires at least one dot so
    plain quantities ("30 days", "5 years") are not mistaken for clause numbers.
    """
    out: list[str] = []
    for match in RE_NAMED_SECTION.finditer(query):
        section_id = match.group(1)
        if section_id not in out:
            out.append(section_id)
    return out


def format_clauses(chunks: list[RetrievedChunk]) -> str:
    """Render chunks for an LLM prompt with the citation tag on each header.

    The tag is shown in exactly the form the generator must emit, so producing a
    correct citation is copying rather than constructing.
    """
    blocks = []
    for chunk in chunks:
        title = chunk.payload.get("section_title") or "(untitled)"
        path = chunk.payload.get("path") or ""
        header = f"[{chunk.doc_id} §{chunk.section_id}] {title}"
        if path:
            header += f"\n({path})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- planner


def make_planner(deps: AgentDeps):
    def planner(state: AgentState) -> dict[str, Any]:
        query = state["raw_query"]
        try:
            out = deps.llm.parse(
                system=planner_system(deps.roster_text),
                user=f"Question: {query}",
                schema=PlannerOutput,
                node="planner",
                fast=True,  # mechanical work; runs on the cheaper model
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            # Retrieval on the raw query is the safe default -- better to search
            # unnecessarily than to refuse a question because planning failed.
            return {
                "query_type": "cross_referential",
                "needs_retrieval": True,
                "sub_queries": [query],
                "active_queries": [query],
                "planner_reasoning": f"planner failed, defaulting to retrieval: {exc}",
            }

        sub_queries = [q.strip() for q in out.sub_queries if q.strip()] or [query]
        known = {d.doc_id for d in deps.roster}

        # The model's pick is trusted only if it names a real document; otherwise fall
        # back to matching the question against the roster ourselves.
        doc_id = out.doc_id if out.doc_id in known else resolve_doc(deps.roster, query)

        # Belt and braces over the prompt: naming a document or a section number is
        # objective evidence the question is about contract content, whatever the
        # model concluded. Refusing such a question is never the right call.
        names_section = bool(sections_named_in(query))
        needs_retrieval = out.needs_retrieval or doc_id is not None or names_section
        query_type = out.query_type
        if query_type == "out_of_scope" and (doc_id is not None or names_section):
            query_type = "overview"

        return {
            "query_type": query_type,
            "needs_retrieval": needs_retrieval,
            "doc_id": doc_id or state.get("doc_id"),
            "sub_queries": sub_queries if needs_retrieval else [],
            "active_queries": sub_queries if needs_retrieval else [],
            "planner_reasoning": out.reasoning,
        }

    return planner


# -------------------------------------------------------------------------- retrieve


def make_retrieve(deps: AgentDeps):
    def retrieve(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        config = deps.retrieval_config()
        doc_id = state.get("doc_id")
        found: list[RetrievedChunk] = []

        # If the user named a section of a known document, fetch it by id first.
        # Searching for "Section 2.2.2" semantically tends to return 2.2.1 or the
        # same number from a different agreement; constructing the key does not.
        named: list[str] = []
        if doc_id and state.get("iteration", 0) == 0:
            wanted = [
                make_chunk_id(doc_id, section_id)
                for section_id in sections_named_in(state["raw_query"])
            ]
            if wanted:
                direct = deps.pipeline.fetch_by_ids(wanted)
                for chunk in direct:
                    chunk.source = "direct_lookup"
                found.extend(direct)
                named = [c.section_id for c in direct]

        for sub_query in state.get("active_queries") or [state["raw_query"]]:
            try:
                found.extend(deps.pipeline.retrieve(sub_query, config, doc_id=doc_id))
            except Exception as exc:  # noqa: BLE001
                return {"error": f"retrieval failed: {exc}"}

        context = merge_context(state.get("context", []), found)
        iteration = state.get("iteration", 0) + 1
        elapsed = int((time.perf_counter() - started) * 1000)

        record = {
            "iteration": iteration,
            "trigger": state.get("refine_strategy", "none"),
            "sub_queries": list(state.get("active_queries", [])),
            "direct_lookups": named,
            "retrieved_chunks": [c.to_trace() for c in found],
            "latency_ms": elapsed,
        }
        latency = dict(state.get("latency_ms", {}))
        latency["retrieval"] = latency.get("retrieval", 0) + elapsed

        return {
            "context": context,
            "iteration": iteration,
            "iterations": [*state.get("iterations", []), record],
            "latency_ms": latency,
        }

    return retrieve


# ---------------------------------------------------------------------- sufficiency


def make_sufficiency(deps: AgentDeps):
    def sufficiency(state: AgentState) -> dict[str, Any]:
        context = state.get("context", [])
        if not context:
            return {
                "sufficient": True,  # nothing retrieved; let Generate say so
                "missing_info": "no clauses were retrieved for this question",
                "sections_needed": [],
                "terms_needed": [],
            }

        # Candidates come from the precomputed graph, so the model picks from real
        # edges instead of inventing section numbers.
        section_candidates = referenced_but_absent(context)
        term_candidates = definitions_absent(context)

        parts = [f"Question: {state['raw_query']}", "", "Retrieved clauses:", ""]
        parts.append(format_clauses(context[:MAX_CONTEXT_CHUNKS]))
        if section_candidates:
            parts += [
                "",
                "These sections are referenced by the clauses above but are NOT present.",
                "Choose from this list only:",
                ", ".join(sorted(section_candidates)),
            ]
        if term_candidates:
            parts += [
                "",
                "These defined terms are used above but their definitions are NOT present.",
                "Choose from this list only:",
                ", ".join(sorted(term_candidates)),
            ]

        try:
            out = deps.llm.parse(
                system=SUFFICIENCY_SYSTEM,
                user="\n".join(parts),
                schema=SufficiencyOutput,
                node="sufficiency",
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            # Proceed to Generate rather than looping on a broken check.
            return {
                "sufficient": True,
                "missing_info": f"sufficiency check failed: {exc}",
                "sections_needed": [],
                "terms_needed": [],
            }

        # Keep only choices that map to real edges -- guards against invented ids.
        sections = [s for s in out.referenced_sections_needed if s in section_candidates]
        terms = [t for t in out.defined_terms_needed if t in term_candidates]

        iterations = list(state.get("iterations", []))
        if iterations:
            iterations[-1] = {
                **iterations[-1],
                "sufficiency": {
                    "sufficient": out.sufficient,
                    "missing": out.missing_info,
                    "sections_needed": sections,
                    "terms_needed": terms,
                },
            }

        return {
            "sufficient": out.sufficient,
            "missing_info": out.missing_info,
            "sections_needed": sections,
            "terms_needed": terms,
            "iterations": iterations,
        }

    return sufficiency


# --------------------------------------------------------------------------- refine


def make_refine(deps: AgentDeps):
    def refine(state: AgentState) -> dict[str, Any]:
        context = state.get("context", [])
        section_candidates = referenced_but_absent(context)
        term_candidates = definitions_absent(context)

        # Candidate keys are already the doc-qualified labels the sufficiency node
        # showed the model, so they go straight into the trace as chosen.
        wanted_ids: list[str] = []
        pulled: list[str] = []
        for label in [*state.get("sections_needed", []), *state.get("terms_needed", [])]:
            chunk_id = section_candidates.get(label) or term_candidates.get(label)
            if chunk_id and chunk_id not in wanted_ids:
                wanted_ids.append(chunk_id)
                pulled.append(label)

        # Architecture.md §7: prefer graph traversal when the gap is an explicit
        # reference. It is a key lookup rather than a search -- cheaper, exact, and
        # it cannot return a neighbouring clause.
        if wanted_ids:
            started = time.perf_counter()
            fetched = deps.pipeline.fetch_by_ids(wanted_ids)
            elapsed = int((time.perf_counter() - started) * 1000)
            iterations = [
                *state.get("iterations", []),
                {
                    "iteration": state.get("iteration", 0) + 1,
                    "trigger": "graph_traversal",
                    "pulled_sections": pulled,
                    "retrieved_chunks": [c.to_trace() for c in fetched],
                    "latency_ms": elapsed,
                },
            ]
            return {
                "context": merge_context(context, fetched),
                "iteration": state.get("iteration", 0) + 1,
                "iterations": iterations,
                "refine_strategy": "graph_traversal",
                "active_queries": [],
            }

        # Otherwise rewrite the query and search again.
        previous = state.get("active_queries") or [state["raw_query"]]
        try:
            out = deps.llm.parse(
                system=REWRITE_SYSTEM,
                user=(
                    f"Original question: {state['raw_query']}\n"
                    f"Query that failed: {previous[0]}\n"
                    f"What was missing: {state.get('missing_info', 'unknown')}"
                ),
                schema=RewrittenQuery,
                node="refine",
                max_tokens=512,
            )
            rewritten = out.rewritten_query.strip() or previous[0]
            rationale = out.rationale
        except Exception:  # noqa: BLE001
            # Widening with the original wording is a weak but safe fallback.
            rewritten = f"{state['raw_query']} {state.get('missing_info', '')}".strip()
            rationale = "rewrite failed; widened with the original question"

        return {
            "active_queries": [rewritten],
            "refine_strategy": "query_rewrite",
            "iterations": [
                *state.get("iterations", []),
                {
                    "iteration": state.get("iteration", 0) + 1,
                    "trigger": "query_rewrite",
                    "rewritten_query": rewritten,
                    "rationale": rationale,
                },
            ],
        }

    return refine


# ------------------------------------------------------------------------- generate


def make_generate(deps: AgentDeps):
    def generate(state: AgentState) -> dict[str, Any]:
        context = state.get("context", [])[:MAX_CONTEXT_CHUNKS]

        if not state.get("needs_retrieval", True):
            # Bulleted rather than the prompt's indented form: the answer renderer
            # groups consecutive bullet lines into a list, and would otherwise run the
            # fifteen roster lines together into one paragraph.
            roster_lines = "\n".join(
                f"- {line.strip()}" for line in deps.roster_text.splitlines() if line.strip()
            )
            return {
                "summary": "I answer only from the clauses in the fifteen indexed "
                "agreements, so I cannot help with that one.",
                "final_answer": "\n".join(
                    [
                        "The corpus holds these contracts:",
                        "",
                        roster_lines,
                        "",
                        "Ask about obligations, definitions, termination, payment terms "
                        "or cross-references in any of them.",
                    ]
                ),
                "citations": [],
                "unverified_citations": [],
                "confidence": "high",
            }

        if not context:
            return {
                "summary": "I could not find any clauses in the indexed contracts "
                "relevant to that question.",
                "final_answer": (
                    "Retrieval returned nothing usable. The subject may not be covered "
                    "by these fifteen agreements, or the wording may not match how the "
                    "clauses are drafted -- try naming the company, the section number, "
                    "or the legal term the contract would use."
                ),
                "citations": [],
                "unverified_citations": [],
                "confidence": "low",
                "low_confidence": True,
            }

        low_confidence = bool(state.get("low_confidence"))
        user = [f"Question: {state['raw_query']}", "", "Clauses:", "", format_clauses(context)]
        if low_confidence and state.get("missing_info"):
            user += [
                "",
                "NOTE: retrieval hit its iteration limit and this context may be "
                f"incomplete. Specifically: {state['missing_info']}. Answer from what "
                "is present and state what could not be confirmed.",
            ]

        try:
            out = deps.llm.parse(
                system=GENERATE_SYSTEM,
                user="\n".join(user),
                schema=GeneratedAnswer,
                node="generate",
                max_tokens=2048,
            )
            summary, answer, confidence = out.summary.strip(), out.answer, out.confidence
        except Exception as exc:  # noqa: BLE001
            return {
                "summary": "",
                "final_answer": f"Answer generation failed: {exc}",
                "citations": [],
                "unverified_citations": [],
                "confidence": "low",
                "low_confidence": True,
                "error": str(exc),
            }

        # Cheap mechanical check before the LLM faithfulness pass: does every citation
        # tag name a chunk that was actually in context? (Architecture.md §7, node 5)
        # The summary is checked on the same terms as the body: it is the line most
        # readers will act on, so an unverifiable citation there matters most.
        in_context = {(c.doc_id, c.section_id): c for c in context}
        citations: list[dict[str, Any]] = []
        unverified: list[str] = []
        for doc_id, section_id in RE_CITATION.findall(f"{summary}\n\n{answer}"):
            key = (doc_id, section_id)
            tag = f"[{doc_id} §{section_id}]"
            if key in in_context:
                chunk = in_context[key]
                if not any(c["chunk_id"] == chunk.chunk_id for c in citations):
                    citations.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": doc_id,
                            "section_id": section_id,
                            "section_title": chunk.payload.get("section_title", ""),
                            "verified": True,
                        }
                    )
            elif tag not in unverified:
                unverified.append(tag)

        if unverified:
            answer += (
                "\n\n⚠️ Citations that do not correspond to a retrieved clause and could "
                "not be verified: " + ", ".join(unverified)
            )

        return {
            "summary": summary,
            "final_answer": answer,
            "citations": citations,
            "unverified_citations": unverified,
            "confidence": "low" if low_confidence else confidence,
        }

    return generate


# --------------------------------------------------------------------- faithfulness


def make_faithfulness(deps: AgentDeps):
    def faithfulness(state: AgentState) -> dict[str, Any]:
        # The summary carries claims like any other line, and is the one line a reader
        # may act on without reading further -- so it is checked, not exempted.
        answer = "\n".join(filter(None, [state.get("summary", ""), state.get("final_answer", "")]))
        citations = state.get("citations", [])
        if not answer or not citations:
            return {"faithfulness": {"claims_checked": 0, "flagged": [], "skipped": True}}

        by_key = {(c.doc_id, c.section_id): c for c in state.get("context", [])}

        # Each claim is paired with ONLY its own cited clause. Passing the whole
        # context would let a claim be "supported" by some other clause, which turns
        # a per-citation check into a vague overall grounding score.
        claims: list[str] = []
        for sentence in _split_sentences(answer):
            found = RE_CITATION.findall(sentence)
            if not found:
                continue
            doc_id, section_id = found[0]
            chunk = by_key.get((doc_id, section_id))
            if chunk is None:
                continue
            claims.append(
                f"CLAIM {len(claims) + 1}: {sentence.strip()}\n"
                f"CITED CLAUSE [{doc_id} §{section_id}]:\n{chunk.text}"
            )

        if not claims:
            return {"faithfulness": {"claims_checked": 0, "flagged": [], "skipped": True}}

        try:
            out = deps.llm.parse(
                system=FAITHFULNESS_SYSTEM,
                user="\n\n---\n\n".join(claims),
                schema=FaithfulnessOutput,
                node="faithfulness",
                max_tokens=2048,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "faithfulness": {
                    "claims_checked": len(claims),
                    "flagged": [],
                    "error": str(exc),
                }
            }

        flagged = [
            {"claim": c.claim, "cited_section": c.cited_section, "issue": c.issue}
            for c in out.claims
            if not c.supported
        ]
        return {
            "faithfulness": {
                "claims_checked": len(out.claims),
                "flagged": flagged,
            },
            "token_usage": deps.usage.total_tokens(),
            "latency_ms": {**state.get("latency_ms", {}), **deps.usage.latency_by_node()},
        }

    return faithfulness


def _split_sentences(text: str) -> list[str]:
    """Split into claims, without breaking on 'Section 4.2.' or '[doc §8.2].'

    Lines are split first, then sentences within a line. Now that answers are written
    as bullets, a line break is the stronger claim boundary: a bullet often has no
    terminal full stop, and the next one starts with "- **Quorum**" rather than a
    capital letter, so sentence splitting alone would fuse several bullets into one
    claim and check them all against whichever clause the first one cited.
    """
    out: list[str] = []
    for raw in text.split("\n"):
        # Drop the bullet marker and the bold label's asterisks: the checker judges the
        # claim, and the flagged-claim UI shows this text back to the user, so neither
        # should carry markdown punctuation.
        line = re.sub(r"^\s*[-*•]\s+", "", raw).replace("**", "").strip()
        if line:
            out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", line) if s.strip()]
    return out
