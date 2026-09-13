"""Ask the agent a question and watch the reasoning trace.

The terminal stand-in for the React trace panel (Architecture.md §9) -- same
information, same order, streamed from the same ``graph.stream()`` the SSE endpoint
will use. Useful for manual testing before the UI exists, and for the eval loop after.

Usage:
    python scripts/ask.py "Can the JGC waive compliance with the agreement?"
    python scripts/ask.py "What is Confidential Information?" --doc doc_001
    python scripts/ask.py                      # interactive; blank line or Ctrl-D exits
    python scripts/ask.py --examples           # suggested questions to try
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# Contract clauses use § and en-dashes; Windows consoles default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.deps import AgentDeps, build_deps  # noqa: E402
from app.agent.graph import stream_query  # noqa: E402

BAR = "─" * 74

EXAMPLES = [
    ("cross-reference", "How may this agreement be amended or waived?"),
    ("cross-reference", "Can the Joint Governance Committee waive compliance?"),
    ("definitional", "What counts as Confidential Information?"),
    ("definitional", "What is a Detail Report and where is it specified?"),
    ("comparative", "How do termination provisions differ across these agreements?"),
    ("single-clause", "What are the obligations around pharmacovigilance?"),
    ("out-of-scope", "What is the capital of France?"),
    ("out-of-scope", "Hello, what can you do?"),
]


def show(node: str, partial: dict) -> None:
    """Render one node's output the way the trace panel will."""
    if node == "planner":
        kind = partial.get("query_type", "?")
        retr = partial.get("needs_retrieval")
        print(f"\n  PLANNER      {kind}   needs_retrieval={retr}")
        if partial.get("planner_reasoning"):
            print(f"               {partial['planner_reasoning']}")
        for sub in partial.get("sub_queries", []):
            print(f"               → {sub}")

    elif node == "retrieve":
        iterations = partial.get("iterations", [])
        latest = iterations[-1] if iterations else {}
        hits = latest.get("retrieved_chunks", [])
        trigger = latest.get("trigger", "none")
        label = "RETRIEVE" if trigger in ("none", "") else f"RETRIEVE ({trigger})"
        print(f"\n  {label:<12} {len(hits)} hits, {len(partial.get('context', []))} in context")
        for hit in hits[:6]:
            ranks = f"d={hit.get('dense_rank')} s={hit.get('sparse_rank')}"
            title = (hit.get("section_title") or "")[:34]
            print(f"      [{hit['doc_id']} §{hit['section_id']}]  {ranks:<12} {title}")

    elif node == "sufficiency":
        ok = partial.get("sufficient")
        print(f"\n  SUFFICIENCY  {'SUFFICIENT' if ok else 'INSUFFICIENT'}")
        if not ok:
            if partial.get("missing_info"):
                print(f"               missing: {partial['missing_info']}")
            if partial.get("sections_needed"):
                print(f"               sections needed: {partial['sections_needed']}")
            if partial.get("terms_needed"):
                print(f"               terms needed: {partial['terms_needed']}")

    elif node == "refine":
        strategy = partial.get("refine_strategy", "?")
        latest = (partial.get("iterations") or [{}])[-1]
        if strategy == "graph_traversal":
            pulled = latest.get("pulled_sections", [])
            print(f"\n  REFINE       GRAPH TRAVERSAL → pulled {pulled}")
            for hit in latest.get("retrieved_chunks", []):
                title = (hit.get("section_title") or "")[:40]
                print(f"      [{hit['doc_id']} §{hit['section_id']}]  {title}")
        else:
            print(f"\n  REFINE       QUERY REWRITE → {latest.get('rewritten_query', '?')!r}")
            if latest.get("rationale"):
                print(f"               {latest['rationale']}")

    elif node == "flag" and partial.get("low_confidence"):
        print("\n  FLAG         low confidence — answering from incomplete context")

    elif node == "faithfulness":
        report = partial.get("faithfulness", {})
        checked = report.get("claims_checked", 0)
        flagged = report.get("flagged", [])
        status = "all supported" if not flagged else f"{len(flagged)} FLAGGED"
        print(f"\n  FAITHFULNESS {checked} claims checked — {status}")
        for flag in flagged:
            print(f"      ⚠ [{flag.get('cited_section')}] {flag.get('claim', '')[:90]}")
            print(f"        {flag.get('issue', '')}")


def ask(deps: AgentDeps, question: str, doc_id: str | None) -> None:
    print(f"\n{BAR}\n  {question}\n{BAR}")
    started = time.time()
    final: dict = {}
    try:
        for node, partial in stream_query(
            deps, question, query_id=str(uuid.uuid4()), doc_id=doc_id
        ):
            final.update(partial)
            show(node, partial)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  run failed: {exc}")
        return

    print(f"\n{BAR}\n  ANSWER\n{BAR}\n")
    print(final.get("final_answer", "(no answer)"))

    cites = final.get("citations", [])
    if cites:
        print("\n  Citations:")
        for c in cites:
            print(f"    [{c['doc_id']} §{c['section_id']}]  {c.get('section_title', '')}")
    if final.get("unverified_citations"):
        print(f"\n  ⚠ Unverified citations: {final['unverified_citations']}")

    usage = final.get("token_usage", {})
    print(
        f"\n  {time.time() - started:.1f}s"
        f" · {final.get('iteration', 0)} iteration(s)"
        f" · {len(final.get('context', []))} clauses in context"
        f" · confidence={final.get('confidence', '?')}"
        f" · tokens {usage.get('input', 0)}in/{usage.get('output', 0)}out"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question; omit for interactive mode")
    ap.add_argument("--doc", default=None, help="restrict to one document, e.g. doc_001")
    ap.add_argument("--examples", action="store_true", help="print suggested questions")
    args = ap.parse_args()

    if args.examples:
        print("\nQuestions worth trying:\n")
        for kind, q in EXAMPLES:
            print(f"  {kind:<16} {q}")
        print()
        return 0

    print("Loading index and building BM25...")
    deps = build_deps()
    print(f"Ready — {len(deps.bm25.chunk_ids)} chunks, model {deps.settings.llm_model}")

    if args.question:
        ask(deps, " ".join(args.question), args.doc)
        return 0

    print("Interactive mode. Blank line or Ctrl-D to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        ask(deps, question, args.doc)


if __name__ == "__main__":
    sys.exit(main())
