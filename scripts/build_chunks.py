"""Parse data/raw/*.txt into data/processed/chunks.jsonl and report on the graph.

Day 1's deliverable (Architecture.md §12) is the chunk file *plus* a manual check that
the cross-reference edges are real. The ``--inspect`` mode prints sampled edges with
both endpoints' text so that check takes a minute instead of an afternoon.

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --inspect 10
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ingest.chunker import build_corpus  # noqa: E402
from app.ingest.models import Chunk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "chunks.jsonl"


def report(chunks: list[Chunk]) -> None:
    by_doc: Counter[str] = Counter(c.doc_id for c in chunks)
    edges = sum(len(c.cross_reference_ids) for c in chunks)
    def_edges = sum(len(c.definition_ids) for c in chunks)
    with_edges = sum(1 for c in chunks if c.cross_reference_ids)
    defined = {t for c in chunks for t in c.defines_terms}
    lengths = sorted(len(c.text) for c in chunks)

    print("\n=== Corpus ===")
    print(f"documents            {len(by_doc)}")
    print(f"chunks               {len(chunks)}")
    print(f"chunks/doc  median   {sorted(by_doc.values())[len(by_doc) // 2]}")
    print(f"chunk chars median   {lengths[len(lengths) // 2]}")
    print(f"chunk chars p95      {lengths[int(len(lengths) * 0.95)]}")

    print("\n=== Cross-reference graph ===")
    print(f"section edges        {edges}")
    print(f"definition edges     {def_edges}")
    print(f"chunks with edges    {with_edges} ({100 * with_edges / max(len(chunks), 1):.0f}%)")
    print(f"distinct defined terms {len(defined)}")

    # A dangling edge means resolution produced an id that was never indexed --
    # that would silently break graph traversal at query time.
    known = {c.chunk_id for c in chunks}
    dangling = [
        (c.chunk_id, e)
        for c in chunks
        for e in (*c.cross_reference_ids, *c.definition_ids)
        if e not in known
    ]
    print(f"dangling edges       {len(dangling)}" + ("  <-- BUG" if dangling else "  (ok)"))
    for src, dst in dangling[:5]:
        print(f"    {src} -> {dst}")


def inspect(chunks: list[Chunk], n: int, seed: int = 0) -> None:
    """Print sampled edges with both endpoints so they can be eyeballed."""
    by_id = {c.chunk_id: c for c in chunks}
    candidates = [c for c in chunks if c.cross_reference_ids]
    if not candidates:
        print("\nNo cross-reference edges to inspect.")
        return
    rng = random.Random(seed)
    for chunk in rng.sample(candidates, min(n, len(candidates))):
        target_id = chunk.cross_reference_ids[0]
        target = by_id.get(target_id)
        print("\n" + "=" * 78)
        print(f"SOURCE {chunk.chunk_id}  ({chunk.section_title or 'untitled'})")
        print(f"  {_excerpt(chunk.text)}")
        print(f"  refs: {chunk.cross_references}")
        if chunk.defined_terms_used:
            print(f"  uses defined terms: {chunk.defined_terms_used[:4]}")
        print(f"TARGET {target_id}  ({target.section_title if target else 'MISSING'})")
        if target:
            print(f"  {_excerpt(target.text)}")


def _excerpt(text: str, limit: int = 260) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=int, default=0, help="sample N edges to eyeball")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not any(args.raw_dir.glob("*.txt")):
        raise SystemExit(f"No .txt files in {args.raw_dir}. Run scripts/fetch_cuad.py first.")

    chunks = build_corpus(args.raw_dir, args.out)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")
    report(chunks)
    if args.inspect:
        inspect(chunks, args.inspect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
