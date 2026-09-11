"""Reciprocal Rank Fusion, written out rather than imported (Architecture.md §6).

    score(d) = Σ_i  1 / (k + rank_i(d))

Why RRF rather than normalizing and summing the retrievers' own scores: cosine
similarity and BM25 are on incomparable scales, and BM25's range shifts with corpus
statistics, so any weighted score blend needs re-tuning per corpus. RRF only reads
*rank*, so it is scale-free. ``k`` (default 60) damps the head: without it the top
rank dominates, and a single retriever's #1 would win every fusion outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedResult:
    key: str
    score: float
    # Per-retriever rank, e.g. {"dense": 3, "sparse": 11}. Surfaced in the trace UI
    # so it is visible *why* something ranked where it did.
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedResult]:
    """Fuse ranked id lists into one ordering.

    ``ranked_lists`` maps a retriever name to its ids in rank order (best first).
    Ties break on the best single rank achieved, so a document ranked #1 by one
    retriever outranks one that was #5 in both when their fused scores collide.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for retriever, ids in ranked_lists.items():
        weight = weights.get(retriever, 1.0)
        for zero_based, key in enumerate(ids):
            rank = zero_based + 1  # RRF is 1-indexed; rank 0 would divide unevenly
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            ranks.setdefault(key, {})[retriever] = rank

    fused = [FusedResult(key=key, score=score, ranks=ranks[key]) for key, score in scores.items()]
    fused.sort(key=lambda r: (-r.score, min(r.ranks.values()), r.key))
    return fused
