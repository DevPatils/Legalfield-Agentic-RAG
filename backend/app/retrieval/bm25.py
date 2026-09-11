"""In-process BM25 index over the chunk corpus.

Architecture.md §6 allows BM25 either via ``rank_bm25`` or Qdrant sparse vectors.
``rank_bm25`` is the right trade here: the corpus is 10-20 contracts (low thousands of
chunks), so the index is a few MB and builds in well under a second, and it avoids
standing up a SPLADE model just to produce sparse vectors.

Sparse retrieval is not a formality in this domain -- legal queries lean on exact
tokens ("Section 8.2", "indemnify", "Force Majeure") that dense embeddings blur.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

# Keeps section numbers ("8.2") and hyphenated legal compounds intact as single tokens.
RE_TOKEN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)*")

# Deliberately short: legal text is formulaic, and dropping "shall"/"party" would
# discard real signal. Only true function words are removed.
STOPWORDS = frozenset(
    """a an the of to in for on by or and is are be been was were with as at from
    that this these those it its such any all each other than then so if not no
    will would may might can could shall should must have has had do does did""".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for raw in RE_TOKEN.findall(text.lower())
        if (token := raw) not in STOPWORDS and len(token) > 1
    ]


class BM25Index:
    """Wraps BM25Okapi with a chunk_id mapping."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.chunk_ids = [p["chunk_id"] for p in payloads]
        self.payloads = {p["chunk_id"]: p for p in payloads}
        tokenized = [tokenize(self._document_text(p)) for p in payloads]
        self.token_sets = [set(tokens) for tokens in tokenized]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _document_text(payload: dict[str, Any]) -> str:
        """Index the heading alongside the body.

        The section number and title carry a lot of the lexical signal for queries
        like "what does Section 8.2 say", and they live outside the clause body.
        """
        return " ".join(
            filter(
                None,
                [
                    payload.get("section_id", ""),
                    payload.get("section_title", ""),
                    payload.get("text", ""),
                ],
            )
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` best-first."""
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        query_tokens = set(tokens)
        # Filter on actual token overlap, not on score sign. BM25 IDF is
        # log((N-df+0.5)/(df+0.5)), which is legitimately <= 0 for a term that appears
        # in most documents -- so a `score > 0` filter would silently drop real
        # matches on common legal vocabulary and behave differently as the corpus grows.
        candidates = [
            (cid, float(score))
            for cid, score, doc_tokens in zip(
                self.chunk_ids, scores, self.token_sets, strict=True
            )
            if query_tokens & doc_tokens
        ]
        candidates.sort(key=lambda row: -row[1])
        return candidates[:limit]
