"""The retrieval pipeline of Architecture.md §6: dense + sparse -> RRF -> rerank.

``RetrievalConfig`` exposes the stages as switches because the headline eval number is
a *delta* (§11): recall@k before vs. after hybrid+rerank. That comparison is only
honest if the ablations run through this same code path rather than a parallel
implementation written for the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..embeddings.provider import QUERY, EmbeddingProvider
from ..storage.qdrant_store import Hit, QdrantStore
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .rerank import Reranker


@dataclass
class RetrievalConfig:
    top_n: int = 40          # candidates per retriever, pre-fusion
    top_k: int = 6           # returned after rerank
    rrf_k: int = 60
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """One result, carrying the provenance the trace panel renders."""

    chunk_id: str
    score: float
    payload: dict[str, Any]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    source: str = "search"  # "search" | "graph_traversal"

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def section_id(self) -> str:
        return self.payload.get("section_id", "")

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")

    @property
    def citation(self) -> str:
        return f"[{self.doc_id} §{self.section_id}]"

    def to_trace(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "section_title": self.payload.get("section_title", ""),
            "score": round(self.score, 4),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "fused_score": round(self.fused_score, 5) if self.fused_score is not None else None,
            "rerank_score": (
                round(self.rerank_score, 4) if self.rerank_score is not None else None
            ),
            "source": self.source,
            "excerpt": self.text[:300],
        }


class RetrievalPipeline:
    def __init__(
        self,
        store: QdrantStore,
        embedder: EmbeddingProvider,
        bm25: BM25Index,
        reranker: Reranker,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings or get_settings()

    def retrieve(
        self, query: str, config: RetrievalConfig | None = None, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        cfg = config or RetrievalConfig(
            top_n=self.settings.retrieve_top_n,
            top_k=self.settings.rerank_top_k,
            rrf_k=self.settings.rrf_k,
        )

        ranked_lists: dict[str, list[str]] = {}
        payloads: dict[str, dict[str, Any]] = {}

        if cfg.use_dense:
            vector = self.embedder.embed([query], kind=QUERY)[0]
            dense_hits = self.store.search(vector, limit=cfg.top_n, doc_id=doc_id)
            ranked_lists["dense"] = [h.chunk_id for h in dense_hits]
            payloads.update({h.chunk_id: h.payload for h in dense_hits})

        if cfg.use_sparse:
            sparse_hits = self.bm25.search(query, limit=cfg.top_n)
            sparse_ids = [cid for cid, _ in sparse_hits]
            if doc_id:
                sparse_ids = [
                    cid for cid in sparse_ids if self.bm25.payloads[cid]["doc_id"] == doc_id
                ]
            ranked_lists["sparse"] = sparse_ids
            for cid in sparse_ids:
                payloads.setdefault(cid, self.bm25.payloads[cid])

        if not ranked_lists:
            raise ValueError("RetrievalConfig disabled every retriever.")

        fused = reciprocal_rank_fusion(ranked_lists, k=cfg.rrf_k, weights=cfg.weights)
        candidates = fused[: cfg.top_n]

        results = [
            RetrievedChunk(
                chunk_id=f.key,
                score=f.score,
                payload=payloads.get(f.key, {}),
                dense_rank=f.ranks.get("dense"),
                sparse_rank=f.ranks.get("sparse"),
                fused_score=f.score,
            )
            for f in candidates
        ]
        # A single retriever needs no fusion step to be meaningful, but RRF still
        # produces a valid ordering, so the ablation paths stay identical.
        if not cfg.use_rerank or len(results) <= 1:
            return results[: cfg.top_k]

        ranked = self.reranker.rerank(query, [r.text for r in results], top_k=cfg.top_k)
        out: list[RetrievedChunk] = []
        for index, relevance in ranked:
            chunk = results[index]
            chunk.rerank_score = relevance
            chunk.score = relevance
            out.append(chunk)
        return out

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Exact fetch for the Refine node's graph traversal -- no search involved."""
        hits: list[Hit] = self.store.fetch_by_chunk_ids(chunk_ids)
        return [
            RetrievedChunk(
                chunk_id=h.chunk_id, score=1.0, payload=h.payload, source="graph_traversal"
            )
            for h in hits
        ]
