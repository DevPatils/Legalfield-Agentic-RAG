"""Cross-encoder reranking of the fused candidate set.

Dense and sparse retrieval both score a query against a *precomputed* document
representation. A cross-encoder instead reads the query and the clause together, which
is what lets it tell "termination for convenience" from "termination for cause" --
a distinction the bi-encoder collapses because the two clauses are near-identical in
embedding space but opposite in meaning. That is exactly the error mode that matters
in contracts, so the rerank stage is doing real work here, not just polishing.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Settings, get_settings


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return ``(original_index, relevance)`` best-first, at most ``top_k``."""
        ...


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        import cohere

        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        result = self.client.rerank(
            model=self.model, query=query, documents=documents, top_n=min(top_k, len(documents))
        )
        return [(r.index, float(r.relevance_score)) for r in result.results]


class LocalReranker:
    """bge-reranker-large via sentence-transformers. No API key; needs torch."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        scores = self.model.predict([(query, doc) for doc in documents])
        ranked = sorted(enumerate(float(s) for s in scores), key=lambda row: -row[1])
        return ranked[:top_k]


class NoopReranker:
    """Passthrough. Keeps the pipeline runnable without a rerank provider, and is
    what the eval uses to measure the rerank stage's contribution (Architecture.md §11).
    """

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        return [(i, 1.0 / (i + 1)) for i in range(min(top_k, len(documents)))]


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    provider = settings.rerank_provider.lower()
    if provider == "none":
        return NoopReranker()
    if provider == "cohere":
        if not settings.cohere_api_key:
            raise RuntimeError("RERANK_PROVIDER=cohere but COHERE_API_KEY is unset.")
        return CohereReranker(settings.cohere_api_key)
    if provider == "local":
        return LocalReranker()
    raise ValueError(f"Unknown RERANK_PROVIDER: {settings.rerank_provider!r}")
