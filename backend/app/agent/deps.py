"""Everything the agent nodes need, assembled once.

Built explicitly rather than imported at module scope so tests can substitute fakes
and the FastAPI layer can hold a single instance across requests -- the BM25 index in
particular is expensive enough that rebuilding it per query would dominate latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..embeddings.provider import EmbeddingProvider, get_embedding_provider
from ..ingest.roster import DocumentInfo, format_roster, load_roster
from ..retrieval.bm25 import BM25Index
from ..retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from ..retrieval.rerank import Reranker, get_reranker
from ..storage.qdrant_store import QdrantStore
from .llm import LLMClient, UsageLog, get_llm


@dataclass
class AgentDeps:
    settings: Settings
    store: QdrantStore
    embedder: EmbeddingProvider
    bm25: BM25Index
    reranker: Reranker
    llm: LLMClient
    usage: UsageLog
    pipeline: RetrievalPipeline
    roster: tuple[DocumentInfo, ...] = ()

    @property
    def roster_text(self) -> str:
        return format_roster(self.roster)

    def retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            top_n=self.settings.retrieve_top_n,
            top_k=self.settings.rerank_top_k,
            rrf_k=self.settings.rrf_k,
            use_rerank=self.settings.rerank_provider.lower() != "none",
        )


def build_deps(settings: Settings | None = None) -> AgentDeps:
    """Wire the real stack. Requires a populated Qdrant collection."""
    settings = settings or get_settings()
    store = QdrantStore(settings)
    embedder = get_embedding_provider(settings)
    reranker = get_reranker(settings)
    usage = UsageLog()
    llm = get_llm(usage, settings)

    payloads = store.all_chunks()
    if not payloads:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} is empty. "
            "Run scripts/build_index.py first."
        )
    bm25 = BM25Index(payloads)

    return AgentDeps(
        settings=settings,
        store=store,
        embedder=embedder,
        bm25=bm25,
        reranker=reranker,
        llm=llm,
        usage=usage,
        pipeline=RetrievalPipeline(store, embedder, bm25, reranker, settings),
        roster=load_roster(settings.raw_dir / "manifest.json"),
    )
