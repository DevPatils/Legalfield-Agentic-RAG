"""Embedding providers behind one interface.

Architecture.md §4 leaves the choice open between ``voyage-law-2`` (legal-domain, the
interesting answer) and ``text-embedding-3-large``. Keeping it behind a protocol means
swapping is a .env change, and means the eval can be re-run on both to show the
domain model actually earns its place.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from ..config import Settings, get_settings
from .ratelimit import RateLimiter, batch_by_tokens, estimate_tokens

# Retrieval quality depends on the query and the document being embedded for the
# same task; both providers below distinguish the two.
QUERY = "query"
DOCUMENT = "document"


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], kind: str = DOCUMENT) -> list[list[float]]: ...


class VoyageEmbeddings:
    """voyage-law-2 -- trained on legal text, 1024-dim.

    Batching is driven by the token ceiling rather than a fixed item count: chunk
    sizes vary by an order of magnitude (a one-line cross-reference vs. a four-page
    indemnity clause), so a fixed batch of N can be well under or well over the
    per-request budget depending on which clauses land in it.
    """

    MAX_ITEMS_PER_REQUEST = 96

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-law-2",
        limiter: RateLimiter | None = None,
        max_request_tokens: int = 8_000,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dim = 1024
        self.limiter = limiter
        self.max_request_tokens = max_request_tokens
        self.progress = progress

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Exact per-text token counts via Voyage's own tokenizer (local and free).

        Worth doing: the chars/4 heuristic undershoots legal text by ~10%, and an
        undershoot means requests the limiter believes fit are rejected by the server,
        costing a full backoff cycle each time. Falls back to the heuristic if the
        tokenizer is unavailable.
        """
        try:
            return [self.client.count_tokens([t], model=self.model) for t in texts]
        except Exception:
            return [estimate_tokens(t) for t in texts]

    def embed(self, texts: list[str], kind: str = DOCUMENT) -> list[list[float]]:
        import voyageai

        input_type = "query" if kind == QUERY else "document"
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        costs = self.count_tokens_batch(texts)

        for indices in batch_by_tokens(
            costs, self.max_request_tokens, self.MAX_ITEMS_PER_REQUEST
        ):
            batch = [texts[i] for i in indices]
            cost = sum(costs[i] for i in indices)
            if self.limiter:
                self.limiter.acquire(cost, on_wait=self._note_wait)

            for attempt in range(6):
                try:
                    result = self.client.embed(batch, model=self.model, input_type=input_type)
                    break
                except voyageai.error.RateLimitError:
                    # The limiter estimates token cost; the server counts exactly.
                    # Back off and retry rather than losing the batch.
                    delay = min(60, 5 * 2**attempt)
                    self._note_wait(delay)
                    time.sleep(delay)
            else:
                raise RuntimeError("Voyage rate limit persisted after 6 retries.")

            for slot, vector in zip(indices, result.embeddings, strict=True):
                out[slot] = vector
        return out

    def _note_wait(self, seconds: float) -> None:
        if self.progress:
            self.progress(f"rate limit: waiting {seconds:.0f}s")


class OpenAIEmbeddings:
    """text-embedding-3-large -- general purpose, 3072-dim."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = 3072

    def embed(self, texts: list[str], kind: str = DOCUMENT) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = [t.replace("\n", " ") for t in texts[i : i + 128]]
            result = self.client.embeddings.create(input=batch, model=self.model)
            out.extend(item.embedding for item in result.data)
        return out


def get_embedding_provider(
    settings: Settings | None = None, progress: Callable[[str], None] | None = None
) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY is unset.")
        return VoyageEmbeddings(
            settings.voyage_api_key,
            settings.embedding_model,
            limiter=RateLimiter(settings.embed_max_rpm, settings.embed_max_tpm),
            # Leave headroom under the per-minute ceiling so one request never
            # consumes the entire budget on its own.
            max_request_tokens=max(1_000, int(settings.embed_max_tpm * 0.8)),
            progress=progress,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset.")
        return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
