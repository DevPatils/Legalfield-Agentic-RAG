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

            # A full pass takes over an hour at the free-tier cap, so a transient
            # network drop part-way through is a realistic event. Retrying the batch
            # costs seconds; letting the exception escape costs the whole run.
            transient = (
                voyageai.error.RateLimitError,
                voyageai.error.APIConnectionError,
                voyageai.error.ServiceUnavailableError,
                voyageai.error.Timeout,
            )
            # Patience is close to free here: the run is already rate-capped, and the
            # alternative to waiting is aborting a job with an hour of work behind it.
            # The limiter's window is per-process, so a resumed run starts blind to
            # requests the previous process made, and the server can throttle for
            # several minutes until those age out.
            for attempt in range(14):
                try:
                    result = self.client.embed(batch, model=self.model, input_type=input_type)
                    break
                except transient as exc:
                    kind = type(exc).__name__
                    delay = min(120, 5 * 2**attempt)
                    self._note_wait_labelled(delay, kind)
                    time.sleep(delay)
            else:
                raise RuntimeError("Voyage kept failing after 14 retries; run again to resume.")

            for slot, vector in zip(indices, result.embeddings, strict=True):
                out[slot] = vector
        return out

    def _note_wait(self, seconds: float) -> None:
        if self.progress:
            self.progress(f"rate limit: waiting {seconds:.0f}s")

    def _note_wait_labelled(self, seconds: float, kind: str) -> None:
        if self.progress:
            self.progress(f"{kind}: retrying in {seconds:.0f}s")


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
        # Size each request at roughly TPM / RPM, so the allowed number of requests
        # per minute adds up to the token budget and no single request can consume it.
        # Sizing to a large fraction of TPM instead means one request nearly exhausts
        # the minute, and any undercount in the local estimate trips a server refusal
        # -- which then costs a full backoff cycle and, repeated, stalls the run.
        per_request = settings.embed_max_tpm / max(settings.embed_max_rpm, 1)
        return VoyageEmbeddings(
            settings.voyage_api_key,
            settings.embedding_model,
            limiter=RateLimiter(settings.embed_max_rpm, settings.embed_max_tpm),
            max_request_tokens=max(1_000, int(per_request * 0.9)),
            progress=progress,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset.")
        return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
