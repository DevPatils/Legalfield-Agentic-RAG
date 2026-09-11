"""Embedding providers behind one interface.

Architecture.md §4 leaves the choice open between ``voyage-law-2`` (legal-domain, the
interesting answer) and ``text-embedding-3-large``. Keeping it behind a protocol means
swapping is a .env change, and means the eval can be re-run on both to show the
domain model actually earns its place.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Settings, get_settings

# Retrieval quality depends on the query and the document being embedded for the
# same task; both providers below distinguish the two.
QUERY = "query"
DOCUMENT = "document"


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], kind: str = DOCUMENT) -> list[list[float]]: ...


class VoyageEmbeddings:
    """voyage-law-2 -- trained on legal text, 1024-dim."""

    def __init__(self, api_key: str, model: str = "voyage-law-2") -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dim = 1024

    def embed(self, texts: list[str], kind: str = DOCUMENT) -> list[list[float]]:
        input_type = "query" if kind == QUERY else "document"
        out: list[list[float]] = []
        # Voyage caps batch size; chunk conservatively.
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            result = self.client.embed(batch, model=self.model, input_type=input_type)
            out.extend(result.embeddings)
        return out


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


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY is unset.")
        return VoyageEmbeddings(settings.voyage_api_key, settings.embedding_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset.")
        return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
