"""Qdrant access layer.

Qdrant owns vectors, chunk text, chunk metadata, and the cross-reference graph --
all of it as point payload. MongoDB never holds any of this (Architecture.md §14).

Two distinct read paths matter here:

* ``search`` -- ANN over dense vectors, used by the Retrieve node.
* ``fetch_by_chunk_ids`` -- exact lookup, used by the Refine node's graph traversal.
  Traversal already knows exactly which clause it needs, so re-searching for it would
  be slower and less precise.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from ..config import Settings, get_settings
from ..ingest.models import Chunk

NAMESPACE = uuid.UUID("6f1b8a1e-4f3a-4a2b-9a6e-2c1d4f5a7b30")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk_id -- Qdrant point ids must be int or UUID."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]

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


class QdrantStore:
    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or QdrantClient(url=self.settings.qdrant_url)
        self.collection = self.settings.qdrant_collection

    # --- write path ---

    def recreate_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # Payload indexes for the filters the agent actually uses.
        for field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("chunk_id", models.PayloadSchemaType.KEYWORD),
            ("section_id", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(self.collection, field, field_schema=schema)

    def ensure_collection(self, dim: int) -> bool:
        """Create the collection if absent. Returns True if it already existed with
        the right dimension, meaning stored vectors can be reused."""
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim == dim:
                return True
            # A different embedding model: old vectors are not comparable to new ones.
            self.client.delete_collection(self.collection)
        self.recreate_collection(dim)
        return False

    def existing_hashes(self) -> dict[str, str]:
        """``{chunk_id: content_hash}`` for everything already indexed.

        Lets a re-index skip chunks whose embedding input has not changed, which makes
        an interrupted run resumable instead of a restart.
        """
        out: dict[str, str] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=["chunk_id", "content_hash"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                if payload.get("chunk_id"):
                    out[payload["chunk_id"]] = payload.get("content_hash", "")
            if offset is None:
                return out

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        for i in range(0, len(chunk_ids), 256):
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(
                    points=[point_id(cid) for cid in chunk_ids[i : i + 256]]
                ),
                wait=True,
            )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        hashes: list[str] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            payload = chunk.to_payload()
            if hashes:
                payload["content_hash"] = hashes[i]
            points.append(
                models.PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=payload)
            )
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128], wait=True)

    # --- read paths ---

    def search(self, vector: list[float], limit: int, doc_id: str | None = None) -> list[Hit]:
        flt = (
            models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
            if doc_id
            else None
        )
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [
            Hit(chunk_id=p.payload["chunk_id"], score=p.score, payload=p.payload)
            for p in result.points
        ]

    def fetch_by_chunk_ids(self, chunk_ids: list[str]) -> list[Hit]:
        """Exact fetch used by graph traversal. Preserves the requested order."""
        if not chunk_ids:
            return []
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id(cid) for cid in chunk_ids],
            with_payload=True,
        )
        by_id = {r.payload["chunk_id"]: r for r in records if r.payload}
        return [
            Hit(chunk_id=cid, score=1.0, payload=by_id[cid].payload)
            for cid in chunk_ids
            if cid in by_id
        ]

    def all_chunks(self) -> list[dict[str, Any]]:
        """Stream every payload -- used to build the in-process BM25 index."""
        out: list[dict[str, Any]] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(r.payload for r in records if r.payload)
            if offset is None:
                break
        return out

    def list_documents(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for payload in self.all_chunks():
            counts[payload["doc_id"]] = counts.get(payload["doc_id"], 0) + 1
        return [
            {"doc_id": doc_id, "num_sections": n} for doc_id, n in sorted(counts.items())
        ]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
