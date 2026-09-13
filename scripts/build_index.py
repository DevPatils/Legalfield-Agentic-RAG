"""Embed data/processed/chunks.jsonl and load it into Qdrant.

Completes Day 1 of Architecture.md §12. Requires the embedding provider's key in .env
and a running Qdrant (``docker compose up -d``).

Indexing is **incremental**. Each point stores a hash of the exact string that was
embedded; on a re-run, chunks whose hash is unchanged keep their existing vector and
are never sent to the API. Two things follow from that:

* Editing the parser re-embeds only the chunks that actually changed.
* An interrupted run resumes. At the free tier's rate cap a full pass takes over an
  hour, which is long enough that a laptop sleeping or Docker restarting mid-run is a
  realistic event rather than a hypothetical one -- and losing an hour of work to it
  is avoidable.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --dry-run     # report the plan, no API calls
    python scripts/build_index.py --rebuild     # discard existing vectors and redo all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.embeddings.provider import get_embedding_provider  # noqa: E402
from app.embeddings.ratelimit import estimate_tokens  # noqa: E402
from app.ingest.chunker import load_chunks  # noqa: E402
from app.storage.qdrant_store import QdrantStore  # noqa: E402


def embedding_text(chunk) -> str:
    """What actually gets embedded.

    The clause body alone is ambiguous out of context -- half the sections in a
    contract begin "Except as provided above, ...". Prefixing the breadcrumb and title
    gives the embedding the structural context a human reader would have from the page.
    """
    header = " ".join(filter(None, [chunk.path, chunk.section_title]))
    return f"{header}\n\n{chunk.text}" if header else chunk.text


def content_hash(text: str, model: str) -> str:
    """Identifies the embedding input. Includes the model, so switching models
    invalidates every hash rather than silently mixing vector spaces."""
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()[:32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report the plan, no writes")
    ap.add_argument("--rebuild", action="store_true", help="re-embed everything")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    settings = get_settings()
    if not settings.chunks_path.exists():
        raise SystemExit(f"{settings.chunks_path} not found. Run scripts/build_chunks.py first.")

    chunks = load_chunks(settings.chunks_path)
    texts = [embedding_text(c) for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {settings.chunks_path.name}")
    print(f"Provider: {settings.embedding_provider} / {settings.embedding_model}")

    store = QdrantStore(settings)
    embedder = get_embedding_provider(settings, progress=lambda msg: print(f"  {msg}"))

    hashes = [content_hash(t, settings.embedding_model) for t in texts]
    reusable: dict[str, str] = {}
    if not args.rebuild:
        try:
            if store.client.collection_exists(store.collection):
                reusable = store.existing_hashes()
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read existing index ({exc}); treating as empty")

    todo = [i for i, c in enumerate(chunks) if reusable.get(c.chunk_id) != hashes[i]]
    stale = sorted(set(reusable) - {c.chunk_id for c in chunks})

    skipped = len(chunks) - len(todo)
    pending_tokens = sum(estimate_tokens(texts[i]) for i in todo)
    minutes = pending_tokens / max(settings.embed_max_tpm, 1)
    print(
        f"\n  already current : {skipped}"
        f"\n  to embed        : {len(todo)}"
        f"\n  to delete       : {len(stale)}"
        f"\n  ~{pending_tokens:,} tokens at {settings.embed_max_tpm:,} TPM"
        f" -> approx {minutes:.0f} min"
    )

    if args.dry_run:
        print("\nDry run -- nothing embedded or written.")
        return 0
    if not todo and not stale:
        print("\nIndex is already up to date.")
        return 0

    reused = store.ensure_collection(embedder.dim) and not args.rebuild
    if not reused:
        todo = list(range(len(chunks)))  # fresh collection: everything needs embedding
        print(f"  collection created fresh -> embedding all {len(todo)}")

    if stale:
        store.delete_by_chunk_ids(stale)
        print(f"  deleted {len(stale)} chunks no longer in the corpus")

    started = time.time()
    done = 0
    for start in range(0, len(todo), args.batch):
        batch_idx = todo[start : start + args.batch]
        batch = [chunks[i] for i in batch_idx]
        vectors = embedder.embed([texts[i] for i in batch_idx])
        store.upsert(batch, vectors, hashes=[hashes[i] for i in batch_idx])
        done += len(batch)
        rate = done / max(time.time() - started, 1e-6)
        eta = (len(todo) - done) / rate / 60 if rate else 0
        print(f"  embedded {done}/{len(todo)}  ({eta:.0f} min left)", flush=True)

    print(f"Done in {(time.time() - started) / 60:.1f} min. Collection holds {store.count()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
