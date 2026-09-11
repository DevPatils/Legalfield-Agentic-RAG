"""Embed data/processed/chunks.jsonl and load it into Qdrant.

Completes Day 1 of Architecture.md §12. Requires the embedding provider's key in .env
and a running Qdrant (``docker compose up -d``).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --dry-run     # parse + count, no API calls, no writes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.embeddings.provider import get_embedding_provider  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no API calls, no writes")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    settings = get_settings()
    if not settings.chunks_path.exists():
        raise SystemExit(
            f"{settings.chunks_path} not found. Run scripts/build_chunks.py first."
        )

    chunks = load_chunks(settings.chunks_path)
    print(f"Loaded {len(chunks)} chunks from {settings.chunks_path.name}")
    print(f"Provider: {settings.embedding_provider} / {settings.embedding_model}")

    if args.dry_run:
        sample = embedding_text(chunks[0])
        print(f"\nDry run -- nothing embedded or written.\nFirst embedding input:\n{sample[:400]}")
        return 0

    embedder = get_embedding_provider(settings)
    store = QdrantStore(settings)

    print(f"Recreating collection {settings.qdrant_collection!r} (dim={embedder.dim})")
    store.recreate_collection(embedder.dim)

    texts = [embedding_text(c) for c in chunks]
    for start in range(0, len(chunks), args.batch):
        batch = chunks[start : start + args.batch]
        vectors = embedder.embed(texts[start : start + args.batch])
        store.upsert(batch, vectors)
        print(f"\r  indexed {min(start + args.batch, len(chunks))}/{len(chunks)}", end="")

    print(f"\nDone. Collection holds {store.count()} points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
