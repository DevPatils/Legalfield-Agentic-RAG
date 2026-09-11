# Agentic RAG for Legal/Compliance Documents

An agentic RAG system for dense, cross-referencing commercial contracts. Unlike naive
"chunk + embed + top-k" RAG, it plans and decomposes queries, retrieves with hybrid
search + reranking, **checks whether the retrieved context is actually sufficient**,
self-corrects by traversing an extracted clause cross-reference graph, and verifies
every citation against its source before returning an answer.

Full design rationale lives in [Architecture.md](Architecture.md).

## Why contracts

Two properties make legal text break naive RAG in ways that are easy to demonstrate:

1. **Dense cross-referencing** — the answer often isn't in the top-k chunk, it's in a
   chunk that chunk points at (`"subject to Section 4.2"`).
2. **Definitional dependency** — `"Confidential Information"` is defined once in
   Section 1 and used everywhere else. A chunk can be semantically similar to the query
   and still unusable without its definition.

Both are resolved by extracting a real cross-reference graph at index time and letting
the agent walk it, rather than asking an LLM to guess where to look next.

## Status

| Stage | State |
|---|---|
| Corpus selection + download (CUAD) | Done |
| Hierarchical clause parser | Done, 81 tests |
| Cross-reference + definition graph | Done, 0 dangling edges over 15 contracts |
| Retrieval: BM25, RRF fusion, rerank | Implemented, not yet run against a live index |
| Embedding + Qdrant indexing | Implemented, needs an API key to run |
| LangGraph agent | Not started |
| FastAPI + SSE | Not started |
| React UI | Not started |
| Eval | Not started |

## Setup

Requires Python 3.11+, Docker (for Qdrant + MongoDB), and Node 20+ (for the UI, later).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[dev,voyage,anthropic,cohere]"

cp .env.example .env            # then fill in your API keys
docker compose up -d            # Qdrant on :6333, MongoDB on :27017
```

### Building the index

```bash
python scripts/fetch_cuad.py --count 15    # download CUAD, pick the densest contracts
python scripts/build_chunks.py             # parse -> data/processed/chunks.jsonl
python scripts/build_index.py              # embed + load into Qdrant  (needs a key)
```

`fetch_cuad.py` scores all ~510 CUAD contracts on cross-reference and definition
density and keeps the top N — a random sample would understate what the agentic layer
buys you. `build_chunks.py --inspect 10` prints sampled graph edges with both endpoints
so you can eyeball that they're real.

### Tests

```bash
pytest backend/tests -q                    # all
pytest backend/tests/test_crossref.py -q   # one file
pytest backend/tests -q -k "roman"         # one pattern
ruff check backend scripts
```

## Layout

```
backend/app/
  ingest/      hierarchical parser, cross-reference extraction, chunking
  retrieval/   BM25, explicit RRF fusion, cross-encoder rerank, pipeline
  embeddings/  provider abstraction (voyage-law-2 | text-embedding-3-large)
  storage/     Qdrant access (vectors, chunk text, graph edges)
  agent/       LangGraph nodes + structured-output schemas
scripts/       fetch_cuad, build_chunks, build_index
eval/          eval set + metrics (Architecture.md §11)
```

## Design decisions worth knowing

**Chunk granularity is the leaf numbered section** (`4.2`), with sub-clauses folded in.
An isolated `"(a) is publicly available"` is useless to retrieve; `4.2 Exceptions` with
its `(a)`–`(d)` inline is a real answer unit.

**Chunk IDs are structural** — `{doc_id}__sec_{section_id}`. Graph traversal fetches
clauses directly by id, so the format is load-bearing rather than cosmetic.

**Qdrant owns all document data**; MongoDB is strictly session/trace history. Chunk
text, vectors, and graph edges never go in Mongo.

**RRF is written out, not imported.** Cosine similarity and BM25 are on incomparable
scales, so any weighted score blend needs re-tuning per corpus; RRF reads only rank.

**Sparse retrieval is not a formality here** — legal queries lean on exact tokens
(`"Section 8.2"`, `"indemnify"`, `"Force Majeure"`) that dense embeddings blur.

**Faithfulness is checked per-citation** — each atomic claim is entailment-checked
against only its own cited chunk, never the whole context, which would degrade it into
a vague overall grounding score.

## Corpus

[CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset),
~510 real commercial contracts. This project indexes 15 of them, selected for
structural density: the chosen set averages ~150 explicit section references and ~60
defined terms per contract.

Raw contracts and generated chunks are gitignored — rerun the scripts above to
reproduce them.
