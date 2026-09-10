# Agentic RAG for Legal/Compliance Documents — Build Spec

## 1. Project Idea (one-liner)

An agentic Retrieval-Augmented Generation system purpose-built for dense, cross-referencing legal/compliance documents (commercial contracts). Unlike naive RAG, it plans queries, decomposes them, retrieves with hybrid search + reranking, checks whether retrieved context is actually sufficient to answer, self-refines (rewrites the query or traverses explicit clause cross-reference links) up to a capped number of iterations, generates an answer with mandatory clause-level citations, and verifies those citations are faithful to the source before returning the answer.

This is a portfolio/resume project. Priorities in order: (1) it must actually work end-to-end on a real corpus, (2) it must be demo-able live with a visible reasoning trace, (3) it must produce real evaluation numbers, (4) code quality/structure should be clean enough to walk through in an interview.

**Timebox: 4 days.**

---

## 2. Why Legal/Compliance Docs Specifically

Contracts and regulations have two properties that make naive "chunk + embed + top-k" RAG fail in ways that are easy to demonstrate and easy to fix with an agentic loop:

1. **Dense cross-referencing** — "subject to Section 4.2", "as defined in Article IX", "notwithstanding Clause 7(b)(i)" — the answer to a question often isn't in the top-k retrieved chunk, it's in a chunk that chunk points to.
2. **Definitional dependency** — terms are defined once (e.g., "Confidential Information" in Section 1) and used everywhere else without re-explanation. A chunk can be semantically similar to the query but unusable without its definition chunk.

Both of these are the exact failure modes the sufficiency-check + refinement loop is designed to catch. This is the "why is this hard, and how did you solve it" story for interviews.

---

## 3. Corpus / Data

- **Source**: [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad) — public dataset of ~500 real commercial contracts with clause-level annotations.
- **Scope for this project**: select **10–20 contracts** from CUAD. Enough density and cross-referencing to be non-trivial; small enough to index and eval quickly within 4 days.
- Store raw contract text as flat files (`data/raw/{doc_id}.txt` or `.pdf`), not in a database.
- No fine-tuning of any model. All intelligence comes from prompting, structured outputs, and the retrieval/agentic pipeline.

---

## 4. Full Architecture

```
┌─────────────┐       SSE stream        ┌──────────────────┐
│   React     │ <──────────────────────>│    FastAPI        │
│  Frontend   │   POST /api/query        │    Backend        │
└─────────────┘                          └─────────┬─────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │   LangGraph Agent    │
                                          │   (orchestration)    │
                                          └──────────┬──────────┘
                                                     │
                        ┌────────────────────────────┼────────────────────────────┐
                        │                             │                             │
                ┌───────▼────────┐         ┌──────────▼─────────┐         ┌────────▼────────┐
                │     Qdrant      │         │      MongoDB        │         │   LLM API        │
                │ (vectors +      │         │ (session/query       │         │ (Claude/GPT-4o-  │
                │  chunk payload  │         │  traces, history)     │         │  mini) for all    │
                │  + metadata)    │         │                       │         │  agent reasoning  │
                └─────────────────┘         └──────────────────────┘         └──────────────────┘
```

**Stack summary**
| Layer | Choice | Why |
|---|---|---|
| Frontend | React | Chat UI + live reasoning trace panel + citation click-through |
| Backend | FastAPI (async, SSE streaming) | Native async, Pydantic structured outputs, auto docs |
| Orchestration | LangGraph | Sufficiency-check refinement loop is a cyclic graph with conditional edges — this is what LangGraph is for |
| Vector DB | Qdrant | Native hybrid (dense + sparse) search, stores chunk text + metadata as payload alongside vectors |
| Document DB | MongoDB | Query traces are variable-depth nested JSON (variable iteration count, variable sub-query fan-out) — natural fit for a document store, awkward to normalize into relational tables |
| Embeddings | `voyage-law-2` (Voyage AI legal-domain model) or `text-embedding-3-large` | Domain-specific embedding model is a genuine differentiator to mention in interviews |
| Reranker | `bge-reranker-large` (local) or Cohere Rerank API | Cross-encoder rerank of hybrid-fused candidates |
| LLM | Claude or GPT-4o-mini via API | Powers planner, sufficiency check, generation, faithfulness check — all via structured/JSON output, no fine-tuning |

---

## 5. Chunking Strategy (do this carefully — it's the foundation everything else depends on)

**Do not** naive-chunk by fixed token count. Parse each contract into a **hierarchical structure**:

```
Document
 └── Article / Section (e.g. "Section 4: Confidentiality")
      └── Sub-clause (e.g. "4.2", "4.2(a)", "4.2(a)(i)")
```

Use rule-based/regex parsing on common legal numbering patterns:
- `Section \d+(\.\d+)*`
- `Article [IVXLC]+`
- `\(\w\)`, `\(\d+\)`, `\(i{1,3}\)` for nested sub-clauses

**Each chunk stores metadata:**
```json
{
  "chunk_id": "doc_001__sec_4.2",
  "doc_id": "doc_001",
  "section_id": "4.2",
  "parent_section_id": "4",
  "section_title": "Confidential Information — Exceptions",
  "text": "...",
  "cross_references": ["4.1", "1.3"]
}
```

**Cross-reference extraction (do this at index time, not query time):**
Regex-scan each chunk's text for patterns like `Section \d+\.\d+`, `Article [IVX]+`, `as defined in`, `subject to`, `pursuant to`. Resolve matched section numbers to `chunk_id`s within the same document. Store as an edge list / adjacency structure — this becomes a real graph you can traverse programmatically during the refinement step, instead of asking the LLM to guess where to look next.

This cross-reference graph is the single best "I solved a real problem" talking point for this project — build it properly.

---

## 6. Retrieval Pipeline

For each sub-query:

1. **Dense search**: embed sub-query, top-N (~30-50) nearest chunks from Qdrant dense vectors.
2. **Sparse search**: BM25 (via `rank_bm25` or Qdrant sparse/SPLADE vectors), top-N.
3. **Fusion**: Reciprocal Rank Fusion (RRF), implemented explicitly (not just relying on a library) so it can be explained:
   ```
   score(doc) = Σ over retrievers [ 1 / (k + rank_i(doc)) ]     # k ≈ 60 typical
   ```
4. **Rerank**: cross-encoder reranks top ~30-50 fused candidates → keep top-k (5-8).
5. Return top-k chunks with scores + metadata to the agent state.

---

## 7. Agentic Pipeline (LangGraph)

### Nodes

1. **Planner**
   - LLM call, structured JSON output.
   - Classifies query type: `definitional | cross_referential | comparative | out_of_scope`.
   - Decides `needs_retrieval: bool` (skip retrieval for greetings/meta-questions).
   - Decomposes compound questions into `sub_queries: [str]`.

2. **Retrieve** (per sub-query, can run in parallel)
   - Runs the retrieval pipeline in §6.
   - Appends results to state under `iterations[i].retrieved_chunks`.

3. **Sufficiency Check**
   - LLM call, structured JSON output:
     ```json
     {"sufficient": false, "missing_info": "definition of 'Confidential Information' referenced but not present", "referenced_sections_needed": ["1.3"]}
     ```
   - **Conditional edge**:
     - If `sufficient == true` → go to Generate.
     - If `sufficient == false` and iteration count < max (2-3) → go to Refine.
     - If iteration count == max → proceed to Generate anyway, flag low confidence.

4. **Refine** (two strategies, both used)
   - **Query rewriting**: LLM rewrites/expands the sub-query with missing terms/synonyms → loop back to Retrieve.
   - **Graph traversal**: if `referenced_sections_needed` matches known cross-reference edges from §5, directly fetch those chunks from Qdrant by `chunk_id` (no re-search needed) → loop back to Sufficiency Check.
   - Prefer graph traversal over query rewriting when the missing info is an explicit section reference (cheaper, more precise, and a better demo of the cross-reference graph).

5. **Generate**
   - LLM call, prompted to require a citation `[doc_id §section_id]` for every factual claim, with a few-shot example enforcing the format.
   - Post-process: verify every citation tag in the output actually corresponds to a `chunk_id` that was in context. Flag/strip any that don't (cheap hallucination check before the faithfulness step).

6. **Faithfulness Check**
   - LLM call: decompose the generated answer into atomic claims.
   - For each claim, run an entailment check **against only its cited chunk** (not the whole context — this is what makes it a real per-citation faithfulness check, not a vague overall grounding score).
   - Output: `{"claims_checked": 5, "flagged": [{"claim": "...", "cited_section": "8.2", "issue": "not supported"}]}`.
   - Flags are surfaced in the UI, not silently swallowed.

### Graph shape
```
Planner → Retrieve → Sufficiency Check ──(sufficient)──> Generate → Faithfulness Check → END
                            │
                     (insufficient, iter < max)
                            │
                            ▼
                         Refine ──> back to Retrieve (query rewrite)
                            or  ──> back to Sufficiency Check (graph traversal fetch)
```

---

## 8. Backend (FastAPI)

### Routes
```
POST /api/query              -> runs full pipeline, returns final answer + full trace (non-streaming)
GET  /api/query/stream        -> SSE endpoint, streams each LangGraph node's output as it completes
GET  /api/documents            -> list indexed documents (id, title, section count)
GET  /api/documents/{id}       -> fetch full doc or a specific section (for citation click-through)
GET  /api/sessions/{id}/history -> past queries for this session, from MongoDB
GET  /api/query/{id}/trace     -> refetch a previously stored full trace without rerunning the pipeline
```

- Use LangGraph's `.stream()` method to emit state after each node — wire this directly to the SSE endpoint. This is close to free given the graph is already built correctly.
- Write to MongoDB incrementally as each node completes (not just once at the end) — gives crash recovery and lets you inspect partial traces.
- Never call the LLM API key from the frontend — all LLM calls happen server-side.

---

## 9. Frontend (React)

- Chat-style main panel: user query in, streamed final answer out, citations rendered as clickable chips.
- **Reasoning trace panel** (side-by-side with the chat, not hidden in a collapsed log) showing, live as they stream in:
  - Sub-queries generated by the planner
  - Retrieved chunks per iteration, with RRF/rerank scores
  - Sufficiency verdicts and why a refinement loop triggered
  - Which refinement strategy was used (query rewrite vs. graph traversal) and what was fetched
  - Faithfulness flags (⚠️ marker on any claim that failed verification)
- Clicking a citation chip opens the actual source clause (fetched via `/api/documents/{id}`) in a side panel.
- Session history sidebar: list of past queries (from `/api/sessions/{id}/history`), clicking one replays the stored trace via `/api/query/{id}/trace` instead of rerunning the LLM (fast, reliable, no burnt API calls — use this for the actual demo/interview).

---

## 10. MongoDB Schema

**Collection: `sessions`**
```json
{
  "_id": "session_uuid",
  "created_at": "ISODate",
  "queries": ["query_id_1", "query_id_2"]
}
```

**Collection: `query_traces`**
```json
{
  "_id": "query_uuid",
  "session_id": "session_uuid",
  "raw_query": "What are the termination conditions under Section 8?",
  "planner": {
    "query_type": "cross_referential",
    "needs_retrieval": true,
    "sub_queries": ["..."]
  },
  "iterations": [
    {
      "iteration": 1,
      "sub_query": "...",
      "retrieved_chunks": [{"chunk_id": "...", "score": 0.82, "section_id": "8.2"}],
      "sufficiency": {"sufficient": false, "missing": "reference to Section 4.2 definitions"}
    },
    {
      "iteration": 2,
      "trigger": "graph_traversal",
      "pulled_sections": ["4.2"],
      "sufficiency": {"sufficient": true}
    }
  ],
  "final_answer": "...",
  "citations": [{"text": "...", "doc_id": "...", "section_id": "8.2", "verified": true}],
  "faithfulness": {"claims_checked": 5, "flagged": []},
  "latency_ms": {"planner": 400, "retrieval": 800, "generation": 1200}
}
```

**Collection: `documents`** (lightweight, for the React sidebar listing)
```json
{"doc_id": "doc_001", "title": "...", "num_sections": 42, "indexed_at": "ISODate"}
```

Mongo does **not** store chunk text, vectors, or the cross-reference graph — that all lives in Qdrant payloads. Mongo is strictly session/trace/history data.

---

## 11. Evaluation Plan

Build a hand-written eval set of **15-25 questions** against the selected contracts, each with a known-correct source section. Deliberately include:
- A few questions answerable from a single top-k chunk (baseline sanity check).
- A few questions that require cross-reference graph traversal (definition lookups, "subject to Section X" cases).
- A few genuinely out-of-scope or ambiguous questions (to test the planner's `needs_retrieval` gate and low-confidence flagging).

**Metrics to report:**
- Retrieval recall@k (before vs. after adding hybrid+rerank, and before vs. after the agentic refinement loop — this delta is the actual headline result)
- Citation accuracy (% of citations pointing to a real, correct chunk)
- Faithfulness flag rate (% of claims flagged as unsupported)
- Iteration count distribution (how often 1 vs. 2 vs. 3 iterations were needed, and for which query types)

---

## 12. 4-Day Build Schedule

**Day 1 — Data + chunking + indexing**
- Pull 10-20 CUAD contracts.
- Write hierarchical parser + cross-reference edge extractor.
- Stand up Qdrant (docker), embed + index all chunks (dense + sparse).
- Manually inspect ~10 chunks to confirm cross-ref edges are correct.
- Deliverable: `chunks.jsonl` + populated Qdrant collection.

**Day 2 — Retrieval pipeline**
- Implement hybrid search + RRF fusion + cross-encoder rerank.
- Write the 15-25 question eval set now (easier while corpus is fresh).
- Measure baseline recall@k (retrieval only, no agentic layer yet).
- Deliverable: given a raw query, correct top-k chunks are retrieved and scored.

**Day 3 — Agentic layer**
- Build the LangGraph graph exactly as in §7: Planner → Retrieve → Sufficiency Check → (conditional: Refine loop | Generate) → Faithfulness Check.
- Wire up both refinement strategies (query rewrite + graph traversal).
- Run 5-10 manual queries end-to-end, including at least one requiring 2 iterations and one requiring cross-reference traversal — pick eval questions that guarantee this happens.
- Deliverable: full pipeline runs end-to-end and produces a correct, cited, faithfulness-checked answer.

**Day 4 — Backend/frontend wiring + eval + polish**
- FastAPI: implement all routes in §8, wire LangGraph `.stream()` to SSE.
- MongoDB: incremental trace writes, session history route.
- React: chat panel + reasoning trace panel + citation click-through + session history sidebar.
- Run full eval set, compute all §11 metrics.
- Pre-record 4-5 strong traces (stored in Mongo) for reliable instant-replay demo — don't rely on live LLM calls during an actual interview demo.
- Write README: architecture diagram, design decisions (why hierarchical chunking, why RRF over pure dense, why graph traversal for cross-refs, why Mongo for traces vs. relational), eval numbers, known limitations.

---

## 13. Resume Line (fill in numbers after Day 4 eval)

> Built an agentic RAG system for dense, cross-referencing legal contracts with hierarchical clause-aware chunking, hybrid retrieval (BM25 + dense + RRF fusion + cross-encoder reranking), and a self-correcting LangGraph pipeline that traverses extracted cross-reference graphs and iteratively refines queries when context is insufficient; achieved **[X]%** citation accuracy and **[Y]%** faithfulness on a 20-question eval set. Full-stack: React, FastAPI (SSE streaming), Qdrant, MongoDB.

---

## 14. Explicit Non-Goals (to keep scope inside 4 days)

- No fine-tuning of any model.
- No multi-user auth system.
- No production deployment/CI/CD — local demo is sufficient.
- No attempt to handle the full 500-contract CUAD corpus — 10-20 is enough to prove the architecture.
- MongoDB is for session/trace history only — never used to store vectors, chunk text, or the cross-reference graph (that's Qdrant's job).