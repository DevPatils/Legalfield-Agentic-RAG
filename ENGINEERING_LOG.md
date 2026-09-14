# Engineering Log

A record of what broke while building this system, why, and what the fix was.

Most of these are not typos. They are cases where the code did exactly what it was
told and the instruction was wrong — which is the failure mode that actually costs
time, because nothing crashes and the output looks plausible. A RAG pipeline is
especially good at hiding these: a bad chunk still embeds, still retrieves, and still
produces a fluent answer.

Three things caught more bugs than reading the code did, and they are the parts worth
talking about:

- **The faithfulness check found a bug in the indexer.** It flagged an answer claim as
  supported only by "a table of contents listing section titles." That was the
  system's own quality gate reporting a defect two stages upstream of itself.
- **Drawing the data structure by hand.** Sketching the chunk tree on paper surfaced
  a whole class of text that was being silently discarded (#4). Nothing in the code
  or the tests pointed at it.
- **Counting things.** Several bugs showed up as a number being slightly wrong —
  1 reference where there should have been 3, 51 lines where there should have been
  85 — long before anyone noticed a bad answer.

Current state: 15 contracts, 2,878 chunks, 2,607 resolved cross-reference edges,
8,943 definition edges, 168 passing tests.

---

## 1. The parser built the tree out of the table of contents

**Symptom.** Asked what Section 9.1 said, the system answered with a list of section
titles. The trace showed a real retrieval, a real citation, and the faithfulness check
flagging the claim: the cited clause was "only a table of contents listing section
titles."

**Cause.** The parser walks the document top to bottom, and a table of contents comes
first. A TOC line like `2.1 Formation of the JGC .......... 14` matches the same
numbering regex as a real heading, so `2.1` was registered from the TOC and claimed
that ID. When the *real* Section 2.1 appeared 40 pages later, its sub-sections looked
up parent `2.1`, found the TOC node already sitting there, and attached to it.

The whole document tree was rooted in its own table of contents. Every real clause
became a sibling of the index rather than a child of its section.

**Fix.** `strip_toc()` in [parser.py](backend/app/ingest/parser.py) detects and removes
the TOC span before parsing begins. Detection is deliberately conservative — a TOC line
must end in a page number, at least `MIN_TOC_ENTRIES` (4) of them must cluster
together, and the search only looks in the first `TOC_SEARCH_FRACTION` (40%) of the
document. A contract that genuinely has no TOC loses nothing.

**Generalizes to.** Structure-aware parsers need to know which parts of the document
*describe* structure rather than *being* structure. Headers, footers, indexes,
signature blocks and exhibit lists all imitate the thing the parser is looking for.

---

## 2. The TOC fix worked, then truncated halfway

**Symptom.** After #1, most documents were clean, but `doc_001` still produced TOC
chunks. The stripper was firing — just stopping at line 51 of an 85-line TOC.

**Cause.** `TOC_GAP_TOLERANCE` allows a few non-matching lines between TOC entries, so
a page break or a wrapped title doesn't end the span. The gap was measured in *raw
lines*, and this TOC had blank lines between entries. Six blank lines exhausted a
tolerance meant for six lines of unexpected content.

**Fix.** Count only non-blank lines toward the gap:

```python
between = sum(1 for i in range(end + 1, index) if lines[i].strip())
```

**Generalizes to.** A tolerance is a statement about how much *signal* may be missing.
Measuring it in units that include noise makes it mean something else entirely.

---

## 3. Wrapped sentences became phantom sections

**Symptom.** Sub-clauses were attaching to sections that did not exist in the contract.

**Cause.** Contract text wraps. A sentence ending `...as set forth in` followed by
`Section 4.1 shall not apply` puts `Section 4.1` at the start of a line, which is
exactly what a heading looks like. The parser created a section for it, and the real
`4.1(a)`-`(d)` that followed attached to the phantom instead of the real clause.

**Fix.** A line may only open a heading if the previous line ended a sentence, or was
itself a bare heading:

```python
heading_allowed = prev_ends_sentence or prev_was_bare_heading
```

**Generalizes to.** When a pattern is ambiguous in isolation, the disambiguating
signal is usually in the neighbouring context, not in a more elaborate pattern. Two
lines of state beat a longer regex.

---

## 4. Section intro text was silently unretrievable

**Symptom.** None. Nothing failed. This was found by drawing the chunk tree on paper
and noticing a box with no arrow out of it.

**Cause.** Chunking happens at the *leaf* section, so `4.2` with its `(a)`–`(d)` folded
in becomes one self-contained chunk. But a section with numbered *sub-sections* is a
container, and containers were skipped entirely to avoid duplicating their children's
text. Contracts routinely put a scope statement in the container:

> **2.2 General Provisions.** The following apply to the JGC and survive termination.
> **2.2.1** ...
> **2.2.2** ...

That lead-in belongs to no leaf. It reached no chunk. It was not in the index at all.

**Fix.** `_container_lead_in()` in [chunker.py](backend/app/ingest/chunker.py) emits a
container's *own* text — excluding children — as its own chunk when it exceeds
`MIN_CONTAINER_CHARS` (60). No duplication, because the child chunks never contained
it. 56 chunks across the corpus, all previously invisible.

**Generalizes to.** "Skip the container to avoid duplication" assumed containers hold
nothing but their children. Worth checking before relying on it — and worth noticing
that the failure is invisible: unretrievable text produces no error, just a slightly
worse answer to a question nobody has asked yet.

---

## 5. `^` in a regex used with `.match(text, pos)`

**Symptom.** `subject to Sections 4.1, 4.2 and 4.3` resolved to exactly one edge.
Every multi-target reference in the corpus was truncated to its first target.

**Cause.** Reference lists are parsed by scanning: match a target, match a connector,
match the next target, advancing `pos` each time. The connector pattern was anchored:

```python
RE_CONNECTOR = re.compile(r"^[\s,;]*(?:and|or|through|to|-)?[\s,;]*")
```

`re.match(text, pos)` starts matching *at* `pos`, but `^` still means "start of
string." At any `pos > 0` it can never match. The scan matched the first target, failed
the connector, and stopped.

**Fix.** Drop the anchor — `.match(text, pos)` is already anchored at `pos`.
[crossref.py:35](backend/app/ingest/crossref.py#L35) carries a comment saying so, since
the anchor looks correct and adding it back is the obvious "cleanup."

**Generalizes to.** `^` and `.match(s, pos)` mean different anchors, and combining them
fails silently rather than erroring.

---

## 6. A lookahead that excluded sentence-final references

**Symptom.** References at the end of a sentence — `...as provided in Section 5.1.` —
produced no edge.

**Cause.** The target pattern ended with `(?![\w.])` to prevent `4.2` matching inside
`4.2.1`. But the full stop ending the *sentence* is also a `.`, so the lookahead
rejected the match entirely.

**Fix.** `(?!\w)` — [crossref.py:30](backend/app/ingest/crossref.py#L30). The greedy
`(?:\.\d+)*` already consumes `4.2.1` correctly; excluding `.` was solving a problem
the quantifier had solved.

**Generalizes to.** Negative lookaheads are easy to over-specify. Each character added
to the exclusion class is a silent rejection of some real input.

---

## 7. Defined terms broken across lines

**Symptom.** `Confidential Information` matched in some chunks and not others, with no
obvious pattern.

**Cause.** The non-matching ones had the term wrapped across a line break, so the
literal space in the compiled term did not match a newline.

**Fix.** Spaces in the term pattern compile to `\s+`, and matches are whitespace-
normalised before comparison to the canonical term:

```python
used = {re.sub(r"\s+", " ", m.group(0)) for m in term_pattern.finditer(c.text)}
```

**Generalizes to.** In text extracted from PDFs, a space and a newline are the same
character semantically. Any multi-word literal needs to treat them that way.

---

## 8. Filtering BM25 results on `score > 0`

**Symptom.** Sparse retrieval returned fewer results than expected for common legal
terms, and hybrid fusion silently lost its sparse half on exactly the queries where
keyword matching should have helped most.

**Cause.** BM25 weights by inverse document frequency. A term appearing in most
documents gets a *negative* IDF — that's the formula working correctly, saying "this
term is not discriminating." Filtering on `score > 0` discarded real keyword matches
for the most common contract vocabulary.

**Fix.** Filter on token overlap rather than score
([bm25.py:78](backend/app/retrieval/bm25.py#L78)):

```python
self.token_sets = [set(tokens) for tokens in tokenized]
```

A document is a candidate if it actually contains a query token, whatever BM25 thinks
of that token's discriminating power.

**Generalizes to.** `score > 0` is an assumption about a scoring function's range, not
a general test for relevance. Check the range before filtering on it.

---

## 9. A unit test that asserted the opposite of the truth

**Symptom.** The RRF tests passed. The tests were wrong.

**Cause.** Reciprocal Rank Fusion is `score(d) = Σᵢ 1/(k + rankᵢ(d))`. The tests were
written on the premise that a *small* `k` rewards documents both retrievers agree on.
It is the reverse: small `k` makes the curve steep, so a single top-1 placement
dominates and disagreement wins; large `k` flattens it, so consistent mid-rank
agreement accumulates.

**Fix.** Both tests redesigned around the actual behaviour, with the reasoning written
into the test docstrings — the value of RRF here is being able to explain it, so a test
that encodes a wrong explanation is worse than no test.

**Generalizes to.** A passing test proves the code matches the test's premise. If the
premise is wrong, the test locks in the bug and adds confidence to it.

---

## 10. A prompt that taught the model the contrapositive

**Symptom.** After a fix intended to stop the planner refusing questions about indexed
companies, the planner began refusing questions that named *no* company — including
plainly answerable ones like "how may this agreement be amended?"

**Cause.** The prompt said:

> a question naming any company is IN SCOPE... never out_of_scope

The model learned the converse: a question naming *no* company may be out of scope.
The rule was stated as a property of the *question's surface form* rather than of its
*subject*.

**Fix.** Rewrote `needs_retrieval` guidance around topic, not answerability — "is this
about contract content at all" — with the reasoning stated explicitly:

> You are deciding whether to look, not whether an answer exists — and you cannot know
> whether an answer exists without looking.

**Generalizes to.** Models generalise from prompts the way they generalise from
anything: they pick up the implied rule, not only the stated one. Positive rules stated
narrowly ("X is in scope") invite the negative inference ("not-X may not be"). State
the *principle*, and give the model the reason, so the inference it draws is the one
intended.

Also worth recording: this bug was diagnosed before it was fixed, at the user's
insistence — *"don't change anything but analyze the problem."* The first instinct was
to revert the previous change, which would have restored the original bug.

---

## 11. Token estimates that undershot by 10%

**Symptom.** The embedding run slowed to a crawl and then stalled, with escalating
backoff waits.

**Cause.** Batches were sized using a `len(text) / 4` token heuristic. Legal text runs
closer to 3.64 characters per token — dense with long words and no contractions — so
every batch was ~10% larger than the limiter believed. The server rejected the request,
which cost a full backoff cycle, which made the next attempt slower, and so on.

**Fix.** Voyage's tokenizer runs locally and free — use it:

```python
return [self.client.count_tokens([t], model=self.model) for t in texts]
```

with the heuristic kept only as a fallback.

**Generalizes to.** An estimate used for *reporting* can be rough. An estimate used to
stay under a hard limit needs to be exact or conservative; being optimistic converts a
small error into a compounding one.

---

## 12. An O(n²) batcher that was also wrong

**Symptom.** Found in review, not in production.

**Cause.** The first version of `batch_by_tokens` took a list of texts and a cost
function, and the call site passed `lambda t: costs[texts.index(t)]`. That is `O(n²)`,
and on a corpus with duplicate short chunks it returns the *first* match's cost — the
wrong one.

**Fix.** Pass precomputed costs positionally. The function now takes `costs: list[int]`
and yields index batches, so identity is by position and never by value.

**Generalizes to.** Looking an item up by value inside a loop over the same collection
is almost always both a complexity bug and a correctness bug waiting on a duplicate.

---

## 13. Three interrupted index runs, and what made the fourth survivable

**Symptom.** Three separate failures partway through a ~75-minute embedding run: Docker
Desktop quitting (~600 points in), a network `RemoteDisconnected` (1,088 in), and
sustained rate limiting (1,408 in).

**Cause.** Not a bug — a run long enough that ordinary interruptions become likely.
The bug was that the first failure lost all of its work.

**Fix.** Content-hash-addressed incremental indexing in
[build_index.py](scripts/build_index.py). Every point stores
`sha256(f"{model}\x00{text}")[:32]`. A re-run reads back the existing hashes and embeds
only what changed:

```python
todo = [i for i, c in enumerate(chunks) if reusable.get(c.chunk_id) != hashes[i]]
```

Two properties fall out of this. An interrupted run resumes where it stopped. And
editing the parser re-embeds only the chunks the edit actually changed — which is what
made fixing #1–#4 affordable, since each parser fix would otherwise have meant a full
re-embed of the corpus.

Including the model name in the hash means switching embedding models invalidates
everything, rather than silently mixing two vector spaces in one collection.

Point IDs are `uuid5(NAMESPACE, chunk_id)`, so a re-upsert overwrites the existing
point rather than creating a duplicate.

**Generalizes to.** Any job long enough to be interrupted should be written so that
being interrupted is cheap. That is usually a small amount of bookkeeping decided
early, and very expensive to retrofit after the first loss.

---

## 14. The rate-limit fix that didn't fix anything

**Symptom.** Runs kept stalling on `RateLimitError` at the free tier.

**Attempted fix.** Request size was reduced from 8,000 to 3,000 tokens, reasoning that
a request sized at roughly `TPM / RPM` means the permitted requests-per-minute add up
to exactly the token budget, and no single request can exhaust the minute.

**Result.** The reasoning is sound and the sizing is still in the code
([provider.py:158](backend/app/embeddings/provider.py#L158)). It did not resolve the
stall. The limit being hit was an account-level one that request shaping cannot work
around.

**Recorded because it is a real outcome.** The local limiter models a server-side
policy it cannot observe, and a per-process sliding window starts each run blind to
what the previous process spent. Retries with capped exponential backoff
(`min(120, 5 * 2**attempt)`, 14 attempts) make this survivable, and #13 makes it
resumable, but neither makes it fast.

---

## 15. Claim splitting vs. bulleted answers

**Symptom.** Caught before it shipped, while changing answers from prose to labelled
bullets.

**Cause.** The faithfulness check splits an answer into claims and checks each against
*only* its own cited clause — that per-citation isolation is the whole point of the
check. The splitter was sentence-based:

```python
re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", text)
```

Bullets break both halves of that: they often have no terminal full stop, and the next
one starts `- **Quorum**`, not a capital letter. Several bullets would have fused into
one claim and been checked against whichever clause the first one cited — silently
turning a per-citation check back into the vague overall grounding score it exists to
avoid.

**Fix.** Split on line breaks first, then sentences within a line
([nodes.py](backend/app/agent/nodes.py)). Three tests pin the behaviour: bullets split
individually, prose still splits on sentences, and `Section 4.2.` still does not count
as a sentence boundary.

**Generalizes to.** Changing an output format changes every consumer that parses it.
The renderer was the obvious consumer; the verifier was the one that mattered.

---

---

## 16. Two lists that stopped being parallel, and 41% of the graph went dark

**Symptom.** Traversal fired correctly on "what confidentiality obligations apply to a
consultant attending a JGC meeting" — and fetched **doc_012's** Article 9, from an
unrelated contract. It then did it twice more, one clause per iteration, hit the
iteration cap and answered with low confidence. The sufficiency check was right every
time: Article 9's actual terms were never in context.

**Cause.** Two defects compounding, both in one function.

At index time the two payload lists are built in lockstep and then only one of them is
deduplicated ([chunker.py](backend/app/ingest/chunker.py)):

```python
c.cross_references    = _dedupe(targets)   # ['ART-9']    -- 1
c.cross_reference_ids = target_ids         # 9.1 ... 9.8  -- 8
```

At query time they were zipped back together as though still parallel:

```python
for section_id, chunk_id in zip(refs, ids, strict=False):
```

`strict=False` takes one pair and discards seven, with no error. Measured across the
corpus: **196 chunks misaligned, 1,065 of 2,607 edges — 41% — unreachable by
traversal.** Every reference to a whole article resolved to its first section only.

The second defect was the dictionary key:

```python
out[section_id] = chunk_id     # "ART-9"
```

All fifteen contracts have an Article 9. Each document in context overwrote the
previous one's entry, so traversal fetched whichever agreement happened to sit last in
the retrieved list — doc_012, in this case, because its §3.1 was ranked sixth.

**Fix.** Treat the resolved ids as authoritative and derive the label from them, rather
than trusting two lists to line up:

```python
for chunk_id in chunk.payload.get("cross_reference_ids") or []:
    ...
    out[label_for(chunk_id)] = chunk_id     # "doc_001 §9.1"
```

Structural chunk ids make the label a string split rather than a lookup. Three things
fall out: all eight children of an article become candidates instead of one, the label
is document-qualified so traversal can tell two Article 9s apart, and the sufficiency
model can name several in one go — turning eight iterations into one. The same fix
applies to the definitions path, where the lists are parallel across the corpus today
but nothing enforces it; a length mismatch there is now skipped rather than paired off,
because a term attached to the wrong clause is worse than a term with no clause.

**No re-index was needed** — every id was already in the Qdrant payload, correct and
complete. The indexer had been right the whole time; only the reader was wrong.

**Generalizes to.** Two collections are only parallel if something keeps them that way.
`_dedupe` on one and not the other silently broke an invariant three files away, and
`zip(..., strict=False)` is precisely the construct that turns that breakage into
missing data instead of an exception. Where one list is derivable from the other,
derive it — don't store both and hope.

This is the third bug in this log (with #5 and #12) whose entire shape is "a list was
silently truncated to its first element." That is worth knowing about oneself.

---

## 17. The generator and the verifier disagreed about what evidence is

**Symptom.** With the traversal fix in place, the consultant question finally reached
doc_001's Article 9 — and the faithfulness check flagged the answer's best claim. The
claim said non-representative attendees owe duties "equivalent to those set forth in
Article 9, meaning… not disclosing it to third parties, and not using it outside the
Agreement's purposes," citing both §2.2.2 and §9.2. The checker objected that §2.2.2
does not contain those specifics.

It was right about §2.2.2. And §9.2 says almost verbatim: "keep confidential and not
publish or otherwise disclose to a Third Party and not use, directly or indirectly, for
any purpose." The claim was well supported. The checker had never been shown the clause
that supported it.

**Cause.** The claim was paired with its first citation only:

```python
found = RE_CITATION.findall(sentence)
doc_id, section_id = found[0]
```

Meanwhile GENERATE_SYSTEM instructs: "One citation per claim. Two clauses supporting
one claim get two citations." So the generator was told to cite both, did, and was
penalised for it. Two components with an unstated disagreement about what constitutes a
claim's evidence — the kind of bug that lives in the gap between two files, where
neither one looks wrong on its own.

**Fix.** `build_claims()` in [nodes.py](backend/app/agent/nodes.py) attaches every
clause a claim cites, capped at three. The isolation property the check exists for is
untouched: a claim still sees only what it cited, never the surrounding context. The
cap is what keeps "every clause it cited" from drifting back toward "the whole
context" for a claim that cites eight.

Extracted as a pure function so the invariant is testable without an LLM. Six tests now
pin it: both clauses attached, uncited context never shown, repeated citations
de-duplicated, hallucinated tags dropped, evidence capped.

**Generalizes to.** When one component produces a format and another consumes it, the
contract between them is real whether or not anyone wrote it down. Here it was written
down — in the generate prompt — and the consumer had simply never been updated to
match.

And this is the **fourth** bug in this log whose entire shape is *a list silently
truncated to its first element*: the `^` anchor (#5), `texts.index()` (#12), the
mismatched `zip` (#16), and now `found[0]`. Four instances is not coincidence, it is a
habit. The common thread is that all four were written while thinking about the
singular case — one reference, one cost, one edge, one citation — in code whose type is
plural. None of them raised; each returned a plausible answer built on a fraction of
the data. Worth a standing rule: when indexing `[0]` out of a collection, say out loud
why the rest cannot matter.

**Footnote.** Applying this fix reproduced #10 — a heredoc mangled the string escapes
and wrote literal newlines into a Python string literal, exactly as recorded. The log
entry saying "use the Edit tool for this" existed and was not read. A lesson written
down is not the same as a lesson learned.

---

## 18. A type annotation that lied, and a guard that only caught half the failures

**Symptom.** "(no answer returned)", 0 citations, 3 iterations. The trace ended at a
REFINE step with no generate and no faithfulness. The run had died silently.

What had gone *right* first is worth recording: this query chained both traversal
types for the first time. Section 4.1 (Default) points at Section 7.1 (Aggregate Limit
of Liability) by cross-reference; Section 7.1 leans on "Maximum Liability", a term
defined thirty pages earlier. The agent followed the reference edge, then followed the
definitional edge, both by id, no second search. Then it fell over.

**Cause.** `AnthropicLLM.parse()` is annotated to return a validated model:

```python
def parse(..., schema: type[T], ...) -> T:
    ...
    return response.parsed_output
```

`messages.parse` does not raise when the model fails to produce something matching the
schema — it hands back `parsed_output=None`. Usually that means the response hit
`max_tokens` part-way through the JSON. Every node wraps its call:

```python
try:
    out = deps.llm.parse(...)
except Exception:
    ...degrade to a defensible default
```

That guard catches raises. A `None` walks straight past it and detonates on the next
line, `out.referenced_sections_needed` — outside the try — which propagates up and
kills the entire graph run. `last_node: error`, no answer, nothing rendered.

The trigger was #16. Doc-qualifying the candidate labels made them longer, and once
traversal reached a *definitions* chunk — which is exactly where traversal goes when
a term is missing — that chunk contributed a long tail of defined terms. At
`max_tokens=1024` the sufficiency model ran out mid-structure.

**Fix.** Three parts, in order of importance:

1. `_require_parsed()` raises instead of returning None, naming the node and the
   `stop_reason`. This is the whole fix: it turns a run-killing crash into an
   exception the existing per-node guards already know how to absorb.
2. Terms are capped at 20 against 40 for sections, because definitions chunks flood
   the list and their labels are longer.
3. The sufficiency call gets 2048 tokens rather than 1024.

Verified by handing the sufficiency node a client that returns None the way the SDK
did: it now reports `sufficiency check failed: model returned no parseable output
(stop_reason=max_tokens)` and returns `sufficient=True`, so the graph proceeds to
generate and answers from what it has.

**Generalizes to.** Two things, and the second is the one that cost the time.

A return type of `T` that can be `None` is a lie the type checker cannot catch at a
boundary like this, and every caller downstream is written against the lie.

And a `try/except` around a call is not a guard on the *result* of that call. This one
looked complete — it was the same shape as the guards on every other node, it had a
sensible fallback, and it had been reviewed. It simply protected the wrong line. When
a failure mode is "returns a bad value" rather than "raises", exception handling is not
protection; validating the value is.

**Footnote.** This bug was reachable only because of the previous fix. That is the
ordinary cost of changing a system's behaviour, not an argument against the change —
but it is why #16 and #18 belong next to each other in this log.

## Still open

- **Reranking is disabled.** `RERANK_PROVIDER=none` makes the cross-encoder stage a
  passthrough, so the before/after-rerank recall delta — the headline eval number —
  cannot be measured yet.
- **No eval set.** 15–25 questions spanning single-chunk, cross-reference-requiring,
  and out-of-scope cases. CUAD ships 13,000+ expert annotations that could seed it.
- **29 chunks carry `~N` suffixes** — duplicate section numbers the parser
  disambiguated rather than resolved. Down from 257 after #1, and the remainder are
  believed legitimate (genuinely repeated numbering across exhibits), but unverified.
- **Qdrant client 1.19 against server 1.12.4** emits a version warning. Harmless so
  far; not investigated.

---

## What generalizes

**Fluent output is not evidence of a working pipeline.** Every parser bug above still
produced grammatical, confident, cited answers. The TOC bug produced an answer citing a
real section ID that contained a page-number index. Without a faithfulness check
reading the *actual cited text*, none of this surfaces as a failure — it surfaces as a
system that is quietly a bit worse than it looks.

**Put the quality gates upstream of where you think the bugs are.** The faithfulness
check was built to catch model hallucination. It caught an indexing bug instead,
because it was the only component that compared a claim against the ground truth it
named.

**The cheapest debugging tool was counting.** Reference counts, TOC line counts, chunk
counts per document, chunks with empty text. Every one of these is a one-line script,
and each surfaced something that reading the code had not.

**Draw the data structure.** #4 was invisible in code, in tests, and in output. It was
obvious the moment the tree was sketched by hand.
