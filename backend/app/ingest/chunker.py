"""Turn raw contract text into indexable chunks with a resolved cross-reference graph.

Chunk granularity is the *leaf numbered section* (4.2), with its sub-clauses folded
into the body. That matches the chunk_id example in Architecture.md §5 and keeps
chunks self-contained: an isolated "(a) is publicly available" is useless to retrieve,
but "4.2 Exceptions" with its (a)-(d) inline is a real answer unit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .crossref import (
    CrossRefResolver,
    extract_defined_terms,
    extract_references,
)
from .models import Chunk, Section, make_chunk_id
from .parser import parse_document, section_path, subtree_text

# ~1k tokens. Sections longer than this are split on paragraph boundaries.
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 40


# A container's own lead-in has to be worth retrieving on its own before it earns a
# chunk; a stray line of whitespace or a trailing fragment does not.
MIN_CONTAINER_CHARS = 60


def _is_container(sections: dict[str, Section], section_id: str) -> bool:
    node = sections[section_id]
    return any(sections[c].kind in ("section", "article") for c in node.children)


def _is_chunk_node(sections: dict[str, Section], section_id: str) -> bool:
    """A node becomes its own chunk if it has no numbered-section children."""
    node = sections[section_id]
    if node.kind == "subclause":
        return False
    return not _is_container(sections, section_id)


def _container_lead_in(sections: dict[str, Section], section_id: str) -> str:
    """A container section's own text, excluding its children.

    Contracts routinely write a scope statement before the sub-sections it governs --
    "2.2 General Provisions. The following apply to the JGC and survive termination."
    followed by 2.2.1, 2.2.2. That lead-in is substantive, but it belongs to no leaf,
    so without this it reaches no chunk at all and is silently unretrievable.
    """
    node = sections[section_id]
    if node.kind == "subclause" or not _is_container(sections, section_id):
        return ""
    own = node.own_text()
    return own if len(own) >= MIN_CONTAINER_CHARS else ""


def _split_oversized(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on blank lines, packing paragraphs up to ``limit``."""
    if len(text) <= limit:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > limit:
            parts.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        parts.append(buf.strip())
    return parts or [text]


def build_chunks(raw: str, doc_id: str) -> list[Chunk]:
    """Parse a contract and emit fully-populated chunks, graph edges included."""
    sections = parse_document(raw, doc_id)
    children_map = {sid: list(node.children) for sid, node in sections.items()}

    # --- Pass 1: decide chunk boundaries and materialize text. ---
    chunks: list[Chunk] = []
    for section_id in sections:
        node = sections[section_id]

        if _is_chunk_node(sections, section_id):
            text = subtree_text(sections, section_id)
        else:
            # Container: emit only its own lead-in, so nothing is duplicated between
            # it and the child chunks below it.
            text = _container_lead_in(sections, section_id)
            if not text:
                continue

        # A titled node with no text is parser residue, not a retrievable clause.
        if not text.strip():
            continue
        if len(text) < MIN_CHUNK_CHARS and not node.title:
            continue
        parts = _split_oversized(text)
        for i, part_text in enumerate(parts, start=1):
            sid = section_id if len(parts) == 1 else f"{section_id}__p{i}"
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, sid),
                    doc_id=doc_id,
                    section_id=sid,
                    parent_section_id=node.parent_id,
                    section_title=node.title,
                    text=part_text,
                    kind=node.kind,
                    level=node.level,
                    path=section_path(sections, section_id),
                    part=i,
                    num_parts=len(parts),
                )
            )

    # Parts share a logical section: a reference to "4.2" should hit part 1.
    chunked_sections = {c.section_id for c in chunks}
    for c in chunks:
        if c.num_parts > 1 and c.part == 1:
            chunked_sections.add(c.section_id.split("__p")[0])
    base_to_chunk = {
        c.section_id.split("__p")[0]: c.section_id for c in chunks if c.part == 1
    }
    resolver = CrossRefResolver(chunked_sections, children_map)

    # --- Pass 2: definition index for the whole document. ---
    definition_index: dict[str, str] = {}
    for c in chunks:
        c.defines_terms = extract_defined_terms(c.text)
        for term in c.defines_terms:
            definition_index.setdefault(term, c.chunk_id)

    term_pattern = _compile_term_pattern(definition_index)

    # --- Pass 3: resolve edges. ---
    for c in chunks:
        base_section = c.section_id.split("__p")[0]
        raw_refs = extract_references(c.text)
        targets: list[str] = []
        target_ids: list[str] = []
        for ref in raw_refs:
            for resolved in resolver.resolve(ref.target):
                resolved = base_to_chunk.get(resolved, resolved)
                if resolved in (c.section_id, base_section):
                    continue  # self-reference
                chunk_id = make_chunk_id(doc_id, resolved)
                if chunk_id not in target_ids:
                    targets.append(ref.target)
                    target_ids.append(chunk_id)
        c.cross_references = _dedupe(targets)
        c.cross_reference_ids = target_ids

        if term_pattern is not None:
            # Terms wrap across lines in the source, so matches are whitespace-
            # normalized before being compared to the canonical term.
            used = {re.sub(r"\s+", " ", m.group(0)) for m in term_pattern.finditer(c.text)}
            used -= set(c.defines_terms)
            c.defined_terms_used = sorted(used)
            c.definition_ids = [
                definition_index[t]
                for t in c.defined_terms_used
                if definition_index.get(t) and definition_index[t] != c.chunk_id
            ]

    return chunks


def _compile_term_pattern(definition_index: dict[str, str]) -> re.Pattern[str] | None:
    """One alternation over all defined terms, longest-first so 'X Y' beats 'X'."""
    terms = [t for t in definition_index if len(t) > 3]
    if not terms:
        return None
    terms.sort(key=len, reverse=True)
    # Spaces become \s+ so a term wrapped across a line break still matches.
    parts = [re.sub(r"(?:\\?\s)+", r"\\s+", re.escape(t)) for t in terms]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_corpus(raw_dir: Path, out_path: Path) -> list[Chunk]:
    """Parse every ``.txt`` in ``raw_dir`` and write ``chunks.jsonl``."""
    all_chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("*.txt")):
        doc_id = path.stem
        chunks = build_chunks(path.read_text(encoding="utf-8", errors="ignore"), doc_id)
        all_chunks.extend(chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")
    return all_chunks


def load_chunks(path: Path) -> list[Chunk]:
    """Read back ``chunks.jsonl``."""
    out: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Chunk(**json.loads(line)))
    return out
