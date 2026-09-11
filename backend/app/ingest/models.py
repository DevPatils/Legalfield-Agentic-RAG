"""Data model for parsed contract structure and indexable chunks.

The chunk payload here is exactly what gets written into Qdrant (Architecture.md §5).
MongoDB never sees any of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SectionKind = Literal["article", "section", "subclause", "preamble"]


def make_chunk_id(doc_id: str, section_id: str) -> str:
    """Structural chunk id: ``{doc_id}__sec_{section_id}``.

    Load-bearing — graph traversal fetches chunks directly by this id rather than
    re-searching, so the format must stay derivable from (doc_id, section_id).
    """
    return f"{doc_id}__sec_{section_id}"


@dataclass
class Section:
    """A node in the parsed document tree (Document -> Article/Section -> Sub-clause)."""

    section_id: str
    kind: SectionKind
    level: int
    title: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    # Raw body lines belonging to this node only (not its children).
    body: list[str] = field(default_factory=list)
    line_start: int = 0

    def own_text(self) -> str:
        return "\n".join(self.body).strip()


@dataclass
class Chunk:
    """One indexable unit. Serialized to ``data/processed/chunks.jsonl``."""

    chunk_id: str
    doc_id: str
    section_id: str
    parent_section_id: str | None
    section_title: str
    text: str
    # Raw section numbers referenced by this chunk's text, e.g. ["4.1", "1.3"].
    cross_references: list[str] = field(default_factory=list)
    # Those references resolved to chunk_ids in the same document. This is the
    # edge list the Refine node traverses.
    cross_reference_ids: list[str] = field(default_factory=list)
    # Defined terms this chunk *establishes* (e.g. '"Confidential Information" means ...').
    defines_terms: list[str] = field(default_factory=list)
    # Defined terms this chunk *uses* without defining -- the definitional-dependency
    # failure mode from Architecture.md §2.
    defined_terms_used: list[str] = field(default_factory=list)
    # chunk_ids of the definitions for `defined_terms_used`.
    definition_ids: list[str] = field(default_factory=list)
    kind: SectionKind = "section"
    level: int = 1
    # Breadcrumb for display + LLM context, e.g. "ARTICLE IV > Section 4.2".
    path: str = ""
    part: int = 1
    num_parts: int = 1

    def to_payload(self) -> dict:
        """Qdrant point payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "parent_section_id": self.parent_section_id,
            "section_title": self.section_title,
            "text": self.text,
            "cross_references": self.cross_references,
            "cross_reference_ids": self.cross_reference_ids,
            "defines_terms": self.defines_terms,
            "defined_terms_used": self.defined_terms_used,
            "definition_ids": self.definition_ids,
            "kind": self.kind,
            "level": self.level,
            "path": self.path,
            "part": self.part,
            "num_parts": self.num_parts,
        }

    @property
    def citation(self) -> str:
        """The tag the Generate node must emit for this chunk."""
        return f"[{self.doc_id} §{self.section_id}]"
