"""Cross-reference and defined-term extraction.

This runs at *index time*, never at query time (Architecture.md §5). The output is a
real edge list the Refine node walks programmatically -- the agent never asks the LLM
to guess which section to look at next.

Two edge types, matching the two failure modes in Architecture.md §2:

* **Explicit section references** -- "subject to Section 4.2", "as defined in Article IX".
* **Definitional dependency** -- a chunk that uses "Confidential Information" without
  defining it gets an edge to the chunk that does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import roman_to_int

# Keyword that opens a reference, e.g. "Section", "Articles", "§§".
RE_REF_KEYWORD = re.compile(
    r"(?P<kw>§§?|\b(?:Sections?|Clauses?|Articles?|Subsections?|Paragraphs?)\b)\.?\s*",
    re.I,
)

# A single reference target following the keyword: 4, 4.2, 4.2(a)(i), IX.
# The trailing lookahead must not exclude ".", or a sentence-final "Section 5.1."
# would fail to match at all.
RE_TARGET = re.compile(r"(?P<num>\d+(?:\.\d+)*(?:\([A-Za-z0-9]{1,4}\))*|[IVXLCDM]{1,7})(?!\w)")

# Connectors that continue a reference list: "4.1, 4.2 and 4.3", "4.1 through 4.3".
# No "^" anchor: this is applied with .match(text, pos), and "^" would still assert
# start-of-string rather than start-at-pos, silently truncating every list to one item.
RE_CONNECTOR = re.compile(r"[\s,;]*(?:and|or|through|thru|to|-|–|—)?[\s,;]*", re.I)

# Trigger phrases looked for immediately before the keyword; they label the edge.
RELATION_TRIGGERS: list[tuple[str, str]] = [
    (r"as\s+defined\s+in", "defined_in"),
    (r"defined\s+in", "defined_in"),
    (r"subject\s+to", "subject_to"),
    (r"pursuant\s+to", "pursuant_to"),
    (r"in\s+accordance\s+with", "in_accordance_with"),
    (r"notwithstanding", "notwithstanding"),
    (r"set\s+forth\s+in", "set_forth_in"),
    (r"described\s+in", "set_forth_in"),
    (r"provided\s+in", "set_forth_in"),
    (r"referred\s+to\s+in", "set_forth_in"),
    (r"except\s+as", "exception"),
    (r"in\s+compliance\s+with", "in_accordance_with"),
    (r"under", "under"),
    (r"see", "see"),
]
_COMPILED_TRIGGERS = [(re.compile(p + r"\s*$", re.I), rel) for p, rel in RELATION_TRIGGERS]

# '"Confidential Information" means ...' / 'shall mean' / 'has the meaning'
RE_DEFINITION = re.compile(
    r"[“\"'](?P<term>[A-Z][A-Za-z0-9 ,'\-/&]{2,60}?)[”\"']\s*"
    r"(?:\([^)]{0,40}\)\s*)?"
    r"(?:shall\s+mean|means|shall\s+have\s+the\s+meaning|has\s+the\s+meaning|refers\s+to)",
)
# '(the "Agreement")' / '(hereinafter, the "Disclosing Party")'
RE_PAREN_DEFINITION = re.compile(
    r"\((?:\s*(?:the|hereinafter|collectively|each\s+a|individually)\b[\s,]*)+"
    r"[“\"'](?P<term>[A-Z][A-Za-z0-9 ,'\-/&]{2,60}?)[”\"']\s*\)",
    re.I,
)

MAX_FANOUT = 8


@dataclass(frozen=True)
class Reference:
    """One extracted cross-reference, before resolution to a chunk_id."""

    raw: str
    target: str
    relation: str
    kind: str  # "section" | "article"


def _normalize_target(kw: str, num: str) -> tuple[str, str] | None:
    """Normalize a (keyword, number) pair to a canonical section id."""
    is_article = kw.lower().startswith("article")
    if re.fullmatch(r"[IVXLCDM]{1,7}", num, re.I):
        value = roman_to_int(num)
        if not value:
            return None
        # A bare roman numeral is only a reference when it follows "Article".
        if not is_article:
            return None
        return f"ART-{value}", "article"
    if is_article:
        return f"ART-{num}", "article"
    return num, "section"


def _relation_for(text: str, at: int) -> str:
    """Label the edge from the phrase immediately preceding the keyword."""
    window = text[max(0, at - 40) : at]
    for pattern, rel in _COMPILED_TRIGGERS:
        if pattern.search(window):
            return rel
    return "plain"


def extract_references(text: str) -> list[Reference]:
    """Extract every explicit section/article reference in ``text``.

    Handles lists and ranges: "Sections 4.1, 4.2 and 4.3" yields three references.
    """
    refs: list[Reference] = []
    seen: set[tuple[str, str]] = set()

    for kw_match in RE_REF_KEYWORD.finditer(text):
        kw = kw_match.group("kw")
        relation = _relation_for(text, kw_match.start())
        pos = kw_match.end()
        pending_range: str | None = None

        # Walk the list of targets that follow the keyword.
        while True:
            target_match = RE_TARGET.match(text, pos)
            if not target_match:
                break
            num = target_match.group("num")
            normalized = _normalize_target(kw, num)
            if normalized is None:
                break
            target, kind = normalized

            if pending_range is not None:
                for expanded in _expand_range(pending_range, target):
                    if (expanded, relation) not in seen:
                        seen.add((expanded, relation))
                        refs.append(Reference(num, expanded, relation, kind))
                pending_range = None
            elif (target, relation) not in seen:
                seen.add((target, relation))
                refs.append(Reference(target_match.group(0), target, relation, kind))

            pos = target_match.end()
            conn = RE_CONNECTOR.match(text, pos)
            if not conn or conn.end() == pos:
                break
            connector_text = text[pos : conn.end()].lower()
            if re.search(r"through|thru|\bto\b|[-–—]", connector_text):
                pending_range = target
            pos = conn.end()

    return refs


def _expand_range(start: str, end: str) -> list[str]:
    """Expand "4.1 through 4.3" into [4.1, 4.2, 4.3]; bail out if not same-depth siblings."""
    s_parts, e_parts = start.split("."), end.split(".")
    if len(s_parts) != len(e_parts) or s_parts[:-1] != e_parts[:-1]:
        return [start, end]
    if not (s_parts[-1].isdigit() and e_parts[-1].isdigit()):
        return [start, end]
    lo, hi = int(s_parts[-1]), int(e_parts[-1])
    if not 0 <= hi - lo <= MAX_FANOUT:
        return [start, end]
    prefix = ".".join(s_parts[:-1])
    return [f"{prefix}.{i}" if prefix else str(i) for i in range(lo, hi + 1)]


def extract_defined_terms(text: str) -> list[str]:
    """Extract defined terms this text *establishes*."""
    terms: list[str] = []
    for pattern in (RE_DEFINITION, RE_PAREN_DEFINITION):
        for m in pattern.finditer(text):
            term = re.sub(r"\s+", " ", m.group("term")).strip(" ,")
            if term and term not in terms and len(term) > 2:
                terms.append(term)
    return terms


class CrossRefResolver:
    """Resolves section references to chunk_ids within a single document.

    A reference can point at a node that was never chunked on its own (an article
    header, or a sub-clause folded into its parent's chunk). Resolution walks *up*
    to the nearest ancestor that owns a chunk, and for container nodes fans *down*
    to their chunked descendants -- so "see Article IX" resolves to the sections
    inside Article IX rather than to nothing.
    """

    def __init__(
        self,
        chunked_sections: set[str],
        all_sections: dict[str, list[str]] | None = None,
    ) -> None:
        self.chunked = chunked_sections
        # section_id -> child section_ids, for fan-down on container nodes.
        self.children = all_sections or {}

    def resolve(self, target: str) -> list[str]:
        """Return the section_ids of the chunks that a reference points at."""
        if target in self.chunked:
            return [target]

        # Walk up: 4.2(a)(i) -> 4.2(a) -> 4.2 ; 4.2.1 -> 4.2 -> 4
        for ancestor in _ancestors(target):
            if ancestor in self.chunked:
                return [ancestor]

        # Fan down: an article or container section.
        descendants = self._chunked_descendants(target)
        if descendants:
            return descendants[:MAX_FANOUT]
        return []

    def _chunked_descendants(self, section_id: str) -> list[str]:
        out: list[str] = []
        stack = list(self.children.get(section_id, []))
        while stack and len(out) < MAX_FANOUT:
            node = stack.pop(0)
            if node in self.chunked:
                out.append(node)
            else:
                stack.extend(self.children.get(node, []))
        return out


def _ancestors(section_id: str) -> list[str]:
    """Progressively less specific forms of a section id."""
    out: list[str] = []
    cur = section_id
    # Strip trailing "(a)" groups first.
    while True:
        stripped = re.sub(r"\([A-Za-z0-9]{1,4}\)$", "", cur)
        if stripped == cur:
            break
        cur = stripped
        if cur:
            out.append(cur)
    # Then strip dotted components.
    while "." in cur:
        cur = cur.rsplit(".", 1)[0]
        out.append(cur)
    return out
