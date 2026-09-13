"""Rule-based hierarchical parser for contract text.

Builds Document -> Article/Section -> Sub-clause trees by regex on the numbering
conventions US commercial contracts actually use (Architecture.md §5). No LLM is
involved: the structure has to be deterministic and cheap because every chunk id,
citation, and cross-reference edge is derived from it.

Two things make this harder than a fixed-token chunker, and both are handled here:

* Headings are frequently inline with their body -- ``4.2 Exceptions. The obligations
  under Section 4.1 shall not apply ...`` is one line, not two.
* ``(i)`` is ambiguous: it is the 9th alpha marker *and* the 1st roman marker. It is
  disambiguated from the surrounding sequence, not from the token alone.
"""

from __future__ import annotations

import re

from .models import Section

# --- Heading patterns, tried in this order (most specific first). ---

RE_ARTICLE = re.compile(r"^\s*ARTICLE\s+([IVXLCDM]+|\d+)\b[.\-–—:]?\s*(.*)$", re.I)
RE_SECTION_KW = re.compile(
    r"^\s*(?:SECTION|CLAUSE)\s+(\d+(?:\.\d+)*)\b[.\-–—:]?\s*(.*)$", re.I
)
# "4.2 Exceptions." / "4.2. Exceptions"
RE_DOTTED = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+(\S.*)$")
# "4. CONFIDENTIALITY." -- requires an uppercase start to avoid eating list items.
RE_TOPNUM = re.compile(r"^\s*(\d+)\.\s+([A-Z]\S*.*)$")
RE_PAREN = re.compile(r"^\s*\(([A-Za-z]{1,5}|\d{1,2})\)\s*(.*)$")
# Unnumbered block headings: RECITALS, WITNESSETH, DEFINITIONS.
RE_CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 ,'&()\-/]{2,79})\.?\s*$")

RE_ROMAN = re.compile(r"^(?=[ivxlcdm]+$)m*(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$")

# Page furniture that shows up in CUAD text dumps. "Source: ACME CORP, 10-K, 3/12/2020"
# is a footer the scraper stamps on every page, not contract text.
RE_PAGE_ARTIFACT = re.compile(
    r"^\s*(?:page\s+\d+\s*(?:of\s*\d+)?"
    r"|[-–—]\s*[ivxlcdm\d]{1,6}\s*[-–—]"  # "- 14 -" and roman "- iii -"
    r"|\[?page\s*break\]?|_{3,}|\f"
    r"|source:\s*.{0,80}?\d{1,2}/\d{1,2}/\d{2,4}\s*)\s*$",
    re.I,
)

# --- Table-of-contents detection -------------------------------------------------
#
# Contracts open with a TOC, and left in place it does real damage rather than merely
# adding noise: the TOC entry for "2.1" is parsed first and claims that section id, so
# the *real* 2.1 is deduplicated to "2.1~2" -- and then the real sub-sections (2.1.1,
# 2.1.2) look up their parent, find the TOC node, and attach to it. The document tree
# ends up rooted in its own table of contents, with broken paths and empty chunks.
#
# Skipping the region before parsing fixes every one of those at once.

RE_TOC_HEADING = re.compile(r"^\s*TABLE\s+OF\s+CONTENTS\s*$", re.I)

# "2.1 Joint Governance Committee. 18" -- a numbered entry ending in a page number.
RE_TOC_ENTRY = re.compile(r"\d+(?:\.\d+)*\.?\s+[A-Z][^.\n]{2,70}\.?\s+\d{1,3}(?!\d)")
# "ARTICLE 1 DEFINITIONS 1" -- article entry with a trailing page number.
RE_TOC_ARTICLE = re.compile(r"^\s*ARTICLE\s+[IVXLC0-9]+\s+[A-Z][^.\n]{2,70}?\s+\d{1,3}\s*$", re.I)

# How far into the document a TOC can plausibly start.
TOC_SEARCH_FRACTION = 0.4
MIN_TOC_ENTRIES = 4
# Ends the region after this many consecutive *non-blank* non-TOC lines. Blank lines
# are not counted: extracted TOCs are full of them, and counting them truncated the
# span mid-table, leaving the rest of the listing to be parsed as clauses.
TOC_GAP_TOLERANCE = 6


def _is_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if RE_TOC_HEADING.match(stripped) or RE_TOC_ARTICLE.match(stripped):
        return True
    # A body line can mention one section number; a TOC line packs several entries,
    # each trailed by a page number.
    return len(RE_TOC_ENTRY.findall(stripped)) >= 1 and bool(
        re.search(r"\s\d{1,3}\s*$|\.\s+\d{1,3}\s", stripped)
    )


def find_toc_span(lines: list[str]) -> tuple[int, int] | None:
    """Locate the table-of-contents region as ``(start, end)`` line indices, inclusive.

    Returns None when the document has no TOC, which many exhibits do not.
    """
    horizon = max(30, int(len(lines) * TOC_SEARCH_FRACTION))
    candidates = [i for i in range(min(horizon, len(lines))) if _is_toc_line(lines[i])]
    if len(candidates) < MIN_TOC_ENTRIES:
        return None

    # Take the first run of candidates, allowing small gaps for the article headings
    # and stray page numbers that sit between entries.
    start = candidates[0]
    end = start
    for index in candidates[1:]:
        between = sum(1 for i in range(end + 1, index) if lines[i].strip())
        if between <= TOC_GAP_TOLERANCE:
            end = index
        else:
            break

    if sum(1 for i in range(start, end + 1) if _is_toc_line(lines[i])) < MIN_TOC_ENTRIES:
        return None

    # Pull in an immediately preceding "TABLE OF CONTENTS" heading.
    for i in range(max(0, start - 3), start):
        if RE_TOC_HEADING.match(lines[i].strip()):
            start = i
            break
    return start, end


def strip_toc(lines: list[str]) -> list[str]:
    """Remove the table-of-contents region, if there is one."""
    span = find_toc_span(lines)
    if span is None:
        return lines
    start, end = span
    return lines[:start] + lines[end + 1 :]

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(s: str) -> int:
    """Convert a roman numeral to an int. Returns 0 if not a valid numeral."""
    s = s.lower()
    if not RE_ROMAN.match(s):
        return 0
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def normalize_text(raw: str) -> list[str]:
    """Strip page furniture and normalize whitespace; return a list of lines."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    # Non-breaking space and the section sign's typical stray spacing.
    raw = raw.replace(" ", " ")
    lines: list[str] = []
    for line in raw.split("\n"):
        if RE_PAGE_ARTIFACT.match(line):
            continue
        lines.append(re.sub(r"[ \t]+", " ", line).rstrip())
    return lines


RE_SENTENCE_END = re.compile(r"[.:;]['\"”’)\]]*$")


def _ends_sentence(line: str) -> bool:
    return bool(RE_SENTENCE_END.search(line.strip()))


def _split_title_body(rest: str) -> tuple[str, str]:
    """Separate an inline heading title from the body that follows it.

    ``"Exceptions. The obligations under ..."`` -> ``("Exceptions", "The obligations ...")``
    ``"The Receiving Party shall ..."``          -> ``("", "The Receiving Party shall ...")``

    A leading fragment only counts as a title if it is short and clause-like: real
    section titles are a handful of words, not sentences.
    """
    rest = rest.strip()
    if not rest:
        return "", ""
    m = re.match(r"^(.{1,80}?)\.\s+(\S.*)$", rest, re.S)
    if m:
        candidate, body = m.group(1).strip(), m.group(2).strip()
        if len(candidate.split()) <= 12 and candidate[:1].isupper():
            return candidate, body
        return "", rest
    # No body after a period: the whole line is a title if it is short enough.
    stripped = rest.rstrip(".")
    if len(stripped.split()) <= 12 and stripped[:1].isupper():
        return stripped, ""
    return "", rest


def _classify_marker(marker: str, open_kinds: list[tuple[str, str]]) -> str:
    """Classify a ``(x)`` marker as alpha / roman / upper / num.

    ``open_kinds`` is the currently open sub-clause stack as (kind, last_marker)
    pairs, which is what resolves the ``(i)`` ambiguity: if an alpha level is open
    and its last marker was ``(h)``, then ``(i)`` continues that alpha run rather
    than opening a roman one.
    """
    if marker.isdigit():
        return "num"
    if marker.isupper():
        return "upper"
    if len(marker) == 1 and marker != "i":
        return "alpha"
    if marker == "i":
        for kind, last in open_kinds:
            if kind == "alpha":
                return "alpha" if last == "h" else "roman"
        # No alpha run open: a bare leading "(i)" is a roman enumeration.
        return "roman"
    return "roman" if roman_to_int(marker) else "alpha"


class _TreeBuilder:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.sections: dict[str, Section] = {}
        self.order: list[str] = []
        self.current_article: str | None = None
        self.current_section: str | None = None
        # Open sub-clause levels under `current_section`: list of (kind, marker).
        self.sub_stack: list[tuple[str, str]] = []
        self.cursor: str | None = None
        self.unnumbered_count = 0

    def add(
        self,
        section_id: str,
        kind: str,
        title: str,
        parent_id: str | None,
        line_no: int,
    ) -> str:
        # Duplicate numbering happens (amendments, exhibits). Suffix rather than clobber.
        if section_id in self.sections:
            n = 2
            while f"{section_id}~{n}" in self.sections:
                n += 1
            section_id = f"{section_id}~{n}"
        level = 1 if parent_id is None else self.sections[parent_id].level + 1
        self.sections[section_id] = Section(
            section_id=section_id,
            kind=kind,  # type: ignore[arg-type]
            level=level,
            title=title,
            parent_id=parent_id,
            line_start=line_no,
        )
        self.order.append(section_id)
        if parent_id is not None:
            self.sections[parent_id].children.append(section_id)
        self.cursor = section_id
        return section_id

    def append_body(self, line: str) -> None:
        if self.cursor is None:
            if line.strip():
                self.add("preamble", "preamble", "Preamble", None, 0)
            else:
                return
        self.sections[self.cursor].body.append(line)

    def parent_for_numbered(self, section_id: str) -> str | None:
        """Parent of "4.2" is "4" when it exists, else the enclosing article."""
        if "." in section_id:
            parent = section_id.rsplit(".", 1)[0]
            if parent in self.sections:
                return parent
        return self.current_article

    def push_subclause(self, marker: str, title: str, line_no: int) -> str | None:
        if self.current_section is None:
            return None
        kind = _classify_marker(marker, self.sub_stack)
        existing = [k for k, _ in self.sub_stack]
        if kind in existing:
            # Sibling: unwind back to that level.
            idx = existing.index(kind)
            self.sub_stack = self.sub_stack[:idx]
        self.sub_stack.append((kind, marker))
        composite = self.current_section + "".join(f"({m})" for _, m in self.sub_stack)
        parent = (
            self.current_section
            if len(self.sub_stack) == 1
            else self.current_section + "".join(f"({m})" for _, m in self.sub_stack[:-1])
        )
        parent = parent if parent in self.sections else self.current_section
        return self.add(composite, "subclause", title, parent, line_no)


def parse_document(raw: str, doc_id: str) -> dict[str, Section]:
    """Parse contract text into a flat ``{section_id: Section}`` tree.

    Nodes carry ``parent_id``/``children``, so the tree is reconstructable, and
    ``body`` holds only the lines belonging to that node (children hold their own).
    """
    builder = _TreeBuilder(doc_id)
    lines = strip_toc(normalize_text(raw))

    # A numbered token only opens a new section if the *previous* line closed a
    # sentence (or was a bare heading). Without this, a wrapped line such as
    #   "... the obligations set forth in\nSection 4.1 shall not apply to ..."
    # is misparsed as a heading for Section 4.1, silently stealing every following
    # sub-clause from its real parent and corrupting the whole cross-reference graph.
    prev_ends_sentence = True
    prev_was_bare_heading = True

    for line_no, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            builder.append_body("")
            prev_ends_sentence = True
            prev_was_bare_heading = False
            continue

        heading_allowed = prev_ends_sentence or prev_was_bare_heading
        matched_heading = False
        has_inline_body = False

        if heading_allowed:
            m = RE_ARTICLE.match(line)
            if m:
                num, rest = m.group(1), _strip_heading_punct(m.group(2))
                article_id = num if num.isdigit() else str(roman_to_int(num) or num).upper()
                title, body = _split_title_body(rest)
                sid = builder.add(f"ART-{article_id}", "article", title, None, line_no)
                builder.current_article = sid
                builder.current_section = None
                builder.sub_stack = []
                if body:
                    builder.append_body(body)
                matched_heading, has_inline_body = True, bool(body)

            if not matched_heading:
                m = RE_SECTION_KW.match(line) or RE_DOTTED.match(line) or RE_TOPNUM.match(line)
                if m:
                    num, rest = m.group(1), _strip_heading_punct(m.group(2))
                    title, body = _split_title_body(rest)
                    parent = builder.parent_for_numbered(num)
                    sid = builder.add(num, "section", title, parent, line_no)
                    builder.current_section = sid
                    builder.sub_stack = []
                    if body:
                        builder.append_body(body)
                    matched_heading, has_inline_body = True, bool(body)

            if not matched_heading and builder.current_section is not None:
                m = RE_PAREN.match(line)
                if m:
                    marker, rest = m.group(1), m.group(2)
                    title, body = _split_title_body(rest)
                    sid = builder.push_subclause(marker, title, line_no)
                    if sid is not None:
                        if body:
                            builder.append_body(body)
                        elif title:
                            # Short sub-clauses are body, not headings.
                            builder.append_body(title)
                        matched_heading = True
                        has_inline_body = bool(body or title)

            if not matched_heading:
                m = RE_CAPS_HEADING.match(line)
                if m and len(m.group(1).split()) <= 8:
                    heading = m.group(1).strip()
                    builder.unnumbered_count += 1
                    builder.add(
                        f"UNNUM-{builder.unnumbered_count}",
                        "section",
                        heading.title(),
                        None,
                        line_no,
                    )
                    builder.current_article = None
                    builder.current_section = builder.cursor
                    builder.sub_stack = []
                    matched_heading = True

        if not matched_heading:
            builder.append_body(line)

        prev_ends_sentence = _ends_sentence(stripped)
        prev_was_bare_heading = matched_heading and not has_inline_body

    return builder.sections


def _strip_heading_punct(rest: str) -> str:
    """Drop the separator between a heading number and its title: 'IV - TERM' -> 'TERM'."""
    return re.sub(r"^[\s\-–—:.]+", "", rest)


def section_path(sections: dict[str, Section], section_id: str) -> str:
    """Breadcrumb from root to this section, e.g. ``ARTICLE IV > 4.2 Exceptions``."""
    parts: list[str] = []
    cur: str | None = section_id
    seen: set[str] = set()
    while cur is not None and cur in sections and cur not in seen:
        seen.add(cur)
        node = sections[cur]
        label = node.section_id.replace("ART-", "ARTICLE ")
        parts.append(f"{label} {node.title}".strip())
        cur = node.parent_id
    return " > ".join(reversed(parts))


def subtree_text(sections: dict[str, Section], section_id: str) -> str:
    """Full text of a node including all descendants, in document order."""
    node = sections[section_id]
    parts = [node.own_text()]
    for child_id in node.children:
        child = sections[child_id]
        marker = child.section_id[len(section_id) :] if child.kind == "subclause" else ""
        label = f"{marker} " if marker else f"{child.section_id} "
        child_text = subtree_text(sections, child_id)
        heading = f"{label}{child.title}".strip()
        parts.append(f"{heading}\n{child_text}".strip() if child.title else f"{label}{child_text}")
    return "\n".join(p for p in parts if p.strip()).strip()
