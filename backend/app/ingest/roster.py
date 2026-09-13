"""The document roster: which contract is which.

Chunk payloads carry ``doc_id`` but no human identity, so without this the agent
cannot connect "the Harpoon agreement" to ``doc_001`` -- it sees fifteen opaque ids.
The roster is derived from the CUAD filenames recorded in ``manifest.json`` and
injected into the planner prompt, which costs nothing at query time and needs no
re-indexing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Ordered longest-first so "Development And Option Agreement" beats "Option Agreement".
AGREEMENT_KINDS = [
    "Intellectual Property Agreement",
    "Exclusive Agency And Marketing Agreement",
    "Development And Option Agreement",
    "Collaboration Agreement",
    "Manufacturing Agreement",
    "Distributor Agreement",
    "Development Agreement",
    "Maintenance Agreement",
    "Promotion Agreement",
    "Marketing Agreement",
    "Licensing Agreement",
    "Reseller Agreement",
    "Hosting Agreement",
    "Service Agreement",
    "Supply Agreement",
    "Agency Agreement",
    "License Agreement",
    "Joint Venture Agreement",
    "Franchise Agreement",
    "Transportation Agreement",
    "Endorsement Agreement",
    "Consulting Agreement",
    "Outsourcing Agreement",
    "Strategic Alliance Agreement",
    "Sponsorship Agreement",
    "Agreement",
]

# "HarpoonTherapeuticsInc" -> "Harpoon Therapeutics Inc"
RE_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# SEC filing furniture that carries no meaning for a reader.
RE_FILING_NOISE = re.compile(
    r"\b(?:\d{6,}|\d{8}|EX-?\d+(?:\.\d+)*|10-[KQ]|8-K|S-1|10-12G|DRSA|"
    r"\(on\s+[^)]*\)|\d{1,2}[_ ]\d{1,2}[_ ]\d{4})\b",
    re.I,
)


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    company: str
    kind: str
    num_sections: int = 0

    @property
    def label(self) -> str:
        return f"{self.company} — {self.kind}"


# Many CUAD filenames are ALL-CAPS with no separators ("KINGPHARMACEUTICALSINC"),
# where camel-case splitting has nothing to work with. Greedy longest-match against
# the vocabulary that actually appears in company names recovers most of them.
COMPANY_WORDS = [
    "PHARMACEUTICALS",
    "BIOSCIENCES",
    "LABORATORIES",
    "TECHNOLOGIES",
    "THERAPEUTICS",
    "DIAGNOSTICS",
    "INTERNATIONAL",
    "COMMUNICATIONS",
    "IMMUNOMEDICS",
    "MACROGENICS",
    "PHARMACEUTICAL",
    "BIOPHARMA",
    "HOLDINGS",
    "SCIENCES",
    "SYSTEMS",
    "MEDICAL",
    "NETWORKS",
    "SOFTWARE",
    "SOLUTIONS",
    "ENERGY",
    "PHARMA",
    "HEALTH",
    "LIGHTS",
    "GROUP",
    "LABS",
    "BIO",
    "CORP",
    "INC",
    "LLC",
    "LTD",
    "PLC",
    "CO",
]
_SORTED_WORDS = sorted(COMPANY_WORDS, key=len, reverse=True)
SUFFIXES = {"INC", "LLC", "LTD", "CORP", "PLC", "CO"}


def _segment(blob: str) -> list[str]:
    """Split a run-together ALL-CAPS name into words, longest match first."""
    parts: list[str] = []
    rest = blob
    while rest:
        for word in _SORTED_WORDS:
            if rest.endswith(word) and len(rest) > len(word):
                parts.insert(0, word)
                rest = rest[: -len(word)]
                break
        else:
            parts.insert(0, rest)
            break
    return parts


def _clean_company(raw: str) -> str:
    name = raw.replace("_", " ").replace(",", ", ").strip(" -,")
    name = RE_CAMEL.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Segment per token: "BERKELEYLIGHTS, INC" has a space but still needs splitting.
    tokens: list[str] = []
    for token in name.split():
        bare = token.strip(",")
        trailing = "," if token.endswith(",") else ""
        if bare.isupper() and len(bare) > 6 and bare.upper() not in SUFFIXES:
            segmented = _segment(bare)
            segmented[-1] += trailing
            tokens.extend(segmented)
        else:
            tokens.append(token)

    words = [
        w.upper() if w.strip(",").upper() in SUFFIXES else w.capitalize() for w in tokens
    ]
    return " ".join(words)


def normalize(text: str) -> str:
    """Lowercase, letters only -- so 'King Pharmaceuticals' matches 'KINGPHARMACEUTICALSINC'."""
    return re.sub(r"[^a-z]", "", text.lower())


def parse_source_file(source_file: str, doc_id: str) -> DocumentInfo:
    """Best-effort company + agreement type from a CUAD filename."""
    stem = re.sub(r"\.txt$", "", source_file, flags=re.I)
    normalized = stem.replace("_", " ").replace("-", " ")

    kind = "Agreement"
    for candidate in AGREEMENT_KINDS:
        if re.search(re.escape(candidate), normalized, re.I):
            kind = candidate
            break

    # The company is whatever precedes the first filing-metadata token.
    head = re.split(r"[_-]", stem)[0]
    if len(head) < 3:
        head = stem
    company = _clean_company(RE_FILING_NOISE.sub("", head)) or doc_id
    return DocumentInfo(doc_id=doc_id, company=company, kind=kind)


@lru_cache
def load_roster(manifest_path: Path) -> tuple[DocumentInfo, ...]:
    """Read ``manifest.json``. Returns an empty roster if it is missing."""
    if not manifest_path.exists():
        return ()
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tuple(
        parse_source_file(row.get("source_file", ""), row["doc_id"])
        for row in rows
        if row.get("doc_id")
    )


def format_roster(roster: tuple[DocumentInfo, ...]) -> str:
    """Render for a prompt, one document per line."""
    if not roster:
        return "(document roster unavailable)"
    return "\n".join(f"  {d.doc_id}  {d.label}" for d in roster)


# Shared across a pharma-heavy corpus, so they identify nothing on their own.
GENERIC_WORDS = frozenset(
    {
        "therapeutics",
        "pharmaceuticals",
        "pharmaceutical",
        "technologies",
        "company",
        "sciences",
        "biosciences",
        "holdings",
        "international",
        "group",
        "pharma",
        "health",
        "energy",
        "medical",
    }
)


def resolve_doc(roster: tuple[DocumentInfo, ...], text: str) -> str | None:
    """Find the doc_id a question refers to, if any.

    Matching is on letters-only normalized forms, so "King Pharmaceuticals" reaches
    "KINGPHARMACEUTICALSINC" even though the filename has no word boundaries.
    """
    lowered = text.lower()
    for doc in roster:
        if doc.doc_id.lower() in lowered:
            return doc.doc_id

    normalized_query = normalize(text)
    # Longest company name first: a distinctive full name beats a shared first word.
    for doc in sorted(roster, key=lambda d: -len(normalize(d.company))):
        company = normalize(doc.company)
        stripped = re.sub(r"(inc|llc|ltd|corp|plc|co)$", "", company)
        if len(stripped) >= 6 and stripped in normalized_query:
            return doc.doc_id

    for doc in roster:
        for word in re.findall(r"[A-Za-z]{4,}", doc.company):
            if word.lower() in GENERIC_WORDS:
                continue
            if word.lower() in lowered:
                return doc.doc_id
    return None
