"""Table-of-contents handling and container lead-in text.

Both defects were invisible to the original suite because the synthetic fixture
contract has neither a TOC nor a container section with its own prose -- the two
things every real contract has.
"""

from app.ingest.chunker import build_chunks
from app.ingest.parser import find_toc_span, parse_document, strip_toc

# Shaped like a real filing: TOC first, then the body repeating the same numbers.
WITH_TOC = """\
DEVELOPMENT AND OPTION AGREEMENT

TABLE OF CONTENTS

ARTICLE 1 DEFINITIONS 1

ARTICLE 2 COLLABORATION MANAGEMENT 18

2.1 Joint Governance Committee. 18 2.2 General Provisions. 19 2.3 Discontinuation. 20

ARTICLE 9 CONFIDENTIALITY 52

9.1 Product Information. 52 9.2 Confidentiality Obligations. 53

ARTICLE 1 DEFINITIONS

1.1 Affiliate. "Affiliate" means any entity controlling a Party.

ARTICLE 2 COLLABORATION MANAGEMENT

2.1 Joint Governance Committee. The Parties shall establish a JGC within thirty days
of the Effective Date to oversee the Collaboration.

2.1.1 Formation. The JGC shall comprise three representatives of each Party.

2.2 General Provisions. The following provisions govern the JGC and survive
termination of this Agreement.

2.2.2 Procedural Rules. The JGC shall adopt such standing rules as are necessary,
subject to Article 9.

ARTICLE 9 CONFIDENTIALITY

9.2 Confidentiality Obligations. Each Party shall keep confidential all Confidential
Information disclosed by the other Party.
"""


class TestTocDetection:
    def test_finds_the_toc_span(self):
        span = find_toc_span(WITH_TOC.split("\n"))
        assert span is not None
        start, end = span
        lines = WITH_TOC.split("\n")
        assert "TABLE OF CONTENTS" in lines[start]
        # The span must stop before the real body begins.
        assert end < lines.index("ARTICLE 1 DEFINITIONS")

    def test_strip_removes_page_number_entries(self):
        remaining = "\n".join(strip_toc(WITH_TOC.split("\n")))
        assert "TABLE OF CONTENTS" not in remaining
        assert "2.1 Joint Governance Committee. 18 2.2" not in remaining
        # Real content survives.
        assert "shall establish a JGC within thirty days" in remaining

    def test_document_without_a_toc_is_untouched(self):
        plain = "4.1 Scope. The parties agree.\n4.2 Term. One year.\n"
        assert find_toc_span(plain.split("\n")) is None
        assert strip_toc(plain.split("\n")) == plain.split("\n")

    def test_body_mentioning_a_section_is_not_a_toc(self):
        body = "\n".join(
            f"{i}.1 Clause. The obligations under Section 4.2 shall apply here."
            for i in range(1, 9)
        )
        assert find_toc_span(body.split("\n")) is None


class TestTocRepair:
    def test_real_sections_keep_clean_ids(self):
        sections = parse_document(WITH_TOC, "doc_x")
        # Previously the TOC claimed "2.1" and the real clause became "2.1~2".
        assert "2.1~2" not in sections
        assert "2.1" in sections
        assert "establish a JGC" in sections["2.1"].own_text()

    def test_no_duplicate_numbering_survives(self):
        sections = parse_document(WITH_TOC, "doc_x")
        assert [s for s in sections if "~" in s] == []

    def test_subsections_attach_to_the_real_parent(self):
        # The splice: 2.1.1 used to find the TOC's "2.1" and hang off it.
        sections = parse_document(WITH_TOC, "doc_x")
        assert sections["2.1.1"].parent_id == "2.1"
        assert "establish a JGC" in sections[sections["2.1.1"].parent_id].own_text()

    def test_article_titles_lose_their_page_numbers(self):
        sections = parse_document(WITH_TOC, "doc_x")
        assert sections["ART-9"].title == "CONFIDENTIALITY"

    def test_article_nine_resolves_to_real_text_not_a_listing(self):
        # The failure that broke graph traversal: ART-9 fanned down to a TOC entry.
        chunks = {c.section_id: c for c in build_chunks(WITH_TOC, "doc_x")}
        assert "9.2" in chunks
        assert "keep confidential" in chunks["9.2"].text
        assert "52" not in chunks["9.2"].text  # no page numbers bleeding through

    def test_no_empty_chunks(self):
        chunks = build_chunks(WITH_TOC, "doc_x")
        assert [c.chunk_id for c in chunks if not c.text.strip()] == []


class TestContainerLeadIn:
    def test_container_intro_text_is_retrievable(self):
        chunks = {c.section_id: c for c in build_chunks(WITH_TOC, "doc_x")}
        assert "2.2" in chunks
        assert "survive" in chunks["2.2"].text

    def test_container_chunk_excludes_its_children(self):
        # Otherwise 2.2.2's text would be indexed twice.
        chunks = {c.section_id: c for c in build_chunks(WITH_TOC, "doc_x")}
        assert "standing rules" not in chunks["2.2"].text
        assert "standing rules" in chunks["2.2.2"].text

    def test_container_without_own_text_emits_nothing(self):
        text = "1. HEADING.\n1.1 Scope. The parties agree to the following terms here.\n"
        chunks = {c.section_id for c in build_chunks(text, "doc_y")}
        assert "1" not in chunks
        assert "1.1" in chunks

    def test_short_fragments_do_not_become_chunks(self):
        text = "2.2 X.\n2.2.1 Real. " + ("Body text here. " * 10) + "\n"
        chunks = {c.section_id for c in build_chunks(text, "doc_y")}
        assert "2.2" not in chunks
