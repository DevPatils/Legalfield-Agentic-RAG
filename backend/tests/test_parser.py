from app.ingest.parser import (
    _classify_marker,
    _split_title_body,
    parse_document,
    roman_to_int,
    section_path,
    subtree_text,
)


class TestRomanNumerals:
    def test_valid(self):
        assert roman_to_int("IV") == 4
        assert roman_to_int("IX") == 9
        assert roman_to_int("VIII") == 8
        assert roman_to_int("XIV") == 14

    def test_invalid_returns_zero(self):
        assert roman_to_int("IIII") == 0
        assert roman_to_int("ABC") == 0


class TestTitleBodySplit:
    def test_inline_heading_is_separated(self):
        title, body = _split_title_body("Exceptions. The obligations shall not apply.")
        assert title == "Exceptions"
        assert body == "The obligations shall not apply."

    def test_sentence_is_not_mistaken_for_a_title(self):
        long_sentence = (
            "The Receiving Party shall protect all Confidential Information "
            "with the same degree of care. It shall not disclose it."
        )
        title, body = _split_title_body(long_sentence)
        assert title == ""
        assert body == long_sentence

    def test_bare_title(self):
        title, body = _split_title_body("CONFIDENTIALITY.")
        assert title == "CONFIDENTIALITY"
        assert body == ""


class TestMarkerClassification:
    def test_plain_letters_are_alpha(self):
        assert _classify_marker("a", []) == "alpha"
        assert _classify_marker("b", []) == "alpha"

    def test_i_after_h_continues_the_alpha_run(self):
        assert _classify_marker("i", [("alpha", "h")]) == "alpha"

    def test_i_under_an_open_alpha_level_is_roman(self):
        assert _classify_marker("i", [("alpha", "b")]) == "roman"

    def test_multi_char_romans(self):
        assert _classify_marker("iii", []) == "roman"
        assert _classify_marker("iv", []) == "roman"

    def test_digits_and_uppercase(self):
        assert _classify_marker("3", []) == "num"
        assert _classify_marker("B", []) == "upper"


class TestParseDocument:
    def test_articles_are_captured_with_roman_ids(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        assert "ART-1" in sections
        assert "ART-4" in sections
        assert "ART-8" in sections
        assert sections["ART-4"].title == "CONFIDENTIALITY"

    def test_sections_nest_under_their_article(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        # No bare "4" heading exists in the sample, so 4.x hangs off ARTICLE IV.
        assert "4" not in sections
        assert sections["4.2"].parent_id == "ART-4"
        assert sections["8.2"].parent_id == "ART-8"

    def test_wrapped_line_starting_with_section_is_not_a_heading(self, doc_id):
        text = (
            "Section 4.2 Exceptions. The obligations set forth in\n"
            "Section 4.1 shall not apply to public information.\n"
        )
        sections = parse_document(text, doc_id)
        assert "4.1" not in sections
        assert "shall not apply" in sections["4.2"].own_text()

    def test_dotted_sections_nest_under_their_parent_number(self, doc_id):
        text = "1. DEFINITIONS.\n1.1 Scope. Something.\n1.2 Other. Something else.\n"
        sections = parse_document(text, doc_id)
        assert sections["1.1"].parent_id == "1"
        assert sections["1.2"].parent_id == "1"

    def test_inline_titles_are_extracted(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        assert sections["4.2"].title == "Exceptions"
        assert sections["8.2"].title == "Termination"

    def test_subclauses_build_a_nested_tree(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        assert "4.2(a)" in sections
        assert "4.2(b)" in sections
        # (i)/(ii) sit under (b), not as siblings of (a).
        assert "4.2(b)(i)" in sections
        assert sections["4.2(b)(i)"].parent_id == "4.2(b)"
        # (c) returns to the alpha level rather than staying under (b).
        assert sections["4.2(c)"].parent_id == "4.2"

    def test_body_text_lands_on_the_right_node(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        assert "publicly available" in sections["4.2(a)"].own_text()
        assert "thirty (30) days" in sections["8.2"].own_text()

    def test_preamble_is_captured(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        preamble = [s for s in sections.values() if s.kind == "preamble"]
        recitals = [s for s in sections.values() if "Effective Date" in s.own_text()]
        assert preamble or recitals

    def test_duplicate_numbering_does_not_clobber(self, doc_id):
        text = "1. First. Alpha.\n\n1. Second. Beta.\n"
        sections = parse_document(text, doc_id)
        assert "1" in sections
        assert "1~2" in sections

    def test_page_artifacts_are_stripped(self, doc_id):
        text = "4.1 Scope. Alpha.\nPage 3 of 12\n_______\nMore text.\n"
        sections = parse_document(text, doc_id)
        body = sections["4.1"].own_text()
        assert "Page 3" not in body
        assert "More text." in body


class TestDerivedViews:
    def test_section_path_is_a_breadcrumb(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        path = section_path(sections, "4.2")
        assert "ARTICLE 4" in path
        assert "Exceptions" in path

    def test_subtree_text_includes_subclauses(self, sample_contract, doc_id):
        sections = parse_document(sample_contract, doc_id)
        text = subtree_text(sections, "4.2")
        assert "publicly available" in text
        assert "documented in writing" in text
