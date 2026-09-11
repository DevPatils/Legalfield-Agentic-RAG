from app.ingest.crossref import (
    CrossRefResolver,
    extract_defined_terms,
    extract_references,
)


def targets(text: str) -> list[str]:
    return [r.target for r in extract_references(text)]


def relation_for(text: str, target: str) -> str:
    return next(r.relation for r in extract_references(text) if r.target == target)


class TestReferenceExtraction:
    def test_plain_section_reference(self):
        assert targets("as provided in Section 4.2 hereof") == ["4.2"]

    def test_subclause_reference(self):
        assert targets("see Section 7.1(b)(i) for details") == ["7.1(b)(i)"]

    def test_article_reference_normalizes_roman(self):
        assert targets("pursuant to Article IX") == ["ART-9"]

    def test_section_symbol(self):
        assert targets("under § 12.4 of this Agreement") == ["12.4"]

    def test_reference_lists(self):
        assert targets("described in Sections 4.1, 4.2 and 4.5") == ["4.1", "4.2", "4.5"]

    def test_ranges_expand(self):
        assert targets("Sections 4.1 through 4.3 survive") == ["4.1", "4.2", "4.3"]

    def test_absurd_ranges_do_not_expand(self):
        assert targets("Sections 1.1 through 1.900") == ["1.1", "1.900"]

    def test_bare_roman_without_article_is_not_a_reference(self):
        assert targets("The parties agree (IV) to the following") == []

    def test_no_false_positive_on_plain_numbers(self):
        assert targets("within thirty (30) days of the closing") == []

    def test_deduplicates_repeats(self):
        assert targets("Section 4.1 and again Section 4.1") == ["4.1"]


class TestRelationLabelling:
    def test_subject_to(self):
        assert relation_for("subject to Section 8.3", "8.3") == "subject_to"

    def test_as_defined_in(self):
        assert relation_for("as defined in Section 1.3", "1.3") == "defined_in"

    def test_notwithstanding(self):
        assert relation_for("Notwithstanding Section 4.1", "4.1") == "notwithstanding"

    def test_pursuant_to(self):
        assert relation_for("pursuant to Article IV", "ART-4") == "pursuant_to"

    def test_unlabelled_reference_is_plain(self):
        assert relation_for("The parties acknowledge Section 5.1.", "5.1") == "plain"


class TestDefinedTerms:
    def test_means_definition(self):
        text = '"Confidential Information" means any non-public information.'
        assert extract_defined_terms(text) == ["Confidential Information"]

    def test_shall_mean_definition(self):
        text = '"Territory" shall mean the United States.'
        assert extract_defined_terms(text) == ["Territory"]

    def test_parenthetical_definition(self):
        text = 'disclosed by one party (the "Disclosing Party") to the other.'
        assert extract_defined_terms(text) == ["Disclosing Party"]

    def test_has_the_meaning(self):
        text = '"Affiliate" has the meaning given in Section 1.1.'
        assert extract_defined_terms(text) == ["Affiliate"]

    def test_quoted_non_definition_is_ignored(self):
        text = 'the party may deliver a "notice" to the other party'
        assert extract_defined_terms(text) == []


class TestResolver:
    def test_exact_match(self):
        resolver = CrossRefResolver({"4.2"}, {})
        assert resolver.resolve("4.2") == ["4.2"]

    def test_subclause_resolves_up_to_its_chunk(self):
        resolver = CrossRefResolver({"4.2"}, {})
        assert resolver.resolve("4.2(a)(i)") == ["4.2"]

    def test_deeper_numbering_resolves_up(self):
        resolver = CrossRefResolver({"4.2"}, {})
        assert resolver.resolve("4.2.1") == ["4.2"]

    def test_article_fans_down_to_its_sections(self):
        children = {"ART-4": ["4.1", "4.2"], "4.1": [], "4.2": []}
        resolver = CrossRefResolver({"4.1", "4.2"}, children)
        assert resolver.resolve("ART-4") == ["4.1", "4.2"]

    def test_fan_down_is_capped(self):
        children = {"ART-4": [f"4.{i}" for i in range(1, 20)]}
        resolver = CrossRefResolver({f"4.{i}" for i in range(1, 20)}, children)
        assert len(resolver.resolve("ART-4")) <= 8

    def test_unknown_target_resolves_to_nothing(self):
        resolver = CrossRefResolver({"4.2"}, {})
        assert resolver.resolve("99.9") == []
