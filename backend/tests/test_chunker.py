import pytest
from app.ingest.chunker import build_chunks
from app.ingest.models import make_chunk_id


@pytest.fixture
def chunks(sample_contract, doc_id):
    return build_chunks(sample_contract, doc_id)


@pytest.fixture
def by_section(chunks):
    return {c.section_id: c for c in chunks}


class TestChunkShape:
    def test_chunk_ids_follow_the_structural_convention(self, chunks, doc_id):
        for c in chunks:
            assert c.chunk_id == make_chunk_id(doc_id, c.section_id)

    def test_leaf_sections_become_chunks(self, by_section):
        for section_id in ("4.1", "4.2", "4.3", "8.2", "8.3"):
            assert section_id in by_section

    def test_container_nodes_do_not_become_chunks(self, by_section):
        # ARTICLE IV holds sections, so it is not itself a chunk.
        assert "ART-4" not in by_section

    def test_subclauses_are_folded_into_their_parent(self, by_section):
        assert "4.2(a)" not in by_section
        assert "publicly available" in by_section["4.2"].text
        assert "documented in writing" in by_section["4.2"].text

    def test_titles_and_paths_are_populated(self, by_section):
        chunk = by_section["4.2"]
        assert chunk.section_title == "Exceptions"
        assert "ARTICLE 4" in chunk.path

    def test_citation_format_matches_the_generate_contract(self, by_section):
        assert by_section["4.2"].citation == "[doc_001 §4.2]"


class TestCrossReferenceGraph:
    def test_explicit_references_become_edges(self, by_section, doc_id):
        chunk = by_section["4.2"]
        assert "4.1" in chunk.cross_references
        assert make_chunk_id(doc_id, "4.1") in chunk.cross_reference_ids

    def test_range_references_expand_into_edges(self, by_section, doc_id):
        # 8.3 says "Sections 4.1 through 4.3".
        chunk = by_section["8.3"]
        for section_id in ("4.1", "4.2", "4.3"):
            assert make_chunk_id(doc_id, section_id) in chunk.cross_reference_ids

    def test_article_reference_fans_down(self, by_section, doc_id):
        # 8.2 says "pursuant to Article IV".
        chunk = by_section["8.2"]
        assert make_chunk_id(doc_id, "4.1") in chunk.cross_reference_ids

    def test_self_references_are_dropped(self, by_section, doc_id):
        for chunk in by_section.values():
            assert chunk.chunk_id not in chunk.cross_reference_ids

    def test_edges_point_at_real_chunks(self, chunks):
        known = {c.chunk_id for c in chunks}
        for chunk in chunks:
            for edge in chunk.cross_reference_ids:
                assert edge in known, f"{chunk.chunk_id} -> dangling {edge}"


class TestDefinitionalDependency:
    def test_definition_chunk_is_identified(self, by_section):
        assert "Confidential Information" in by_section["1.3"].defines_terms

    def test_usage_links_back_to_the_definition(self, by_section, doc_id):
        chunk = by_section["4.1"]
        assert "Confidential Information" in chunk.defined_terms_used
        assert make_chunk_id(doc_id, "1.3") in chunk.definition_ids

    def test_defining_chunk_does_not_depend_on_itself(self, by_section):
        chunk = by_section["1.3"]
        assert "Confidential Information" not in chunk.defined_terms_used
        assert chunk.chunk_id not in chunk.definition_ids


class TestOversizedSections:
    def test_long_sections_split_into_parts(self, doc_id):
        body = "\n\n".join(f"Paragraph {i} of the clause body." for i in range(400))
        chunks = build_chunks(f"4.1 Long Clause. Intro.\n\n{body}\n", doc_id)
        parts = [c for c in chunks if c.section_id.startswith("4.1")]
        assert len(parts) > 1
        assert all(c.num_parts == len(parts) for c in parts)
        assert [c.part for c in parts] == list(range(1, len(parts) + 1))

    def test_references_to_a_split_section_hit_part_one(self, doc_id):
        body = "\n\n".join(f"Paragraph {i}." for i in range(400))
        text = f"4.1 Long Clause. Intro.\n\n{body}\n\n9.1 Pointer. See Section 4.1 above.\n"
        chunks = build_chunks(text, doc_id)
        pointer = next(c for c in chunks if c.section_id == "9.1")
        assert make_chunk_id(doc_id, "4.1__p1") in pointer.cross_reference_ids
