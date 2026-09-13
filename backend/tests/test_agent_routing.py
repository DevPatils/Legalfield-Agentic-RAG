"""Routing, state and citation-verification logic -- everything that doesn't call an LLM.

The conditional edges are where a cyclic graph goes wrong (infinite loops, dead ends,
premature exits), so they get tested directly rather than only through a live run.
"""

from app.agent.graph import (
    make_route_after_sufficiency,
    route_after_planner,
    route_after_refine,
)
from app.agent.nodes import RE_CITATION, _split_sentences, format_clauses
from app.agent.state import (
    definitions_absent,
    merge_context,
    new_state,
    referenced_but_absent,
)
from app.retrieval.pipeline import RetrievedChunk


def chunk(section_id, doc="doc_001", refs=(), ref_ids=(), terms=(), defn_ids=(), text="body"):
    return RetrievedChunk(
        chunk_id=f"{doc}__sec_{section_id}",
        score=1.0,
        payload={
            "chunk_id": f"{doc}__sec_{section_id}",
            "doc_id": doc,
            "section_id": section_id,
            "section_title": f"Section {section_id}",
            "text": text,
            "cross_references": list(refs),
            "cross_reference_ids": list(ref_ids),
            "defined_terms_used": list(terms),
            "definition_ids": list(defn_ids),
        },
    )


class TestPlannerRouting:
    def test_retrieval_path(self):
        assert route_after_planner({"needs_retrieval": True}) == "retrieve"

    def test_skip_retrieval_for_out_of_scope(self):
        assert route_after_planner({"needs_retrieval": False}) == "generate"

    def test_defaults_to_retrieving(self):
        assert route_after_planner({}) == "retrieve"


class TestSufficiencyRouting:
    route = staticmethod(make_route_after_sufficiency(3))

    def test_sufficient_goes_to_generate(self):
        assert self.route({"sufficient": True, "iteration": 1}) == "generate"

    def test_insufficient_under_cap_refines(self):
        assert (
            self.route({"sufficient": False, "iteration": 1, "sections_needed": ["4.1"]})
            == "refine"
        )

    def test_iteration_cap_forces_generate(self):
        # The invariant that keeps the graph terminating.
        assert (
            self.route({"sufficient": False, "iteration": 3, "sections_needed": ["4.1"]})
            == "generate"
        )

    def test_past_cap_still_generates(self):
        assert self.route({"sufficient": False, "iteration": 9}) == "generate"

    def test_error_short_circuits(self):
        assert self.route({"sufficient": False, "iteration": 1, "error": "boom"}) == "generate"

    def test_first_pass_with_nothing_to_pull_still_tries_a_rewrite(self):
        state = {"sufficient": False, "iteration": 1, "refine_strategy": "none"}
        assert self.route(state) == "refine"

    def test_rewrite_that_found_nothing_gives_up(self):
        # A second rewrite with no graph edges to follow would repeat the same search.
        state = {"sufficient": False, "iteration": 2, "refine_strategy": "query_rewrite"}
        assert self.route(state) == "generate"


class TestRefineRouting:
    def test_graph_traversal_rechecks_sufficiency(self):
        # Traversal already fetched the clauses; re-searching would be wasted work.
        assert route_after_refine({"refine_strategy": "graph_traversal"}) == "sufficiency"

    def test_query_rewrite_goes_back_to_retrieval(self):
        assert route_after_refine({"refine_strategy": "query_rewrite"}) == "retrieve"


class TestMergeContext:
    def test_appends_new_chunks(self):
        merged = merge_context([chunk("4.1")], [chunk("4.2")])
        assert [c.section_id for c in merged] == ["4.1", "4.2"]

    def test_deduplicates(self):
        merged = merge_context([chunk("4.1")], [chunk("4.1")])
        assert len(merged) == 1

    def test_first_occurrence_wins_preserving_provenance(self):
        traversed = chunk("4.1")
        traversed.source = "graph_traversal"
        merged = merge_context([traversed], [chunk("4.1")])
        assert merged[0].source == "graph_traversal"

    def test_preserves_order(self):
        merged = merge_context([chunk("1.1")], [chunk("9.9"), chunk("1.1"), chunk("5.5")])
        assert [c.section_id for c in merged] == ["1.1", "9.9", "5.5"]


class TestGraphCandidates:
    def test_referenced_but_absent_finds_the_gap(self):
        ctx = [chunk("4.2", refs=["13.9"], ref_ids=["doc_001__sec_13.9"])]
        assert referenced_but_absent(ctx) == {"13.9": "doc_001__sec_13.9"}

    def test_present_references_are_not_candidates(self):
        ctx = [chunk("4.2", refs=["13.9"], ref_ids=["doc_001__sec_13.9"]), chunk("13.9")]
        assert referenced_but_absent(ctx) == {}

    def test_definitions_absent(self):
        ctx = [chunk("4.1", terms=["Confidential Information"], defn_ids=["doc_001__sec_1.3"])]
        assert definitions_absent(ctx) == {"Confidential Information": "doc_001__sec_1.3"}

    def test_definition_present_is_not_a_candidate(self):
        ctx = [
            chunk("4.1", terms=["Confidential Information"], defn_ids=["doc_001__sec_1.3"]),
            chunk("1.3"),
        ]
        assert definitions_absent(ctx) == {}

    def test_handles_missing_payload_fields(self):
        bare = RetrievedChunk(chunk_id="x", score=1.0, payload={"doc_id": "d", "section_id": "1"})
        assert referenced_but_absent([bare]) == {}
        assert definitions_absent([bare]) == {}


class TestCitationParsing:
    def test_matches_the_generate_contract(self):
        assert RE_CITATION.findall("as stated [doc_001 §4.2].") == [("doc_001", "4.2")]

    def test_tolerates_spacing(self):
        assert RE_CITATION.findall("[doc_001 § 4.2]") == [("doc_001", "4.2")]

    def test_multiple_citations(self):
        found = RE_CITATION.findall("first [doc_001 §4.1] and second [doc_002 §8.2]")
        assert found == [("doc_001", "4.1"), ("doc_002", "8.2")]

    def test_subclause_and_part_ids(self):
        assert RE_CITATION.findall("[doc_001 §4.2(a)(i)]") == [("doc_001", "4.2(a)(i)")]
        assert RE_CITATION.findall("[doc_001 §4.1__p2]") == [("doc_001", "4.1__p2")]

    def test_ignores_plain_brackets(self):
        assert RE_CITATION.findall("[see above] and [***]") == []


class TestSentenceSplitting:
    def test_does_not_split_on_section_numbers(self):
        text = "The term is defined in Section 4.2. It applies broadly [doc_001 §4.2]."
        assert len(_split_sentences(text)) == 2

    def test_does_not_split_inside_a_citation(self):
        assert len(_split_sentences("Obligations apply [doc_001 §8.2.1].")) == 1


class TestClauseFormatting:
    def test_header_matches_the_citation_the_model_must_emit(self):
        rendered = format_clauses([chunk("4.2", text="Some clause text.")])
        assert "[doc_001 §4.2]" in rendered
        assert "Some clause text." in rendered


class TestInitialState:
    def test_counters_start_clean(self):
        state = new_state("q", query_id="qid")
        assert state["iteration"] == 0
        assert state["context"] == []
        assert state["refine_strategy"] == "none"
        assert state["low_confidence"] is False
