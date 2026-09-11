import pytest
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.fusion import reciprocal_rank_fusion


def order(results):
    return [r.key for r in results]


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"]})
        assert order(fused) == ["a", "b", "c"]

    def test_agreement_between_retrievers_wins(self):
        # "b" is 2nd in both lists; "a" and "c" are 1st in one list each. Appearing
        # in both beats topping one -- the reason hybrid retrieval helps at all.
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": ["c", "b"]})
        assert order(fused)[0] == "b"

    def test_scores_follow_the_formula(self):
        fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["a"]}, k=60)
        # 1/(60+1) from each retriever.
        assert fused[0].score == pytest.approx(2 / 61)

    def test_ranks_are_reported_per_retriever(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": ["b"]})
        by_key = {r.key: r for r in fused}
        assert by_key["b"].ranks == {"dense": 2, "sparse": 1}
        assert by_key["a"].ranks == {"dense": 1}

    def test_k_damps_the_head(self):
        # "a" is #1 in one list; "x" is #4 in both. Small k makes the top slot
        # decisive, large k lets cross-retriever agreement overtake it.
        lists = {"dense": ["a", "p", "q", "x"], "sparse": ["b", "r", "s", "x"]}
        assert reciprocal_rank_fusion(lists, k=1)[0].key in ("a", "b")
        assert reciprocal_rank_fusion(lists, k=60)[0].key == "x"

    def test_weights_are_applied(self):
        fused = reciprocal_rank_fusion(
            {"dense": ["a", "b"], "sparse": ["b", "a"]}, weights={"dense": 5.0}
        )
        assert order(fused)[0] == "a"

    def test_union_not_intersection(self):
        fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]})
        assert set(order(fused)) == {"a", "b"}

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"dense": []}) == []

    def test_ordering_is_deterministic(self):
        lists = {"dense": ["a", "b"], "sparse": ["b", "a"]}
        assert order(reciprocal_rank_fusion(lists)) == order(reciprocal_rank_fusion(lists))


class TestTokenizer:
    def test_section_numbers_survive_as_tokens(self):
        assert "8.2" in tokenize("obligations under Section 8.2 hereof")

    def test_hyphenated_compounds_stay_whole(self):
        assert "non-disclosure" in tokenize("a non-disclosure obligation")

    def test_stopwords_removed_but_legal_terms_kept(self):
        tokens = tokenize("the party shall indemnify the other")
        assert "the" not in tokens
        assert "indemnify" in tokens
        assert "party" in tokens


@pytest.fixture
def bm25_index():
    payloads = [
        {
            "chunk_id": "d__sec_8.2",
            "doc_id": "d",
            "section_id": "8.2",
            "section_title": "Termination for Cause",
            "text": "Either party may terminate this Agreement for material breach.",
        },
        {
            "chunk_id": "d__sec_4.1",
            "doc_id": "d",
            "section_id": "4.1",
            "section_title": "Confidentiality Obligations",
            "text": "The Receiving Party shall protect Confidential Information.",
        },
    ]
    return BM25Index(payloads)


class TestBM25Index:
    def test_lexical_match_ranks_first(self, bm25_index):
        results = bm25_index.search("material breach termination", limit=5)
        assert results[0][0] == "d__sec_8.2"

    def test_section_number_query_hits_via_the_heading(self, bm25_index):
        results = bm25_index.search("Section 8.2", limit=5)
        assert results and results[0][0] == "d__sec_8.2"

    def test_zero_score_results_are_dropped(self, bm25_index):
        assert bm25_index.search("xylophone unrelated", limit=5) == []

    def test_empty_query(self, bm25_index):
        assert bm25_index.search("the of and", limit=5) == []

    def test_empty_corpus(self):
        assert BM25Index([]).search("anything", limit=5) == []
