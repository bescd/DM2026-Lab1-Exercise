import pytest


def mine(tools, category, method, **kwargs):
    return tools["mine_patterns_tool"].invoke({"category_name": category, "filtering_method": method, **kwargs})


@pytest.mark.parametrize("method", ["variance", "term_frequency"])
@pytest.mark.parametrize("category", ["catA", "catB"])
def test_known_answer_17_zero_terms_survive_cleanly(loaded, method, category):
    result = mine(loaded, category, method, min_sup=1)
    assert result["n_terms_before_filter"] == 2
    assert result["n_terms_after_filter"] == 0
    assert result["patterns"] == []
    assert result["note"] == "no terms survived filtering"


@pytest.mark.parametrize("category, kept, support", [("catA", "alpha", 3), ("catB", "gamma", 4)])
def test_known_answer_17_tfidf_fpgrowth(loaded, session, category, kept, support):
    result = mine(loaded, category, "tfidf", min_sup=1)
    assert result["kept_terms_preview"] == [kept]
    assert result["patterns"] == [{"terms": [kept], "support": support}]
    assert session.artifacts[f"patterns_{category}_tfidf_fpgrowth"] == result["patterns"]


def test_known_answer_17_min_sup_range(loaded):
    assert mine(loaded, "catA", "tfidf", min_sup=3)["n_patterns"] == 1
    assert mine(loaded, "catA", "tfidf", min_sup=4)["n_patterns"] == 0
    assert mine(loaded, "catB", "tfidf", min_sup=4)["n_patterns"] == 1
    assert mine(loaded, "catB", "tfidf", min_sup=5)["n_patterns"] == 0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_known_answer_20_topk(loaded, k):
    assert mine(loaded, "catA", "tfidf", algorithm="topk", k=k)["patterns"] == [{"terms": ["alpha"], "support": 3}]
    assert mine(loaded, "catB", "tfidf", algorithm="topk", k=k)["patterns"] == [{"terms": ["gamma"], "support": 4}]


@pytest.mark.parametrize("min_sup", [1, 4])
@pytest.mark.parametrize("category", ["catA", "catB"])
def test_known_answer_20_maxfpgrowth_is_empty(loaded, category, min_sup):
    result = mine(loaded, category, "tfidf", algorithm="maxfpgrowth", min_sup=min_sup)
    assert result["n_patterns"] == 0 and result["patterns"] == []


def test_mining_is_per_category_and_independent_of_the_global_dtm(loaded, session):
    mine(loaded, "catA", "tfidf", min_sup=1)
    assert session.feature_matrix is None


def test_argument_checks(loaded):
    assert "error" in mine(loaded, "catA", "tfidf")  # fpgrowth without min_sup
    assert "error" in mine(loaded, "catA", "tfidf", algorithm="topk")  # topk without k
    assert "error" in mine(loaded, "catA", "bogus", min_sup=1)
    assert "error" in mine(loaded, "catA", "tfidf", algorithm="bogus", min_sup=1)
    unknown = mine(loaded, "nope", "tfidf", min_sup=1)
    assert unknown["available_categories"] == ["catA", "catB"]
