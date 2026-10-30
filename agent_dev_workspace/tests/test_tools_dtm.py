import pytest


def test_build_dtm_known_answer_4(loaded, session):
    result = loaded["build_dtm_tool"].invoke({})
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix.shape == (8, 5)
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5)
    assert "count_vectorizer" in session.artifacts
    assert session.categories == ["catA", "catB"]


def test_build_dtm_stop_words_and_ngrams(loaded, session):
    loaded["build_dtm_tool"].invoke({"stop_words": "english"})
    assert "always" not in session.feature_names
    loaded["build_dtm_tool"].invoke({"ngram_min": 1, "ngram_max": 2})
    assert "always alpha" in session.feature_names
    assert "error" in loaded["build_dtm_tool"].invoke({"ngram_min": 2, "ngram_max": 1})


def test_term_frequency_known_answer_14(dtm):
    result = dtm["term_frequency_tool"].invoke({})
    assert {t["term"]: t["frequency"] for t in result["top_terms"]} == {
        "alpha": 6, "always": 7, "beta": 3, "delta": 4, "gamma": 7,
    }
    assert result["top_terms"][0]["frequency"] == 7
    assert result["total_count"] == 27


def test_term_frequency_result_can_be_charted(dtm, session):
    result = dtm["term_frequency_tool"].invoke({"top_n": 3})
    assert len(result["top_terms"]) == 3
    chart = dtm["visualize_result_tool"].invoke({"result_id": result["result_id"]})
    assert chart["rows_plotted"] == 5
    assert session.pending_figure is not None


def test_dtm_heatmap_known_answer_21(dtm, session):
    full = dtm["dtm_heatmap_tool"].invoke({})
    assert full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert full["matrix"] == [
        [3, 1, 1, 0, 0], [2, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2], [0, 1, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0],
    ]
    assert session.pending_figure is not None

    corner = dtm["dtm_heatmap_tool"].invoke({"n_terms": 2, "n_documents": 3})
    assert corner["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert corner["terms"] == ["alpha", "always"]


def test_dtm_tools_ask_for_a_matrix_first(loaded):
    for name in ("term_frequency_tool", "dtm_heatmap_tool"):
        assert "error" in loaded[name].invoke({}), name
