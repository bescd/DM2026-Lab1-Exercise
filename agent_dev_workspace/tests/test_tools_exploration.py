import numpy as np
import pytest


def similarity(tools, a, b):
    return tools["cosine_similarity_tool"].invoke({"doc_index_1": a, "doc_index_2": b})["cosine_similarity"]


def test_cosine_similarity_known_answer_16(dtm):
    assert similarity(dtm, 0, 1) == pytest.approx(0.9847, abs=1e-4)
    assert similarity(dtm, 0, 3) == pytest.approx(0.0909, abs=1e-4)
    assert similarity(dtm, 3, 4) == pytest.approx(0.9847, abs=1e-4)
    assert similarity(dtm, 0, 7) == 0.0  # the empty-text row: exactly 0, not NaN


def test_cosine_similarity_with_itself_is_one_and_checks_range(dtm):
    assert similarity(dtm, 2, 2) == pytest.approx(1.0)
    assert "error" in dtm["cosine_similarity_tool"].invoke({"doc_index_1": 0, "doc_index_2": 99})


def test_feature_correlation_known_answer_19(dtm, session):
    result = dtm["feature_correlation_matrix_tool"].invoke({})
    assert result["terms"] == ["alpha", "gamma", "delta", "beta", "always"]
    expected = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]
    assert np.allclose(result["correlation_matrix"], expected, atol=1e-4)
    assert np.allclose(np.diag(result["correlation_matrix"]), 1.0)
    assert session.pending_figure is not None


def test_exploration_tools_need_a_matrix(loaded):
    assert "error" in loaded["cosine_similarity_tool"].invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "error" in loaded["feature_correlation_matrix_tool"].invoke({})
