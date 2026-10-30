import pytest


def test_variance_filter_known_answer_5(dtm):
    result = dtm["variance_filter_tool"].invoke({"threshold": 0.15})
    variances = {t["term"]: t["variance"] for t in result["top_terms"]}
    assert variances["alpha"] == pytest.approx(1.1875)
    assert variances["always"] == pytest.approx(0.109375)
    assert variances["beta"] == pytest.approx(0.234375)
    assert variances["delta"] == pytest.approx(0.25)
    assert variances["gamma"] == pytest.approx(1.109375)
    assert sorted(result["kept_terms_preview"]) == ["alpha", "beta", "delta", "gamma"]
    assert result["removed_terms_preview"] == ["always"]
    assert (result["n_kept"], result["n_removed"]) == (4, 1)


def test_variance_filter_does_not_change_the_matrix(dtm, session):
    dtm["variance_filter_tool"].invoke({"threshold": 0.15})
    assert session.feature_matrix.shape == (8, 5)


def test_variance_filter_result_can_be_charted(dtm):
    result = dtm["variance_filter_tool"].invoke({"threshold": 0.15})
    chart = dtm["visualize_result_tool"].invoke({"result_id": result["result_id"], "top_n": 3})
    assert chart["rows_plotted"] == 3


def test_full_report_has_term_first_value_second_sorted(dtm, session):
    for tool_name, args, value_column in (
        ("variance_filter_tool", {}, "variance"),
        ("pearson_filter_tool", {"target_class": "catB"}, "pearson_r"),
        ("spearman_filter_tool", {"target_class": "catB"}, "spearman_r"),
    ):
        result = dtm[tool_name].invoke(args)
        report = session.results_store[result["result_id"]]["full_report"]
        assert list(report.columns[:2]) == ["feature", value_column], tool_name
        values = report[value_column] if value_column == "variance" else report[value_column].abs()
        assert values.is_monotonic_decreasing, tool_name


def test_pearson_known_answer_6(dtm):
    result = dtm["pearson_filter_tool"].invoke({"target_class": "catB"})
    r = {t["term"]: t["pearson_r"] for t in result["top_terms"]}
    assert r["alpha"] == pytest.approx(-0.6882, abs=1e-4)
    assert r["always"] == pytest.approx(0.3780, abs=1e-4)
    assert r["beta"] == pytest.approx(-0.7746, abs=1e-4)
    assert r["delta"] == pytest.approx(1.0, abs=1e-4)
    assert r["gamma"] == pytest.approx(0.8307, abs=1e-4)
    assert result["top_terms"][0]["term"] == "delta"


def test_spearman_known_answer_6(dtm):
    result = dtm["spearman_filter_tool"].invoke({"target_class": "catB"})
    r = {t["term"]: t["spearman_r"] for t in result["top_terms"]}
    assert r["alpha"] == pytest.approx(-0.7500, abs=1e-4)
    assert r["always"] == pytest.approx(0.3780, abs=1e-4)
    assert r["beta"] == pytest.approx(-0.7746, abs=1e-4)
    assert r["delta"] == pytest.approx(1.0, abs=1e-4)
    assert r["gamma"] == pytest.approx(0.9363, abs=1e-4)


def test_correlation_filters_reject_an_unknown_class(dtm):
    for name in ("pearson_filter_tool", "spearman_filter_tool"):
        result = dtm[name].invoke({"target_class": "nope"})
        assert "error" in result and result["available_categories"] == ["catA", "catB"]


def test_filters_ask_for_a_matrix_first(loaded):
    assert "error" in loaded["variance_filter_tool"].invoke({})
    assert "error" in loaded["pearson_filter_tool"].invoke({"target_class": "catB"})
