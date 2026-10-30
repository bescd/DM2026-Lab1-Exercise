import pytest


def test_all_data_tools_are_loaded(tools):
    expected = {
        "check_missing_tool", "tokenize_tool", "list_files_tool", "inspect_data_tool",
        "check_duplicates_tool", "sample_data_tool", "describe_data_tool",
    }
    assert expected <= set(tools)


def test_check_missing_known_answer_2(loaded):
    result = loaded["check_missing_tool"].invoke({})
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_known_answer_3(loaded, session):
    report = loaded["check_duplicates_tool"].invoke({"drop": False})
    assert report["n_duplicates"] == 1
    assert report["dropped"] is False
    assert len(session.dataframe) == 8

    dropped = loaded["check_duplicates_tool"].invoke({"drop": True})
    # keep=False: both copies of "always gamma delta" go, not only the extra one.
    assert dropped["n_rows_after"] == 6
    assert len(session.dataframe) == 6
    assert 5 not in session.dataframe.index and 6 not in session.dataframe.index


def test_drop_duplicates_clears_the_stale_matrix(dtm, session):
    result = dtm["check_duplicates_tool"].invoke({"drop": True})
    assert session.feature_matrix is None
    assert "note" in result


def test_tokenize_known_answer_10(loaded, session):
    result = loaded["tokenize_tool"].invoke({})
    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[4] == ["always", "gamma", "gamma", "delta"]
    assert unigrams[7] == []
    assert result["n_empty_documents"] == 1
    assert result["total_tokens"] == 5 + 4 + 3 + 5 + 4 + 3 + 3 + 0


def test_check_duplicates_still_works_after_tokenizing(loaded):
    loaded["tokenize_tool"].invoke({})
    assert loaded["check_duplicates_tool"].invoke({})["n_duplicates"] == 1


def test_list_files_known_answer_11(tools):
    result = tools["list_files_tool"].invoke({"subdirectory": "newdataset"})
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_refuses_paths_outside_the_repo(tools):
    assert "error" in tools["list_files_tool"].invoke({"subdirectory": ".."})
    assert "error" in tools["list_files_tool"].invoke({"subdirectory": "no_such_folder"})


def test_sample_data_known_answer_13(loaded, session):
    result = loaded["sample_data_tool"].invoke({"n": 4, "random_state": 42})
    assert session.dataframe.index.tolist() == [1, 5, 0, 7]
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["n_after"] == 4
    assert len(session.artifacts["dataframe_before_sampling"]) == 8


def test_sample_data_rejects_too_large_n(loaded):
    assert "error" in loaded["sample_data_tool"].invoke({"n": 99})


def test_describe_data_known_answer_18(loaded, session):
    result = loaded["describe_data_tool"].invoke({})
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-4)
    assert (overall["min"], overall["25%"], overall["50%"], overall["75%"], overall["max"]) == (0, 17.75, 20.5, 25.25, 30)
    cat_a, cat_b = result["per_category"]["catA"], result["per_category"]["catB"]
    assert cat_a["mean"] == 17.25 and cat_a["std"] == 12.5 and cat_a["50%"] == 20.0
    assert cat_b["mean"] == 22.5 and cat_b["std"] == pytest.approx(5.7446, abs=1e-4) and cat_b["max"] == 30
    assert session.pending_figure is not None, "the box plot must be set on session.pending_figure"


def test_inspect_data_shows_rows(loaded):
    """No known answer exists for this tool: expectations written by hand from the fixture."""
    result = loaded["inspect_data_tool"].invoke({"n_rows": 3})
    assert result["shape"] == [8, 3]
    assert [r["text"] for r in result["rows"]] == ["always alpha alpha alpha beta", "always alpha alpha beta", "always alpha beta"]
    assert result["rows"][0]["category_name"] == "catA"


def test_inspect_data_truncates_and_handles_the_end(loaded):
    result = loaded["inspect_data_tool"].invoke({"n_rows": 10, "start": 6, "max_chars": 8})
    assert [r["position"] for r in result["rows"]] == [6, 7]
    assert result["rows"][0]["text"] == "always g..."
    assert result["rows"][0]["text_length"] == 18


def test_data_tools_ask_for_a_dataset_first(tools):
    for name in ("check_missing_tool", "tokenize_tool", "inspect_data_tool", "check_duplicates_tool", "describe_data_tool"):
        assert "error" in tools[name].invoke({}), name
