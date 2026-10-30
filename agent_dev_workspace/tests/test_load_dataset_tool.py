from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_fixture():
    # Instantiate session state
    session = SessionState()

    # Load premade tools from the premade_tools directory
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Invoke load_dataset_tool with the test fixture CSV
    res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Assert standard known answer #1 from TEST_FIXTURE.md: 8 total documents, split 4 catA and 4 catB
    assert res["n_documents"] == 8
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}
