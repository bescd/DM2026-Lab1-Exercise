import matplotlib

# Headless backend for tests: some tools build a matplotlib Figure (describe_data_tool,
# dtm_heatmap_tool, feature_correlation_matrix_tool, reduce_dimensions_tool,
# visualize_result_tool). Outside Jupyter (which picks its own inline backend), matplotlib
# otherwise defaults to TkAgg, which can crash here with a broken/incomplete Tk install.
# Agg never touches Tk and is what the tools actually need: they only set
# session.pending_figure, never plt.show().
matplotlib.use("Agg")

import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

FIXTURE = {"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"}


@pytest.fixture
def session():
    return SessionState()


@pytest.fixture
def tools(session):
    """Every tool (workspace + provided), by name, bound to one fresh session."""
    workspace_tools, problems = load_workspace_tools("agent_dev_workspace", session)
    premade_tools, premade_problems = load_workspace_tools("premade_tools", session)
    assert not problems and not premade_problems
    return {t.name: t for t in workspace_tools + premade_tools}


@pytest.fixture
def loaded(tools):
    """The 8-row fixture already loaded into the session."""
    result = tools["load_dataset_tool"].invoke(FIXTURE)
    assert result["n_documents"] == 8
    return tools


@pytest.fixture
def dtm(loaded):
    """The fixture loaded and its document-term matrix built."""
    loaded["build_dtm_tool"].invoke({})
    return loaded
