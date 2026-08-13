"""Example workflow tests.

The examples in examples/ are the first thing a new user loads, so a broken
one is worse than none at all. These checks run offline.

The important test here is widget *order*: the ComfyAgent node's
widgets_values is positional, so if agent_node.py ever reorders its inputs,
every example silently misconfigures — mode ends up in task_category, and the
node does something other than what its documentation says. That drift is
invisible without this test.

    pytest tests/                    # collected as test_* functions
    python tests/test_examples.py    # standalone, no pytest needed
"""

import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_DIR = BASE / "examples"

# Modes the node actually offers. EXPLAIN/DEBUG/SUGGEST were superseded by
# CHAT and must never reappear in a shipped example.
VALID_MODES = {"BUILD", "MODIFY", "CHAT"}


def _examples():
    return sorted(EXAMPLES_DIR.glob("*.json"))


def _load(path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _agent_nodes(workflow):
    return [n for n in workflow["nodes"] if n.get("type") == "ComfyAgent"]


def test_examples_exist():
    found = _examples()
    assert found, "examples/ should ship at least one workflow"
    print(f"ok: {len(found)} example workflow(s) present")


def test_examples_are_valid_workflow_json():
    for path in _examples():
        workflow = _load(path)
        for key in ("nodes", "links", "version", "last_node_id"):
            assert key in workflow, f"{path.name} is missing '{key}'"
        assert isinstance(workflow["nodes"], list) and workflow["nodes"]
        assert isinstance(workflow["links"], list)
    print("ok: every example has the keys ComfyUI needs to load it")


def test_example_node_ids_are_unique_and_consistent():
    for path in _examples():
        workflow = _load(path)
        ids = [n["id"] for n in workflow["nodes"]]
        assert len(ids) == len(set(ids)), f"{path.name} has duplicate node ids"
        assert workflow["last_node_id"] == max(ids), (
            f"{path.name}: last_node_id must match the highest node id, or "
            "ComfyUI reuses an id already in the graph"
        )
    print("ok: node ids are unique and last_node_id is correct")


def test_example_links_reference_real_nodes():
    for path in _examples():
        workflow = _load(path)
        ids = {n["id"] for n in workflow["nodes"]}
        for link in workflow["links"]:
            # LiteGraph link: [link_id, from_node, from_slot, to_node, to_slot, type]
            assert link[1] in ids, f"{path.name}: link from unknown node {link[1]}"
            assert link[3] in ids, f"{path.name}: link to unknown node {link[3]}"
    print("ok: every link references a node that exists in the same file")


def test_every_example_contains_an_agent_node():
    for path in _examples():
        assert _agent_nodes(_load(path)), f"{path.name} has no ComfyAgent node"
    print("ok: every example demonstrates the agent node")


def test_example_modes_are_offered_by_the_node():
    """A superseded mode in an example would send the user straight into an
    error message."""
    for path in _examples():
        for node in _agent_nodes(_load(path)):
            mode = node["widgets_values"][1]
            assert mode in VALID_MODES, f"{path.name} uses retired mode {mode!r}"
    print("ok: examples only use modes the node still offers")


def test_agent_widget_order_matches_the_node_schema():
    """The guard that matters: widgets_values is positional, so a reordering
    in agent_node.py silently rewires every example."""
    source = (BASE / "agent_node.py").read_text(encoding="utf-8")
    # Input names in declaration order, e.g. io.String.Input("task_prompt",
    declared = re.findall(r'io\.\w+\.Input\(\s*"(\w+)"', source)
    # context_input is optional and carries no widget, so it is not serialized.
    expected = [n for n in declared if n != "context_input"]

    assert expected[:6] == [
        "task_prompt",
        "mode",
        "task_category",
        "model_family",
        "max_nodes",
        "use_blueprint",
    ], f"agent_node.py input order changed: {expected}"

    for path in _examples():
        for node in _agent_nodes(_load(path)):
            values = node["widgets_values"]
            assert len(values) == len(expected), (
                f"{path.name}: {len(values)} widget values for "
                f"{len(expected)} declared inputs {expected}"
            )
    print(f"ok: example widget order matches the node schema {expected}")


def test_example_widget_values_have_the_right_types():
    for path in _examples():
        for node in _agent_nodes(_load(path)):
            prompt, mode, category, family, max_nodes, use_blueprint = node["widgets_values"]
            assert isinstance(prompt, str)
            assert isinstance(mode, str)
            assert isinstance(category, str)
            assert isinstance(family, str)
            # 4..20 mirrors the bounds in agent_node.py and agent_routes.py.
            assert isinstance(max_nodes, int) and 4 <= max_nodes <= 20
            assert isinstance(use_blueprint, bool)
    print("ok: widget values match their declared types and bounds")


def test_examples_are_documented():
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for path in _examples():
        assert path.name in readme, f"{path.name} is not described in examples/README.md"
    print("ok: every example file is documented")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall example tests passed")
