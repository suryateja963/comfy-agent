"""MODIFY mode tests: diff validation against the real canvas and registry.

This is the code path that deletes and rewrites work the user already has, so
the tests below are mostly about what must be *rejected*. Ollama and ComfyUI
are stubbed; nothing here reaches the network.

    pytest tests/                  # collected as test_* functions
    python tests/test_modify.py    # standalone, no pytest needed
"""

import asyncio
import importlib.util
import pathlib
import sys
import types

BASE = str(pathlib.Path(__file__).resolve().parent.parent)


def _register_pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__file__ = path
    sys.modules[name] = mod
    return mod


def _load(name, path, parent=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[1], mod)
    return mod


_root = _register_pkg("comfy_agent", BASE)
_load("comfy_agent.config", BASE + "/config.py", _root)
_load("comfy_agent.llm_client", BASE + "/llm_client.py", _root)
_knowledge = _register_pkg("comfy_agent.knowledge", BASE + "/knowledge")
_load("comfy_agent.knowledge.type_validator", BASE + "/knowledge/type_validator.py", _knowledge)
_generation = _register_pkg("comfy_agent.generation", BASE + "/generation")
stage_m = _load(
    "comfy_agent.generation.stage_m_modify", BASE + "/generation/stage_m_modify.py", _generation
)


SCHEMA = {
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "steps": ["INT", {"min": 1, "max": 150, "default": 20}],
                "cfg": ["FLOAT", {"min": 0.0, "max": 100.0}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "SaveImage": {
        "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}},
        "output": [],
        "output_name": [],
    },
    "UpscaleModelLoader": {
        "input": {"required": {"model_name": [["4x-ESRGAN.pth"]]}},
        "output": ["UPSCALE_MODEL"],
        "output_name": ["UPSCALE_MODEL"],
    },
}

CANVAS = {"nodes": [{"id": 1, "type": "KSampler"}, {"id": 2, "type": "SaveImage"}]}


def test_canvas_nodes_reads_ids_and_types():
    nodes = stage_m.canvas_nodes(CANVAS)
    assert nodes == {1: "KSampler", 2: "SaveImage"}
    print("ok: canvas nodes are read as {id: class_type}")


def test_canvas_nodes_skips_malformed_entries():
    graph = {
        "nodes": [
            {"id": 1, "type": "KSampler"},
            {"id": "not-an-int", "type": "SaveImage"},
            {"id": 3},  # no type
            "not a dict",
            # A boolean id would pass isinstance(x, int) and corrupt lookups.
            {"id": True, "type": "SaveImage"},
        ]
    }
    assert stage_m.canvas_nodes(graph) == {1: "KSampler"}
    print("ok: malformed canvas entries are skipped, including boolean ids")


def test_valid_diff_passes_through():
    diff = stage_m.validate_diff(
        {
            "add_nodes": ["UpscaleModelLoader"],
            "remove_node_ids": [2],
            "set_widgets": [{"node_id": 1, "widget": "steps", "value": 30}],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.add_nodes == ["UpscaleModelLoader"]
    assert diff.remove_node_ids == [2]
    assert diff.set_widgets == [{"node_id": 1, "widget": "steps", "value": 30}]
    assert diff.rejected == []
    assert not diff.is_empty
    print("ok: a fully valid diff passes through unchanged")


def test_unknown_node_type_is_rejected():
    diff = stage_m.validate_diff(
        {"add_nodes": ["HallucinatedNode"], "remove_node_ids": [], "set_widgets": []},
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.add_nodes == []
    assert any("HallucinatedNode" in r for r in diff.rejected)
    print("ok: adding an uninstalled node type is rejected")


def test_removing_a_nonexistent_node_is_rejected():
    """The most dangerous failure mode: deleting by an id the model invented."""
    diff = stage_m.validate_diff(
        {"add_nodes": [], "remove_node_ids": [1, 99], "set_widgets": []},
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.remove_node_ids == [1], "only the real id survives"
    assert any("99" in r for r in diff.rejected)
    print("ok: removing an id that is not on the canvas is rejected")


def test_widget_change_on_unknown_widget_is_rejected():
    diff = stage_m.validate_diff(
        {
            "add_nodes": [],
            "remove_node_ids": [],
            "set_widgets": [{"node_id": 1, "widget": "nonexistent", "value": 1}],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.set_widgets == []
    assert any("nonexistent" in r for r in diff.rejected)
    print("ok: setting a widget the node does not have is rejected")


def test_widget_change_on_a_removed_node_is_rejected():
    diff = stage_m.validate_diff(
        {
            "add_nodes": [],
            "remove_node_ids": [1],
            "set_widgets": [{"node_id": 1, "widget": "steps", "value": 30}],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.remove_node_ids == [1]
    assert diff.set_widgets == [], "a node being deleted cannot also be edited"
    print("ok: editing a node the same diff removes is rejected")


def test_duplicate_widget_targets_keep_the_first():
    diff = stage_m.validate_diff(
        {
            "add_nodes": [],
            "remove_node_ids": [],
            "set_widgets": [
                {"node_id": 1, "widget": "steps", "value": 30},
                {"node_id": 1, "widget": "steps", "value": 45},
            ],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.set_widgets == [{"node_id": 1, "widget": "steps", "value": 30}]
    print("ok: a duplicated widget target resolves deterministically")


def test_malformed_entries_do_not_crash_validation():
    diff = stage_m.validate_diff(
        {
            "add_nodes": [None, 42],
            "remove_node_ids": ["2", None, True],
            "set_widgets": ["not a dict", {"node_id": 1}, {"widget": "steps"}],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    assert diff.is_empty
    assert len(diff.rejected) >= 5
    print("ok: junk in every diff field is rejected without crashing")


def test_missing_diff_keys_are_treated_as_empty():
    diff = stage_m.validate_diff({}, stage_m.canvas_nodes(CANVAS), SCHEMA)
    assert diff.is_empty
    assert diff.rejected == []
    print("ok: a diff with no keys at all is empty rather than an error")


def test_object_wrapped_values_are_rejected():
    """Measured against a live Ollama: every model size returned the value
    wrapped in a descriptor object ({"i": 35}) rather than the value itself.
    A widget holds a scalar, so a container is always wrong here — and writing
    one produces a node that looks configured and cannot run."""
    for bad in ({"i": 35}, {"default_value": 20, "type": "int"}, [35]):
        diff = stage_m.validate_diff(
            {
                "add_nodes": [],
                "remove_node_ids": [],
                "set_widgets": [{"node_id": 1, "widget": "steps", "value": bad}],
            },
            stage_m.canvas_nodes(CANVAS),
            SCHEMA,
        )
        assert diff.set_widgets == [], f"{bad!r} must not reach the canvas"
        assert any("scalar" in r for r in diff.rejected)
    print("ok: object- and list-wrapped widget values are rejected")


def test_widget_values_are_coerced_and_clamped():
    """Stage M applies the same coercion as Stage B, so a MODIFY cannot write
    a value that a BUILD would have rejected."""
    diff = stage_m.validate_diff(
        {
            "add_nodes": [],
            "remove_node_ids": [],
            "set_widgets": [
                {"node_id": 1, "widget": "steps", "value": "35"},   # string -> int
                {"node_id": 1, "widget": "cfg", "value": 999.0},    # above max -> clamped
            ],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    by_widget = {c["widget"]: c["value"] for c in diff.set_widgets}
    assert by_widget["steps"] == 35 and isinstance(by_widget["steps"], int)
    assert by_widget["cfg"] == 100.0, "out-of-range values are clamped, not passed through"
    print("ok: widget values are type-coerced and range-clamped")


def test_destructive_diffs_are_flagged_for_escalation():
    """The 0.8B proposed deleting two of three canvas nodes to change a widget
    value. Every id was real, so only the shape of the edit reveals it."""
    nodes = stage_m.canvas_nodes(CANVAS)  # 2 nodes
    wipe = stage_m.ModifyDiff([], [1, 2], [], [])
    assert stage_m._is_implausibly_destructive(wipe, nodes)

    one_of_two = stage_m.ModifyDiff([], [1], [], [])
    assert not stage_m._is_implausibly_destructive(one_of_two, nodes), (
        "removing a single node is a legitimate edit and must not be flagged"
    )
    assert not stage_m._is_implausibly_destructive(stage_m.ModifyDiff([], [], [], []), nodes)
    print("ok: wholesale deletion is flagged, a single removal is not")


def test_modify_prefers_the_escalation_model():
    """MODIFY routes to the fallback model first, unlike every other stage.

    Measured against a live install: asked to set steps to 35, qwen3.5:4b and
    :9b returned exactly the right edit while the 0.8B edited an unrelated
    node with an unrelated value. A wrong MODIFY corrupts existing work, so
    the extra latency is worth it here even though BUILD does not pay it.
    """
    called = {}

    async def fake_resolve(force=False):
        return ("small-primary", "big-fallback")

    async def fake_generate_json(prompt, schema, *, model=None, system=None):
        called["model"] = model
        return {"add_nodes": [], "remove_node_ids": [], "set_widgets": []}

    async def fail_if_used(*a, **kw):
        raise AssertionError("generate_json_with_fallback must not be the first choice")

    llm = sys.modules["comfy_agent.generation.stage_m_modify"].llm_client
    orig = (llm.resolve_models, llm.generate_json, llm.generate_json_with_fallback)
    llm.resolve_models = fake_resolve
    llm.generate_json = fake_generate_json
    llm.generate_json_with_fallback = fail_if_used
    try:
        asyncio.run(
            stage_m.generate_diff("set steps to 35", {1: "KSampler"}, SCHEMA, "KSampler: in=[] out=[]")
        )
    finally:
        llm.resolve_models, llm.generate_json, llm.generate_json_with_fallback = orig

    assert called["model"] == "big-fallback", (
        f"MODIFY should try the escalation model first, used {called.get('model')!r}"
    )
    print("ok: MODIFY prefers the escalation model over the primary")


def test_summary_describes_the_edit():
    diff = stage_m.validate_diff(
        {
            "add_nodes": ["UpscaleModelLoader"],
            "remove_node_ids": [2],
            "set_widgets": [{"node_id": 1, "widget": "steps", "value": 30}],
        },
        stage_m.canvas_nodes(CANVAS),
        SCHEMA,
    )
    summary = diff.summary()
    assert "UpscaleModelLoader" in summary
    assert "1 widget change" in summary
    print(f"ok: summary reads as {summary!r}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall modify tests passed")
