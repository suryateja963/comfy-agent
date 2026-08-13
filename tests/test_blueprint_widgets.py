"""Blueprint widget-mapping regression tests.

Every blueprint node carries a `widgets_named` map so injection is by NAME,
not by position. Pure positional injection silently misassigns values whenever
a node's widget layout differs from what the blueprint author assumed — which
was already happening: KSampler blueprints supplied 7 values for 6 widgets
(a legacy `control_after_generate` token at index 1), so `steps` received
'fixed', `cfg` received 20, `sampler_name` received 7.0, and `denoise` fell
off the end entirely. That produces a workflow that runs and generates wrong
images, which is worse than a crash because it is invisible.

These tests are schema-shape checks and run offline. The values were generated
and validated against a live ComfyUI /object_info (1717 nodes); to re-validate
against a running instance, see the `--live` flag below.

    python tests/test_blueprint_widgets.py
    python tests/test_blueprint_widgets.py --live   # requires ComfyUI running
"""

import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent
BLUEPRINT_DIR = BASE / "blueprint_defs"

# Widget order per node type, captured from a live /object_info. Used to prove
# that positional injection would misassign, without needing ComfyUI running.
KNOWN_WIDGET_ORDER = {
    "KSampler": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "LoadImage": ["image"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CLIPTextEncode": ["text"],
    "SaveImage": ["filename_prefix"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "VAELoader": ["vae_name"],
    "LatentUpscale": ["upscale_method", "width", "height", "crop"],
    "ControlNetLoader": ["control_net_name"],
    "ControlNetApply": ["strength"],
    "SaveAnimatedWEBP": ["filename_prefix", "fps", "lossless", "quality", "method"],
}


def _blueprints():
    for path in sorted(BLUEPRINT_DIR.glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def test_every_blueprint_node_with_values_has_named_widgets():
    """A node with widgets_values must also carry widgets_named."""
    missing = []
    for path, data in _blueprints():
        for node in data["nodes"]:
            if node.get("widgets_values") and not node.get("widgets_named"):
                missing.append(f"{path.name}:{node['type']}")
    assert not missing, f"nodes missing widgets_named: {missing}"


def test_named_widgets_match_known_widget_order():
    """Every named widget must be a real widget of that node type."""
    bad = []
    for path, data in _blueprints():
        for node in data["nodes"]:
            names = KNOWN_WIDGET_ORDER.get(node["type"])
            if names is None:
                continue  # type not in the offline snapshot; covered by --live
            for key in (node.get("widgets_named") or {}):
                if key not in names:
                    bad.append(f"{path.name}:{node['type']}.{key} not in {names}")
    assert not bad, "unknown widget names: " + "; ".join(bad)


def test_positional_injection_would_misassign_ksampler():
    """Guards the reason widgets_named exists.

    If this ever stops holding, the blueprints were rewritten to match the
    real widget order and the named mapping could in principle be dropped —
    but until then, positional injection is provably wrong.
    """
    order = KNOWN_WIDGET_ORDER["KSampler"]
    for path, data in _blueprints():
        for node in data["nodes"]:
            if node["type"] != "KSampler":
                continue
            values = node.get("widgets_values") or []
            named = node.get("widgets_named") or {}
            assert len(values) > len(order), (
                f"{path.name}: KSampler now has {len(values)} values for "
                f"{len(order)} widgets — re-evaluate this test"
            )
            positional = dict(zip(order, values))
            assert positional != named, f"{path.name}: expected positional/named divergence"
            # The specific corruption: steps receives the legacy token.
            assert positional["steps"] == "fixed", positional
            assert named["steps"] == values[2], named


def test_named_values_have_sane_types():
    """Catch a mapping that assigns a string where a number belongs."""
    numeric = {"seed", "steps", "cfg", "denoise", "width", "height", "batch_size",
               "strength", "strength_model", "strength_clip", "fps", "quality",
               "guide_size", "max_size", "feather"}
    bad = []
    for path, data in _blueprints():
        for node in data["nodes"]:
            for key, value in (node.get("widgets_named") or {}).items():
                if key in numeric and not isinstance(value, (int, float)):
                    bad.append(f"{path.name}:{node['type']}.{key}={value!r}")
                if key in ("sampler_name", "scheduler", "text") and not isinstance(value, str):
                    bad.append(f"{path.name}:{node['type']}.{key}={value!r}")
    assert not bad, "type-implausible widget values: " + "; ".join(bad)


def main() -> None:
    live = "--live" in sys.argv
    tests = [
        test_every_blueprint_node_with_values_has_named_widgets,
        test_named_widgets_match_known_widget_order,
        test_positional_injection_would_misassign_ksampler,
        test_named_values_have_sane_types,
    ]
    for fn in tests:
        fn()
        print(f"  PASS {fn.__name__}")
    if live:
        _run_live()
    print(f"BLUEPRINT WIDGET TESTS OK ({len(tests)} tests)")


def _run_live() -> None:
    """Inline live validation against /object_info (only with --live)."""
    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=20) as resp:
        object_info = json.load(resp)

    def spec_of(node_type, name):
        inputs = object_info[node_type].get("input", {})
        for group in ("required", "optional"):
            spec = (inputs.get(group) or {}).get(name)
            if spec is not None:
                return spec
        return None

    problems = []
    checked = 0
    for path, data in _blueprints():
        for node in data["nodes"]:
            node_type = node["type"]
            if node_type not in object_info:
                problems.append(f"{path.name}: unknown node type {node_type}")
                continue
            for name, value in (node.get("widgets_named") or {}).items():
                checked += 1
                spec = spec_of(node_type, name)
                if spec is None:
                    problems.append(f"{path.name}:{node_type}.{name} is not a widget")
                    continue
                type_val = spec[0] if isinstance(spec, list) and spec else spec
                if isinstance(type_val, dict):
                    type_val = type_val.get("type")
                meta = spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                if type_val == "INT" and (not isinstance(value, int) or isinstance(value, bool)):
                    problems.append(f"{path.name}:{node_type}.{name} expected INT, got {value!r}")
                if type_val == "BOOLEAN" and not isinstance(value, bool):
                    problems.append(f"{path.name}:{node_type}.{name} expected BOOLEAN, got {value!r}")
                if type_val in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
                    lo, hi = meta.get("min"), meta.get("max")
                    if lo is not None and value < lo:
                        problems.append(f"{path.name}:{node_type}.{name}={value} below min {lo}")
                    if hi is not None and value > hi:
                        problems.append(f"{path.name}:{node_type}.{name}={value} above max {hi}")

    assert not problems, "live validation failed:\n  " + "\n  ".join(problems)
    print(f"  PASS test_live_object_info ({checked} assignments validated)")


if __name__ == "__main__":
    main()
