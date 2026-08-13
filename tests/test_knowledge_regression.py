"""Regression tests for the knowledge layer and Stage C link validation.

Covers the defects that previously had zero test coverage:
  - malformed/unhashable LLM link fields crashing the build (TypeError -> 500)
  - wildcard ("*") inputs being rejected as type mismatches
  - truncated output_name making real outputs unaddressable
  - the schema budget evicting essential nodes from the candidate pool

Run standalone (no pytest required):

    python tests/test_knowledge_regression.py
"""

import asyncio
import importlib.util
import pathlib
import sys
import types

BASE = str(pathlib.Path(__file__).resolve().parent.parent)


def _pkg(name, path):
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


_root = _pkg("comfy_agent", BASE)
config_mod = _load("comfy_agent.config", BASE + "/config.py", _root)
_load("comfy_agent.llm_client", BASE + "/llm_client.py", _root)
_knowledge = _pkg("comfy_agent.knowledge", BASE + "/knowledge")
type_validator = _load(
    "comfy_agent.knowledge.type_validator", BASE + "/knowledge/type_validator.py", _knowledge
)
schema_compactor = _load(
    "comfy_agent.knowledge.schema_compactor", BASE + "/knowledge/schema_compactor.py", _knowledge
)
_generation = _pkg("comfy_agent.generation", BASE + "/generation")
stage_a = _load(
    "comfy_agent.generation.stage_a_identify", BASE + "/generation/stage_a_identify.py", _generation
)
stage_c = _load(
    "comfy_agent.generation.stage_c_connect", BASE + "/generation/stage_c_connect.py", _generation
)
stage_d = _load(
    "comfy_agent.generation.stage_d_validate", BASE + "/generation/stage_d_validate.py", _generation
)

SIMPLE_SCHEMA = {
    "X": {"output": ["MODEL"], "output_name": ["MODEL"], "input": {"required": {}}},
    "Y": {"output": [], "input": {"required": {"model": ["MODEL"]}}},
}
SIMPLE_INSTANCES = {"x1": "X", "y1": "Y"}


def test_malformed_links_never_raise():
    """Non-string ids/slots must be rejected, not crash.

    These previously raised TypeError('unhashable type') out of
    node_instances.get()/seen_targets.add(), escaping StageCError and
    surfacing as an HTTP 500 that killed the whole build.
    """
    malformed_batches = [
        [{"from_node": "x1", "from_output": "MODEL", "to_node": "y1", "to_input": ["model"]}],
        [{"from_node": ["x1"], "from_output": "MODEL", "to_node": "y1", "to_input": "model"}],
        [{"from_node": "x1", "from_output": {"a": 1}, "to_node": "y1", "to_input": "model"}],
        [{"from_node": {"k": "v"}, "to_node": {"k": "v"}, "from_output": [], "to_input": []}],
        [{"from_node": "x1", "to_node": "y1"}],  # missing slot keys entirely
        ["garbage", None, 42, {"from_node": "nope"}],
    ]
    for batch in malformed_batches:
        accepted, rejected = stage_c._validate_links(batch, SIMPLE_INSTANCES, SIMPLE_SCHEMA)
        assert accepted == [], f"malformed batch was accepted: {batch} -> {accepted}"
        assert len(rejected) == len(batch), f"not every entry rejected: {batch} -> {rejected}"


def test_valid_link_still_accepted():
    """The hardening must not reject well-formed links."""
    good = [{"from_node": "x1", "from_output": "MODEL", "to_node": "y1", "to_input": "model"}]
    accepted, rejected = stage_c._validate_links(good, SIMPLE_INSTANCES, SIMPLE_SCHEMA)
    assert len(accepted) == 1, f"valid link rejected: {rejected}"


def test_duplicate_target_input_rejected():
    dup = [
        {"from_node": "x1", "from_output": "MODEL", "to_node": "y1", "to_input": "model"},
        {"from_node": "x1", "from_output": "MODEL", "to_node": "y1", "to_input": "model"},
    ]
    accepted, rejected = stage_c._validate_links(dup, SIMPLE_INSTANCES, SIMPLE_SCHEMA)
    assert len(accepted) == 1 and len(rejected) == 1


def test_wildcard_input_accepts_any_type():
    """"*" inputs (Reroute, most custom node packs) connect to anything."""
    schema = {
        "A": {"output": ["IMAGE"], "output_name": ["IMAGE"], "input": {"required": {}}},
        "B": {"output": [], "input": {"required": {"any": ["*"]}}},
    }
    type_validator.validate_link("A", "IMAGE", "B", "any", schema)  # must not raise


def test_real_type_mismatch_still_rejected():
    """Wildcard support must not weaken genuine type checking."""
    schema = {
        "A": {"output": ["IMAGE"], "output_name": ["IMAGE"], "input": {"required": {}}},
        "B": {"output": [], "input": {"required": {"m": ["MODEL"]}}},
    }
    try:
        type_validator.validate_link("A", "IMAGE", "B", "m", schema)
    except type_validator.LinkValidationError:
        return
    raise AssertionError("IMAGE -> MODEL mismatch was accepted")


def test_truncated_output_name_keeps_all_outputs_addressable():
    """output_name shorter than output must not hide the trailing outputs."""
    node = {"output": ["MODEL", "CLIP", "VAE"], "output_name": ["MODEL"]}
    names = type_validator.node_output_names(node)
    assert names == ["MODEL", "CLIP", "VAE"], names

    schema = {
        "A": dict(node, input={"required": {}}),
        "B": {"output": [], "input": {"required": {"clip": ["CLIP"]}}},
    }
    type_validator.validate_link("A", "CLIP", "B", "clip", schema)  # must not raise


def test_blank_entries_in_output_name_fall_back_to_type():
    node = {"output": ["MODEL", "CLIP"], "output_name": ["", "CLIP"]}
    assert type_validator.node_output_names(node) == ["MODEL", "CLIP"]


def test_dict_form_outputs_supported():
    node = {"output": [{"name": "LATENT", "type": "LATENT"}], "output_name": []}
    assert type_validator.node_output_names(node) == ["LATENT"]


def test_widget_vs_link_input_partition():
    """Stage B and Stage C must partition inputs as exact complements."""
    assert type_validator.is_widget_input(["INT", {"default": 0}]) is True
    assert type_validator.is_widget_input(["STRING", {}]) is True
    assert type_validator.is_widget_input([["euler", "dpmpp_2m"], {}]) is True  # COMBO
    assert type_validator.is_widget_input({"type": "FLOAT"}) is True
    assert type_validator.is_widget_input(["MODEL"]) is False
    assert type_validator.is_widget_input(["LATENT"]) is False


def test_essential_nodes_survive_the_schema_budget():
    """A late-sorting SaveImage/VAEDecode must never be evicted by the budget.

    Eviction made a correct workflow structurally impossible for Stage A while
    presenting only as a generic downstream "generation failed".
    """
    registry = {
        f"KSamplerVariant{i}": {
            "input": {"required": {"model": ["MODEL"], "positive": ["CONDITIONING"]}},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        }
        for i in range(500)
    }
    registry["SaveImage"] = {
        "input": {"required": {"images": ["IMAGE"]}},
        "output": [],
        "output_name": [],
    }
    registry["VAEDecode"] = {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    }

    compacted = schema_compactor.compact_schema(registry, ["sampling", "image"])
    assert "SaveImage" in compacted, "SaveImage evicted by the schema budget"
    assert "VAEDecode" in compacted, "VAEDecode evicted by the schema budget"


def test_compact_schema_skips_malformed_entries():
    registry = {
        "Good": {"input": {"required": {}}, "output": []},
        "BadString": "not-a-dict",
        "BadNone": None,
    }
    compacted = schema_compactor.compact_schema(registry, ["custom"])
    assert "Good" in compacted
    assert "BadString" not in compacted and "BadNone" not in compacted


def test_orphan_is_a_warning_not_a_fatal_error():
    """A dangling node must be surfaced, not silently dropped.

    Orphans used to be appended to `errors` while leaving valid=True, so the
    pipeline's `error if not valid` branch discarded the text entirely and the
    route still reported execution_ready.
    """
    result = stage_d.validate_spec(
        {"a": "X", "b": "Y", "c": "Z"}, [{"from_node": "a", "to_node": "b"}]
    )
    assert result.valid is True, "an orphan must not block the build"
    assert result.errors == [], result.errors
    assert any("orphan" in w for w in result.warnings), result.warnings


def test_fully_disconnected_graph_is_fatal():
    result = stage_d.validate_spec({"a": "X", "b": "Y"}, [])
    assert result.valid is False
    assert any("disconnected" in e for e in result.errors), result.errors


def test_link_count_ignores_links_naming_unknown_nodes():
    """Bogus links must not inflate link_count or mask an orphan."""
    result = stage_d.validate_spec({"a": "X"}, [{"bogus": 1}, {"from_node": "a", "to_node": "zzz"}])
    assert result.link_count == 0, result.link_count


def test_empty_spec_is_fatal():
    result = stage_d.validate_spec({}, [])
    assert result.valid is False
    assert any("no nodes" in e for e in result.errors), result.errors


def test_stage_a_truncation_keeps_essential_nodes():
    """Truncating in raw model order could clip away the output stage.

    Drives the real identify_nodes() with a stubbed model that lists the
    essential nodes last, then asserts they survive a max_nodes clip.
    """
    proposed = ["ExtraA", "ExtraB", "ExtraC", "ExtraD", "KSampler", "SaveImage", "VAEDecode"]

    async def _stub_with_fallback(prompt, schema, **kwargs):
        return {"node_types": proposed}, config_mod.PRIMARY_MODEL

    async def _stub_generate(prompt, schema, **kwargs):
        return {"node_types": proposed}

    original_fallback = stage_a.llm_client.generate_json_with_fallback
    original_generate = stage_a.llm_client.generate_json
    stage_a.llm_client.generate_json_with_fallback = _stub_with_fallback
    stage_a.llm_client.generate_json = _stub_generate
    try:
        got = asyncio.run(
            stage_a.identify_nodes("task", "block", 5, known_class_types=set(proposed))
        )
    finally:
        stage_a.llm_client.generate_json_with_fallback = original_fallback
        stage_a.llm_client.generate_json = original_generate

    assert len(got) == 5, got
    assert "SaveImage" in got, f"output stage clipped away: {got}"
    assert "VAEDecode" in got, f"decoder clipped away: {got}"
    assert "KSampler" in got, f"sampler clipped away: {got}"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"KNOWLEDGE REGRESSION OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
