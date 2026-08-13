"""End-to-end free-form pipeline test with the LLM and ComfyUI stubbed.

Runs `pipeline.generate_workflow` (Pass 1 + Stages A-D) against a fake
/object_info registry and canned model responses, so the full async wiring
is exercised without Ollama or ComfyUI.

    pytest tests/                              # collected as test_* functions
    python tests/test_pipeline_integration.py  # standalone, no pytest needed
"""

import asyncio
import contextlib
import importlib.util
import pathlib
import re
import sys
import types

# Derived from this file's location so the suite runs on any machine / in CI.
BASE = str(pathlib.Path(__file__).resolve().parent.parent)
PKG = BASE


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


# Build the package tree by hand, in dependency order, without running
# comfy_agent/__init__.py (which pulls in ComfyUI-only comfy_api / server).
_root = _register_pkg("comfy_agent", PKG)
config = _load("comfy_agent.config", PKG + "/config.py", _root)
llm_client = _load("comfy_agent.llm_client", PKG + "/llm_client.py", _root)
knowledge = _register_pkg("comfy_agent.knowledge", PKG + "/knowledge")
_load("comfy_agent.knowledge.type_validator", PKG + "/knowledge/type_validator.py", knowledge)
schema_compactor = _load(
    "comfy_agent.knowledge.schema_compactor", PKG + "/knowledge/schema_compactor.py", knowledge
)
generation = _register_pkg("comfy_agent.generation", PKG + "/generation")
classify = _load("comfy_agent.generation.classify", PKG + "/generation/classify.py", generation)
_load("comfy_agent.generation.stage_a_identify", PKG + "/generation/stage_a_identify.py", generation)
_load("comfy_agent.generation.stage_b_configure", PKG + "/generation/stage_b_configure.py", generation)
_load("comfy_agent.generation.stage_c_connect", PKG + "/generation/stage_c_connect.py", generation)
_load("comfy_agent.generation.stage_d_validate", PKG + "/generation/stage_d_validate.py", generation)
pipeline = _load("comfy_agent.generation.pipeline", PKG + "/generation/pipeline.py", generation)

FAKE_REGISTRY = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["a.safetensors"], {}]}},
        "output": ["MODEL", "CLIP", "VAE"],
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"default": ""}], "clip": ["CLIP"]}},
        "output": ["CONDITIONING"],
    },
    "EmptyLatentImage": {
        "input": {"required": {"width": ["INT", {"default": 512}], "height": ["INT", {}], "batch_size": ["INT", {}]}},
        "output": ["LATENT"],
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
                "seed": ["INT", {"default": 0}],
                "steps": ["INT", {"default": 20, "min": 1, "max": 150}],
                "cfg": ["FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}],
                "sampler_name": [["euler", "dpmpp_2m"], {}],
                "scheduler": [["normal", "karras"], {}],
                "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
            }
        },
        "output": ["LATENT"],
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
    },
    "SaveImage": {
        "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
        "output": [],
    },
}

STAGE_A_NODE_TYPES = [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
]

POSITIVE_NEGATIVE_LINKS = [
    ("CheckpointLoaderSimple", "MODEL", "KSampler", "model"),
    ("CheckpointLoaderSimple", "CLIP", "CLIPTextEncode", "clip"),
    ("CheckpointLoaderSimple", "CLIP", "CLIPTextEncode", "clip"),
    ("EmptyLatentImage", "LATENT", "KSampler", "latent_image"),
    ("CLIPTextEncode", "CONDITIONING", "KSampler", "positive"),
    ("CLIPTextEncode", "CONDITIONING", "KSampler", "negative"),
    ("KSampler", "LATENT", "VAEDecode", "samples"),
    ("CheckpointLoaderSimple", "VAE", "VAEDecode", "vae"),
    ("VAEDecode", "IMAGE", "SaveImage", "images"),
]


def _route_by_schema(prompt: str, schema: dict):
    props = set((schema.get("properties") or {}).keys())
    if "categories" in props:
        return {"categories": []}
    if "node_types" in props:
        return {"node_types": STAGE_A_NODE_TYPES}
    if "widgets" in props:
        if "Node: CLIPTextEncode" in prompt:
            return {"widgets": {"text": "a cat in a wizard hat"}}
        if "Node: KSampler" in prompt:
            return {"widgets": {"steps": "7"}}
        return {"widgets": {}}
    if "links" in props:
        ids = {ctype: [] for ctype in set(STAGE_A_NODE_TYPES)}
        for instance_id, ctype in re.findall(r"id=(\S+) type=(\S+)", prompt):
            ids.setdefault(ctype, []).append(instance_id)
        counter = {ctype: 0 for ctype in set(STAGE_A_NODE_TYPES)}
        links = []
        for from_type, out, to_type, inp in POSITIVE_NEGATIVE_LINKS:
            from_id = ids[from_type][counter[from_type] % len(ids[from_type])]
            counter[from_type] += 1
            to_id = ids[to_type][counter[to_type] % len(ids[to_type])]
            counter[to_type] += 1
            links.append({"from_node": from_id, "from_output": out, "to_node": to_id, "to_input": inp})
        return {"links": links}
    raise AssertionError(f"stub hit an unknown schema: {props}")


async def _fake_with_fallback(prompt, schema, **kw):
    return _route_by_schema(prompt, schema), config.PRIMARY_MODEL


async def _fake_generate(prompt, schema, **kw):
    return _route_by_schema(prompt, schema)


async def _fake_fetch_object_info():
    return FAKE_REGISTRY


# Garbage Stage C output. Includes UNHASHABLE ids/slot names: the earlier
# version of this list was all-hashable, so it never exercised the dict/set
# lookups and a TypeError('unhashable type') escaped Stage C as an HTTP 500.
JUNK_LINKS = [
    "garbage",
    None,
    42,
    {"from_node": "nope"},
    {"from_node": ["x"], "from_output": "MODEL", "to_node": "y", "to_input": "model"},
    {"from_node": "x", "from_output": "MODEL", "to_node": "y", "to_input": ["model"]},
    {"from_node": "x", "from_output": {"a": 1}, "to_node": "y", "to_input": "model"},
]


async def _junk_router(prompt, schema, **kw):
    """Second run: Stage C gets garbage (non-dict / malformed / unhashable)."""
    props = set((schema.get("properties") or {}).keys())
    if "links" in props:
        return {"links": JUNK_LINKS}, config.PRIMARY_MODEL
    return _route_by_schema(prompt, schema), config.PRIMARY_MODEL


async def _junk_generate(prompt, schema, **kw):
    props = set((schema.get("properties") or {}).keys())
    if "links" in props:
        return {"links": JUNK_LINKS}
    return _route_by_schema(prompt, schema)


async def _fake_resolve_models():
    """Stage A and Stage C consult this to decide whether to escalate.

    Stubbed alongside the generators because it otherwise reaches /api/tags —
    the whole point of this suite is that it runs with nothing installed.
    Returns a distinct fallback so escalation paths stay reachable.
    """
    return config.PRIMARY_MODEL, config.FALLBACK_MODEL


@contextlib.contextmanager
def _stubbed(with_fallback, generate, resolve=_fake_resolve_models):
    """Install LLM/registry stubs and always restore them.

    Restoration matters: these patch module globals, so leaking them made the
    tests order-dependent and let a later test silently inherit an earlier
    test's model behaviour.
    """
    saved = (
        llm_client.generate_json_with_fallback,
        llm_client.generate_json,
        llm_client.resolve_models,
        schema_compactor.fetch_object_info,
    )
    llm_client.generate_json_with_fallback = with_fallback
    llm_client.generate_json = generate
    llm_client.resolve_models = resolve
    schema_compactor.fetch_object_info = _fake_fetch_object_info
    try:
        yield
    finally:
        (
            llm_client.generate_json_with_fallback,
            llm_client.generate_json,
            llm_client.resolve_models,
            schema_compactor.fetch_object_info,
        ) = saved


def _run_happy_path():
    with _stubbed(_fake_with_fallback, _fake_generate):
        return asyncio.run(pipeline.generate_workflow("a cat in a wizard hat", max_nodes=12))


def test_happy_path_builds_full_workflow():
    result = _run_happy_path()
    assert result.ok, f"pipeline not ok: {result.error}"
    assert len(result.node_instances) == 7, f"expected 7 nodes, got {len(result.node_instances)}"
    assert len(result.links) == 9, f"expected 9 links, got {len(result.links)}"
    assert result.validation is not None and result.validation.valid


def test_happy_path_has_no_warnings():
    """A fully-connected graph must report no orphan warnings."""
    result = _run_happy_path()
    assert result.warnings == [], result.warnings


def test_prompt_text_reaches_both_clip_encoders():
    result = _run_happy_path()
    encode_widgets = [w for iid, w in result.widgets.items() if iid.startswith("CLIPTextEncode_")]
    assert len(encode_widgets) == 2, encode_widgets
    for widgets in encode_widgets:
        assert widgets.get("text") == "a cat in a wizard hat", widgets


def test_widget_values_are_type_coerced():
    """Stage B must coerce the model's string "7" to a real int."""
    result = _run_happy_path()
    sampler_widgets = next(w for iid, w in result.widgets.items() if iid.startswith("KSampler_"))
    assert sampler_widgets.get("steps") == 7, sampler_widgets
    assert isinstance(sampler_widgets["steps"], int), type(sampler_widgets["steps"])


def test_junk_links_fail_gracefully_without_crashing():
    """Stage C garbage — including unhashable ids/slots — must be rejected,
    leaving a reported failure rather than an unhandled TypeError/HTTP 500."""
    with _stubbed(_junk_router, _junk_generate):
        result = asyncio.run(pipeline.generate_workflow("a cat in a wizard hat", max_nodes=12))
    assert not result.ok, "expected graceful failure on junk links"
    assert "disconnected graph" in (result.error or ""), result.error
    assert len(result.links) == 0, f"expected 0 accepted links, got {len(result.links)}"


# --- Gap 8: partially connected graphs -------------------------------------
#
# The exact observed failure: every node has *some* link, so the old
# "orphans == all nodes" check passes and the old "len(accepted) >= N-1"
# heuristic is satisfied (8 links, 7 nodes) — yet VAEDecode -> SaveImage is
# missing, so SaveImage.images is unfilled and the graph cannot run.
MISSING_SAVE_LINK = [l for l in POSITIVE_NEGATIVE_LINKS if l[2] != "SaveImage"]


def _links_for(prompt: str, spec_links) -> dict:
    ids: dict = {}
    for instance_id, ctype in re.findall(r"id=(\S+) type=(\S+)", prompt):
        ids.setdefault(ctype, []).append(instance_id)
    counter = {ctype: 0 for ctype in ids}
    links = []
    for from_type, out, to_type, inp in spec_links:
        if from_type not in ids or to_type not in ids:
            continue
        from_id = ids[from_type][counter[from_type] % len(ids[from_type])]
        counter[from_type] += 1
        to_id = ids[to_type][counter[to_type] % len(ids[to_type])]
        counter[to_type] += 1
        links.append({"from_node": from_id, "from_output": out, "to_node": to_id, "to_input": inp})
    return {"links": links}


def _incomplete_router_factory(repair_links):
    """Primary model omits VAEDecode->SaveImage; the repair pass returns
    `repair_links` (possibly empty, to model a failed repair)."""
    state = {"repair_called": 0}

    async def with_fallback(prompt, schema, **kw):
        props = set((schema.get("properties") or {}).keys())
        if "links" in props:
            return _links_for(prompt, MISSING_SAVE_LINK), config.PRIMARY_MODEL
        return _route_by_schema(prompt, schema), config.PRIMARY_MODEL

    async def generate(prompt, schema, **kw):
        props = set((schema.get("properties") or {}).keys())
        if "links" in props:
            # The repair pass is the only caller that reaches generate_json
            # with a links schema, and it must be given only the gaps.
            state["repair_called"] += 1
            assert "Unconnected required inputs" in prompt, prompt[:200]
            assert "images" in prompt, "repair prompt must name the missing input"
            return _links_for(prompt, repair_links)
        return _route_by_schema(prompt, schema)

    return with_fallback, generate, state


def test_unconnected_required_input_is_detected_not_counted():
    """A graph with N-1+ links but an unfilled required input must be caught.

    The old check was `len(accepted) < len(nodes) - 1`. Here 8 links across 7
    nodes satisfies it, and every node is linked so the orphan check passes
    too — yet SaveImage.images has nothing feeding it.
    """
    with_fallback, generate, state = _incomplete_router_factory(repair_links=[])
    with _stubbed(with_fallback, generate):
        result = asyncio.run(pipeline.generate_workflow("a cat in a wizard hat", max_nodes=12))

    assert state["repair_called"] == 1, "the repair pass should have been attempted once"
    assert len(result.links) >= len(result.node_instances) - 1, (
        "precondition: the old link-count heuristic must be satisfied, "
        f"got {len(result.links)} links for {len(result.node_instances)} nodes"
    )
    missing_inputs = {name for _, name in result.incomplete}
    assert "images" in missing_inputs, result.incomplete
    assert not result.execution_ready, "a graph missing a required input is not runnable"


def test_repair_pass_fills_the_gap_and_restores_readiness():
    """When the fallback model supplies the missing link, it is merged in and
    the graph becomes execution_ready — without re-rolling the other links."""
    with_fallback, generate, state = _incomplete_router_factory(
        repair_links=[("VAEDecode", "IMAGE", "SaveImage", "images")]
    )
    with _stubbed(with_fallback, generate):
        result = asyncio.run(pipeline.generate_workflow("a cat in a wizard hat", max_nodes=12))

    assert state["repair_called"] == 1
    assert result.incomplete == [], f"expected no gaps after repair, got {result.incomplete}"
    assert result.execution_ready, "graph should be runnable once the gap is filled"
    save_targets = [l for l in result.links if l["to_input"] == "images"]
    assert len(save_targets) == 1, f"exactly one link into images, got {save_targets}"


def test_repair_cannot_double_fill_an_already_connected_input():
    """The repair pass proposing a link into a filled input must be rejected —
    an input accepts exactly one link, so a duplicate silently drops one."""
    with_fallback, generate, _ = _incomplete_router_factory(
        # Proposes the genuinely missing link AND a duplicate of one already made.
        repair_links=[
            ("VAEDecode", "IMAGE", "SaveImage", "images"),
            ("EmptyLatentImage", "LATENT", "KSampler", "latent_image"),
        ]
    )
    with _stubbed(with_fallback, generate):
        result = asyncio.run(pipeline.generate_workflow("a cat in a wizard hat", max_nodes=12))

    latent_targets = [l for l in result.links if l["to_input"] == "latent_image"]
    assert len(latent_targets) == 1, f"duplicate target accepted: {latent_targets}"
    assert result.execution_ready


def test_widget_backed_required_inputs_are_not_reported_missing():
    """KSampler.steps and SaveImage.filename_prefix are required but live in
    widgets. Counting them as unconnected would make every graph incomplete."""
    result = _run_happy_path()
    missing_inputs = {name for _, name in result.incomplete}
    assert "steps" not in missing_inputs, result.incomplete
    assert "filename_prefix" not in missing_inputs, result.incomplete
    assert result.execution_ready, result.incomplete


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"INTEGRATION TEST OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
