"""Install-index cache tests: incremental re-embedding and vector alignment.

The index is stored keyed by node name so that installing or removing one node
pack re-embeds only what changed. The property that matters most here is
alignment: retrieval indexes `vectors` positionally against `names`, so a
vector reattached to the wrong node silently corrupts every answer rather than
failing loudly. Ollama and ComfyUI are both stubbed.

    pytest tests/                      # collected as test_* functions
    python tests/test_index_cache.py   # standalone, no pytest needed
"""

import asyncio
import importlib.util
import json
import pathlib
import sys
import tempfile
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


# Registered for their side effect on sys.modules only. The tests deliberately
# resolve these through _index_mod() / sys.modules at call time instead of
# binding them here — see _install_stubs() for why.
_root = _register_pkg("comfy_agent", BASE)
_load("comfy_agent.config", BASE + "/config.py", _root)
_load("comfy_agent.llm_client", BASE + "/llm_client.py", _root)
_knowledge = _register_pkg("comfy_agent.knowledge", BASE + "/knowledge")
_load("comfy_agent.knowledge.type_validator", BASE + "/knowledge/type_validator.py", _knowledge)
_load(
    "comfy_agent.knowledge.schema_compactor", BASE + "/knowledge/schema_compactor.py", _knowledge
)
_load("comfy_agent.install_index", BASE + "/install_index.py", _root)


def _node():
    return {"input": {"required": {}}, "output": [], "output_name": [], "category": "testing"}


THREE_NODES = {name: _node() for name in ("Alpha", "Beta", "Gamma")}

# Mutated per test; the stubbed fetch_object_info reads it.
_current_registry = dict(THREE_NODES)
_embedded: list[str] = []


async def _fake_object_info():
    return dict(_current_registry)


async def _fake_harvest_models():
    return {}


async def _fake_embed(texts, model=None):
    """Deterministic vector per text, so identity can be asserted exactly."""
    _embedded.extend(texts)
    return [[float(len(t)), float(sum(map(ord, t)) % 97)] for t in texts]


def _index_mod():
    """The live install_index module, not the one captured at import."""
    return sys.modules["comfy_agent.install_index"]


def _install_stubs():
    """(Re)bind the stubs this module depends on.

    Applied per test rather than once at import, and resolved through
    sys.modules rather than the module object captured at import time:
    install_index.build() does `from .knowledge import schema_compactor`
    *inside* the function, so it picks up whichever object sys.modules holds
    when it runs. Another test file re-registers that key, which would
    otherwise leave this file's stub on an orphaned module and let the real
    HTTP path run.
    """
    index_mod = _index_mod()
    sys.modules["comfy_agent.knowledge.schema_compactor"].fetch_object_info = _fake_object_info
    index_mod.harvest_models = _fake_harvest_models
    # Stub the llm_client object install_index itself holds, not the one in
    # sys.modules: install_index bound `llm_client` at its own import time, so
    # after another test file re-registers that key the two are different
    # objects and only this one is on the path build() actually takes.
    index_mod.llm_client.embed = _fake_embed


def _setup(registry):
    """Point the cache at a scratch file and reset all build state."""
    global _current_registry
    _install_stubs()
    _current_registry = dict(registry)
    _index_mod()._CACHE_PATH = pathlib.Path(tempfile.mkdtemp()) / "index.cache.json"
    _index_mod()._index = None
    _index_mod()._ready_event = None
    _embedded.clear()


def _rebuild(registry):
    """Rebuild against a changed registry, keeping the existing cache file."""
    global _current_registry
    _install_stubs()
    _current_registry = dict(registry)
    _index_mod()._index = None
    _index_mod()._ready_event = None
    _embedded.clear()
    return asyncio.run(_index_mod().build())


def _vector_for(index, name):
    return index["vectors"][index["names"].index(name)]


def test_unchanged_registry_reuses_cache_without_embedding():
    _setup(THREE_NODES)
    first = asyncio.run(_index_mod().build())
    assert len(_embedded) == 3
    baseline = {n: _vector_for(first, n) for n in THREE_NODES}

    second = _rebuild(THREE_NODES)
    assert _embedded == [], "an unchanged registry must not re-embed anything"
    assert _vector_for(second, "Beta") == baseline["Beta"]
    print("ok: unchanged registry reuses the cache with zero embed calls")


def test_adding_a_node_embeds_only_that_node():
    _setup(THREE_NODES)
    first = asyncio.run(_index_mod().build())
    baseline = {n: _vector_for(first, n) for n in THREE_NODES}

    grown = dict(THREE_NODES)
    grown["Delta"] = _node()
    index = _rebuild(grown)

    assert len(_embedded) == 1, f"expected one new embedding, got {len(_embedded)}"
    assert "Delta" in _embedded[0]
    for name in THREE_NODES:
        assert _vector_for(index, name) == baseline[name], f"{name} was re-embedded"
    print("ok: installing a node pack embeds only the new nodes")


def test_vectors_stay_aligned_across_a_mid_list_insertion():
    """The alignment guarantee: "Delta" sorts between Beta and Gamma, so every
    later node shifts position. A name-keyed cache must follow the name, not
    the old index."""
    _setup(THREE_NODES)
    first = asyncio.run(_index_mod().build())
    baseline = {n: _vector_for(first, n) for n in THREE_NODES}

    grown = dict(THREE_NODES)
    grown["Delta"] = _node()
    index = _rebuild(grown)

    assert index["names"] == ["Alpha", "Beta", "Delta", "Gamma"]
    for name in THREE_NODES:
        assert _vector_for(index, name) == baseline[name]
    # And the newcomer's vector matches its own description, not a neighbour's.
    delta_description = index["descriptions"][index["names"].index("Delta")]
    assert _vector_for(index, "Delta") == [
        float(len(delta_description)),
        float(sum(map(ord, delta_description)) % 97),
    ]
    print("ok: vectors follow their node across a mid-list insertion")


def test_removing_a_node_re_embeds_nothing():
    _setup(THREE_NODES)
    first = asyncio.run(_index_mod().build())
    baseline = {n: _vector_for(first, n) for n in THREE_NODES}

    index = _rebuild({"Alpha": _node(), "Gamma": _node()})
    assert _embedded == [], "removing a node must not re-embed the survivors"
    assert index["names"] == ["Alpha", "Gamma"]
    assert _vector_for(index, "Alpha") == baseline["Alpha"]
    assert _vector_for(index, "Gamma") == baseline["Gamma"]
    print("ok: removing a node re-embeds nothing and keeps alignment")


def test_legacy_cache_is_not_salvaged_into_a_changed_registry():
    """A v1 cache is a bare positional list, meaningful only against the exact
    name list that produced it. Salvaging it after a change would attach
    vectors to the wrong nodes."""
    _setup(THREE_NODES)
    _index_mod()._CACHE_PATH.write_text(
        json.dumps({"fingerprint": "stale", "vectors": [[9.0, 9.0], [8.0, 8.0]]}),
        encoding="utf-8",
    )
    _rebuild(THREE_NODES)
    assert len(_embedded) == 3, "a v1 cache must not be partially reused"
    print("ok: a legacy cache is discarded rather than misaligned")


def test_legacy_cache_still_loads_on_an_exact_fingerprint_match():
    """Backward compatibility: an unchanged install upgrading to this version
    must not pay for a full rebuild."""
    _setup(THREE_NODES)
    fingerprint = _index_mod()._fingerprint(THREE_NODES)
    _index_mod()._CACHE_PATH.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "vectors": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]}
        ),
        encoding="utf-8",
    )
    index = _rebuild(THREE_NODES)
    assert _embedded == [], "an exact v1 match should not re-embed"
    assert _vector_for(index, "Alpha") == [1.0, 1.0]
    print("ok: a legacy cache still loads when the registry is unchanged")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall index cache tests passed")
