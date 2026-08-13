"""/agent/health tests: per-service readiness reporting.

Ollama, ComfyUI and ComfyUI's `server` module are all stubbed, so this runs
with none of them installed.

    pytest tests/                 # collected as test_* functions
    python tests/test_health.py   # standalone, no pytest needed
"""

import asyncio
import importlib.util
import pathlib
import sys
import types

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


# --- Stub ComfyUI's `server` module: agent_routes imports PromptServer at
# module scope, and it is only importable inside a running ComfyUI. ---
class _FakeRoutes:
    """Collects the handlers agent_routes registers, keyed by path."""

    def __init__(self):
        self.handlers = {}

    def _register(self, path):
        def decorator(fn):
            self.handlers[path] = fn
            return fn

        return decorator

    def get(self, path):
        return self._register(path)

    def post(self, path):
        return self._register(path)


_fake_routes = _FakeRoutes()
_server_mod = types.ModuleType("server")
_server_mod.PromptServer = types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=_fake_routes, send_sync=lambda *a, **k: None)
)
sys.modules["server"] = _server_mod

_root = _register_pkg("comfy_agent", PKG)
config = _load("comfy_agent.config", PKG + "/config.py", _root)
llm_client = _load("comfy_agent.llm_client", PKG + "/llm_client.py", _root)
knowledge = _register_pkg("comfy_agent.knowledge", PKG + "/knowledge")
_load("comfy_agent.knowledge.type_validator", PKG + "/knowledge/type_validator.py", knowledge)
_load("comfy_agent.knowledge.schema_compactor", PKG + "/knowledge/schema_compactor.py", knowledge)
blueprint_matcher = _load(
    "comfy_agent.knowledge.blueprint_matcher", PKG + "/knowledge/blueprint_matcher.py", knowledge
)
install_index = _load("comfy_agent.install_index", PKG + "/install_index.py", _root)
_load("comfy_agent.chat_context", PKG + "/chat_context.py", _root)
generation = _register_pkg("comfy_agent.generation", PKG + "/generation")
for _stage in ("classify", "stage_a_identify", "stage_b_configure", "stage_c_connect", "stage_d_validate"):
    _load(f"comfy_agent.generation.{_stage}", f"{PKG}/generation/{_stage}.py", generation)
_load("comfy_agent.generation.pipeline", PKG + "/generation/pipeline.py", generation)
agent_routes = _load("comfy_agent.agent_routes", PKG + "/agent_routes.py", _root)


class _FakeRequest:
    """Minimal stand-in for aiohttp's Request: only `remote` and `headers`."""

    def __init__(self, remote="127.0.0.1"):
        self.remote = remote
        self.headers = {}


def _call_health(request=None):
    """Invoke the registered /agent/health handler and return its JSON body."""
    handler = _fake_routes.handlers["/agent/health"]
    response = asyncio.run(handler(request or _FakeRequest()))
    return response


def _patch(available, installed):
    """Stub the two Ollama probes health() depends on."""
    async def fake_is_available():
        return available

    async def fake_list_installed():
        return set(installed)

    llm_client.is_available = fake_is_available
    llm_client.list_installed = fake_list_installed
    # resolve_models() caches process-wide; clear it so each case resolves fresh.
    llm_client._resolved = None


def test_health_reports_ollama_down():
    _patch(available=False, installed=[])
    body = _call_health().body
    assert b'"ollama": false' in body
    # A down Ollama must still be a 200: the frontend has to distinguish an
    # unhealthy backend from an unreachable one.
    assert _call_health().status == 200
    assert b"ollama serve" in body, "the fix command must be in the hints"
    print("ok: health reports an unreachable Ollama with a fix hint")


def test_health_reports_all_models_present():
    _patch(
        available=True,
        installed=[config.BASE_MODEL, config.FALLBACK_CHAIN[0], config.EMBED_MODEL],
    )
    body = _call_health().body
    assert b'"ollama": true' in body
    assert b'"hints": []' in body, "no hints expected when everything is installed"
    print("ok: health reports a fully provisioned install with no hints")


def test_health_hints_missing_embed_model():
    _patch(available=True, installed=[config.BASE_MODEL, config.FALLBACK_CHAIN[0]])
    body = _call_health().body.decode()
    assert config.EMBED_MODEL in body
    assert f"ollama pull {config.EMBED_MODEL}" in body
    print("ok: a missing embed model is named with its pull command")


def test_health_rejects_unauthorized_peer():
    _patch(available=True, installed=[])
    response = _call_health(_FakeRequest(remote="10.1.2.3"))
    assert response.status == 403
    print("ok: health refuses a non-localhost peer")


def test_health_reports_index_not_ready():
    _patch(available=True, installed=[config.BASE_MODEL])
    install_index._index = None
    body = _call_health().body
    assert b'"ready": false' in body
    print("ok: health reports an unbuilt index without triggering a build")


def test_health_counts_blueprints():
    _patch(available=True, installed=[config.BASE_MODEL])
    body = _call_health().body.decode()
    expected = len(blueprint_matcher.list_blueprints())
    assert f'"blueprints": {expected}' in body
    assert expected > 0, "the blueprint library should not be empty"
    print(f"ok: health counts {expected} blueprints")


def test_superseded_modes_name_their_replacement():
    """A workflow saved before EXPLAIN/DEBUG/SUGGEST were removed from the
    node still sends them. The reply must point at CHAT rather than reading
    as a missing feature."""
    for mode in ("EXPLAIN", "DEBUG", "SUGGEST"):
        result = asyncio.run(agent_routes._handle_superseded_mode(mode))
        assert "CHAT" in result["error_detail"]
        assert mode in result["error_detail"]
        assert result["execution_ready"] is False
    print("ok: superseded modes point the user at CHAT")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall health tests passed")
