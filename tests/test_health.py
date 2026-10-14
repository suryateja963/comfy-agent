"""/agent/health tests: per-service readiness reporting.

Ollama, ComfyUI and ComfyUI's `server` module are all stubbed, so this runs
with none of them installed.

    pytest tests/                 # collected as test_* functions
    python tests/test_health.py   # standalone, no pytest needed
"""

import asyncio
import importlib.util
import json
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


def _authorized(remote, *, token="", allow_network=False, cidrs=(), header=None):
    saved = (config.AGENT_API_TOKEN, config.ALLOW_NETWORK, config.ALLOWED_NETWORK_CIDRS)
    config.AGENT_API_TOKEN = token
    config.ALLOW_NETWORK = allow_network
    config.ALLOWED_NETWORK_CIDRS = list(cidrs)
    try:
        request = _FakeRequest(remote=remote)
        if header is not None:
            request.headers["X-Agent-Token"] = header
        return agent_routes._is_authorized(request)
    finally:
        config.AGENT_API_TOKEN, config.ALLOW_NETWORK, config.ALLOWED_NETWORK_CIDRS = saved


def test_token_is_not_generated_when_unset():
    """A random token nobody can read locked remote clients out while
    looking configured."""
    import os

    if not os.getenv("AGENT_API_TOKEN"):
        assert config.AGENT_API_TOKEN == ""
    print("ok: no token is invented when none is configured")


def test_localhost_never_needs_a_token():
    assert _authorized("127.0.0.1", token="s3cret", allow_network=True)
    assert _authorized("::1", token="s3cret", cidrs=["10.0.0.0/8"])
    print("ok: localhost needs no token")


def test_cidr_peer_must_send_a_configured_token():
    """The token used to be ignored for CIDR peers entirely."""
    cidrs = ["10.0.0.0/8"]
    assert not _authorized("10.1.2.3", token="s3cret", cidrs=cidrs)
    assert not _authorized("10.1.2.3", token="s3cret", cidrs=cidrs, header="wrong")
    assert _authorized("10.1.2.3", token="s3cret", cidrs=cidrs, header="s3cret")
    print("ok: a configured token is required from allow-listed networks")


def test_cidr_peer_without_a_token_configured_is_allowed():
    assert _authorized("10.1.2.3", cidrs=["10.0.0.0/8"])
    assert not _authorized("192.168.1.5", cidrs=["10.0.0.0/8"])
    print("ok: without a token, the CIDR allowlist alone decides")


def test_allow_network_requires_a_matching_token():
    assert not _authorized("203.0.113.9", allow_network=True)
    assert not _authorized("203.0.113.9", allow_network=True, header="")
    assert not _authorized("203.0.113.9", token="s3cret", allow_network=True, header="nope")
    assert _authorized("203.0.113.9", token="s3cret", allow_network=True, header="s3cret")
    print("ok: ALLOW_NETWORK admits only requests with the configured token")


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


def test_health_warns_when_only_one_model_is_installed():
    """The case that actually bites: models resolve fine, but there is nothing
    better than the primary to escalate to. MODIFY runs on the escalation
    model by design, so this is a correctness warning, not a quality note."""
    _patch(available=True, installed=[config.BASE_MODEL, config.EMBED_MODEL])
    body = _call_health().body.decode()
    assert "MODIFY is unreliable" in body, "a single-model install must warn about MODIFY"
    assert f"ollama pull {config.FALLBACK_CHAIN[1]}" in body
    print("ok: a single-model install is warned that MODIFY is unreliable")


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


# --- /agent/chat: prompt budgeting. Lives here because this file already
# builds the stubbed route harness; a missing import in agent_routes (the
# count_prompt_tokens regression) fails this whole module at collection. ---

chat_context = agent_routes.chat_context


class _FakeJSONRequest(_FakeRequest):
    def __init__(self, body):
        super().__init__()
        self._body = body

    async def json(self):
        return self._body


def _chat_turns(n_pairs, size):
    turns = []
    for i in range(n_pairs):
        turns.append({"role": "user", "content": f"q{i} " + "u" * size})
        turns.append({"role": "assistant", "content": f"a{i} " + "a" * size})
    turns.append({"role": "user", "content": "THE QUESTION"})
    return turns


def _run_chat(body, *, system_for_graph, summarize=None, reply="ok", fail=None, ctx=1000):
    """Drive /agent/chat with every Ollama and context call stubbed.

    Returns (response, captured) where captured records what reached the
    model and how the system prompt was built.
    """
    captured = {"stream_calls": [], "system_builds": []}

    async def fake_is_available():
        return True

    async def fake_build_system_prompt(question, graph, last_error=""):
        captured["system_builds"].append(graph)
        return system_for_graph(graph)

    async def fake_summarize(messages):
        if summarize is None:
            raise AssertionError("summarize_history should not have been called")
        return summarize(messages)

    async def fake_chat_stream(messages, *, system=None, model=None, on_token=None):
        captured["stream_calls"].append({"messages": list(messages), "system": system})
        if fail is not None:
            raise fail
        return reply

    saved = (
        llm_client.is_available,
        llm_client.chat_stream,
        chat_context.build_system_prompt,
        chat_context.summarize_history,
        config.OLLAMA_CHAT_NUM_CTX,
    )
    llm_client.is_available = fake_is_available
    llm_client.chat_stream = fake_chat_stream
    chat_context.build_system_prompt = fake_build_system_prompt
    chat_context.summarize_history = fake_summarize
    config.OLLAMA_CHAT_NUM_CTX = ctx
    try:
        handler = _fake_routes.handlers["/agent/chat"]
        response = asyncio.run(handler(_FakeJSONRequest(body)))
    finally:
        (
            llm_client.is_available,
            llm_client.chat_stream,
            chat_context.build_system_prompt,
            chat_context.summarize_history,
            config.OLLAMA_CHAT_NUM_CTX,
        ) = saved
    return response, captured


def test_prompt_token_estimate_counts_the_system_prompt():
    """The system block (canvas, inventory, schemas) is most of a real chat
    request; an estimate that ignores it never trips the budget."""
    messages = [{"role": "user", "content": "x" * 400}]
    without = chat_context.count_prompt_tokens(messages)
    with_system = chat_context.count_prompt_tokens(messages, "s" * 4000)
    assert with_system > without * 5, (without, with_system)
    print("ok: token estimate includes the system prompt")


def test_small_chat_is_sent_unchanged():
    body = {"messages": [{"role": "user", "content": "hi"}], "graph": {"nodes": []}}
    response, captured = _run_chat(body, system_for_graph=lambda g: "SYS")
    assert response.status == 200
    assert len(captured["stream_calls"]) == 1
    assert captured["stream_calls"][0]["messages"] == body["messages"]
    assert captured["stream_calls"][0]["system"] == "SYS"
    print("ok: a chat under budget is sent as-is")


def test_oversized_history_is_summarised_and_still_ends_on_the_question():
    """The summary must not become a trailing assistant turn — that left the
    model with no question to answer."""
    body = {"messages": _chat_turns(n_pairs=5, size=600), "graph": None}

    def summarize(messages):
        return "- user is building an SDXL workflow", messages[-3:-1]

    response, captured = _run_chat(
        body, system_for_graph=lambda g: "SYS", summarize=summarize
    )
    assert response.status == 200, response.body
    sent = captured["stream_calls"][0]
    assert sent["messages"][-1] == {"role": "user", "content": "THE QUESTION"}
    assert all(m["role"] in ("user", "assistant") for m in sent["messages"])
    assert "SDXL workflow" in sent["system"], "summary belongs in the system prompt"
    assert len(sent["messages"]) == 3
    print("ok: summarised history keeps the question last")


def test_canvas_is_dropped_when_summarising_is_not_enough():
    """Dropping the graph only helps if the system prompt is rebuilt — the
    old code set graph = None after the prompt was already rendered."""
    body = {
        "messages": [{"role": "user", "content": "why is this broken?"}],
        "graph": {"nodes": [{"id": 1, "type": "KSampler"}]},
    }

    def system_for_graph(graph):
        return "G" * 20000 if graph else "SMALL"

    response, captured = _run_chat(body, system_for_graph=system_for_graph)
    assert response.status == 200, response.body
    assert captured["system_builds"][-1] is None, "must rebuild without the canvas"
    assert captured["stream_calls"][0]["system"] == "SMALL"
    print("ok: an oversized canvas is dropped from the prompt")


def test_budget_follows_the_configured_context_window():
    """Lowering OLLAMA_CHAT_NUM_CTX is the documented VRAM fix, so the
    budget must follow it rather than a hardcoded 65536."""
    body = {"messages": [{"role": "user", "content": "hi"}], "graph": None}
    response, _ = _run_chat(body, system_for_graph=lambda g: "SYS", ctx=16384)
    assert json.loads(response.body)["token_budget"]["total"] == 16384
    print("ok: budget is reported against the configured window")


def test_failed_reply_is_not_retried_without_the_question():
    body = {"messages": _chat_turns(n_pairs=1, size=10), "graph": None}
    response, captured = _run_chat(
        body,
        system_for_graph=lambda g: "SYS",
        fail=llm_client.OllamaGenerationError("boom"),
    )
    assert response.status == 502
    assert len(captured["stream_calls"]) == 1, "no silent retry on a different question"
    print("ok: a failed reply is reported, not retried")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall health tests passed")
