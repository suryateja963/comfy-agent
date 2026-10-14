"""CHAT mode tests: context assembly, message validation, and token streaming.

Ollama and ComfyUI are both stubbed, so this runs with neither installed.

    pytest tests/                # collected as test_* functions
    python tests/test_chat.py    # standalone, no pytest needed
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


_root = _register_pkg("comfy_agent", PKG)
config = _load("comfy_agent.config", PKG + "/config.py", _root)
llm_client = _load("comfy_agent.llm_client", PKG + "/llm_client.py", _root)
knowledge = _register_pkg("comfy_agent.knowledge", PKG + "/knowledge")
_load("comfy_agent.knowledge.type_validator", PKG + "/knowledge/type_validator.py", knowledge)
schema_compactor = _load(
    "comfy_agent.knowledge.schema_compactor", PKG + "/knowledge/schema_compactor.py", knowledge
)
install_index = _load("comfy_agent.install_index", PKG + "/install_index.py", _root)
chat_context = _load("comfy_agent.chat_context", PKG + "/chat_context.py", _root)


FAKE_OBJECT_INFO = {
    "KSampler": {
        "input": {"required": {"model": ["MODEL"], "steps": ["INT"], "cfg": ["FLOAT"]}},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
    },
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["a.safetensors"]]}},
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
        "category": "loaders",
    },
    "WeirdCustomNode": {
        "input": {"required": {"x": ["INT"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "custom/experimental",
    },
}

SAMPLE_GRAPH = {
    "nodes": [
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "pos": [10, 20],
            "size": [300, 100],
            "flags": {},
            "widgets_values": ["sdxl.safetensors"],
        },
        {
            "id": 2,
            "type": "KSampler",
            "pos": [400, 20],
            "size": [300, 260],
            "widgets_values": [0, "randomize", 20, 8.0],
            "inputs": [{"name": "model", "link": 5}, {"name": "latent_image", "link": None}],
        },
    ],
    "links": [[5, 1, 0, 2, 0, "MODEL"]],
}


def test_graph_block_includes_nodes_and_drops_presentation():
    block = chat_context.render_graph_block(SAMPLE_GRAPH)
    assert "KSampler" in block
    assert "sdxl.safetensors" in block
    # Position/size/flags are pure presentation and must not eat the budget.
    assert "flags" not in block
    assert '"pos"' not in block
    assert '"size"' not in block
    # A connected input is retained; an unconnected one is dropped.
    assert "connected_inputs" in block
    print("ok: graph block keeps semantics, drops presentation")


def test_graph_block_summarizes_when_oversized():
    big = {"nodes": [dict(SAMPLE_GRAPH["nodes"][1], id=i) for i in range(4000)], "links": []}
    block = chat_context.render_graph_block(big)
    assert "too large" in block
    assert "KSampler" in block
    assert len(block) < config.CHAT_MAX_GRAPH_CHARS
    print("ok: oversized graph degrades to a summary within budget")


def test_graph_block_handles_empty_canvas():
    assert "empty" in chat_context.render_graph_block({"nodes": []})
    assert "empty" in chat_context.render_graph_block(None)
    print("ok: empty canvas handled")


def test_schema_block_includes_canvas_types_without_keyword_match():
    """A question naming no category must still get the user's own nodes."""
    async def fake_fetch():
        return FAKE_OBJECT_INFO

    schema_compactor.fetch_object_info = fake_fetch
    graph = {"nodes": [{"id": 1, "type": "WeirdCustomNode"}]}
    block = asyncio.run(chat_context.render_schema_block("why is this broken?", graph))
    assert "WeirdCustomNode" in block
    print("ok: canvas node types are always included in the schema block")


def test_system_prompt_survives_object_info_failure():
    async def boom():
        raise RuntimeError("/object_info unreachable")

    schema_compactor.fetch_object_info = boom
    prompt = asyncio.run(chat_context.build_system_prompt("what is a KSampler?", SAMPLE_GRAPH))
    assert "ComfyUI assistant" in prompt
    assert "KSampler" in prompt  # still present via the graph block
    print("ok: chat degrades gracefully when /object_info is down")


def test_system_prompt_includes_last_error():
    async def fake_fetch():
        return FAKE_OBJECT_INFO

    schema_compactor.fetch_object_info = fake_fetch
    prompt = asyncio.run(
        chat_context.build_system_prompt("why did it fail?", SAMPLE_GRAPH, "CUDA out of memory")
    )
    assert "CUDA out of memory" in prompt
    assert "EXECUTION ERROR" in prompt
    print("ok: last execution error reaches the prompt")


# --- streaming ---


class _FakeContent:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()


class _FakeResponse:
    def __init__(self, lines, status=200):
        self.status = status
        self.content = _FakeContent(lines)

    async def text(self):
        return "error body"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, *a, **k):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _chunk(text, done=False):
    return json.dumps({"message": {"content": text}, "done": done}).encode()


def test_chat_stream_assembles_tokens_in_order():
    lines = [_chunk("Hel"), _chunk("lo "), _chunk("world"), _chunk("", done=True)]
    llm_client.aiohttp.ClientSession = lambda *a, **k: _FakeSession(_FakeResponse(lines))
    llm_client._resolved = ("m", "m")

    seen = []

    async def on_token(t):
        seen.append(t)

    reply = asyncio.run(
        llm_client.chat_stream([{"role": "user", "content": "hi"}], on_token=on_token)
    )
    assert reply == "Hello world"
    assert seen == ["Hel", "lo ", "world"]
    print("ok: tokens stream in order and assemble to the full reply")


def test_chat_stream_skips_malformed_chunks():
    lines = [_chunk("a"), b"{not json", _chunk("b"), _chunk("", done=True)]
    llm_client.aiohttp.ClientSession = lambda *a, **k: _FakeSession(_FakeResponse(lines))
    llm_client._resolved = ("m", "m")
    reply = asyncio.run(llm_client.chat_stream([{"role": "user", "content": "hi"}]))
    assert reply == "ab"
    print("ok: a malformed mid-stream frame does not abort the reply")


def test_chat_stream_rejects_empty_reply():
    llm_client.aiohttp.ClientSession = lambda *a, **k: _FakeSession(
        _FakeResponse([_chunk("", done=True)])
    )
    llm_client._resolved = ("m", "m")
    try:
        asyncio.run(llm_client.chat_stream([{"role": "user", "content": "hi"}]))
    except llm_client.OllamaGenerationError:
        print("ok: empty reply raises rather than returning a blank message")
    else:
        raise AssertionError("expected OllamaGenerationError on empty reply")


def test_chat_stream_sends_no_format_constraint():
    """The build path forces JSON; chat must not, or answers come back as JSON."""
    captured = {}

    class _CapturingSession(_FakeSession):
        def post(self, url, json=None, **k):
            captured.update(json or {})
            return self._response

    llm_client.aiohttp.ClientSession = lambda *a, **k: _CapturingSession(
        _FakeResponse([_chunk("hi", done=True)])
    )
    llm_client._resolved = ("m", "m")
    asyncio.run(llm_client.chat_stream([{"role": "user", "content": "q"}], system="SYS"))
    assert "format" not in captured
    assert captured["stream"] is True
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    print("ok: chat sends no JSON-schema constraint and prepends the system prompt")


# --- install index / retrieval ---


def test_camelcase_is_split_for_embedding():
    """Embedders tokenize CamelCase poorly; the split form is what makes a
    name-carried concept searchable."""
    text = install_index._describe_node("FaceDetailer", {"description": "enhances details"})
    assert "Face Detailer" in text
    assert install_index._describe_node("VAEDecode", {}).count("VAE Decode") == 1
    print("ok: node names are split into words for embedding")


def test_stopwords_do_not_drive_name_ranking():
    """Regression: 'how do I MAKE faces look better?' ranked MakeTrainingDataset
    above FaceDetailer purely on the filler verb 'make'."""
    names = ["MakeTrainingDataset", "FaceDetailer"]
    bonus = install_index._name_match_bonus("how do I make faces look better?", names)
    assert bonus[0] == 0.0, "filler verb 'make' must not boost a node name"
    assert bonus[1] > 0.0, "'faces' should still match FaceDetailer"
    print("ok: stopwords excluded from name-match ranking")


def test_exact_name_outranks_its_own_variants():
    """Regression: 'what does KSampler do' returned KSamplerAdvanced and
    KSamplerProvider but not plain KSampler."""
    names = ["KSamplerAdvanced", "KSamplerProvider", "KSampler"]
    bonus = install_index._name_match_bonus("what does KSampler do", names)
    assert bonus[2] > bonus[0] and bonus[2] > bonus[1]
    print("ok: an exact node name outranks its longer variants")


def test_model_inventory_lists_only_weights():
    install_index._index = {
        "names": [], "descriptions": [], "schema_lines": [], "vectors": [],
        "models": {"checkpoints": ["sdxl.safetensors", "notes.md", ".editorconfig"]},
        "fingerprint": "x",
    }
    rendered = install_index.render_model_inventory()
    assert "sdxl.safetensors" in rendered
    print("ok: model inventory renders installed weights")
    install_index._index = None


def test_model_inventory_caps_huge_folders():
    many = [f"lora_{i}.safetensors" for i in range(500)]
    install_index._index = {
        "names": [], "descriptions": [], "schema_lines": [], "vectors": [],
        "models": {"loras": many}, "fingerprint": "x",
    }
    rendered = install_index.render_model_inventory()
    assert "more)" in rendered, "an oversized folder must be summarised, not dumped"
    assert len(rendered) < 8000
    install_index._index = None
    print("ok: a huge model folder cannot flood the prompt")


def test_task_category_is_read_from_the_users_words():
    """An earlier version matched mixed-case node names against the
    lowercased question, so every question classified as "custom"."""
    assert install_index._task_categories("make me a video workflow") == ["video"]
    assert install_index._task_categories("how do I upscale this?") == ["upscale"]
    assert install_index._task_categories("copy the pose with controlnet") == ["controlnet"]
    assert install_index._task_categories("what CLIP model should I use?") == []
    assert install_index._task_categories("I want a cat") == [], "'want' is not 'wan'"
    print("ok: task categories come from the question's own words")


def _fake_index(names, descriptions):
    return {
        "names": names,
        "descriptions": descriptions,
        "schema_lines": [f"{n}: line" for n in names],
        "vectors": [],
        "models": {},
        "fingerprint": "x",
    }


def test_category_boost_reorders_matching_nodes():
    """Both nodes match "animation" equally; the question is about video, so
    the animated-output node must rank first. The old rerank ran after the
    list was built and changed nothing."""
    install_index._index = _fake_index(
        ["SaveImage", "SaveAnimatedWEBP"],
        ["save image animation frames", "save animated webp animation"],
    )
    try:
        lines = asyncio.run(install_index.search("save my animation"))
    finally:
        install_index._index = None
    assert lines[0].startswith("SaveAnimatedWEBP"), lines
    print("ok: the category boost changes the ranking")


def test_category_boost_does_not_add_irrelevant_nodes():
    # AnimateDiffLoader carries the video marker ("animat") but nothing in the
    # question matches it, so its score is 0 and the boost must keep it at 0.
    install_index._index = _fake_index(
        ["LatentUpscale", "AnimateDiffLoader"],
        ["upscale latent image", "load a file from disk"],
    )
    try:
        lines = asyncio.run(install_index.search("how do I upscale a video?"))
    finally:
        install_index._index = None
    assert [line.split(":")[0] for line in lines] == ["LatentUpscale"], lines
    print("ok: a boost only reorders nodes that already matched")


def test_graph_budget_reads_model_size_from_the_name():
    """The hardcoded list missed the default fine-tune's name, so the model
    the node ships configured for got the large-model budget."""
    full = config.CHAT_MAX_GRAPH_CHARS
    assert chat_context.get_graph_char_budget("qwen3.5-0.8b-comfyagent") == full // 2
    assert chat_context.get_graph_char_budget("qwen3.5:0.8b") == full // 2
    assert chat_context.get_graph_char_budget("qwen3.5:4b") == full // 2
    assert chat_context.get_graph_char_budget("qwen3.5:9b") == full
    assert chat_context.get_graph_char_budget("llama3.1:70b") == full
    assert chat_context.get_graph_char_budget("mystery-model") == full
    assert chat_context.get_graph_char_budget("") == full
    print("ok: graph budget follows the model's parameter count")


def test_cosine_handles_zero_vectors():
    assert install_index._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert abs(install_index._cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    print("ok: cosine similarity is safe on zero vectors")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall chat tests passed")
