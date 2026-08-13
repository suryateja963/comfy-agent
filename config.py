"""All model names and budget numbers live here and nowhere else. Section 4A."""

from __future__ import annotations

import os


def _normalize_host(raw: str, default_port: int) -> str:
    """Turn a loosely-specified host into a URL a client can actually call.

    ``OLLAMA_HOST`` is Ollama's *server bind* variable and is commonly set to
    something like ``0.0.0.0:11434`` — no scheme, and a bind-any address that
    is not a valid connect target. Reusing that name for a client URL is
    convenient but needs this normalisation, otherwise every request fails
    with "Request URL is missing an 'http://' or 'https://' protocol" far from
    the cause.

    Kept byte-identical to comfyagent/agent/config.py's version on purpose:
    two copies that drift are worse than two copies that don't, and the custom
    node cannot import from the standalone agent package.
    """
    host = (raw or "").strip()
    if not host:
        return f"http://127.0.0.1:{default_port}"
    if "://" not in host:
        host = f"http://{host}"
    # 0.0.0.0 means "listen on every interface"; to connect, use loopback.
    host = host.replace("://0.0.0.0", "://127.0.0.1")
    return host.rstrip("/")


OLLAMA_HOST = _normalize_host(os.getenv("OLLAMA_HOST", ""), 11434)

# The supported default: on the Ollama registry, so `ollama pull qwen3.5:0.8b`
# is all a new user needs. Everything below degrades to this.
BASE_MODEL = "qwen3.5:0.8b"

# Opt-in upgrade. LoRA fine-tuned on 545 verified ComfyUI workflow-generation
# examples (8 blueprints reversed + 68 synthetic prompts validated through the
# real pipeline). Measured +25% avg link-completeness over BASE_MODEL across 5
# held-out test prompts x 3 runs each — see ARCHITECTURE.md for the full
# before/after numbers.
#
# NOT PUBLISHED on the Ollama registry: it is built locally from the Modelfile.
# If it is absent, resolve_models() falls back to BASE_MODEL with a warning
# rather than failing every Stage A call with an opaque "all attempts failed".
PRIMARY_MODEL = "qwen3.5-0.8b-comfyagent"

# Escalation targets, tried in order; the first one actually installed wins.
# A chain rather than a single name because a 9B is a large ask on an 8 GB
# card, and a user who has only the 4B should still get escalation rather than
# silently getting none.
FALLBACK_CHAIN = ("qwen3.5:9b", "qwen3.5:4b", BASE_MODEL)
FALLBACK_MODEL = FALLBACK_CHAIN[0]

# Stage generation retry policy (Section 4A model-swap risk notes).
# 0.8B inference is cheap; escalate to the fallback model fast rather than
# burning three same-model retries.
STAGE_RETRY_CAP = 1
FALLBACK_RETRY_CAP = 2

# Pass 2 schema payload budget — biased toward the low end of the
# 2000-8000 token range in Section 2D, since small models lose coherence
# faster as filtered /object_info context grows.
PASS2_MAX_SCHEMA_TOKENS = 2500

# Subgraph wrapping threshold (Section 2F).
SUBGRAPH_WRAP_THRESHOLD = 6

# Ollama request timeout, seconds. Integration test target: 503 within 5s
# when Ollama is offline (Section 6).
OLLAMA_TIMEOUT_SECONDS = 5.0
OLLAMA_GENERATE_TIMEOUT_SECONDS = 60.0

# Chat replies stream token-by-token, so a long answer is visibly progressing
# rather than blocking — a more generous ceiling than the one-shot generate
# path is safe here, and prose on a small local model is genuinely slower.
OLLAMA_CHAT_TIMEOUT_SECONDS = 180.0

# --- CHAT mode ---
# Conversation turns (user+assistant messages) kept and resent per request.
# The whole history rides in every call, so this trades recall against the
# context window shared with the canvas snapshot below.
CHAT_MAX_HISTORY_MESSAGES = 12

# Hard cap on a single chat message, characters. Prevents a paste of a giant
# log from blowing the context window out on a 0.8B model.
CHAT_MAX_MESSAGE_CHARS = 8000

# Char budget for the serialized canvas graph injected into the system prompt.
# A large workflow serializes to far more than a small model can hold; past
# this the snapshot is summarised to node types + link count instead.
CHAT_MAX_GRAPH_CHARS = 6000

# Concurrent chat requests served at once.
MAX_CONCURRENT_CHATS = 2

# --- Install knowledge index (semantic retrieval) ---
# The full /object_info on a real install is ~2.7 MB / 1767 node types — about
# 680k tokens, an order of magnitude past even OLLAMA_CHAT_NUM_CTX. So the
# index is searched per question and only the top matches are injected.
EMBED_MODEL = "nomic-embed-text"

# Nodes whose schemas are injected per question. The limit is answer quality,
# not context: a small model reasons better over 24 well-chosen nodes than over
# hundreds of marginal ones.
CHAT_RETRIEVAL_TOP_K = 24

# Embedding requests batched per Ollama call while building the index.
EMBED_BATCH_SIZE = 64

# Where the built index is cached, relative to this package. Rebuilding embeds
# 1767 node descriptions (~1-2 min), so it is cached and reused across restarts.
INDEX_CACHE_FILENAME = "install_index.cache.json"

# Model folders enumerated for the inventory. Empty means "every folder the
# server reports at /models", minus MODEL_FOLDER_DENYLIST below.
MODEL_FOLDERS: list[str] = []

# /models lists some folders that hold source code rather than weights. On this
# install "custom_nodes" alone returns 1400 entries — .editorconfig, .eslintrc,
# GitHub issue templates — which would crowd the real checkpoints out of the
# prompt and invite the model to cite a linter config as a usable model.
MODEL_FOLDER_DENYLIST = frozenset({"custom_nodes", "configs", "datasets", "classifiers"})

# Only files that look like model weights are listed, for the same reason.
MODEL_FILE_EXTENSIONS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
    ".onnx", ".sft", ".patch", ".engine",
)

# Cap on filenames listed per model folder before summarising with a count, so
# an install with thousands of LoRAs cannot flood the prompt.
MODEL_LIST_CAP = 60

# Ollama context window (tokens) for the BUILD pipeline. Must comfortably hold
# PASS2_MAX_SCHEMA_TOKENS plus prompt/system overhead. Ollama silently truncates
# context when num_ctx is unset (defaults can be as low as 2048 on some models),
# which would drop Stage A's node list mid-prompt.
OLLAMA_NUM_CTX = 8192

# Separate, much larger window for CHAT. Retrieval keeps the prompt far below
# this in practice; the headroom is for long conversations and big graphs.
#
# VRAM WARNING: the KV cache grows linearly with this value — roughly 3-4 GB at
# 65536 on top of the model weights, on a card with ~7.4 GB free. If a large
# SDXL/Flux generation OOMs while chatting, LOWER THIS FIRST. 32768 and 16384
# are both safe steps down, and retrieval means answers barely change.
OLLAMA_CHAT_NUM_CTX = 65536

COMFY_HOST = "http://127.0.0.1:8188"
COMFY_TIMEOUT_SECONDS = 5.0

# Network deployment opt-in (Section 4C). Off by default.
#
# SECURITY: setting this True removes the ONLY access control on
# /agent/build_workflow — an endpoint that drives local LLM inference and
# writes to the canvas. Prefer ALLOWED_NETWORK_CIDRS below, which keeps the
# peer check in place and narrows it to specific subnets, and set
# AGENT_API_TOKEN whenever the endpoint is reachable off-host.
ALLOW_NETWORK = False

# Optional allowlist of CIDR ranges permitted in addition to localhost, e.g.
# ["192.168.1.0/24"]. Empty means localhost-only. Ignored when ALLOW_NETWORK
# is True (that flag already allows everything).
ALLOWED_NETWORK_CIDRS: list[str] = []

# Shared secret required on non-localhost requests, sent as X-Agent-Token.
# Strongly recommended whenever ALLOWED_NETWORK_CIDRS is non-empty; an empty
# value disables the check.
AGENT_API_TOKEN = ""

BLUEPRINT_DIR_NAME = "blueprint_defs"

# --- Canvas layout (js/canvas_writer.js reads these via the route response) ---
# Horizontal pitch between free-form nodes, and the assumed row height used to
# size the free-zone search box.
FREEFORM_COLUMN_PITCH = 340
FREEFORM_ROW_HEIGHT = 260

# --- Prompt shaping ---
# How many COMBO options to show a model before truncating with ", ...".
WIDGET_OPTION_DISPLAY_CAP = 12

# Candidate-pool ceiling for the Pass 2 schema block, independent of the char
# budget derived from PASS2_MAX_SCHEMA_TOKENS.
COMPACT_SCHEMA_MAX_NODES = 200

# Rough chars-per-token ratio used to convert the token budget into a char
# budget for the compacted schema block.
CHARS_PER_TOKEN_ESTIMATE = 4
