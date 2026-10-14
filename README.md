# Comfy Agent

A ComfyUI node that builds workflows from plain English. Describe what you
want — *"SDXL portrait with a LoRA"*, *"upscale this image"* — and it places
the nodes, sets their values and wires them together on your canvas. It can
also edit the workflow already on the canvas, and answer questions about
ComfyUI, your graph and your own installation.

Everything runs locally through [Ollama](https://ollama.com). Nothing is sent
to a cloud service.

**New here?** [`docs/comfy_agent_explained.pdf`](docs/comfy_agent_explained.pdf)
is a nine-page walkthrough of how the node works: the three modes, the
generation pipeline, undo, and what happens when things fail.

## What it does

The node has three modes, chosen with its `mode` widget:

| Mode | What it does | Needs Ollama? |
|---|---|---|
| **BUILD** | Builds a new workflow next to the node. It first looks for a matching ready-made template (a *blueprint*); if none matches, a local LLM builds the workflow step by step | Only when no blueprint matches |
| **MODIFY** | Edits the workflow already on the canvas: adds, removes or changes nodes. Every change is checked against your real canvas and installed nodes before it is applied | Yes |
| **CHAT** | A chat panel on the node that answers questions using your actual installed nodes, model files, current canvas and last error | Yes |

The 8 blueprints in `blueprint_defs/` cover SDXL and Flux text-to-image,
img2img, latent upscale, face detailing, Wan video, ControlNet and LoRA
stacks. They are pre-built and checked, so they work without Ollama and are
the most reliable path.

Every BUILD and MODIFY can be reverted with **Undo Last Agent Action**.

## Install

### 1. Add the node to ComfyUI

Clone into ComfyUI's `custom_nodes` folder **as `comfy_agent`** (the repo name
has a hyphen, which Python cannot import and the test suite cannot run from):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/suryateja963/comfy-agent comfy_agent
```

Restart ComfyUI. The node appears under **agent → Comfy Agent**. Its only
Python dependency is `aiohttp`, which ComfyUI already ships.

### 2. Install Ollama and pull the models

Install Ollama from https://ollama.com, then pull:

```bash
# Required: the main model for building, classifying and chatting.
ollama pull qwen3.5:0.8b

# Strongly recommended: the "escalation" model, used when the small model's
# answer is incomplete. MODIFY runs on it first and is unreliable without it.
# Fits an 8 GB GPU.
ollama pull qwen3.5:4b

# Recommended: embeddings for CHAT's search over your installed nodes.
# Without it, CHAT falls back to keyword matching.
ollama pull nomic-embed-text
```

`qwen3.5:9b` is also supported as the escalation model if you have the VRAM;
the node uses the first of `qwen3.5:9b`, `qwen3.5:4b` that is installed.

> **About `qwen3.5-0.8b-comfyagent`.** `config.PRIMARY_MODEL` names a
> fine-tuned version of the 0.8B model. It is not on the Ollama registry, and
> its Modelfile and weights are not in this repository. You don't need it: when
> it is missing the node automatically uses `qwen3.5:0.8b` and logs a warning.
> To silence the warning, set `PRIMARY_MODEL = BASE_MODEL` in `config.py`.

### 3. Check that it works

Drag [`examples/01_build_blueprint.json`](examples/01_build_blueprint.json)
onto the canvas and press **Build Workflow**. It uses a blueprint, so it works
even with Ollama stopped. The other examples cover free-form BUILD, MODIFY and
CHAT; see [`examples/README.md`](examples/README.md).

When the node is created it checks its setup and shows the first problem it
finds on its status line, with the command that fixes it.

## Using the node

### BUILD

Write a description in `task_prompt` and press **Build Workflow**.

| Widget | Effect |
|---|---|
| `task_prompt` | What you want built |
| `task_category` | Optional hint (`txt2img`, `img2img`, `video`, `upscale`) that helps pick a blueprint |
| `model_family` | Optional hint (`sdxl`, `flux`, `sd15`, `wan`) that helps pick a blueprint |
| `max_nodes` | Upper limit on nodes for free-form builds (4–20) |
| `use_blueprint` | Turn off to always use the LLM instead of a blueprint |

Blueprints are chosen by keywords in your prompt, so naming the model or task
("sdxl", "flux", "img2img", "upscale", "controlnet", "lora", "video") makes a
blueprint match more likely.

If the LLM builds a workflow but cannot wire every input, the nodes are still
placed and the missing connections are highlighted in amber. Connect them by
hand, or press **Undo Last Agent Action** and try again.

A built workflow is never queued automatically — you run it yourself.

### MODIFY

Set `mode` to `MODIFY`, describe the change in `task_prompt` (for example
*"set steps to 30 and add a LoRA loader"*) and press **Modify Workflow**. The
node edits the existing canvas and reports what it changed. Install
`qwen3.5:4b` first; on the 0.8B model alone, MODIFY often edits the wrong node.

### CHAT

Set `mode` to `CHAT`. The Build button is replaced by a chat panel. Enter
sends, Shift+Enter adds a new line, and the reply appears as it is generated.

It answers from your installation rather than from general knowledge:

- *"What checkpoints do I have?"* — reads your model folders
- *"How do I improve faces?"* — finds `FaceDetailer` if Impact Pack is installed
- *"Why is this broken?"* — reads the current canvas
- *"Why did that fail?"* — the last execution error is attached automatically

**The first question after installing or removing node packs can take
20–30 seconds** while the node index is rebuilt. After that it is cached and
fast. The ComfyUI console shows `building install index: embedding N nodes...`
while this happens.

**Topics.** The row above the messages has one tab per conversation topic.
Type a name into *New topic* and press Enter to start a separate thread, or
click a tab to go back to one. Each topic has its own history, so the model
only sees the thread you are in. You cannot switch topics while a reply is
streaming.

All topics are saved with the workflow. A workflow saved before topics existed
opens with its chat under `default`. **Clear Chat** deletes every topic.

**Long conversations.** If a request would use more than 60% of the chat
context window (`OLLAMA_CHAT_NUM_CTX`), older messages are summarised and only
the recent ones are sent. If it is still too large, the canvas is left out of
that request. Your question is always sent.

## Configuration

Settings live in [`config.py`](config.py): model names, retry limits, context
sizes, timeouts and network access. Two can also be set as environment
variables, which survive updating the node:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama is reached. Ollama's own `0.0.0.0:11434` form is accepted |
| `AGENT_API_TOKEN` | empty | Token remote clients must send in the `X-Agent-Token` header (see [Security](#security)) |

Everything else is edited in `config.py`. Note that `git pull` will conflict
with local edits to that file.

**If image generation runs out of VRAM while chatting**, lower
`OLLAMA_CHAT_NUM_CTX` first. The chat model's memory grows with it — roughly
3–4 GB at the default 65536, on top of the model and competing with ComfyUI
for the same GPU. 32768 and 16384 are safe steps down; answers barely change,
and long conversations are simply summarised sooner.

## Troubleshooting

Check the node's status line first — it reports the first blocking problem and
how to fix it. For the full picture, with ComfyUI running:

```bash
curl http://127.0.0.1:8188/agent/health
```

This lists whether Ollama is reachable, which models are installed, whether
the node index is built, and a `hints` array with each fix.

| Symptom | Cause | Fix |
|---|---|---|
| `Ollama is not reachable at http://127.0.0.1:11434` | Ollama is not running | `ollama serve`. Blueprint builds keep working without it — try `examples/01_build_blueprint.json` |
| Node says *Primary model missing* | No model pulled | `ollama pull qwen3.5:0.8b` |
| Console warns `PRIMARY_MODEL 'qwen3.5-0.8b-comfyagent' is not installed` | The fine-tune is not available | Expected; the base model is used. Set `PRIMARY_MODEL = BASE_MODEL` to silence it |
| Builds succeed but wire poorly | Only the 0.8B model is installed | `ollama pull qwen3.5:4b` — it is used automatically as the escalation model |
| `Stage A generation failed` | The model returned unusable output | Usually a too-small model. Pull `qwen3.5:4b`, or use a prompt that matches a blueprint |
| First CHAT message pauses ~20 s | The node index is being built | Expected once after node packs change; the result is cached |
| CHAT answers vaguely about your nodes | Embedding model missing | `ollama pull nomic-embed-text` |
| Build reports *N link(s) not wired* | The model could not connect everything | Finish the amber-highlighted wiring by hand, or **Undo Last Agent Action** and retry |
| `too many concurrent builds in progress` | Two builds at once | Wait for the first to finish (at most 2 run at once) |
| MODIFY says *the canvas is empty* | Nothing to modify | Use BUILD first |
| MODIFY makes the wrong edit | No escalation model installed | `ollama pull qwen3.5:4b`, then **Undo Last Agent Action** and retry |
| Mode says it *has been replaced by CHAT* | Workflow saved with the retired EXPLAIN/DEBUG/SUGGEST modes | Set `mode = CHAT` |
| Buttons missing / node shows only widgets | The node's JavaScript failed to load | Check the browser console (F12) for `comfy_agent` errors |
| Node does not appear in ComfyUI at all | The Python side failed to import | Check the ComfyUI console for `comfy_agent`. `no usable comfy_api version found` means this ComfyUI is too old |
| Everything is slow | Model running on CPU | Expected. Blueprint builds don't use the model |

**A built workflow fails when you run it.** The node shows the error on its
status line and keeps it. Switch to CHAT and ask why — the error is already in
its context.

**Undo** restores the whole canvas to how it was just before the last BUILD or
MODIFY, so edits you made by hand since then are reverted too. Only the last
agent action can be undone.

## Limitations

- **Free-form builds are only as good as the model.** On `qwen3.5:0.8b` a
  generated workflow sometimes misses one or two connections. Blueprints don't
  have this problem; `qwen3.5:4b` reduces it.
- **CHAT's explanations are basic on the 0.8B model.** It names real installed
  nodes and files correctly, but its prose can be loose. Pointing
  `PRIMARY_MODEL` at `qwen3.5:4b` gives noticeably better answers at some VRAM
  cost.
- **The node's outputs and `context_input` are not implemented.** The node
  works through its buttons, not through the queue; queueing it outputs empty
  values, and `context_input` is accepted but ignored.

## Security

- **Local only by default.** The `/agent/*` endpoints refuse requests that
  don't come from this machine. Build and chat requests sent by other websites
  open in your browser are also rejected.
- **Opening it to other machines.** Set a token with the `AGENT_API_TOKEN`
  environment variable; remote clients send it in the `X-Agent-Token` header.
  Then choose one of:
  - `ALLOWED_NETWORK_CIDRS` (e.g. `["192.168.1.0/24"]`) — allows those
    addresses, and requires the token when one is set. Without a token anyone
    on those networks can use the agent, and the ComfyUI console warns you.
  - `ALLOW_NETWORK = True` — allows any address, but only with the token. If
    no token is set, every remote request is refused, and the console says so.
- **Behind a reverse proxy or tunnel**, every request appears to come from the
  local machine, so the local-only check no longer protects you. Add
  authentication at the proxy.
- **The node's JavaScript can read and change your whole canvas.** Don't
  install it on a ComfyUI instance exposed to an untrusted network.
- **What CHAT sends to the model:** the current canvas, your model
  *filenames*, details of relevant nodes, and the last error. With the default
  local Ollama this never leaves your machine; if you point `OLLAMA_HOST` at a
  remote server, it goes there.
- `install_index.cache.json` (about 18 MB) is generated, gitignored, and holds
  only embeddings of node descriptions — no model weights or user content.

## Development

Run the tests from a clone named `comfy_agent` (see [Install](#1-add-the-node-to-comfyui)):

```bash
pytest tests/
node --test tests/canvas_writer.test.mjs tests/health_summary.test.mjs tests/js_syntax.test.mjs
```

That is 93 Python and 42 JavaScript tests. Both suites run offline, with
Ollama, ComfyUI and LiteGraph stubbed. List the `.mjs` files explicitly: a
quoted `tests/*.test.mjs` glob is not expanded by every shell and silently runs
nothing.

Two checks use a real install:

```bash
python tests/test_blueprint_widgets.py --live   # needs ComfyUI running on :8188
pytest tests/test_frontend_api.py               # skips unless comfyui_frontend_package is installed
```

Regenerate the PDF explainer with `python docs/make_pdf.py` (needs matplotlib,
which ComfyUI ships).

**Do not add `NODE_CLASS_MAPPINGS`** to `__init__.py`. ComfyUI checks for it
before the V3 `comfy_entrypoint` and stops there if it exists — even as an
empty dict — so the node would never register.

Further reading:

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — design and the build log of what broke and why
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — per-module notes and what has been verified on a live install
- [`MASTER_PLAN.md`](MASTER_PLAN.md) — the original phased plan

## License

[MIT](LICENSE)
