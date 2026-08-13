# comfy_agent

A meta-node that reads natural language and builds ComfyUI workflows around
itself directly on the canvas — and, in `CHAT` mode, answers questions about
ComfyUI, your own graph, and your specific installation from a live index of
every node and model you have. See `../MASTER_PLAN.md` (repo root) for the
full architecture, phased implementation plan, and risk notes.

## Status

Phase 0 ("Hello Canvas") + Phase 1 ("Blueprint Mode Works") + Phase 2
("Free-Form Mode Works") are implemented and **have been run end-to-end
against a live ComfyUI Desktop install + local Ollama**, including a full
bug-fixing pass driven by real failures observed in the browser and via
direct API calls — not just static review. See below for what was found and
fixed, and what remains a known limitation rather than a bug.

What exists:

- `agent_node.py` — V3-schema node. Imports the newest *pinned* `comfy_api`
  version available (`v0_0_3`, then `v0_0_2`), falling back to
  `comfy_api.latest` only if no pinned version is importable, so a ComfyUI
  release that retires a version cannot make the node unloadable. Confirmed
  on ComfyUI v0.32.0: only `v0_0_1`, `v0_0_2` and `latest` exist, so the
  ladder resolves `v0_0_2` — `v0_0_3` is there for when it ships. Verified
  against the real `comfy_api.latest._io.py` on a live install (`io.Schema`,
  `io.String`/`Combo`/`Int`/`Boolean` `.Input`/`.Output`, `NodeOutput` arg
  order, `ComfyExtension.get_node_list`) — not a guess anymore.
- `__init__.py` — registers via `comfy_entrypoint` only. **Do not add
  `NODE_CLASS_MAPPINGS`** — the real loader in `nodes.py` checks for it
  before `comfy_entrypoint` and short-circuits if present (even as `{}`),
  so the V3 path would never run.
- `agent_routes.py` — `/agent/ping`, `/agent/health`, `/agent/build_workflow`
  (BUILD mode: blueprint-first, falls back to free-form), `/agent/chat`
  (CHAT mode), `/agent/build_workflow` with `mode = MODIFY` (edits the
  workflow already on the canvas). EXPLAIN/DEBUG/SUGGEST were superseded by
  CHAT and are no longer offered by the node; the dispatch still accepts them
  so an older saved workflow gets a message naming the replacement.
- **MODIFY mode** — edits the current canvas instead of building a new
  workflow. Returns a *diff* (nodes to add, nodes to remove, widget values to
  change), every part validated against the real canvas ids and the installed
  registry before it is applied. The canvas is snapshotted first, so
  **Undo Last Agent Action** reverts the whole edit.
- `/agent/health` reports what is actually working — Ollama reachability, the
  primary/fallback/embed models, index readiness and blueprint count — with
  the exact `ollama pull` command for anything missing. The node calls it on
  creation and shows the first blocking problem in its status widget, so a
  missing model is visible before a build fails rather than after.
- **CHAT mode** — free-text Q&A about ComfyUI, *this* installation, the user's
  own graph, and execution errors, rendered in a DOM panel on the node itself.
  Selecting `mode = CHAT` swaps the Build button for the chat panel.
  - `/agent/chat` streams tokens over the `comfy_agent.chat_token` WS event
    (routed by `chat_id`, like builds are by `build_id`) and also returns the
    full reply, so a client that missed the stream still renders.
  - `chat_context.py` builds the system prompt: a trimmed canvas snapshot
    (presentation keys dropped, summarised past `CHAT_MAX_GRAPH_CHARS`), the
    installed-model inventory, retrieved node schemas **plus every type on the
    canvas** (a question like "why is this broken?" may not rank the user's own
    nodes highly, but they are exactly what needs explaining), and the last
    execution error — captured automatically from ComfyUI's `execution_error`
    event, so "why did that fail?" needs no copy-paste.
  - `llm_client.chat_stream()` deliberately sends **no** `format` constraint —
    every other call here is schema-constrained, which is right for building a
    graph and exactly wrong for conversation.
  - Chat history is saved with the workflow via `onSerialize`, not through a
    widget value.
- **`install_index.py` — semantic index over the whole installation.** This is
  what lets CHAT answer about any installed node rather than a keyword-matched
  subset. The registry cannot simply be prompted: on this install `/object_info`
  is **~2.7 MB across 1767 node types (~680k tokens)** — roughly 10x past even
  `OLLAMA_CHAT_NUM_CTX`, and a sub-1B model degrades long before that ceiling.
  So every node is embedded once with `nomic-embed-text` and each question
  retrieves only its top-K matches (measured: a real prompt lands ~1300 tokens,
  about 2% of the chat window).
  - **Cached to disk**, keyed by a fingerprint of the sorted node-type names.
    The ~22s build happens once and is reused until a node pack is installed or
    removed. The cache is generated, ~18 MB, and gitignored.
  - **Degrades rather than fails**: if Ollama or the embed model is unavailable,
    retrieval falls back to keyword scoring. Worse lookup beats no chat.
  - Ranking blends three signals, each added to fix a failure observed against
    the real registry:
    - *Semantic cosine* finds paraphrases ("remove the background" →
      `RemoveBackground`).
    - *Exact-name bonus* — pure vector similarity ranked `KSamplerAdvanced`,
      `KSamplerProvider` and `KSampler (Efficient)` above **`KSampler` itself**
      for "what does KSampler do", because longer descriptive text scores
      better. An outright name mention now dominates.
    - *Stopwords* — the name bonus fired on filler verbs, so "how do I **make**
      faces look better?" ranked `MakeTrainingDataset` (1.035) above
      `FaceDetailer` (0.641) purely on "make". With stopwords excluded,
      `FaceDetailer` ranks **#1**.
  - Node names are CamelCase-split before embedding (`FaceDetailer` →
    `Face Detailer`): embedders tokenize CamelCase poorly, and node names carry
    most of the meaning here — `FaceDetailer`'s own description never says
    "face". Query plurals are also singularised, since users type "faces" while
    node names say `Face`.
  - **Model inventory** is read live from `/models`, filtered to weight
    extensions and with source-code folders denylisted. Unfiltered,
    `/models/custom_nodes` alone returned 1400 entries — `.editorconfig`,
    `.eslintrc.cjs`, GitHub issue templates — which crowded out the real
    checkpoints and invited the model to cite a linter config as a usable model.
- `llm_client.py` — async Ollama client. Sends `think: false` — without it,
  qwen3.5-family "thinking" models put the answer in Ollama's `thinking`
  field and leave `response` empty, which read as total generation failure
  until this was found via a raw `/api/generate` call. Falls back to parsing
  `thinking` defensively in case a future model ignores the flag.
- `knowledge/schema_compactor.py` — `/object_info` fetch + category
  filtering. Now actually enforces `config.PASS2_MAX_SCHEMA_TOKENS` (it was
  declared but never wired up) — on this install, the base category set is
  1700+ nodes and ~3300 tokens uncapped, well past what the 0.8B primary
  model can hold onto.
- `knowledge/type_validator.py`, `knowledge/blueprint_matcher.py` — type-safe
  link validation, keyword-based blueprint selection. Blueprint JSONs live in
  `blueprint_defs/` (renamed from `subgraphs/` — that name made ComfyUI's own
  frontend try to auto-load them as native "Global Subgraphs", a feature we
  don't use, producing 8 console errors on every load).
- `generation/classify.py` — Pass 1 classification. Always unions the
  model's answer with a fixed core category set (loaders/sampling/latent/
  conditioning/image) rather than trusting the classifier alone. Confirmed
  live: without this, the 0.8B model regularly classified a plain txt2img
  request as `['sampling']` only, which made `CheckpointLoaderSimple` and
  `CLIPTextEncode` structurally impossible for Stage A to ever select —
  they were never in the candidate pool it saw.
- `generation/stage_a_identify.py` — Node identification. Escalates to the
  fallback model when the primary returns a suspiciously sparse node list
  (<5) — needed because a small model can return *valid* JSON that's still
  an incomplete node set, which the error-based retry logic never catches.
- `generation/stage_b_configure.py` — Widget configuration. Now validates
  COMBO values against the node's real allowed options (previously
  `sampler_name: "SDXL"` — not a real option — passed straight through) and
  gives the model the actual option list / numeric range in-prompt instead
  of a bare widget name.
- `generation/stage_c_connect.py` — Connection generation. Previously never
  told the model each node's actual input/output names (the `schema` param
  was accepted but only used for post-hoc validation) — Stage C was
  connecting nodes blind. Now includes a concrete `id=... INPUT=[...]
  OUTPUT=[...]` block per instance plus a worked example, and escalates to
  the fallback model when the accepted-link count is implausibly low.
- `generation/stage_d_validate.py` — Now actually fails a multi-node result
  with zero links between any of them (previously this was collected as a
  warning string but never affected the `valid` flag, so a fully
  disconnected "workflow" could report `execution_ready: true`).
- `js/canvas_writer.js` — `findFreeZone` now sizes its overlap check to the
  actual spec footprint instead of a fixed 300×200 box (a multi-node
  blueprint could previously still land on top of existing nodes), and
  `injectWorkflow` pans the view to the result afterward — previously a
  successful build could place nodes entirely outside the current viewport
  with no indication anything happened.
- `blueprint_defs/*.json` — all 8 blueprints. Verified against this
  install's real `/object_info`: every node type resolves, every link's
  slot index and type checked against the real node definitions (zero
  mismatches after correcting the check itself to exclude widget-only
  inputs from slot numbering), and all 8 inject cleanly via the real
  `LiteGraph.createNode()` path with zero failed types and zero
  `undefined` nodes. None of the 8 nests a subgraph inside another,
  confirming they're unaffected by the ordering bug in issue #10522.

**Known limitation, not a bug:** free-form generation quality is bounded by
the 0.8B primary model's reasoning, exactly as Section 4A predicted. Even
after the fixes above, Stage C sometimes leaves one or two plausible links
(e.g. `VAEDecode → SaveImage`) unconnected on some runs — the escalation
logic catches the worst failures (0 nodes, 0 links, garbage widget values)
but doesn't guarantee a fully complete graph every time. Blueprint mode has
no such limitation since it's pre-authored, not generated.

**CHAT mode caveat:** retrieval has been verified against this install's real
1767-node registry (the three ranking failures above were found and fixed that
way), but answer *quality* is still bounded by the 0.8B model, which is tuned
for workflow generation rather than explanation. Its grounding is now accurate
— it cites real installed nodes and real model filenames — while its prose can
still be loose. Pointing `PRIMARY_MODEL` at `qwen3.5:4b` or `:9b` gives
noticeably better explanations at some VRAM cost.

What is explicitly not done: Phase 3's EXPLAIN/DEBUG/SUGGEST modes (CHAT
covers much of what EXPLAIN/DEBUG were meant to do, but the dedicated modes
still return "not implemented"), context auto-clear, Phase 4
(cross-platform/version testing).

MODIFY is implemented as a validated, undoable diff. It has been smoke-tested
against a live ComfyUI (1767 nodes) plus a live Ollama, which is how three
defects in it were found and fixed: widget values arriving wrapped in a
descriptor object (`{"i": 35}`) from every model size, the 0.8B proposing
wholesale deletion to change one value, and the 0.8B editing an unrelated
node with an unrelated value. **MODIFY now runs on the escalation model
first** — measured, `qwen3.5:4b` and `:9b` produce the correct minimal diff
where the 0.8B does not — so `ollama pull qwen3.5:4b` is effectively required
for MODIFY even though BUILD works without it.

Verified against that live install: the V3 node loads and registers
(`comfy_api.v0_0_3` does not exist yet, so the import ladder correctly
resolves `v0_0_2`), all 8 blueprints reference node types that are actually
present, the free-form pipeline builds and honestly reports unwired inputs,
`/agent/health` reports accurately, and the semantic index builds over 1767
nodes in ~21s and then re-uses its cache in ~1s. The JS layer (canvas
injection, Undo, `applyDiff`) is still covered only by its own test suite. The Week 10 quality re-baselining (VND/NodeComp/
LinkComp against a real 20-case golden set) also hasn't happened — what's been
done here is bug-fixing against observed failures, not the full metric suite.
Retrieval likewise has no formal recall benchmark; it was tuned against
hand-checked queries.

## Setup

### 1. Install the node

Clone or copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.
`aiohttp` is the only Python dependency and ComfyUI already ships it.

### 2. Install Ollama and pull the models

**Blueprint mode works without Ollama.** Ollama is required for free-form
generation (the four-stage pipeline used when no blueprint matches, or when
`use_blueprint` is off) and for CHAT mode.

Install Ollama from https://ollama.com, then pull the models named in
`config.py`:

```bash
# Fallback model — pull this first; free-form generation works with it alone.
ollama pull qwen3.5:9b

# Base model for the primary slot.
ollama pull qwen3.5:0.8b

# Embeddings for CHAT's install index (config.EMBED_MODEL).
# Without it, CHAT still works but retrieval degrades to keyword matching.
ollama pull nomic-embed-text
```

`config.PRIMARY_MODEL` defaults to **`qwen3.5-0.8b-comfyagent`**, a LoRA
fine-tune that is *not* published on the Ollama registry — `ollama pull` will
not find it. You have two options:

- **Use the base model instead** (recommended to start). Edit `config.py`:
  ```python
  PRIMARY_MODEL = "qwen3.5:0.8b"
  ```
- **Build the fine-tune yourself** from a Modelfile, then
  `ollama create qwen3.5-0.8b-comfyagent -f Modelfile`.

Verify what Ollama actually has:

```bash
ollama list
```

If `PRIMARY_MODEL` is missing, every free-form build fails at Stage A with an
Ollama "model not found" error. Blueprint mode is unaffected.

### 3. Using CHAT mode

Set the node's `mode` widget to **`CHAT`**. The Build button is replaced by a
chat panel: scrollback above, input below. Enter sends, Shift+Enter inserts a
newline, and the reply streams in token by token.

It answers from live install data rather than training memory, so questions
like these resolve against reality:

- *"What checkpoints do I have?"* — reads the actual model folders
- *"How do I improve faces?"* — finds `FaceDetailer` if Impact Pack is installed
- *"Why is this broken?"* — reads the current canvas
- *"Why did that fail?"* — the last execution error is attached automatically

**First question after a node-pack change takes ~20-30s** while the index
rebuilds; every question after that is fast. Watch the ComfyUI console for
`building install index: embedding N nodes...` if it seems to hang.

`Clear Chat` discards the thread. History is otherwise saved with the workflow.

### 4. Configuration

All tunable values live in `config.py` — model names, retry caps, token
budgets, timeouts, and the network/security settings described below. There
are no environment variables and no `.env` file; edit `config.py` directly.

**VRAM note for CHAT.** `OLLAMA_CHAT_NUM_CTX` is deliberately separate from the
BUILD pipeline's `OLLAMA_NUM_CTX`, and the KV cache grows linearly with it —
roughly 3-4 GB at 65536, on top of the model weights, competing with ComfyUI
for the same GPU. **If a large SDXL/Flux generation OOMs while chatting, lower
this first**; 32768 and 16384 are safe steps down. Because retrieval keeps real
prompts near ~1300 tokens, answers barely change — the headroom exists for long
conversations and large graphs, not for typical questions.

### 5. Run the tests

```bash
pytest tests/                          # 87 Python tests
node --test tests/canvas_writer.test.mjs tests/health_summary.test.mjs   # 35 JS tests
```

(Pass the `.mjs` file explicitly — a quoted `tests/*.test.mjs` glob is not
expanded by every shell and silently runs zero tests.)

Both suites run fully offline — the LLM, ComfyUI, and LiteGraph are stubbed.
Two extra checks need live services:

```bash
python tests/test_blueprint_widgets.py --live   # needs ComfyUI on :8188
```

## Troubleshooting

**Check the node's status line first.** On creation it calls `/agent/health`
and reports the first blocking problem it finds, with the command that fixes
it. Most of the table below is something the node will have already told you.

For the full picture, with ComfyUI running:

```bash
curl http://127.0.0.1:8188/agent/health
```

That returns Ollama reachability, which models are installed, whether the
semantic index is built, and a `hints` array naming each fix.

| Symptom | Cause | Fix |
|---|---|---|
| `Ollama is not reachable at http://127.0.0.1:11434` | Ollama is not running | `ollama serve`. Blueprint builds keep working without it — try `examples/01_build_blueprint.json` |
| Node says *Primary model missing* | No model pulled | `ollama pull qwen3.5:0.8b` |
| Builds succeed but wire poorly | Running on the 0.8B base model | `ollama pull qwen3.5:4b` (or `:9b`) — it is used automatically as the escalation model |
| `Stage A generation failed` | Model returned unusable JSON | Usually a too-small model. Pull a larger fallback, or use a prompt that matches a blueprint |
| First CHAT message hangs ~20s | Index building | Expected once per registry change. It now warms at startup, so this should be rare; the result is cached to disk |
| CHAT answers vaguely about your nodes | Embedding model missing | `ollama pull nomic-embed-text`. Without it retrieval falls back to keyword matching |
| Build reports *N link(s) not wired* | Model could not connect everything | The nodes are placed and highlighted in amber — finish the wiring by hand, or press **Undo Last Agent Action** and retry |
| `too many concurrent builds in progress` | Two builds at once | Wait for the first to finish (`MAX_CONCURRENT_BUILDS = 2`) |
| MODIFY says *the canvas is empty* | Nothing to modify | MODIFY edits an existing workflow. Use BUILD first |
| MODIFY makes the wrong edit | Only the 0.8B is installed | MODIFY runs on the escalation model by design; without one it falls back to the primary, which is measurably unreliable here. `ollama pull qwen3.5:4b`. Press **Undo Last Agent Action** and retry |
| Mode says it *has been replaced by CHAT* | Workflow saved before EXPLAIN/DEBUG/SUGGEST were retired | Set `mode = CHAT`, which covers all three |
| Node does not appear in ComfyUI at all | Import failed | Check the ComfyUI console for `comfy_agent`. A `no usable comfy_api version found` line means this ComfyUI is too old for the V3 schema |
| Everything is slow | Small model on CPU | Expected. Blueprint builds bypass the model entirely |

**A build that fails to run.** If a workflow the agent just built errors when
queued, the node reports it on its own status line and keeps the traceback.
Switch to CHAT and ask why — it already has the error in context.

**Undo.** Every BUILD and MODIFY snapshots the canvas first. **Undo Last Agent
Action** reverts the last one. It restores the whole canvas to that moment, so
manual edits made since are reverted too.

## Security notice

- All backend routes (`agent_routes.py`) reject non-localhost requests by
  default. To allow specific remote hosts, prefer
  `config.ALLOWED_NETWORK_CIDRS` (plus `config.AGENT_API_TOKEN`, sent as the
  `X-Agent-Token` header) over `config.ALLOW_NETWORK`, which disables every
  peer check at once.
- `WEB_DIRECTORY` (`js/`) is served statically by ComfyUI's frontend — this
  JS has full canvas access (it can create nodes, wire links, and read the
  entire graph). Do not load this extension against a ComfyUI instance
  exposed to an untrusted network.
- The agent's LLM backend (Ollama) is called at `http://localhost:11434`,
  hardcoded, not user-configurable by default.
- No credentials, API keys, or model file paths ever leave the machine.
- **CHAT sends more context than BUILD does.** Each question ships the current
  canvas, your installed-model *filenames*, retrieved node schemas, and the
  last execution error to the LLM. With the default local Ollama this never
  leaves the machine — but if you repoint `config.OLLAMA_HOST` at a remote or
  hosted endpoint, that inventory goes with it. `/agent/chat` enforces the same
  localhost/origin checks as `/agent/build_workflow`.
- `install_index.cache.json` is generated (~18 MB) and gitignored. It holds
  embeddings of node metadata only — no model weights, no user content.
- A generated workflow is drawn to the canvas but is never auto-queued for
  execution — the user must trigger execution themselves.
