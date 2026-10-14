# Development notes

Per-module implementation notes and the record of what has been verified
against a live ComfyUI + Ollama install. Moved out of the README, which is
now a user guide. For the design rationale and the full build log, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); for the original phased plan, see
[`../MASTER_PLAN.md`](../MASTER_PLAN.md).

## Status

Phase 0 ("Hello Canvas") + Phase 1 ("Blueprint Mode Works") + Phase 2
("Free-Form Mode Works") are implemented and **have been run end-to-end
against a live ComfyUI Desktop install + local Ollama**, including a full
bug-fixing pass driven by real failures observed in the browser and via
direct API calls — not just static review. See below for what was found and
fixed, and what remains a known limitation rather than a bug.

## Modules

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
  - Chat history, split into topics, is saved with the workflow via
    `onSerialize`, not through a widget value or `localStorage` (which is
    shared by every workflow and leaked one workflow's chats into another).
  - Prompt size is budgeted against `OLLAMA_CHAT_NUM_CTX`, counting the system
    prompt too. Past the budget, older turns are summarised into the system
    prompt and then the canvas is dropped; the request always ends on the
    user's question.
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

## Known limitations

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

Not done: context auto-clear and Phase 4 (cross-platform/version testing).
EXPLAIN/DEBUG/SUGGEST were retired in favour of CHAT; a workflow that still
sends one gets a message naming CHAT as the replacement.

## What has been verified

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
nodes in ~21s and then re-uses its cache in ~1s.

MODIFY was then exercised on a real 7-node canvas across widget edits, adds,
removals, a node replacement, and a three-part edit — all produced the
correct minimal diff on `qwen3.5:4b`. The JS layer's *logic* is covered by
its own suite against stubs, and `tests/test_frontend_api.py` checks the
LiteGraph APIs it calls still exist in the installed frontend; the rendered
UI itself has not been driven in a browser. The Week 10 quality re-baselining (VND/NodeComp/
LinkComp against a real 20-case golden set) also hasn't happened — what's been
done here is bug-fixing against observed failures, not the full metric suite.
Retrieval likewise has no formal recall benchmark; it was tuned against
hand-checked queries.
