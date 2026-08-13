# Master Plan: ComfyUI AI Agent Custom Node
### A Comprehensive, Error-Free, Architect-Grade Blueprint

*(Revision note: primary LLM changed from qwen2.5:14b to qwen3.5:0.8b per user request — already installed locally. See Section 4A and the model-swap risk notes below for the implications of this change.)*

---

## SECTION 1 — EXECUTIVE VISION & PRECISE SYSTEM DEFINITION

### What This System Is — Defined With Absolute Precision

This is a meta-node: a first-class ComfyUI canvas object that reads natural language instructions and constructs executable ComfyUI workflows around itself in real time. It is not a sidebar. It is not an external tool. It lives on the canvas, executes as a meta-operator, and its output is new nodes and connections — not image data flowing downstream.

This distinction is the cornerstone of every architectural decision that follows.

### The Three Jobs This System Must Do Perfectly


```
JOB 1 — UNDERSTAND
  Comprehend the user's intent from natural language
  Map that intent to specific ComfyUI node operations
  Know exactly which nodes are installed right now

JOB 2 — REASON
  Determine which nodes are needed
  Determine how they connect (type-safe links)
  Determine what widget values to set
  Validate every decision before committing to canvas

JOB 3 — BUILD
  Draw nodes on the live canvas without overlap
  Wire connections using correct slot indices
  Package output as Subgraph Blueprints for reuse
  Report status to the user inline
```


### Why This Is Harder Than It Looks — The Research Baseline

Despite the powerful effects ComfyUI can achieve, workflows are always manually crafted, which can be expensive. Expert-level skills are required to figure out the appropriate nodes and parameters that can produce the desired results, and creating new workflows is laborious and time-consuming. This is exactly the gap the agent node fills.

However, workflow generation carries a lot of context, so context length must be carefully controlled — otherwise the process is easy to interrupt. It is recommended to frequently clear conversation context to control conversation length. This concern is *more* acute, not less, with the small primary model now chosen (see Section 4A) — small models degrade faster as context grows, so context hygiene becomes load-bearing rather than a nice-to-have.

The most current academic research confirms the difficulty. Given a user instruction, accurate adherence to the task description requires evaluation from both structural and semantic perspectives using four complementary metrics: Valid Node Diversity (VND), Node Completeness (NodeComp), Link Completeness (LinkComp), and Task Consistency (TaskCons). These four metrics will form your internal quality scoring system.

---

## SECTION 2 — TECHNICAL ARCHITECTURE (ALL LAYERS)

### 2A. The API Foundation — Async-First, Non-Negotiable

ComfyUI's API is built on aiohttp, serving on port 8188 by default, exposing HTTP routes and a single WebSocket channel that let clients submit workflows, manage the execution queue, introspect the node catalogue, and receive real-time execution events.

Your agent backend must operate fully async. Blocking that event loop is catastrophic. The correct pattern, confirmed by production ComfyUI extensions, is:

Using `async with aiohttp.ClientSession() as session:` with `async with session.post(...)` and `await response.json()` — never blocking synchronous `requests` calls inside a PromptServer route handler.

Understanding async/await patterns in both Python and JavaScript is essential. Async processing must not block the UI while providing detailed progress information — users should always know what's happening. Long-running operations should be broken into discrete steps with progress reported after each step, and cancellation capabilities should be implemented so users can abort operations taking too long.

### 2B. The Node API — V3 Schema Is the Only Forward Path

The ComfyUI V3 schema introduces a more organized way of defining nodes, and future extensions to node features will only be added to the V3 schema. This guide helps migrate existing V1 nodes to the new V3 schema. The V3 schema is kept on the new versioned Comfy API, meaning future revisions will be backwards compatible. `comfy_api.latest` points to the latest numbered API still under development.

**Corrected versioning note (verified against current docs.comfy.org):** `v0_0_2` is the *current and first* API version, and it is **not yet stable** — the docs explicitly warn "more changes will be made to it without warning." There is no stable predecessor yet; `comfy_api.latest` currently just points to v0_0_2 itself. Pinning to `comfy_api.v0_0_2` today means accepting that the API can change under you without notice until a `v0_0_3` is cut and v0_0_2 is retroactively considered stable. Build the import as a pinned, swappable reference — `from comfy_api.v0_0_2 import ComfyAPI` — with async-ready API calls like `await api.set_progress(0.5)`, and revisit the pin the moment v0_0_3 ships.

Currently, even minor updates to ComfyUI can disrupt custom nodes due to the absence of a clear separation between internal and public APIs. Nodes v3 introduces a stable public API with guarantees for backward compatibility. This is the insurance policy for your node's longevity. While dynamic inputs/outputs and custom widgets are possible today, their implementation remains fragile. Nodes v3 aims to formalize these features with first-class support and a more intuitive schema.

### 2C. The Subgraph Blueprint System — The Strategic Architecture Choice

This is the most important architectural discovery from the research phase. Your agent should not build flat node lists. It should build Subgraph Blueprints — the most powerful organizational primitive in modern ComfyUI.

The Blueprints and Subgraph system provides a mechanism for defining, loading, and serving pre-built workflow templates and reusable nested graph structures. Blueprints are stored as JSON files and can be served to the frontend as "Global Subgraphs," allowing complex node logic to be encapsulated and reused. The SubgraphManager class is the central authority for discovering, caching, and serving subgraph definitions from multiple sources.

Subgraph blueprints allow users to quickly add pre-built node groups from the subgraph menu. As a node developer, all you have to do is create a `subgraphs` folder in your custom node directory and place `.json` files there (confirmed current: ComfyUI scans all custom node directories for subgraph files and serves them via the `/global_subgraphs` API endpoint; the JSON format is identical to standard workflow JSON — build the composition in-app, convert to subgraph, export, and drop the file in).

The subgraph system implements a hierarchical graph structure where a Subgraph is instantiated as a SubgraphNode within a parent graph, establishing a clear containment and boundary relationship, with dedicated input/output nodes at the subgraph edges.

**Critical stability note (verified against GitHub issue #10522):** Subgraphs are a new feature and have known bugs. The specific, confirmed failure mode is in **nested** subgraphs — a subgraph that references another subgraph. If the referencing subgraph appears earlier in the JSON than the subgraph it depends on, the reference fails to resolve, and the node renders as a red "undefined" box after a browser refresh. The fix (per the reporter) is to load all subgraphs into a dictionary before dereferencing IDs, rather than resolving references in file order. Since several of the 8 shipped blueprints below (face_detailer, controlnet_base, lora_stack) are plausible candidates for being composed from smaller sub-templates, explicitly test whether any of your 8 blueprints nest a subgraph inside another during Week 3 authoring — if none do, this risk doesn't apply to you yet, but confirm it rather than assume it. Plan all blueprint loading with graceful fallback to flat node injection regardless.

Also noted: "Failed to load subgraph blueprints — Error: Loaded subgraph blueprint does not contain valid subgraph" is a live reported error class distinct from the nesting bug — treat blueprint load as a operation that can fail outright, not just misbehave.

### 2D. The Knowledge Brain — Multi-Pass Retrieval Architecture

The agent's intelligence must be live and retrieved, never baked in. Node documentation provides complete documentation for the 3205 nodes registered in ComfyUI as of the last public count (September 2024) — the ecosystem has grown substantially since, so this must never be treated as a fixed number; always query live.

**The Two-Pass Context Strategy:**


```
PASS 1 (Fast — Task Classification)
  Input: User's raw natural language prompt
  LLM call: "Which of these categories does this task involve?"
  Output: Filtered category list (e.g. ["sampling", "loaders", "latent"])
  Cost: ~200 tokens. Very fast.

PASS 2 (Full — Schema-Aware Generation)
  Input: Filtered /object_info + user prompt + task type
  LLM call: Full workflow spec generation
  Schema-constrained output: enforced JSON structure
  Cost: ~2000-8000 tokens. Model-dependent.
```


This solves the context window overflow problem without losing coverage. With the 0.8B primary model, keep the Pass 2 context budget toward the low end of that range — small models lose coherence faster as filtered schema context grows, so aggressive category filtering in Pass 1 matters more here than it would with a 14B model.

### 2E. The LLM Generation Layer — Four-Stage Pipeline (Not One-Shot)

Workflows are converted into equivalent code so that LLMs can better understand them. The agent can learn from existing workflows and autonomously design new ones.

Based on academic research into ComfyUI workflow generation, single-pass generation fails at complexity. A central LLM-based assistant agent can either respond directly to user instructions based on conversation history (short-term memory), or collaborate with specialized worker agents, supported by curated ComfyUI knowledge bases.

Your generation pipeline must be:


```
STAGE A — Node Identification
  LLM outputs ONLY node class_type names needed
  System validates every name against /object_info keys
  Reject unknown names BEFORE proceeding

STAGE B — Widget Configuration
  LLM sets widget values for each validated node
  System checks value types against /object_info widget specs
  Type coerce: string → int/float/bool where needed

STAGE C — Connection Generation
  LLM generates only links between already-validated nodes
  System checks: does from_output name exist on that node?
  System checks: does to_input name exist on that node?
  System checks: do the type strings match exactly (case-normalized)?

STAGE D — Canvas Validation
  JS side: LiteGraph.createNode(class_type) — does it return non-null?
  Check existing canvas for overlap before placement
  Dry-run all connections before committing
  Wrap in SubgraphNode if node count > 6
```


**Small-model adaptation:** with qwen3.5:0.8b as primary, keep each stage's prompt and expected output as narrow and single-purpose as possible — e.g. Stage A should request *only* a JSON array of class_type strings, nothing else. Lean on Ollama's `format` grammar constraint at every stage, not just the final one. Small models do meaningfully better with tightly scoped, single-responsibility prompts than with one large schema-constrained blob, so the four-stage split — already in this design — is doing more load-bearing work than it would with a larger model.

### 2F. The Canvas Writer — Overlap-Aware, Type-Safe Injection

The combination of Python's processing power and modern JavaScript frontend capabilities creates opportunities for innovation. Custom nodes can now provide experiences that rival dedicated applications while remaining integrated with ComfyUI's powerful workflow system.

The canvas writer must handle:

| Operation | Implementation Detail |
|---|---|
| Node placement | Read `app.graph._nodes` first — map all occupied regions, place in first free zone |
| Widget assignment | Positional fallback when `widget.name` lookup fails |
| Link creation | `fromNode.connect(outSlotIndex, toNode, inSlotIndex)` — indices from validated schema |
| Type normalization | Uppercase all type strings before comparison — "model" → "MODEL" |
| Error isolation | Wrap `createNode()` in try/catch — report failed types without crashing the whole build |
| Canvas refresh | Always call `app.graph.setDirtyCanvas(true, true)` after all operations |
| Subgraph wrapping | For workflows > 6 nodes: convert to SubgraphNode automatically |

### 2G. The Feedback Channel — WebSocket Progress Streaming

The agent must stream its thinking to the user in real time, not just return a final result after 60+ seconds of silence. ComfyUI's PromptServer supports WebSocket messaging — use it.


```
Generation Stage Events to Stream:
  → "Fetching node registry..."       (object_info call)
  → "Classifying task type..."        (Pass 1 LLM)
  → "Identifying nodes..."            (Stage A)
  → "Validating X of Y nodes..."      (Schema check)
  → "Generating connections..."       (Stage C)
  → "Drawing on canvas..."            (JS injection)
  → "Done — built N nodes, M links"   (Complete)
  → "Error in Stage X: [detail]"      (Fail-explicit)
```


---

## SECTION 3 — KEY FUNCTIONALITIES (COMPLETE SPECIFICATION)

### 3A. Five Operating Modes

| Mode | Trigger | Description |
|---|---|---|
| BUILD | Default | Create full workflow from natural language on blank canvas region |
| MODIFY | "change / add / remove" keywords | Read existing canvas state, surgically alter it |
| EXPLAIN | "what does this do / explain" | Read selected nodes, return human-readable explanation |
| DEBUG | "fix / broken / error" | Scan canvas for type mismatches, broken links, missing nodes |
| SUGGEST | "what should I add" | Analyze current workflow's open output slots, recommend next nodes |

> **AS-BUILT AMENDMENT — a sixth mode, `CHAT`.** The mode-per-intent design
> above assumes the agent can route a request by keyword. In practice the
> EXPLAIN/DEBUG/SUGGEST distinctions collapsed: a user asking "why is my image
> black?" wants explanation, diagnosis and a suggestion in one answer, and
> forcing them to pick a mode first is friction with no payoff. `CHAT` was
> implemented instead — one conversational panel in the node body that covers
> that whole space, keeps multi-turn history, and reads the canvas plus the
> last execution error automatically. The three original modes remain wired
> into the dispatch but unimplemented. See ARCHITECTURE.md §11.

### 3B. Knowledge Sources (Priority Order)


```
1. Live /object_info              ← Ground truth for installed nodes RIGHT NOW
2. Shipped Blueprint Library      ← Pre-validated subgraph templates
3. ComfyUI Built-in Help Pages    ← ComfyUI now features built-in help pages
4. Conversation History           ← Multi-turn correction context
5. Task Classification Context    ← Category-filtered schema subset
```


ComfyUI now features built-in help pages directly accessible within the interface to simplify node understanding and usage. These help pages are manually curated and available in multiple languages. Your agent's knowledge layer can reference these pages as supplementary semantic context for nodes that `/object_info` only describes syntactically.

**With a small primary model, weight source #2 (the blueprint library) more heavily than the priority order above might suggest** — see the Section 4A model-swap risk notes. Default `use_blueprint` to `true` and treat free-form generation as the exception path, not a coequal mode.

> **AS-BUILT AMENDMENT — two sources added, one not yet used.** The plan's
> "category-filtered schema subset" (#5) is sufficient for BUILD, where the
> task category is known up front. It is *not* sufficient for CHAT, where a
> question may concern any of 1767 node types in any phrasing. Two sources
> were added:
>
> ```
> 6. Semantic index over ALL installed nodes  ← install_index.py, embedded once
> 7. Installed model inventory (/models)      ← real checkpoints/LoRAs/VAEs on disk
> ```
>
> Source #6 exists because the registry cannot be prompted directly: at ~2.7 MB
> / ~680k tokens it is roughly 10x the largest usable context window here, and
> raising `num_ctx` cannot close that gap (the KV cache alone would exceed the
> GPU). Source #7 is what lets the agent answer "what checkpoints do I have?"
> from disk rather than from model memory.
>
> Source #3 (ComfyUI's built-in help pages) is still **not** used — node
> `description` fields from `/object_info` are harvested into the index
> instead, which covers much of the same semantic ground without depending on
> an undocumented internal format.

### 3C. Blueprint Library (Ship With the Node)

Pre-build and validate these subgraph blueprints before release. Each lives in `custom_nodes/comfy_agent/subgraphs/`:


```
comfy_agent/subgraphs/
  txt2img_sdxl.json          ← CheckpointLoader → CLIP → KSampler → VAE → Save
  txt2img_flux.json          ← Flux-specific node chain
  img2img_base.json          ← With image input + denoise control
  upscale_latent.json        ← Latent upscale + Hi-Res fix
  face_detailer.json         ← Face region detect + inpaint
  video_wan.json             ← Wan2.x video generation pipeline
  controlnet_base.json       ← ControlNet preprocessor + apply chain
  lora_stack.json            ← Multi-LoRA loader + apply stack
```


These are the fast path. When user intent matches a template, instantiate and customize — do not generate from scratch. This drops hallucination risk substantially and, with the small primary model, is now the *primary* reliability mechanism rather than an optimization.

### 3D. Input/Output Slot Design


```python
# Agent Node Slot Specification (V3 Schema)
INPUTS (Widgets):
  task_prompt     → MULTILINE STRING   "Describe the workflow you want"
  mode            → COMBO              [BUILD, MODIFY, EXPLAIN, DEBUG, SUGGEST]
  task_category   → COMBO              [auto, txt2img, img2img, video, upscale, custom]
  model_family    → STRING             "sdxl / flux / sd15 / wan / auto"
  max_nodes       → INT (4–20)         Complexity ceiling for generation
  use_blueprint   → BOOLEAN            Prefer pre-validated templates when available (default TRUE)
  context_input   → STRING (optional)  Pipe text from upstream node

OUTPUTS:
  status_message  → STRING             "Built N nodes, M connections"
  error_detail    → STRING             Error description if build fails
  workflow_json   → STRING             The generated spec (for logging/chaining)
  execution_ready → BOOLEAN            True if generated workflow passed all validation
```


---

## SECTION 4 — TECHNICAL SPECIFICATION STACK

### 4A. Full Technology Decisions

| Layer | Technology | Version / Spec | Rationale |
|---|---|---|---|
| Node Schema | ComfyUI V3 API | `comfy_api.v0_0_2` pinned (not yet stable — re-pin on v0_0_3) | Only forward-compatible path |
| Backend Language | Python | 3.10+ (match ComfyUI requirement) | Mandatory |
| Async HTTP Client | `aiohttp.ClientSession` | Latest with ComfyUI | Non-blocking Ollama calls |
| Frontend Language | JavaScript (ES Modules) | Modern — `import { app }` pattern | ComfyUI standard |
| Frontend Framework | Vanilla JS + LiteGraph API | ComfyUI fork (not external litegraph) | Use ComfyUI's own fork only |
| LLM Backend | Ollama | Latest stable | Local-first, format schema support |
| **Primary Model** | **qwen3.5:0.8b** | **0.8B parameter** | **Already installed locally; fast inference, low resource footprint — see risk notes below** |
| Fallback Model | llama3.1:8b | 8B parameter | Escalation when small-model output fails schema validation |
| JSON Constraint | Ollama `format` parameter | Full JSON Schema object | Grammar-enforced, not hinted — more important than ever with a small primary model |
| WebSocket | ComfyUI PromptServer WS | Built-in | Progress streaming to frontend |
| Blueprint Storage | JSON files | `custom_nodes/comfy_agent/subgraphs/*.json` | Native SubgraphManager discovery |
| Dependencies | Zero new pip deps | aiohttp already in ComfyUI env | No dependency conflict risk |

**Model-swap risk notes (qwen2.5:14b → qwen3.5:0.8b as primary):**

- **Blueprint reliance goes up.** At 0.8B parameters, expect materially lower success rates on free-form generation than the plan's original 70% target for 5–12 node workflows. `use_blueprint` now defaults to `true` in the slot spec above, and free-form generation should be treated as the exception path, not a coequal mode.
- **Prompt scope must shrink.** Each of the four generation stages should ask the model for the narrowest possible output (e.g. Stage A: a bare JSON array of class_type strings, nothing else). Combined with aggressive `format`-based grammar constraints at every stage, this is the main lever for keeping a sub-1B model's output usable.
- **Quality thresholds are provisional, not gates.** The VND/NodeComp/LinkComp minimums in Section 6 were calibrated with a 14B model in mind. Re-measure against qwen3.5:0.8b in Week 10 before treating those numbers as pass/fail gates — expect them to land lower on genuinely correct small-model output, and adjust the bar based on real measurement rather than holding the original target as sacred.
- **Retry cost changes.** A 0.8B model is cheap enough that the existing 3-retry cap per stage is now nearly free in wall-clock time — consider whether escalating to the llama3.1:8b fallback after just 1 failed retry (rather than 3) is a better trade of latency vs. reliability, and tune this empirically in Phase 2.
- **Context-window discipline matters more.** Small models degrade faster as filtered `/object_info` context grows in Pass 2. Keep category filtering in Pass 1 aggressive, and consider capping the Pass 2 schema payload well below the "2000–8000 tokens" range noted in Section 2D, biasing toward the low end.
### 4B. Package Structure (Final)


```
ComfyUI/custom_nodes/comfy_agent/
│
├── __init__.py                    ← V3 schema registration + route setup
├── agent_node.py                  ← Node class definition (V3 schema)
├── agent_routes.py                ← Async aiohttp route handlers
├── knowledge/
│   ├── schema_compactor.py        ← /object_info filtering + compression
│   ├── type_validator.py          ← Type-safe link validation
│   └── blueprint_matcher.py       ← Template selection logic
├── generation/
│   ├── stage_a_identify.py        ← Node identification LLM call
│   ├── stage_b_configure.py       ← Widget value LLM call
│   ├── stage_c_connect.py         ← Link generation LLM call
│   └── stage_d_validate.py        ← Full spec validation
├── subgraphs/                     ← Blueprint library (JSON files)
│   ├── txt2img_sdxl.json
│   ├── txt2img_flux.json
│   └── ... (8 base templates)
├── js/
│   ├── agent_node_ui.js           ← Node widget rendering
│   ├── canvas_writer.js           ← LiteGraph injection logic
│   ├── progress_listener.js       ← WebSocket progress display
│   └── agent_style.css            ← Styling
└── README.md                      ← Security notice: localhost-only
```

> **AS-BUILT AMENDMENT — the shipped tree differs in three ways.** See
> ARCHITECTURE.md §4 for the current structure.
>
> - **`subgraphs/` → `blueprint_defs/`.** The planned name made ComfyUI's own
>   frontend try to auto-load the blueprints as native "Global Subgraphs" — a
>   feature this node does not use — producing 8 console errors on every page
>   load. The directory name itself was the trigger.
> - **Added `chat_context.py` and `install_index.py`** at package root, plus
>   `js/chat_panel.js`, for CHAT mode and its retrieval layer.
> - **Added `generation/classify.py` and `generation/pipeline.py`** — the plan
>   listed the four stages but not the Pass-1 classifier or the orchestrator
>   that sequences them.
>
> `install_index.cache.json` also appears at runtime (~18 MB, gitignored).

### 4C. Security Specification

- All backend routes validate origin IP — reject non-localhost requests by default
- WEB_DIRECTORY assets are served statically — document this in README with explicit note that JS has canvas access
- No credentials, API keys, or model paths leave the machine
- Ollama call is always to `http://localhost:11434` — hardcoded, not user-configurable by default
- Configurable opt-in (`allow_network: true` in a config file) for network deployments — off by default
---

## SECTION 5 — IMPLEMENTATION TIMELINE

### Phase 0 — Foundation (Weeks 1–2)
**Milestone: "Hello Canvas"** — A node that exists on canvas and draws one node


```
Week 1:
  □ Package structure created
  □ V3 schema node class registered successfully
  □ async aiohttp route `/agent/ping` returns 200
  □ JS extension loads — no console errors
  □ `app.graph._nodes` read successfully from JS
  □ Friendly error state if Ollama is not running locally (no silent hang)

Week 2:
  □ /object_info fetched async — no server block
  □ schema_compactor.py tested: input 3000 nodes → output 80 filtered
  □ LiteGraph.createNode() tested for 10 known node types
  □ One hardcoded node drawn on canvas at correct position
  □ One hardcoded link wired correctly
  □ Subgraph wrapper tested — 5 nodes → 1 SubgraphNode
```

Gate: All items checked before Phase 1 begins.

### Phase 1 — Blueprint Fast Path (Weeks 3–5)
**Milestone: "Blueprint Mode Works"** — Agent selects and instantiates pre-validated templates


```
Week 3:
  □ All 8 blueprint JSON files created and validated manually
  □ Explicitly test whether any blueprint nests a subgraph inside another —
    if so, stress-test against the known ordering bug in issue #10522
  □ SubgraphManager discovery confirmed — blueprints appear in ComfyUI menu
  □ blueprint_matcher.py: keyword → template selection logic
  □ Blueprint instantiation via SubgraphNode API tested

Week 4:
  □ Task classification LLM call (Pass 1) implemented async, on qwen3.5:0.8b
  □ Blueprint selection based on classification result
  □ Widget customization from user prompt (model name, prompt text)
  □ WebSocket progress streaming to JS frontend

Week 5:
  □ Full blueprint path end-to-end tested: "make a txt2img workflow"
  □ Error handling: blueprint load failure → flat node fallback
  □ Subgraph buggy-load guard implemented (try/catch on blueprint load)
  □ User-facing status messages all working
```

Gate: Blueprint path works for all 8 templates. Success rate > 95% on those templates.

### Phase 2 — Four-Stage LLM Generation (Weeks 6–10)
**Milestone: "Free-Form Mode Works"** — Agent generates novel workflows from scratch


```
Week 6:
  □ Stage A (Node Identification) implemented + tested in isolation on qwen3.5:0.8b
  □ Schema validation against /object_info keys
  □ Unknown node type rejection with helpful error message
  □ Retry loop with hard cap (max 3 retries) — measure whether 1-retry-then-escalate
    to llama3.1:8b beats 3-retries-then-fail on the small model

Week 7:
  □ Stage B (Widget Configuration) implemented
  □ Type coercion logic (string → int/float/bool)
  □ Positional widget fallback when name lookup fails

Week 8:
  □ Stage C (Connection Generation) implemented
  □ Type string normalization (uppercase)
  □ Slot name validation against /object_info output/input names
  □ Type compatibility check before accepting any link

Week 9:
  □ Stage D (Canvas Validation) — JS-side dry-run
  □ Overlap detection reading app.graph._nodes positions
  □ Grid layout with topology-aware column ordering

Week 10:
  □ Full four-stage pipeline end-to-end tested
  □ 20 diverse workflow requests tested, results scored on VND/NodeComp/LinkComp/TaskCons
  □ Re-baseline quality thresholds against actual qwen3.5:0.8b output (originals were
    calibrated for a 14B model — treat as provisional until this measurement)
  □ Failure modes documented, retry/escalation strategy tuned
```

Gate: Free-form generation success rate measured and documented; target adjusted based on real small-model performance rather than held at the original 70% figure without re-validation.

### Phase 3 — Canvas Awareness & Multi-Mode (Weeks 11–14)
**Milestone: "Full Agent"** — All five modes working, canvas-aware


```
Week 11:
  □ Canvas state reader: serialize existing nodes to context
  □ MODIFY mode: diff existing canvas against desired state
  □ Surgical node add/remove without disrupting existing links

Week 12:
  □ EXPLAIN mode: read selected nodes → natural language description
  □ DEBUG mode: type mismatch scanner, broken link detector
  □ SUGGEST mode: open output slot analyzer → next node recommendation

Week 13:
  □ Multi-turn conversation history management
  □ Context window management: auto-clear when approaching limit (more urgent with
    a small model — degrades faster as context grows)
  □ Session persistence across ComfyUI restarts

Week 14:
  □ Full five-mode end-to-end testing
  □ Performance profiling — total latency from prompt to drawn canvas
  □ Memory leak check on JS side (event listener cleanup)
  □ Security audit: localhost-only confirmed, WEB_DIRECTORY reviewed
```

Gate: All five modes functional. Latency target re-measured against qwen3.5:0.8b's actual throughput rather than the original 45s/14B assumption.

> **AS-BUILT AMENDMENT — Phase 3 partially delivered, via `CHAT` rather than
> the five modes.** What actually shipped from the list above:
>
> | Week 11-13 item | Status |
> |---|---|
> | Canvas state reader: serialize existing nodes to context | **Done** — `chat_context.render_graph_block()`, trimmed and budgeted |
> | Multi-turn conversation history management | **Done** — per-node, persisted with the workflow via `onSerialize` |
> | Session persistence across ComfyUI restarts | **Done** — history rides in the saved workflow JSON |
> | EXPLAIN / DEBUG / SUGGEST as separate modes | **Superseded** by `CHAT` (see §3A amendment) |
> | MODIFY mode: diff canvas, surgical add/remove | **Not done** — still the main outstanding Phase 3 work |
> | Context auto-clear near the limit | **Not done** — retrieval keeps real prompts near ~1300 tokens of a 65536 window, so the pressure this was meant to relieve has not materialised |
> | JS memory-leak check (listener cleanup) | **Not done** — CHAT adds DOM widgets and WS listeners, so this is now more relevant than when written |
>
> The security audit item was partially addressed: `/agent/chat` enforces the
> same localhost + origin checks as `/agent/build_workflow`, and CHAT never
> writes to the canvas — it reads the graph but cannot modify a workflow.

### Phase 4 — Production Hardening (Weeks 15–16)
**Milestone: "Publishable"** — Ready for ComfyUI Manager listing


```
Week 15:
  □ Node 2.0 / Nodes v3 compatibility review — re-check comfy_api pin (v0_0_2 vs v0_0_3)
  □ Large canvas performance test (50+ existing nodes)
  □ Cross-platform test: Windows / Linux / macOS
  □ ComfyUI version compatibility: test on last 3 major versions

Week 16:
  □ README with full security notice written
  □ ComfyUI Manager pyproject.toml metadata prepared
  □ License file added (MIT recommended)
  □ Example workflow JSON files included
  □ Final regression suite: all 8 blueprints + 20 free-form cases pass
```


Consider reserving buffer time (e.g. one week in Phase 3 or 4) for absorbing an upstream ComfyUI frontend/backend change mid-project, since the plan itself notes minor ComfyUI updates can disrupt custom nodes without warning.

---

## SECTION 6 — TESTING STRATEGY

### Testing Pyramid


```
                    ┌───────────────┐
                    │  E2E Tests    │  ← Full workflow generation on live ComfyUI
                    │  (10 cases)   │
                  ┌─┴───────────────┴─┐
                  │ Integration Tests  │  ← Route → LLM → Spec → Canvas
                  │    (30 cases)      │
                ┌─┴────────────────────┴─┐
                │     Unit Tests          │  ← Each stage, each validator
                │     (100+ cases)        │
              ┌─┴──────────────────────────┴─┐
              │   Static Analysis / Linting   │  ← pyflakes, eslint, type checking
              └───────────────────────────────┘
```


### Unit Test Targets

| Module | Tests |
|---|---|
| schema_compactor.py | Input: 3000-node /object_info. Output size, category accuracy, malformed node skip |
| type_validator.py | All type strings normalized to uppercase. Mismatch detection. |
| blueprint_matcher.py | 50 different prompts → correct template selected vs not |
| stage_a_identify.py | Unknown node type rejected. Known node type accepted. |
| stage_c_connect.py | Valid link accepted. Type mismatch rejected. Missing node rejected. |
| canvas_writer.js | createNode null → caught. Overlap detection. Widget positional fallback. |

### Integration Test Targets

| Scenario | Expected |
|---|---|
| `/agent/build_workflow` called while Ollama is offline | 503 returned within 5s, no server block |
| `/object_info` returns malformed node entry | Skipped silently, rest of schema used |
| LLM generates unknown class_type | Rejected with valid_hint list in response |
| Blueprint file is malformed JSON | Fallback to flat node injection |
| 50+ nodes already on canvas | New nodes placed in empty zone, no overlap |
| User sends 3 correction requests in a row | History maintained, context managed |

### Four-Metric Quality Scoring (From Academic Research)

Generated workflows should be evaluated from both structural and semantic perspectives using four complementary metrics: Valid Node Diversity (VND), Node Completeness (NodeComp), Link Completeness (LinkComp), and Task Consistency (TaskCons).

Implement these as automated test scorers against a golden set of 20 known-correct workflows:


```
VND        → % of generated node types that are valid and distinct
NodeComp   → % of expected nodes present in generated spec
LinkComp   → % of expected links correctly generated
TaskCons   → Does the workflow actually accomplish the stated task? (manual review)

ORIGINAL TARGET SCORES (calibrated for qwen2.5:14b — treat as provisional
pending Week 10 re-measurement against qwen3.5:0.8b, per Section 4A):
  Blueprint path:     VND > 0.98, NodeComp > 0.97, LinkComp > 0.95
  Free-form path:     VND > 0.90, NodeComp > 0.80, LinkComp > 0.75
```


### Regression Suite — Critical Cases


```
"Make a basic txt2img workflow"                → Blueprint path. 6 nodes. All links valid.
"txt2img with FLUX and LoRA"                   → Blueprint + modification. LoRA node added.
"img2img with ControlNet depth map"            → Multi-category. Cross-category filter test.
"Fix the broken connection in my workflow"     → DEBUG mode. Finds type mismatch.
"Explain what the KSampler does"               → EXPLAIN mode. Accurate description.
"Add an upscaler after the KSampler"           → MODIFY mode. Canvas-aware injection.
"Build a face swap workflow"                   → Complex. Tests context window management.
[Malformed /object_info]                       → Partial schema, graceful degradation.
[Ollama timeout]                               → Error returned, no freeze.
[50 nodes on canvas]                            → New nodes placed without overlap.
```


### Performance Baselines

| Metric | Target | Measurement Method |
|---|---|---|
| Blueprint path latency | < 15 seconds end-to-end | Timed from button press to canvas drawn |
| Free-form path latency | Re-measure against qwen3.5:0.8b; original 60s target assumed a 14B model | qwen3.5:0.8b on local hardware |
| /object_info fetch | < 5 seconds | Async with 5s timeout |
| JS canvas injection | < 500ms | Measured in browser DevTools |
| Memory (JS side) | No growth over 10 agent calls | Chrome DevTools heap snapshot |
| Server blocking | Zero — event loop never blocked | aiohttp concurrent request test |

---

## MASTER CHECKLIST — BEFORE ANY CODE IS WRITTEN


```
ARCHITECTURE DECISIONS (Lock these in planning):
  □ V3 Schema confirmed as foundation — no V1 node class
  □ Async-only backend — no synchronous requests calls anywhere
  □ Blueprint-first generation — LLM scratch generation is the fallback
    (weighted more heavily now given the small primary model — default
    use_blueprint = true)
  □ Four-stage generation — no single-pass LLM calls for full workflows
  □ Two-pass context — classification first, schema second
  □ ComfyUI's litegraph fork as sole JS reference — no external litegraph docs
  □ Subgraph wrapping for output > 6 nodes — flat injection for ≤ 6
  □ Hard retry cap at 3 — return best partial on failure, not empty error
    (re-tune vs. faster escalation to llama3.1:8b fallback, see 4A)
  □ localhost-only security model — documented in README before publish

RISK MITIGATIONS (Address before Phase 1):
  □ Blueprint fallback path tested before free-form path built
  □ /object_info malformed node skip logic unit-tested
  □ Subgraph load failure guard implemented before blueprint library shipped
  □ Explicitly checked whether any of the 8 shipped blueprints nest a subgraph
    inside another (known ordering bug in issue #10522)
  □ Overlap detection working before any multi-node generation released
  □ WebSocket progress streaming working before latency-heavy LLM calls enabled
  □ Friendly Ollama-offline error state built in Phase 0, not deferred to Phase 2

KNOWN ECOSYSTEM RISKS (Monitor throughout):
  □ comfy_api v0_0_2 is not yet stable — re-pin import when v0_0_3 goes stable
  □ LiteGraph fork changes — check ComfyUI frontend changelog on each update
  □ Subgraph system bugs — track github.com/Comfy-Org/ComfyUI issue #10522
  □ /object_info V1/V3 dual format — parser handles both, always
  □ Small-model (0.8B) reliability on free-form generation — re-validate
    quality thresholds against real measurements, don't hold 14B-era targets
    as fixed gates
```
