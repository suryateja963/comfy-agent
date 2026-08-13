# Example workflows

Drag any of these onto the ComfyUI canvas (or **Workflow → Open**) to see the
node working. They are the fastest way to confirm the install is healthy.

Start with `01` — it is the only one that works with Ollama stopped.

| File | Mode | What it shows | Needs Ollama? |
|---|---|---|---|
| `01_build_blueprint.json` | BUILD | Blueprint match — instant, no LLM call | No |
| `02_build_freeform.json` | BUILD | The four-stage generation pipeline | Yes |
| `03_modify_existing.json` | MODIFY | Editing a workflow that already exists | Yes |
| `04_chat_about_workflow.json` | CHAT | Answering questions about *this* canvas | Yes |

## 01 — Build from a blueprint

One agent node on an empty canvas. Press **Build Workflow**.

The prompt (`a photorealistic portrait, sdxl`) matches the `txt2img_sdxl`
blueprint, so the node instantiates a pre-validated workflow with no model
call at all. If this works and `02` does not, the problem is Ollama, not the
node — check the node's own status line, which reports exactly what is
missing.

The generated workflow references `sd_xl_base_1.0.safetensors`. If you do not
have that checkpoint, the workflow still builds; pick one you do have from the
loader's dropdown before queueing.

## 02 — Build free-form

Same node with `use_blueprint` set to false and a prompt no blueprint covers,
which forces the full pipeline: classify → identify nodes → configure widgets
→ connect → validate.

This is the path that depends on model quality. A build that places nodes but
leaves some unwired is reported honestly — the node says how many connections
are missing and highlights them in amber rather than claiming success.

## 03 — Modify an existing workflow

A complete txt2img workflow plus an agent node in MODIFY mode. Press
**Modify Workflow** and it edits what is already on the canvas — adding,
removing, or changing widget values — instead of building something new.

Every proposed edit is checked against the real node ids and your installed
nodes before anything is applied. **Undo Last Agent Action** reverts the whole
edit, so this is safe to try.

## 04 — Chat about the canvas

The same workflow with the agent in CHAT mode. Ask it something concrete:

- *what does the KSampler here do?*
- *why would this produce a black image?*
- *what checkpoints do I have installed?*

CHAT reads this canvas, the nodes actually installed in your ComfyUI, and your
model files — so it answers about your machine rather than about ComfyUI in
general. The first question may pause while the node index builds; after that
it is cached.
