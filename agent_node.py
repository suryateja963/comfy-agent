"""Agent meta-node — V3 schema. See MASTER_PLAN.md Section 3D for the slot spec.

Prefers a *pinned* comfy_api version, newest first, and only falls back to
`comfy_api.latest` if no pinned version this file knows about is importable.
That order is deliberate (MASTER_PLAN.md Section 2B / 4A): `latest` tracks
whichever numbered API is still under development and can change without
warning, so it is a last resort rather than the default.

A bare `from comfy_api.v0_0_2 import ...` made the whole node unloadable the
day a ComfyUI release stops shipping that version — with an ImportError that
says nothing about which versions were tried.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("comfy_agent")

# Newest pinned version first; `latest` last. Extend the pinned entries as new
# versions are cut rather than switching the preference to `latest`.
_API_MODULES = ("comfy_api.v0_0_3", "comfy_api.v0_0_2", "comfy_api.latest")

ComfyExtension = None
io = None
_import_errors: list[str] = []

for _module_name in _API_MODULES:
    try:
        _module = __import__(_module_name, fromlist=["ComfyExtension", "io"])
        ComfyExtension = _module.ComfyExtension
        io = _module.io
    except (ImportError, AttributeError) as _exc:
        _import_errors.append(f"{_module_name}: {_exc}")
        continue
    if _module_name != _API_MODULES[0]:
        logger.info(
            "comfy_agent: using %s (preferred %s unavailable)",
            _module_name,
            _API_MODULES[0],
        )
    break
else:
    raise ImportError(
        "comfy_agent: no usable comfy_api version found. Tried: "
        + "; ".join(_import_errors)
    )

del _API_MODULES, _import_errors, _module_name, _module


class ComfyAgentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyAgent",
            display_name="Comfy Agent",
            category="agent",
            inputs=[
                io.String.Input("task_prompt", multiline=True, tooltip="Describe the workflow you want"),
                io.Combo.Input(
                    "mode",
                    options=["BUILD", "CHAT", "MODIFY", "EXPLAIN", "DEBUG", "SUGGEST"],
                    default="BUILD",
                    tooltip="BUILD writes a workflow to the canvas; CHAT opens a conversation panel",
                ),
                io.Combo.Input(
                    "task_category",
                    options=["auto", "txt2img", "img2img", "video", "upscale", "custom"],
                    default="auto",
                ),
                io.String.Input("model_family", default="auto", tooltip="sdxl / flux / sd15 / wan / auto"),
                io.Int.Input("max_nodes", default=12, min=4, max=20),
                io.Boolean.Input("use_blueprint", default=True),
                io.String.Input(
                    "context_input",
                    optional=True,
                    tooltip=(
                        "Pipe text from upstream node "
                        "(NOT YET IMPLEMENTED — accepted but ignored, see MASTER_PLAN.md Phase 3)"
                    ),
                ),
            ],
            outputs=[
                io.String.Output("status_message"),
                io.String.Output("error_detail"),
                io.String.Output("workflow_json"),
                io.Boolean.Output("execution_ready"),
            ],
        )

    @classmethod
    def execute(cls, task_prompt, mode, task_category, model_family, max_nodes, use_blueprint, context_input=None):
        # Phase 0 stub: this node is a meta-operator that writes to the canvas via
        # /agent/build_workflow (see agent_routes.py) and the JS frontend
        # (js/canvas_writer.js). It does not process image/latent data downstream.
        # execute() exists to satisfy the schema; real orchestration happens through
        # the async route + WebSocket progress channel, not the graph execution path.
        return io.NodeOutput("", "not implemented (Phase 0 stub)", "{}", False)


class ComfyAgentExtension(ComfyExtension):
    async def get_node_list(self):
        return [ComfyAgentNode]


async def comfy_entrypoint() -> ComfyAgentExtension:
    return ComfyAgentExtension()
