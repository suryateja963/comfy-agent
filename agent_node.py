"""Agent meta-node — V3 schema. See MASTER_PLAN.md Section 3D for the slot spec.

Pinned to comfy_api.v0_0_2 (current, not yet stable — re-pin on v0_0_3, see
MASTER_PLAN.md Section 2B / 4A).
"""

from __future__ import annotations

from comfy_api.v0_0_2 import ComfyExtension, io


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
