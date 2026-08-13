"""MODIFY — edit the graph that is already on the canvas.

Unlike BUILD, which emits a whole workflow, this emits a *diff*: nodes to add,
nodes to remove, and widget values to change. A diff rather than a regenerated
workflow for two reasons:

* The user asked to change one thing. Regenerating everything re-rolls parts
  they were happy with, on a model whose whole reliability strategy is being
  asked narrow questions (Section 4A).
* A diff is reviewable and reversible. "Remove node 7, add an UpscaleModel"
  can be reported to the user and undone; a wholesale replacement cannot.

Every element of the diff is validated against the real canvas and the real
/object_info before it reaches the frontend: ids must exist, class types must
be installed, and widget names must exist on the node being changed. The LLM's
output is a proposal, never applied directly.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import llm_client
from ..knowledge.type_validator import is_widget_input

logger = logging.getLogger("comfy_agent.generation.stage_m")

SCHEMA = {
    "type": "object",
    "properties": {
        "add_nodes": {"type": "array", "items": {"type": "string"}},
        "remove_node_ids": {"type": "array", "items": {"type": "integer"}},
        "set_widgets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer"},
                    "widget": {"type": "string"},
                    "value": {},
                },
                "required": ["node_id", "widget", "value"],
            },
        },
    },
    "required": ["add_nodes", "remove_node_ids", "set_widgets"],
}

SYSTEM = (
    "You edit an existing ComfyUI workflow. You are given the nodes currently "
    "on the canvas (each with its numeric id and class_type) and a list of "
    "node types that may be added.\n"
    "Produce the SMALLEST edit that satisfies the request:\n"
    "1. add_nodes: class_type names to add, copied exactly from the available "
    "list. Empty if nothing needs adding.\n"
    "2. remove_node_ids: numeric ids of nodes to delete, copied exactly from "
    "the canvas list. Empty if nothing needs removing.\n"
    "3. set_widgets: widget values to change on nodes that stay, as "
    "{node_id, widget, value}. Empty if no values change.\n"
    "Change ONLY what the request asks for. Do not remove a node just because "
    "you are adding one. Do not restate unchanged values. Output ONLY the "
    "JSON object."
)


class StageMError(RuntimeError):
    pass


class ModifyDiff:
    """A validated edit to the current canvas."""

    def __init__(
        self,
        add_nodes: list[str],
        remove_node_ids: list[int],
        set_widgets: list[dict[str, Any]],
        rejected: list[str],
    ) -> None:
        self.add_nodes = add_nodes
        self.remove_node_ids = remove_node_ids
        self.set_widgets = set_widgets
        self.rejected = rejected

    @property
    def is_empty(self) -> bool:
        return not (self.add_nodes or self.remove_node_ids or self.set_widgets)

    def summary(self) -> str:
        parts = []
        if self.add_nodes:
            parts.append(f"+{len(self.add_nodes)} node(s): {', '.join(self.add_nodes)}")
        if self.remove_node_ids:
            parts.append(f"-{len(self.remove_node_ids)} node(s)")
        if self.set_widgets:
            parts.append(f"{len(self.set_widgets)} widget change(s)")
        return "; ".join(parts) if parts else "no changes"

    def to_dict(self) -> dict[str, Any]:
        return {
            "add_nodes": self.add_nodes,
            "remove_node_ids": self.remove_node_ids,
            "set_widgets": self.set_widgets,
        }


def canvas_nodes(graph: Any) -> dict[int, str]:
    """{node_id: class_type} for everything currently on the canvas."""
    if not isinstance(graph, dict):
        return {}
    found: dict[int, str] = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        # Booleans are ints in Python; an id of True would corrupt lookups.
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            continue
        if not isinstance(node_type, str):
            continue
        found[node_id] = node_type
    return found


def _canvas_block(nodes: dict[int, str]) -> str:
    return "\n".join(f"id={nid} type={ctype}" for nid, ctype in sorted(nodes.items()))


def build_prompt(task_prompt: str, nodes: dict[int, str], schema_block: str) -> str:
    return (
        f"Requested change: {task_prompt}\n\n"
        f"Nodes currently on the canvas:\n{_canvas_block(nodes)}\n\n"
        f"Node types available to add:\n{schema_block}"
    )


def _widget_names(node_schema: dict[str, Any]) -> set[str]:
    inputs = node_schema.get("input", {})
    if not isinstance(inputs, dict):
        return set()
    names: set[str] = set()
    for group in ("required", "optional"):
        group_inputs = inputs.get(group)
        if not isinstance(group_inputs, dict):
            continue
        for name, spec in group_inputs.items():
            if is_widget_input(spec):
                names.add(name)
    return names


def validate_diff(
    proposed: dict[str, Any],
    nodes: dict[int, str],
    schema: dict[str, dict[str, Any]],
) -> ModifyDiff:
    """Drop anything that does not correspond to the real canvas and registry.

    Silently applying an unvalidated diff is worse here than in BUILD: BUILD
    adds nodes to empty space, while MODIFY deletes and rewrites work the user
    already has.
    """
    rejected: list[str] = []

    add_nodes: list[str] = []
    for class_type in proposed.get("add_nodes") or []:
        if not isinstance(class_type, str):
            rejected.append(f"add {class_type!r} (not a string)")
        elif class_type not in schema:
            rejected.append(f"add '{class_type}' (unknown node type)")
        else:
            add_nodes.append(class_type)

    remove_ids: list[int] = []
    for node_id in proposed.get("remove_node_ids") or []:
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            rejected.append(f"remove {node_id!r} (not a node id)")
        elif node_id not in nodes:
            rejected.append(f"remove {node_id} (no such node on the canvas)")
        elif node_id in remove_ids:
            continue  # duplicate, not an error
        else:
            remove_ids.append(node_id)

    set_widgets: list[dict[str, Any]] = []
    seen_targets: set[tuple[int, str]] = set()
    for entry in proposed.get("set_widgets") or []:
        if not isinstance(entry, dict):
            rejected.append(f"{entry!r} (malformed widget change)")
            continue
        node_id = entry.get("node_id")
        widget = entry.get("widget")
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id not in nodes:
            rejected.append(f"set widget on {node_id!r} (no such node on the canvas)")
            continue
        if not isinstance(widget, str):
            rejected.append(f"set {widget!r} on {node_id} (widget name must be a string)")
            continue
        if node_id in remove_ids:
            # Setting a value on a node the same diff deletes is incoherent;
            # the removal wins because it is the more explicit instruction.
            rejected.append(f"set '{widget}' on {node_id} (node is also being removed)")
            continue
        known = _widget_names(schema.get(nodes[node_id], {}))
        if known and widget not in known:
            rejected.append(
                f"set '{widget}' on {node_id} ({nodes[node_id]} has no such widget)"
            )
            continue
        target = (node_id, widget)
        if target in seen_targets:
            continue  # last-write-wins would be arbitrary; keep the first
        seen_targets.add(target)
        set_widgets.append({"node_id": node_id, "widget": widget, "value": entry.get("value")})

    if rejected:
        logger.info("stage M rejected %d proposed edit(s): %s", len(rejected), rejected)

    return ModifyDiff(add_nodes, remove_ids, set_widgets, rejected)


async def generate_diff(
    task_prompt: str,
    nodes: dict[int, str],
    schema: dict[str, dict[str, Any]],
    schema_block: str,
) -> ModifyDiff:
    """Ask for an edit to the current canvas and validate every part of it."""
    if not nodes:
        raise StageMError(
            "the canvas is empty - use BUILD to create a workflow before modifying one"
        )

    prompt = build_prompt(task_prompt, nodes, schema_block)
    try:
        result, model_used = await llm_client.generate_json_with_fallback(
            prompt=prompt, schema=SCHEMA, system=SYSTEM
        )
    except llm_client.OllamaGenerationError as exc:
        raise StageMError(f"MODIFY generation failed: {exc}") from exc

    diff = validate_diff(result, nodes, schema)
    logger.info("stage M (%s): %s", model_used, diff.summary())
    return diff
