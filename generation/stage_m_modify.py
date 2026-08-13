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

from .. import config, llm_client
from ..knowledge.type_validator import is_widget_input
from .stage_b_configure import _coerce as coerce_widget_value

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
                    # Constrained to scalars: an unconstrained `{}` let every
                    # model size return the value wrapped in a descriptor
                    # object ({"i": 35}) instead of the value itself.
                    "value": {"type": ["string", "number", "integer", "boolean"]},
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
    "{node_id, widget, value}. Empty if no values change. 'value' must be the "
    "bare new value itself - the number 35, not {\"i\": 35} and not an object "
    "describing the widget.\n"
    "Changing a value NEVER requires removing or re-adding a node: to change "
    "steps, emit one set_widgets entry and leave add_nodes and "
    "remove_node_ids empty.\n"
    "Change ONLY what the request asks for. Do not remove a node just because "
    "you are adding one. Do not restate unchanged values. Output ONLY the "
    "JSON object.\n\n"
    "Example - given canvas 'id=1 type=KSampler' and the request 'set steps "
    "to 30', the correct and complete answer is: "
    '{"add_nodes": [], "remove_node_ids": [], '
    '"set_widgets": [{"node_id": 1, "widget": "steps", "value": 30}]}'
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

        raw_value = entry.get("value")
        # Observed on every model size, including the 9B: the value arrives
        # wrapped in an object describing the widget rather than as the value
        # itself — {"i": 35}, {"default_value": 20, "type": "int", ...}. A
        # widget holds a scalar, so a container is always wrong here, and
        # writing one to the canvas produces a node that looks configured and
        # cannot run.
        if isinstance(raw_value, (dict, list)):
            rejected.append(
                f"set '{widget}' on {node_id} (value must be a scalar, got {type(raw_value).__name__})"
            )
            continue

        # Same coercion Stage B applies: type-cast, clamp INT/FLOAT to the
        # declared range, and validate COMBO values against the real options.
        coerced = coerce_widget_value(raw_value, schema.get(nodes[node_id], {}), widget)
        if coerced is None:
            rejected.append(f"set '{widget}' on {node_id} (value {raw_value!r} is not valid for this widget)")
            continue

        seen_targets.add(target)
        set_widgets.append({"node_id": node_id, "widget": widget, "value": coerced})

    if rejected:
        logger.info("stage M rejected %d proposed edit(s): %s", len(rejected), rejected)

    return ModifyDiff(add_nodes, remove_ids, set_widgets, rejected)


# A removal touching more than this share of the canvas is treated as
# suspicious rather than intentional. Deliberately loose: deleting one node of
# three is a legitimate edit, deleting two of three to change a widget value
# is the observed small-model failure.
DESTRUCTIVE_REMOVAL_RATIO = 0.5


def _is_implausibly_destructive(diff: ModifyDiff, nodes: dict[int, str]) -> bool:
    """True when a diff deletes a large share of the canvas.

    Only ever used to trigger a second opinion, never to reject outright: a
    user really can ask for most of their workflow to be removed, and this
    heuristic cannot tell that apart from a model that lost the plot.
    """
    if not diff.remove_node_ids or not nodes:
        return False
    return len(diff.remove_node_ids) / len(nodes) > DESTRUCTIVE_REMOVAL_RATIO


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

    # MODIFY goes to the escalation model first, unlike every other stage.
    # Measured against a live install (1767 nodes, real blueprint canvas):
    # asked to set steps to 35, qwen3.5:4b and :9b both returned exactly
    # {node_id: 5, widget: "steps", value: 35}, while the 0.8B edited an
    # unrelated node with an unrelated value on every attempt. Reasoning over
    # existing node ids is materially harder than the additive stages, and a
    # wrong MODIFY silently corrupts work the user already has - so paying the
    # extra latency is the right trade here even though BUILD does not.
    primary, fallback = await llm_client.resolve_models()
    preferred = fallback if fallback != primary else primary

    try:
        if preferred != primary:
            try:
                result = await llm_client.generate_json(
                    prompt=prompt, schema=SCHEMA, model=preferred, system=SYSTEM
                )
                model_used = preferred
            except llm_client.OllamaGenerationError as exc:
                logger.info("stage M: %s failed (%s); falling back to %s", preferred, exc, primary)
                result, model_used = await llm_client.generate_json_with_fallback(
                    prompt=prompt, schema=SCHEMA, system=SYSTEM
                )
        else:
            # Only one model installed. It still runs, but a small model is
            # unreliable here and the caller should be able to say so.
            logger.warning(
                "stage M: only %s is installed. MODIFY is unreliable on a sub-1B "
                "model - `ollama pull %s` for accurate edits.",
                primary, config.FALLBACK_CHAIN[1],
            )
            result, model_used = await llm_client.generate_json_with_fallback(
                prompt=prompt, schema=SCHEMA, system=SYSTEM
            )
    except llm_client.OllamaGenerationError as exc:
        raise StageMError(f"MODIFY generation failed: {exc}") from exc

    diff = validate_diff(result, nodes, schema)

    # Escalate wholesale deletion rather than applying it. Measured against a
    # live Ollama: asked only to change a widget value, the 0.8B proposed
    # deleting two of three canvas nodes, while the 4B and 9B returned exactly
    # the one widget change. Every id it named was real, so no amount of
    # validation catches this - the check has to be on the *shape* of the edit.
    # Still worth checking after the preferred-model call above: it only
    # applies when MODIFY had to run on the primary (nothing better installed,
    # or the escalation model failed), which is exactly when this happens.
    if (
        _is_implausibly_destructive(diff, nodes)
        and model_used == primary
        and fallback != primary
    ):
        logger.info(
            "stage M: %s proposed removing %d of %d node(s); re-asking %s",
            model_used, len(diff.remove_node_ids), len(nodes), fallback,
        )
        try:
            escalated = await llm_client.generate_json(
                prompt=prompt, schema=SCHEMA, model=fallback, system=SYSTEM
            )
            candidate = validate_diff(escalated, nodes, schema)
            if not _is_implausibly_destructive(candidate, nodes) and not candidate.is_empty:
                diff = candidate
                model_used = fallback
        except llm_client.OllamaGenerationError as exc:
            logger.info("stage M escalation failed, keeping the original diff: %s", exc)

    logger.info("stage M (%s): %s", model_used, diff.summary())
    return diff
