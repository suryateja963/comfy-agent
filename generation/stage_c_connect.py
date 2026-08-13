"""Stage C — Connection Generation. Section 2E.

Named as the highest-risk stage for a 0.8B model in the plan's critique:
it requires holding multiple validated node schemas in context at once and
reasoning about slot compatibility. Every proposed link is independently
re-validated against type_validator before acceptance — the LLM's output
is a proposal, never trusted directly.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import config, llm_client
from ..knowledge.type_validator import (
    LinkValidationError,
    is_widget_input,
    node_input_type,
    node_output_names,
    validate_link,
)

logger = logging.getLogger("comfy_agent.generation.stage_c")

SCHEMA = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "from_output": {"type": "string"},
                    "to_node": {"type": "string"},
                    "to_input": {"type": "string"},
                },
                "required": ["from_node", "from_output", "to_node", "to_input"],
            },
        }
    },
    "required": ["links"],
}

SYSTEM = (
    "You connect ComfyUI nodes into links. Each node instance below is "
    "listed with its exact instance ID, then its INPUT names, then its "
    "OUTPUT names.\n"
    "Rules, all mandatory:\n"
    "1. from_node and to_node must each be copied EXACTLY from the 'id=' "
    "value of a listed instance — never a description, never a class_type.\n"
    "2. from_output must be exactly ONE single name copied from that node's "
    "OUTPUT list — never the whole list, never comma-separated.\n"
    "3. to_input must be exactly ONE single name copied from that node's "
    "INPUT list — never the whole list, never comma-separated.\n"
    "4. Only propose a link where the task actually needs that connection.\n"
    "Output ONLY a JSON object with a 'links' array of {from_node, "
    "from_output, to_node, to_input}.\n\n"
    "Example — given these instances:\n"
    "id=Loader_1 type=CheckpointLoaderSimple INPUT=[] OUTPUT=[MODEL, CLIP, VAE]\n"
    "id=Sampler_1 type=KSampler INPUT=[model, positive, negative, latent_image] OUTPUT=[LATENT]\n"
    "a correct link is exactly: "
    '{"from_node": "Loader_1", "from_output": "MODEL", "to_node": "Sampler_1", "to_input": "model"}'
)


REPAIR_SYSTEM = (
    "You are completing a PARTIALLY connected ComfyUI graph. Most links are "
    "already correct and are NOT shown. Below is the list of inputs that are "
    "still unconnected, and the nodes available to feed them.\n"
    "For EACH unconnected input listed, propose exactly one link that fills "
    "it. Do not propose links for anything not in that list.\n"
    "from_node and to_node must be copied EXACTLY from an 'id=' value. "
    "from_output must be ONE name from that node's OUTPUT list. to_input must "
    "be exactly the input name given in the unconnected list.\n"
    "Output ONLY a JSON object with a 'links' array of {from_node, "
    "from_output, to_node, to_input}."
)


class StageCError(RuntimeError):
    pass


def unconnected_required(
    node_instances: dict[str, str],
    accepted: list[dict[str, str]],
    schema: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Required, link-capable inputs with no incoming link.

    This is the structural completeness check that replaces a link *count*.
    A count is wrong in both directions: KSampler alone needs four incoming
    links, so a 7-node graph legitimately needs more than 6 — while a graph
    can reach N-1 links and still leave VAEDecode -> SaveImage unconnected,
    because a total says nothing about *which* inputs got filled. That second
    case is the observed failure this function exists to catch.

    Widget inputs are excluded via the same `is_widget_input` classifier Stage
    B uses, so a value supplied by a widget never counts as missing.
    """
    filled = {(link["to_node"], link["to_input"]) for link in accepted}
    missing: list[tuple[str, str]] = []

    for instance_id, class_type in node_instances.items():
        node_schema = schema.get(class_type, {})
        inputs = node_schema.get("input", {})
        if not isinstance(inputs, dict):
            continue
        required = inputs.get("required")
        if not isinstance(required, dict):
            continue
        for name, spec in required.items():
            if is_widget_input(spec):
                continue  # lives in node.widgets, not on the graph
            if (instance_id, name) not in filled:
                missing.append((instance_id, name))
    return missing


def _repair_prompt(
    task_prompt: str,
    missing: list[tuple[str, str]],
    node_instances: dict[str, str],
    schema: dict[str, dict[str, Any]],
) -> str:
    """Ask only about the gaps.

    Re-running the full connection prompt discards links the model already got
    right and re-rolls every one of them. A narrow ask is the same adaptation
    that makes Stage A reliable: the smaller the question, the better a sub-1B
    model answers it.
    """
    instance_block = "\n".join(
        _instance_line(iid, ctype, schema.get(ctype, {}))
        for iid, ctype in node_instances.items()
    )
    wanted = "\n".join(
        f"- node id={iid} (type {node_instances[iid]}) needs input '{name}' "
        f"of type {node_input_type(schema.get(node_instances[iid], {}), name)}"
        for iid, name in missing
    )
    return (
        f"Task: {task_prompt}\n\n"
        f"Node instances:\n{instance_block}\n\n"
        f"Unconnected required inputs to fill:\n{wanted}"
    )


def _link_capable_inputs(node_schema: dict[str, Any]) -> list[str]:
    """Inputs that can actually receive links — i.e. everything except
    widget-only primitives (STRING/INT/FLOAT/BOOLEAN/COMBO), which live in the
    node's widgets, not the graph. Listing them to the LLM invites junk links."""
    inputs = node_schema.get("input", {})
    if not isinstance(inputs, dict):
        return []
    names: list[str] = []
    for group in ("required", "optional"):
        group_inputs = inputs.get(group)
        if not isinstance(group_inputs, dict):
            continue
        for name, spec in group_inputs.items():
            # Shared classifier with Stage B, so the widget/link partitions
            # are exact complements and cannot drift apart.
            if is_widget_input(spec):
                continue
            names.append(name)
    return names


def _instance_line(instance_id: str, class_type: str, node_schema: dict[str, Any]) -> str:
    """Describe one node instance's actual available input/output names, so
    the LLM has something concrete to connect rather than guessing slot
    names blind. Without this, Stage C had no way to know e.g. that
    KSampler's inputs are model/positive/negative/latent_image."""
    input_names = _link_capable_inputs(node_schema)
    output_names = node_output_names(node_schema)
    return f"id={instance_id} type={class_type} INPUT=[{', '.join(input_names)}] OUTPUT=[{', '.join(output_names)}]"


def _validate_links(
    proposed: list[dict[str, Any]],
    node_instances: dict[str, str],
    schema: dict[str, dict[str, Any]],
    existing: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Validate proposed links, dropping anything unsafe or duplicated.

    ``existing`` seeds the taken-target set from links already accepted in an
    earlier pass. Without it the repair pass could propose a second link into
    an input the first pass already filled, and a single input accepts exactly
    one link — so the graph would silently lose whichever one LiteGraph
    applied second.
    """
    accepted: list[dict[str, str]] = []
    rejected: list[str] = []
    seen_targets: set[tuple[str, str]] = {
        (link["to_node"], link["to_input"]) for link in (existing or [])
    }

    for link in proposed:
        if not isinstance(link, dict):
            rejected.append(f"{link} (malformed link entry, expected object)")
            continue

        from_id = link.get("from_node")
        to_id = link.get("to_node")
        from_output = link.get("from_output")
        to_input = link.get("to_input")

        # Every field must be a plain string before it reaches any dict lookup
        # or set insertion. The model can emit a list/dict here, and an
        # unhashable id or slot name previously raised TypeError out of
        # node_instances.get()/seen_targets.add() — which is neither
        # LinkValidationError nor KeyError, so it escaped this function,
        # escaped StageCError, and surfaced as an HTTP 500 that killed the
        # whole build over one bad link.
        if not all(isinstance(v, str) for v in (from_id, to_id, from_output, to_input)):
            rejected.append(f"{link} (non-string node id or slot name)")
            continue

        from_type = node_instances.get(from_id)
        to_type = node_instances.get(to_id)

        if from_type is None or to_type is None:
            rejected.append(f"{link} (unknown node instance id)")
            continue

        if from_id == to_id:
            rejected.append(f"{link} (self-link)")
            continue

        target = (to_id, to_input)
        if target in seen_targets:
            rejected.append(f"{link} (duplicate target input — a single input accepts one link)")
            continue

        try:
            validate_link(from_type, from_output, to_type, to_input, schema)
            seen_targets.add(target)
            accepted.append(link)
        except (LinkValidationError, KeyError, TypeError) as exc:
            # TypeError kept as defence in depth against malformed schema
            # entries reached through validate_link.
            rejected.append(f"{link}: {exc}")

    return accepted, rejected


async def generate_connections(
    task_prompt: str,
    node_instances: dict[str, str],
    schema: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """node_instances: {instance_id: class_type} for every node placed by Stage A."""
    instance_block = "\n".join(
        _instance_line(iid, ctype, schema.get(ctype, {})) for iid, ctype in node_instances.items()
    )
    prompt = f"Task: {task_prompt}\n\nNode instances:\n{instance_block}"

    try:
        result, model_used = await llm_client.generate_json_with_fallback(prompt=prompt, schema=SCHEMA, system=SYSTEM)
    except llm_client.OllamaGenerationError as exc:
        raise StageCError(f"Stage C generation failed: {exc}") from exc

    proposed = result.get("links", [])
    accepted, rejected = _validate_links(proposed, node_instances, schema)

    # Same class of gap as Stage A: the primary model can return syntactically
    # valid JSON that is still mostly garbage (copying the whole bracketed
    # input/output list as one value, wrong instance ids), so error-based
    # escalation never triggers and result *quality* has to be checked
    # directly.
    #
    # The check is structural, not a link count. A count cannot distinguish a
    # graph that is legitimately dense (KSampler alone takes four inputs) from
    # one that is missing precisely the link that matters, which is how
    # VAEDecode -> SaveImage kept slipping through.
    primary, fallback = await llm_client.resolve_models()
    missing = unconnected_required(node_instances, accepted, schema)

    if missing and model_used == primary and fallback != primary:
        # Targeted repair: ask only about the gaps and merge, rather than
        # re-running the full connection step and re-rolling every link the
        # model already got right.
        logger.info(
            "stage C: %d required input(s) unconnected after %s; repairing with %s: %s",
            len(missing), model_used, fallback, missing,
        )
        try:
            repaired = await llm_client.generate_json(
                prompt=_repair_prompt(task_prompt, missing, node_instances, schema),
                schema=SCHEMA,
                model=fallback,
                system=REPAIR_SYSTEM,
            )
            # Validate against the links already accepted, so the repair pass
            # cannot double-fill an input that is now taken.
            extra, extra_rejected = _validate_links(
                repaired.get("links", []), node_instances, schema, existing=accepted
            )
            if extra:
                accepted = accepted + extra
                missing = unconnected_required(node_instances, accepted, schema)
                logger.info(
                    "stage C repair: added %d link(s), %d still unconnected",
                    len(extra), len(missing),
                )
            if extra_rejected:
                logger.info("stage C repair: rejected %s", extra_rejected)
        except llm_client.OllamaGenerationError as exc:
            # Keep the incomplete-but-valid result; Stage D reports the gap.
            logger.info("stage C repair attempt failed, keeping partial graph: %s", exc)

    if rejected:
        # Non-fatal: report what got dropped, keep what's valid (Section 2F:
        # error isolation — don't crash the whole build over one bad link).
        logger.info("stage C (%s): rejected %d/%d proposed links: %s", model_used, len(rejected), len(proposed), rejected)

    return accepted
