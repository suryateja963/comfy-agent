"""Stage A — Node Identification. Section 2E.

Narrowest possible ask: a bare JSON array of class_type strings, nothing
else (Section 4A small-model adaptation). Every name is validated against
the compacted /object_info before Stage B ever runs.
"""

from __future__ import annotations

from .. import config, llm_client
from ..knowledge.schema_compactor import ESSENTIAL_NODES

SCHEMA = {
    "type": "object",
    "properties": {"node_types": {"type": "array", "items": {"type": "string"}}},
    "required": ["node_types"],
}

SYSTEM = (
    "You select ComfyUI node class_type names needed for a task, from the "
    "provided node list ONLY. A complete image-generation workflow needs one "
    "node from EACH of these roles if relevant to the task: model/checkpoint "
    "loader, positive prompt encoder, negative prompt encoder, latent image "
    "source, sampler, image decoder, image save/output. Include one node per "
    "role, not just the sampler. If positive and negative prompts both need "
    "encoding with the same node type, list that class_type TWICE — once per "
    "role — so two separate instances are created. Output ONLY a JSON "
    "object with a 'node_types' array of exact class_type strings copied "
    "from the list. Do not invent names. Do not explain."
)

# Below this many valid nodes, a result is almost certainly an incomplete
# workflow rather than a genuinely small one — worth the extra latency of an
# escalation attempt. Empirically, qwen3.5:0.8b returns syntactically valid
# JSON with only 2 nodes for prompts that need 6+; because it's valid JSON,
# generate_json_with_fallback's error-based escalation never triggers, so
# this checks result *quality*, not just whether the call succeeded.
MIN_PLAUSIBLE_NODES = 5


class StageAError(RuntimeError):
    pass


def _build_prompt(task_prompt: str, compact_schema_block: str, max_nodes: int) -> str:
    return (
        f"Task: {task_prompt}\n\n"
        f"Available nodes (class_type: in=[...] out=[...]):\n{compact_schema_block}\n\n"
        f"Select at most {max_nodes} node class_types needed to accomplish the task."
    )


async def identify_nodes(
    task_prompt: str,
    compact_schema_block: str,
    max_nodes: int,
    known_class_types: set[str],
) -> list[str]:
    prompt = _build_prompt(task_prompt, compact_schema_block, max_nodes)

    try:
        result, model_used = await llm_client.generate_json_with_fallback(prompt=prompt, schema=SCHEMA, system=SYSTEM)
    except llm_client.OllamaGenerationError as exc:
        raise StageAError(f"Stage A generation failed: {exc}") from exc

    candidates = result.get("node_types", [])
    valid = [c for c in candidates if c in known_class_types]
    unknown = [c for c in candidates if c not in known_class_types]

    # Compare against the *resolved* primary, not config.PRIMARY_MODEL: when the
    # tuned LoRA is absent the run happens on the base model, and testing
    # against the configured name would silently skip escalation in exactly the
    # case where the weaker model makes it most necessary.
    primary, fallback = await llm_client.resolve_models()
    if model_used == primary and fallback != primary and len(valid) < MIN_PLAUSIBLE_NODES:
        try:
            escalated = await llm_client.generate_json(prompt=prompt, schema=SCHEMA, model=fallback, system=SYSTEM)
            escalated_valid = [c for c in escalated.get("node_types", []) if c in known_class_types]
            if len(escalated_valid) > len(valid):
                valid = escalated_valid
        except llm_client.OllamaGenerationError:
            pass  # keep the sparse-but-valid primary result rather than failing outright

    if not valid:
        raise StageAError(
            f"Stage A produced no valid node types (model={model_used}). "
            f"Rejected unknown names: {unknown}. Valid hint: pick from the provided node list."
        )

    if len(valid) > max_nodes:
        # Truncating in raw model-output order can decapitate the workflow —
        # if SaveImage/VAEDecode happen to be listed last they get clipped,
        # leaving a graph with no output stage that Stage D won't flag
        # (it checks connectivity, not terminal nodes). Order essentials
        # first so the clip only ever drops optional extras. Relative order
        # within each partition is preserved.
        essential = [c for c in valid if c in ESSENTIAL_NODES]
        rest = [c for c in valid if c not in ESSENTIAL_NODES]
        valid = essential + rest

    return valid[:max_nodes]
