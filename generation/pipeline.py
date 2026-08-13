"""Free-form generation orchestrator. Wires Pass 1 + Stages A-D together.

This is the exception path (Section 3B / 4A) — agent_routes.py tries a
blueprint match first and only falls through to this pipeline when nothing
matches or use_blueprint=False.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import classify, stage_a_identify, stage_b_configure, stage_c_connect, stage_d_validate
from ..knowledge import schema_compactor
from .. import config, llm_client

logger = logging.getLogger("comfy_agent.generation.pipeline")

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class GenerationResult:
    ok: bool
    node_instances: dict[str, str] = field(default_factory=dict)  # instance_id -> class_type
    widgets: dict[str, dict[str, Any]] = field(default_factory=dict)  # instance_id -> {widget: value}
    links: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    # (instance_id, input_name) pairs the canvas should flag for the user.
    incomplete: list[tuple[str, str]] = field(default_factory=list)
    validation: stage_d_validate.ValidationResult | None = None

    @property
    def execution_ready(self) -> bool:
        return self.validation.execution_ready if self.validation else False


async def _noop_progress(_msg: str) -> None:
    return None


async def generate_workflow(
    task_prompt: str,
    max_nodes: int,
    on_progress: ProgressCallback = _noop_progress,
) -> GenerationResult:
    try:
        await on_progress("Fetching node registry...")
        object_info = await schema_compactor.fetch_object_info()
    except RuntimeError as exc:
        return GenerationResult(ok=False, error=str(exc))

    await on_progress("Classifying task type...")
    try:
        categories = await classify.classify_task(task_prompt)
    except (llm_client.OllamaGenerationError, llm_client.OllamaUnavailableError):
        # Pass 1 must never take down a build, but only *expected* LLM failures
        # are absorbed here. A blanket `except Exception` also swallowed real
        # defects in classification, converting them into an invisible
        # performance regression (a much larger context for the 0.8B model)
        # instead of a reported failure.
        logger.warning("Pass 1 classification failed; using broad category pool", exc_info=True)
        categories = [c for c in classify.CATEGORY_LIST if c != "custom"]


    compacted = schema_compactor.compact_schema(object_info, categories)
    if not compacted:
        return GenerationResult(ok=False, error="no matching nodes found for classified categories")
    schema_block = schema_compactor.to_compact_prompt_block(compacted)
    logger.info(
        "schema_block ~%d chars (~%d tokens est) vs budget %d tokens",
        len(schema_block),
        len(schema_block) // 4,
        config.PASS2_MAX_SCHEMA_TOKENS,
    )

    await on_progress("Identifying nodes...")
    try:
        class_types = await stage_a_identify.identify_nodes(
            task_prompt, schema_block, max_nodes, known_class_types=set(compacted.keys())
        )
    except stage_a_identify.StageAError as exc:
        return GenerationResult(ok=False, error=str(exc))

    node_instances: dict[str, str] = {}
    for class_type in class_types:
        instance_id = f"{class_type}_{uuid.uuid4().hex[:6]}"
        node_instances[instance_id] = class_type

    await on_progress(f"Validating {len(node_instances)} of {len(class_types)} nodes...")

    # Role-aware Stage B: duplicate class types (e.g. two CLIPTextEncode for
    # positive/negative) are told their ordinal so the model can differentiate
    # them; all widget calls run concurrently.
    class_type_counts: dict[str, int] = {}
    for _class_type in class_types:
        class_type_counts[_class_type] = class_type_counts.get(_class_type, 0) + 1

    role_by_instance: dict[str, int] = {}
    seen: dict[str, int] = {}
    for instance_id, class_type in node_instances.items():
        role_by_instance[instance_id] = seen.get(class_type, 0)
        seen[class_type] = role_by_instance[instance_id] + 1

    async def _configure_one(instance_id: str, class_type: str) -> dict[str, Any]:
        return await stage_b_configure.configure_widgets(
            task_prompt,
            class_type,
            compacted[class_type],
            role_index=role_by_instance[instance_id],
            role_count=class_type_counts[class_type],
        )

    stage_b_results = await asyncio.gather(
        *(_configure_one(iid, ctype) for iid, ctype in node_instances.items()),
        return_exceptions=True,
    )
    widgets: dict[str, dict[str, Any]] = {}
    for instance_id, result in zip(node_instances.keys(), stage_b_results):
        if isinstance(result, Exception):
            logger.warning("Stage B failed for %s: %s", instance_id, result)
            widgets[instance_id] = {}
        else:
            widgets[instance_id] = result

    await on_progress("Generating connections...")
    try:
        links = await stage_c_connect.generate_connections(task_prompt, node_instances, compacted)
    except stage_c_connect.StageCError as exc:
        await on_progress(f"Error in Stage C: {exc}")
        return GenerationResult(ok=False, error=str(exc))

    # Recomputed here rather than returned from Stage C so the check covers the
    # links as they finally stand, including anything the repair pass added.
    incomplete = stage_c_connect.unconnected_required(node_instances, links, compacted)
    validation = stage_d_validate.validate_spec(node_instances, links, incomplete=incomplete)
    for warning in validation.warnings:
        logger.info("stage D warning: %s", warning)
        await on_progress(f"Warning: {warning}")

    return GenerationResult(
        ok=validation.valid,
        node_instances=node_instances,
        widgets=widgets,
        links=links,
        error="; ".join(validation.errors) if validation.errors else None,
        warnings=list(validation.warnings),
        incomplete=list(incomplete),
        validation=validation,
    )
