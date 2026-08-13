"""Fetch and filter /object_info. Section 2D — Two-Pass Context Strategy.

Pass 1 classifies task category; this module then filters the full node
registry down to only the categories relevant to the task, keeping Pass 2's
context small enough for a sub-1B model to stay coherent (Section 4A).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

from .. import config
from .type_validator import node_output_names

logger = logging.getLogger("comfy_agent.knowledge.schema_compactor")

# Nodes without which a workflow cannot be assembled at all. These are admitted
# to the candidate pool before the char budget is applied and are exempt from
# it: the pool is scanned in arbitrary /object_info order, so on an install
# with many custom nodes a late-sorting SaveImage/VAEDecode was being evicted
# by the budget. Stage A can only pick from this pool, so an eviction made a
# correct workflow structurally impossible while presenting as a generic
# "generation failed".
ESSENTIAL_NODES = frozenset(
    {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "VAEEncode",
        "SaveImage",
        "LoadImage",
    }
)

# Keyword -> category buckets. A node is assigned to every category whose
# keyword appears in its class_type or category field (case-insensitive).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "loaders": ["loader", "checkpoint", "unet", "clip", "vae", "lora"],
    "sampling": ["sampler", "ksampler", "scheduler", "guider", "noise"],
    "latent": ["latent", "empty"],
    "conditioning": ["conditioning", "clip text", "cliptext"],
    "image": ["image", "save", "preview", "load image"],
    "upscaling": ["upscale", "hires", "esrgan"],
    "controlnet": ["controlnet", "control_net", "preprocessor"],
    "video": ["video", "wan", "animate"],
    "mask": ["mask", "inpaint", "segment"],
    "custom": [],
}


async def fetch_object_info() -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=config.COMFY_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{config.COMFY_HOST}/object_info") as resp:
                if resp.status != 200:
                    raise RuntimeError(f"/object_info returned HTTP {resp.status}")
                return await resp.json()
    except (TimeoutError, asyncio.TimeoutError) as exc:
        # Checked before ClientError: aiohttp.ServerTimeoutError subclasses
        # ClientError, so the reverse order reported timeouts as "unreachable".
        # asyncio.TimeoutError is the builtin TimeoutError only on 3.11+.
        raise RuntimeError(f"/object_info timed out after {config.COMFY_TIMEOUT_SECONDS}s") from exc
    except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
        # aiohttp raises ContentTypeError (a ClientError subclass) for a
        # non-JSON body, so this must precede the ClientError handler.
        raise RuntimeError(f"/object_info returned a non-JSON body: {exc}") from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"/object_info unreachable at {config.COMFY_HOST}: {exc}") from exc


def _classify_node(class_type: str, entry: dict[str, Any]) -> set[str]:
    haystack = f"{class_type} {entry.get('category', '')}".lower()
    hits = {cat for cat, keywords in CATEGORY_KEYWORDS.items() if any(kw in haystack for kw in keywords)}
    return hits or {"custom"}


def _input_names(entry: dict[str, Any]) -> list[str]:
    inputs = entry.get("input", {})
    if not isinstance(inputs, dict):
        return []
    names: list[str] = []
    for group in ("required", "optional"):
        group_inputs = inputs.get(group)
        if isinstance(group_inputs, dict):
            names.extend(group_inputs.keys())
    return names


def _estimate_line_chars(class_type: str, entry: dict[str, Any]) -> int:
    inputs = _input_names(entry)
    outputs = node_output_names(entry)
    return len(class_type) + sum(len(str(i)) + 1 for i in inputs) + sum(len(str(o)) + 1 for o in outputs) + 16


def _compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": entry.get("input", {}),
        "output": entry.get("output", []),
        "output_name": entry.get("output_name", []),
    }


def compact_schema(
    object_info: dict[str, Any],
    categories: list[str],
    *,
    max_nodes: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Filter the full /object_info dict down to the given categories.

    Skips malformed node entries silently rather than failing the whole
    build (Section 6 integration test: malformed /object_info entry).

    `max_chars` enforces config.PASS2_MAX_SCHEMA_TOKENS (Section 2D/4A):
    on a real install this category set alone can produce a 1700+ node,
    3000+ token candidate pool before any budget is applied — well past
    what a sub-1B model can hold onto, which is exactly the failure mode
    the plan warns about. Defaults to the configured budget so this is
    enforced by default rather than opt-in.
    """
    if max_chars is None:
        max_chars = config.PASS2_MAX_SCHEMA_TOKENS * config.CHARS_PER_TOKEN_ESTIMATE
    if max_nodes is None:
        max_nodes = config.COMPACT_SCHEMA_MAX_NODES

    filtered: dict[str, Any] = {}
    running_chars = 0
    wanted = set(categories) or set(CATEGORY_KEYWORDS.keys())

    # Phase 1 — admit essential nodes unconditionally, ignoring both budgets.
    # Without a workflow's terminal/loader nodes there is nothing for Stage A
    # to build with, so they are never the right thing to evict.
    for class_type in sorted(ESSENTIAL_NODES):
        entry = object_info.get(class_type)
        if not isinstance(entry, dict):
            continue
        try:
            filtered[class_type] = _compact_entry(entry)
            running_chars += _estimate_line_chars(class_type, entry)
        except (TypeError, AttributeError, KeyError) as exc:
            logger.debug("skipping malformed essential node entry %s: %s", class_type, exc)

    # Phase 2 — budget-limited scan for everything else in the wanted categories.
    dropped = 0
    for class_type, entry in object_info.items():
        if class_type in filtered:
            continue
        if len(filtered) >= max_nodes:
            dropped += 1
            continue
        try:
            node_categories = _classify_node(class_type, entry)
            if node_categories & wanted:
                line_chars = _estimate_line_chars(class_type, entry)
                if running_chars + line_chars > max_chars and filtered:
                    dropped += 1
                    continue  # skip this one, but keep scanning for smaller entries
                filtered[class_type] = _compact_entry(entry)
                running_chars += line_chars
        except (TypeError, AttributeError, KeyError) as exc:
            logger.debug("skipping malformed node entry %s: %s", class_type, exc)
            continue

    if dropped:
        # Make truncation visible: a silently amputated pool used to surface
        # only as an unexplained downstream "generation failed".
        logger.warning(
            "schema budget dropped %d candidate node(s); kept %d (~%d chars, budget %d)",
            dropped,
            len(filtered),
            running_chars,
            max_chars,
        )

    return filtered


def to_compact_prompt_block(compacted: dict[str, Any]) -> str:
    """Render the compacted schema as a terse text block for LLM prompts."""
    lines = []
    for class_type, entry in compacted.items():
        inputs = _input_names(entry)
        outputs = node_output_names(entry)
        lines.append(f"{class_type}: in=[{','.join(inputs)}] out=[{','.join(str(o) for o in outputs)}]")
    return "\n".join(lines)
