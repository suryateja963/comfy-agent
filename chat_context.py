"""System-prompt assembly for CHAT mode.

The point of an in-node chat (versus a browser tab with a general chatbot) is
that it can see the graph you are actually looking at. This module turns the
canvas snapshot the frontend sends up, plus the live /object_info registry,
into a compact context block a sub-1B model can still reason over.

Everything here is budget-aware: a real workflow serializes to tens of
thousands of characters, and an unbudgeted dump silently pushes the user's
actual question out of the context window (Section 2D / 4A).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import config, install_index
from .knowledge import schema_compactor

logger = logging.getLogger("comfy_agent.chat_context")

SYSTEM_PROMPT = """You are a ComfyUI assistant embedded inside a node on the user's canvas.

You have live knowledge of THIS specific ComfyUI installation: the nodes that
are actually installed and the model files actually present on disk. The
sections below are read from the running server, not from memory.

You help with:
- how ComfyUI works, and what specific nodes do
- what is installed here: nodes, checkpoints, LoRAs, VAEs, ControlNets, upscalers
- reading and debugging the user's current workflow
- diagnosing errors and giving concrete fixes

Rules:
- Answer in plain prose. Never reply with JSON.
- Be concise and concrete. Prefer naming the exact node and widget to change
  over general advice.
- When the user's workflow is shown below, refer to their actual nodes by name.
- Recommend ONLY nodes and model files that appear in the sections below. If
  something useful is not installed, say it is not installed rather than
  suggesting it as though it were available.
- The node list below is the top matches for this question, not the full
  registry — if you do not see a node you expect, say so instead of asserting
  it is missing from the installation.
- Keep answers short unless asked to elaborate."""


def _summarize_graph(graph: dict[str, Any]) -> str:
    """Fallback rendering when the full serialization exceeds the budget:
    node types and counts only. Loses widget values, keeps the shape."""
    nodes = graph.get("nodes") or []
    counts: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "unknown")
        counts[node_type] = counts.get(node_type, 0) + 1
    link_count = len(graph.get("links") or [])
    listing = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(counts.items()))
    return f"{len(nodes)} nodes ({listing}), {link_count} links. " \
           "(Workflow too large to include in full; widget values omitted.)"


def _trim_node(node: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields that matter for reasoning about a graph.

    Drops position, size, colour, flags and other pure-presentation keys,
    which otherwise account for most of the serialized bytes.
    """
    trimmed: dict[str, Any] = {
        "id": node.get("id"),
        "type": node.get("type"),
    }
    if node.get("title"):
        trimmed["title"] = node["title"]
    widgets = node.get("widgets_values")
    if widgets:
        trimmed["widgets_values"] = widgets
    # Inputs matter only for what they are connected to.
    inputs = node.get("inputs")
    if isinstance(inputs, list):
        links = [
            {"name": i.get("name"), "link": i.get("link")}
            for i in inputs
            if isinstance(i, dict) and i.get("link") is not None
        ]
        if links:
            trimmed["connected_inputs"] = links
    return trimmed


def render_graph_block(graph: Any) -> str:
    """Render the canvas snapshot for the system prompt, within budget."""
    if not isinstance(graph, dict) or not graph.get("nodes"):
        return "The user's canvas is currently empty."

    try:
        trimmed = {
            "nodes": [_trim_node(n) for n in graph.get("nodes") or [] if isinstance(n, dict)],
            "links": graph.get("links") or [],
        }
        rendered = json.dumps(trimmed, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        logger.debug("could not serialize graph snapshot: %s", exc)
        return "The user's workflow could not be read."

    if len(rendered) > config.CHAT_MAX_GRAPH_CHARS:
        return _summarize_graph(graph)
    return rendered


async def render_schema_block(question: str, graph: Any) -> str:
    """Schemas for the nodes most relevant to this question.

    Backed by the semantic index over the whole install (install_index), which
    is why this can answer about any of the ~1767 installed node types rather
    than only the keyword-matched categories the BUILD pipeline uses.

    Nodes already on the canvas are appended unconditionally: a question like
    "why is this broken?" may not rank the user's own nodes highly, but they
    are exactly what needs explaining.
    """
    lines: list[str] = []
    try:
        lines = await install_index.search(question)
    except Exception as exc:  # noqa: BLE001
        # Retrieval is an enhancement; chat must still answer without it.
        logger.debug("chat: index retrieval unavailable: %s", exc)

    canvas_types: set[str] = set()
    if isinstance(graph, dict):
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("type"):
                canvas_types.add(str(node["type"]))

    if canvas_types:
        try:
            object_info = await schema_compactor.fetch_object_info()
            present = {line.split(":", 1)[0] for line in lines}
            for class_type in sorted(canvas_types - present):
                entry = object_info.get(class_type)
                if isinstance(entry, dict):
                    lines.append(install_index.schema_line(class_type, entry))
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat: could not add canvas node schemas: %s", exc)

    return "\n".join(lines)


async def build_system_prompt(question: str, graph: Any, last_error: str = "") -> str:
    """Assemble the full CHAT system prompt."""
    sections = [SYSTEM_PROMPT]

    # Ensure the install index exists before anything reads from it. Build
    # failures are swallowed inside build(); retrieval degrades to keyword
    # scoring rather than taking chat down with it.
    try:
        await install_index.build()
    except Exception as exc:  # noqa: BLE001
        logger.debug("chat: install index unavailable: %s", exc)

    graph_block = render_graph_block(graph)
    sections.append(f"--- USER'S CURRENT WORKFLOW ---\n{graph_block}")

    models_block = install_index.render_model_inventory()
    if models_block:
        sections.append(
            "--- MODELS INSTALLED ON THIS MACHINE (folder (count): files) ---\n"
            f"{models_block}"
        )

    schema_block = await render_schema_block(question, graph)
    if schema_block:
        sections.append(
            "--- INSTALLED NODES MOST RELEVANT TO THIS QUESTION "
            f"(name: inputs / outputs) ---\n{schema_block}"
        )

    if last_error:
        sections.append(
            f"--- MOST RECENT EXECUTION ERROR ---\n{last_error[: config.CHAT_MAX_MESSAGE_CHARS]}"
        )

    return "\n\n".join(sections)
