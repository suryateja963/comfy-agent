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

from . import config, install_index, llm_client
from .knowledge import schema_compactor

logger = logging.getLogger("comfy_agent.chat_context")


def count_prompt_tokens(messages: list[dict[str, str]], system: str = "") -> int:
    """Estimate the tokens one chat request sends: system prompt plus messages.

    The system prompt must be counted — it carries the canvas, the model
    inventory, the retrieved schemas and the last error, and is usually most
    of the request. Character-based (config.CHARS_PER_TOKEN_ESTIMATE) rather
    than a real tokenizer: the budget check needs an order of magnitude, and a
    tokenizer would be a new dependency.
    """
    chars = len(system or "")
    for message in messages:
        chars += len(str(message.get("content", "")))
    return chars // max(1, config.CHARS_PER_TOKEN_ESTIMATE)


def get_graph_char_budget(model_name: str = "") -> int:
    """Return CHAT_MAX_GRAPH_CHARS based on model parameters.

    Small models (0.8b, 4b) get a conservative budget to avoid OOM;
    larger models can afford more graph context.
    """
    small_models = ["qwen3.5:0.8b", "qwen3.5:4b", "llama3.1:8b"]
    if any(m in model_name.lower() for m in small_models):
        return config.CHAT_MAX_GRAPH_CHARS // 2  # halve the budget for small models
    return config.CHAT_MAX_GRAPH_CHARS  # full budget for larger models


async def summarize_history(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Summarize conversation history to preserve key info within token budget.

    Calls LLM to produce exactly 3 bullet points (prefixed with '-') preserving:
    - task goals
    - model preferences
    - node types mentioned
    - widget values set

    Output format: one '- item per line

    Returns (summary_message, trimmed_messages) where trimmed_messages
    keeps only the most recent 4 exchanges (2 user+assistant pairs),
    excluding the current user message that triggered summarization.
    """
    # Build prompt for LLM summarization
    prompt = "Summarize the following conversation history in exactly 3 bullet points (prefixed with '-'). "
    prompt += "Preserve: task goals, model preferences, node types mentioned, widget values set. "
    prompt += "Output format: one '- item per line.\n\n"
    prompt += "Conversation history:\n"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += "\nSummary (3 bullet points only, each prefixed with '-'):"

    # Call LLM to generate the summary via chat_stream
    system_prompt = "You are a conversation summarizer. Produce exactly 3 bullet points (prefixed with '-') summarizing the conversation history. Preserve task goals, model preferences, node types mentioned, and widget values set. Output one '- item per line only."

    summary = await llm_client.chat_stream(
        [{"role": "user", "content": prompt}],
        system=system_prompt,
    )

    # Parse the 3 bullet points from the summary
    bullets: list[str] = []
    for line in summary.split("\n"):
        stripped = line.strip()
        if stripped.startswith("-"):
            bullets.append(stripped[1:].strip())
        if len(bullets) >= 3:
            break

    summary_text = "\n".join(bullets) if bullets else summary.strip()

    # Trim messages: keep only the most recent 4 exchanges (8 messages),
    # excluding the current user message that triggered summarization
    if len(messages) <= 8:
        trimmed = messages[:-1]
    else:
        trimmed = messages[-9:-1]

    summary_message = summary_text.strip() if summary_text else ""

    return summary_message, trimmed


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

    # Model-aware graph character budgeting (Section 4A): small models get
    # a reduced graph context to avoid OOM; larger models can afford more.
    model_name = ""
    try:
        installed = await llm_client.list_installed()
        primary = config.PRIMARY_MODEL
        primary, _ = await llm_client.resolve_models()
        if primary in installed:
            model_name = primary
    except Exception:  # noqa: BLE001
        pass
    graph_budget = get_graph_char_budget(model_name)

    graph_block = render_graph_block(graph)
    # Truncate if exceeding the model-aware budget
    if len(graph_block) > graph_budget:
        logger.debug(
            "chat: graph block %d chars exceeds budget %d for model %s; truncating",
            len(graph_block),
            graph_budget,
            model_name or "unknown",
        )
        # Re-render with stricter trimming: node types and counts only
        graph_block = _summarize_graph(graph)
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
