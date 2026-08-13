"""comfy_agent — meta-node that builds ComfyUI workflows from natural language.

See MASTER_PLAN.md for the full architecture. This file wires up:
  - V3 schema node registration via `comfy_entrypoint` (Section 2B)
  - aiohttp route setup (agent_routes.py)
  - WEB_DIRECTORY for the frontend extension (js/)

IMPORTANT: nodes.py's custom-node loader checks for `NODE_CLASS_MAPPINGS`
*before* `comfy_entrypoint` and short-circuits (returns True) the moment it
finds a `NODE_CLASS_MAPPINGS` attribute on this module — even an empty dict,
since the check is `is not None`, not truthiness. Do NOT define
NODE_CLASS_MAPPINGS anywhere this module re-exports it, or the V3
`comfy_entrypoint` branch below will never run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("comfy_agent")

WEB_DIRECTORY = "./js"

from .agent_node import comfy_entrypoint  # noqa: E402, F401

try:
    from . import agent_routes  # noqa: F401  (registers aiohttp routes on import)
except Exception:  # noqa: BLE001
    logger.exception("comfy_agent: failed to register routes")

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
