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


def _apply_log_level() -> None:
    """Set the level for this package's logger tree from config.LOG_LEVEL.

    Only the "comfy_agent" logger is touched, never the root: this runs inside
    someone else's application, and reconfiguring global logging from a custom
    node would be a rude surprise. LOG_LEVEL = None opts out entirely.
    """
    from . import config

    level = getattr(config, "LOG_LEVEL", None)
    if not level:
        return
    resolved = logging.getLevelName(str(level).upper())
    if not isinstance(resolved, int):
        logger.warning("ignoring invalid LOG_LEVEL %r", level)
        return
    logger.setLevel(resolved)


try:
    _apply_log_level()
except Exception:  # noqa: BLE001
    logger.debug("comfy_agent: could not apply LOG_LEVEL", exc_info=True)

from .agent_node import comfy_entrypoint  # noqa: E402, F401

try:
    from . import agent_routes  # noqa: F401  (registers aiohttp routes on import)
except Exception:  # noqa: BLE001
    logger.exception("comfy_agent: failed to register routes")


# Held on the module so the warm task is not garbage-collected mid-run:
# asyncio keeps only a weak reference to running tasks.
_index_warm_task = None


def _schedule_index_warm() -> None:
    """Start the semantic index build in the background once the server runs.

    Built lazily, the 1767-node index build runs inside the first CHAT request
    and blocks it for the whole build. Warming needs the loop ComfyUI actually
    runs, which does not exist at import time — and `get_event_loop()` outside
    a running loop returns a *different*, never-run loop, so scheduling there
    would silently never fire. aiohttp's on_startup signal is the hook that
    runs on the right loop.

    Any failure here leaves build()'s lazy path in place, so the worst case is
    the behaviour we had before rather than a broken node.
    """
    import asyncio

    from server import PromptServer

    from . import install_index

    async def _on_startup(_app) -> None:
        global _index_warm_task
        _index_warm_task = asyncio.create_task(install_index.warm())

    app = getattr(PromptServer.instance, "app", None)
    if app is None or not hasattr(app, "on_startup"):
        logger.debug("comfy_agent: no aiohttp app to hook; index will build lazily")
        return
    app.on_startup.append(_on_startup)


try:
    _schedule_index_warm()
except Exception:  # noqa: BLE001
    # Warming is an optimisation; never let it stop the node from loading.
    logger.debug("comfy_agent: could not schedule index warm", exc_info=True)

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
