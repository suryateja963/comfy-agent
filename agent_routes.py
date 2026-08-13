"""Async aiohttp routes for the agent backend. Section 2A / 2G / 4C.

Every route is async. No blocking `requests` calls — use aiohttp.ClientSession
for outbound calls (e.g. to Ollama). All routes are localhost-only by default
(Section 4C) until an explicit opt-in config is added.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import secrets
from typing import Any

from aiohttp import web
from server import PromptServer

from . import chat_context, config, install_index, llm_client
from .generation import pipeline
from .knowledge import blueprint_matcher

logger = logging.getLogger("comfy_agent.routes")

routes = PromptServer.instance.routes

PROGRESS_EVENT = "comfy_agent.progress"
CHAT_TOKEN_EVENT = "comfy_agent.chat_token"

# max_nodes bounds, mirroring the node schema in agent_node.py. The schema is
# only enforced by the ComfyUI frontend; a direct POST bypasses it entirely,
# and an unbounded value fans out that many concurrent Stage B LLM calls.
MIN_MAX_NODES = 4
MAX_MAX_NODES = 20

# Server-side concurrency cap. The frontend's `comfyAgentBuilding` flag is a
# UX guard only and is bypassed by any direct POST, so this is the real limit
# on how much LLM work one client can start at once.
MAX_CONCURRENT_BUILDS = 2
_build_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BUILDS)


LOCALHOST_PEERS = ("127.0.0.1", "::1", "localhost")


def _is_localhost(request: web.Request) -> bool:
    return request.remote in LOCALHOST_PEERS


def _peer_in_allowed_cidrs(request: web.Request) -> bool:
    cidrs = getattr(config, "ALLOWED_NETWORK_CIDRS", None) or []
    peer = request.remote
    if not cidrs or not peer:
        return False
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if address in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            logger.warning("ignoring malformed entry in ALLOWED_NETWORK_CIDRS: %r", cidr)
    return False


def _is_authorized(request: web.Request) -> bool:
    """Access control for the agent routes.

    localhost is always allowed. Beyond that, a peer must fall in an explicit
    ALLOWED_NETWORK_CIDRS range AND present AGENT_API_TOKEN when one is set.
    ALLOW_NETWORK remains as a blanket escape hatch but is deliberately the
    least preferred option — it disables every check at once.
    """
    if _is_localhost(request):
        return True
    if config.ALLOW_NETWORK:
        return True
    if not _peer_in_allowed_cidrs(request):
        return False
    token = getattr(config, "AGENT_API_TOKEN", "") or ""
    if not token:
        return True
    return secrets.compare_digest(request.headers.get("X-Agent-Token", ""), token)


def _is_allowed_origin(request: web.Request) -> bool:
    """Reject cross-origin POSTs.

    `_is_localhost` only checks the socket peer, which any page loaded in the
    user's browser satisfies — including a malicious site that POSTs here to
    burn local LLM inference and inject nodes onto the canvas. A same-origin
    request from the ComfyUI frontend either omits Origin or sends its own.
    """
    if config.ALLOW_NETWORK:
        return True
    origin = request.headers.get("Origin")
    if origin is None:
        return True  # non-browser client (curl, tests) or same-origin navigation
    host = request.headers.get("Host", "")
    allowed = {
        f"http://{host}",
        f"https://{host}",
        "http://127.0.0.1:8188",
        "http://localhost:8188",
    }
    return origin in allowed


async def _ws_send(event: str, payload: dict, sid: str | None) -> None:
    """Emit one WebSocket event, tolerating both the async and sync send APIs.

    Prefers PromptServer.send() — the native coroutine — since these call
    sites are already inside the event loop; send_sync() is meant for calls
    from worker threads and hops through call_soon_threadsafe to get back
    here. Falls back to send_sync when the async API is unavailable, so this
    keeps working across ComfyUI versions.
    """
    try:
        server = PromptServer.instance
        send = getattr(server, "send", None)
        if send is not None and inspect.iscoroutinefunction(send):
            await send(event, payload, sid)
        else:
            server.send_sync(event, payload, sid)
    except Exception:  # noqa: BLE001
        # Streaming is best-effort — never let a WS hiccup fail the request.
        logger.debug("failed to send %s event", event, exc_info=True)


async def _send_progress(sid: str | None, message: str, build_id: str | None = None) -> None:
    """Emit one progress line. `build_id` lets the frontend route the message
    to the node that started this build instead of every agent node on the
    canvas (which otherwise stomp each other's status).

    Prefers PromptServer.send() — the native coroutine — since these call
    sites are already inside the event loop; send_sync() is meant for calls
    from worker threads and hops through call_soon_threadsafe to get back
    here. Falls back to send_sync when the async API is unavailable, so this
    keeps working across ComfyUI versions.
    """
    logger.info("[progress] %s", message)
    payload: dict[str, str] = {"message": message}
    if build_id:
        payload["build_id"] = build_id
    await _ws_send(PROGRESS_EVENT, payload, sid)


@routes.get("/agent/ping")
async def ping(request: web.Request) -> web.Response:
    if not _is_authorized(request):
        return web.json_response({"error": "forbidden"}, status=403)
    return web.json_response({"status": "ok"})


@routes.get("/agent/health")
async def health(request: web.Request) -> web.Response:
    """Per-service readiness, so a broken setup is visible in the node itself.

    Every failure this reports used to surface only as an opaque mid-build
    error ("all generation attempts failed") or a console warning the user
    never sees. `hint` carries the exact command that fixes each problem.

    Always HTTP 200 when authorized: "Ollama is down" is a successful answer
    to this question, and a non-200 would make the frontend unable to tell a
    reachable-but-unhealthy backend from an unreachable one.
    """
    if not _is_authorized(request):
        return web.json_response({"error": "forbidden"}, status=403)

    ollama_up = await llm_client.is_available()

    installed: set[str] = set()
    primary = config.PRIMARY_MODEL
    fallback = config.FALLBACK_MODEL
    if ollama_up:
        installed = await llm_client.list_installed()
        # The names actually used at run time, after the LoRA-missing and
        # fallback-chain degradations in resolve_models().
        primary, fallback = await llm_client.resolve_models()
        # resolve_models() caches for the process lifetime, so a user who fixes
        # a missing model after the first resolve would keep being told it is
        # missing — precisely when this endpoint has to be right. Re-resolve
        # against the freshly listed models when the cached answer looks stale.
        if primary not in installed or fallback not in installed:
            primary, fallback = await llm_client.resolve_models(force=True)

    index_stats = install_index.stats()

    hints: list[str] = []
    if not ollama_up:
        hints.append(f"Ollama is not reachable at {config.OLLAMA_HOST} — start it with `ollama serve`.")
    else:
        if primary not in installed:
            hints.append(f"Primary model missing — run `ollama pull {config.BASE_MODEL}`.")
        if fallback not in installed:
            hints.append(
                "No escalation model installed — run "
                f"`ollama pull {config.FALLBACK_CHAIN[0]}` for better free-form builds."
            )
        if config.EMBED_MODEL not in installed:
            hints.append(
                f"Embedding model missing — run `ollama pull {config.EMBED_MODEL}` "
                "for semantic CHAT retrieval (keyword search is used meanwhile)."
            )

    return web.json_response(
        {
            "ollama": ollama_up,
            "ollama_host": config.OLLAMA_HOST,
            "primary_model": {"name": primary, "installed": ollama_up and primary in installed},
            "fallback_model": {"name": fallback, "installed": ollama_up and fallback in installed},
            "embed_model": {
                "name": config.EMBED_MODEL,
                "installed": ollama_up and config.EMBED_MODEL in installed,
            },
            "index": index_stats,
            "blueprints": len(blueprint_matcher.list_blueprints()),
            "hints": hints,
        }
    )


def _error_result(detail: str) -> dict:
    return {
        "status_message": "",
        "error_detail": detail,
        "workflow_json": "{}",
        "execution_ready": False,
    }


async def _handle_build(body: dict, sid: str | None, build_id: str | None = None) -> dict:
    """BUILD mode: blueprint-first, falling back to four-stage free-form
    generation. Section 3A / 3B / 3C."""
    # Bind build_id once so every progress line in this handler carries it.
    async def _progress(message: str) -> None:
        await _send_progress(sid, message, build_id)

    task_prompt = body.get("task_prompt", "")
    task_category = body.get("task_category", "auto")
    model_family = str(body.get("model_family") or "auto").strip().lower()
    use_blueprint = bool(body.get("use_blueprint", True))

    if not isinstance(task_prompt, str):
        return _error_result("task_prompt must be a string")

    try:
        max_nodes = int(body.get("max_nodes", 12))
    except (TypeError, ValueError):
        return _error_result("max_nodes must be an integer")
    # Clamp rather than reject: the schema already constrains the UI path, and
    # an out-of-range value from a direct POST is better bounded than fatal.
    max_nodes = max(MIN_MAX_NODES, min(max_nodes, MAX_MAX_NODES))

    if not task_prompt.strip():
        return _error_result("task_prompt is empty")

    if use_blueprint:
        await _progress("Checking blueprint library...")
        blueprint_name = blueprint_matcher.match(task_prompt, task_category, model_family)
        if blueprint_name:
            spec = blueprint_matcher.load_blueprint(blueprint_name)
            if spec is not None:
                # load_blueprint guarantees list-valued 'nodes' and 'links'.
                node_count = len(spec["nodes"])
                link_count = len(spec["links"])
                await _progress(f"Instantiating blueprint '{blueprint_name}'...")
                await _progress(f"Done — built {node_count} nodes, {link_count} links")
                return {
                    "status_message": f"Built from blueprint '{blueprint_name}' "
                    f"({node_count} nodes, {link_count} links)",
                    "error_detail": "",
                    "workflow_json": json.dumps(spec),
                    "execution_ready": True,
                    "source": "blueprint",
                    "blueprint_name": blueprint_name,
                    "subgraph_wrap_threshold": config.SUBGRAPH_WRAP_THRESHOLD,
                }
            # Blueprint load failed outright (Section 6) — fall through to free-form.
            await _progress(f"Blueprint '{blueprint_name}' failed to load — falling back to free-form generation")

    if not await llm_client.is_available():
        return _error_result(
            f"Ollama is not reachable at {config.OLLAMA_HOST}. Start Ollama and try again."
        )

    async def on_progress(message: str) -> None:
        await _progress(message)

    result = await pipeline.generate_workflow(task_prompt, max_nodes, on_progress=on_progress)

    if not result.ok:
        await _progress(f"Error: {result.error}")
        return _error_result(result.error or "generation failed")

    spec = {
        "node_instances": result.node_instances,
        "widgets": result.widgets,
        "links": result.links,
    }
    status = f"Built {len(result.node_instances)} nodes, {len(result.links)} links"
    if result.warnings:
        # Non-fatal problems (e.g. a dangling node) must still reach the user:
        # the build succeeded, but the workflow may not run as intended.
        status += f" — warnings: {'; '.join(result.warnings)}"
    await _progress(f"Done — {status}")

    # Never hardcode this True. A graph with unconnected required inputs is
    # still worth placing (the user can finish the wiring), but it will fail
    # the instant it is queued — announcing it as ready is a false success.
    ready = result.execution_ready
    if not ready and not result.error:
        await _progress(
            "Placed, but not runnable yet — highlighted nodes need connections"
        )
    return {
        "status_message": status,
        "error_detail": "",
        "warnings": result.warnings,
        "workflow_json": json.dumps(spec),
        "execution_ready": ready,
        # Consumed by canvas_writer.js to highlight what was left incomplete.
        "incomplete_inputs": [
            {"node": iid, "input": name} for iid, name in result.incomplete
        ],
        "source": "free-form",
        "subgraph_wrap_threshold": config.SUBGRAPH_WRAP_THRESHOLD,
    }


async def _handle_superseded_mode(mode: str) -> dict:
    """EXPLAIN/DEBUG/SUGGEST were superseded by CHAT, which answers all three
    kinds of question in one conversation (MASTER_PLAN.md Section 3A
    amendment). They are no longer offered by the node, but a workflow saved
    before that change can still send one, so the reply names the replacement
    rather than reading as a missing feature.
    """
    return _error_result(
        f"mode '{mode}' has been replaced by CHAT, which explains nodes, "
        "diagnoses errors and suggests next steps in one conversation. "
        "Set mode = CHAT on this node."
    )


async def _handle_modify(body: dict, sid: str | None, build_id: str | None = None) -> dict:
    """MODIFY mode: edit the workflow already on the canvas.

    Returns a validated diff rather than a workflow. The frontend snapshots
    the canvas before applying it, so the whole edit is undoable.
    """
    async def _progress(message: str) -> None:
        await _send_progress(sid, message, build_id)

    task_prompt = body.get("task_prompt", "")
    if not isinstance(task_prompt, str):
        return _error_result("task_prompt must be a string")
    if not task_prompt.strip():
        return _error_result("task_prompt is empty")

    graph = body.get("graph")
    if not isinstance(graph, dict):
        return _error_result(
            "MODIFY needs the current canvas; none was sent with the request"
        )

    result = await pipeline.modify_workflow(task_prompt, graph, on_progress=_progress)
    if not result.ok or result.diff is None:
        await _progress(f"Error: {result.error}")
        return _error_result(result.error or "modification failed")

    diff = result.diff
    status = f"Modified: {diff.summary()}"
    if result.warnings:
        status += f" - warnings: {'; '.join(result.warnings)}"

    return {
        "status_message": status,
        "error_detail": "",
        "warnings": result.warnings,
        # Not a workflow: the frontend applies this as an edit, not an
        # injection, and must not route it through the BUILD spec path.
        "workflow_json": "{}",
        "diff": diff.to_dict(),
        "execution_ready": False,
        "source": "modify",
    }


_chat_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_CHATS)

VALID_CHAT_ROLES = ("user", "assistant")


def _normalize_messages(raw: Any) -> list[dict[str, str]] | str:
    """Validate the client's conversation history.

    Returns the cleaned message list, or an error string. The history comes
    from the frontend and is replayed straight into the model, so a bad shape
    must be rejected here rather than raising deep inside the LLM client.
    """
    if not isinstance(raw, list) or not raw:
        return "messages must be a non-empty array"

    cleaned: list[dict[str, str]] = []
    # Trim oldest-first: the newest turns carry the actual question.
    for entry in raw[-config.CHAT_MAX_HISTORY_MESSAGES :]:
        if not isinstance(entry, dict):
            return "each message must be an object"
        role = entry.get("role")
        content = entry.get("content")
        if role not in VALID_CHAT_ROLES:
            return f"message role must be one of {VALID_CHAT_ROLES}"
        if not isinstance(content, str) or not content.strip():
            return "message content must be a non-empty string"
        cleaned.append({"role": role, "content": content[: config.CHAT_MAX_MESSAGE_CHARS]})

    if cleaned[-1]["role"] != "user":
        return "the last message must be from the user"
    return cleaned


@routes.post("/agent/chat")
async def chat(request: web.Request) -> web.Response:
    """CHAT mode: free-text Q&A about ComfyUI, the user's graph, and errors.

    Tokens stream over the WebSocket as they arrive; the JSON response carries
    the complete reply so a client that missed the stream still renders it.
    """
    if not _is_authorized(request):
        return web.json_response({"error": "forbidden"}, status=403)
    if not _is_allowed_origin(request):
        return web.json_response({"error": "cross-origin request rejected"}, status=403)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "request body must be a JSON object"}, status=400)

    messages = _normalize_messages(body.get("messages"))
    if isinstance(messages, str):
        return web.json_response({"error": messages}, status=400)

    sid = body.get("sid")
    raw_chat_id = body.get("chat_id")
    chat_id = raw_chat_id if isinstance(raw_chat_id, str) else None
    graph = body.get("graph")
    last_error = body.get("last_error")
    last_error = last_error if isinstance(last_error, str) else ""

    if _chat_semaphore.locked():
        return web.json_response(
            {"error": "too many concurrent chats in progress — retry shortly"}, status=429
        )

    async with _chat_semaphore:
        if not await llm_client.is_available():
            return web.json_response(
                {"error": f"Ollama is not reachable at {config.OLLAMA_HOST}. Start Ollama and try again."},
                status=503,
            )

        question = messages[-1]["content"]
        try:
            system = await chat_context.build_system_prompt(question, graph, last_error)
        except Exception:  # noqa: BLE001
            # Context assembly is an enhancement, not a prerequisite — a
            # failure here must degrade to an unsituated answer, not a 500.
            logger.exception("chat: failed to build system prompt; continuing without context")
            system = chat_context.SYSTEM_PROMPT

        async def on_token(token: str) -> None:
            payload = {"token": token}
            if chat_id:
                payload["chat_id"] = chat_id
            await _ws_send(CHAT_TOKEN_EVENT, payload, sid)

        try:
            reply = await llm_client.chat_stream(messages, system=system, on_token=on_token)
        except llm_client.OllamaUnavailableError as exc:
            return web.json_response({"error": str(exc)}, status=503)
        except llm_client.OllamaGenerationError as exc:
            return web.json_response({"error": str(exc)}, status=502)
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed unexpectedly")
            return web.json_response({"error": f"internal error: {exc}"}, status=500)

    return web.json_response({"reply": reply, "chat_id": chat_id})


@routes.post("/agent/build_workflow")
async def build_workflow(request: web.Request) -> web.Response:
    if not _is_authorized(request):
        return web.json_response({"error": "forbidden"}, status=403)
    if not _is_allowed_origin(request):
        return web.json_response({"error": "cross-origin request rejected"}, status=403)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "request body must be a JSON object"}, status=400)

    mode = body.get("mode", "BUILD")
    sid = body.get("sid")
    # Echoed back on every progress event so the frontend can route messages
    # to the originating node rather than broadcasting to all agent nodes.
    raw_build_id = body.get("build_id")
    build_id = raw_build_id if isinstance(raw_build_id, str) else None

    # Mode is checked first so the BUILD path — which can succeed via a
    # blueprint with Ollama entirely absent — never pays for this probe.
    # Evaluating it eagerly cost every blueprint build a round-trip, up to the
    # full timeout when Ollama was offline, which is exactly the case the
    # blueprint path exists to serve.
    if mode != "BUILD" and not await llm_client.is_available():
        return web.json_response(
            _error_result(f"Ollama is not reachable at {config.OLLAMA_HOST}."),
            status=503,
        )

    if _build_semaphore.locked():
        return web.json_response(
            _error_result("too many concurrent builds in progress — retry shortly"),
            status=429,
        )

    try:
        async with _build_semaphore:
            if mode == "BUILD":
                result = await _handle_build(body, sid, build_id)
            elif mode == "CHAT":
                # CHAT is served by /agent/chat, which streams tokens; routing
                # it through the build response shape would drop the stream.
                return web.json_response(
                    _error_result("mode 'CHAT' is served by POST /agent/chat"), status=400
                )
            elif mode == "MODIFY":
                result = await _handle_modify(body, sid, build_id)
            elif mode in ("EXPLAIN", "DEBUG", "SUGGEST"):
                # Kept in the dispatch though the node no longer offers them:
                # a workflow saved before they were removed still sends one.
                result = await _handle_superseded_mode(mode)
            else:
                return web.json_response({"error": f"unknown mode '{mode}'"}, status=400)
    except llm_client.OllamaUnavailableError as exc:
        return web.json_response(_error_result(str(exc)), status=503)
    except Exception as exc:  # noqa: BLE001
        # Safety net: an unexpected pipeline failure must never leak a raw 500
        # stack trace to the frontend. Report it in the same shape as the other
        # error paths and log the traceback server-side.
        logger.exception("build_workflow failed unexpectedly")
        return web.json_response(_error_result(f"internal error: {exc}"), status=500)

    return web.json_response(result)
