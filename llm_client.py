"""Async Ollama client. Section 2A — never a blocking `requests` call.

Every generation stage goes through `generate_json`, which enforces a JSON
schema via Ollama's `format` grammar constraint rather than merely hinting
at one in the prompt (Section 2E / 4A: this matters more with a sub-1B model).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

from . import config

logger = logging.getLogger("comfy_agent.llm")


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama cannot be reached at all (offline, wrong port, etc.)."""


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama responded but generation failed or returned invalid JSON."""


async def is_available() -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=config.OLLAMA_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{config.OLLAMA_HOST}/api/tags") as resp:
                return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


async def list_installed() -> set[str]:
    """Model names Ollama actually holds, from /api/tags.

    Returns an empty set on any failure rather than raising: the caller treats
    "cannot enumerate" the same as "cannot verify", and falls through to the
    configured names so a transient tags failure never blocks a build that
    would otherwise work.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=config.OLLAMA_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{config.OLLAMA_HOST}/api/tags") as resp:
                if resp.status != 200:
                    return set()
                body = await resp.json()
    except Exception:  # noqa: BLE001
        return set()

    names: set[str] = set()
    for entry in body.get("models", []) or []:
        name = entry.get("name") or entry.get("model") or ""
        if not name:
            continue
        names.add(name)
        # Ollama reports "qwen3.5:0.8b"; a Modelfile build is often referenced
        # without its ":latest" suffix. Index both so either spelling resolves.
        if name.endswith(":latest"):
            names.add(name[: -len(":latest")])
        else:
            names.add(f"{name}:latest")
    return names


# Resolved once per process. The alternative is an /api/tags round trip per
# stage — and Stage B fans out one call per node under asyncio.gather, which
# would turn a 12-node build into 12 redundant probes.
_resolved: tuple[str, str] | None = None
_resolve_lock = asyncio.Lock()


async def resolve_models(force: bool = False) -> tuple[str, str]:
    """Return (primary, fallback) that are actually installed.

    PRIMARY_MODEL is a locally-built LoRA that is deliberately not on the
    Ollama registry (see config). When it is missing, every Stage A call
    otherwise fails with HTTP 404 -> retry -> escalate to a fallback that is
    usually missing too, surfacing as "all generation attempts failed" with no
    indication that the cause is a model name rather than a bad prompt.

    Degrading to BASE_MODEL with a loud warning is strictly better: the node
    works out of the box, and the log says exactly how to get the tuned model.
    """
    global _resolved
    if _resolved is not None and not force:
        return _resolved

    async with _resolve_lock:
        if _resolved is not None and not force:
            return _resolved

        installed = await list_installed()
        primary = config.PRIMARY_MODEL
        fallback = config.FALLBACK_MODEL

        if not installed:
            # Could not enumerate. Trust the config rather than guessing.
            logger.debug("could not list installed models; using configured names")
            _resolved = (primary, fallback)
            return _resolved

        if primary not in installed:
            # ASCII only: ComfyUI's console on Windows is cp1252, and a mojibake
            # dash in the one message a stuck user actually reads is a poor
            # trade for typography.
            logger.warning(
                "PRIMARY_MODEL %r is not installed - falling back to %r. "
                "Workflow quality will be lower (the tuned model measured +25%% "
                "link completeness). To build it: `ollama create %s -f Modelfile`. "
                "To silence this, set PRIMARY_MODEL = BASE_MODEL in config.py.",
                primary,
                config.BASE_MODEL,
                primary,
            )
            primary = config.BASE_MODEL

        if primary not in installed:
            logger.error(
                "Neither PRIMARY_MODEL %r nor BASE_MODEL %r is installed. "
                "Run `ollama pull %s` before using the agent node.",
                config.PRIMARY_MODEL,
                config.BASE_MODEL,
                config.BASE_MODEL,
            )

        if fallback not in installed:
            chosen = next(
                (m for m in config.FALLBACK_CHAIN if m in installed), primary
            )
            if chosen != fallback:
                logger.info(
                    "FALLBACK_MODEL %r is not installed; escalating to %r instead",
                    fallback,
                    chosen,
                )
            fallback = chosen

        _resolved = (primary, fallback)
        return _resolved


async def generate_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str = config.PRIMARY_MODEL,
    system: str | None = None,
) -> dict[str, Any]:
    """Call Ollama's /api/generate with a JSON-schema `format` constraint.

    Raises OllamaUnavailableError if Ollama cannot be reached, and
    OllamaGenerationError if it returns something that doesn't parse as JSON
    matching the schema shape (caller decides whether to retry / escalate).
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    if system:
        payload["system"] = system

    timeout = aiohttp.ClientTimeout(total=config.OLLAMA_GENERATE_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{config.OLLAMA_HOST}/api/generate", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise OllamaGenerationError(f"Ollama returned HTTP {resp.status}: {text[:300]}")
                try:
                    body = await resp.json()
                except json.JSONDecodeError as exc:
                    raise OllamaGenerationError(
                        f"Ollama returned a non-JSON body (HTTP {resp.status})"
                    ) from exc
    except aiohttp.ClientError as exc:
        raise OllamaUnavailableError(f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}") from exc
    except (TimeoutError, asyncio.TimeoutError) as exc:
        # asyncio.TimeoutError is the builtin TimeoutError only on Python 3.11+;
        # on 3.10 they are distinct classes, so catch both explicitly.
        raise OllamaUnavailableError(f"Ollama timed out after {config.OLLAMA_GENERATE_TIMEOUT_SECONDS}s") from exc

    # Thinking models (e.g. the qwen3.5 family) can still route the actual
    # answer into the "thinking" field and leave "response" empty even with
    # think=False honored inconsistently across versions — fall back to
    # parsing "thinking" rather than treating an empty "response" as failure.
    raw = body.get("response", "") or body.get("thinking", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaGenerationError(f"Ollama response was not valid JSON: {raw[:300]}") from exc


async def chat_stream(
    messages: list[dict[str, str]],
    *,
    system: str | None = None,
    model: str | None = None,
    on_token=None,
) -> str:
    """Stream a free-text reply from Ollama's /api/chat.

    Deliberately does NOT pass `format`: every other call in this module
    constrains generation to a JSON schema, which is right for the build
    pipeline and exactly wrong for conversation — a schema-constrained model
    answers "why is my image black?" with a JSON object.

    `on_token` is awaited once per chunk so the frontend can render the reply
    as it arrives; a 0.8B model on CPU can take 30s for a paragraph, and a
    node that sits blank for that long reads as hung. Returns the full text.
    """
    if model is None:
        model, _ = await resolve_models()

    payload: dict[str, Any] = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        "stream": True,
        "think": False,
        # Higher than the 0.1 used for graph generation: this is prose, and
        # near-greedy decoding makes explanations repetitive and stilted.
        "options": {"temperature": 0.6, "num_ctx": config.OLLAMA_CHAT_NUM_CTX},
    }

    timeout = aiohttp.ClientTimeout(total=config.OLLAMA_CHAT_TIMEOUT_SECONDS)
    parts: list[str] = []
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{config.OLLAMA_HOST}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise OllamaGenerationError(f"Ollama returned HTTP {resp.status}: {text[:300]}")
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # A partial frame mid-stream is not fatal; the next
                        # newline-delimited object usually completes fine.
                        logger.debug("skipping unparseable chat chunk: %r", line[:120])
                        continue
                    token = (chunk.get("message") or {}).get("content", "")
                    if token:
                        parts.append(token)
                        if on_token is not None:
                            await on_token(token)
                    if chunk.get("done"):
                        break
    except aiohttp.ClientError as exc:
        raise OllamaUnavailableError(f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}") from exc
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise OllamaUnavailableError(
            f"Ollama timed out after {config.OLLAMA_CHAT_TIMEOUT_SECONDS}s"
        ) from exc

    text = "".join(parts).strip()
    if not text:
        raise OllamaGenerationError("Ollama returned an empty reply")
    return text


async def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts via Ollama's /api/embed.

    Returns one vector per input, in order. Raises OllamaUnavailableError if
    Ollama cannot be reached so the caller can degrade to keyword matching
    rather than losing retrieval entirely.
    """
    if not texts:
        return []
    payload = {"model": model or config.EMBED_MODEL, "input": texts}

    timeout = aiohttp.ClientTimeout(total=config.OLLAMA_GENERATE_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{config.OLLAMA_HOST}/api/embed", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise OllamaGenerationError(f"embed returned HTTP {resp.status}: {text[:200]}")
                body = await resp.json()
    except aiohttp.ClientError as exc:
        raise OllamaUnavailableError(f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}") from exc
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise OllamaUnavailableError("Ollama timed out while embedding") from exc

    vectors = body.get("embeddings") or []
    if len(vectors) != len(texts):
        raise OllamaGenerationError(
            f"embed returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors


async def generate_json_with_fallback(
    prompt: str,
    schema: dict[str, Any],
    *,
    system: str | None = None,
    stage_retry_cap: int = config.STAGE_RETRY_CAP,
    fallback_retry_cap: int = config.FALLBACK_RETRY_CAP,
) -> tuple[dict[str, Any], str]:
    """Retry on the primary model, then escalate to the fallback model.

    Returns (result, model_used). Raises OllamaGenerationError if every
    attempt across both models fails.
    """
    last_error: Exception | None = None

    # Resolved, not configured: PRIMARY_MODEL may name a LoRA this machine has
    # never built. Callers compare the returned name against config.PRIMARY_MODEL
    # to decide whether to escalate, so returning the *resolved* name keeps that
    # comparison honest — a run on the base model must not look like a run on
    # the tuned one.
    primary, fallback = await resolve_models()

    for attempt in range(stage_retry_cap + 1):
        try:
            result = await generate_json(prompt, schema, model=primary, system=system)
            return result, primary
        except OllamaGenerationError as exc:
            last_error = exc
            logger.warning("primary model attempt %d/%d failed: %s", attempt + 1, stage_retry_cap + 1, exc)

    if fallback == primary:
        # Nothing left to escalate to; say so rather than burning identical retries.
        raise OllamaGenerationError(
            f"generation failed on {primary} and no distinct fallback model is "
            f"installed; last error: {last_error}"
        )

    logger.info("escalating to fallback model %s", fallback)
    for attempt in range(fallback_retry_cap + 1):
        try:
            result = await generate_json(prompt, schema, model=fallback, system=system)
            return result, fallback
        except OllamaGenerationError as exc:
            last_error = exc
            logger.warning("fallback model attempt %d/%d failed: %s", attempt + 1, fallback_retry_cap + 1, exc)

    raise OllamaGenerationError(f"all generation attempts failed; last error: {last_error}")
