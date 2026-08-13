"""Semantic index over everything installed in this ComfyUI.

The whole registry cannot be prompted directly: a real install is ~2.7 MB /
1767 node types (~680k tokens), far past even OLLAMA_CHAT_NUM_CTX, and a small
model degrades badly long before that limit anyway. So every node is embedded
once, cached to disk, and each question retrieves only its top-K matches.

Two deliberate properties:

* Embeddings are cached keyed by a fingerprint of the node registry, so the
  1-2 minute build happens once and is reused until you install or remove
  custom nodes.
* Retrieval degrades to keyword scoring when Ollama or the embed model is
  unavailable. A chat with slightly worse node lookup beats a chat that is
  down, and this path also covers a first run where the index is still warming.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import pathlib
import re
import time
from typing import Any

import aiohttp

from . import config, llm_client
from .knowledge.type_validator import node_output_names

logger = logging.getLogger("comfy_agent.install_index")

# Splits CamelCase and acronym boundaries: "FaceDetailer" -> "Face Detailer",
# "VAEDecode" -> "VAE Decode".
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_CACHE_PATH = pathlib.Path(__file__).parent / config.INDEX_CACHE_FILENAME

# Build state. The index is built lazily on first use and guarded so that N
# concurrent chats trigger exactly one build rather than N.
_index: dict[str, Any] | None = None
_build_lock = asyncio.Lock()
_build_failed_reason: str = ""


# --------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------

def _describe_node(class_type: str, entry: dict[str, Any]) -> str:
    """One embeddable line per node: what it is called, what it does, what it
    connects to. This text is what a question is matched against, so it should
    read like something a user would ask about."""
    parts = [class_type]

    # Embedding models tokenize CamelCase poorly: "FaceDetailer" does not
    # reliably match "make faces look better", because the concept the user
    # asked about is fused into one token. Splitting to "Face Detailer" is what
    # makes name-carried meaning searchable — and node names carry most of the
    # meaning here, since many descriptions never repeat the key word.
    spaced = _CAMEL_RE.sub(" ", class_type).replace("_", " ")
    if spaced != class_type:
        parts.append(spaced)

    display = entry.get("display_name")
    if display and display != class_type:
        parts.append(str(display))

    category = entry.get("category")
    if category:
        parts.append(str(category).replace("/", " "))

    description = entry.get("description")
    if description:
        # Descriptions can be paragraphs; the leading sentence carries the
        # signal and keeps every node's vector comparable in length.
        parts.append(str(description).strip().split("\n")[0][:300])

    inputs = entry.get("input", {})
    names: list[str] = []
    if isinstance(inputs, dict):
        for group in ("required", "optional"):
            group_inputs = inputs.get(group)
            if isinstance(group_inputs, dict):
                names.extend(str(k) for k in group_inputs)
    if names:
        parts.append("inputs: " + ", ".join(names[:12]))

    try:
        outputs = node_output_names(entry)
    except Exception:  # noqa: BLE001
        outputs = []
    if outputs:
        parts.append("outputs: " + ", ".join(str(o) for o in outputs[:12]))

    return " | ".join(parts)


def schema_line(class_type: str, entry: dict[str, Any]) -> str:
    """Compact schema line injected into the prompt for a retrieved node."""
    inputs = entry.get("input", {})
    required: list[str] = []
    optional: list[str] = []
    if isinstance(inputs, dict):
        req = inputs.get("required")
        if isinstance(req, dict):
            required = [str(k) for k in req]
        opt = inputs.get("optional")
        if isinstance(opt, dict):
            optional = [str(k) for k in opt]
    try:
        outputs = [str(o) for o in node_output_names(entry)]
    except Exception:  # noqa: BLE001
        outputs = []

    line = f"{class_type}: in=[{','.join(required)}]"
    if optional:
        line += f" opt=[{','.join(optional)}]"
    line += f" out=[{','.join(outputs)}]"
    category = entry.get("category")
    if category:
        line += f" ({category})"
    return line


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{url} returned HTTP {resp.status}")
        return await resp.json()


async def harvest_models() -> dict[str, list[str]]:
    """Enumerate installed model files per folder via /models.

    Folders are fetched concurrently; a folder that errors is skipped rather
    than failing the whole inventory, since one bad path should not cost the
    user their entire model list.
    """
    timeout = aiohttp.ClientTimeout(total=config.COMFY_TIMEOUT_SECONDS * 4)
    inventory: dict[str, list[str]] = {}
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            folders = config.MODEL_FOLDERS or await _fetch_json(
                session, f"{config.COMFY_HOST}/models"
            )
            if not isinstance(folders, list):
                return {}

            async def one(folder: str) -> tuple[str, list[str]]:
                try:
                    files = await _fetch_json(
                        session, f"{config.COMFY_HOST}/models/{folder}"
                    )
                    if not isinstance(files, list):
                        return folder, []
                    # Weights only: these folders also surface source files
                    # (.editorconfig, .md, .py), which are not models and must
                    # not be offered to the user as though they were.
                    kept = [
                        str(f)
                        for f in files
                        if str(f).lower().endswith(config.MODEL_FILE_EXTENSIONS)
                    ]
                    return folder, kept
                except Exception:  # noqa: BLE001
                    logger.debug("could not list model folder %r", folder, exc_info=True)
                    return folder, []

            wanted = [
                str(f) for f in folders if str(f) not in config.MODEL_FOLDER_DENYLIST
            ]
            results = await asyncio.gather(*(one(f) for f in wanted))
            for folder, files in results:
                if files:
                    inventory[folder] = files
    except Exception as exc:  # noqa: BLE001
        logger.warning("model inventory unavailable: %s", exc)
        return {}
    return inventory


def _fingerprint(object_info: dict[str, Any]) -> str:
    """Identity of the current node registry.

    Keyed on the sorted node-type names, so installing or removing a custom
    node pack invalidates the cache but a mere server restart does not.
    """
    joined = "\x00".join(sorted(object_info.keys()))
    return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _load_cache(fingerprint: str) -> dict[str, Any] | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("ignoring unreadable index cache: %s", exc)
        return None
    if cached.get("fingerprint") != fingerprint:
        logger.info("node registry changed — rebuilding install index")
        return None
    if not cached.get("vectors"):
        return None
    return cached


def _save_cache(payload: dict[str, Any]) -> None:
    try:
        tmp = _CACHE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        # Atomic swap: a half-written cache would otherwise be loaded on the
        # next start and silently poison retrieval.
        tmp.replace(_CACHE_PATH)
    except OSError as exc:
        logger.warning("could not write index cache: %s", exc)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

async def _embed_all(texts: list[str], on_progress=None) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        vectors.extend(await llm_client.embed(batch))
        if on_progress is not None:
            await on_progress(min(start + len(batch), total), total)
    return vectors


async def build(force: bool = False, on_progress=None) -> dict[str, Any]:
    """Build (or load) the index. Safe to call concurrently."""
    global _index, _build_failed_reason

    if _index is not None and not force:
        return _index

    async with _build_lock:
        if _index is not None and not force:
            return _index

        from .knowledge import schema_compactor

        object_info = await schema_compactor.fetch_object_info()
        fingerprint = _fingerprint(object_info)

        names = sorted(object_info.keys())
        entries = [object_info[n] if isinstance(object_info[n], dict) else {} for n in names]
        descriptions = [_describe_node(n, e) for n, e in zip(names, entries)]
        schema_lines = [schema_line(n, e) for n, e in zip(names, entries)]

        models = await harvest_models()

        vectors: list[list[float]] = []
        cached = None if force else _load_cache(fingerprint)
        if cached and len(cached["vectors"]) == len(names):
            vectors = cached["vectors"]
            # Clear any reason recorded by an earlier failed build in this
            # process: retrieval is no longer degraded, and a stale reason is
            # reported verbatim by stats() (and so by /agent/health).
            _build_failed_reason = ""
            logger.info("loaded install index from cache (%d nodes)", len(names))
        else:
            started = time.monotonic()
            logger.info("building install index: embedding %d nodes...", len(names))
            try:
                vectors = await _embed_all(descriptions, on_progress=on_progress)
                _save_cache({"fingerprint": fingerprint, "vectors": vectors})
                logger.info(
                    "install index built in %.1fs (%d nodes)",
                    time.monotonic() - started,
                    len(names),
                )
                _build_failed_reason = ""
            except Exception as exc:  # noqa: BLE001
                # Keyword retrieval still works without vectors, so this is a
                # degradation rather than a failure.
                logger.warning("embedding failed (%s) — falling back to keyword retrieval", exc)
                _build_failed_reason = str(exc)
                vectors = []

        _index = {
            "names": names,
            "descriptions": descriptions,
            "schema_lines": schema_lines,
            "vectors": vectors,
            "models": models,
            "fingerprint": fingerprint,
        }
        return _index


def is_ready() -> bool:
    return _index is not None


# Set once the first successful build finishes, so a caller can distinguish
# "still warming" from "built" without racing the build lock.
_ready_event: asyncio.Event | None = None


def ready_event() -> asyncio.Event:
    """Signalled when the index is usable.

    Created lazily rather than at import: asyncio.Event binds to the running
    loop on first await in older Python versions, and this module is imported
    long before ComfyUI's loop exists.
    """
    global _ready_event
    if _ready_event is None:
        _ready_event = asyncio.Event()
    if _index is not None:
        _ready_event.set()
    return _ready_event


async def warm(retry_delay: float = 5.0, max_attempts: int = 3) -> None:
    """Build the index in the background, once the server can answer.

    Without this the 1767-node build ran inside the first CHAT request, which
    blocked that request for the full build. Retries because this starts with
    ComfyUI itself: /object_info is not served for the first moments after
    launch, and a single failed attempt would leave every later chat paying
    the cost this function exists to remove.
    """
    event = ready_event()
    for attempt in range(1, max_attempts + 1):
        try:
            await build()
            event.set()
            logger.info("install index warm complete (%s)", stats())
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                # Not fatal: retrieval degrades to keyword scoring, and the
                # next chat request rebuilds via build()'s own lazy path.
                logger.warning(
                    "install index warm failed after %d attempts (%s); "
                    "retrieval will degrade to keyword search",
                    max_attempts,
                    exc,
                )
                return
            logger.debug("index warm attempt %d failed (%s); retrying", attempt, exc)
            await asyncio.sleep(retry_delay)


def stats() -> dict[str, Any]:
    if _index is None:
        # `building` distinguishes a warm already in flight from an index that
        # nothing has asked for yet, so the UI can say "building" rather than
        # implying retrieval is broken.
        return {"ready": False, "building": _build_lock.locked()}
    return {
        "ready": True,
        "nodes": len(_index["names"]),
        "embedded": bool(_index["vectors"]),
        "model_folders": len(_index["models"]),
        "total_model_files": sum(len(v) for v in _index["models"].values()),
        "degraded_reason": _build_failed_reason,
    }


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that are filler in a question but structural in node names. Boosting
# them wrecks ranking: "how do I MAKE faces look better?" scored
# MakeTrainingDataset (1.035) above FaceDetailer (0.641) purely on "make".
# Only the name bonus consults this — the embedding still sees the full text.
_STOPWORDS = frozenset(
    """
    make making made get getting got set setting put use using used
    load loading save saving show add adding create creating build building
    do does doing how what which where when why who whom whose
    can could should would will shall may might must
    the a an and or but if then than that this these those there here
    is are was were be been being am i you it its my me we our your
    to of in on at for with from into by as about like over under
    have has had want need help please tell explain thing things stuff
    my mine better best good bad new old more less most least
    """.split()
)


def _keyword_scores(question: str, descriptions: list[str]) -> list[float]:
    terms = {
        t
        for t in _WORD_RE.findall(question.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }
    if not terms:
        return [0.0] * len(descriptions)
    scores = []
    for text in descriptions:
        haystack = text.lower()
        # Substring rather than token equality so "upscal" matches "upscale"
        # and a partial node name still hits.
        scores.append(float(sum(1 for t in terms if t in haystack)))
    return scores


def _name_match_bonus(question: str, names: list[str]) -> list[float]:
    """Boost nodes whose *name* the question actually says.

    Pure vector similarity systematically buries the plain node under its own
    variants: "what does KSampler do" ranked KSamplerAdvanced, KSamplerProvider
    and KSampler (Efficient) above KSampler itself, because longer descriptive
    text scores better. An exact name mention is the strongest signal a user
    can give, so it is scored separately from semantics.
    """
    terms = [
        t
        for t in _WORD_RE.findall(question.lower())
        if len(t) > 2 and t not in _STOPWORDS
    ]
    # Users type plurals ("faces", "loras", "samplers"); node names are almost
    # always singular ("FaceDetailer", "LoraLoader"), so without this the exact
    # word the user cared about scores zero on the name.
    for term in list(terms):
        if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            singular = term[:-1]
            if singular not in _STOPWORDS and singular not in terms:
                terms.append(singular)

    if not terms:
        return [0.0] * len(names)

    bonuses = []
    for name in names:
        lowered = name.lower()
        compact = _WORD_RE.sub("", lowered)  # "KSampler (Efficient)" -> "ksamplerefficient"
        best = 0.0
        for term in terms:
            if lowered == term:
                # Exact class name — the user named this node outright.
                best = max(best, 1.0)
            elif compact == term:
                best = max(best, 0.9)
            elif lowered.startswith(term) or compact.startswith(term):
                # "ksampler" matching "KSamplerAdvanced": relevant, but must
                # not outrank the exact node it is a variant of.
                best = max(best, 0.45)
            elif term in lowered:
                best = max(best, 0.2)
        bonuses.append(best)
    return bonuses


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


async def search(question: str, top_k: int | None = None) -> list[str]:
    """Return schema lines for the nodes most relevant to `question`."""
    index = await build()
    if top_k is None:
        top_k = config.CHAT_RETRIEVAL_TOP_K

    names = index["names"]
    scores = _keyword_scores(question, index["descriptions"])
    name_bonus = _name_match_bonus(question, names)

    if index["vectors"]:
        try:
            query_vec = (await llm_client.embed([question]))[0]
            for i, vec in enumerate(index["vectors"]):
                # Blend three signals: semantics find paraphrases ("make faces
                # better" -> FaceDetailer), keyword overlap grounds it, and the
                # name bonus dominates when the user names a node outright.
                # Cosine here lands ~0.4-0.7, so a 1.0 exact-name bonus wins.
                scores[i] = _cosine(query_vec, vec) + 0.1 * scores[i] + name_bonus[i]
        except Exception as exc:  # noqa: BLE001
            logger.debug("query embedding failed, keyword-only for this turn: %s", exc)
            for i in range(len(scores)):
                scores[i] += 2.0 * name_bonus[i]
    else:
        for i in range(len(scores)):
            scores[i] += 2.0 * name_bonus[i]

    ranked = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
    return [index["schema_lines"][i] for i in ranked[:top_k] if scores[i] > 0]


def render_model_inventory() -> str:
    """Installed models per folder, capped so a huge library cannot flood the
    prompt."""
    if _index is None or not _index["models"]:
        return ""
    lines = []
    for folder, files in sorted(_index["models"].items()):
        if not files:
            continue
        shown = files[: config.MODEL_LIST_CAP]
        suffix = "" if len(files) <= config.MODEL_LIST_CAP else f" (+{len(files) - len(shown)} more)"
        lines.append(f"{folder} ({len(files)}): {', '.join(shown)}{suffix}")
    return "\n".join(lines)
