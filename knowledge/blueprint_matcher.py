"""Keyword -> blueprint template selection. Section 3C / 3B.

Blueprint-first is the primary reliability mechanism with the 0.8B model
(Section 4A), so this runs before any LLM call, not after a failed one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("comfy_agent.knowledge.blueprint_matcher")

SUBGRAPHS_DIR = Path(__file__).resolve().parent.parent / "blueprint_defs"

# Each blueprint's trigger keywords, checked against the lowercased task_prompt.
# Order matters only for tie-breaking (first match with the highest score wins).
BLUEPRINT_KEYWORDS: dict[str, list[str]] = {
    "txt2img_sdxl": ["sdxl", "txt2img", "text to image", "text-to-image"],
    "txt2img_flux": ["flux"],
    "img2img_base": ["img2img", "image to image", "image-to-image", "denoise"],
    "upscale_latent": ["upscale", "hires", "hi-res", "hi res"],
    "face_detailer": ["face", "detailer", "face fix", "face swap"],
    "video_wan": ["video", "wan", "animate"],
    "controlnet_base": ["controlnet", "control net", "depth map", "pose"],
    "lora_stack": ["lora", "loras", "multi-lora", "multi lora"],
}


def list_blueprints() -> list[str]:
    if not SUBGRAPHS_DIR.exists():
        return []
    return sorted(p.stem for p in SUBGRAPHS_DIR.glob("*.json"))


def load_blueprint(name: str) -> dict[str, Any] | None:
    """Load a blueprint JSON by name. Returns None (not an exception) on any
    failure — Section 6: blueprint load can fail outright, callers must treat
    it as a fallback trigger, not a crash.
    """
    path = SUBGRAPHS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("blueprint %s failed to load: %s", name, exc)
        return None

    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        logger.warning("blueprint %s does not contain a valid subgraph (missing/invalid 'nodes')", name)
        return None

    # Callers (agent_routes, js/canvas_writer) index both 'nodes' and 'links'
    # directly, so normalize 'links' here rather than letting a hand-edited
    # blueprint without it raise KeyError out of the request handler.
    links = data.setdefault("links", [])
    if not isinstance(links, list):
        logger.warning("blueprint %s has a non-list 'links'", name)
        return None

    return data


# Model-name keywords that override a generic category match — "flux txt2img"
# must pick the FLUX blueprint, not the SDXL one, which is what a flat
# first-highest-score dict-order scan did before.
MODEL_KEYWORD_WEIGHT = 3

# Keywords that name a model family rather than a task. They carry extra
# weight so a generic category match cannot win ("flux txt2img" must pick
# the FLUX blueprint, not the SDXL one). Keep in sync with
# BLUEPRINT_KEYWORDS when adding a model-specific blueprint.
MODEL_KEYWORDS = frozenset({"flux", "sdxl"})


def match(task_prompt: str, task_category: str = "auto", model_family: str = "auto") -> str | None:
    """Return the best-matching blueprint name, or None if nothing scores > 0.

    Scored by weighted keyword hits (model-name keywords like flux/sdxl carry
    extra weight), plus a boost for an explicit model_family or task_category
    matching the blueprint name. Ties break by longest matched keyword instead
    of dict order, so "flux txt2img" resolves to txt2img_flux."""
    text = task_prompt.lower()
    best_name: str | None = None
    best_score = 0
    best_tiebreak = ""

    model_family = (model_family or "auto").strip().lower()

    for name, keywords in BLUEPRINT_KEYWORDS.items():
        score = 0
        longest_hit = ""
        for kw in keywords:
            if kw in text:
                score += MODEL_KEYWORD_WEIGHT if kw in MODEL_KEYWORDS else 1
                if len(kw) > len(longest_hit):
                    longest_hit = kw
        if model_family not in ("auto", "") and model_family in name:
            score += 2
        if task_category != "auto" and task_category in name:
            score += 1
        if score > best_score or (score == best_score and len(longest_hit) > len(best_tiebreak)):
            best_score = score
            best_tiebreak = longest_hit
            best_name = name

    return best_name
