"""Pass 1 — fast task classification. Section 2D.

~200 tokens, narrow output: which schema categories does this task touch.
Feeds schema_compactor.compact_schema() so Pass 2 never sees the full
/object_info payload.
"""

from __future__ import annotations

from .. import llm_client
from ..knowledge.schema_compactor import CATEGORY_KEYWORDS

CATEGORY_LIST = list(CATEGORY_KEYWORDS.keys())

# Brief descriptions per category — a bare category name like "loaders" or
# "conditioning" gives a small model nothing to map "SD1.5 checkpoint" or
# "positive/negative prompts" onto. Confirmed empirically: without these
# hints, qwen3.5:0.8b regularly classified a plain txt2img request as
# ['sampling'] only, dropping 'loaders' and 'conditioning' entirely — which
# then made CheckpointLoaderSimple/CLIPTextEncode structurally impossible
# for Stage A to select, since they were never even in its candidate pool.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "loaders": "loaders (checkpoint, model, unet, clip, vae, lora files)",
    "sampling": "sampling (KSampler, schedulers, samplers)",
    "latent": "latent (empty/blank latent images)",
    "conditioning": "conditioning (positive/negative prompt text encoding)",
    "image": "image (loading, saving, previewing images)",
    "upscaling": "upscaling (hires fix, image/latent upscale)",
    "controlnet": "controlnet (pose/depth/edge guidance)",
    "video": "video (video generation, animation)",
    "mask": "mask (inpainting, segmentation, masking)",
    "custom": "custom (anything not covered above)",
}

# Any task that reaches free-form generation is building a ComfyUI image/video
# workflow, which virtually always needs some combination of these. Pass 1
# ADDS specialized categories (controlnet, upscaling, video, mask) on top of
# this baseline rather than deciding whether to include the baseline at all —
# a restrictive classifier that can exclude essentials is worse than the
# token-bloat problem the two-pass split was meant to solve (Section 2D).
CORE_CATEGORIES = ["loaders", "sampling", "latent", "conditioning", "image"]

SCHEMA = {
    "type": "object",
    "properties": {"categories": {"type": "array", "items": {"type": "string", "enum": CATEGORY_LIST}}},
    "required": ["categories"],
}

SYSTEM = (
    "You classify a ComfyUI workflow request into schema categories, in "
    "addition to a baseline that is always included. Only name EXTRA "
    "specialized categories this request needs, from: "
    f"{'; '.join(CATEGORY_DESCRIPTIONS[c] for c in CATEGORY_LIST if c not in CORE_CATEGORIES)}. "
    "Output ONLY the extra categories relevant to the request as a JSON "
    "array. If none apply, output an empty array. Nothing else."
)


async def classify_task(task_prompt: str) -> list[str]:
    try:
        result, _model = await llm_client.generate_json_with_fallback(
            prompt=task_prompt,
            schema=SCHEMA,
            system=SYSTEM,
        )
    except (llm_client.OllamaGenerationError, llm_client.OllamaUnavailableError):
        # Fail open: hand Pass 2 every category rather than blocking the build.
        # Exclude the catch-all "custom" bucket — including it would match every
        # uncategorized node and turn the candidate pool into "first N nodes of
        # the registry" regardless of task.
        return [c for c in CATEGORY_LIST if c != "custom"]

    categories = result.get("categories", [])
    valid = [c for c in categories if c in CATEGORY_LIST]
    return sorted(set(CORE_CATEGORIES) | set(valid))
