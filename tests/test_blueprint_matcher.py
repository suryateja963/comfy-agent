"""Blueprint keyword matching. Pure Python; needs neither ComfyUI nor Ollama.

    pytest tests/                          # collected as test_* functions
    python tests/test_blueprint_matcher.py # standalone, no pytest needed
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "knowledge" / "blueprint_matcher.py"
_spec = importlib.util.spec_from_file_location("blueprint_matcher_under_test", _PATH)
blueprint_matcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blueprint_matcher)
match = blueprint_matcher.match


def test_keywords_inside_other_words_do_not_match():
    """Substring matching built a video workflow for "I want...", because
    "wan" is inside "want". Each of these used to pick a blueprint."""
    for prompt in (
        "I want a photo of a cat",
        "a cozy interface mockup on a laptop surface",
        "compose a portrait for my purpose",
        "a swan on a lake",
        "make an explorer character",
    ):
        assert match(prompt) is None, (prompt, match(prompt))
    print("ok: keywords inside other words are ignored")


def test_whole_word_keywords_still_match():
    cases = {
        "a photorealistic portrait, sdxl": "txt2img_sdxl",
        "flux txt2img of a forest": "txt2img_flux",
        "upscale this render": "upscale_latent",
        "fix the face": "face_detailer",
        "generate a wan video of waves": "video_wan",
        "openpose controlnet": "controlnet_base",
        "stack two lora files": "lora_stack",
        "img2img with low denoise": "img2img_base",
    }
    for prompt, expected in cases.items():
        assert match(prompt) == expected, (prompt, match(prompt))
    print("ok: whole-word keywords pick the right blueprint")


def test_plurals_match():
    assert match("fix the faces") == "face_detailer"
    assert match("apply three loras") == "lora_stack"
    assert match("copy the poses with a depth map") == "controlnet_base"
    print("ok: plural keywords match")


def test_multi_word_keywords_match():
    assert match("text to image of a castle") == "txt2img_sdxl"
    assert match("hi res fix please") == "upscale_latent"
    print("ok: multi-word keywords match")


def test_model_keyword_outranks_category():
    """"flux txt2img" names both blueprints' keywords; the model name wins."""
    assert match("flux txt2img") == "txt2img_flux"
    assert match("txt2img", model_family="flux") == "txt2img_flux"
    print("ok: model keywords outrank category keywords")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall blueprint matcher tests passed")
