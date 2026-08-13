"""Check the LiteGraph APIs the JS layer calls actually exist in the frontend.

js/*.js is tested against stubs (tests/canvas_writer.test.mjs), which proves
the logic but not the *contract*: a stub cannot tell you that
`graph.convertToSubgraph` was renamed, or that `remove()` no longer fires
`onRemoved`. Those break silently in the browser, where nothing in the offline
suite would notice.

This test reads the real `comfyui_frontend_package` when one is installed and
verifies each API is still present. It SKIPS cleanly when the package is
absent, so the offline suite stays runnable everywhere.

Verified against comfyui_frontend_package 1.48.7.

    pytest tests/                       # skips without the frontend package
    python tests/test_frontend_api.py   # standalone
"""

import pathlib
import sys

# Every LiteGraph/ComfyUI API the JS layer depends on, with the file that
# calls it. Keep this in sync when js/ starts using a new frontend API.
REQUIRED_APIS = {
    "createNode": "canvas_writer.js — node instantiation",
    "setDirtyCanvas": "canvas_writer.js — repaint after injection",
    "convertToSubgraph": "canvas_writer.js — subgraph wrapping above threshold",
    "centerOnNode": "canvas_writer.js — pan to the result after a build",
    "addDOMWidget": "agent_node_ui.js — hosts the CHAT panel",
    "onRemoved": "agent_node_ui.js — per-node teardown",
    "onSerialize": "agent_node_ui.js — persists chat history with the workflow",
    "onConfigure": "agent_node_ui.js — restores chat history on load",
    "comfyClass": "agent_node_ui.js — identifies our nodes on the canvas",
    "registerExtension": "agent_node_ui.js — extension entry point",
    "nodeCreated": "agent_node_ui.js — per-node setup hook",
}

# Frontend behaviours the JS relies on that a stub cannot model.
REQUIRED_BEHAVIOURS = {
    "disconnectInput": "graph.remove() must drop incoming links (applyDiff)",
    "disconnectOutput": "graph.remove() must drop outgoing links (applyDiff)",
}


def _asset_dirs():
    """Frontend asset directories, searched across installed environments."""
    found = []
    for base in sys.path:
        candidate = pathlib.Path(base) / "comfyui_frontend_package" / "static" / "assets"
        if candidate.is_dir():
            found.append(candidate)
    return found


def _js_sources(assets: pathlib.Path):
    # .map files are source maps, not shipped code — searching them would
    # report an API as present after it was removed from the bundle.
    return [p for p in assets.glob("*.js") if not p.name.endswith(".map")]


def _haystack(assets: pathlib.Path) -> str:
    parts = []
    for path in _js_sources(assets):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def test_frontend_apis_still_exist():
    dirs = _asset_dirs()
    if not dirs:
        print("skip: comfyui_frontend_package not installed in this environment")
        return

    haystack = _haystack(dirs[0])
    assert haystack, f"no readable JS found in {dirs[0]}"

    missing = [name for name in REQUIRED_APIS if name not in haystack]
    assert not missing, (
        "frontend APIs used by js/ are absent from the installed frontend: "
        + ", ".join(f"{n} ({REQUIRED_APIS[n]})" for n in missing)
    )
    print(f"ok: all {len(REQUIRED_APIS)} frontend APIs present in {dirs[0].parent.parent.name}")


def test_frontend_behaviours_still_hold():
    """applyDiff calls graph.remove() and does not clean links itself, because
    LiteGraph's remove() disconnects them. If that stops being true, removing a
    node would leave dangling links behind."""
    dirs = _asset_dirs()
    if not dirs:
        print("skip: comfyui_frontend_package not installed in this environment")
        return

    haystack = _haystack(dirs[0])
    missing = [name for name in REQUIRED_BEHAVIOURS if name not in haystack]
    assert not missing, (
        "frontend behaviours js/ relies on are gone: "
        + ", ".join(f"{n} ({REQUIRED_BEHAVIOURS[n]})" for n in missing)
    )
    print(f"ok: all {len(REQUIRED_BEHAVIOURS)} relied-on frontend behaviours present")


def test_our_extension_files_are_all_present():
    """WEB_DIRECTORY ships js/; a missing file 404s at runtime with no error
    anywhere in the Python logs."""
    js_dir = pathlib.Path(__file__).resolve().parent.parent / "js"
    expected = {
        "agent_node_ui.js",
        "canvas_writer.js",
        "chat_panel.js",
        "progress_listener.js",
        "agent_style.css",
    }
    present = {p.name for p in js_dir.iterdir() if p.is_file()}
    assert expected <= present, f"missing from js/: {sorted(expected - present)}"
    print(f"ok: all {len(expected)} extension assets present")


def test_js_imports_resolve_to_real_frontend_modules():
    """js/ imports ../../scripts/app.js and api.js. Those are shims re-exporting
    window.comfyAPI; if the names change, every import fails at load."""
    js_dir = pathlib.Path(__file__).resolve().parent.parent / "js"
    imported = set()
    for path in js_dir.glob("*.js"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "../../scripts/" in line:
                imported.add(line.split("../../scripts/")[1].split('"')[0])
    assert imported <= {"app.js", "api.js"}, (
        f"js/ imports an unexpected core module: {sorted(imported)}"
    )
    print(f"ok: js/ imports only known core modules: {sorted(imported)}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall frontend API tests passed")
