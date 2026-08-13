"""Stage B — Widget Configuration. Section 2E.

One LLM call per validated node (parallelized in pipeline.py), each asking
only for that node's widget values. Type-coerces string -> int/float/bool
against the /object_info widget spec, validates COMBO values against the real
option list, and clamps INT/FLOAT to the widget's declared min/max. Unknown
widget names are dropped (left at node default) — free-form injection is
name-only, there is no positional fallback (Section 2F).
"""

from __future__ import annotations

from typing import Any

from .. import config, llm_client
from ..knowledge.type_validator import is_widget_input, spec_type as _shared_spec_type

SCHEMA = {
    "type": "object",
    "properties": {"widgets": {"type": "object"}},
    "required": ["widgets"],
}

SYSTEM = (
    "You set ComfyUI node widget values for a single node, based on the "
    "task description. Output ONLY a JSON object with a 'widgets' object "
    "mapping widget name to value. Only set widgets relevant to the task; "
    "leave others out to use node defaults."
)


def _widget_spec(node_schema: dict[str, Any], name: str) -> Any:
    for group in ("required", "optional"):
        spec = node_schema.get("input", {}).get(group, {}).get(name)
        if spec is not None:
            return spec
    return None


def _spec_type(spec: Any) -> Any:
    """Type token from an /object_info input spec (V1 list or V3 dict form).

    Delegates to the shared implementation so Stage B and Stage C cannot
    drift apart in how they read a spec.
    """
    return _shared_spec_type(spec)


def _spec_meta(spec: Any) -> dict[str, Any]:
    """Extract the metadata dict from an input spec (default/min/max for
    primitives; option list for combos)."""
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def _combo_options(spec: Any) -> list[Any]:
    """Allowed values for a COMBO input, from either V1 (type is the option
    list) or V3 (meta carries "options")."""
    meta = _spec_meta(spec)
    for key in ("options", "allowed"):
        opts = meta.get(key)
        if isinstance(opts, list):
            return list(opts)
    type_val = _spec_type(spec)
    if isinstance(type_val, list):
        return list(type_val)
    return []


def _widget_names(node_schema: dict[str, Any]) -> list[str]:
    """Widget inputs are required/optional inputs whose type is a primitive
    (STRING/INT/FLOAT/BOOLEAN/COMBO) rather than a link type."""
    inputs = node_schema.get("input", {})
    if not isinstance(inputs, dict):
        return []
    names: list[str] = []
    for group in ("required", "optional"):
        group_inputs = inputs.get(group)
        if not isinstance(group_inputs, dict):
            continue
        for name, spec in group_inputs.items():
            # Shared classifier with Stage C — these two partitions are exact
            # complements and must stay that way.
            if is_widget_input(spec):
                names.append(name)
    return names


def _widget_constraint_hint(node_schema: dict[str, Any], name: str) -> str:
    """Describe the allowed values/range for one widget, so the LLM has enough
    information to produce a value that will actually validate, instead of
    hallucinating (e.g. sampler_name is a fixed enum, not free text)."""
    spec = _widget_spec(node_schema, name)
    if spec is None:
        # `not spec` also matched a legitimately empty spec, dropping the
        # type/range hint the model needs to produce a valid value.
        return name
    type_val = _spec_type(spec)
    meta = _spec_meta(spec)
    options = _combo_options(spec)
    if isinstance(type_val, list) or type_val == "COMBO":
        cap = config.WIDGET_OPTION_DISPLAY_CAP
        shown = options[:cap]
        suffix = ", ..." if len(options) > cap else ""
        return f"{name} (choose exactly one of: {', '.join(str(o) for o in shown)}{suffix})"
    if type_val == "INT" or type_val == "FLOAT":
        lo, hi, default = meta.get("min"), meta.get("max"), meta.get("default")
        bounds = f"range {lo}-{hi}" if lo is not None and hi is not None else ""
        return f"{name} ({type_val}{', ' + bounds if bounds else ''}{', default ' + str(default) if default is not None else ''})"
    if type_val == "BOOLEAN":
        return f"{name} (true/false)"
    return f"{name} (text)"


def _coerce(value: Any, node_schema: dict[str, Any], widget_name: str) -> Any | None:
    """Returns None if the value is invalid for this widget and should be
    dropped (left at the node's own default) rather than injected as-is —
    this is what stops a hallucinated combo value like "SDXL" for
    sampler_name from ever reaching the canvas."""
    spec = _widget_spec(node_schema, widget_name)
    if spec is None:
        return value
    type_val = _spec_type(spec)

    options = _combo_options(spec)
    if isinstance(type_val, list) or type_val == "COMBO":
        # COMBO: only accept values that are actually in the allowed option list.
        if value in options:
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for opt in options:
                if str(opt).lower() == lowered:
                    return opt
        return None

    try:
        if type_val == "INT":
            coerced = int(value)
        elif type_val == "FLOAT":
            coerced = float(value)
        elif type_val == "BOOLEAN":
            coerced = bool(value) if not isinstance(value, str) else value.lower() in ("true", "1", "yes")
        else:
            return value
    except (TypeError, ValueError):
        return None

    if type_val in ("INT", "FLOAT"):
        meta = _spec_meta(spec)
        if meta.get("min") is not None:
            coerced = max(coerced, meta["min"])
        if meta.get("max") is not None:
            coerced = min(coerced, meta["max"])
    return coerced


async def configure_widgets(
    task_prompt: str,
    class_type: str,
    node_schema: dict[str, Any],
    role_index: int | None = None,
    role_count: int | None = None,
) -> dict[str, Any]:
    widget_names = _widget_names(node_schema)
    if not widget_names:
        return {}

    hints = [_widget_constraint_hint(node_schema, name) for name in widget_names]
    role_context = ""
    if role_index is not None and role_count is not None and role_count > 1:
        role_context = f"\nThis is instance {role_index + 1} of {role_count} of this node type."
        if class_type.startswith("CLIPTextEncode"):
            role_context += " (positive prompt encoder)" if role_index == 0 else " (negative prompt encoder)"
    prompt = f"Task: {task_prompt}\n\nNode: {class_type}{role_context}\nWidgets available: {'; '.join(hints)}"

    try:
        result, _model = await llm_client.generate_json_with_fallback(prompt=prompt, schema=SCHEMA, system=SYSTEM)
    except llm_client.OllamaGenerationError:
        # Non-fatal: fall back to node defaults rather than failing the whole build.
        return {}

    raw_widgets = result.get("widgets", {})
    coerced: dict[str, Any] = {}
    for name, value in raw_widgets.items():
        if name not in widget_names:
            # Unrecognized widget names are dropped here (left at the node's own
            # default). Free-form injection is name-only — there is no positional
            # fallback, unlike the blueprint path, because free-form specs have no
            # stable widget order to index against.
            continue
        result_value = _coerce(value, node_schema, name)
        if result_value is not None:
            coerced[name] = result_value
    return coerced
