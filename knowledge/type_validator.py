"""Type-safe link validation. Section 2E Stage C / 2F.

All type strings are normalized to uppercase before comparison, per
Section 2F ("model" -> "MODEL"), so a case mismatch between what the LLM
emits and what /object_info reports never silently breaks a link.
"""

from __future__ import annotations

from typing import Any


# Types that connect to anything. ComfyUI uses "*" for Reroute and for the
# any-type inputs that a large share of custom node packs rely on; an empty
# type string is treated the same way rather than as a real mismatch.
WILDCARD_TYPES = frozenset({"*", "ANY", ""})

# Input types that live in the node's widgets rather than on the graph.
WIDGET_TYPES = frozenset({"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"})


def normalize_type(type_str: Any) -> str:
    if isinstance(type_str, dict):
        # V3 /object_info format: entries may be {"type": "...", "name": "..."}.
        type_str = type_str.get("type") or type_str.get("name") or ""
    if isinstance(type_str, list):
        # COMBO inputs are represented as a list of options in /object_info;
        # treat as a single opaque type.
        return "COMBO"
    return str(type_str).upper().strip()


def spec_type(spec: Any) -> Any:
    """Extract the type token from an /object_info input spec, tolerating both
    V1 list form ([type, {meta}]) and V3 dict form ({"type": ..., ...}).

    Canonical implementation — Stage B and Stage C both classify inputs
    through this so their widget/link partitions can never drift apart.
    """
    if isinstance(spec, dict):
        return spec.get("type")
    if isinstance(spec, list) and spec:
        return spec[0]
    return spec


def is_widget_input(spec: Any) -> bool:
    """True if this input is a widget primitive (lives in node.widgets), False
    if it can receive a graph link. Stage B keeps these; Stage C drops them."""
    type_val = spec_type(spec)
    if isinstance(type_val, list):
        return True  # COMBO, represented as its option list
    return type_val in WIDGET_TYPES


def node_output_types(node_schema: dict[str, Any]) -> list[str]:
    outputs = node_schema.get("output") or []
    if not isinstance(outputs, list):
        return []
    return [normalize_type(t) for t in outputs]


def node_output_names(node_schema: dict[str, Any]) -> list[str]:
    """Output slot names, index-aligned with the `output` type list.

    `output_name` is merged positionally rather than trusted wholesale: some
    nodes declare a shorter or partially-blank output_name, and returning it
    as-is made the trailing real outputs unaddressable (e.g. a node with
    output=[MODEL, CLIP] and output_name=[MODEL] could never be linked on
    CLIP, so Stage C could not wire a text encoder).
    """
    outputs = node_schema.get("output") or []
    if not isinstance(outputs, list):
        outputs = []
    declared = node_schema.get("output_name") or []
    if not isinstance(declared, list):
        declared = []

    names: list[str] = []
    for i, out in enumerate(outputs):
        # V3 /object_info tolerance: output entries may be dicts
        # {"name": "...", "type": "..."} instead of bare type strings.
        if isinstance(out, dict):
            derived = out.get("name") or out.get("type") or ""
        else:
            derived = out
        if i < len(declared) and declared[i]:
            names.append(str(declared[i]))
        else:
            names.append(str(derived))

    # An output_name longer than output (rare, but seen on nodes that declare
    # names ahead of a dynamic output list) still contributes its extra names.
    names.extend(str(n) for n in declared[len(outputs):] if n)
    return names


def node_input_type(node_schema: dict[str, Any], input_name: str) -> str | None:
    inputs = node_schema.get("input", {})
    if not isinstance(inputs, dict):
        return None
    for group in ("required", "optional"):
        group_inputs = inputs.get(group)
        if not isinstance(group_inputs, dict):
            continue
        spec = group_inputs.get(input_name)
        if spec is not None:
            return normalize_type(spec_type(spec))
    return None


class LinkValidationError(ValueError):
    pass


def validate_link(
    from_node_type: str,
    from_output: str,
    to_node_type: str,
    to_input: str,
    schema: dict[str, dict[str, Any]],
) -> None:
    """Raise LinkValidationError if the link is not type-safe.

    Checks (Section 2E Stage C):
      - from_node_type exists in schema
      - to_node_type exists in schema
      - from_output name exists on the from node
      - to_input name exists on the to node
      - the type strings match exactly (case-normalized), unless either side
        is a wildcard ("*"), which connects to anything
    """
    if not isinstance(from_output, str) or not isinstance(to_input, str):
        raise LinkValidationError(
            f"slot names must be strings, got from_output={from_output!r} to_input={to_input!r}"
        )
    from_schema = schema.get(from_node_type)
    if from_schema is None:
        raise LinkValidationError(f"unknown source node type: {from_node_type}")

    to_schema = schema.get(to_node_type)
    if to_schema is None:
        raise LinkValidationError(f"unknown target node type: {to_node_type}")

    out_names = node_output_names(from_schema)
    if from_output not in out_names:
        raise LinkValidationError(f"{from_node_type} has no output named '{from_output}' (has {out_names})")
    out_index = out_names.index(from_output)
    out_types = node_output_types(from_schema)
    if out_index >= len(out_types):
        raise LinkValidationError(f"{from_node_type}.{from_output} has no declared type")
    out_type = out_types[out_index]

    in_type = node_input_type(to_schema, to_input)
    if in_type is None:
        raise LinkValidationError(f"{to_node_type} has no input named '{to_input}'")

    # A wildcard on either end accepts any counterpart type — Reroute and the
    # any-type inputs common in custom node packs would otherwise have every
    # link rejected as a mismatch.
    if out_type in WILDCARD_TYPES or in_type in WILDCARD_TYPES:
        return

    if out_type != in_type:
        raise LinkValidationError(
            f"type mismatch: {from_node_type}.{from_output} ({out_type}) -> {to_node_type}.{to_input} ({in_type})"
        )
