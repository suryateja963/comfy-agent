"""Stage D — Full spec validation (server side). Section 2E.

This is the pre-flight check before the spec is handed to the JS frontend,
which does its own canvas-side dry-run (overlap detection, createNode()
null-checks, subgraph wrapping — see js/canvas_writer.js). This stage
catches structural problems (orphan nodes, empty spec, duplicate ids)
before a single byte goes over the wire to the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config


@dataclass
class ValidationResult:
    """Outcome of the pre-flight structural check.

    Three severities, not two, because a partially connected graph is neither:

    * `errors` are fatal — the spec is not a workflow at all (empty, or zero
      links between multiple nodes).
    * `incomplete` lists (instance_id, input_name) for required inputs left
      unconnected. The nodes are still worth placing — the user can finish the
      wiring by hand, which beats refusing to build — but the graph cannot
      run, so `execution_ready` must be False. Reporting these as merely
      "valid with warnings" is what let a KSampler missing `latent_image` be
      announced to the user as a successful build.
    * `warnings` are non-fatal and cosmetic (e.g. a dangling node).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    incomplete: list[tuple[str, str]] = field(default_factory=list)
    node_count: int = 0
    link_count: int = 0
    needs_subgraph_wrap: bool = False

    @property
    def execution_ready(self) -> bool:
        """Can ComfyUI actually run this? Not the same question as `valid`."""
        return self.valid and not self.incomplete

    def incomplete_summary(self) -> str:
        return ", ".join(f"{iid}.{name}" for iid, name in self.incomplete)


def validate_spec(
    node_instances: dict[str, str],
    links: list[dict[str, str]],
    incomplete: list[tuple[str, str]] | None = None,
) -> ValidationResult:
    """Pre-flight structural check.

    ``incomplete`` comes from Stage C's own required-input analysis, which has
    the /object_info schema in scope; this stage deliberately does not re-fetch
    it just to recompute the same answer.
    """
    errors: list[str] = []
    warnings: list[str] = []
    incomplete = list(incomplete or [])

    if not node_instances:
        errors.append("no nodes in spec")

    # Only count links that actually name two known nodes. A link missing
    # from_node/to_node used to insert None into this set (silently masking an
    # orphan), and every entry counted toward link_count regardless of whether
    # it referenced anything real.
    linked_ids: set[str] = set()
    valid_link_count = 0
    for link in links:
        if not isinstance(link, dict):
            continue
        from_id = link.get("from_node")
        to_id = link.get("to_node")
        if not (isinstance(from_id, str) and isinstance(to_id, str)):
            continue
        if from_id not in node_instances or to_id not in node_instances:
            continue
        linked_ids.add(from_id)
        linked_ids.add(to_id)
        valid_link_count += 1

    orphans = set(node_instances) - linked_ids
    fully_disconnected = bool(orphans) and orphans == set(node_instances) and len(node_instances) > 1
    if fully_disconnected:
        # Fatal — a multi-node result with zero links between any of them
        # isn't a workflow, it's a pile of unrelated nodes (Stage C failed
        # to connect anything). Don't report this as execution_ready.
        errors.append(
            f"no links between any of the {len(node_instances)} generated nodes "
            "— Stage C produced a disconnected graph"
        )
    elif orphans and len(node_instances) > 1:
        # Not fatal — a single unconnected node (e.g. a lone SaveImage with no
        # upstream match) is surfaced to the user but doesn't block the build.
        warnings.append(f"orphan nodes with no links: {sorted(orphans)}")

    if incomplete:
        # Not an error (the nodes are still worth placing) but not a warning
        # either — this graph will fail the moment it is queued, so it must not
        # be reported as execution_ready.
        warnings.append(
            f"{len(incomplete)} required input(s) left unconnected: "
            + ", ".join(f"{iid}.{name}" for iid, name in incomplete)
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        incomplete=incomplete,
        node_count=len(node_instances),
        link_count=valid_link_count,
        needs_subgraph_wrap=len(node_instances) > config.SUBGRAPH_WRAP_THRESHOLD,
    )
