// LiteGraph injection logic. Section 2F.
// Uses ComfyUI's own litegraph fork exclusively — no external litegraph APIs.

const SEARCH_LIMIT = 4000;
const SEARCH_STEP = 40;
// Layout pitch for free-form specs. Mirrors config.FREEFORM_COLUMN_PITCH /
// config.FREEFORM_ROW_HEIGHT.
const COLUMN_PITCH = 340;
const ROW_HEIGHT = 260;

export function findFreeZone(graph, width = 300, height = 200) {
    const occupied = (graph._nodes || [])
        .filter((n) => n && Array.isArray(n.pos) && Array.isArray(n.size))
        .map((n) => ({ x: n.pos[0], y: n.pos[1], w: n.size[0], h: n.size[1] }));

    if (!occupied.length) return [100, 100];

    // Bound the scan by the actual content extent instead of a fixed 4000px
    // box: past maxOccupiedX/Y nothing can overlap, so the row scan stops
    // there rather than stepping through empty space on every row.
    const maxOccupiedX = occupied.reduce((m, o) => Math.max(m, o.x + o.w), 0);
    const maxOccupiedY = occupied.reduce((m, o) => Math.max(m, o.y + o.h), 0);
    const xLimit = Math.min(maxOccupiedX, SEARCH_LIMIT);
    const yLimit = Math.min(maxOccupiedY, SEARCH_LIMIT);

    const overlaps = (rx, ry) =>
        occupied.some((o) => rx < o.x + o.w && rx + width > o.x && ry < o.y + o.h && ry + height > o.y);

    for (let y = 100; y <= yLimit; y += SEARCH_STEP) {
        for (let x = 100; x <= xLimit; x += SEARCH_STEP) {
            if (!overlaps(x, y)) return [x, y];
        }
    }

    // Nothing free within the occupied region — drop clear below everything.
    // Previously this returned the last-tested (still overlapping) coordinate,
    // stamping new nodes on top of existing canvas content.
    return [100, maxOccupiedY + 80];
}

/**
 * Serialize the whole graph so an agent action can be undone.
 *
 * Taken before any mutation. Returns null if the graph cannot be serialized,
 * which the caller must treat as "no undo available" rather than proceeding
 * silently — the point of the snapshot is that the user can always get back.
 */
export function snapshotGraph(graph) {
    try {
        return graph?.serialize?.() ?? null;
    } catch (err) {
        console.warn("[comfy_agent] could not snapshot graph:", err);
        return null;
    }
}

/**
 * Restore a snapshot taken by snapshotGraph. Returns true on success.
 *
 * configure() replaces the graph wholesale, so this reverts every change made
 * since the snapshot — not only the agent's. That is the honest behaviour for
 * an "undo the agent's last action" button placed immediately after the
 * action, and the caller warns when other edits may be caught up in it.
 */
export function restoreGraph(app, snapshot) {
    if (!snapshot) return false;
    try {
        app.graph.configure(snapshot);
        app.graph.setDirtyCanvas(true, true);
        return true;
    } catch (err) {
        console.error("[comfy_agent] graph restore failed:", err);
        return false;
    }
}

export function createNodeSafe(graph, classType) {
    try {
        const node = LiteGraph.createNode(classType);
        if (!node) {
            console.warn(`[comfy_agent] createNode returned null for ${classType}`);
            return null;
        }
        return node;
    } catch (err) {
        console.warn(`[comfy_agent] createNode threw for ${classType}:`, err);
        return null;
    }
}

/** Returns true if the value was actually applied to a widget.
 *  Exported for tests — this is the guard that keeps a stale blueprint from
 *  silently corrupting a neighbouring widget slot. */
export function setWidgetValue(node, widgetName, value, positionalIndex) {
    if (!node.widgets) return false;
    if (widgetName !== undefined && widgetName !== null) {
        const widget = node.widgets.find((w) => w.name === widgetName);
        if (widget) {
            widget.value = value;
            return true;
        }
        // A named lookup that misses must NOT fall through to the positional
        // path: writing the value to whatever happens to sit at that index is
        // exactly the silent corruption naming exists to prevent.
        return false;
    }
    // Positional fallback per Section 2F — used only when no name was given.
    if (positionalIndex !== undefined && node.widgets[positionalIndex]) {
        node.widgets[positionalIndex].value = value;
        return true;
    }
    return false;
}

function findOutputSlot(node, outputName) {
    if (!node.outputs) return -1;
    const idx = node.outputs.findIndex((o) => o.name === outputName);
    return idx;
}

function findInputSlot(node, inputName) {
    if (!node.inputs) return -1;
    const idx = node.inputs.findIndex((i) => i.name === inputName);
    return idx;
}

/**
 * Inject a blueprint (Section 2C) — a full litegraph nodes/links structure
 * exported from a saved workflow — offset into a free canvas zone.
 */
function injectBlueprint(graph, spec, originOffset) {
    const idMap = new Map();
    const failedTypes = [];
    const createdNodes = [];

    for (const nodeDef of spec.nodes) {
        const node = createNodeSafe(graph, nodeDef.type);
        if (!node) {
            failedTypes.push(nodeDef.type);
            continue;
        }
        node.pos = [nodeDef.pos[0] + originOffset[0], nodeDef.pos[1] + originOffset[1]];
        if (nodeDef.size) node.size = nodeDef.size;
        // Prefer names when the blueprint provides them. Pure positional
        // injection silently misassigns every widget whenever a node's widget
        // layout shifts between ComfyUI versions (e.g. control_after_generate
        // appearing/moving), producing a workflow that runs but generates
        // wrong output — a failure the user cannot see.
        if (nodeDef.widgets_named && typeof nodeDef.widgets_named === "object") {
            for (const [name, val] of Object.entries(nodeDef.widgets_named)) {
                const applied = setWidgetValue(node, name, val);
                if (!applied) {
                    // Surface a stale blueprint instead of silently leaving the
                    // widget at its default — the whole point of naming is that
                    // a mismatch becomes visible rather than corrupting a
                    // neighbouring slot.
                    console.warn(
                        `[comfy_agent] blueprint widget '${name}' not found on ${nodeDef.type} — left at default`,
                    );
                }
            }
        } else if (Array.isArray(nodeDef.widgets_values)) {
            nodeDef.widgets_values.forEach((val, i) => setWidgetValue(node, undefined, val, i));
        }
        graph.add(node);
        idMap.set(nodeDef.id, node);
        createdNodes.push(node);
    }

    let linkCount = 0;
    const skippedLinks = [];
    for (const link of spec.links) {
        const [, fromId, fromSlot, toId, toSlot] = link;
        const fromNode = idMap.get(fromId);
        const toNode = idMap.get(toId);
        if (!fromNode || !toNode) {
            // One side failed to create — skip, don't crash. Recorded so the
            // caller can report it rather than leaving a silently sparser graph.
            skippedLinks.push(`node ${fromId} -> node ${toId} (node not created)`);
            continue;
        }
        try {
            fromNode.connect(fromSlot, toNode, toSlot);
            linkCount += 1;
        } catch (err) {
            console.warn("[comfy_agent] link failed during blueprint injection:", link, err);
            skippedLinks.push(`node ${fromId} -> node ${toId} (connect failed)`);
        }
    }

    return { createdNodes, failedTypes, linkCount, idMap, skippedLinks };
}

/**
 * Inject a free-form spec produced by the four-stage generation pipeline
 * (Section 2E): { node_instances: {id: class_type}, widgets: {id: {name: value}},
 * links: [{from_node, from_output, to_node, to_input}] }.
 */
function injectFreeForm(graph, spec, originOffset) {
    const idMap = new Map();
    const failedTypes = [];
    const createdNodes = [];

    let col = 0;
    // `|| {}` rather than a bare read: every server error path returns
    // workflow_json "{}", so a spec with no node_instances at all is reachable.
    // Object.entries(undefined) throws a TypeError that surfaces as the
    // unrelated "Error drawing on canvas" instead of the real cause.
    for (const [instanceId, classType] of Object.entries(spec.node_instances || {})) {
        const node = createNodeSafe(graph, classType);
        if (!node) {
            failedTypes.push(classType);
            continue;
        }
        node.pos = [originOffset[0] + col * COLUMN_PITCH, originOffset[1]];
        col += 1;

        const widgets = spec.widgets?.[instanceId] || {};
        for (const [name, value] of Object.entries(widgets)) {
            setWidgetValue(node, name, value);
        }

        graph.add(node);
        idMap.set(instanceId, node);
        createdNodes.push(node);
    }

    let linkCount = 0;
    // Links the server asked for that did not make it onto the canvas. These
    // were previously console-only, so a build reporting "6 nodes, 4 links"
    // gave no indication that a fifth link was requested and dropped — the
    // user found out when the workflow failed to queue.
    const skippedLinks = [];
    const describe = (link) =>
        `${link.from_node}[${link.from_output}] -> ${link.to_node}[${link.to_input}]`;

    for (const link of spec.links || []) {
        const fromNode = idMap.get(link.from_node);
        const toNode = idMap.get(link.to_node);
        if (!fromNode || !toNode) {
            // The node itself failed to create; already reported via failedTypes.
            skippedLinks.push(`${describe(link)} (node not created)`);
            continue;
        }

        const fromSlot = findOutputSlot(fromNode, link.from_output);
        const toSlot = findInputSlot(toNode, link.to_input);
        if (fromSlot < 0 || toSlot < 0) {
            const which = fromSlot < 0 ? `output '${link.from_output}'` : `input '${link.to_input}'`;
            console.warn("[comfy_agent] slot not found for link:", link);
            skippedLinks.push(`${describe(link)} (no ${which})`);
            continue;
        }
        try {
            fromNode.connect(fromSlot, toNode, toSlot);
            linkCount += 1;
        } catch (err) {
            console.warn("[comfy_agent] link failed during free-form injection:", link, err);
            skippedLinks.push(`${describe(link)} (connect failed)`);
        }
    }

    return { createdNodes, failedTypes, linkCount, idMap, skippedLinks };
}

/** Wrap a set of just-created nodes into a SubgraphNode if over threshold
 * (Section 2F). Falls back to leaving nodes flat if the subgraph API isn't
 * available on this ComfyUI frontend version — never blocks the build. */
function maybeWrapSubgraph(graph, createdNodes, threshold) {
    if (createdNodes.length <= threshold) return false;
    try {
        if (typeof graph.convertToSubgraph === "function") {
            const ids = createdNodes.map((n) => n.id);
            graph.convertToSubgraph(ids);
            return true;
        }
    } catch (err) {
        console.warn("[comfy_agent] subgraph wrap failed, leaving nodes flat:", err);
    }
    return false;
}

/**
 * Compute how much canvas space a spec actually needs, and (for blueprints)
 * the shift required so the leftmost/topmost node lands exactly at the
 * chosen free-zone origin. Without this, findFreeZone only ever checked a
 * fixed 300x200 box regardless of how many nodes the spec contains, so a
 * multi-node blueprint could still overlap existing canvas content.
 */
function computeSpecFootprint(spec) {
    if (Array.isArray(spec.nodes)) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const n of spec.nodes) {
            const w = (n.size && n.size[0]) || 200;
            const h = (n.size && n.size[1]) || 100;
            minX = Math.min(minX, n.pos[0]);
            minY = Math.min(minY, n.pos[1]);
            maxX = Math.max(maxX, n.pos[0] + w);
            maxY = Math.max(maxY, n.pos[1] + h);
        }
        return { width: maxX - minX, height: maxY - minY, shift: [-minX, -minY] };
    }
    const count = Object.keys(spec.node_instances || {}).length || 1;
    return { width: count * COLUMN_PITCH, height: ROW_HEIGHT, shift: [0, 0] };
}

/**
 * Top-level entry point called from agent_node_ui.js after a successful
 * /agent/build_workflow response. Handles both blueprint and free-form spec
 * shapes, overlap-aware placement (sized to the actual spec, not a fixed
 * default), subgraph wrapping, and panning the view to the result so a
 * successful build is never mistaken for "nothing happened".
 */
// Warning colours for nodes the agent could not finish wiring. LiteGraph reads
// `color` (title bar) and `bgcolor` (body) directly, so no new API is involved
// and this degrades to a no-op on frontends that ignore them.
const INCOMPLETE_TITLE_COLOR = "#8a5a1e";
const INCOMPLETE_BODY_COLOR = "#3a2a12";

/**
 * Mark nodes that still have unconnected required inputs. Without this the
 * user is told the build is not runnable but has no way to see *which* of a
 * dozen freshly placed nodes is the problem.
 *
 * Exported for tests — the mapping from instance id to placed node is the part
 * that silently breaks if injectFreeForm's id handling ever changes.
 */
export function markIncomplete(idMap, incompleteInputs) {
    if (!Array.isArray(incompleteInputs) || !incompleteInputs.length) return 0;

    const byNode = new Map();
    for (const entry of incompleteInputs) {
        if (!entry || typeof entry.node !== "string") continue;
        if (!byNode.has(entry.node)) byNode.set(entry.node, []);
        byNode.get(entry.node).push(entry.input);
    }

    let marked = 0;
    for (const [instanceId, inputs] of byNode) {
        const node = idMap.get(instanceId);
        if (!node) continue;
        node.color = INCOMPLETE_TITLE_COLOR;
        node.bgcolor = INCOMPLETE_BODY_COLOR;
        // Say what is missing, not just that something is.
        const label = inputs.filter(Boolean).join(", ");
        if (label && !String(node.title || "").includes("needs:")) {
            node.title = `${node.title || ""} ⚠ needs: ${label}`.trim();
        }
        marked += 1;
    }
    return marked;
}

/**
 * Apply a MODIFY diff to the live canvas: remove nodes, change widget values,
 * then add new nodes in a free zone.
 *
 * Order matters. Removals run first so the free-zone search sees the space
 * they release, and widget changes are resolved against the pre-removal graph
 * so an id that no longer exists is reported rather than silently skipped.
 *
 * The caller is expected to have snapshotted the graph first — this deletes
 * the user's nodes, and every failure mode here is one they will want to undo.
 */
export function applyDiff(app, diff) {
    const graph = app.graph;
    const applied = { removed: 0, widgetsSet: 0, added: 0, failed: [] };

    const byId = new Map((graph._nodes || []).map((n) => [n.id, n]));

    // Widget changes first, while every referenced node is still present.
    for (const change of diff.set_widgets || []) {
        const target = byId.get(change.node_id);
        if (!target) {
            applied.failed.push(`widget '${change.widget}' (node ${change.node_id} not found)`);
            continue;
        }
        if (setWidgetValue(target, change.widget, change.value)) {
            applied.widgetsSet += 1;
        } else {
            applied.failed.push(`widget '${change.widget}' on node ${change.node_id}`);
        }
    }

    for (const nodeId of diff.remove_node_ids || []) {
        const target = byId.get(nodeId);
        if (!target) {
            applied.failed.push(`remove node ${nodeId} (not found)`);
            continue;
        }
        try {
            graph.remove(target);
            applied.removed += 1;
        } catch (err) {
            console.warn("[comfy_agent] could not remove node:", nodeId, err);
            applied.failed.push(`remove node ${nodeId}`);
        }
    }

    const addTypes = diff.add_nodes || [];
    if (addTypes.length) {
        const [zx, zy] = findFreeZone(graph, addTypes.length * COLUMN_PITCH, ROW_HEIGHT);
        let col = 0;
        for (const classType of addTypes) {
            const node = createNodeSafe(graph, classType);
            if (!node) {
                applied.failed.push(`add ${classType}`);
                continue;
            }
            node.pos = [zx + col * COLUMN_PITCH, zy];
            col += 1;
            graph.add(node);
            applied.added += 1;
        }
    }

    graph.setDirtyCanvas(true, true);
    return applied;
}

export function injectWorkflow(app, spec, subgraphWrapThreshold = 6, incompleteInputs = []) {
    const graph = app.graph;
    const footprint = computeSpecFootprint(spec);
    const [zx, zy] = findFreeZone(graph, footprint.width, footprint.height);
    const origin = [zx + footprint.shift[0], zy + footprint.shift[1]];

    const isBlueprint = Array.isArray(spec.nodes) && Array.isArray(spec.links);
    const { createdNodes, failedTypes, linkCount, idMap, skippedLinks } = isBlueprint
        ? injectBlueprint(graph, spec, origin)
        : injectFreeForm(graph, spec, origin);

    // Before the subgraph wrap: once nodes are collapsed into a subgraph they
    // are no longer individually visible on the canvas, so the highlight has
    // to be applied while they still are.
    const incompleteCount = markIncomplete(idMap, incompleteInputs);

    maybeWrapSubgraph(graph, createdNodes, subgraphWrapThreshold);

    graph.setDirtyCanvas(true, true);

    if (createdNodes.length > 0 && typeof app.canvas?.centerOnNode === "function") {
        app.canvas.centerOnNode(createdNodes[Math.floor(createdNodes.length / 2)]);
        app.canvas.setDirty(true, true);
    }

    return {
        nodeCount: createdNodes.length,
        linkCount,
        failedTypes,
        incompleteCount,
        skippedLinks,
        // LiteGraph assigns ids on graph.add(), so these are only meaningful
        // after injection. Used to attribute a later execution error to this
        // build rather than to unrelated nodes on the canvas.
        createdNodeIds: createdNodes.map((n) => n.id),
    };
}
