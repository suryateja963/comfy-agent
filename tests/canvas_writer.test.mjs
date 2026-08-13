// Tests for js/canvas_writer.js — the canvas injection layer, which had no
// coverage of any kind despite holding the widget-assignment and node-placement
// logic. Uses Node's built-in test runner, so there is no npm dependency and
// nothing to install:
//
//     node --test tests/
//
// LiteGraph and the ComfyUI app object are stubbed; the widget layouts below
// mirror a real /object_info (verified against a live instance with 1717 nodes).

import test from "node:test";
import assert from "node:assert/strict";
import {
    applyDiff,
    findFreeZone,
    injectWorkflow,
    restoreGraph,
    setWidgetValue,
    snapshotGraph,
} from "../js/canvas_writer.js";

// Real widget order, as reported by a live ComfyUI /object_info. KSampler has
// SIX widgets — blueprints authored against the legacy layout supply seven
// values (an extra control_after_generate token at index 1), which is what
// makes positional injection misassign every slot after `seed`.
const WIDGET_LAYOUT = {
    KSampler: ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    CheckpointLoaderSimple: ["ckpt_name"],
    CLIPTextEncode: ["text"],
    EmptyLatentImage: ["width", "height", "batch_size"],
    VAEDecode: [],
    SaveImage: ["filename_prefix"],
};

function installLiteGraph({ failTypes = [] } = {}) {
    globalThis.LiteGraph = {
        createNode(type) {
            if (failTypes.includes(type)) return null;
            if (!(type in WIDGET_LAYOUT)) return null;
            return {
                type,
                pos: [0, 0],
                size: [200, 100],
                widgets: WIDGET_LAYOUT[type].map((n) => ({ name: n, value: "<default>" })),
                inputs: [{ name: "in", type: "*" }],
                outputs: [{ name: "out", type: "*" }],
                connect() {
                    return true;
                },
            };
        },
    };
}

function makeApp(existingNodes = []) {
    const graph = {
        _nodes: [...existingNodes],
        _nextId: existingNodes.length + 1,
        add(n) {
            // LiteGraph assigns the id on add(), not on createNode(). The
            // stub must do the same or createdNodeIds looks populated while
            // being entirely undefined.
            if (n.id === undefined) n.id = this._nextId++;
            this._nodes.push(n);
        },
        remove(n) {
            const i = this._nodes.indexOf(n);
            if (i >= 0) this._nodes.splice(i, 1);
        },
        setDirtyCanvas() {},
        // Mirrors LiteGraph's save/load pair closely enough to test the
        // snapshot/undo contract: serialize() must not alias live state, and
        // configure() replaces the graph wholesale.
        serialize() {
            return JSON.parse(JSON.stringify({ nodes: this._nodes }));
        },
        configure(data) {
            this._nodes = JSON.parse(JSON.stringify(data.nodes ?? []));
        },
    };
    return { graph, canvas: null };
}

const node = (x, y, w = 300, h = 200) => ({ pos: [x, y], size: [w, h] });

const widgetsOf = (n) => Object.fromEntries(n.widgets.map((w) => [w.name, w.value]));

// --- findFreeZone -----------------------------------------------------------

test("findFreeZone returns the origin on an empty canvas", () => {
    assert.deepEqual(findFreeZone(makeApp().graph, 300, 200), [100, 100]);
});

test("findFreeZone never returns an overlapping position", () => {
    // Densely tile the search region so no gap exists inside it.
    const occupied = [];
    for (let y = 100; y <= 1000; y += 40) {
        for (let x = 100; x <= 1000; x += 40) occupied.push(node(x, y, 400, 300));
    }
    const [fx, fy] = findFreeZone(makeApp(occupied).graph, 300, 200);
    const overlaps = occupied.some(
        (o) =>
            fx < o.pos[0] + o.size[0] &&
            fx + 300 > o.pos[0] &&
            fy < o.pos[1] + o.size[1] &&
            fy + 200 > o.pos[1],
    );
    // Previously the loop gave up and returned the last-tested (overlapping)
    // coordinate, stamping new nodes on top of existing canvas content.
    assert.equal(overlaps, false, `returned overlapping position [${fx}, ${fy}]`);
});

test("findFreeZone tolerates malformed nodes without throwing", () => {
    const graph = makeApp([{ pos: null, size: null }, node(100, 100)]).graph;
    const [x, y] = findFreeZone(graph, 300, 200);
    assert.ok(Number.isFinite(x) && Number.isFinite(y));
});

test("findFreeZone stays fast on a large canvas", () => {
    const many = [];
    for (let i = 0; i < 800; i++) {
        many.push(node(100 + (i % 40) * 40, 100 + Math.floor(i / 40) * 40));
    }
    const started = Date.now();
    findFreeZone(makeApp(many).graph, 300, 200);
    assert.ok(Date.now() - started < 1000, "findFreeZone took over 1s on 800 nodes");
});

// --- blueprint widget injection --------------------------------------------

const KSAMPLER_LEGACY_VALUES = [0, "fixed", 20, 7.0, "dpmpp_2m", "karras", 1.0];

function blueprintWith(nodeDef) {
    return { nodes: [nodeDef], links: [] };
}

test("widgets_named assigns every value to the correct slot", () => {
    installLiteGraph();
    const app = makeApp();
    injectWorkflow(
        app,
        blueprintWith({
            id: 1,
            type: "KSampler",
            pos: [0, 0],
            widgets_values: KSAMPLER_LEGACY_VALUES,
            widgets_named: {
                seed: 0,
                steps: 20,
                cfg: 7.0,
                sampler_name: "dpmpp_2m",
                scheduler: "karras",
                denoise: 1.0,
            },
        }),
        6,
    );
    assert.deepEqual(widgetsOf(app.graph._nodes[0]), {
        seed: 0,
        steps: 20,
        cfg: 7.0,
        sampler_name: "dpmpp_2m",
        scheduler: "karras",
        denoise: 1.0,
    });
});

test("positional fallback misassigns — the reason widgets_named exists", () => {
    installLiteGraph();
    const app = makeApp();
    injectWorkflow(
        app,
        blueprintWith({ id: 1, type: "KSampler", pos: [0, 0], widgets_values: KSAMPLER_LEGACY_VALUES }),
        6,
    );
    const got = widgetsOf(app.graph._nodes[0]);
    // Documents the corruption: 7 values into 6 slots shifts everything after
    // `seed`, and `denoise` falls off the end.
    assert.equal(got.steps, "fixed");
    assert.equal(got.cfg, 20);
    assert.equal(got.sampler_name, 7.0);
    assert.notEqual(got.denoise, 1.0);
});

test("an unknown widget name does not fall through to a positional write", () => {
    installLiteGraph();
    const app = makeApp();
    injectWorkflow(
        app,
        blueprintWith({
            id: 1,
            type: "KSampler",
            pos: [0, 0],
            widgets_named: { not_a_real_widget: 999, steps: 25 },
        }),
        6,
    );
    const got = widgetsOf(app.graph._nodes[0]);
    assert.equal(got.steps, 25, "the valid name should still apply");
    assert.ok(
        !Object.values(got).includes(999),
        "a missed name must not corrupt a neighbouring slot",
    );
});

// --- setWidgetValue contract ------------------------------------------------
// Tested directly: injectBlueprint passes no positionalIndex alongside a name,
// so going through it cannot prove the no-fall-through guard actually holds.

const sampleNode = () => ({
    widgets: WIDGET_LAYOUT.KSampler.map((n) => ({ name: n, value: "<default>" })),
});

test("setWidgetValue applies a value by name and reports success", () => {
    const n = sampleNode();
    assert.equal(setWidgetValue(n, "steps", 25), true);
    assert.equal(widgetsOf(n).steps, 25);
});

test("setWidgetValue reports failure for an unknown name", () => {
    const n = sampleNode();
    assert.equal(setWidgetValue(n, "not_a_real_widget", 999), false);
});

test("a missed NAME never falls through to the positional index", () => {
    const n = sampleNode();
    // Index 0 is `seed`. With a name supplied, a miss must NOT write there —
    // that fall-through is exactly the silent corruption naming prevents.
    assert.equal(setWidgetValue(n, "not_a_real_widget", 999, 0), false);
    assert.equal(widgetsOf(n).seed, "<default>", "positional fallback corrupted `seed`");
    assert.ok(!Object.values(widgetsOf(n)).includes(999));
});

test("positional write still works when no name is given", () => {
    const n = sampleNode();
    assert.equal(setWidgetValue(n, undefined, 42, 1), true);
    assert.equal(widgetsOf(n).steps, 42);
});

test("setWidgetValue is safe on a node with no widgets", () => {
    assert.equal(setWidgetValue({}, "steps", 1), false);
});

// --- free-form injection ----------------------------------------------------

test("free-form spec injects nodes and applies named widgets", () => {
    installLiteGraph();
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: { ks_1: { steps: 30 }, save_1: { filename_prefix: "out" } },
            links: [],
        },
        6,
    );
    assert.equal(result.nodeCount, 2);
    assert.equal(widgetsOf(app.graph._nodes[0]).steps, 30);
    assert.equal(widgetsOf(app.graph._nodes[1]).filename_prefix, "out");
});

test("a node type that fails to create is reported, not fatal", () => {
    installLiteGraph({ failTypes: ["SaveImage"] });
    const app = makeApp();
    const result = injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler", save_1: "SaveImage" }, widgets: {}, links: [] },
        6,
    );
    assert.equal(result.nodeCount, 1);
    assert.deepEqual(result.failedTypes, ["SaveImage"]);
});

test("links referencing a failed node are skipped without throwing", () => {
    installLiteGraph({ failTypes: ["SaveImage"] });
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: {},
            links: [{ from_node: "ks_1", from_output: "out", to_node: "save_1", to_input: "in" }],
        },
        6,
    );
    assert.equal(result.linkCount, 0);
    // The dropped link must be reported, not merely absent: "0 links" alone
    // does not tell the user a link was requested and lost.
    assert.equal(result.skippedLinks.length, 1);
    assert.match(result.skippedLinks[0], /node not created/);
});

test("a link naming a slot that does not exist is reported as skipped", () => {
    installLiteGraph();
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: {},
            links: [
                // SaveImage has no input called "nonexistent".
                { from_node: "ks_1", from_output: "out", to_node: "save_1", to_input: "nonexistent" },
            ],
        },
        6,
    );
    assert.equal(result.linkCount, 0);
    assert.equal(result.skippedLinks.length, 1);
    assert.match(result.skippedLinks[0], /nonexistent/);
});

test("applyDiff removes, edits and adds in one pass", () => {
    installLiteGraph();
    const app = makeApp();
    // Two real nodes to edit against.
    injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler", save_1: "SaveImage" }, widgets: {}, links: [] },
        99, // above the wrap threshold check, so nothing is collapsed
    );
    const [ks, save] = app.graph._nodes;

    const outcome = applyDiff(app, {
        add_nodes: ["SaveImage"],
        remove_node_ids: [save.id],
        set_widgets: [{ node_id: ks.id, widget: "steps", value: 42 }],
    });

    assert.equal(outcome.removed, 1);
    assert.equal(outcome.widgetsSet, 1);
    assert.equal(outcome.added, 1);
    assert.deepEqual(outcome.failed, []);
    assert.equal(ks.widgets.find((w) => w.name === "steps").value, 42);
    assert.ok(!app.graph._nodes.includes(save), "the removed node is off the canvas");
});

test("applyDiff edits widgets before removals take effect", () => {
    // A diff that edits and removes different nodes must not lose the edit
    // because the removal reindexed the graph first.
    installLiteGraph();
    const app = makeApp();
    injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler", save_1: "SaveImage" }, widgets: {}, links: [] },
        99,
    );
    const [ks, save] = app.graph._nodes;

    const outcome = applyDiff(app, {
        add_nodes: [],
        remove_node_ids: [save.id],
        set_widgets: [{ node_id: ks.id, widget: "cfg", value: 7.5 }],
    });
    assert.equal(outcome.widgetsSet, 1);
    assert.equal(ks.widgets.find((w) => w.name === "cfg").value, 7.5);
});

test("applyDiff reports edits it could not apply instead of throwing", () => {
    installLiteGraph();
    const app = makeApp();
    injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler" }, widgets: {}, links: [] },
        99,
    );
    const outcome = applyDiff(app, {
        add_nodes: [],
        remove_node_ids: [9999],
        set_widgets: [{ node_id: 9999, widget: "steps", value: 1 }],
    });
    assert.equal(outcome.removed, 0);
    assert.equal(outcome.widgetsSet, 0);
    assert.equal(outcome.failed.length, 2);
});

test("applyDiff on an empty diff changes nothing", () => {
    installLiteGraph();
    const app = makeApp([node(100, 100)]);
    const outcome = applyDiff(app, { add_nodes: [], remove_node_ids: [], set_widgets: [] });
    assert.deepEqual(outcome, { removed: 0, widgetsSet: 0, added: 0, failed: [] });
    assert.equal(app.graph._nodes.length, 1);
});

test("a snapshot taken before injection restores the original canvas", () => {
    installLiteGraph();
    const app = makeApp([node(100, 100)]);
    const before = snapshotGraph(app.graph);
    assert.equal(app.graph._nodes.length, 1);

    injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler", save_1: "SaveImage" }, widgets: {}, links: [] },
        6,
    );
    assert.equal(app.graph._nodes.length, 3, "injection should have added nodes");

    assert.equal(restoreGraph(app, before), true);
    assert.equal(app.graph._nodes.length, 1, "undo must remove the injected nodes");
});

test("a snapshot does not alias live graph state", () => {
    installLiteGraph();
    const app = makeApp([node(100, 100)]);
    const before = snapshotGraph(app.graph);
    // Mutating the graph after snapshotting must not change the snapshot; a
    // shallow copy here would make undo restore the post-change state.
    app.graph._nodes[0].pos = [999, 999];
    assert.deepEqual(before.nodes[0].pos, [100, 100]);
});

test("restoring a null snapshot is refused rather than clearing the canvas", () => {
    installLiteGraph();
    const app = makeApp([node(100, 100)]);
    assert.equal(restoreGraph(app, null), false);
    assert.equal(app.graph._nodes.length, 1, "a failed undo must not wipe the graph");
});

test("snapshotGraph reports failure instead of throwing", () => {
    const broken = { serialize() { throw new Error("unserializable node"); } };
    assert.equal(snapshotGraph(broken), null);
});

test("createdNodeIds reports the ids of every placed node", () => {
    installLiteGraph();
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: {},
            links: [],
        },
        6,
    );
    assert.equal(result.createdNodeIds.length, 2);
    // Attribution of a later execution error depends on these being real ids,
    // so an all-undefined list must fail rather than look populated.
    assert.ok(result.createdNodeIds.every((id) => id !== undefined));
    assert.deepEqual(
        result.createdNodeIds,
        app.graph._nodes.map((n) => n.id),
    );
});

test("createdNodeIds omits nodes that failed to create", () => {
    installLiteGraph({ failTypes: ["SaveImage"] });
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: {},
            links: [],
        },
        6,
    );
    assert.equal(result.createdNodeIds.length, 1);
});

test("a fully wired spec reports no skipped links", () => {
    installLiteGraph();
    const app = makeApp();
    const result = injectWorkflow(
        app,
        {
            node_instances: { ks_1: "KSampler", save_1: "SaveImage" },
            widgets: {},
            links: [
                { from_node: "ks_1", from_output: "out", to_node: "save_1", to_input: "in" },
            ],
        },
        6,
    );
    assert.equal(result.skippedLinks.length, 0);
});

test("injected nodes do not overlap existing canvas content", () => {
    installLiteGraph();
    const existing = node(100, 100, 600, 400);
    const app = makeApp([existing]);
    injectWorkflow(
        app,
        { node_instances: { ks_1: "KSampler" }, widgets: {}, links: [] },
        6,
    );
    const created = app.graph._nodes[1];
    const overlaps =
        created.pos[0] < existing.pos[0] + existing.size[0] &&
        created.pos[0] + 200 > existing.pos[0] &&
        created.pos[1] < existing.pos[1] + existing.size[1] &&
        created.pos[1] + 100 > existing.pos[1];
    assert.equal(overlaps, false, `injected node overlaps existing content`);
});
