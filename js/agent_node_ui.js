import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { applyDiff, injectWorkflow, restoreGraph, snapshotGraph } from "./canvas_writer.js";
import { listenForAgentProgress, listenForChatTokens } from "./progress_listener.js";
import { createChatPanel } from "./chat_panel.js";

// Fallback only — the server sends subgraph_wrap_threshold on every response.
// Must match config.SUBGRAPH_WRAP_THRESHOLD.
const SUBGRAPH_WRAP_THRESHOLD = 6;

// How long after a build an execution error is still attributed to it.
// Bounded so the agent does not claim responsibility for a failure in a
// workflow the user has since rewritten by hand.
const BUILD_ERROR_WINDOW_MS = 5 * 60 * 1000;

// ComfyUI auto-loads .js from WEB_DIRECTORY but not .css, so the stylesheet
// has to be linked in explicitly or the CHAT panel renders unstyled.
function injectStylesheet() {
    const id = "comfy-agent-stylesheet";
    if (document.getElementById(id)) return;
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = new URL("./agent_style.css", import.meta.url).href;
    document.head.append(link);
}

/**
 * Serialize the canvas for the server. Wrapped because graph serialization
 * can throw on a graph containing a malformed node, and MODIFY should report
 * that clearly rather than failing inside JSON.stringify.
 */
function captureGraph(appRef) {
    try {
        return appRef.graph?.serialize?.() ?? null;
    } catch (err) {
        console.warn("[comfy_agent] could not serialize graph:", err);
        return null;
    }
}

function newBuildId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `build_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function setStatus(node, text) {
    if (!node.comfyAgentStatusWidget) return;
    node.comfyAgentStatusWidget.value = text;
    app.graph.setDirtyCanvas(true, true);
}

/**
 * Ask the server what is actually working. Returns null if health itself is
 * unreachable (older server, or ComfyUI still starting), which the caller
 * treats as "say nothing" rather than as a failure to report.
 */
async function fetchHealth() {
    try {
        const resp = await api.fetchApi("/agent/health");
        if (!resp.ok) return null;
        return await resp.json();
    } catch {
        return null;
    }
}

/**
 * One status line describing the first real problem, with the command that
 * fixes it. The server already ranks hints most-blocking-first, so leading
 * with hints[0] avoids burying "Ollama is down" under a note about the embed
 * model. Blueprint builds need no Ollama at all, so an Ollama problem is
 * reported as a limitation rather than as a dead node.
 */
function healthSummary(health) {
    if (!health) return null;
    const hints = Array.isArray(health.hints) ? health.hints : [];
    if (!hints.length) return null;
    if (!health.ollama) return `${hints[0]} Blueprint builds still work.`;
    return hints[0];
}

async function triggerBuild(node) {
    const getWidget = (name) => node.widgets?.find((w) => w.name === name);

    const taskPrompt = getWidget("task_prompt")?.value ?? "";
    const mode = getWidget("mode")?.value ?? "BUILD";
    const taskCategory = getWidget("task_category")?.value ?? "auto";
    const modelFamily = getWidget("model_family")?.value ?? "auto";
    const maxNodes = getWidget("max_nodes")?.value ?? 12;
    const useBlueprint = getWidget("use_blueprint")?.value ?? true;

    if (!taskPrompt.trim()) {
        setStatus(node, "Enter a task_prompt first.");
        return;
    }

    if (node.comfyAgentBuilding) {
        setStatus(node, "A build is already in progress...");
        return;
    }
    node.comfyAgentBuilding = true;
    const buildId = newBuildId();
    node.comfyAgentBuildId = buildId;

    try {
        setStatus(node, "Fetching node registry...");

        let resp;
        try {
            resp = await api.fetchApi("/agent/build_workflow", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    task_prompt: taskPrompt,
                    mode,
                    task_category: taskCategory,
                    model_family: modelFamily,
                    max_nodes: maxNodes,
                    use_blueprint: useBlueprint,
                    sid: api.clientId,
                    build_id: buildId,
                    // MODIFY edits what is already on the canvas, so it needs
                    // to see it. Serialized defensively — a graph containing a
                    // malformed node must not stop the request.
                    graph: mode === "MODIFY" ? captureGraph(app) : undefined,
                }),
            });
        } catch (err) {
            setStatus(node, `Error: request failed (${err})`);
            return;
        }

        if (resp.status === 503) {
            setStatus(node, "Ollama is not running. Start it and try again.");
            return;
        }
        if (!resp.ok) {
            let detail = "";
            try {
                const errBody = await resp.json();
                detail = errBody?.error_detail || errBody?.error || "";
            } catch (err) {
                // Don't swallow silently — without this the real cause of a
                // malformed error body is invisible in the console.
                console.warn("[comfy_agent] could not parse error body:", err);
            }
            setStatus(node, `Error: server returned ${resp.status}${detail ? ` (${detail})` : ""}`);
            return;
        }

        let result;
        try {
            result = await resp.json();
        } catch (err) {
            setStatus(node, `Error: invalid response (${err})`);
            return;
        }
        // `execution_ready` is NOT the failure test. The server deliberately
        // returns a real spec with execution_ready=false for a graph whose
        // required inputs are not all wired: those nodes are still worth
        // placing, and the whole incomplete-build path below (markIncomplete,
        // the amber highlight, the "needs connections" message) exists to hand
        // that case to the user. Gating on it here made all of that
        // unreachable and reported a partial build as an outright failure.
        //
        // A genuine failure is the one the server signals with error_detail
        // and an empty spec, so that is what is tested.
        if (result.error_detail) {
            setStatus(node, `Error: ${result.error_detail}`);
            return;
        }

        // MODIFY returns an edit to apply, not a workflow to inject.
        if (result.diff) {
            try {
                node.comfyAgentUndoSnapshot = snapshotGraph(app.graph);
                const outcome = applyDiff(app, result.diff);
                let msg =
                    `Modified: ${outcome.removed} removed, ` +
                    `${outcome.widgetsSet} widget(s) changed, ${outcome.added} added`;
                if (outcome.failed.length) {
                    msg += ` — ${outcome.failed.length} edit(s) failed: ${outcome.failed.join("; ")}`;
                }
                if (Array.isArray(result.warnings) && result.warnings.length) {
                    msg += ` — warnings: ${result.warnings.join("; ")}`;
                }
                msg += " (Undo Last Agent Action reverts this)";
                setStatus(node, msg);
            } catch (err) {
                setStatus(node, `Error applying changes: ${err}`);
                console.error("[comfy_agent] diff application failed:", err);
            }
            return;
        }

        let spec;
        try {
            spec = JSON.parse(result.workflow_json);
        } catch (err) {
            setStatus(node, `Error: could not parse workflow_json (${err})`);
            return;
        }

        try {
            // Taken immediately before the only mutation in this function, so
            // Undo reverts exactly this action and nothing that happened while
            // the request was in flight.
            node.comfyAgentUndoSnapshot = snapshotGraph(app.graph);

            const wrapThreshold = result.subgraph_wrap_threshold ?? SUBGRAPH_WRAP_THRESHOLD;
            const injectResult = injectWorkflow(
                app, spec, wrapThreshold, result.incomplete_inputs ?? [],
            );
            // Remember what this build put on the canvas, so a later execution
            // error can be attributed to it rather than to an unrelated part
            // of the user's workflow.
            node.comfyAgentBuiltNodeIds = new Set(injectResult.createdNodeIds ?? []);
            node.comfyAgentBuiltAt = Date.now();
            let msg = `Done — built ${injectResult.nodeCount} nodes, ${injectResult.linkCount} links`;
            if (injectResult.failedTypes.length) {
                msg += ` (failed: ${injectResult.failedTypes.join(", ")})`;
            }
            // Lead with runnability. A graph missing required inputs is placed
            // but cannot be queued, and saying "Done" alone reads as success.
            if (result.execution_ready === false && injectResult.incompleteCount) {
                msg = `Placed ${injectResult.nodeCount} nodes — ${injectResult.incompleteCount} ` +
                      `highlighted node(s) still need connections before this can run`;
            }
            // Links the server asked for that could not be wired. Naming them
            // turns a silently sparser graph into something the user can go
            // and fix by hand.
            if (injectResult.skippedLinks?.length) {
                msg += ` — ${injectResult.skippedLinks.length} link(s) not wired: ` +
                       injectResult.skippedLinks.join("; ");
            }
            // Non-fatal server-side findings (e.g. orphan nodes): the build
            // succeeded but the workflow may not run as intended.
            if (Array.isArray(result.warnings) && result.warnings.length) {
                msg += ` — warnings: ${result.warnings.join("; ")}`;
            }
            setStatus(node, msg);
        } catch (err) {
            setStatus(node, `Error drawing on canvas: ${err}`);
            console.error("[comfy_agent] canvas injection failed:", err);
        }
    } finally {
        node.comfyAgentBuilding = false;
        // Stop accepting progress for a build that is no longer running, so a
        // late-arriving event cannot overwrite the final status.
        if (node.comfyAgentBuildId === buildId) node.comfyAgentBuildId = null;
    }
}

app.registerExtension({
    name: "comfy_agent.ui",
    async setup() {
        console.log("[comfy_agent] extension loaded");
        injectStylesheet();
        listenForChatTokens((detail) => {
            if (!detail?.token) return;
            const nodes = (app.graph?._nodes || []).filter((n) => n.comfyClass === "ComfyAgent");
            // Route strictly by chat_id: an untagged broadcast would splice one
            // node's reply into another node's open conversation.
            const target = detail.chat_id
                ? nodes.find((n) => n.comfyAgentChatId === detail.chat_id)
                : null;
            target?.comfyAgentOnChatToken?.(detail.token);
        });

        // Capture execution errors so CHAT can answer "why did this fail?"
        // without the user pasting the traceback back in by hand.
        api.addEventListener("execution_error", (event) => {
            const detail = event.detail || {};
            const text = [
                detail.node_type ? `Node: ${detail.node_type}` : "",
                detail.exception_type ? `${detail.exception_type}: ${detail.exception_message ?? ""}` : "",
                Array.isArray(detail.traceback) ? detail.traceback.join("") : "",
            ].filter(Boolean).join("\n");
            if (!text) return;
            for (const n of app.graph?._nodes || []) {
                if (n.comfyClass !== "ComfyAgent") continue;
                n.comfyAgentLastError = text;

                // Close the loop on BUILD: if the failing node is one this
                // agent just placed, say so on the node itself. Otherwise the
                // only signal was a generic ComfyUI error dialog, leaving the
                // agent reporting a build it had no idea was broken.
                const built = n.comfyAgentBuiltNodeIds;
                const fresh =
                    n.comfyAgentBuiltAt &&
                    Date.now() - n.comfyAgentBuiltAt < BUILD_ERROR_WINDOW_MS;
                // `==` not `===`: LiteGraph node ids are numbers while the
                // event reports them as strings in some ComfyUI versions.
                const isOurs =
                    built &&
                    detail.node_id !== undefined &&
                    [...built].some((id) => id == detail.node_id);
                if (fresh && isOurs && !n.comfyAgentBuilding) {
                    const label = detail.node_type || `node ${detail.node_id}`;
                    setStatus(
                        n,
                        `This build failed to run: ${label} — ` +
                        `${detail.exception_type || "error"}. ` +
                        "Switch to CHAT and ask why for a diagnosis.",
                    );
                }
            }
        });

        listenForAgentProgress((detail) => {
            if (!detail?.message) return;
            const nodes = (app.graph?._nodes || []).filter(
                (n) => n.comfyClass === "ComfyAgent" && n.comfyAgentStatusWidget,
            );
            // Route to the node that started this build. Without the build_id
            // filter, two agent nodes on one canvas overwrite each other's
            // status — including stomping a finished result with a stale
            // in-progress line from the other build.
            const target = detail.build_id
                ? nodes.find((n) => n.comfyAgentBuildId === detail.build_id)
                : null;
            if (target) {
                setStatus(target, detail.message);
                return;
            }
            if (!detail.build_id) {
                // Untagged event (older server, or a non-build notice): fall
                // back to the previous broadcast behaviour.
                for (const node of nodes) setStatus(node, detail.message);
            }
        });
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "ComfyAgent") return;

        node.comfyAgentStatusWidget = node.addWidget("text", "status", "Idle.", () => {}, {
            multiline: true,
        });
        node.comfyAgentStatusWidget.disabled = true;

        const buildButton = node.addWidget("button", "Build Workflow", null, () => {
            triggerBuild(node);
        });

        // Undo is what makes an agent safe to try on a workflow you care
        // about. Without it, the first bad build teaches the user never to
        // press the button again.
        const undoButton = node.addWidget("button", "Undo Last Agent Action", null, () => {
            if (node.comfyAgentBuilding) {
                setStatus(node, "Cannot undo while a build is running.");
                return;
            }
            if (!node.comfyAgentUndoSnapshot) {
                setStatus(node, "Nothing to undo — no agent action this session.");
                return;
            }
            if (restoreGraph(app, node.comfyAgentUndoSnapshot)) {
                // One level of undo only: keeping the snapshot would let a
                // second press "undo" a state that is already restored.
                node.comfyAgentUndoSnapshot = null;
                node.comfyAgentBuiltNodeIds = null;
                setStatus(node, "Reverted the canvas to before the last agent action.");
            } else {
                setStatus(node, "Undo failed — the canvas was left unchanged.");
            }
        });

        // --- CHAT panel ---
        const chat = createChatPanel(app, node);
        const chatWidget = node.addDOMWidget("chat", "div", chat.root, {
            // Chat history is persisted via onSerialize below, not through the
            // widget value — serializing the DOM node itself would break save.
            serialize: false,
            hideOnZoom: false,
        });

        const clearButton = node.addWidget("button", "Clear Chat", null, () => chat.clear());

        function applyMode() {
            const mode = node.widgets?.find((w) => w.name === "mode")?.value ?? "BUILD";
            const chatting = mode === "CHAT";

            // The button does different things in different modes; saying so
            // is cheaper than a user pressing "Build Workflow" and being
            // surprised that it edited what was already there.
            if (buildButton) {
                buildButton.name = mode === "MODIFY" ? "Modify Workflow" : "Build Workflow";
            }

            // DOM widgets are hidden by moving them offscreen; LiteGraph does
            // not honour the `hidden` flag for them the way it does for
            // native widgets.
            chat.root.style.display = chatting ? "flex" : "none";
            if (chatWidget) chatWidget.computeSize = chatting ? undefined : () => [0, -4];

            // Build is for the non-chat modes, Clear Chat only for CHAT.
            // Both are buttons, so restoring `type` is unconditional.
            for (const [w, hide] of [
                [buildButton, chatting],
                [undoButton, chatting],
                [clearButton, !chatting],
            ]) {
                if (!w) continue;
                w.hidden = hide;
                // LiteGraph checks type against "hidden" in some paths.
                w.type = hide ? "hidden" : "button";
            }

            node.setSize([node.size[0], node.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        }

        const modeWidget = node.widgets?.find((w) => w.name === "mode");
        if (modeWidget) {
            const originalCallback = modeWidget.callback;
            modeWidget.callback = function (...args) {
                const ret = originalCallback?.apply(this, args);
                applyMode();
                return ret;
            };
        }

        // Persist the conversation with the saved workflow, so reopening a
        // .json restores the thread rather than an empty panel.
        const originalOnSerialize = node.onSerialize;
        node.onSerialize = function (info) {
            originalOnSerialize?.apply(this, arguments);
            info.comfyAgentChatHistory = this.comfyAgentChatHistory || [];
        };
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            originalOnConfigure?.apply(this, arguments);
            if (Array.isArray(info?.comfyAgentChatHistory)) {
                this.comfyAgentChatHistory = info.comfyAgentChatHistory;
                chat.renderHistory();
            }
            applyMode();
        };

        // Tear down per-node state when the node leaves the canvas. Without
        // this, removing and re-adding the agent node accumulates orphaned
        // chat listeners and leaves a streaming callback holding DOM that is
        // no longer on screen.
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            try {
                chat.destroy();
            } catch (err) {
                console.warn("[comfy_agent] chat panel teardown failed:", err);
            }
            // Stop any in-flight build or chat from writing to a dead node.
            this.comfyAgentBuildId = null;
            this.comfyAgentChatId = null;
            this.comfyAgentStatusWidget = null;
            this.comfyAgentBuiltNodeIds = null;
            // A full graph serialization; not worth retaining for a node that
            // is no longer on the canvas.
            this.comfyAgentUndoSnapshot = null;
            return originalOnRemoved?.apply(this, arguments);
        };

        applyMode();

        // Report a broken setup in the node itself. The same information is
        // logged server-side, but only on first use and only to a console the
        // user is unlikely to be reading when the build fails.
        fetchHealth().then((health) => {
            const summary = healthSummary(health);
            // Only overwrite a still-idle node: a build started before health
            // returned must keep its own status.
            if (summary && node.comfyAgentStatusWidget?.value === "Idle.") {
                setStatus(node, summary);
            }
        });
    },
});
