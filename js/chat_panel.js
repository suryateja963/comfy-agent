// CHAT mode UI. Builds a DOM chat panel hosted inside the node via
// node.addDOMWidget, and drives POST /agent/chat with WS token streaming.
import { api } from "../../scripts/api.js";
import { generateSubgraphFromSuggestions } from "./blueprint_exporter.js";

// Kept in sync with config.CHAT_MAX_HISTORY_MESSAGES. The server trims to its
// own limit regardless; trimming here too keeps the request small.
const MAX_HISTORY = 12;

function newChatId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `chat_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

/**
 * Serialize the current canvas for context. Wrapped because graph
 * serialization can throw on a graph containing a malformed node, and losing
 * canvas context is far better than losing the ability to ask a question.
 */
function captureGraph(app) {
    try {
        return app.graph?.serialize?.() ?? null;
    } catch (err) {
        console.warn("[comfy_agent] could not serialize graph for chat context:", err);
        return null;
    }
}

export function createChatPanel(app, node) {
    const root = el("div", "comfy-agent-chat");

    const log = el("div", "comfy-agent-chat-log");

    // --- Topic branching UI ---
    const topicRow = el("div", "comfy-agent-chat-topic-row");
    const topicInput = document.createElement("input");
    topicInput.className = "comfy-agent-chat-topic-input";
    topicInput.type = "text";
    topicInput.placeholder = "Topic (optional)";
    const sendBtn = el("button", "comfy-agent-chat-send", "Send");
    const saveBlueprintBtn = el(
        "button",
        "comfy-agent-chat-save-blueprint",
        "Save as blueprint"
    );
    const closeChatBtn = el("button", "comfy-agent-chat-close", "Close");

    topicRow.append(topicInput, sendBtn, saveBlueprintBtn, closeChatBtn);
    root.append(log, topicRow);
    // --------------------------

    // Conversation state lives on the node so it survives re-renders and is
    // captured by workflow save/load via onSerialize below.
    node.comfyAgentChatHistory = node.comfyAgentChatHistory || [];
    node.comfyAgentChatting = false;

    // History branches: keyed by topic tag, each value is an array of {role, content}
    node.comfyAgentHistoryBranches = node.comfyAgentHistoryBranches || {};

    // Selected topic tag; "default" is the fallback when none provided
    node.comfyAgentSelectedTopic = node.comfyAgentSelectedTopic || "default";

    // Persistent chat_id shared across topic switches — stored on node
    node.comfyAgentChatId = node.comfyAgentChatId || newChatId();

    // WS event listeners attached during CHAT session
    node.comfyAgentWSListeners = node.comfyAgentWSListeners || [];

    // LiteGraph node widgets created during CHAT session
    node.comfyAgentLiteGraphWidgets = node.comfyAgentLiteGraphWidgets || [];

    // Load branches from localStorage on panel creation
    const stored = localStorage.getItem("comfy_agent_chat_branches");
    if (stored) {
        try {
            Object.assign(node.comfyAgentHistoryBranches, JSON.parse(stored));
        } catch (err) {
            console.warn("[comfy_agent] could not parse chat branches from localStorage:", err);
            node.comfyAgentHistoryBranches = {};
        }
    }
    // Ensure the selected topic branch exists
    if (!node.comfyAgentHistoryBranches[node.comfyAgentSelectedTopic]) {
        node.comfyAgentHistoryBranches[node.comfyAgentSelectedTopic] = [];
    }

    function appendMessage(role, text) {
        const row = el("div", `comfy-agent-chat-msg comfy-agent-chat-${role}`);
        row.append(el("div", "comfy-agent-chat-role", role === "user" ? "You" : "Agent"));
        row.append(el("div", "comfy-agent-chat-text", text));
        log.append(row);
        log.scrollTop = log.scrollHeight;
        return row.querySelector(".comfy-agent-chat-text");
    }

    function renderHistory(topic) {
        const branch = node.comfyAgentHistoryBranches[topic] || [];
        log.replaceChildren();
        for (const msg of branch) {
            appendMessage(msg.role, msg.content);
        }
    }

    function saveBranchesToLocalStorage() {
        try {
            localStorage.setItem("comfy_agent_chat_branches", JSON.stringify(node.comfyAgentHistoryBranches));
        } catch (err) {
            console.warn("[comfy_agent] could not save chat branches to localStorage:", err);
        }
    }

    function setBusy(busy) {
        node.comfyAgentChatting = busy;
        sendBtn.disabled = busy;
        topicInput.disabled = busy;
        sendBtn.textContent = busy ? "..." : "Send";
    }

    async function send() {
        const question = input.value.trim();
        const topic = (topicInput.value || "default").trim() || "default";
        if (!question) return;
        if (node.comfyAgentChatting) return;

        // Ensure the selected topic branch exists
        if (!node.comfyAgentHistoryBranches[topic]) {
            node.comfyAgentHistoryBranches[topic] = [];
        }

        const branchHistory = node.comfyAgentHistoryBranches[topic];
        // Append user question to the selected topic branch
        branchHistory.push({ role: "user", content: question });

        // Switch to this topic's rendering immediately so the UI reflects the new message
        node.comfyAgentSelectedTopic = topic;
        renderHistory(topic);

        input.value = "";
        topicInput.value = "";

        setBusy(true);

        const chatId = node.comfyAgentChatId;
        // chat_id stays constant across topic switches

        // Streaming target. Tokens arrive over the WS and are appended here;
        // created up front so the reply area appears immediately rather than
        // after the first token, which can be seconds away on a local model.
        const replyEl = appendMessage("assistant", "");
        let streamed = "";
        node.comfyAgentOnChatToken = (token) => {
            streamed += token;
            replyEl.textContent = streamed;
            log.scrollTop = log.scrollHeight;
        };

        try {
            const resp = await api.fetchApi("/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: branchHistory.slice(-MAX_HISTORY),
                    graph: captureGraph(app),
                    last_error: node.comfyAgentLastError || "",
                    sid: api.clientId,
                    chat_id: chatId,
                    topic: topic,
                }),
            });

            if (!resp.ok) {
                let detail = "";
                try {
                    detail = (await resp.json())?.error || "";
                } catch (err) {
                    console.warn("[comfy_agent] could not parse chat error body:", err);
                }
                replyEl.textContent = `Error: ${detail || `server returned ${resp.status}`}`;
                replyEl.classList.add("comfy-agent-chat-error");
                // Drop the unanswered question so the next turn does not resend
                // a history ending in two consecutive user messages.
                branchHistory.pop();
                node.comfyAgentHistoryBranches[topic] = branchHistory;
                // Re-render from updated branch history
                renderHistory(topic);
                saveBranchesToLocalStorage();
                return;
            }

            const result = await resp.json();
            const reply = result.reply ?? streamed;
            replyEl.textContent = reply;
            branchHistory.push({ role: "assistant", content: reply });
            node.comfyAgentHistoryBranches[topic] = branchHistory;
            // Re-render from updated branch history (includes new assistant message)
            renderHistory(topic);
            saveBranchesToLocalStorage();

            // If topic changed in response, ensure branch exists
            if (result.topic && result.topic !== topic) {
                node.comfyAgentHistoryBranches[result.topic] = node.comfyAgentHistoryBranches[result.topic] || [];
            }
        } catch (err) {
            replyEl.textContent = `Error: request failed (${err})`;
            replyEl.classList.add("comfy-agent-chat-error");
            branchHistory.pop();
            node.comfyAgentHistoryBranches[topic] = branchHistory;
            renderHistory(topic);
            saveBranchesToLocalStorage();
        } finally {
            node.comfyAgentOnChatToken = null;
            // chat_id stays constant — do not reset
            setBusy(false);
            log.scrollTop = log.scrollHeight;
        }
    }

    // --- Topic tab switching UI ---
    const topicTabsContainer = el("div", "comfy-agent-chat-topic-tabs");
    root.insertBefore(topicTabsContainer, log);
    // --------------------------------

    function renderTopicTabs() {
        // Collect all known topics (including default)
        const topics = ["default"];
        for (const key of Object.keys(node.comfyAgentHistoryBranches)) {
            if (key && key !== "default") topics.push(key);
        }
        // Remove old tabs
        topicTabsContainer.replaceChildren();
        // Create a tab per topic
        for (const topic of topics) {
            const tab = el("span", `comfy-agent-chat-topic-tab${topic === node.comfyAgentSelectedTopic ? " comfy-agent-chat-topic-tab-active" : ""}`, topic);
            tab.title = `Switch to ${topic} branch`;
            on(tab, "click", () => {
                node.comfyAgentSelectedTopic = topic;
                renderHistory(topic);
                // Do NOT clear input; new messages will go to this topic
            });
            topicTabsContainer.append(tab);
        }
    }

    on(sendBtn, "click", send);
    on(input, "keydown", (event) => {
        // Enter sends; Shift+Enter newlines. Stop propagation either way so
        // LiteGraph's canvas keybindings don't also act on the keystroke.
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.stopPropagation();
            send();
        }
    });
    // Typing in a node must not trigger canvas shortcuts (delete node, etc.).
    for (const evt of ["keydown", "keyup", "keypress"]) {
        on(input, evt, (e) => e.stopPropagation());
    }
    on(topicInput, "keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
        }
    });

    closeChatBtn.onclick = cleanupChat;

    // Save as blueprint button handler
    on(saveBlueprintBtn, "click", () => {
        // Use default node types for txt2img workflow, or could extract from conversation
        const defaultNodeTypes = [
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "EmptyLatentImage",
            "KSampler",
            "VAEDecode",
            "SaveImage",
        ];

        // Build widget values from defaults (could be enhanced to use actual chat context)
        const widgetValues = {};
        defaultNodeTypes.forEach((type, idx) => {
            const typeDef = {
                CheckpointLoaderSimple: { widgets_values: ["sd_xl_base_1.0.safetensors"], widgets_named: { ckpt_name: "sd_xl_base_1.0.safetensors" } },
                CLIPTextEncode: { widgets_values: ["masterpiece, best quality"], widgets_named: { text: "masterpiece, best quality" } },
                EmptyLatentImage: { widgets_values: [1024, 1024, 1], widgets_named: { width: 1024, height: 1024, batch_size: 1 } },
                KSampler: { widgets_values: [0, "fixed", 20, 7.0, "dpmpp_2m", "karras", 1.0], widgets_named: { seed: 0, steps: 20, cfg: 7.0, sampler_name: "dpmpp_2m", scheduler: "karras", denoise: 1.0 } },
                VAEDecode: {},
                SaveImage: { widgets_values: ["ComfyUI"], widgets_named: { filename_prefix: "ComfyUI" } },
            }[type];
            if (typeDef) {
                widgetValues[type] = typeDef;
            }
        });

        const result = generateSubgraphFromSuggestions(defaultNodeTypes, widgetValues);
        if (!result) {
            alert("Failed to generate blueprint: unknown node types");
            return;
        }

        // Validate each node type with LiteGraph.createNode()
        let allValid = true;
        for (const nodeDef of result.nodes) {
            try {
                const testNode = LiteGraph.createNode(nodeDef.type);
                if (!testNode) {
                    console.warn(`[comfy_agent] LiteGraph.createNode returned null for ${nodeDef.type}`);
                    allValid = false;
                }
            } catch (err) {
                console.warn(`[comfy_agent] LiteGraph.createNode threw for ${nodeDef.type}:`, err);
                allValid = false;
            }
        }

        if (!allValid) {
            alert("Blueprint generated but some node types could not be validated with LiteGraph");
            return;
        }

        // Save to blueprint_defs/ directory
        const blueprintName = `${result.extra.comfy_agent_blueprint.name}_${Date.now()}.json`;
        const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        // Try to prompt save, or auto-drop in blueprint_defs/
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    types: [{ suggestedName: blueprintName, accept: { ".json": [".json"] } }],
                });
                const fileHandle = await window.showSaveFilePicker({
                    types: [{ suggestedName: blueprintName, accept: { ".json": [".json"] } }],
                });
                // Write to file
                const writable = await fileHandle.createWritable();
                await writable.write(JSON.stringify(result, null, 2));
                await writable.close();
                alert(`Blueprint saved to ${blueprintName}`);
            } catch (e) {
                // Fallback: just show the JSON and suggest user save it
                console.log("Blueprint JSON:", JSON.stringify(result, null, 2));
                alert(`Blueprint generated: ${blueprintName}. Copy the JSON above and save to blueprint_defs/`);
            }
        } else {
            // No file picker available - just alert with the JSON and suggest save location
            console.log("Blueprint JSON:", JSON.stringify(result, null, 2));
            alert(`Blueprint generated: ${blueprintName}. Save to blueprint_defs/ directory.`);
        }
    });

    // Initial render: history of selected topic, tabs
    renderHistory(node.comfyAgentSelectedTopic);
    renderTopicTabs();

    return {
        root,
        renderHistory,
        clear() {
            node.comfyAgentChatHistory = [];
            node.comfyAgentHistoryBranches = {};
            node.comfyAgentSelectedTopic = "default";
            node.comfyAgentChatId = newChatId();
            // Clear localStorage
            try { localStorage.removeItem("comfy_agent_chat_branches"); } catch (e) {}
            renderHistory();
            renderTopicTabs();
        },
        /** Detach every listener and drop the streaming callback. Called from
         *  the node's onRemoved so a deleted node leaves nothing behind. */
        destroy() {
            window.removeEventListener("beforeunload", cleanupChat);
            for (const [target, type, handler] of listeners) {
                target.removeEventListener(type, handler);
            }
            listeners.length = 0;
            node.comfyAgentOnChatToken = null;
            log.replaceChildren();
            // Persist branches on destroy
            saveBranchesToLocalStorage();
            // Remove topic tabs and input row from DOM
            topicTabsContainer.replaceChildren();
            topicRow.replaceChildren();
        },

        /** Clean up all CHAT resources when the panel closes */
        cleanupChat() {
            // Remove all WS event listeners attached during the CHAT session
            if (node.comfyAgentWSListeners) {
                for (const [target, type, handler] of node.comfyAgentWSListeners) {
                    target.removeEventListener(type, handler);
                }
                node.comfyAgentWSListeners.length = 0;
            }
            // Dereference LiteGraph node widgets created during CHAT
            if (node.comfyAgentLiteGraphWidgets) {
                for (const widget of node.comfyAgentLiteGraphWidgets) {
                    if (widget && typeof widget.destroy === "function") {
                        widget.destroy();
                    } else {
                        widget = null;
                    }
                }
                node.comfyAgentLiteGraphWidgets.length = 0;
            }
            // Clear conversation history from localStorage if persisted
            try { localStorage.removeItem("comfy_agent_chat_branches"); } catch (e) {}
            // Dispatch a chat:cleaned event for other modules to react
            document.dispatchEvent(new Event("chat:cleaned"));
        }

        // Ensure cleanup also happens on explicit "Close" button click,
        // not just page unload (handled above via closeChatBtn.onclick)

        window.addEventListener("beforeunload", cleanupChat);

        return {
            root,
            renderHistory,
            clear() {
            node.comfyAgentChatHistory = [];
            node.comfyAgentHistoryBranches = {};
            node.comfyAgentSelectedTopic = "default";
            node.comfyAgentChatId = newChatId();
            // Clear localStorage
            try { localStorage.removeItem("comfy_agent_chat_branches"); } catch (e) {}
            renderHistory();
            renderTopicTabs();
        },
        destroy,
    };
}