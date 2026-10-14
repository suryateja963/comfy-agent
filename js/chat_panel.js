// CHAT mode UI. Builds a DOM chat panel hosted inside the node via
// node.addDOMWidget, and drives POST /agent/chat with WS token streaming.
//
// Conversations are split into topics: each topic is its own thread with its
// own history, so "help with my LoRA setup" does not crowd out "why is my
// image black?" in the context the model sees. Topics are saved with the
// workflow through serializeInto()/restore(), called from the node's
// onSerialize/onConfigure — never localStorage, which is shared by every node
// in every workflow and leaked one workflow's chats into the next.
import { api } from "../../scripts/api.js";

// Kept in sync with config.CHAT_MAX_HISTORY_MESSAGES. The server trims to its
// own limit regardless; trimming here too keeps the request small.
const MAX_HISTORY = 12;

const DEFAULT_TOPIC = "default";
const MAX_TOPIC_CHARS = 40;

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

function normalizeTopic(raw) {
    const topic = String(raw ?? "").trim().slice(0, MAX_TOPIC_CHARS);
    return topic || DEFAULT_TOPIC;
}

// Null-prototype store: topic names are user-typed, and a plain object would
// let a topic called "__proto__" or "constructor" collide with Object's own.
function emptyBranches() {
    const branches = Object.create(null);
    branches[DEFAULT_TOPIC] = [];
    return branches;
}

function isMessage(msg) {
    return (
        msg &&
        (msg.role === "user" || msg.role === "assistant") &&
        typeof msg.content === "string"
    );
}

export function createChatPanel(app, node) {
    const root = el("div", "comfy-agent-chat");

    const topicTabs = el("div", "comfy-agent-chat-topic-tabs");
    const topicInput = document.createElement("input");
    topicInput.type = "text";
    topicInput.className = "comfy-agent-chat-topic-input";
    topicInput.placeholder = "New topic + Enter";
    topicInput.maxLength = MAX_TOPIC_CHARS;
    const topicRow = el("div", "comfy-agent-chat-topic-row");
    topicRow.append(topicTabs, topicInput);

    const log = el("div", "comfy-agent-chat-log");
    const inputRow = el("div", "comfy-agent-chat-input-row");
    const input = document.createElement("textarea");
    input.className = "comfy-agent-chat-input";
    input.rows = 2;
    input.placeholder = "Ask about your workflow, a node, or an error...";
    const sendBtn = el("button", "comfy-agent-chat-send", "Send");

    inputRow.append(input, sendBtn);
    root.append(topicRow, log, inputRow);

    // Conversation state lives on the node so it survives re-renders and is
    // captured by workflow save/load via serializeInto()/restore().
    node.comfyAgentHistoryBranches = node.comfyAgentHistoryBranches || emptyBranches();
    node.comfyAgentSelectedTopic = normalizeTopic(node.comfyAgentSelectedTopic);
    node.comfyAgentChatting = false;

    function branch(topic) {
        const branches = node.comfyAgentHistoryBranches;
        if (!Array.isArray(branches[topic])) branches[topic] = [];
        return branches[topic];
    }

    function appendMessage(role, text) {
        const row = el("div", `comfy-agent-chat-msg comfy-agent-chat-${role}`);
        row.append(el("div", "comfy-agent-chat-role", role === "user" ? "You" : "Agent"));
        row.append(el("div", "comfy-agent-chat-text", text));
        log.append(row);
        log.scrollTop = log.scrollHeight;
        return row.querySelector(".comfy-agent-chat-text");
    }

    function renderTopicTabs() {
        topicTabs.replaceChildren();
        const topics = Object.keys(node.comfyAgentHistoryBranches);
        if (!topics.includes(DEFAULT_TOPIC)) topics.unshift(DEFAULT_TOPIC);
        for (const topic of topics) {
            const active = topic === node.comfyAgentSelectedTopic;
            const tab = el(
                "button",
                `comfy-agent-chat-topic-tab${active ? " comfy-agent-chat-topic-tab-active" : ""}`,
                topic
            );
            tab.title = `Switch to the "${topic}" conversation`;
            tab.disabled = node.comfyAgentChatting;
            // Tabs are rebuilt on every render, so these listeners go away
            // with the elements rather than through destroy().
            tab.addEventListener("click", () => selectTopic(topic));
            topicTabs.append(tab);
        }
    }

    function renderHistory() {
        log.replaceChildren();
        for (const msg of branch(node.comfyAgentSelectedTopic)) {
            appendMessage(msg.role, msg.content);
        }
        renderTopicTabs();
    }

    function selectTopic(raw) {
        // Switching mid-reply would stream the answer into the wrong thread.
        if (node.comfyAgentChatting) return;
        const topic = normalizeTopic(raw);
        branch(topic);
        node.comfyAgentSelectedTopic = topic;
        renderHistory();
    }

    function setBusy(busy) {
        node.comfyAgentChatting = busy;
        sendBtn.disabled = busy;
        input.disabled = busy;
        topicInput.disabled = busy;
        sendBtn.textContent = busy ? "..." : "Send";
        renderTopicTabs();
    }

    async function send() {
        const question = input.value.trim();
        if (!question) return;
        if (node.comfyAgentChatting) return;

        // Captured once: the reply belongs to the topic the question was asked
        // in, whatever happens to the selection while it streams.
        const history = branch(node.comfyAgentSelectedTopic);

        input.value = "";
        history.push({ role: "user", content: question });
        appendMessage("user", question);

        // A fresh id per request: the WS token router matches on it, and an
        // id saved with the workflow would be shared by a copy-pasted node.
        const chatId = newChatId();
        node.comfyAgentChatId = chatId;
        setBusy(true);

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
                    messages: history.slice(-MAX_HISTORY),
                    graph: captureGraph(app),
                    last_error: node.comfyAgentLastError || "",
                    sid: api.clientId,
                    chat_id: chatId,
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
                history.pop();
                return;
            }

            const result = await resp.json();
            const reply = result.reply ?? streamed;
            replyEl.textContent = reply;
            history.push({ role: "assistant", content: reply });
        } catch (err) {
            replyEl.textContent = `Error: request failed (${err})`;
            replyEl.classList.add("comfy-agent-chat-error");
            history.pop();
        } finally {
            node.comfyAgentOnChatToken = null;
            if (node.comfyAgentChatId === chatId) node.comfyAgentChatId = null;
            setBusy(false);
            log.scrollTop = log.scrollHeight;
        }
    }

    // Tracked so destroy() can detach them. Removing a node and adding another
    // otherwise leaves the old panel's handlers attached to DOM that is still
    // reachable through the closures below.
    const listeners = [];
    function on(target, type, handler) {
        target.addEventListener(type, handler);
        listeners.push([target, type, handler]);
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
    on(topicInput, "keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            if (topicInput.value.trim()) selectTopic(topicInput.value);
            topicInput.value = "";
        }
    });
    // Typing in a node must not trigger canvas shortcuts (delete node, etc.).
    for (const field of [input, topicInput]) {
        for (const evt of ["keydown", "keyup", "keypress"]) {
            on(field, evt, (e) => e.stopPropagation());
        }
    }

    renderHistory();

    return {
        root,
        renderHistory,

        /** Write the conversation into a workflow-save payload. */
        serializeInto(info) {
            info.comfyAgentHistoryBranches = node.comfyAgentHistoryBranches;
            info.comfyAgentSelectedTopic = node.comfyAgentSelectedTopic;
            // The pre-topics key, so a workflow saved here still shows its
            // main thread in an older version of the node.
            info.comfyAgentChatHistory = branch(DEFAULT_TOPIC);
        },

        /** Load the conversation from a saved workflow. Validates every
         *  message, since a hand-edited .json is replayed into the model. */
        restore(info) {
            const branches = emptyBranches();
            const saved = info?.comfyAgentHistoryBranches;
            if (saved && typeof saved === "object") {
                for (const [rawTopic, messages] of Object.entries(saved)) {
                    if (!Array.isArray(messages)) continue;
                    branches[normalizeTopic(rawTopic)] = messages.filter(isMessage);
                }
            } else if (Array.isArray(info?.comfyAgentChatHistory)) {
                // Saved before topics existed: the single thread is the default.
                branches[DEFAULT_TOPIC] = info.comfyAgentChatHistory.filter(isMessage);
            }
            node.comfyAgentHistoryBranches = branches;
            const topic = normalizeTopic(info?.comfyAgentSelectedTopic);
            node.comfyAgentSelectedTopic = topic in branches ? topic : DEFAULT_TOPIC;
            renderHistory();
        },

        /** Clear Chat: discard every topic. */
        clear() {
            if (node.comfyAgentChatting) return;
            node.comfyAgentHistoryBranches = emptyBranches();
            node.comfyAgentSelectedTopic = DEFAULT_TOPIC;
            renderHistory();
        },

        /** Detach every listener and drop the streaming callback. Called from
         *  the node's onRemoved so a deleted node leaves nothing behind. */
        destroy() {
            for (const [target, type, handler] of listeners) {
                target.removeEventListener(type, handler);
            }
            listeners.length = 0;
            node.comfyAgentOnChatToken = null;
            log.replaceChildren();
            topicTabs.replaceChildren();
        },
    };
}
