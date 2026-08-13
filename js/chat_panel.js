// CHAT mode UI. Builds a DOM chat panel hosted inside the node via
// node.addDOMWidget, and drives POST /agent/chat with WS token streaming.
import { api } from "../../scripts/api.js";

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
    const inputRow = el("div", "comfy-agent-chat-input-row");
    const input = document.createElement("textarea");
    input.className = "comfy-agent-chat-input";
    input.rows = 2;
    input.placeholder = "Ask about your workflow, a node, or an error...";
    const sendBtn = el("button", "comfy-agent-chat-send", "Send");

    inputRow.append(input, sendBtn);
    root.append(log, inputRow);

    // Conversation state lives on the node so it survives re-renders and is
    // captured by workflow save/load via onSerialize below.
    node.comfyAgentChatHistory = node.comfyAgentChatHistory || [];
    node.comfyAgentChatting = false;

    function appendMessage(role, text) {
        const row = el("div", `comfy-agent-chat-msg comfy-agent-chat-${role}`);
        row.append(el("div", "comfy-agent-chat-role", role === "user" ? "You" : "Agent"));
        row.append(el("div", "comfy-agent-chat-text", text));
        log.append(row);
        log.scrollTop = log.scrollHeight;
        return row.querySelector(".comfy-agent-chat-text");
    }

    function renderHistory() {
        log.replaceChildren();
        for (const msg of node.comfyAgentChatHistory) {
            appendMessage(msg.role, msg.content);
        }
    }

    function setBusy(busy) {
        node.comfyAgentChatting = busy;
        sendBtn.disabled = busy;
        input.disabled = busy;
        sendBtn.textContent = busy ? "..." : "Send";
    }

    async function send() {
        const question = input.value.trim();
        if (!question) return;
        if (node.comfyAgentChatting) return;

        input.value = "";
        node.comfyAgentChatHistory.push({ role: "user", content: question });
        appendMessage("user", question);

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
                    messages: node.comfyAgentChatHistory.slice(-MAX_HISTORY),
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
                node.comfyAgentChatHistory.pop();
                return;
            }

            const result = await resp.json();
            const reply = result.reply ?? streamed;
            replyEl.textContent = reply;
            node.comfyAgentChatHistory.push({ role: "assistant", content: reply });
        } catch (err) {
            replyEl.textContent = `Error: request failed (${err})`;
            replyEl.classList.add("comfy-agent-chat-error");
            node.comfyAgentChatHistory.pop();
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
    // Typing in a node must not trigger canvas shortcuts (delete node, etc.).
    for (const evt of ["keydown", "keyup", "keypress"]) {
        on(input, evt, (e) => e.stopPropagation());
    }

    renderHistory();

    return {
        root,
        renderHistory,
        clear() {
            node.comfyAgentChatHistory = [];
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
        },
    };
}
