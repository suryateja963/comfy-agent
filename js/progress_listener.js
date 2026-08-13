// WebSocket progress streaming. Section 2G.
import { api } from "../../scripts/api.js";

export function listenForAgentProgress(onMessage) {
    api.addEventListener("comfy_agent.progress", (event) => {
        onMessage(event.detail);
    });
}

// CHAT mode token stream. Fires once per token so the reply renders as it is
// generated rather than appearing all at once when the request completes.
export function listenForChatTokens(onToken) {
    api.addEventListener("comfy_agent.chat_token", (event) => {
        onToken(event.detail);
    });
}
