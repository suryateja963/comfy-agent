// Turning /agent/health into one line for the node's status widget.
//
// Deliberately a separate module with no ComfyUI imports: agent_node_ui.js
// imports ../../scripts/app.js, which only resolves inside a running ComfyUI,
// so anything living there cannot be unit-tested at all. This is the logic
// that decides what a user is told about a broken setup, which is worth
// covering.

/**
 * One status line describing the first real problem, or null when there is
 * nothing worth saying.
 *
 * The server ranks hints most-blocking-first, so leading with hints[0] avoids
 * burying "Ollama is down" under a note about the embedding model.
 *
 * @param {object|null} health - parsed /agent/health response, or null when
 *   the endpoint could not be reached (older server, or still starting).
 * @returns {string|null} status text, or null to leave the status untouched.
 */
export function healthSummary(health) {
    if (!health) return null;
    const hints = Array.isArray(health.hints) ? health.hints : [];
    if (!hints.length) return null;
    // Blueprint builds need no model at all, so a dead Ollama is a limitation
    // rather than a dead node — say so, or the user assumes nothing works.
    if (!health.ollama) return `${hints[0]} Blueprint builds still work.`;
    return hints[0];
}
