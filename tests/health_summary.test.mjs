// Tests for js/health_summary.js — the logic that decides what a user is told
// when their setup is broken. Previously untestable because it lived in
// agent_node_ui.js, which imports ComfyUI's app.js and cannot be loaded
// outside a running ComfyUI.
//
//     node --test tests/health_summary.test.mjs

import test from "node:test";
import assert from "node:assert/strict";
import { healthSummary } from "../js/health_summary.js";

test("a healthy install produces no status line", () => {
    // Nothing to say is better than a reassuring message the user has to read
    // and dismiss on every node they create.
    assert.equal(healthSummary({ ollama: true, hints: [] }), null);
});

test("an unreachable health endpoint says nothing", () => {
    // Older server, or ComfyUI still starting. Not knowing is not the same as
    // being broken, and claiming otherwise would be a false alarm.
    assert.equal(healthSummary(null), null);
});

test("a down Ollama is reported as a limitation, not a dead node", () => {
    const msg = healthSummary({
        ollama: false,
        hints: ["Ollama is not reachable at http://127.0.0.1:11434 — start it with `ollama serve`."],
    });
    assert.match(msg, /ollama serve/);
    // The blueprint path needs no model at all; without this the user assumes
    // the whole node is unusable.
    assert.match(msg, /Blueprint builds still work/);
});

test("the first hint wins so the most blocking problem leads", () => {
    // The server ranks hints most-blocking-first. Reporting a later one would
    // bury "Ollama is down" under a note about the embedding model.
    const msg = healthSummary({
        ollama: true,
        hints: ["Primary model missing — run `ollama pull qwen3.5:0.8b`.", "Embedding model missing."],
    });
    assert.match(msg, /Primary model missing/);
    assert.doesNotMatch(msg, /Embedding/);
});

test("a healthy Ollama does not get the blueprint consolation", () => {
    const msg = healthSummary({ ollama: true, hints: ["Embedding model missing."] });
    assert.doesNotMatch(msg, /Blueprint builds still work/);
});

test("a malformed hints field is treated as no hints", () => {
    // The response is parsed JSON from the network; a non-array here must not
    // throw inside node creation and break the whole extension.
    for (const bad of [undefined, null, "not an array", 42, {}]) {
        assert.equal(healthSummary({ ollama: true, hints: bad }), null);
    }
});

test("the MODIFY warning reaches the user", () => {
    // Regression guard: this hint is the only in-product signal that MODIFY
    // will produce unreliable edits on this machine.
    const msg = healthSummary({
        ollama: true,
        hints: ["No escalation model installed — run `ollama pull qwen3.5:4b`. MODIFY is unreliable without one."],
    });
    assert.match(msg, /MODIFY is unreliable/);
    assert.match(msg, /ollama pull qwen3\.5:4b/);
});
