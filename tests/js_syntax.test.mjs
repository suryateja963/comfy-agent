// Parse every file ComfyUI loads from js/. The behaviour suites can only
// import modules with no ComfyUI imports, so a syntax error in, say,
// chat_panel.js passed every test while agent_node_ui.js — which imports it —
// failed to load in the browser and took the whole node UI down with it.
//
//     node --test tests/js_syntax.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const JS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "js");

const files = readdirSync(JS_DIR).filter((name) => name.endsWith(".js"));

test("js/ is not empty", () => {
    assert.ok(files.length > 0, `no .js files found in ${JS_DIR}`);
});

for (const name of files) {
    test(`${name} parses as an ES module`, () => {
        // --check decides module vs script by extension, and these files use
        // import/export, so check a .mjs copy rather than the .js itself.
        const dir = mkdtempSync(join(tmpdir(), "comfy-agent-js-"));
        try {
            const copy = join(dir, name.replace(/\.js$/, ".mjs"));
            copyFileSync(join(JS_DIR, name), copy);
            const result = spawnSync(process.execPath, ["--check", copy], { encoding: "utf8" });
            assert.equal(result.status, 0, `${name} has a syntax error:\n${result.stderr}`);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
}
