"""Render docs/comfy_agent_explained.pdf — how the node works.

Uses matplotlib's PDF backend, which is already present in the ComfyUI
environment, so building the document installs nothing. Run with any Python
that has matplotlib:

    python docs/make_pdf.py [output.pdf]

Content is deliberately kept in PAGES below rather than parsed from the
Markdown docs: this explains the *runtime behaviour* and is written to be read
in order, which is a different document from the reference material in
README.md and ARCHITECTURE.md.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# A4 portrait, in inches.
PAGE = (8.27, 11.69)

INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#1f5fa8"
WARN = "#8a5a1e"
RULE = "#d0d0d0"
BOX_BG = "#f4f6f8"
CODE_BG = "#eef1f4"

LEFT = 0.085
RIGHT = 0.915
WIDTH = RIGHT - LEFT


class Page:
    """One page, laid out top-down in figure coordinates."""

    def __init__(self, pdf: PdfPages, title: str, subtitle: str = "", number: int | None = None):
        self.pdf = pdf
        self.fig = plt.figure(figsize=PAGE)
        self.y = 0.945
        self.number = number

        self.fig.text(LEFT, self.y, title, size=19, weight="bold", color=INK, va="top")
        self.y -= 0.030
        if subtitle:
            self.fig.text(LEFT, self.y, subtitle, size=9.5, color=MUTED, va="top", style="italic")
            self.y -= 0.018
        self.rule()
        self.y -= 0.018

    # --- primitives -----------------------------------------------------

    def rule(self, color: str = RULE) -> None:
        self.fig.add_artist(
            plt.Line2D([LEFT, RIGHT], [self.y, self.y], color=color, linewidth=0.9)
        )

    def space(self, amount: float = 0.014) -> None:
        self.y -= amount

    def heading(self, text: str) -> None:
        self.space(0.012)
        self.fig.text(LEFT, self.y, text, size=12, weight="bold", color=ACCENT, va="top")
        self.y -= 0.024

    def para(self, text: str, size: float = 9.6, color: str = INK, indent: float = 0.0) -> None:
        """Wrapped body text. Line count drives the vertical advance, so pages
        never silently overlap."""
        wrapped = self._wrap(text, chars=int(101 - indent * 190))
        self.fig.text(
            LEFT + indent, self.y, "\n".join(wrapped),
            size=size, color=color, va="top", linespacing=1.55,
        )
        # The trailing gap is what separates consecutive paragraphs; too small
        # and they read as one block.
        self.y -= 0.0148 * len(wrapped) * (size / 9.6) + 0.013

    def bullet(self, text: str, marker: str = "•", color: str = INK) -> None:
        wrapped = self._wrap(text, chars=96)
        self.fig.text(LEFT + 0.012, self.y, marker, size=9.6, color=color, va="top")
        self.fig.text(
            LEFT + 0.032, self.y, "\n".join(wrapped),
            size=9.6, color=color, va="top", linespacing=1.55,
        )
        self.y -= 0.0148 * len(wrapped) + 0.007

    def code(self, lines: list[str]) -> None:
        height = 0.0138 * len(lines) + 0.012
        self.fig.add_artist(
            FancyBboxPatch(
                (LEFT, self.y - height), WIDTH, height,
                boxstyle="round,pad=0.004", facecolor=CODE_BG,
                edgecolor="#dde2e7", linewidth=0.8, transform=self.fig.transFigure,
            )
        )
        self.fig.text(
            LEFT + 0.014, self.y - 0.010, "\n".join(lines),
            size=8.3, family="monospace", color=INK, va="top", linespacing=1.5,
        )
        self.y -= height + 0.010

    def callout(self, title: str, text: str, color: str = WARN) -> None:
        # A callout usually follows bullets or a table, which end tight against
        # the next element; without this the box lands on the last line.
        self.space(0.010)
        wrapped = self._wrap(text, chars=92)
        # 0.030 covers the title and top padding; the extra 0.010 is bottom
        # padding, without which the final line rides the box border.
        height = 0.0148 * len(wrapped) + 0.040
        self.fig.add_artist(
            FancyBboxPatch(
                (LEFT, self.y - height), WIDTH, height,
                boxstyle="round,pad=0.005", facecolor=BOX_BG,
                edgecolor=color, linewidth=1.1, transform=self.fig.transFigure,
            )
        )
        self.fig.text(LEFT + 0.016, self.y - 0.013, title, size=9.4, weight="bold", color=color, va="top")
        self.fig.text(
            LEFT + 0.016, self.y - 0.030, "\n".join(wrapped),
            size=9.2, color=INK, va="top", linespacing=1.5,
        )
        self.y -= height + 0.012

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        xs, acc = [], LEFT + 0.004
        for w in widths:
            xs.append(acc)
            acc += w * WIDTH

        for x, h in zip(xs, headers):
            self.fig.text(x, self.y, h, size=8.8, weight="bold", color=ACCENT, va="top")
        self.y -= 0.015
        self.rule()
        self.y -= 0.011

        for row in rows:
            tallest = 1
            for x, cell, w in zip(xs, row, widths):
                wrapped = self._wrap(cell, chars=max(8, int(w * 101)))
                tallest = max(tallest, len(wrapped))
                self.fig.text(
                    x, self.y, "\n".join(wrapped),
                    size=8.6, color=INK, va="top", linespacing=1.45,
                )
            self.y -= 0.0138 * tallest + 0.006
        self.y -= 0.004

    @staticmethod
    def _wrap(text: str, chars: int) -> list[str]:
        out: list[str] = []
        for para in text.split("\n"):
            if not para.strip():
                out.append("")
                continue
            line = ""
            for word in para.split():
                if len(line) + len(word) + 1 > chars and line:
                    out.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            out.append(line)
        return out

    def save(self) -> None:
        if self.number is not None:
            self.fig.text(
                0.5, 0.035, str(self.number), size=8.5, color=MUTED, ha="center"
            )
            self.fig.text(
                RIGHT, 0.035, "comfy_agent", size=8, color=MUTED, ha="right"
            )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def _box(fig, x, y, w, h, label, sub="", fc=BOX_BG, ec=ACCENT):
    fig.add_artist(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004", facecolor=fc,
            edgecolor=ec, linewidth=1.2, transform=fig.transFigure,
        )
    )
    fig.text(x + w / 2, y + h / 2 + (0.006 if sub else 0), label,
             size=8.8, weight="bold", color=INK, ha="center", va="center")
    if sub:
        fig.text(x + w / 2, y + h / 2 - 0.009, sub, size=7.4, color=MUTED,
                 ha="center", va="center")


def _arrow(fig, x1, y1, x2, y2, color=ACCENT):
    fig.add_artist(
        FancyArrowPatch(
            (x1, y1), (x2, y2), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.1,
        )
    )


def build(out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        _cover(pdf)
        _what_it_is(pdf, 2)
        _architecture(pdf, 3)
        _build_mode(pdf, 4)
        _pipeline(pdf, 5)
        _modify_mode(pdf, 6)
        _chat_mode(pdf, 7)
        _safety(pdf, 8)
        _operating(pdf, 9)

    meta = pdf.infodict if hasattr(pdf, "infodict") else None  # noqa: F841


# --- pages --------------------------------------------------------------

def _cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(LEFT, 0.70, "comfy_agent", size=40, weight="bold", color=INK)
    fig.text(LEFT, 0.655, "How the node works", size=17, color=ACCENT)
    fig.add_artist(plt.Line2D([LEFT, RIGHT], [0.632, 0.632], color=RULE, linewidth=1.2))
    fig.text(
        LEFT, 0.600,
        "A ComfyUI meta-node that builds workflows from a natural-language\n"
        "prompt, edits the workflow already on your canvas, and answers\n"
        "questions about your specific installation - entirely on your own\n"
        "machine, using a local Ollama model.",
        size=11.5, color=INK, va="top", linespacing=1.7,
    )
    for i, (k, v) in enumerate([
        ("Modes", "BUILD, MODIFY, CHAT"),
        ("Runs on", "ComfyUI V3 node API + local Ollama"),
        ("Network", "localhost only by default; nothing leaves the machine"),
        ("Verified against", "ComfyUI v0.32.0, 1767 installed node types"),
    ]):
        y = 0.455 - i * 0.030
        fig.text(LEFT, y, k, size=9.5, weight="bold", color=ACCENT)
        fig.text(LEFT + 0.20, y, v, size=9.5, color=INK)
    fig.text(
        LEFT, 0.10,
        "Generated from the source at docs/make_pdf.py.\n"
        "Every figure in this document was measured, not estimated.",
        size=8.4, color=MUTED, va="bottom", linespacing=1.6,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _what_it_is(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "1. What the node is", "and the problem it solves", n)
    p.para(
        "ComfyUI workflows are graphs. Building one means knowing which of ~1,700 installed "
        "node types you need, what each one's inputs and outputs are called, and how they "
        "connect. That knowledge is the barrier for new users and tedious for everyone else."
    )
    p.para(
        "comfy_agent is a node you drop on the canvas. You type what you want in plain "
        "language; it writes the workflow around itself. It is a meta-node: it does not "
        "process images or latents. It reads the ComfyUI registry, talks to a local language "
        "model, and draws on the canvas."
    )

    p.heading("Three modes")
    p.table(
        ["Mode", "What it does", "Needs a model?"],
        [
            ["BUILD", "Creates a new workflow from a prompt. Tries a pre-validated blueprint "
                      "first; falls back to generating one.", "Only for free-form"],
            ["MODIFY", "Edits the workflow already on your canvas - add, remove, or change "
                       "widget values. Fully undoable.", "Yes"],
            ["CHAT", "Answers questions about ComfyUI, your installed nodes and models, your "
                     "current graph, and your last error.", "Yes"],
        ],
        [0.13, 0.62, 0.25],
    )

    p.heading("What makes it different from a chatbot in a browser tab")
    p.bullet("It reads your actual installation. Node suggestions come from the ~1,700 node "
             "types you really have, not from what the model remembers about ComfyUI.")
    p.bullet("It reads your actual canvas. CHAT and MODIFY both receive the current graph, so "
             "\"why is this broken?\" is a question about your workflow.")
    p.bullet("It writes to the canvas directly. There is no copy-pasting JSON.")
    p.bullet("It runs locally. No prompt, workflow, or file listing leaves the machine.")

    p.callout(
        "The core design constraint",
        "The default model is small - under a billion parameters - so it fits alongside "
        "image generation on one GPU. Almost every design decision in this node exists to "
        "make a small model reliable: narrow questions, filtered context, and validating "
        "every answer against the real registry rather than trusting it.",
        color=ACCENT,
    )
    p.save()


def _architecture(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "2. How the pieces fit together", "three layers, one round trip", n)
    p.para(
        "The node is split across a Python backend, a JavaScript frontend, and the local "
        "Ollama server. The frontend owns the canvas; the backend owns generation. They meet "
        "at one HTTP request and a WebSocket for progress."
    )
    fig = p.fig
    y0 = p.y - 0.245

    _box(fig, LEFT, y0 + 0.175, 0.40, 0.055, "ComfyUI canvas (browser)", "the agent node + your graph")
    _box(fig, LEFT + 0.47, y0 + 0.175, 0.36, 0.055, "js/ frontend", "reads + writes the graph")
    _box(fig, LEFT, y0 + 0.085, 0.40, 0.055, "agent_routes.py", "HTTP routes, access control")
    _box(fig, LEFT + 0.47, y0 + 0.085, 0.36, 0.055, "generation/", "the staged pipeline")
    _box(fig, LEFT, y0, 0.40, 0.055, "Ollama (localhost)", "the language model")
    _box(fig, LEFT + 0.47, y0, 0.36, 0.055, "ComfyUI /object_info", "the real node registry")

    # Request path: canvas -> js -> routes -> generation, then each side's
    # dependency below it. Endpoints sit on box edges so no arrow floats.
    _arrow(fig, LEFT + 0.40, y0 + 0.202, LEFT + 0.47, y0 + 0.202)   # canvas -> js
    _arrow(fig, LEFT + 0.20, y0 + 0.175, LEFT + 0.20, y0 + 0.140)   # canvas -> routes
    _arrow(fig, LEFT + 0.40, y0 + 0.112, LEFT + 0.47, y0 + 0.112)   # routes -> generation
    _arrow(fig, LEFT + 0.20, y0 + 0.085, LEFT + 0.20, y0 + 0.055)   # routes -> Ollama
    _arrow(fig, LEFT + 0.65, y0 + 0.085, LEFT + 0.65, y0 + 0.056)   # generation -> registry
    p.y = y0 - 0.022

    p.heading("What each layer is responsible for")
    p.table(
        ["Layer", "Responsibility"],
        [
            ["js/canvas_writer.js", "Creating nodes, wiring links, finding empty canvas space, "
                                    "snapshotting for undo, applying edits."],
            ["js/agent_node_ui.js", "The node's buttons and status line; sends the request; "
                                    "reports what happened."],
            ["agent_routes.py", "Four HTTP routes, localhost-only access control, concurrency "
                                "limits."],
            ["generation/", "The staged pipeline that turns a prompt into a validated graph."],
            ["knowledge/", "Filtering the registry, validating link types, matching blueprints."],
            ["install_index.py", "A searchable index of every installed node, for CHAT."],
        ],
        [0.27, 0.73],
    )

    p.callout(
        "Why the backend never touches the canvas",
        "ComfyUI's graph lives in the browser. The backend only ever returns a description of "
        "what to draw; the frontend decides where it fits and applies it. That split is why an "
        "agent action can be snapshotted and undone.",
        color=ACCENT,
    )
    p.save()


def _build_mode(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "3. BUILD mode", "two paths, and why the fast one exists", n)
    p.para(
        "When you press Build Workflow, the node takes one of two paths. The choice is made "
        "before any model is consulted."
    )

    p.heading("Path 1 - blueprint (instant, no model)")
    p.para(
        "Eight complete workflows ship with the node, each validated against a real ComfyUI "
        "install. Your prompt is scored against their trigger keywords; a match is instantiated "
        "directly. This takes milliseconds, cannot hallucinate, and works with Ollama stopped."
    )
    p.code([
        "txt2img_sdxl     txt2img_flux     img2img_base     controlnet_base",
        "lora_stack       face_detailer    upscale_latent   video_wan",
    ])
    p.para(
        "Model-family keywords carry extra weight, so \"flux txt2img\" selects the FLUX "
        "blueprint rather than the SDXL one.",
        color=MUTED, size=9.2,
    )

    p.heading("Path 2 - free-form (the staged pipeline)")
    p.para(
        "If nothing matches, the node generates a workflow from scratch. This is where the "
        "small-model problem is real, and it is covered on the next page."
    )

    p.heading("What you get back")
    p.table(
        ["Outcome", "What the node does"],
        [
            ["Complete workflow", "Places it, pans the canvas to it, reports node and link counts."],
            ["Incomplete workflow", "Still places it. Nodes with unconnected required inputs are "
                                    "coloured amber and titled with what they need. The status "
                                    "says it cannot run yet."],
            ["Failure", "Nothing is drawn; the status names the cause."],
        ],
        [0.22, 0.78],
    )
    p.callout(
        "An honest partial result beats a false success",
        "A graph missing one connection is still worth having - you can finish it by hand in "
        "seconds. What is not acceptable is calling it ready and letting you discover the gap "
        "when the queue fails. The node distinguishes 'built' from 'runnable' everywhere.",
        color=WARN,
    )
    p.save()


def _pipeline(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "4. The generation pipeline", "why one big question becomes five small ones", n)
    p.para(
        "Asking a small model to emit a whole workflow in one go produces confident nonsense. "
        "Instead the work is split into stages, each asking for one narrow thing, and each "
        "output is checked against the real registry before the next stage runs."
    )

    fig = p.fig
    y = p.y - 0.075
    labels = [
        ("Pass 1", "classify"), ("Stage A", "pick nodes"), ("Stage B", "set widgets"),
        ("Stage C", "connect"), ("Stage D", "validate"),
    ]
    bw, gap = 0.145, 0.023
    for i, (a, b) in enumerate(labels):
        x = LEFT + i * (bw + gap)
        _box(fig, x, y, bw, 0.058, a, b)
        if i < len(labels) - 1:
            _arrow(fig, x + bw, y + 0.029, x + bw + gap, y + 0.029)
    p.y = y - 0.028

    p.table(
        ["Stage", "Asks the model", "Then checks"],
        [
            ["Pass 1", "Which categories does this task touch?",
             "Core categories are always added, so a bad guess cannot drop the loaders."],
            ["Stage A", "Which node types are needed?",
             "Every name must exist in the registry. Invented names are dropped; a suspiciously "
             "short list triggers a retry on a bigger model."],
            ["Stage B", "What values for this one node's widgets?",
             "Types coerced, numbers clamped to the declared range, dropdown values checked "
             "against the real options. One call per node, run in parallel."],
            ["Stage C", "Which outputs connect to which inputs?",
             "Every link re-validated for type compatibility. Unfilled required inputs trigger "
             "a targeted repair pass that asks only about the gaps."],
            ["Stage D", "(nothing - pure code)",
             "Structural check: no orphans, no empty spec, and whether the graph can actually run."],
        ],
        [0.10, 0.36, 0.54],
    )

    p.heading("Keeping the context small")
    p.para(
        "The full registry is about 2.7 MB - far more than a small model can hold. Pass 1's "
        "categories filter it, and a character budget trims what remains to roughly 2,500 "
        "tokens. Essential nodes such as the checkpoint loader and image saver are admitted "
        "before the budget applies, so they can never be squeezed out by a large custom pack."
    )
    p.save()


def _modify_mode(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "5. MODIFY mode", "editing a workflow you already have", n)
    p.para(
        "MODIFY changes what is on your canvas instead of building something new. It does not "
        "regenerate the workflow - it produces a diff, listing only what should change."
    )
    p.code([
        '{ "add_nodes":       ["PreviewImage"],',
        '  "remove_node_ids": [7],',
        '  "set_widgets":     [{"node_id": 5, "widget": "steps", "value": 35}] }',
    ])
    p.para(
        "A diff rather than a rewrite for two reasons. You asked to change one thing, so "
        "re-rolling everything would discard the parts you were happy with. And a diff can be "
        "shown to you and reversed; a wholesale replacement cannot."
    )

    p.heading("Everything is checked before anything is applied")
    p.bullet("Node ids must exist on your canvas. An invented id is rejected, not deleted.")
    p.bullet("Added node types must be installed here.")
    p.bullet("Widget names must exist on the node being changed.")
    p.bullet("Values must be scalars, then type-coerced and range-clamped - the same checks "
             "BUILD applies.")
    p.bullet("A diff that would delete most of the canvas triggers a second opinion from a "
             "larger model before it is trusted.")

    p.heading("Undo")
    p.para(
        "Before any agent action writes to the canvas, the whole graph is snapshotted. "
        "Undo Last Agent Action restores it. Undo is what makes the feature usable at all: "
        "without it, one bad edit teaches you never to press the button again."
    )
    p.para(
        "The snapshot restores the entire canvas to that moment, so manual edits made after "
        "the agent ran are reverted too.",
        color=MUTED, size=9.2,
    )

    p.callout(
        "MODIFY needs a capable model",
        "Reasoning about node ids that already exist is the hardest thing this node asks of a "
        "model. Measured on a real 7-node canvas: qwen3.5:4b and :9b produced the correct "
        "minimal edit every time, while the sub-1B model edited the wrong node with the wrong "
        "value. MODIFY therefore runs on the escalation model by design, and warns you on the "
        "node if none is installed. Run: ollama pull qwen3.5:4b",
        color=WARN,
    )
    p.save()


def _chat_mode(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "6. CHAT mode", "questions about your machine, not about ComfyUI in general", n)
    p.para(
        "Selecting CHAT swaps the Build button for a conversation panel inside the node. What "
        "makes it useful is what it can see. Every question is answered with four things in "
        "context:"
    )
    p.bullet("Your current canvas - trimmed to node types, titles, widget values and links.")
    p.bullet("The installed nodes most relevant to your question, retrieved from an index of "
             "every node type on this machine.")
    p.bullet("Your actual model files - checkpoints, LoRAs, VAEs, upscalers.")
    p.bullet("Your most recent execution error, captured automatically. You never paste a "
             "traceback.")

    p.heading("How retrieval works")
    p.para(
        "The registry is far too large to put in a prompt, so every node is described in one "
        "line and embedded once, then cached. Each question retrieves the top matches - "
        "typically 24 nodes - and only those are shown to the model."
    )
    p.para(
        "Three ranking details matter more than they look. Node names are split on case "
        "boundaries, so \"FaceDetailer\" can match \"make faces look better\". Filler words are "
        "excluded from name matching, because \"how do I make...\" otherwise ranks a node "
        "called MakeTrainingDataset first. And naming a node exactly outranks its variants, so "
        "asking about KSampler does not return KSamplerAdvanced instead."
    )

    p.heading("The index")
    p.table(
        ["Property", "Measured on a 1,767-node install"],
        [
            ["First build", "About 21 seconds, once. Starts in the background at ComfyUI startup."],
            ["Later starts", "Loaded from cache, about 1 second."],
            ["After installing a node pack", "Only the new nodes are embedded - about 1 second, "
                                             "not a full rebuild."],
            ["If the embedding model is missing", "Falls back to keyword search. Chat still works."],
        ],
        [0.34, 0.66],
    )
    p.callout(
        "CHAT replaced three planned modes",
        "EXPLAIN, DEBUG and SUGGEST were originally separate. In practice the distinction "
        "collapsed: someone asking \"why is my image black?\" wants an explanation, a diagnosis "
        "and a suggestion in one answer. Making them choose a mode first was friction with no "
        "payoff, so CHAT covers all three.",
        color=ACCENT,
    )
    p.save()


def _safety(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "7. Safety, privacy and failure", "what the node will and will not do", n)

    p.heading("Privacy")
    p.para(
        "Everything runs on your machine. The node talks to ComfyUI on localhost and Ollama on "
        "localhost. Your prompts, your workflows, and the list of your model files are never "
        "sent anywhere else. There are no API keys and no accounts."
    )

    p.heading("Access control")
    p.para(
        "All routes reject non-localhost callers by default. Remote access is opt-in per subnet "
        "and can require a shared token. Cross-origin requests are refused, so a web page you "
        "happen to have open cannot drive your canvas or burn your GPU."
    )

    p.heading("How it fails")
    p.table(
        ["When", "What happens"],
        [
            ["Ollama is stopped", "Blueprint builds still work. The node says so rather than "
                                  "appearing broken."],
            ["A model is missing", "The node reports it on creation, with the exact ollama pull "
                                   "command."],
            ["The model invents a node", "The name is not in the registry, so it is dropped "
                                         "before anything is drawn."],
            ["A link cannot be wired", "The build still lands; the status names each link that "
                                       "was skipped and why."],
            ["A built workflow errors", "The node notices the failure came from nodes it just "
                                        "placed and says so, keeping the error for CHAT."],
            ["Anything else", "The build fails with a stated reason. The canvas is left alone."],
        ],
        [0.26, 0.74],
    )

    p.callout(
        "The principle behind all of it",
        "A wrong workflow that claims to be right costs far more than a refusal. Every stage "
        "validates against the real registry rather than trusting the model, and anything that "
        "cannot be verified is reported rather than hidden.",
        color=ACCENT,
    )
    p.save()


def _operating(pdf: PdfPages, n: int) -> None:
    p = Page(pdf, "8. Running it", "setup, models, and where to look when something breaks", n)

    p.heading("Requirements")
    p.para("ComfyUI with the V3 node API, and Ollama running locally.")
    p.code([
        "ollama pull qwen3.5:0.8b      # base model - free-form builds",
        "ollama pull qwen3.5:4b        # escalation - required for MODIFY",
        "ollama pull nomic-embed-text  # semantic search for CHAT",
    ])
    p.para(
        "Blueprint builds need none of these. Free-form builds work on the base model alone. "
        "MODIFY and answer quality both improve materially with the 4B.",
        color=MUTED, size=9.2,
    )

    p.heading("Check the setup before blaming the node")
    p.para(
        "The node calls a health endpoint when it is created and shows the first real problem "
        "in its own status line, with the command that fixes it. For the full picture:"
    )
    p.code(["curl http://127.0.0.1:8188/agent/health"])
    p.para(
        "That reports whether Ollama is reachable, which models are present, whether the search "
        "index is built, and how many blueprints loaded."
    )

    p.heading("Start here")
    p.para(
        "The examples/ folder holds four loadable workflows, one per mode. The first uses a "
        "blueprint and works with Ollama stopped - which makes it a clean way to tell an "
        "installation problem apart from a model problem."
    )

    p.heading("Tuning")
    p.table(
        ["If", "Change"],
        [
            ["Image generation runs out of memory while chatting",
             "Lower the chat context window in config.py - retrieval means answers barely change."],
            ["Free-form builds wire poorly", "Pull a larger escalation model."],
            ["Builds are too large or too small", "Adjust max_nodes on the node itself (4-20)."],
            ["You want a new instant workflow", "Save a working graph into blueprint_defs/ and "
                                                "add its trigger keywords."],
        ],
        [0.42, 0.58],
    )

    p.space(0.006)
    p.rule()
    p.space(0.012)
    p.para(
        "Reference material lives beside this document: README.md for setup and troubleshooting, "
        "ARCHITECTURE.md for the design record, and examples/README.md for the sample workflows.",
        size=9.0, color=MUTED,
    )
    p.save()


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
        pathlib.Path(__file__).resolve().parent / "comfy_agent_explained.pdf"
    )
    build(target)
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
