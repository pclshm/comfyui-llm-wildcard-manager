// Frontend extension for the LLM Wildcard Manager nodes.
//
//   * LLMWildcardManager — generated prompt panel + categories table with
//                          live entry view per category. The Python side hides
//                          the JSON `categories` widget and we render a nicer
//                          table on top of it.
//   * LLMWildcardReport  — structured collapsible per-slot view + raw text
//                          panel. Stretches with the node body.
//   * LLMServerConfig    — uses ComfyUI's built-in widget rendering. No JS
//                          customisation needed; left out on purpose.
//
// Workflows continue to function without this extension; the categories
// JSON widget remains the source of truth for headless / API runs.

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readCategoriesJSON(widget) {
    try {
        const v = JSON.parse(widget.value || "{}");
        return v && typeof v === "object" && !Array.isArray(v) ? v : {};
    } catch {
        return {};
    }
}

function writeCategoriesJSON(widget, rows) {
    const obj = {};
    for (const { name, desc } of rows) {
        const k = (name || "").trim();
        if (k) obj[k] = desc || "";
    }
    widget.value = JSON.stringify(obj, null, 2);
}

// Design-brief widget helpers.
// The brief mirrors the Python `normalize_brief` shape:
//   { refined_idea: string, fixed_traits: [..], forbidden_axes: [..],
//     scene_bans: [..] }
// Empty/missing widget value means "let the LLM generate one"; any non-empty
// content means "user has edited; use as-is and skip the LLM brief step".
function emptyBrief() {
    return { refined_idea: "", fixed_traits: [],
             forbidden_axes: [], scene_bans: [] };
}

function readBriefJSON(widget) {
    try {
        const raw = JSON.parse(widget.value || "{}");
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
            return emptyBrief();
        }
        const norm = emptyBrief();
        if (typeof raw.refined_idea === "string") {
            norm.refined_idea = raw.refined_idea;
        }
        for (const k of ["fixed_traits", "forbidden_axes", "scene_bans"]) {
            if (Array.isArray(raw[k])) {
                norm[k] = raw[k]
                    .map(s => String(s ?? "").trim())
                    .filter(Boolean);
            }
        }
        return norm;
    } catch {
        return emptyBrief();
    }
}

function briefIsEmpty(brief) {
    return !brief.refined_idea
        && !brief.fixed_traits.length
        && !brief.forbidden_axes.length
        && !brief.scene_bans.length;
}

function writeBriefJSON(widget, brief) {
    // When empty, persist "{}" so the Manager treats it as "regenerate".
    if (briefIsEmpty(brief)) {
        widget.value = "{}";
    } else {
        widget.value = JSON.stringify(brief, null, 2);
    }
}

// Snake_case axis names — keep parity with the Python `_to_snake_case`.
function toSnakeCase(s) {
    return String(s ?? "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, "_")
        .replace(/^_+|_+$/g, "");
}

// Hide a built-in widget without removing it (so its value still serializes).
function hideWidget(node, widget) {
    widget.computeSize = () => [0, -4];
    widget.type = "hidden_" + widget.type;
    // The legacy type-rename trick isn't honored by ComfyUI's newer frontend
    // for native toggles (e.g. the BOOLEAN lock_brief checkbox), which leaves
    // them rendered. `hidden` is the flag the modern frontend respects; set
    // both so the widget stays out of the layout on old and new UIs alike.
    widget.hidden = true;
    if (widget.element) widget.element.style.display = "none";
}

// Shrink ComfyUI's multiline STRING widget. Default sizing is ~120px which
// stacks 3 of them into ~360px of header before the DOM widget gets any
// space. Cap at `height` and resize the textarea inside to match.
function shrinkMultilineWidget(node, name, height = 56) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    w.computeSize = function (width) { return [width, height]; };
    // ComfyUI's multiline widget exposes the textarea as `inputEl`.
    const ta = w.inputEl || w.element;
    if (ta) {
        ta.style.height = `${height - 8}px`;
        ta.style.minHeight = `${height - 8}px`;
        ta.style.resize = "none";
    }
    // Some ComfyUI versions also key off `computedHeight`.
    w.computedHeight = height;
}

// Pin a DOM widget's rendered height to a stable constant. Default behavior
// (`element.scrollHeight`) grows with content and pushes the widget past the
// visible node frame. Anchoring to a constant breaks the feedback loop.
function pinWidgetHeight(domWidget, height) {
    if (!domWidget) return;
    domWidget.computeSize = function (width) { return [width, height]; };
    domWidget.computedHeight = height;
}

// Make a DOM widget grow with the node frame: the widget claims whatever
// vertical space is left after the title bar and any sibling widgets, with a
// floor of `minHeight` so the contents stay usable on a tiny node. Hooks
// `onResize` so dragging the node corner reflows the inner flex children
// (slots list, raw textarea) instead of being clipped.
//
// Sizing must converge — if our `computeSize` reports a height + the chrome
// LiteGraph/ComfyUI insert exceeds `node.size[1]`, ComfyUI grows the node,
// `onResize` fires, we read the larger size, our height grows by the gap, and
// the node inflates a few pixels every frame forever. Estimating the chrome
// with a constant (TITLE + VPADDING + sum of siblings) doesn't match what
// `node.computeSize()` actually returns, which is what reopened the loop.
//
// Instead, probe `node.computeSize()` with our widget pinned to `minHeight` to
// learn the real overhead the node adds around us, then pick our height so
// `our_height + overhead === node.size[1]`. ComfyUI sees no reason to grow,
// onResize doesn't re-fire, and the loop stays broken.
function fillWidgetToNode(node, domWidget, minHeight = 280) {
    domWidget.computedHeight = minHeight;
    domWidget.computeSize = function (width) {
        return [width, domWidget.computedHeight ?? minHeight];
    };
    function recompute() {
        const saved = domWidget.computedHeight;
        domWidget.computedHeight = minHeight;
        const probed = (typeof node.computeSize === "function"
            ? node.computeSize()[1] : minHeight);
        const overhead = Math.max(0, probed - minHeight);
        const target = Math.max(minHeight, node.size[1] - overhead);
        domWidget.computedHeight = target;
        return target !== saved;
    }
    recompute();
    const onResize = node.onResize;
    node.onResize = function (size) {
        onResize?.apply(this, arguments);
        if (recompute()) node.setDirtyCanvas(true, true);
    };
}

// Size a DOM widget + its node to wrap the rendered content. RAF coalesces
// bursts of edits into a single layout pass; the widget reports exactly
// `scrollHeight` (plus a small fudge for borders), so the node grows or
// shrinks to fit and never runs away. Inner panels are responsible for their
// own scrolling — this helper assumes the root has `height:auto` and that
// any list/textarea inside has its own max-height. Returns a function the
// caller invokes after any change that affects rendered height.
function fitWidgetToContent(node, domWidget, root, minWidth = 0) {
    let pending = false;
    return function update() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            const h = Math.ceil(root.scrollHeight) + 8;
            const prev = domWidget.computedHeight || 0;
            if (Math.abs(prev - h) < 2) return;
            domWidget.computeSize = (width) => [width, h];
            domWidget.computedHeight = h;
            const min = node.computeSize();
            const w = Math.max(node.size[0], min[0], minWidth);
            node.setSize([w, min[1]]);
            node.setDirtyCanvas(true, true);
        });
    };
}

// Build a single-line editable div that mimics an <input type="text"> but
// isn't a form field — password managers (Dashlane, 1Password, LastPass,
// Bitwarden) don't autofill into contenteditable divs. The data-* attributes
// are extra belt-and-suspenders for managers that scan beyond <input>.
function makeEditable(placeholder, value) {
    const el = document.createElement("div");
    el.setAttribute("contenteditable", "plaintext-only");
    // Firefox doesn't support plaintext-only — fall back to plain editable.
    if (el.contentEditable !== "plaintext-only") {
        el.setAttribute("contenteditable", "true");
    }
    el.dataset.placeholder = placeholder;
    el.spellcheck = false;
    el.setAttribute("data-form-type", "other");
    el.setAttribute("data-1p-ignore", "true");
    el.setAttribute("data-lpignore", "true");
    el.setAttribute("data-bwignore", "true");
    if (value) el.textContent = value;
    el.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); el.blur(); }
    });
    // For the contenteditable=true fallback, scrub HTML on paste.
    el.addEventListener("paste", (e) => {
        if (el.getAttribute("contenteditable") === "plaintext-only") return;
        e.preventDefault();
        const text = (e.clipboardData || window.clipboardData).getData("text/plain");
        const sel = window.getSelection();
        if (sel?.rangeCount) {
            const range = sel.getRangeAt(0);
            range.deleteContents();
            range.insertNode(document.createTextNode(text));
            range.collapse(false);
            sel.removeAllRanges();
            sel.addRange(range);
        } else {
            el.textContent = (el.textContent || "") + text;
        }
    });
    return el;
}

// One-time CSS for the Manager + Report. Scoped to `.lwm-*` so it can't bleed
// into the rest of ComfyUI.
function injectStyles() {
    if (document.getElementById("lwm-styles")) return;
    const css = `
        /* Root fills the widget container exactly and never overflows; the
           scroll area is delegated to .lwm-scroll inside. The classic flex
           pattern: flex column with overflow:hidden, plus a child with
           flex:1 1 auto + min-height:0 + overflow:auto. */
        .lwm-root { display:flex; flex-direction:column; gap:8px;
            padding:6px; box-sizing:border-box; width:100%; height:100%;
            max-width:100%; min-width:0; min-height:0;
            overflow:hidden;
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                Roboto, sans-serif; color:#dcdcdc;
        }
        /* Manager variant: height tracks content so the node grows with the
           list instead of clipping it. No internal scroll on the categories. */
        .lwm-root.lwm-root-fit { height:auto; overflow:visible; }
        .lwm-root * { box-sizing:border-box; }
        .lwm-fixed { flex:0 0 auto; min-height:0; }
        .lwm-scroll { flex:1 1 auto; min-height:0; min-width:0;
            overflow-y:auto; overflow-x:hidden; padding-right:2px; }
        .lwm-scroll.lwm-cap-cats   { max-height:260px; }
        .lwm-scroll.lwm-cap-slots  { max-height:240px; }
        /* Report uses a flexible split: slots and raw textarea share the
           leftover vertical space and grow with the node frame. */
        .lwm-scroll.lwm-grow       { flex:1 1 0; min-height:120px; }
        .lwm-scroll::-webkit-scrollbar { width:8px; }
        .lwm-scroll::-webkit-scrollbar-thumb {
            background:#2c3138; border-radius:4px; }
        .lwm-scroll::-webkit-scrollbar-thumb:hover { background:#3a4250; }
        .lwm-section-label { font-size:10px; letter-spacing:.06em;
            text-transform:uppercase; color:#7d8693; margin:2px 2px -2px; }
        .lwm-toolbar { display:flex; gap:6px; align-items:center;
            min-width:0; max-width:100%; flex-wrap:wrap; }
        .lwm-toolbar > * { min-width:0; }
        .lwm-toolbar .lwm-spacer { flex:1 1 auto; }
        .lwm-input, .lwm-textarea {
            background:#16181b; color:#e6e6e6;
            border:1px solid #2e3338; border-radius:4px;
            padding:6px 8px; font-size:12px; outline:none;
            box-sizing:border-box; min-width:0; max-width:100%;
            transition:border-color .12s, box-shadow .12s;
        }
        .lwm-input:focus, .lwm-textarea:focus {
            border-color:#4d8cd0; box-shadow:0 0 0 2px rgba(77,140,208,.18);
        }
        /* Contenteditable inputs (used instead of <input> in the manager so
           password managers like Dashlane don't try to autofill them). */
        .lwm-input[contenteditable] {
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
            cursor:text; line-height:1.3;
        }
        .lwm-input[contenteditable]:empty::before {
            content: attr(data-placeholder);
            color:#5a606a; pointer-events:none;
        }
        .lwm-textarea { width:100%; resize:none;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size:11px; line-height:1.4; white-space:pre; }
        /* Report's raw view wraps long lines and keeps a fixed height so the
           node body wraps the content instead of stretching with it. */
        .lwm-textarea.lwm-raw-textarea {
            height:200px; min-height:120px;
            white-space:pre-wrap; word-break:break-word;
            overflow:auto;
        }
        .lwm-btn { padding:6px 10px; font-size:12px;
            background:#2c5b86; color:#fff; border:none; border-radius:4px;
            cursor:pointer; transition:background .12s, transform .04s; }
        .lwm-btn:hover { background:#3b75a8; }
        .lwm-btn:active { transform:translateY(1px); }
        .lwm-btn-ghost { background:#2a2e34; color:#cfd3d8; }
        .lwm-btn-ghost:hover { background:#3a3f47; }
        .lwm-btn-danger { background:#5a2a2a; }
        .lwm-btn-danger:hover { background:#7a3a3a; }
        .lwm-btn-icon { width:26px; padding:5px 0; text-align:center; }
        .lwm-pathline { font-size:10px; color:#6c7480;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            min-width:0; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; }
        .lwm-prompt-panel {
            background:#0f1114; color:#e6e6e6;
            border:1px solid #262a31; border-radius:4px;
            padding:6px 8px; font-size:11.5px; line-height:1.45;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            white-space:pre-wrap; word-break:break-word;
            min-width:0; max-width:100%; max-height:140px; overflow:auto;
        }
        .lwm-prompt-panel.lwm-empty { color:#6c7480; font-style:italic; }
        .lwm-prompt-panel.lwm-error { border-color:#5b2c2c; color:#ff9b9b; }
        .lwm-prompt-panel .lwm-tok { color:#7ec9ff; background:#142235;
            padding:1px 3px; border-radius:3px; }
        .lwm-status-banner {
            padding:6px 8px; border-radius:4px; font-size:11.5px;
            line-height:1.4; border:1px solid #2a2e34; background:#1f2228;
            color:#cfd3d8;
        }
        .lwm-status-banner.lwm-status-parse_failed,
        .lwm-status-banner.lwm-status-no_prompt,
        .lwm-status-banner.lwm-status-llm_error,
        .lwm-status-banner.lwm-status-no_locked_template {
            border-color:#5b2c2c; background:#281616; color:#ff9b9b;
        }
        .lwm-status-banner.lwm-status-no_wildcards,
        .lwm-status-banner.lwm-status-salvaged,
        .lwm-status-banner.lwm-status-fallback_default {
            border-color:#5b4a2c; background:#2a2316; color:#f5d782;
        }
        .lwm-status-banner.lwm-status-locked {
            border-color:#2c4467; background:#1a2434; color:#9ec5ff;
        }
        .lwm-raw-header {
            cursor:pointer; user-select:none; display:flex; gap:6px;
            align-items:center;
        }
        .lwm-raw-header .lwm-raw-toggle {
            display:inline-block; transition:transform .12s;
            width:10px; text-align:center;
        }
        .lwm-raw-header.lwm-open .lwm-raw-toggle { transform:rotate(90deg); }
        .lwm-raw-reply {
            background:#0f1114; color:#cfd3d8;
            border:1px solid #262a31; border-radius:4px;
            padding:6px 8px; font-size:11px; line-height:1.45;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            white-space:pre-wrap; word-break:break-word;
            max-height:240px; overflow:auto; margin:0;
        }
        .lwm-raw-reply.lwm-error-border { border-color:#5b2c2c; }
        .lwm-raw-reply.lwm-empty { color:#6c7480; font-style:italic; }
        .lwm-list { display:flex; flex-direction:column; gap:6px;
            min-width:0; max-width:100%; }
        .lwm-row {
            display:flex; flex-direction:column;
            background: linear-gradient(180deg,#1d2025,#181a1e);
            border:1px solid #2a2e34; border-radius:5px;
            padding:6px 6px 4px;
            min-width:0; max-width:100%;
            transition:border-color .12s, background .12s;
        }
        .lwm-row:hover { border-color:#3a4250; }
        .lwm-row.lwm-row-user { border-color:#3a5a82; }
        .lwm-row-head { display:flex; gap:6px; align-items:center;
            min-width:0; max-width:100%; }
        .lwm-row-head > * { min-width:0; }
        .lwm-expand { width:24px; height:24px; flex:0 0 24px;
            background:#22262b; color:#cfd3d8;
            border:1px solid #2e3338; border-radius:4px;
            cursor:pointer; font-size:11px; line-height:22px; padding:0;
            transition:background .12s, transform .12s;
        }
        .lwm-expand:hover { background:#2c3138; }
        .lwm-expand.lwm-open { transform:rotate(90deg); }
        .lwm-name { flex:0 1 27%; min-width:0; }
        .lwm-desc { flex:1 1 auto; min-width:0; }
        .lwm-badge {
            flex:0 0 38px; text-align:center; font-size:11px;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            border-radius:10px; padding:2px 0;
            background:#1f2228; color:#6c7480; border:1px solid #2a2e34;
        }
        .lwm-badge.lwm-badge-low    { color:#9ec5ff; border-color:#2c4467; background:#1a2434; }
        .lwm-badge.lwm-badge-mid    { color:#9be8a4; border-color:#2c5b3a; background:#15281d; }
        .lwm-badge.lwm-badge-high   { color:#f5d782; border-color:#5b4a2c; background:#2a2316; }
        .lwm-tag { font-size:9px; letter-spacing:.06em;
            color:#9ec5ff; background:#152030; border:1px solid #2c4467;
            border-radius:3px; padding:1px 4px; flex:0 0 auto; }
        .lwm-entries {
            overflow:hidden; max-height:0;
            max-width:100%; min-width:0;
            transition:max-height .18s ease, margin-top .18s ease, padding .18s ease;
            background:#0f1114; color:#cfd3d8;
            border:1px solid transparent; border-radius:4px;
            margin-top:0; padding:0 8px;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size:11px; line-height:1.45; white-space:pre;
        }
        .lwm-entries.lwm-open {
            max-height:220px; margin-top:6px; padding:6px 8px;
            border-color:#262a31; overflow:auto;
        }
        .lwm-entries.lwm-empty { color:#6c7480; font-style:italic; }
        /* Report-specific */
        .lwm-report-header { display:flex; flex-wrap:wrap; gap:6px;
            font-size:11px; color:#cfd3d8;
            background:#1a1c20; border:1px solid #262a31; border-radius:4px;
            padding:6px 8px; }
        .lwm-stat { padding:1px 6px; border-radius:10px;
            background:#1f2228; border:1px solid #2a2e34;
            font-family: ui-monospace, Menlo, Consolas, monospace; font-size:10.5px; }
        .lwm-stat.gen  { color:#9be8a4; border-color:#2c5b3a; background:#15281d; }
        .lwm-stat.reu  { color:#9ec5ff; border-color:#2c4467; background:#1a2434; }
        .lwm-stat.err  { color:#ff9b9b; border-color:#5b2c2c; background:#281616; }
        .lwm-slot { background: linear-gradient(180deg,#1d2025,#181a1e);
            border:1px solid #2a2e34; border-radius:5px;
            padding:6px 6px 4px; min-width:0; }
        .lwm-slot-head { display:flex; gap:8px; align-items:center;
            min-width:0; }
        .lwm-slot-head > .lwm-slot-name { font-weight:600; color:#e6e6e6;
            flex:0 0 auto; }
        .lwm-slot-head > .lwm-slot-value {
            flex:1 1 auto; min-width:0; color:#cfd3d8;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size:11px; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; }
        .lwm-status {
            flex:0 0 auto; padding:1px 6px; font-size:10.5px;
            border-radius:10px; border:1px solid #2a2e34; background:#1f2228;
            color:#cfd3d8;
            font-family: ui-monospace, Menlo, Consolas, monospace; }
        .lwm-status.s-generated_new        { color:#9be8a4; border-color:#2c5b3a; background:#15281d; }
        .lwm-status.s-generated_duplicate  { color:#cfd3d8; border-color:#3a4250; }
        .lwm-status.s-reused               { color:#9ec5ff; border-color:#2c4467; background:#1a2434; }
        .lwm-status.s-cap_reached          { color:#f5d782; border-color:#5b4a2c; background:#2a2316; }
        .lwm-status.s-error                { color:#ff9b9b; border-color:#5b2c2c; background:#281616; }
        .lwm-slot-detail {
            overflow:hidden; max-height:0;
            transition:max-height .18s ease, margin-top .18s ease, padding .18s ease;
            background:#0f1114; color:#cfd3d8;
            border:1px solid transparent; border-radius:4px;
            margin-top:0; padding:0 8px;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size:11px; line-height:1.45; white-space:pre-wrap;
            word-break:break-word;
        }
        .lwm-slot-detail.lwm-open {
            max-height:280px; margin-top:6px; padding:6px 8px;
            border-color:#262a31; overflow:auto;
        }
        .lwm-detail-row { margin-bottom:2px; }
        .lwm-detail-key { color:#7d8693; }
        .lwm-flex-fill { flex:1 1 auto; min-height:0;
            display:flex; flex-direction:column; min-width:0; }
        .lwm-flex-fill > textarea {
            flex:1 1 auto; min-height:0; height:100%; width:100%; }
        /* Design brief panel */
        .lwm-brief-panel {
            background:#13161a; border:1px solid #262a31; border-radius:5px;
            padding:6px 8px; display:flex; flex-direction:column; gap:6px;
            min-width:0; max-width:100%;
        }
        .lwm-brief-row { display:flex; gap:6px; align-items:flex-start;
            min-width:0; max-width:100%; }
        .lwm-brief-label {
            flex:0 0 96px; font-size:10px; letter-spacing:.06em;
            text-transform:uppercase; color:#7d8693; padding-top:6px;
        }
        .lwm-brief-field { flex:1 1 auto; min-width:0;
            display:flex; flex-direction:column; gap:4px; }
        .lwm-brief-idea {
            background:#0f1114; color:#e6e6e6;
            border:1px solid #2e3338; border-radius:4px;
            padding:6px 8px; font-size:12px; line-height:1.4;
            min-height:30px; outline:none;
            font-family: ui-sans-serif, system-ui, sans-serif;
            white-space:pre-wrap; word-break:break-word;
        }
        .lwm-brief-idea[contenteditable]:focus {
            border-color:#4d8cd0; box-shadow:0 0 0 2px rgba(77,140,208,.18);
        }
        .lwm-brief-idea[contenteditable]:empty::before {
            content: attr(data-placeholder);
            color:#5a606a; pointer-events:none;
        }
        .lwm-chip-row { display:flex; flex-wrap:wrap; gap:4px;
            align-items:center; min-width:0; }
        .lwm-chip {
            display:inline-flex; align-items:center; gap:4px;
            padding:2px 4px 2px 8px; font-size:11px;
            background:#1a2434; color:#9ec5ff; border:1px solid #2c4467;
            border-radius:10px; max-width:100%; min-width:0;
        }
        .lwm-chip.lwm-chip-axis  { background:#152030; color:#9ec5ff; border-color:#2c4467; }
        .lwm-chip.lwm-chip-scene { background:#281616; color:#ff9b9b; border-color:#5b2c2c; }
        .lwm-chip.lwm-chip-fixed { background:#15281d; color:#9be8a4; border-color:#2c5b3a; }
        .lwm-chip-text {
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
            max-width:280px;
        }
        .lwm-chip-text[contenteditable] { cursor:text; min-width:14px;
            white-space:nowrap; overflow:hidden; }
        .lwm-chip-x {
            width:16px; height:16px; line-height:14px;
            text-align:center; cursor:pointer; border-radius:50%;
            background:transparent; color:inherit; border:none;
            font-size:12px; opacity:.7; padding:0;
        }
        .lwm-chip-x:hover { opacity:1; background:rgba(255,255,255,.08); }
        .lwm-chip-add {
            background:transparent; color:#7d8693;
            border:1px dashed #3a4250; border-radius:10px;
            padding:2px 8px; font-size:11px; cursor:text;
            min-width:80px; outline:none;
        }
        .lwm-chip-add:focus { color:#e6e6e6; border-color:#4d8cd0; }
        .lwm-chip-add[contenteditable]:empty::before {
            content: attr(data-placeholder);
            color:#5a606a; pointer-events:none;
        }
        .lwm-brief-empty { color:#6c7480; font-size:11px; font-style:italic; }
        /* Template Builder */
        .lwm-blk.lwm-blk-off { opacity:.5; }
        .lwm-blk-kind { flex:0 0 auto; font-size:9px; letter-spacing:.06em;
            text-transform:uppercase; border-radius:3px; padding:2px 6px;
            white-space:nowrap; }
        .lwm-blk-kind.lwm-blk-sentence { color:#9be8a4; background:#15281d;
            border:1px solid #2c5b3a; }
        .lwm-blk-kind.lwm-blk-wild { color:#7ec9ff; background:#142235;
            border:1px solid #2c4467; }
        .lwm-select { background:#16181b; color:#e6e6e6;
            border:1px solid #2e3338; border-radius:4px; padding:3px 6px;
            font-size:11px; outline:none; max-width:140px; min-width:0; }
        .lwm-select:focus { border-color:#4d8cd0; }
        .lwm-preset-select { flex:0 0 auto; max-width:160px; }
        .lwm-range { flex:0 1 110px; min-width:60px; accent-color:#4d8cd0;
            cursor:pointer; }
        .lwm-count { flex:0 0 auto; font-size:11px; color:#9ec5ff;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            min-width:26px; text-align:center; }
        .lwm-fnlabel { display:inline-flex; align-items:center; gap:3px;
            font-size:10px; color:#7d8693; cursor:pointer; white-space:nowrap; }
        .lwm-chk { cursor:pointer; accent-color:#4d8cd0; margin:0; }
        /* Two-class selector so it out-specifies .lwm-input[contenteditable]'s
           nowrap/ellipsis rule and the sentence editor wraps multi-clause text. */
        .lwm-input.lwm-blk-text { white-space:pre-wrap; overflow:visible;
            text-overflow:clip; height:auto; min-height:30px; margin-top:6px;
            line-height:1.4; }
        .lwm-skeleton { font-size:11px; color:#9ec5ff; line-height:1.5;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            background:#0f1114; border:1px solid #262a31; border-radius:4px;
            padding:6px 8px; word-break:break-word; }
        .lwm-entries::-webkit-scrollbar, .lwm-textarea::-webkit-scrollbar,
        .lwm-prompt-panel::-webkit-scrollbar, .lwm-slot-detail::-webkit-scrollbar {
            width:8px; height:8px; }
        .lwm-entries::-webkit-scrollbar-thumb, .lwm-textarea::-webkit-scrollbar-thumb,
        .lwm-prompt-panel::-webkit-scrollbar-thumb,
        .lwm-slot-detail::-webkit-scrollbar-thumb {
            background:#2c3138; border-radius:4px; }
        .lwm-entries::-webkit-scrollbar-thumb:hover,
        .lwm-textarea::-webkit-scrollbar-thumb:hover,
        .lwm-prompt-panel::-webkit-scrollbar-thumb:hover,
        .lwm-slot-detail::-webkit-scrollbar-thumb:hover {
            background:#3a4250; }
    `;
    const style = document.createElement("style");
    style.id = "lwm-styles";
    style.textContent = css;
    document.head.appendChild(style);
}

function escapeHTML(s) {
    return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// Render a prompt template, highlighting __wildcard__ tokens.
function renderTemplateHTML(template) {
    const t = String(template || "");
    if (!t.trim()) return "";
    return escapeHTML(t).replace(
        /__(!?[A-Za-z0-9_-]+)__/g,
        (_, name) => `<span class="lwm-tok">__${escapeHTML(name)}__</span>`
    );
}

// ---------------------------------------------------------------------------
// Template Builder helpers — keep parity with the Python normaliser.
// A block is one of:
//   { kind:"sentence",  enabled, role, text }
//   { kind:"wildcards", enabled, role, count, force_new }
// `role` is an abstract structural label (or "" = undefined). The hidden
// `structure` STRING widget stores { blocks: [...] } and is the source of
// truth for headless / API runs.
// ---------------------------------------------------------------------------
const BUILDER_WILDCARD_ROLES = [
    "subject", "character", "appearance", "age", "outfit", "accessory",
    "pose", "expression", "action", "activity", "setting", "location",
    "background", "time", "weather", "lighting", "mood", "color",
    "material", "texture", "style", "camera", "composition", "detail",
];
const BUILDER_SENTENCE_ROLES = [
    "scene", "subject", "action", "setting", "atmosphere", "style", "closing",
];
const BUILDER_MAX_COUNT = 12;

function defaultStructureBlocks() {
    return [
        { kind: "sentence",  enabled: true, role: "scene",     text: "" },
        { kind: "wildcards", enabled: true, role: "character", count: 3, force_new: false },
        { kind: "sentence",  enabled: true, role: "action",    text: "" },
        { kind: "wildcards", enabled: true, role: "pose",      count: 2, force_new: false },
    ];
}

function normalizeBlocks(raw) {
    let arr = raw;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) arr = raw.blocks;
    if (!Array.isArray(arr)) return [];
    const out = [];
    for (const e of arr) {
        if (!e || typeof e !== "object") continue;
        const kind = String(e.kind || "").toLowerCase();
        const enabled = e.enabled !== false;
        const role = String(e.role ?? "").trim();
        if (kind === "sentence") {
            out.push({ kind: "sentence", enabled, role, text: String(e.text ?? "") });
        } else if (kind === "wildcards") {
            let count = parseInt(e.count, 10);
            if (!Number.isFinite(count)) count = 1;
            count = Math.max(1, Math.min(count, BUILDER_MAX_COUNT));
            out.push({ kind: "wildcards", enabled, role, count,
                       force_new: !!e.force_new });
        }
    }
    return out;
}

function readStructureBlocks(widget) {
    try {
        const blocks = normalizeBlocks(JSON.parse(widget.value || "{}"));
        return blocks.length ? blocks : defaultStructureBlocks();
    } catch {
        return defaultStructureBlocks();
    }
}

function writeStructureBlocks(widget, blocks) {
    widget.value = JSON.stringify({ blocks }, null, 2);
}

// Compact "shape" preview, e.g. "Sentence(scene)  ·  __character__ ×3".
function structureSkeleton(blocks) {
    const segs = [];
    for (const b of blocks) {
        if (!b.enabled) continue;
        if (b.kind === "sentence") {
            segs.push(`Sentence(${b.role || "any"})`);
        } else {
            const base = toSnakeCase(b.role) || "slot";
            segs.push(`__${base}__ ×${b.count}`);
        }
    }
    return segs.join("  ·  ") || "(empty — add a block)";
}

// ---------------------------------------------------------------------------
// Starter presets — ready-made structures the user can load into the editor.
// Each preset is a pure *shape*: empty-text sentence blocks (the AI writes the
// prose) plus role-tagged wildcard groups. The content is intentionally
// generic, so any idea can be poured into the same composition. Loading a
// preset REPLACES the current blocks. Roles are drawn from
// BUILDER_SENTENCE_ROLES / BUILDER_WILDCARD_ROLES.
// ---------------------------------------------------------------------------
function _ps(role) {                          // sentence block (AI-written)
    return { kind: "sentence", enabled: true, role: role || "", text: "" };
}
function _pw(role, count, forceNew) {         // wildcard group
    return { kind: "wildcards", enabled: true, role: role || "",
             count: count || 1, force_new: !!forceNew };
}

const BUILDER_PRESETS = [
    { group: "Portraits & Characters", items: [
        { name: "Simple Character Portrait",
          blocks: [ _ps("subject"), _pw("character", 3), _pw("appearance", 2), _ps("style") ] },
        { name: "Detailed Character Sheet",
          blocks: [ _ps("subject"), _pw("character", 2), _pw("appearance", 3),
                    _pw("outfit", 2), _pw("accessory", 2), _ps("style") ] },
        { name: "Fashion / Outfit Showcase",
          blocks: [ _ps("subject"), _pw("outfit", 4), _pw("accessory", 3),
                    _pw("pose", 1), _ps("style") ] },
        { name: "Close-up Headshot",
          blocks: [ _ps("subject"), _pw("appearance", 3), _pw("expression", 2),
                    _pw("lighting", 2), _ps("closing") ] },
        { name: "Full-Body Character",
          blocks: [ _ps("scene"), _pw("character", 2), _pw("outfit", 3),
                    _pw("pose", 2), _pw("setting", 1), _ps("closing") ] },
        { name: "Character Duo",
          blocks: [ _ps("scene"), _pw("character", 2), _pw("pose", 2),
                    _pw("expression", 2), _ps("action"), _ps("style") ] },
        { name: "Group Scene",
          blocks: [ _ps("scene"), _pw("character", 3), _pw("activity", 2),
                    _pw("setting", 2), _ps("atmosphere") ] },
        { name: "Emotion Study",
          blocks: [ _ps("subject"), _pw("expression", 3), _pw("mood", 2),
                    _pw("lighting", 2), _ps("closing") ] },
        { name: "Action Pose",
          blocks: [ _ps("action"), _pw("character", 2), _pw("pose", 3),
                    _pw("action", 2), _ps("atmosphere"), _ps("style") ] },
        { name: "Costume Focus",
          blocks: [ _ps("subject"), _pw("outfit", 4), _pw("material", 2),
                    _pw("detail", 2), _ps("style") ] },
    ]},
    { group: "Scenes & Environments", items: [
        { name: "Landscape Vista",
          blocks: [ _ps("scene"), _pw("location", 2), _pw("weather", 2),
                    _pw("lighting", 2), _ps("atmosphere"), _ps("style") ] },
        { name: "Cityscape / Urban",
          blocks: [ _ps("scene"), _pw("location", 2), _pw("time", 1),
                    _pw("lighting", 2), _pw("detail", 2), _ps("atmosphere") ] },
        { name: "Interior Room",
          blocks: [ _ps("setting"), _pw("location", 1), _pw("material", 2),
                    _pw("lighting", 2), _pw("detail", 3), _ps("atmosphere") ] },
        { name: "Nature Wilderness",
          blocks: [ _ps("scene"), _pw("location", 2), _pw("weather", 1),
                    _pw("time", 1), _pw("detail", 3), _ps("style") ] },
        { name: "Fantasy Environment",
          blocks: [ _ps("scene"), _pw("setting", 2), _pw("detail", 3),
                    _pw("lighting", 2), _pw("color", 2), _ps("atmosphere") ] },
        { name: "Sci-Fi Setting",
          blocks: [ _ps("scene"), _pw("setting", 2), _pw("material", 2),
                    _pw("lighting", 2), _pw("detail", 2), _ps("style") ] },
        { name: "Seasonal Scene",
          blocks: [ _ps("scene"), _pw("location", 1), _pw("weather", 2),
                    _pw("time", 1), _pw("color", 2), _ps("atmosphere") ] },
        { name: "Weather & Mood",
          blocks: [ _ps("scene"), _pw("weather", 3), _pw("lighting", 2),
                    _pw("mood", 2), _ps("atmosphere") ] },
        { name: "Architectural Study",
          blocks: [ _ps("subject"), _pw("setting", 2), _pw("material", 3),
                    _pw("composition", 2), _pw("lighting", 1), _ps("style") ] },
        { name: "Establishing Shot",
          blocks: [ _ps("scene"), _pw("location", 2), _pw("composition", 2),
                    _pw("lighting", 2), _pw("time", 1), _ps("closing") ] },
    ]},
    { group: "Cinematic & Photography", items: [
        { name: "Cinematic Wide Shot",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("setting", 2),
                    _pw("lighting", 2), _pw("camera", 2), _pw("mood", 1), _ps("style") ] },
        { name: "Film Still",
          blocks: [ _ps("scene"), _pw("character", 1), _pw("action", 1),
                    _pw("lighting", 2), _pw("camera", 2), _pw("color", 1), _ps("atmosphere") ] },
        { name: "Golden Hour Portrait",
          blocks: [ _ps("subject"), _pw("character", 2), _pw("lighting", 3),
                    _pw("time", 1), _pw("mood", 1), _ps("style") ] },
        { name: "Studio Product Shot",
          blocks: [ _ps("subject"), _pw("material", 2), _pw("lighting", 3),
                    _pw("composition", 2), _pw("detail", 2), _ps("closing") ] },
        { name: "Street Photography",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("location", 1),
                    _pw("action", 1), _pw("time", 1), _pw("camera", 1), _ps("atmosphere") ] },
        { name: "Macro Detail",
          blocks: [ _ps("subject"), _pw("texture", 3), _pw("detail", 3),
                    _pw("lighting", 2), _ps("style") ] },
        { name: "Dramatic Lighting",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("lighting", 4),
                    _pw("mood", 2), _pw("color", 1), _ps("closing") ] },
        { name: "Documentary Candid",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("action", 1),
                    _pw("setting", 1), _pw("camera", 1), _ps("atmosphere") ] },
        { name: "Editorial Fashion",
          blocks: [ _ps("subject"), _pw("character", 1), _pw("outfit", 3),
                    _pw("pose", 2), _pw("lighting", 2), _pw("camera", 1), _ps("style") ] },
        { name: "Noir / Moody",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("lighting", 2),
                    _pw("mood", 2), _pw("color", 2), _pw("camera", 1), _ps("atmosphere") ] },
    ]},
    { group: "Stylized & Artistic", items: [
        { name: "Anime / Manga Style",
          blocks: [ _ps("subject"), _pw("character", 2), _pw("expression", 2),
                    _pw("outfit", 2), _pw("style", 2), _ps("closing") ] },
        { name: "Oil Painting",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("color", 2),
                    _pw("texture", 2), _pw("style", 3), _ps("atmosphere") ] },
        { name: "Watercolor Soft",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("color", 3),
                    _pw("texture", 2), _pw("style", 2), _ps("atmosphere") ] },
        { name: "Concept Art Splash",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("setting", 2),
                    _pw("color", 2), _pw("detail", 2), _pw("style", 2), _ps("closing") ] },
        { name: "Flat Vector / Minimal",
          blocks: [ _ps("subject"), _pw("composition", 2), _pw("color", 3),
                    _pw("style", 2), _ps("closing") ] },
        { name: "Surreal Dreamscape",
          blocks: [ _ps("scene"), _pw("subject", 2), _pw("setting", 2),
                    _pw("color", 2), _pw("mood", 2), _pw("style", 1), _ps("atmosphere") ] },
        { name: "Cyberpunk Neon",
          blocks: [ _ps("scene"), _pw("setting", 2), _pw("lighting", 2),
                    _pw("color", 3), _pw("detail", 2), _pw("style", 1), _ps("atmosphere") ] },
        { name: "Steampunk",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("material", 3),
                    _pw("detail", 3), _pw("color", 1), _pw("style", 1), _ps("closing") ] },
        { name: "Vintage / Retro",
          blocks: [ _ps("scene"), _pw("subject", 1), _pw("color", 2),
                    _pw("texture", 2), _pw("style", 2), _ps("atmosphere") ] },
        { name: "Pop Art Bold",
          blocks: [ _ps("subject"), _pw("color", 4), _pw("composition", 2),
                    _pw("style", 2), _ps("closing") ] },
    ]},
    { group: "Subjects & Themes", items: [
        { name: "Animal / Creature",
          blocks: [ _ps("subject"), _pw("subject", 2), _pw("appearance", 2),
                    _pw("setting", 1), _pw("detail", 2), _ps("style") ] },
        { name: "Food / Culinary",
          blocks: [ _ps("subject"), _pw("subject", 2), _pw("material", 2),
                    _pw("color", 2), _pw("lighting", 2), _pw("composition", 1), _ps("style") ] },
        { name: "Vehicle / Machine",
          blocks: [ _ps("subject"), _pw("subject", 2), _pw("material", 2),
                    _pw("detail", 3), _pw("setting", 1), _pw("lighting", 1), _ps("style") ] },
        { name: "Still Life Arrangement",
          blocks: [ _ps("scene"), _pw("subject", 3), _pw("material", 2),
                    _pw("lighting", 2), _pw("composition", 2), _ps("style") ] },
        { name: "Abstract Composition",
          blocks: [ _ps("scene"), _pw("color", 3), _pw("texture", 2),
                    _pw("composition", 3), _pw("style", 1), _ps("closing") ] },
        { name: "Botanical / Floral",
          blocks: [ _ps("subject"), _pw("subject", 2), _pw("color", 3),
                    _pw("texture", 2), _pw("detail", 2), _ps("style") ] },
        { name: "Mythical Creature",
          blocks: [ _ps("scene"), _pw("subject", 2), _pw("appearance", 3),
                    _pw("setting", 1), _pw("detail", 2), _pw("style", 1), _ps("atmosphere") ] },
        { name: "Mecha / Robot",
          blocks: [ _ps("subject"), _pw("subject", 1), _pw("material", 3),
                    _pw("detail", 3), _pw("pose", 1), _pw("lighting", 1), _ps("style") ] },
        { name: "Underwater Scene",
          blocks: [ _ps("scene"), _pw("setting", 2), _pw("color", 2),
                    _pw("lighting", 2), _pw("detail", 2), _ps("atmosphere") ] },
        { name: "Cosmic / Space",
          blocks: [ _ps("scene"), _pw("setting", 2), _pw("color", 3),
                    _pw("lighting", 2), _pw("detail", 2), _ps("atmosphere"), _ps("style") ] },
    ]},
];

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "comfyui.llm_wildcard_manager",

    async beforeRegisterNodeDef(nodeType, nodeData) {

        // -------------------------------------------------------------------
        // LLMWildcardManager
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardManager") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                injectStyles();
                const node = this;

                const jsonWidget = node.widgets.find(w => w.name === "categories");
                if (!jsonWidget) return;
                hideWidget(node, jsonWidget);

                // Hidden JSON widget mirroring the design brief. The Python
                // side reads it; the brief panel below reads/writes it.
                const briefWidget =
                    node.widgets.find(w => w.name === "design_brief");
                if (briefWidget) hideWidget(node, briefWidget);

                // The brief lock is rendered as a toggle button inside the
                // brief panel header, so hide the bare BOOLEAN widget.
                const lockBriefWidget =
                    node.widgets.find(w => w.name === "lock_brief");
                if (lockBriefWidget) hideWidget(node, lockBriefWidget);

                // Shrink the oversized multiline STRING widgets so the node
                // header doesn't eat all the space before the DOM table.
                shrinkMultilineWidget(node, "example_prompt", 64);
                shrinkMultilineWidget(node, "anchors", 48);
                shrinkMultilineWidget(node, "negative_prompt", 64);
                shrinkMultilineWidget(node, "forbidden_placeholders", 48);
                shrinkMultilineWidget(node, "system_prompt_override", 48);

                const root = document.createElement("div");
                root.className = "lwm-root lwm-root-fit";

                // Forward declaration — set to its real impl after the DOM
                // widget exists. Helpers below call it on changes that affect
                // the rendered height.
                let updateManagerSize = () => {};

                // ---- generated prompt panel (fixed at top) ----
                const promptLabel = document.createElement("div");
                promptLabel.className = "lwm-section-label lwm-fixed";
                promptLabel.textContent = "Generated prompt template";
                const promptPanel = document.createElement("div");
                promptPanel.className = "lwm-prompt-panel lwm-empty lwm-fixed";
                promptPanel.textContent =
                    "(no template yet — queue the workflow to generate)";
                root.appendChild(promptLabel);
                root.appendChild(promptPanel);

                // ---- status banner (shown only when status != ok) ----
                const statusBanner = document.createElement("div");
                statusBanner.className = "lwm-status-banner lwm-fixed";
                statusBanner.style.display = "none";
                root.appendChild(statusBanner);

                // ---- raw LLM reply (collapsible) ----
                const rawHeader = document.createElement("div");
                rawHeader.className = "lwm-section-label lwm-fixed lwm-raw-header";
                rawHeader.style.display = "none";
                rawHeader.innerHTML =
                    `<span class="lwm-raw-toggle">▸</span>` +
                    `<span>Last LLM raw reply</span>`;
                root.appendChild(rawHeader);

                const rawPanel = document.createElement("pre");
                rawPanel.className = "lwm-raw-reply lwm-fixed";
                rawPanel.style.display = "none";
                root.appendChild(rawPanel);

                rawHeader.addEventListener("click", () => {
                    const open = rawHeader.classList.toggle("lwm-open");
                    rawPanel.style.display = open ? "block" : "none";
                    updateManagerSize();
                });

                // ---- design brief panel ----
                // Lets the user review and edit the auto-derived design brief
                // (refined idea + fixed traits + forbidden axes + scene bans)
                // before the next queue. Empty widget = "regenerate from the
                // LLM"; any non-empty content = "use as-is and skip the brief
                // LLM call". Edits flow into the hidden `design_brief` widget.

                const briefLabel = document.createElement("div");
                briefLabel.className = "lwm-section-label lwm-fixed";
                briefLabel.style.display = "flex";
                briefLabel.style.gap = "6px";
                briefLabel.style.alignItems = "center";
                const briefLabelText = document.createElement("span");
                briefLabelText.style.flex = "1 1 auto";
                briefLabelText.textContent =
                    "Design brief — locks the LLM to the user's idea";
                const briefLockBtn = document.createElement("button");
                briefLockBtn.className = "lwm-btn lwm-btn-ghost";
                briefLockBtn.style.fontSize = "10px";
                briefLockBtn.style.padding = "3px 8px";
                briefLockBtn.style.letterSpacing = "0";
                briefLockBtn.style.textTransform = "none";
                function syncBriefLockBtn() {
                    const locked = !!(lockBriefWidget && lockBriefWidget.value);
                    briefLockBtn.textContent =
                        locked ? "🔒 Locked" : "🔓 Unlocked";
                    briefLockBtn.title = locked
                        ? "Brief is locked — the edited brief below is reused " +
                          "every queue. Click to unlock and regenerate from " +
                          "the LLM each queue."
                        : "Brief is unlocked — the LLM regenerates it from " +
                          "your idea every queue (edits below are " +
                          "overwritten). Click to lock.";
                }
                briefLockBtn.addEventListener("click", () => {
                    if (!lockBriefWidget) return;
                    lockBriefWidget.value = !lockBriefWidget.value;
                    syncBriefLockBtn();
                    node.setDirtyCanvas(true, true);
                });
                syncBriefLockBtn();
                const briefRegenBtn = document.createElement("button");
                briefRegenBtn.textContent = "↻ Regenerate brief";
                briefRegenBtn.title =
                    "Clear the edited brief so the next queue regenerates it " +
                    "from your idea via the LLM";
                briefRegenBtn.className = "lwm-btn lwm-btn-ghost";
                briefRegenBtn.style.fontSize = "10px";
                briefRegenBtn.style.padding = "3px 8px";
                briefRegenBtn.style.letterSpacing = "0";
                briefRegenBtn.style.textTransform = "none";
                briefLabel.appendChild(briefLabelText);
                briefLabel.appendChild(briefLockBtn);
                briefLabel.appendChild(briefRegenBtn);
                root.appendChild(briefLabel);

                const briefPanel = document.createElement("div");
                briefPanel.className = "lwm-brief-panel lwm-fixed";
                root.appendChild(briefPanel);

                // Local working copy of the brief; UI events mutate this, then
                // commitBrief() pushes it back into the hidden widget.
                let currentBrief = emptyBrief();
                if (briefWidget) currentBrief = readBriefJSON(briefWidget);

                function commitBrief() {
                    if (briefWidget) writeBriefJSON(briefWidget, currentBrief);
                    node.setDirtyCanvas(true, true);
                }

                // Build a removable chip. `value` is the visible text, `kind`
                // controls colour, `onChange` receives the edited text on blur
                // (empty/duplicate values are pruned by the caller), `onRemove`
                // deletes the chip.
                function buildChip(value, kind, onChange, onRemove) {
                    const chip = document.createElement("span");
                    chip.className = `lwm-chip lwm-chip-${kind}`;
                    const txt = makeEditable("", value);
                    txt.className = "lwm-chip-text";
                    txt.addEventListener("blur", () => {
                        const next = (txt.textContent || "").trim();
                        onChange(next);
                    });
                    const rm = document.createElement("button");
                    rm.className = "lwm-chip-x";
                    rm.textContent = "×";
                    rm.title = "Remove";
                    rm.addEventListener("click", () => onRemove());
                    chip.appendChild(txt);
                    chip.appendChild(rm);
                    return chip;
                }

                // Build the "+" adder element for a chip row. On Enter or blur
                // (with content), pushes the value into `list` via `onAdd` and
                // re-renders the row.
                function buildChipAdd(placeholder, onAdd) {
                    const add = document.createElement("span");
                    add.className = "lwm-chip-add";
                    add.setAttribute("contenteditable",
                        add.contentEditable === "plaintext-only"
                            ? "plaintext-only" : "true");
                    add.dataset.placeholder = placeholder;
                    add.spellcheck = false;
                    add.setAttribute("data-1p-ignore", "true");
                    add.setAttribute("data-lpignore", "true");
                    function commitAdd() {
                        const v = (add.textContent || "").trim();
                        add.textContent = "";
                        if (v) onAdd(v);
                    }
                    add.addEventListener("keydown", (e) => {
                        if (e.key === "Enter" || e.key === ",") {
                            e.preventDefault();
                            commitAdd();
                        }
                    });
                    add.addEventListener("blur", commitAdd);
                    return add;
                }

                function renderBrief() {
                    briefPanel.innerHTML = "";

                    // Refined idea row.
                    const ideaRow = document.createElement("div");
                    ideaRow.className = "lwm-brief-row";
                    const ideaLabel = document.createElement("div");
                    ideaLabel.className = "lwm-brief-label";
                    ideaLabel.textContent = "Refined idea";
                    const ideaField = document.createElement("div");
                    ideaField.className = "lwm-brief-field";
                    const ideaEdit = document.createElement("div");
                    ideaEdit.className = "lwm-brief-idea";
                    ideaEdit.setAttribute("contenteditable",
                        ideaEdit.contentEditable === "plaintext-only"
                            ? "plaintext-only" : "true");
                    ideaEdit.dataset.placeholder =
                        "(no brief yet — queue the workflow to generate one " +
                        "from your idea)";
                    ideaEdit.spellcheck = false;
                    ideaEdit.textContent = currentBrief.refined_idea || "";
                    ideaEdit.addEventListener("blur", () => {
                        const v = (ideaEdit.textContent || "").trim();
                        if (v === currentBrief.refined_idea) return;
                        currentBrief.refined_idea = v;
                        commitBrief();
                    });
                    ideaField.appendChild(ideaEdit);
                    ideaRow.appendChild(ideaLabel);
                    ideaRow.appendChild(ideaField);
                    briefPanel.appendChild(ideaRow);

                    // Build a chip-list row for one of the three array fields.
                    function buildListRow(labelText, key, kind, addPh, normFn) {
                        const row = document.createElement("div");
                        row.className = "lwm-brief-row";
                        const lab = document.createElement("div");
                        lab.className = "lwm-brief-label";
                        lab.textContent = labelText;
                        const field = document.createElement("div");
                        field.className = "lwm-brief-field";
                        const chips = document.createElement("div");
                        chips.className = "lwm-chip-row";
                        const items = currentBrief[key] || [];
                        items.forEach((val, idx) => {
                            chips.appendChild(buildChip(val, kind,
                                (next) => {
                                    const v = normFn ? normFn(next) : next;
                                    if (!v) {
                                        currentBrief[key].splice(idx, 1);
                                    } else {
                                        // Dedupe against the rest of the list.
                                        const dup = currentBrief[key].some(
                                            (x, j) => j !== idx && x === v);
                                        if (dup) {
                                            currentBrief[key].splice(idx, 1);
                                        } else {
                                            currentBrief[key][idx] = v;
                                        }
                                    }
                                    commitBrief();
                                    renderBrief();
                                    updateManagerSize();
                                },
                                () => {
                                    currentBrief[key].splice(idx, 1);
                                    commitBrief();
                                    renderBrief();
                                    updateManagerSize();
                                }
                            ));
                        });
                        chips.appendChild(buildChipAdd(addPh, (val) => {
                            const v = normFn ? normFn(val) : val.trim();
                            if (!v) return;
                            if (!currentBrief[key].includes(v)) {
                                currentBrief[key].push(v);
                            }
                            commitBrief();
                            renderBrief();
                            updateManagerSize();
                        }));
                        field.appendChild(chips);
                        row.appendChild(lab);
                        row.appendChild(field);
                        briefPanel.appendChild(row);
                    }

                    buildListRow(
                        "Fixed traits", "fixed_traits", "fixed",
                        "+ add trait",
                        (v) => String(v ?? "").trim(),
                    );
                    buildListRow(
                        "Forbidden axes", "forbidden_axes", "axis",
                        "+ add axis (snake_case)",
                        (v) => toSnakeCase(v),
                    );
                    buildListRow(
                        "Scene bans", "scene_bans", "scene",
                        "+ add ban",
                        (v) => String(v ?? "").trim(),
                    );
                }

                briefRegenBtn.addEventListener("click", () => {
                    currentBrief = emptyBrief();
                    commitBrief();
                    renderBrief();
                    updateManagerSize();
                });

                renderBrief();

                // ---- categories toolbar (fixed) ----
                const headLabel = document.createElement("div");
                headLabel.className = "lwm-section-label lwm-fixed";
                headLabel.textContent =
                    "Categories — descriptions sent to the LLM + entries on disk";
                root.appendChild(headLabel);

                const toolbar = document.createElement("div");
                toolbar.className = "lwm-toolbar lwm-fixed";

                const pathLine = document.createElement("div");
                pathLine.className = "lwm-pathline";
                pathLine.style.flex = "1 1 auto";
                pathLine.textContent = "wildcards: (loading…)";

                const refreshBtn = document.createElement("button");
                refreshBtn.textContent = "↻ Refresh disk";
                refreshBtn.title = "Re-read entries from the wildcards/ folder";
                refreshBtn.className = "lwm-btn lwm-btn-ghost";

                const addBtn = document.createElement("button");
                addBtn.textContent = "+ Add category";
                addBtn.className = "lwm-btn";

                toolbar.appendChild(pathLine);
                toolbar.appendChild(refreshBtn);
                toolbar.appendChild(addBtn);
                root.appendChild(toolbar);

                // ---- categories list ----
                // No scroll wrapper: the node itself grows to fit every row,
                // so the list never produces an internal scrollbar.
                const list = document.createElement("div");
                list.className = "lwm-list";
                root.appendChild(list);

                // ---- helpers ----
                function readRows() {
                    return Array.from(list.children).map(row => ({
                        name: row._nameInput.textContent ?? "",
                        desc: row._descInput.textContent ?? "",
                    }));
                }
                function commit() {
                    writeCategoriesJSON(jsonWidget, readRows());
                    node.setDirtyCanvas(true, true);
                }
                function badgeClass(count, on_disk) {
                    if (!on_disk || !count) return "lwm-badge";
                    if (count < 10) return "lwm-badge lwm-badge-low";
                    if (count < 50) return "lwm-badge lwm-badge-mid";
                    return "lwm-badge lwm-badge-high";
                }
                function buildRow({ name = "", desc = "", entries = null, count = 0,
                                    on_disk = false, user_override = false } = {}) {
                    const row = document.createElement("div");
                    row.className = "lwm-row" + (user_override ? " lwm-row-user" : "");

                    const top = document.createElement("div");
                    top.className = "lwm-row-head";

                    const expandBtn = document.createElement("button");
                    expandBtn.className = "lwm-expand";
                    expandBtn.textContent = "▸";
                    expandBtn.title = "Show entries on disk";

                    const nameI = makeEditable("name", name);
                    nameI.classList.add("lwm-input", "lwm-name");
                    row._nameInput = nameI;

                    const descI = makeEditable(
                        "description sent to the LLM", desc);
                    descI.classList.add("lwm-input", "lwm-desc");
                    row._descInput = descI;

                    const badge = document.createElement("span");
                    badge.className = badgeClass(count, on_disk);
                    badge.textContent = on_disk ? String(count) : "·";
                    badge.title = on_disk
                        ? `${count} ${count === 1 ? "entry" : "entries"} on disk`
                        : "no file on disk yet";

                    const rmBtn = document.createElement("button");
                    rmBtn.textContent = "✕";
                    rmBtn.title =
                        "Remove this override (file on disk is left alone)";
                    rmBtn.className = "lwm-btn lwm-btn-danger lwm-btn-icon";

                    top.appendChild(expandBtn);
                    top.appendChild(nameI);
                    top.appendChild(descI);
                    if (user_override) {
                        const tag = document.createElement("span");
                        tag.className = "lwm-tag";
                        tag.textContent = "OVERRIDE";
                        tag.title = "Description has been edited by the user";
                        top.appendChild(tag);
                    }
                    top.appendChild(badge);
                    top.appendChild(rmBtn);
                    row.appendChild(top);

                    const entriesPanel = document.createElement("div");
                    entriesPanel.className = "lwm-entries";
                    if (entries === null) {
                        entriesPanel.textContent =
                            "(click ↻ Refresh disk to load entries)";
                        entriesPanel.classList.add("lwm-empty");
                    } else if (!entries.length) {
                        entriesPanel.textContent = "(no entries on disk yet)";
                        entriesPanel.classList.add("lwm-empty");
                    } else {
                        entriesPanel.textContent = entries.join("\n");
                    }
                    row.appendChild(entriesPanel);
                    row._entriesPanel = entriesPanel;

                    expandBtn.addEventListener("click", () => {
                        const open = entriesPanel.classList.toggle("lwm-open");
                        expandBtn.classList.toggle("lwm-open", open);
                        updateManagerSize();
                    });

                    nameI.addEventListener("input", commit);
                    descI.addEventListener("input", commit);
                    rmBtn.addEventListener("click", () => {
                        row.remove();
                        commit();
                        updateManagerSize();
                    });

                    return row;
                }

                const FAILURE_PROMPT_TEXT = {
                    parse_failed:
                        "(LLM did not return parseable JSON — see raw reply below)",
                    no_prompt:
                        "(LLM JSON had no 'prompt' field — see raw reply below)",
                    llm_error:
                        "(LLM call failed — see raw reply below)",
                };

                function renderPrompt(template, status) {
                    const t = (template || "").trim();
                    promptPanel.classList.remove("lwm-error");
                    if (!t) {
                        promptPanel.classList.add("lwm-empty");
                        if (FAILURE_PROMPT_TEXT[status]) {
                            promptPanel.classList.add("lwm-error");
                            promptPanel.textContent = FAILURE_PROMPT_TEXT[status];
                        } else {
                            promptPanel.textContent =
                                "(no template yet — queue the workflow to generate)";
                        }
                    } else {
                        promptPanel.classList.remove("lwm-empty");
                        promptPanel.innerHTML = renderTemplateHTML(t);
                    }
                }

                function renderStatus(status, message) {
                    const isFailure =
                        status &&
                        status !== "ok" &&
                        (message || FAILURE_PROMPT_TEXT[status]);
                    if (!isFailure) {
                        statusBanner.style.display = "none";
                        statusBanner.className = "lwm-status-banner lwm-fixed";
                        return;
                    }
                    statusBanner.style.display = "block";
                    statusBanner.className =
                        "lwm-status-banner lwm-fixed lwm-status-" + status;
                    statusBanner.textContent =
                        message || `LLM call status: ${status}`;
                }

                function renderRawReply(raw, status) {
                    const text = String(raw ?? "");
                    const hasContent = text.trim().length > 0;
                    // "locked" means the LLM was deliberately skipped — treat
                    // it as a normal status (no red border, no auto-expand).
                    const isFailure =
                        status && status !== "ok" && status !== "locked";
                    rawPanel.classList.remove("lwm-error-border");
                    if (isFailure && hasContent) {
                        rawPanel.classList.add("lwm-error-border");
                    }
                    rawPanel.textContent = text;
                    rawHeader.style.display = hasContent ? "" : "none";
                    // Auto-open on failure so the user sees what came back; on
                    // success/locked, leave it collapsed unless the user opened it.
                    if (isFailure && hasContent) {
                        rawHeader.classList.add("lwm-open");
                        rawPanel.style.display = "block";
                    }
                }

                function rebuildFromSnapshot(snapshot) {
                    list.innerHTML = "";
                    if (snapshot?.wildcards_dir) {
                        pathLine.textContent =
                            `wildcards: ${snapshot.wildcards_dir}`;
                    }
                    renderPrompt(snapshot?.generated_prompt, snapshot?.status);
                    renderStatus(snapshot?.status, snapshot?.status_message);
                    renderRawReply(snapshot?.raw_reply, snapshot?.status);
                    // Adopt the brief from the snapshot only when the user
                    // hasn't already edited one on the node — preserves
                    // in-flight edits across re-execute / state refreshes.
                    const widgetBrief = briefWidget
                        ? readBriefJSON(briefWidget) : emptyBrief();
                    const userHasEdits = !briefIsEmpty(widgetBrief);
                    if (!userHasEdits && snapshot?.brief) {
                        const snap = snapshot.brief;
                        currentBrief = {
                            refined_idea: snap.refined_idea || "",
                            fixed_traits: Array.isArray(snap.fixed_traits)
                                ? snap.fixed_traits.slice() : [],
                            forbidden_axes: Array.isArray(snap.forbidden_axes)
                                ? snap.forbidden_axes.slice() : [],
                            scene_bans: Array.isArray(snap.scene_bans)
                                ? snap.scene_bans.slice() : [],
                        };
                        commitBrief();
                        renderBrief();
                    }
                    for (const r of (snapshot?.rows || [])) {
                        list.appendChild(buildRow({
                            name: r.name,
                            desc: r.description ?? "",
                            entries: r.entries,
                            count: r.count,
                            on_disk: r.on_disk,
                            user_override: r.user_override,
                        }));
                    }
                    if (!list.children.length) list.appendChild(buildRow());
                    updateManagerSize();
                }

                function rebuildFromWidgetOnly() {
                    list.innerHTML = "";
                    const declared = readCategoriesJSON(jsonWidget);
                    for (const [n, d] of Object.entries(declared)) {
                        list.appendChild(buildRow({
                            name: n, desc: d, user_override: true,
                        }));
                    }
                    if (briefWidget) {
                        currentBrief = readBriefJSON(briefWidget);
                        renderBrief();
                    }
                    updateManagerSize();
                }

                async function refreshFromServer() {
                    try {
                        const resp = await fetch("/llm_wildcard/state");
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        const snap = await resp.json();
                        rebuildFromSnapshot(snap);
                    } catch (e) {
                        console.warn("[LLMWildcardManager] refresh failed:", e);
                        pathLine.textContent =
                            `wildcards: refresh failed (${e.message})`;
                    }
                }
                refreshBtn.addEventListener("click", refreshFromServer);
                addBtn.addEventListener("click", () => {
                    list.appendChild(buildRow({ user_override: true }));
                    commit();
                    updateManagerSize();
                });

                rebuildFromWidgetOnly();

                const managerWidget = node.addDOMWidget(
                    "manager_view", "div", root, { serialize: false });
                node.size = [Math.max(node.size[0], 580), node.size[1]];

                // Dynamic sizing: measure the rendered content height after
                // the next frame, then resize the widget + node to match. The
                // RAF guard collapses bursts of changes into one resize and
                // ensures we measure after layout settles.
                let _sizingPending = false;
                updateManagerSize = function () {
                    if (_sizingPending) return;
                    _sizingPending = true;
                    requestAnimationFrame(() => {
                        _sizingPending = false;
                        const h = Math.ceil(root.scrollHeight) + 8;
                        const prev = managerWidget.computedHeight || 0;
                        if (Math.abs(prev - h) < 2) return;
                        managerWidget.computeSize =
                            (width) => [width, h];
                        managerWidget.computedHeight = h;
                        const min = node.computeSize();
                        const w = Math.max(node.size[0], min[0], 580);
                        node.setSize([w, min[1]]);
                        node.setDirtyCanvas(true, true);
                    });
                };
                updateManagerSize();

                // Re-fit whenever the DOM content itself changes height (text
                // wrapping, chips added, panels expanded) — without this the
                // node only resized on the handful of events that called the
                // updater explicitly, and grown content overlapped below the
                // frame.
                if (typeof ResizeObserver !== "undefined") {
                    const ro = new ResizeObserver(() => updateManagerSize());
                    ro.observe(root);
                }

                // When the user drags the corner, LiteGraph lets the frame
                // shrink past the content. Clamp to the computed minimum so
                // the body always fits inside the node.
                const onResize = node.onResize;
                node.onResize = function (size) {
                    onResize?.apply(this, arguments);
                    const min = node.computeSize();
                    if (size) {
                        if (size[0] < Math.max(min[0], 580)) {
                            size[0] = Math.max(min[0], 580);
                        }
                        if (size[1] < min[1]) size[1] = min[1];
                    }
                };

                const onConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    onConfigure?.apply(this, arguments);
                    setTimeout(() => {
                        rebuildFromWidgetOnly();
                        refreshFromServer();
                    }, 0);
                };
                setTimeout(refreshFromServer, 50);

                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    const raw = (message?.manager_state || [])[0];
                    if (!raw) return;
                    try {
                        const snap = JSON.parse(raw);
                        rebuildFromSnapshot(snap);
                    } catch (e) {
                        console.warn("[LLMWildcardManager] bad snapshot:", e);
                    }
                };
            };
        }

        // -------------------------------------------------------------------
        // LLMWildcardTemplateBuilder — hand-authored structure editor
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardTemplateBuilder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                injectStyles();
                const node = this;

                const structWidget =
                    node.widgets.find(w => w.name === "structure");
                if (!structWidget) return;
                hideWidget(node, structWidget);

                let blocks = readStructureBlocks(structWidget);

                const root = document.createElement("div");
                root.className = "lwm-root lwm-root-fit";

                let updateBuilderSize = () => {};

                // ---- structure editor ----
                const structLabel = document.createElement("div");
                structLabel.className = "lwm-section-label lwm-fixed";
                structLabel.textContent =
                    "Structure — build the prompt shape, then wire into the Manager";
                root.appendChild(structLabel);

                const skeleton = document.createElement("div");
                skeleton.className = "lwm-skeleton lwm-fixed";
                root.appendChild(skeleton);

                const list = document.createElement("div");
                list.className = "lwm-list";
                root.appendChild(list);

                // ---- add-block toolbar ----
                const toolbar = document.createElement("div");
                toolbar.className = "lwm-toolbar lwm-fixed";
                const addSentenceBtn = document.createElement("button");
                addSentenceBtn.className = "lwm-btn lwm-btn-ghost";
                addSentenceBtn.textContent = "+ Sentence";
                const addWildBtn = document.createElement("button");
                addWildBtn.className = "lwm-btn lwm-btn-ghost";
                addWildBtn.textContent = "+ Wildcard group";
                toolbar.appendChild(addSentenceBtn);
                toolbar.appendChild(addWildBtn);
                const tbSpacer = document.createElement("div");
                tbSpacer.className = "lwm-spacer";
                toolbar.appendChild(tbSpacer);

                // ---- starter-preset loader ----
                const presetSelect = document.createElement("select");
                presetSelect.className = "lwm-select lwm-preset-select";
                presetSelect.title =
                    "Load a ready-made structure (replaces the current blocks)";
                const presetPlaceholder = document.createElement("option");
                presetPlaceholder.value = "";
                presetPlaceholder.textContent = "Load preset…";
                presetSelect.appendChild(presetPlaceholder);
                const flatPresets = [];
                for (const grp of BUILDER_PRESETS) {
                    const og = document.createElement("optgroup");
                    og.label = grp.group;
                    for (const it of grp.items) {
                        const o = document.createElement("option");
                        o.value = String(flatPresets.length);
                        o.textContent = it.name;
                        og.appendChild(o);
                        flatPresets.push(it);
                    }
                    presetSelect.appendChild(og);
                }
                presetSelect.addEventListener("change", () => {
                    const it = flatPresets[parseInt(presetSelect.value, 10)];
                    presetSelect.value = "";          // snap back to placeholder
                    if (!it) return;
                    // normalizeBlocks deep-copies + validates, so the shared
                    // preset definition is never mutated by later edits.
                    blocks = normalizeBlocks({ blocks: it.blocks });
                    commit();
                    render();
                });
                toolbar.appendChild(presetSelect);

                root.appendChild(toolbar);

                // ---------- editor logic ----------
                function commit() {
                    writeStructureBlocks(structWidget, blocks);
                    skeleton.textContent = structureSkeleton(blocks);
                    node.setDirtyCanvas(true, true);
                    updateBuilderSize();
                }

                function move(idx, dir) {
                    const j = idx + dir;
                    if (j < 0 || j >= blocks.length) return;
                    const t = blocks[idx]; blocks[idx] = blocks[j]; blocks[j] = t;
                    commit();
                    render();
                }

                function iconBtn(label, title, onClick, extra = "") {
                    const b = document.createElement("button");
                    b.className = "lwm-btn lwm-btn-ghost lwm-btn-icon " + extra;
                    b.textContent = label;
                    b.title = title;
                    b.addEventListener("click", onClick);
                    return b;
                }

                function buildRoleSelect(kind, value, onChange) {
                    const sel = document.createElement("select");
                    sel.className = "lwm-select";
                    sel.title = "Abstract role — structure only, not content";
                    const roles = kind === "sentence"
                        ? BUILDER_SENTENCE_ROLES : BUILDER_WILDCARD_ROLES;
                    const undef = document.createElement("option");
                    undef.value = "";
                    undef.textContent =
                        kind === "sentence" ? "(any)" : "(undefined)";
                    sel.appendChild(undef);
                    for (const r of roles) {
                        const o = document.createElement("option");
                        o.value = r; o.textContent = r;
                        sel.appendChild(o);
                    }
                    if (value && !roles.includes(value)) {
                        const o = document.createElement("option");
                        o.value = value; o.textContent = value + " (custom)";
                        sel.appendChild(o);
                    }
                    sel.value = value || "";
                    sel.addEventListener("change", () => onChange(sel.value));
                    return sel;
                }

                function buildBlockRow(idx) {
                    const blk = blocks[idx];
                    const row = document.createElement("div");
                    row.className = "lwm-row lwm-blk" + (blk.enabled ? "" : " lwm-blk-off");

                    const head = document.createElement("div");
                    head.className = "lwm-row-head";

                    const en = document.createElement("input");
                    en.type = "checkbox"; en.className = "lwm-chk";
                    en.checked = blk.enabled;
                    en.title = "Enable / disable this block";
                    en.addEventListener("change", () => {
                        blk.enabled = en.checked;
                        row.classList.toggle("lwm-blk-off", !en.checked);
                        commit();
                    });

                    const kind = document.createElement("span");
                    kind.className = "lwm-blk-kind " +
                        (blk.kind === "sentence" ? "lwm-blk-sentence" : "lwm-blk-wild");
                    kind.textContent =
                        blk.kind === "sentence" ? "Sentence" : "Wildcards";

                    const role = buildRoleSelect(blk.kind, blk.role, (v) => {
                        blk.role = v; commit();
                    });

                    head.appendChild(en);
                    head.appendChild(kind);
                    head.appendChild(role);

                    if (blk.kind === "wildcards") {
                        const range = document.createElement("input");
                        range.type = "range"; range.min = "1";
                        range.max = String(BUILDER_MAX_COUNT);
                        range.value = String(blk.count);
                        range.className = "lwm-range";
                        range.title = "How many wildcard slots this group emits";
                        const num = document.createElement("span");
                        num.className = "lwm-count";
                        num.textContent = "×" + blk.count;
                        range.addEventListener("input", () => {
                            blk.count = parseInt(range.value, 10) || 1;
                            num.textContent = "×" + blk.count;
                            commit();
                        });

                        const fnWrap = document.createElement("label");
                        fnWrap.className = "lwm-fnlabel";
                        fnWrap.title =
                            "Force a fresh value every run (emits __!name__)";
                        const fn = document.createElement("input");
                        fn.type = "checkbox"; fn.className = "lwm-chk";
                        fn.checked = !!blk.force_new;
                        fn.addEventListener("change", () => {
                            blk.force_new = fn.checked; commit();
                        });
                        fnWrap.appendChild(fn);
                        fnWrap.appendChild(document.createTextNode("new"));

                        head.appendChild(range);
                        head.appendChild(num);
                        head.appendChild(fnWrap);
                    }

                    const spacer = document.createElement("div");
                    spacer.className = "lwm-spacer";
                    head.appendChild(spacer);

                    head.appendChild(iconBtn("↑", "Move up", () => move(idx, -1)));
                    head.appendChild(iconBtn("↓", "Move down", () => move(idx, 1)));
                    head.appendChild(iconBtn("✕", "Remove block", () => {
                        blocks.splice(idx, 1); commit(); render();
                    }, "lwm-btn-danger"));

                    row.appendChild(head);

                    if (blk.kind === "sentence") {
                        const body = makeEditable(
                            "leave empty → AI writes this sentence",
                            blk.text || "");
                        body.className = "lwm-input lwm-blk-text";
                        body.addEventListener("blur", () => {
                            blk.text = (body.textContent || "").trim();
                            commit();
                        });
                        row.appendChild(body);
                    }
                    return row;
                }

                function render() {
                    list.innerHTML = "";
                    blocks.forEach((_, i) => list.appendChild(buildBlockRow(i)));
                    skeleton.textContent = structureSkeleton(blocks);
                    updateBuilderSize();
                }

                addSentenceBtn.addEventListener("click", () => {
                    blocks.push({ kind: "sentence", enabled: true, role: "", text: "" });
                    commit(); render();
                });
                addWildBtn.addEventListener("click", () => {
                    blocks.push({ kind: "wildcards", enabled: true, role: "",
                                  count: 1, force_new: false });
                    commit(); render();
                });

                render();

                const builderWidget = node.addDOMWidget(
                    "builder_view", "div", root, { serialize: false });
                node.size = [Math.max(node.size[0], 560), node.size[1]];

                let _sizingPending = false;
                updateBuilderSize = function () {
                    if (_sizingPending) return;
                    _sizingPending = true;
                    requestAnimationFrame(() => {
                        _sizingPending = false;
                        const h = Math.ceil(root.scrollHeight) + 8;
                        const prev = builderWidget.computedHeight || 0;
                        if (Math.abs(prev - h) < 2) return;
                        builderWidget.computeSize = (width) => [width, h];
                        builderWidget.computedHeight = h;
                        const min = node.computeSize();
                        const w = Math.max(node.size[0], min[0], 560);
                        node.setSize([w, min[1]]);
                        node.setDirtyCanvas(true, true);
                    });
                };
                updateBuilderSize();

                // Auto-refit on any DOM size change (blocks added/removed,
                // text wrapping, expanded panels). Without this, only the
                // explicit `updateBuilderSize()` callsites resized the node
                // and content could overlap below.
                if (typeof ResizeObserver !== "undefined") {
                    const ro = new ResizeObserver(() => updateBuilderSize());
                    ro.observe(root);
                }

                // Prevent the user from dragging the frame below the content's
                // natural size — LiteGraph otherwise lets them shrink past it,
                // leaving the DOM body overflowing the visible node.
                const onResize = node.onResize;
                node.onResize = function (size) {
                    onResize?.apply(this, arguments);
                    const min = node.computeSize();
                    if (size) {
                        if (size[0] < Math.max(min[0], 560)) {
                            size[0] = Math.max(min[0], 560);
                        }
                        if (size[1] < min[1]) size[1] = min[1];
                    }
                };

                const onConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    onConfigure?.apply(this, arguments);
                    setTimeout(() => {
                        blocks = readStructureBlocks(structWidget);
                        render();
                    }, 0);
                };
            };
        }

        // -------------------------------------------------------------------
        // LLMWildcardReport — structured collapsible view
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardReport") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                injectStyles();
                const node = this;

                const root = document.createElement("div");
                // `lwm-root-fit` lets the node wrap its content instead of
                // stretching to fill the frame. Inner panels (slots scroll +
                // raw textarea) own their own scroll, so the node settles at
                // header + slots cap + textarea + padding.
                root.className = "lwm-root lwm-root-fit";

                const header = document.createElement("div");
                header.className = "lwm-report-header lwm-fixed";
                header.textContent = "(no report yet)";
                root.appendChild(header);

                const slotsLabel = document.createElement("div");
                slotsLabel.className = "lwm-section-label lwm-fixed";
                slotsLabel.textContent = "Per-slot details";
                root.appendChild(slotsLabel);

                // slots list scrolls inside its own container, capped at
                // 240px so a long run doesn't push the node off-screen.
                const slotsScroll = document.createElement("div");
                slotsScroll.className = "lwm-scroll lwm-cap-slots";
                const slots = document.createElement("div");
                slots.className = "lwm-list";
                slotsScroll.appendChild(slots);
                root.appendChild(slotsScroll);

                const rawLabel = document.createElement("div");
                rawLabel.className = "lwm-section-label lwm-fixed";
                rawLabel.textContent = "Raw report";
                root.appendChild(rawLabel);

                // Raw textarea wraps long lines and has a fixed height; it
                // owns its own scrollbar instead of growing the node body.
                const rawTA = document.createElement("textarea");
                rawTA.readOnly = true;
                rawTA.spellcheck = false;
                rawTA.className = "lwm-textarea lwm-raw-textarea";
                root.appendChild(rawTA);

                node._rawTA = rawTA;
                node._reportRoot = root;

                function buildSlot(rec) {
                    const slot = document.createElement("div");
                    slot.className = "lwm-slot";

                    const head = document.createElement("div");
                    head.className = "lwm-slot-head";

                    const expand = document.createElement("button");
                    expand.className = "lwm-expand";
                    expand.textContent = "▸";
                    expand.title = "Show LLM call details";

                    const status = document.createElement("span");
                    const s = rec.status || "error";
                    status.className = "lwm-status s-" + s;
                    status.textContent = s;

                    const name = document.createElement("span");
                    name.className = "lwm-slot-name";
                    name.textContent = rec.name || "(unnamed)";

                    const value = document.createElement("span");
                    value.className = "lwm-slot-value";
                    value.textContent = rec.value || "(no value)";
                    value.title = rec.value || "";

                    head.appendChild(expand);
                    head.appendChild(status);
                    head.appendChild(name);
                    head.appendChild(value);
                    slot.appendChild(head);

                    const detail = document.createElement("div");
                    detail.className = "lwm-slot-detail";
                    const lines = [];
                    function row(label, val) {
                        if (val === undefined || val === null || val === "") return;
                        lines.push(
                            `<div class="lwm-detail-row">` +
                            `<span class="lwm-detail-key">${escapeHTML(label)}:</span> ` +
                            `${escapeHTML(String(val))}</div>`
                        );
                    }
                    row("pool size", rec.pool_size);
                    row("sent → LLM", rec.sent);
                    row("LLM reply", rec.raw);
                    row("retry sent", rec.retry_sent);
                    row("retry reply", rec.retry_raw);
                    row("error", rec.err);
                    if (!lines.length) {
                        lines.push(
                            `<div class="lwm-detail-row" style="font-style:italic;color:#6c7480">` +
                            `(no details — value was reused from disk)</div>`
                        );
                    }
                    detail.innerHTML = lines.join("");
                    slot.appendChild(detail);

                    expand.addEventListener("click", () => {
                        const open = detail.classList.toggle("lwm-open");
                        expand.classList.toggle("lwm-open", open);
                    });
                    return slot;
                }

                function renderTallies(t, raw) {
                    const total = t?.total || 0;
                    const gen = t?.generated || 0;
                    const reu = t?.reused || 0;
                    const err = t?.errors || 0;
                    const sub = total === 0
                        ? `(no wildcards in template)`
                        : "";
                    header.innerHTML =
                        `<span class="lwm-stat">total: ${total}</span>` +
                        `<span class="lwm-stat gen">generated: ${gen}</span>` +
                        `<span class="lwm-stat reu">reused: ${reu}</span>` +
                        `<span class="lwm-stat err">errors: ${err}</span>` +
                        (sub ? `<span style="color:#7d8693">${sub}</span>` : "");
                }

                function renderRecords(records) {
                    slots.innerHTML = "";
                    if (!records || !records.length) {
                        const empty = document.createElement("div");
                        empty.style.color = "#6c7480";
                        empty.style.fontStyle = "italic";
                        empty.style.fontSize = "11px";
                        empty.textContent =
                            "(structured records unavailable — see raw report below)";
                        slots.appendChild(empty);
                        return;
                    }
                    for (const r of records) slots.appendChild(buildSlot(r));
                }

                // The node wraps its rendered content: each panel inside
                // owns its own scrollbar (slots cap at 240px, textarea is a
                // fixed 200px), and the node body sizes itself to whatever
                // the root currently measures. No feedback loop with
                // LiteGraph's per-frame size check.
                const reportWidget = node.addDOMWidget(
                    "report_view", "div", root, { serialize: false });
                node.size = [Math.max(node.size[0], 560), node.size[1]];
                const updateReportSize =
                    fitWidgetToContent(node, reportWidget, root, 560);
                updateReportSize();

                node._renderReport = (payload) => {
                    renderTallies(payload?.tallies, payload?.raw);
                    renderRecords(payload?.records);
                    rawTA.value = payload?.raw || "";
                    updateReportSize();
                };
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const raw = (message?.report_state || [])[0];
                if (!raw) return;
                try {
                    const payload = JSON.parse(raw);
                    this._renderReport?.(payload);
                } catch (e) {
                    console.warn("[LLMWildcardReport] bad payload:", e);
                }
            };

            // re-fetch the latest payload when a saved workflow is loaded so
            // the panel doesn't appear empty after reload.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                setTimeout(async () => {
                    try {
                        const resp = await fetch("/llm_wildcard/last_report");
                        if (!resp.ok) return;
                        const payload = await resp.json();
                        this._renderReport?.({
                            tallies: payload.tallies,
                            records: payload.records,
                            raw: payload.text,
                        });
                    } catch {}
                }, 50);
            };
        }
    },
});
