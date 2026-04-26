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

// Hide a built-in widget without removing it (so its value still serializes).
function hideWidget(node, widget) {
    widget.computeSize = () => [0, -4];
    widget.type = "hidden_" + widget.type;
    if (widget.element) widget.element.style.display = "none";
}

// One-time CSS for the Manager + Report. Scoped to `.lwm-*` so it can't bleed
// into the rest of ComfyUI.
function injectStyles() {
    if (document.getElementById("lwm-styles")) return;
    const css = `
        .lwm-root { display:flex; flex-direction:column; gap:10px;
            padding:6px; box-sizing:border-box; width:100%; height:100%;
            max-width:100%; min-width:0;
            overflow-x:hidden; overflow-y:auto;
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                Roboto, sans-serif; color:#dcdcdc;
        }
        .lwm-root::-webkit-scrollbar { width:8px; }
        .lwm-root::-webkit-scrollbar-thumb {
            background:#2c3138; border-radius:4px; }
        .lwm-root::-webkit-scrollbar-thumb:hover { background:#3a4250; }
        .lwm-root * { box-sizing:border-box; }
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
        .lwm-textarea { width:100%; resize:none;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size:11px; line-height:1.4; white-space:pre; }
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
        .lwm-prompt-panel .lwm-tok { color:#7ec9ff; background:#142235;
            padding:1px 3px; border-radius:3px; }
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
        .lwm-flex-fill { flex:1 1 auto; min-height:120px;
            display:flex; flex-direction:column; min-width:0; }
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

                const root = document.createElement("div");
                root.className = "lwm-root";

                // ---- generated prompt panel ----
                const promptLabel = document.createElement("div");
                promptLabel.className = "lwm-section-label";
                promptLabel.textContent = "Generated prompt template";
                const promptPanel = document.createElement("div");
                promptPanel.className = "lwm-prompt-panel lwm-empty";
                promptPanel.textContent =
                    "(no template yet — queue the workflow to generate)";
                root.appendChild(promptLabel);
                root.appendChild(promptPanel);

                // ---- categories toolbar ----
                const headLabel = document.createElement("div");
                headLabel.className = "lwm-section-label";
                headLabel.textContent =
                    "Categories — descriptions sent to the LLM + entries on disk";
                root.appendChild(headLabel);

                const toolbar = document.createElement("div");
                toolbar.className = "lwm-toolbar";

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

                const list = document.createElement("div");
                list.className = "lwm-list";
                root.appendChild(list);

                // ---- helpers ----
                function readRows() {
                    return Array.from(list.children).map(row => ({
                        name: row._nameInput.value,
                        desc: row._descInput.value,
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

                    const nameI = document.createElement("input");
                    nameI.type = "text";
                    nameI.placeholder = "name";
                    nameI.value = name;
                    nameI.className = "lwm-input lwm-name";
                    row._nameInput = nameI;

                    const descI = document.createElement("input");
                    descI.type = "text";
                    descI.placeholder = "description sent to the LLM";
                    descI.value = desc;
                    descI.className = "lwm-input lwm-desc";
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
                    });

                    nameI.addEventListener("input", commit);
                    descI.addEventListener("input", commit);
                    rmBtn.addEventListener("click", () => {
                        row.remove();
                        commit();
                    });

                    return row;
                }

                function renderPrompt(template) {
                    const t = (template || "").trim();
                    if (!t) {
                        promptPanel.classList.add("lwm-empty");
                        promptPanel.textContent =
                            "(no template yet — queue the workflow to generate)";
                        return;
                    }
                    promptPanel.classList.remove("lwm-empty");
                    promptPanel.innerHTML = renderTemplateHTML(t);
                }

                function rebuildFromSnapshot(snapshot) {
                    list.innerHTML = "";
                    if (snapshot?.wildcards_dir) {
                        pathLine.textContent =
                            `wildcards: ${snapshot.wildcards_dir}`;
                    }
                    renderPrompt(snapshot?.generated_prompt);
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
                }

                function rebuildFromWidgetOnly() {
                    list.innerHTML = "";
                    const declared = readCategoriesJSON(jsonWidget);
                    for (const [n, d] of Object.entries(declared)) {
                        list.appendChild(buildRow({
                            name: n, desc: d, user_override: true,
                        }));
                    }
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
                });

                rebuildFromWidgetOnly();

                node.addDOMWidget("manager_view", "div", root, { serialize: false });
                node.size = [
                    Math.max(node.size[0], 580),
                    Math.max(node.size[1], 660),
                ];

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
        // LLMWildcardReport — structured collapsible view
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardReport") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                injectStyles();
                const node = this;

                const root = document.createElement("div");
                root.className = "lwm-root";

                const header = document.createElement("div");
                header.className = "lwm-report-header";
                header.textContent = "(no report yet)";
                root.appendChild(header);

                const slotsLabel = document.createElement("div");
                slotsLabel.className = "lwm-section-label";
                slotsLabel.textContent = "Per-slot details";
                root.appendChild(slotsLabel);

                const slots = document.createElement("div");
                slots.className = "lwm-list";
                root.appendChild(slots);

                const rawLabel = document.createElement("div");
                rawLabel.className = "lwm-section-label";
                rawLabel.textContent = "Raw report";
                root.appendChild(rawLabel);

                const rawWrap = document.createElement("div");
                rawWrap.className = "lwm-flex-fill";
                const rawTA = document.createElement("textarea");
                rawTA.readOnly = true;
                rawTA.spellcheck = false;
                rawTA.className = "lwm-textarea";
                rawTA.style.flex = "1 1 auto";
                rawTA.style.minHeight = "100%";
                rawWrap.appendChild(rawTA);
                root.appendChild(rawWrap);

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

                node.addDOMWidget("report_view", "div", root, { serialize: false });
                node.size = [
                    Math.max(node.size[0], 560),
                    Math.max(node.size[1], 480),
                ];

                node._renderReport = (payload) => {
                    renderTallies(payload?.tallies, payload?.raw);
                    renderRecords(payload?.records);
                    rawTA.value = payload?.raw || "";
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
