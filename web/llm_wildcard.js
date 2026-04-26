// Frontend extension for the LLM Wildcard Manager nodes.
//
//   * LLMWildcardReport          — render the resolver's report inside the node.
//   * LLMWildcardPromptConfig    — clickable add/remove rows for category overrides,
//                                   serialized into the underlying JSON STRING widget.
//   * LLMWildcardManager         — central UI: direction combo (already rendered by
//                                   ComfyUI), category table with live entry view per
//                                   category (read from disk via /llm_wildcard/state),
//                                   inline report panel when `report` is wired.
//
// The Python side stores everything as plain widgets, so workflows continue to
// work without this extension (they just lose the nicer UI).

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

// One-time CSS for the Manager UI. Scoped to `.lwm-*` so it can't bleed into
// the rest of ComfyUI. We inject from JS so the extension stays single-file.
function injectManagerStyles() {
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
            min-width:0; max-width:100%; }
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
        .lwm-textarea { width:100%; resize:vertical;
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
        .lwm-report-wrap { display:none; flex-direction:column; gap:4px;
            margin-top:4px; }
        .lwm-report-wrap.lwm-open { display:flex; }
        .lwm-report-meta { font-size:10px; color:#7d8693; }
        /* nicer scrollbars on dark backgrounds */
        .lwm-entries::-webkit-scrollbar, .lwm-textarea::-webkit-scrollbar { width:8px; height:8px; }
        .lwm-entries::-webkit-scrollbar-thumb, .lwm-textarea::-webkit-scrollbar-thumb {
            background:#2c3138; border-radius:4px; }
        .lwm-entries::-webkit-scrollbar-thumb:hover, .lwm-textarea::-webkit-scrollbar-thumb:hover {
            background:#3a4250; }
    `;
    const style = document.createElement("style");
    style.id = "lwm-styles";
    style.textContent = css;
    document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "comfyui.llm_wildcard_manager",

    async beforeRegisterNodeDef(nodeType, nodeData) {

        // -------------------------------------------------------------------
        // LLMWildcardReport — display incoming report text in the node body
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardReport") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const text = (message?.text || []).join("\n");

                if (!this._reportTA) {
                    const ta = document.createElement("textarea");
                    ta.readOnly = true;
                    ta.spellcheck = false;
                    Object.assign(ta.style, {
                        width: "100%",
                        maxWidth: "100%",
                        minHeight: "260px",
                        fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                        fontSize: "11px",
                        lineHeight: "1.35",
                        background: "#15171a",
                        color: "#d8d8d8",
                        border: "1px solid #444",
                        borderRadius: "4px",
                        padding: "6px 8px",
                        boxSizing: "border-box",
                        whiteSpace: "pre",
                        overflow: "auto",
                        resize: "vertical",
                    });
                    this.addDOMWidget("report_view", "textarea", ta, {
                        serialize: false,
                    });
                    this._reportTA = ta;
                    // give the node a sensible starting size
                    this.size = [Math.max(this.size[0], 520), Math.max(this.size[1], 360)];
                }
                this._reportTA.value = text;
            };

            // restore on workflow load: empty until next run
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
            };
        }

        // -------------------------------------------------------------------
        // LLMWildcardPromptConfig — categories table editor
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardPromptConfig") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const node = this;

                const jsonWidget = node.widgets.find(w => w.name === "category_overrides");
                if (!jsonWidget) return;
                hideWidget(node, jsonWidget);

                // ----- container -----
                const root = document.createElement("div");
                Object.assign(root.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                    padding: "4px",
                    boxSizing: "border-box",
                    width: "100%",
                    maxWidth: "100%",
                    minWidth: "0",
                    overflow: "hidden",
                });

                const header = document.createElement("div");
                header.textContent = "Category overrides (description per wildcard name)";
                Object.assign(header.style, {
                    fontSize: "11px",
                    color: "#aaa",
                    marginBottom: "2px",
                });
                root.appendChild(header);

                const list = document.createElement("div");
                Object.assign(list.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                });
                root.appendChild(list);

                const addBtn = document.createElement("button");
                addBtn.textContent = "+ Add category";
                Object.assign(addBtn.style, {
                    padding: "5px 10px",
                    background: "#2b5d8a",
                    color: "#fff",
                    border: "none",
                    borderRadius: "3px",
                    cursor: "pointer",
                    fontSize: "12px",
                    alignSelf: "flex-start",
                });
                addBtn.addEventListener("mouseenter", () => addBtn.style.background = "#3873a8");
                addBtn.addEventListener("mouseleave", () => addBtn.style.background = "#2b5d8a");
                root.appendChild(addBtn);

                // ----- helpers -----
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
                function styleInput(el) {
                    Object.assign(el.style, {
                        background: "#1a1c1f",
                        color: "#e0e0e0",
                        border: "1px solid #444",
                        borderRadius: "3px",
                        padding: "4px 6px",
                        fontSize: "12px",
                        outline: "none",
                        boxSizing: "border-box",
                        minWidth: "0",
                        maxWidth: "100%",
                    });
                }
                function buildRow(name = "", desc = "") {
                    const row = document.createElement("div");
                    Object.assign(row.style, {
                        display: "flex",
                        gap: "4px",
                        alignItems: "center",
                        minWidth: "0",
                        maxWidth: "100%",
                    });

                    const nameI = document.createElement("input");
                    nameI.type = "text";
                    nameI.placeholder = "name";
                    nameI.value = name;
                    nameI.style.flex = "0 1 32%";
                    styleInput(nameI);
                    row._nameInput = nameI;

                    const descI = document.createElement("input");
                    descI.type = "text";
                    descI.placeholder = "description sent to the LLM";
                    descI.value = desc;
                    descI.style.flex = "1 1 auto";
                    styleInput(descI);
                    row._descInput = descI;

                    const rmBtn = document.createElement("button");
                    rmBtn.textContent = "✕";
                    rmBtn.title = "Remove";
                    Object.assign(rmBtn.style, {
                        flex: "0 0 28px",
                        cursor: "pointer",
                        background: "#5a2a2a",
                        color: "#fff",
                        border: "none",
                        borderRadius: "3px",
                        padding: "4px 0",
                        fontSize: "12px",
                    });
                    rmBtn.addEventListener("mouseenter", () => rmBtn.style.background = "#7a3a3a");
                    rmBtn.addEventListener("mouseleave", () => rmBtn.style.background = "#5a2a2a");

                    nameI.addEventListener("input", commit);
                    descI.addEventListener("input", commit);
                    rmBtn.addEventListener("click", () => { row.remove(); commit(); });

                    row.appendChild(nameI);
                    row.appendChild(descI);
                    row.appendChild(rmBtn);
                    return row;
                }
                function rebuildFromWidget() {
                    list.innerHTML = "";
                    const data = readCategoriesJSON(jsonWidget);
                    for (const [name, desc] of Object.entries(data)) {
                        list.appendChild(buildRow(name, String(desc)));
                    }
                }

                addBtn.addEventListener("click", () => {
                    list.appendChild(buildRow());
                    commit();
                });

                rebuildFromWidget();

                node.addDOMWidget("categories_editor", "div", root, {
                    serialize: false,
                });
                node.size = [Math.max(node.size[0], 520), Math.max(node.size[1], 380)];

                // re-sync after loading a saved workflow
                const onConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    onConfigure?.apply(this, arguments);
                    setTimeout(rebuildFromWidget, 0);
                };
            };
        }

        // -------------------------------------------------------------------
        // LLMWildcardManager — central management UI
        // -------------------------------------------------------------------
        if (nodeData.name === "LLMWildcardManager") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                injectManagerStyles();
                const node = this;

                const jsonWidget = node.widgets.find(w => w.name === "categories");
                if (!jsonWidget) return;
                // Only the categories JSON widget is replaced by the DOM table.
                // direction / extra_flair / system_prompt_override remain as
                // standard ComfyUI widgets so they render inside the node
                // frame and don't overlay the canvas.
                hideWidget(node, jsonWidget);

                // ----- root layout -----
                const root = document.createElement("div");
                root.className = "lwm-root";

                // ----- toolbar: header + refresh + add -----
                const headLabel = document.createElement("div");
                headLabel.className = "lwm-section-label";
                headLabel.textContent = "Categories — descriptions sent to the LLM + entries on disk";
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

                // ----- report panel -----
                const reportWrap = document.createElement("div");
                reportWrap.className = "lwm-report-wrap";
                const reportLabel = document.createElement("div");
                reportLabel.className = "lwm-section-label";
                reportLabel.textContent = "Latest resolver report";
                const reportMeta = document.createElement("div");
                reportMeta.className = "lwm-report-meta";
                const reportTA = document.createElement("textarea");
                reportTA.readOnly = true;
                reportTA.spellcheck = false;
                reportTA.className = "lwm-textarea";
                reportTA.style.minHeight = "180px";
                reportWrap.appendChild(reportLabel);
                reportWrap.appendChild(reportMeta);
                reportWrap.appendChild(reportTA);
                root.appendChild(reportWrap);

                // ----- row helpers -----
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
                                    on_disk = false } = {}) {
                    const row = document.createElement("div");
                    row.className = "lwm-row";

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
                    rmBtn.title = "Remove this category override (file on disk is left alone)";
                    rmBtn.className = "lwm-btn lwm-btn-danger lwm-btn-icon";

                    top.appendChild(expandBtn);
                    top.appendChild(nameI);
                    top.appendChild(descI);
                    top.appendChild(badge);
                    top.appendChild(rmBtn);
                    row.appendChild(top);

                    const entriesPanel = document.createElement("div");
                    entriesPanel.className = "lwm-entries";
                    if (entries === null) {
                        entriesPanel.textContent = "(click ↻ Refresh disk to load entries)";
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
                    rmBtn.addEventListener("click", () => { row.remove(); commit(); });

                    return row;
                }

                function rebuildFromSnapshot(snapshot) {
                    list.innerHTML = "";
                    if (snapshot?.wildcards_dir) {
                        pathLine.textContent = `wildcards: ${snapshot.wildcards_dir}`;
                    }
                    const declared = readCategoriesJSON(jsonWidget);
                    const seen = new Set();
                    for (const r of (snapshot?.rows || [])) {
                        seen.add(r.name);
                        list.appendChild(buildRow({
                            name: r.name,
                            desc: declared[r.name] ?? r.description ?? "",
                            entries: r.entries,
                            count: r.count,
                            on_disk: r.on_disk,
                        }));
                    }
                    for (const [n, d] of Object.entries(declared)) {
                        if (seen.has(n)) continue;
                        list.appendChild(buildRow({ name: n, desc: d, entries: [], count: 0 }));
                    }
                    if (!list.children.length) list.appendChild(buildRow());
                    commit();
                }

                function rebuildFromWidgetOnly() {
                    list.innerHTML = "";
                    const declared = readCategoriesJSON(jsonWidget);
                    for (const [n, d] of Object.entries(declared)) {
                        list.appendChild(buildRow({ name: n, desc: d }));
                    }
                }

                function renderReport(text) {
                    const txt = (text || "").trim();
                    if (txt) {
                        reportWrap.classList.add("lwm-open");
                        reportTA.value = txt;
                        const lines = txt.split("\n");
                        reportMeta.textContent = lines[0]?.trim() || "";
                    } else {
                        reportWrap.classList.remove("lwm-open");
                        reportTA.value = "";
                        reportMeta.textContent = "";
                    }
                }

                async function refreshFromServer() {
                    try {
                        const resp = await fetch("/llm_wildcard/state");
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        const snap = await resp.json();
                        rebuildFromSnapshot(snap);
                        renderReport(snap.report);
                    } catch (e) {
                        console.warn("[LLMWildcardManager] refresh failed:", e);
                        pathLine.textContent = `wildcards: refresh failed (${e.message})`;
                    }
                }
                refreshBtn.addEventListener("click", refreshFromServer);
                addBtn.addEventListener("click", () => {
                    list.appendChild(buildRow());
                    commit();
                });

                rebuildFromWidgetOnly();

                node.addDOMWidget("manager_view", "div", root, { serialize: false });
                node.size = [Math.max(node.size[0], 560), Math.max(node.size[1], 640)];

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
                        renderReport(snap.report);
                    } catch (e) {
                        console.warn("[LLMWildcardManager] bad snapshot:", e);
                    }
                };
            };
        }
    },
});
