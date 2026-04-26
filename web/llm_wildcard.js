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
                    });
                }
                function buildRow(name = "", desc = "") {
                    const row = document.createElement("div");
                    Object.assign(row.style, {
                        display: "flex",
                        gap: "4px",
                        alignItems: "center",
                    });

                    const nameI = document.createElement("input");
                    nameI.type = "text";
                    nameI.placeholder = "name";
                    nameI.value = name;
                    nameI.style.flex = "0 0 32%";
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
                const node = this;

                const jsonWidget = node.widgets.find(w => w.name === "categories");
                if (!jsonWidget) return;
                hideWidget(node, jsonWidget);

                // ----- shared styles -----
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
                    });
                }
                function styleBtn(el, bg = "#2b5d8a", hover = "#3873a8") {
                    Object.assign(el.style, {
                        padding: "5px 10px",
                        background: bg,
                        color: "#fff",
                        border: "none",
                        borderRadius: "3px",
                        cursor: "pointer",
                        fontSize: "12px",
                    });
                    el.addEventListener("mouseenter", () => el.style.background = hover);
                    el.addEventListener("mouseleave", () => el.style.background = bg);
                }

                // ----- root layout -----
                const root = document.createElement("div");
                Object.assign(root.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                    padding: "4px",
                    boxSizing: "border-box",
                    width: "100%",
                });

                // top bar: header + refresh + add
                const topBar = document.createElement("div");
                Object.assign(topBar.style, {
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: "6px",
                });
                const header = document.createElement("div");
                header.textContent = "Categories — descriptions sent to the LLM + entries on disk";
                Object.assign(header.style, {
                    fontSize: "11px",
                    color: "#aaa",
                    flex: "1 1 auto",
                });
                const refreshBtn = document.createElement("button");
                refreshBtn.textContent = "↻ Refresh disk";
                refreshBtn.title = "Re-read entries from the wildcards/ folder";
                styleBtn(refreshBtn, "#3a3a3a", "#505050");
                const addBtn = document.createElement("button");
                addBtn.textContent = "+ Add category";
                styleBtn(addBtn);
                topBar.appendChild(header);
                topBar.appendChild(refreshBtn);
                topBar.appendChild(addBtn);
                root.appendChild(topBar);

                // disk-path indicator
                const pathLine = document.createElement("div");
                Object.assign(pathLine.style, {
                    fontSize: "10px",
                    color: "#666",
                    fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                });
                pathLine.textContent = "wildcards: (run once or click ↻ to load)";
                root.appendChild(pathLine);

                // table list
                const list = document.createElement("div");
                Object.assign(list.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                });
                root.appendChild(list);

                // report panel (hidden until a report comes in)
                const reportLabel = document.createElement("div");
                reportLabel.textContent = "Last report";
                Object.assign(reportLabel.style, {
                    fontSize: "11px",
                    color: "#aaa",
                    marginTop: "6px",
                    display: "none",
                });
                root.appendChild(reportLabel);

                const reportTA = document.createElement("textarea");
                reportTA.readOnly = true;
                reportTA.spellcheck = false;
                Object.assign(reportTA.style, {
                    width: "100%",
                    minHeight: "180px",
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
                    display: "none",
                });
                root.appendChild(reportTA);

                // ----- row-level helpers -----
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

                function buildRow({ name = "", desc = "", entries = null, count = 0,
                                    on_disk = false } = {}) {
                    const row = document.createElement("div");
                    Object.assign(row.style, {
                        display: "flex",
                        flexDirection: "column",
                        gap: "3px",
                        background: "#1c1e22",
                        border: "1px solid #2c2e32",
                        borderRadius: "3px",
                        padding: "4px",
                    });

                    const top = document.createElement("div");
                    Object.assign(top.style, {
                        display: "flex",
                        gap: "4px",
                        alignItems: "center",
                    });

                    const expandBtn = document.createElement("button");
                    expandBtn.textContent = "▸";
                    expandBtn.title = "Show entries on disk";
                    Object.assign(expandBtn.style, {
                        flex: "0 0 22px",
                        cursor: "pointer",
                        background: "#2a2d31",
                        color: "#ccc",
                        border: "1px solid #3a3d41",
                        borderRadius: "3px",
                        padding: "3px 0",
                        fontSize: "11px",
                    });

                    const nameI = document.createElement("input");
                    nameI.type = "text";
                    nameI.placeholder = "name";
                    nameI.value = name;
                    nameI.style.flex = "0 0 26%";
                    styleInput(nameI);
                    row._nameInput = nameI;

                    const descI = document.createElement("input");
                    descI.type = "text";
                    descI.placeholder = "description sent to the LLM";
                    descI.value = desc;
                    descI.style.flex = "1 1 auto";
                    styleInput(descI);
                    row._descInput = descI;

                    const badge = document.createElement("span");
                    badge.textContent = on_disk ? `${count}` : "·";
                    badge.title = on_disk
                        ? `${count} entries currently on disk`
                        : "no file on disk yet";
                    Object.assign(badge.style, {
                        flex: "0 0 36px",
                        textAlign: "center",
                        fontSize: "11px",
                        color: on_disk ? "#9ec5ff" : "#666",
                        fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                    });

                    const rmBtn = document.createElement("button");
                    rmBtn.textContent = "✕";
                    rmBtn.title = "Remove (does not delete the file on disk)";
                    Object.assign(rmBtn.style, {
                        flex: "0 0 26px",
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

                    top.appendChild(expandBtn);
                    top.appendChild(nameI);
                    top.appendChild(descI);
                    top.appendChild(badge);
                    top.appendChild(rmBtn);
                    row.appendChild(top);

                    // entries panel
                    const entriesPanel = document.createElement("div");
                    Object.assign(entriesPanel.style, {
                        display: "none",
                        background: "#15171a",
                        color: "#d8d8d8",
                        border: "1px solid #333",
                        borderRadius: "3px",
                        padding: "4px 6px",
                        marginLeft: "26px",
                        maxHeight: "180px",
                        overflow: "auto",
                        fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                        fontSize: "11px",
                        lineHeight: "1.35",
                        whiteSpace: "pre",
                    });
                    if (entries === null) {
                        entriesPanel.textContent = "(click ↻ Refresh disk to load entries)";
                    } else if (!entries.length) {
                        entriesPanel.textContent = "(no entries on disk yet)";
                    } else {
                        entriesPanel.textContent = entries.join("\n");
                    }
                    row.appendChild(entriesPanel);
                    row._entriesPanel = entriesPanel;

                    expandBtn.addEventListener("click", () => {
                        const open = entriesPanel.style.display !== "none";
                        entriesPanel.style.display = open ? "none" : "block";
                        expandBtn.textContent = open ? "▸" : "▾";
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

                    // First, render every row from the snapshot (authoritative)
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
                    // Then any locally-declared category that didn't come back
                    // from disk (e.g. just typed, not yet generated).
                    for (const [n, d] of Object.entries(declared)) {
                        if (seen.has(n)) continue;
                        list.appendChild(buildRow({ name: n, desc: d, entries: [], count: 0 }));
                    }
                    if (!list.children.length) {
                        list.appendChild(buildRow());
                    }
                    commit();
                }

                function rebuildFromWidgetOnly() {
                    list.innerHTML = "";
                    const declared = readCategoriesJSON(jsonWidget);
                    for (const [n, d] of Object.entries(declared)) {
                        list.appendChild(buildRow({ name: n, desc: d }));
                    }
                }

                // ----- refresh disk state -----
                async function refreshFromServer() {
                    try {
                        const resp = await fetch("/llm_wildcard/state");
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        const snap = await resp.json();
                        rebuildFromSnapshot(snap);
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
                node.size = [Math.max(node.size[0], 600), Math.max(node.size[1], 480)];

                // re-sync after loading a saved workflow, then pull live disk state
                const onConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    onConfigure?.apply(this, arguments);
                    setTimeout(() => {
                        rebuildFromWidgetOnly();
                        refreshFromServer();
                    }, 0);
                };

                // initial best-effort fetch (works after the page is wired up)
                setTimeout(refreshFromServer, 50);

                // ----- on execute: render snapshot + report from python -----
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    const raw = (message?.manager_state || [])[0];
                    if (raw) {
                        try {
                            const snap = JSON.parse(raw);
                            rebuildFromSnapshot(snap);
                            const txt = (snap.report || "").trim();
                            if (txt) {
                                reportLabel.style.display = "";
                                reportTA.style.display = "";
                                reportTA.value = txt;
                            } else {
                                reportLabel.style.display = "none";
                                reportTA.style.display = "none";
                            }
                        } catch (e) {
                            console.warn("[LLMWildcardManager] bad snapshot:", e);
                        }
                    }
                };
            };
        }
    },
});
