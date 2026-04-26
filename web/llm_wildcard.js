// Frontend extension for the LLM Wildcard Manager nodes.
//
//   * LLMWildcardReport          — render the resolver's report inside the node.
//   * LLMWildcardPromptConfig    — clickable add/remove rows for category overrides,
//                                   serialized into the underlying JSON STRING widget.
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
    },
});
