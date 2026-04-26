"""
LLM Wildcard Resolver for ComfyUI
---------------------------------
Resolves __wildcard__ slots in a template by either reusing values from
disk-backed wildcard files or generating new ones with an LLM.

Key design point: the LLM is called PER SLOT in isolation. It never sees
the surrounding prompt, only the category name and the list of forbidden
(existing) values. This prevents the LLM from anchoring on the base prompt
and producing the same enhancement every time.

Compatible with the standard ComfyUI/wildcards/ folder layout used by
Impact Pack and Santodan's Wildcard Manager.
"""

import os
import re
import json
import random
import urllib.request
import urllib.error
from pathlib import Path

try:
    import folder_paths  # ComfyUI module
    COMFY_BASE = Path(folder_paths.base_path)
except Exception:
    # Fallback for dev: assume this file lives in ComfyUI/custom_nodes/<pkg>/
    COMFY_BASE = Path(__file__).resolve().parents[2]

WILDCARDS_DIR = COMFY_BASE / "wildcards"
WILDCARDS_DIR.mkdir(parents=True, exist_ok=True)

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "wildcard_categories.json"

# -----------------------------------------------------------------------------
# Category descriptions guide the LLM. Each one is a short, neutral instruction
# describing what shape of value belongs in that wildcard. You can edit the
# JSON file or pass overrides through the node input.
# -----------------------------------------------------------------------------
DEFAULT_CATEGORIES = {
    "hair": "A short visual description of a person's hair: style, length, and color. One concise phrase, no leading article.",
    "ethnicity": "A single ethnicity or heritage descriptor for a portrait subject. One or two words.",
    "age": "An age descriptor for a portrait subject, e.g. 'young woman in her 20s' or 'middle-aged woman'.",
    "activity": "A single sport or active activity, e.g. 'jogging', 'practicing yoga', 'rock climbing'.",
    "location": "A single outdoor or indoor location suitable for a photoshoot.",
    "time": "A time-of-day or natural lighting condition.",
    "weather": "A weather condition, one phrase.",
    "outfit": "A complete outfit description appropriate to athletic or casual contexts.",
    "pose": "A pose or body-language description, one phrase.",
    "style": "A photographic or illustration style modifier set, comma-separated."
}


def load_category_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[LLMWildcardResolver] Could not parse {CONFIG_PATH}: {e}")
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CATEGORIES, indent=2), encoding="utf-8")
    return dict(DEFAULT_CATEGORIES)


# -----------------------------------------------------------------------------
# Wildcard file I/O
# -----------------------------------------------------------------------------
def read_wildcard_file(name: str) -> list[str]:
    path = WILDCARDS_DIR / f"{name}.txt"
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def append_wildcard(name: str, value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    existing = read_wildcard_file(name)
    if any(e.lower() == value.lower() for e in existing):
        return False
    path = WILDCARDS_DIR / f"{name}.txt"
    needs_newline = path.exists() and path.stat().st_size > 0 \
        and not path.read_text(encoding="utf-8").endswith("\n")
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(value + "\n")
    return True


# -----------------------------------------------------------------------------
# LLM backends — kept dependency-free using urllib
# -----------------------------------------------------------------------------
def _http_post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def call_ollama(endpoint: str, model: str, system: str, user: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    url = endpoint.rstrip("/") + "/api/chat"
    body = _http_post_json(url, payload, {"Content-Type": "application/json"})
    return body.get("message", {}).get("content", "").strip()


def call_openai_compatible(endpoint: str, model: str, system: str, user: str,
                           temperature: float, api_key: str) -> str:
    # llama.cpp / LM Studio / vLLM ignore "model" or accept any string when only
    # one model is loaded — but the field must still be present to satisfy the
    # OpenAI schema. Send a sentinel if the user left it blank.
    payload = {
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = endpoint.rstrip("/") + "/chat/completions"
    body = _http_post_json(url, payload, headers)
    return body["choices"][0]["message"]["content"].strip()


# -----------------------------------------------------------------------------
# The core: ask the LLM for ONE new wildcard value, with strict isolation
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You generate a single concise wildcard value for a stable-diffusion prompt category.\n"
    "Strict rules:\n"
    "1. Output ONLY the value itself. No preamble, no quotes, no markdown, no explanation, no trailing period.\n"
    "2. The value MUST NOT match or paraphrase any item in the forbidden list.\n"
    "3. Stay strictly within the category meaning provided.\n"
    "4. Keep it short — a phrase, not a sentence.\n"
    "5. You are isolated from any larger prompt context. Do not invent unrelated content. "
    "Do not assume what the larger prompt is about beyond what the category meaning says.\n"
    "6. Anatomy and proportions, when relevant, must be plausible and natural."
)


# -----------------------------------------------------------------------------
# Direction presets — pre-baked "flair" lines so users don't have to write
# steering text by hand. The Manager exposes them as autocomplete suggestions
# but accepts free text too: anything that isn't a known preset key is treated
# as raw steering text.
# -----------------------------------------------------------------------------
DIRECTION_PRESETS: dict[str, str] = {
    "none": "",
    "photoreal": "Strict photorealism. Plausible anatomy, realistic light, no stylization.",
    "cinematic": "Cinematic framing, dramatic lighting, filmic colour grading. Avoid clichés.",
    "editorial": "Editorial fashion photography aesthetic, polished and magazine-grade.",
    "vintage_film": "Vintage analog film look — 70s/80s tones, soft grain, slightly faded.",
    "noir": "Film noir — high-contrast monochrome feel, shadow play, mystery.",
    "cyberpunk": "Lean cyberpunk neon noir, dystopian futurism, no clichés.",
    "fantasy": "High fantasy setting, painterly atmosphere, mythic feel.",
    "anime": "Stylized anime/manga aesthetic, clean lineart, expressive features.",
    "dreamlike": "Surreal dreamlike atmosphere, ethereal, soft-focus.",
    "minimal": "Minimal, restrained palette and composition. Nothing busy.",
    "sfw_strict": "Keep all output strictly SFW. No suggestive phrasing.",
}


def resolve_direction(direction: str) -> str:
    """Map a direction string to its steering text.

    Known preset keys expand to their canonical preset text. Anything else is
    returned verbatim — so users can type custom steering directly into the
    direction field without needing to fall back to ``extra_flair``.
    """
    if direction is None:
        return ""
    key = direction.strip()
    if not key:
        return ""
    if key in DIRECTION_PRESETS:
        return DIRECTION_PRESETS[key]
    return key


# -----------------------------------------------------------------------------
# Report relay — the Resolver writes its latest report here on every run so
# the Manager can show it without needing a graph edge back from the Resolver
# (ComfyUI graphs are acyclic).
# -----------------------------------------------------------------------------
LAST_REPORT_PATH = WILDCARDS_DIR / ".last_report.txt"


def write_last_report(text: str) -> None:
    try:
        LAST_REPORT_PATH.write_text(text or "", encoding="utf-8")
    except Exception as e:
        print(f"[LLMWildcard] Could not write last report: {e}")


def read_last_report() -> str:
    if not LAST_REPORT_PATH.exists():
        return ""
    try:
        return LAST_REPORT_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Disk snapshot — used by the Manager UI to show every category and its entries
# -----------------------------------------------------------------------------
def list_disk_categories() -> dict[str, list[str]]:
    """Return every wildcard file currently on disk: {name: [values...]}.

    Skips dotfiles and any file that isn't ``.txt``.
    """
    out: dict[str, list[str]] = {}
    if not WILDCARDS_DIR.exists():
        return out
    for p in sorted(WILDCARDS_DIR.glob("*.txt")):
        if p.name.startswith("."):
            continue
        out[p.stem] = read_wildcard_file(p.stem)
    return out


def build_manager_snapshot(categories: dict, direction: str = "none",
                           extra_flair: str = "",
                           report: str | None = None) -> dict:
    """Snapshot the manager state for the JS frontend."""
    disk = list_disk_categories()
    # Union of: declared categories + DEFAULT_CATEGORIES + categories that
    # already exist on disk. The UI shows them all so nothing is hidden.
    names = sorted(set(categories) | set(DEFAULT_CATEGORIES) | set(disk))
    rows = []
    for name in names:
        rows.append({
            "name": name,
            "description": categories.get(name, DEFAULT_CATEGORIES.get(name, "")),
            "entries": disk.get(name, []),
            "count": len(disk.get(name, [])),
            "on_disk": name in disk,
        })
    # If no explicit report was passed in, fall back to the on-disk relay so
    # the Manager always shows the latest report after a Resolver run.
    final_report = (report or "").strip() or read_last_report()
    return {
        "wildcards_dir": str(WILDCARDS_DIR),
        "direction": direction,
        "direction_text": resolve_direction(direction),
        "direction_presets": DIRECTION_PRESETS,
        "extra_flair": extra_flair,
        "report": final_report,
        "rows": rows,
    }


# -----------------------------------------------------------------------------
# Optional ComfyUI server endpoint so the Manager UI can refresh disk state
# without needing to re-queue the workflow. Safe no-op if the import fails.
# -----------------------------------------------------------------------------
try:
    from server import PromptServer  # type: ignore
    from aiohttp import web as _aiohttp_web  # type: ignore

    @PromptServer.instance.routes.get("/llm_wildcard/state")
    async def _llm_wildcard_state(_request):
        return _aiohttp_web.json_response(
            build_manager_snapshot(load_category_config())
        )
except Exception:  # pragma: no cover — running outside ComfyUI
    pass


def _clean_llm_value(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    # take only first line; many models add a second "explanation" line
    raw = raw.splitlines()[0].strip()
    # strip surrounding quotes / backticks
    raw = raw.strip('"').strip("'").strip("`")
    # strip leading bullet / numbering
    raw = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw)
    # strip trailing period
    if raw.endswith("."):
        raw = raw[:-1].rstrip()
    return raw


def llm_generate_option(category: str, description: str, existing: list[str],
                        backend: str, endpoint: str, model: str, api_key: str,
                        temperature: float, system_prompt: str | None = None) -> str:
    sys_p = system_prompt if (system_prompt and system_prompt.strip()) else SYSTEM_PROMPT
    forbidden = "\n".join(f"- {e}" for e in existing) if existing else "(none yet)"
    user = (
        f"Category: {category}\n"
        f"Category meaning: {description}\n\n"
        f"Forbidden values (do NOT repeat or paraphrase any of these):\n{forbidden}\n\n"
        f"Output exactly one new value for the '{category}' category."
    )
    if backend == "ollama":
        raw = call_ollama(endpoint, model, sys_p, user, temperature)
    else:
        # "openai_compatible" and "llamacpp" share the same wire protocol.
        raw = call_openai_compatible(endpoint, model, sys_p, user, temperature, api_key)
    return _clean_llm_value(raw)


# -----------------------------------------------------------------------------
# Wildcard syntax:
#   __name__   -> use stored value (or generate if file empty / mode says so)
#   __!name__  -> force generate a NEW value, append to file
# -----------------------------------------------------------------------------
WILDCARD_RE = re.compile(r"__(!)?([A-Za-z0-9_\-]+)__")


def format_report(records: list[dict], flair: str = "",
                  using_custom_prompt: bool = False) -> str:
    """Render structured per-slot records into a human-readable report."""
    if not records:
        return "(no wildcards in template)"

    # header tally
    tallies = {"generated_new": 0, "generated_duplicate": 0, "reused": 0,
               "cap_reached": 0, "error": 0}
    for r in records:
        tallies[r.get("status", "error")] = tallies.get(r.get("status", "error"), 0) + 1
    head = (
        f"generated(new): {tallies['generated_new']}   "
        f"generated(dup): {tallies['generated_duplicate']}   "
        f"reused: {tallies['reused']}   "
        f"cap-reused: {tallies['cap_reached']}   "
        f"errors: {tallies['error']}   "
        f"total: {len(records)}"
    )
    meta = []
    if using_custom_prompt:
        meta.append("system_prompt: CUSTOM (from PromptConfig)")
    else:
        meta.append("system_prompt: default")
    if flair:
        meta.append(f"flair: {flair!r}")

    lines = [head, *meta, "=" * 64]

    for r in records:
        lines.append(
            f"[{r['name']}]  status={r['status']}  value={r.get('value','')!r}"
        )
        if "pool_size" in r:
            lines.append(f"    pool       : {r['pool_size']} known values on disk")
        if "sent" in r:
            lines.append(f"    sent → LLM : {r['sent']}")
        if "raw" in r:
            lines.append(f"    LLM reply  : {r['raw']!r}")
        if "retry_sent" in r:
            lines.append(f"    retry sent : {r['retry_sent']}")
        if "retry_raw" in r:
            lines.append(f"    retry reply: {r['retry_raw']!r}")
        if "err" in r:
            lines.append(f"    error      : {r['err']}")
        lines.append("")  # blank line between blocks

    return "\n".join(lines).rstrip()


class LLMWildcardResolver:
    """ComfyUI node: resolves __wildcard__ slots via cache + LLM with anti-repetition."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Digital illustration of a __age__ __ethnicity__ woman with __hair__, "
                        "__activity__ at a __location__, __time__, wearing __outfit__, "
                        "__pose__, __style__, masterpiece, best quality, ultra-detailed"
                    ),
                }),
                "mode": (["reuse_existing", "force_new", "hybrid"], {"default": "hybrid"}),
                "backend": (["ollama", "llamacpp", "openai_compatible"], {"default": "ollama"}),
                # Endpoint examples by backend:
                #   ollama            -> http://localhost:11434
                #   llamacpp          -> http://localhost:8080/v1
                #   openai_compatible -> https://api.openai.com/v1
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),
                "model": ("STRING", {"default": "llama3.1"}),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_per_category": ("INT", {"default": 200, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "category_overrides": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": '{"hair": "Short phrase: hair style + length + color"}',
                }),
                "prompts": ("WILDCARD_PROMPTS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("resolved_prompt", "report")
    FUNCTION = "resolve"
    CATEGORY = "prompt/wildcards"

    # --- mode semantics ------------------------------------------------------
    # reuse_existing : always pick from file; only generate if file is empty
    # force_new      : every slot is regenerated and appended
    # hybrid         : pick from file by default, but __!name__ forces new
    # -------------------------------------------------------------------------

    def resolve(self, template, mode, backend, endpoint, model, api_key,
                temperature, max_per_category, seed,
                category_overrides="", prompts=None):
        rng = random.Random(seed if seed != 0 else None)
        categories = load_category_config()

        # Merge order: defaults < prompts node overrides < inline JSON input
        custom_system_prompt: str | None = None
        flair_text: str = ""
        if isinstance(prompts, dict):
            custom_system_prompt = prompts.get("system_prompt") or None
            flair_text = prompts.get("flair") or ""
            cfg_overrides = prompts.get("category_overrides") or {}
            if isinstance(cfg_overrides, dict):
                categories.update(cfg_overrides)

        if category_overrides.strip():
            try:
                categories.update(json.loads(category_overrides))
            except Exception as e:
                print(f"[LLMWildcardResolver] Bad category_overrides JSON: {e}")

        records: list[dict] = []

        def resolve_slot(match: "re.Match") -> str:
            force_flag = match.group(1)
            name = match.group(2)
            force_new = (force_flag == "!") or (mode == "force_new")
            existing = read_wildcard_file(name)
            description = categories.get(
                name, f"A value for the '{name}' wildcard category."
            )

            rec: dict = {"name": name, "pool_size": len(existing)}

            # Path 1: reuse from file
            should_reuse = (not force_new) and existing and (mode != "force_new")
            if should_reuse:
                value = rng.choice(existing)
                rec.update({"status": "reused", "value": value})
                records.append(rec)
                return value

            # Path 2: generate new — but bail if we're at cap
            if len(existing) >= max_per_category:
                value = rng.choice(existing) if existing else f"[{name}]"
                rec.update({"status": "cap_reached", "value": value})
                records.append(rec)
                return value

            sent = (
                f'category="{name}" | desc={description!r} | '
                f'forbidden={len(existing)} items | model={model!r} | temp={temperature}'
            )
            rec["sent"] = sent
            try:
                value = llm_generate_option(
                    name, description, existing,
                    backend, endpoint, model, api_key, temperature,
                    system_prompt=custom_system_prompt,
                )
                rec["raw"] = value
                if not value:
                    raise RuntimeError("empty LLM response")

                # one retry if the model ignored the forbidden list
                if existing and any(e.lower() == value.lower() for e in existing):
                    bumped = min(2.0, float(temperature) + 0.3)
                    rec["retry_sent"] = f"temp={bumped} (after duplicate)"
                    retry = llm_generate_option(
                        name, description, existing + [value],
                        backend, endpoint, model, api_key, bumped,
                        system_prompt=custom_system_prompt,
                    )
                    rec["retry_raw"] = retry
                    if retry and not any(e.lower() == retry.lower() for e in existing):
                        value = retry

                appended = append_wildcard(name, value)
                rec.update({
                    "status": "generated_new" if appended else "generated_duplicate",
                    "value": value,
                })
                records.append(rec)
                return value

            except Exception as e:
                fallback = rng.choice(existing) if existing else f"__{name}__"
                rec.update({"status": "error", "err": str(e), "value": fallback})
                records.append(rec)
                return fallback

        resolved = WILDCARD_RE.sub(resolve_slot, template)
        report = format_report(records, flair=flair_text,
                               using_custom_prompt=bool(custom_system_prompt))
        # Relay the report to disk so the Manager can pick it up without an
        # explicit graph edge (which would form a cycle).
        write_last_report(report)
        return (resolved, report)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # seed=0 means "fresh roll every run"; non-zero is reproducible
        if kwargs.get("seed", 0) == 0:
            return float("nan")
        return json.dumps(kwargs, sort_keys=True, default=str)


# =============================================================================
# Node 2: LLMWildcardPromptConfig
# Lets the user override the LLM system prompt and define category descriptions
# in the graph instead of editing JSON files. Output is a single bundle that
# plugs into LLMWildcardResolver's optional `prompts` socket.
# =============================================================================
class LLMWildcardPromptConfig:
    """ComfyUI node: bundle a flair direction + optional system-prompt override
    + category descriptions for the LLM. Plug into LLMWildcardResolver `prompts`.

    UI: the `category_overrides` widget is rendered as a clickable add/remove
    table by web/llm_wildcard.js. The underlying value is JSON, so the node
    still works headless (e.g. running workflows via the API).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flair": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Optional steering text appended to the LLM system prompt.\n"
                        "Example: 'Lean cyberpunk neon noir, no clichés.'"
                    ),
                }),
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Leave empty to use the built-in system prompt. "
                        "Fill to fully replace it (advanced)."
                    ),
                }),
                "category_overrides": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "Edited via the table UI on the node.",
                }),
            },
        }

    RETURN_TYPES = ("WILDCARD_PROMPTS",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "build"
    CATEGORY = "prompt/wildcards"

    def build(self, flair, system_prompt_override, category_overrides):
        flair = (flair or "").strip()
        override = (system_prompt_override or "").strip()
        base = override if override else SYSTEM_PROMPT
        effective = base + (
            f"\n\nAdditional direction from the user:\n{flair}" if flair else ""
        )

        cats: dict = {}
        text = (category_overrides or "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    cats = {str(k): str(v) for k, v in parsed.items() if str(k).strip()}
                else:
                    print("[LLMWildcardPromptConfig] category_overrides must be a JSON object")
            except Exception as e:
                print(f"[LLMWildcardPromptConfig] Bad category_overrides JSON: {e}")

        return ({
            "system_prompt": effective,
            "flair": flair,
            "category_overrides": cats,
        },)


# =============================================================================
# Node 3: LLMWildcardReport
# Displays the resolver's report inside the node body (via web/llm_wildcard.js)
# and re-emits it plus parsed counters as outputs.
# =============================================================================
_REPORT_HEAD_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+status=(?P<status>\S+)"
)


class LLMWildcardReport:
    """ComfyUI node: parse the resolver's report into stats and display it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("summary", "generated", "reused", "errors", "total")
    FUNCTION = "parse"
    CATEGORY = "prompt/wildcards"
    OUTPUT_NODE = True

    def parse(self, report):
        generated = reused = errors = total = 0
        for raw in (report or "").splitlines():
            m = _REPORT_HEAD_RE.match(raw.strip())
            if not m:
                continue
            total += 1
            s = m.group("status")
            if s.startswith("generated"):
                generated += 1
            elif s.startswith("reused") or s.startswith("cap"):
                reused += 1
            elif s == "error":
                errors += 1

        summary = (report or "(no report)").strip()
        # web/llm_wildcard.js listens for `text` on this node and renders it.
        return {
            "ui": {"text": [summary]},
            "result": (summary, generated, reused, errors, total),
        }


# =============================================================================
# Node 4: LLMWildcardManager
# Central place to manage everything: category descriptions, generated entries
# (read live from disk), a "direction" preset (so the user doesn't have to
# write steering text by hand), and an optional report passthrough that gets
# rendered inside the same node.
#
# Wire it into the Resolver's `prompts` socket — it is a drop-in replacement
# for LLMWildcardPromptConfig and provides the same WILDCARD_PROMPTS bundle.
# =============================================================================
class LLMWildcardManager:
    """ComfyUI node: central management for wildcard categories, entries,
    direction presets, and (optionally) the resolver's report.

    UI features (rendered by web/llm_wildcard.js):
      * Direction preset combo — fills in steering text for the LLM.
      * Categories table — name + description per row, with an expand button
        that shows the entries currently on disk for that category.
      * "Refresh from disk" button — re-fetches the disk snapshot via the
        /llm_wildcard/state endpoint so newly-generated entries appear without
        needing to re-queue the workflow.
      * If the optional `report` input is connected, the latest report is
        rendered inside the node body too.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Free-text STRING (rendered as autocomplete in the UI).
                # Known preset keys expand to their canonical text; anything
                # else is used as-is. Lets the user type custom directions.
                "direction": ("STRING", {
                    "default": "none",
                    "placeholder": "preset key (e.g. 'cinematic') or any custom steering text",
                }),
                "extra_flair": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Optional extra steering, appended after the direction.\n"
                        "Leave empty to use only the direction value."
                    ),
                }),
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Leave empty to use the built-in system prompt. "
                        "Fill to fully replace it (advanced)."
                    ),
                }),
                "categories": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "Edited via the table UI on the node.",
                }),
            },
            "optional": {
                "report": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("WILDCARD_PROMPTS", "STRING")
    RETURN_NAMES = ("prompts", "summary")
    FUNCTION = "manage"
    CATEGORY = "prompt/wildcards"
    OUTPUT_NODE = True

    def manage(self, direction, extra_flair, system_prompt_override,
               categories, report=None):
        direction = (direction or "").strip() or "none"
        preset = resolve_direction(direction)
        extra = (extra_flair or "").strip()
        flair_parts = [p for p in (preset, extra) if p]
        flair = "\n".join(flair_parts)

        override = (system_prompt_override or "").strip()
        base = override if override else SYSTEM_PROMPT
        effective = base + (
            f"\n\nAdditional direction from the user:\n{flair}" if flair else ""
        )

        cats: dict = {}
        text = (categories or "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    cats = {str(k): str(v) for k, v in parsed.items()
                            if str(k).strip()}
                else:
                    print("[LLMWildcardManager] categories must be a JSON object")
            except Exception as e:
                print(f"[LLMWildcardManager] Bad categories JSON: {e}")

        # Persist current categories to disk so the Resolver-only path picks
        # them up too. We merge with whatever is already there so we never
        # lose entries the user added through the legacy PromptConfig flow.
        merged_disk = load_category_config()
        merged_disk.update(cats)
        try:
            CONFIG_PATH.write_text(
                json.dumps(merged_disk, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[LLMWildcardManager] Could not persist categories: {e}")

        bundle = {
            "system_prompt": effective,
            "flair": flair,
            "category_overrides": cats,
        }

        snapshot = build_manager_snapshot(
            merged_disk, direction=direction, extra_flair=extra, report=report,
        )
        summary = (report or "").strip()

        return {
            # web/llm_wildcard.js reads this to render the table + report.
            "ui": {"manager_state": [json.dumps(snapshot)]},
            "result": (bundle, summary),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so disk state is refreshed in the UI on every run.
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "LLMWildcardResolver": LLMWildcardResolver,
    "LLMWildcardPromptConfig": LLMWildcardPromptConfig,
    "LLMWildcardReport": LLMWildcardReport,
    "LLMWildcardManager": LLMWildcardManager,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMWildcardResolver": "🎲 LLM Wildcard Resolver",
    "LLMWildcardPromptConfig": "🎲 LLM Wildcard Prompt Config",
    "LLMWildcardReport": "🎲 LLM Wildcard Report",
    "LLMWildcardManager": "🎲 LLM Wildcard Manager",
}
