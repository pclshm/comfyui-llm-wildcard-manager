"""
LLM Wildcard Manager for ComfyUI
--------------------------------
A small set of nodes that compose like this:

    [LLMServerConfig] --server--> [LLMWildcardManager] --prompt_template--> [LLMWildcardResolver] --resolved_prompt--> ...
                              \\----server------------------------------------/                  \\---report---> [LLMWildcardReport]

The Manager calls the LLM ONCE to design a prompt template that contains
__wildcard__ placeholders, plus a description for each placeholder. The
Resolver then fills each placeholder by either reusing a value from disk or
asking the LLM for a fresh, anti-repetition value (one slot at a time, in
isolation, with the existing values listed as forbidden).

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
    COMFY_BASE = Path(__file__).resolve().parents[2]

WILDCARDS_DIR = COMFY_BASE / "wildcards"
WILDCARDS_DIR.mkdir(parents=True, exist_ok=True)

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "wildcard_categories.json"

LAST_REPORT_TXT = WILDCARDS_DIR / ".last_report.txt"
LAST_REPORT_JSON = WILDCARDS_DIR / ".last_report.json"
LAST_TEMPLATE_PATH = WILDCARDS_DIR / ".last_template.txt"
LAST_REPLY_PATH = WILDCARDS_DIR / ".last_reply.json"
LAST_RESOLVER_PATH = WILDCARDS_DIR / ".last_resolver.json"


# -----------------------------------------------------------------------------
# Default category descriptions. The Manager will accumulate LLM-suggested
# categories on top of these, and the user can override any of them via the UI.
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
    "style": "A photographic or illustration style modifier set, comma-separated.",
}


def load_category_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if str(k).strip()}
        except Exception as e:
            print(f"[LLMWildcard] Could not parse {CONFIG_PATH}: {e}")
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CATEGORIES, indent=2), encoding="utf-8")
    return dict(DEFAULT_CATEGORIES)


def save_category_config(merged: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[LLMWildcard] Could not persist categories: {e}")


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


def list_disk_categories() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not WILDCARDS_DIR.exists():
        return out
    for p in sorted(WILDCARDS_DIR.glob("*.txt")):
        if p.name.startswith("."):
            continue
        out[p.stem] = read_wildcard_file(p.stem)
    return out


# -----------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# -----------------------------------------------------------------------------
def _http_post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _call_ollama(endpoint: str, model: str, system: str, user: str, temperature: float,
                 request_json: bool = False, seed: int = 0,
                 json_schema: dict | None = None) -> str:
    options = {"temperature": float(temperature)}
    if seed:
        options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    if json_schema is not None:
        # Ollama 0.5+ accepts a full JSON schema in `format` and grammar-constrains
        # output to it. Stronger guarantee than the bare "json" mode below.
        payload["format"] = json_schema
    elif request_json:
        payload["format"] = "json"
    url = endpoint.rstrip("/") + "/api/chat"
    body = _http_post_json(url, payload, {"Content-Type": "application/json"})
    return body.get("message", {}).get("content", "").strip()


def _call_openai_compatible(endpoint: str, model: str, system: str, user: str,
                            temperature: float, api_key: str,
                            request_json: bool = False, seed: int = 0,
                            json_schema: dict | None = None,
                            grammar: str | None = None,
                            backend: str = "openai_compatible") -> str:
    payload = {
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
    }
    if seed:
        payload["seed"] = int(seed)

    # Enforcement path is mutually exclusive on llama.cpp: sending `grammar`
    # alongside `json_schema` (or `response_format`, which it converts to
    # json_schema internally) trips a "Cannot use both json_schema and grammar"
    # 500 server-side. So branch on backend:
    #   * llamacpp     → raw GBNF via `grammar` (most reliable across builds;
    #                    bypasses the schema-to-grammar converter, which has
    #                    historically produced permissive grammars).
    #   * openai_compat → `response_format` (+ top-level `json_schema` for the
    #                    rare older OpenAI-compat server that reads it natively;
    #                    real OpenAI ignores unknown fields).
    if backend == "llamacpp" and grammar:
        payload["grammar"] = grammar
    elif json_schema is not None:
        payload["json_schema"] = json_schema
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "wildcard_template",
                "schema": json_schema,
            },
        }
    elif request_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = endpoint.rstrip("/") + "/chat/completions"
    print(
        f"[LLMWildcard DEBUG] backend={backend!r} url={url!r} "
        f"payload_keys={sorted(payload.keys())} "
        f"grammar_set={'grammar' in payload} "
        f"json_schema_set={'json_schema' in payload} "
        f"response_format_set={'response_format' in payload}"
    )
    body = _http_post_json(url, payload, headers)
    return body["choices"][0]["message"]["content"].strip()


def _server_call(server: dict, system: str, user: str,
                 temperature_override: float | None = None,
                 request_json: bool = False, seed: int = 0,
                 json_schema: dict | None = None,
                 grammar: str | None = None) -> str:
    backend = server.get("backend", "ollama")
    endpoint = server.get("endpoint", "http://localhost:11434")
    model = server.get("model", "")
    api_key = server.get("api_key", "")
    temp = float(temperature_override
                 if temperature_override is not None
                 else server.get("temperature", 0.9))
    if backend == "ollama":
        # Ollama has no `grammar` field — schema is the only enforcement path.
        return _call_ollama(endpoint, model, system, user, temp,
                            request_json=request_json, seed=seed,
                            json_schema=json_schema)
    return _call_openai_compatible(endpoint, model, system, user, temp, api_key,
                                   request_json=request_json, seed=seed,
                                   json_schema=json_schema, grammar=grammar,
                                   backend=backend)


# -----------------------------------------------------------------------------
# Direction presets — pre-baked "flair" lines so users don't have to write
# steering text by hand. Free text is also accepted: anything that isn't a
# known preset key is treated as raw steering.
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
    if direction is None:
        return ""
    key = direction.strip()
    if not key:
        return ""
    if key in DIRECTION_PRESETS:
        return DIRECTION_PRESETS[key]
    return key


# -----------------------------------------------------------------------------
# System prompts — one per small, focused step. No "WRONG OUTPUTS" lists, no
# rule recitations: just say what to produce.
# -----------------------------------------------------------------------------
PARSE_NEGATIVE_SYSTEM_PROMPT = (
    "Classify each item in a user's negative-prompt list as one of two kinds.\n"
    " - axis_ban: the item names a variable attribute axis that must NOT "
    "become a wildcard placeholder. The downstream wildcardifier will refuse "
    "to create that placeholder. Examples: 'no age', 'no gender', "
    "'no ethnicity', 'no body type'. Output the snake_case axis name only "
    "(e.g. 'age', 'gender', 'body_type').\n"
    " - scene_ban: the item names a thing or quality that must NOT appear in "
    "the image (objects, props, atmosphere, composition, text). Output the "
    "item as a short phrase suitable for 'do not depict ___'. Examples: "
    "'no phone', 'no text', 'multiple images in one'.\n"
    "If unsure, prefer scene_ban — only mark as axis_ban when the item is "
    "clearly an attribute axis name (one or two words naming a dimension).\n"
    'Output JSON: {"axis_bans": ["age", "gender"], '
    '"scene_bans": ["phone", "text in the image", ...]}'
)

DRAFT_SYSTEM_PROMPT = (
    "Write one image-prompt sentence from the user's idea and direction. "
    "Concrete, visual, present tense. No preamble, no quotes — sentence only.\n"
    "If a list of forbidden scene elements is provided, the sentence MUST "
    "NOT depict or imply any of them (no clever rephrasings)."
)

WILDCARDIFY_SYSTEM_PROMPT = (
    "Rewrite the image-prompt sentence by replacing variable elements with "
    "__snake_case__ placeholders (double underscores on each side).\n"
    "Pick the most impactful variables: subject action, pose, lighting, "
    "mood, composition, props, materials, color palette, era markers — "
    "whatever fits the sentence.\n"
    "Stay within the placeholder count range you are given. To reach the "
    "minimum, wildcardify additional variables; to stay under the maximum, "
    "leave the rest as concrete words.\n"
    "If a list of forbidden placeholder names is provided, you MUST NOT "
    "create any of those placeholders. Keep that aspect as concrete words.\n"
    'Output JSON: {"prompt": "...with __placeholders__...", '
    '"categories": ["name1", "name2", ...]}'
)

DESCRIBE_SYSTEM_PROMPT = (
    "For each wildcard name, write one short phrase (no full sentence) "
    "describing what kind of value belongs in that slot, tied to the "
    "specific image prompt. Specific enough that off-topic values feel "
    "wrong, broad enough to allow variety.\n"
    "Avoid bland category-only definitions like 'an outfit'. Reference the "
    "tone, era, setting, or aesthetic when relevant.\n"
    "If a list of forbidden scene elements is provided, no description may "
    "invite values that introduce those elements.\n"
    'Output JSON: {"<name>": "<short description>", ...}'
)

ALIGN_SYSTEM_PROMPT = (
    "Smooth the grammar of the image prompt — articles, pluralization, "
    "joining words. Do NOT change, rephrase, or remove any descriptive "
    "phrase. Output the corrected sentence only."
)

LIST_SYSTEM_PROMPT = (
    "Generate distinct values for one image-prompt wildcard slot. Each "
    "value is a concise, specific phrase — not a sentence.\n"
    "Every value must fit the surrounding image prompt and respect its "
    "direction. Use the description to identify the dimensions of the "
    "value (e.g. hair = color + length + texture + style); spread entries "
    "across different dimensional combinations, do not return synonyms or "
    "near-paraphrases.\n"
    "Existing values are forbidden and signal which combinations are "
    "already covered — your new values must explore combinations the pool "
    "has not.\n"
    "If a list of forbidden scene elements is provided, no value may "
    "contain or imply any of them.\n"
    'Output JSON: {"values": ["...", "...", ...]}'
)


# Light per-step JSON schemas. No `pattern` constraints, no GBNF — failures
# surface as parse errors rather than getting masked by salvage paths.
PARSE_NEGATIVE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "axis_bans": {"type": "array", "items": {"type": "string"}},
        "scene_bans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["axis_bans", "scene_bans"],
}

WILDCARDIFY_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompt", "categories"],
}

DESCRIBE_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}

LIST_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "values": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["values"],
}


# -----------------------------------------------------------------------------
# JSON / value extraction helpers
# -----------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


_THINK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think\s*>", re.IGNORECASE)


def _balanced_json_object(source: str) -> dict | None:
    """Walk `source` and return the first balanced {...} that parses as a dict.

    Brace counting respects JSON string literals (so `{` inside a quoted value
    doesn't increase depth). Skipping past mismatched braces lets us recover
    when the model emits prose before the JSON.
    """
    i = 0
    n = len(source)
    while i < n:
        start = source.find("{", i)
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(start, n):
            ch = source[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            return None
        candidate = source[start:end + 1]
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        i = start + 1
    return None


def _extract_json_object(text: str) -> dict | None:
    """Try hard to pull a JSON object out of an LLM reply.

    Order: strip <think> reasoning blocks, direct json.loads, fence-stripped
    json.loads, then balanced-brace extraction (handles prose-before-JSON
    and trailing prose-after-JSON without grabbing nested objects).
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text).strip()

    candidates = [cleaned, _strip_code_fences(cleaned)]
    if cleaned != text.strip():
        candidates.extend([text, _strip_code_fences(text)])
    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except Exception:
            pass

    for source in (cleaned, text):
        v = _balanced_json_object(source)
        if v is not None:
            return v
    return None


def _clean_llm_value(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    raw = raw.splitlines()[0].strip()
    raw = raw.strip('"').strip("'").strip("`")
    raw = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw)
    if raw.endswith("."):
        raw = raw[:-1].rstrip()
    return raw


# Prefixes a user typically writes in front of a forbidden trait. Stripped so
# the term itself is what we match against candidate values.
_NEG_PREFIXES = (
    "no ", "not ", "non ", "non-", "without ", "avoid ", "avoiding ",
    "exclude ", "excluding ", "never ", "anti ", "anti-",
)
_NEG_ARTICLES = ("the ", "a ", "an ")


def _negative_terms(text: str) -> list[str]:
    """Break a free-form negative prompt into a list of forbidden phrases.

    Each comma/semicolon/newline-delimited segment (also split on " and " /
    " or ") is treated as one phrase. Common leading negation words and
    articles are stripped so the user can write "no old people" or "old
    people" interchangeably. Phrases shorter than three characters are dropped
    to avoid stray-letter matches."""
    if not text or not text.strip():
        return []
    parts = re.split(r"[,;\n\r/|]+| and | or ", text.lower())
    terms: set[str] = set()
    for raw in parts:
        p = raw.strip().strip(".!?;:\"'`()[]{}")
        # Strip leading negation words (possibly stacked, e.g. "no the").
        changed = True
        while changed:
            changed = False
            for prefix in _NEG_PREFIXES:
                if p.startswith(prefix):
                    p = p[len(prefix):].strip()
                    changed = True
            for art in _NEG_ARTICLES:
                if p.startswith(art):
                    p = p[len(art):].strip()
                    changed = True
        if len(p) >= 3:
            terms.add(p)
    # Longest first so report output and any overlap checks see specific
    # phrases before their substrings.
    return sorted(terms, key=len, reverse=True)


def _value_violates_negative(value: str, terms: list[str]) -> bool:
    """True if `value` contains any forbidden phrase as a whole-word match.

    Uses an alnum-only word boundary so "old" doesn't match "gold" but does
    match "old-fashioned" or "an old man"."""
    if not value or not terms:
        return False
    v = value.lower()
    for t in terms:
        try:
            if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", v):
                return True
        except re.error:
            if t in v:
                return True
    return False


WILDCARD_RE = re.compile(r"__(!)?([A-Za-z0-9_\-]+)__")


def extract_wildcard_names(template: str) -> list[str]:
    seen = []
    for m in WILDCARD_RE.finditer(template or ""):
        n = m.group(2)
        if n not in seen:
            seen.append(n)
    return seen


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _to_snake_case(name: str) -> str:
    """Normalize an LLM-supplied wildcard name to lower snake_case."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "").strip().lower())
    s = s.strip("_")
    return s


def _parse_forbidden_names(raw: str) -> list[str]:
    """Split a user-supplied 'forbidden placeholders' string (comma- or
    newline-separated, with or without surrounding underscores) into a deduped
    list of bare snake_case names. e.g. '__age__, Ethnicity\\nbody type' →
    ['age', 'ethnicity', 'body_type']."""
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[,\n;]+", str(raw)):
        s = chunk.strip().strip("_").strip()
        if not s:
            continue
        n = _to_snake_case(s)
        if n and _KEY_RE.match(n) and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# -----------------------------------------------------------------------------
# Wildcard-format repair
# -----------------------------------------------------------------------------
def _trim_template_wildcards(template: str, keep_names) -> str:
    """Drop any `__name__` token in `template` whose name isn't in `keep_names`
    by replacing it with a humanized form (snake_case → spaced words). Used as
    a fallback when the LLM exceeds the user's max-categories cap; the
    grammar-align step at resolve time tidies the surrounding sentence."""
    keep = {_to_snake_case(n) for n in (keep_names or []) if str(n).strip()}

    def repl(m: "re.Match") -> str:
        name = m.group(2)
        if name in keep:
            return m.group(0)
        return name.replace("_", " ")

    return WILDCARD_RE.sub(repl, template or "")


def ensure_wildcard_format(template: str, known_names) -> str:
    """For each name in `known_names`, rewrite bare `name`, `_name_`, `_name`,
    or `name_` occurrences in `template` to `__name__`. Tokens already wrapped
    in `__...__` are left alone. Tokens not in `known_names` are left alone."""
    if not template or not known_names:
        return template or ""
    names = sorted({_to_snake_case(n) for n in known_names if str(n).strip()},
                   key=len, reverse=True)
    out = template
    for name in names:
        if not name:
            continue
        # (?<![A-Za-z0-9])  — left boundary excluding alnum (underscores allowed
        #                     so we can match `_name_`, `_name`, etc.).
        # _{0,2}            — optional 0/1/2 leading underscores.
        # name              — the literal category name.
        # _{0,2}            — optional 0/1/2 trailing underscores.
        # (?![A-Za-z0-9_])  — right boundary excluding alnum and underscore so
        #                     we don't chop into the middle of `___name___`.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])_{{0,2}}{re.escape(name)}_{{0,2}}(?![A-Za-z0-9_])"
        )
        out = pattern.sub(f"__{name}__", out)
    return out


# -----------------------------------------------------------------------------
# LLM operations — small, sequential steps. Failures raise; nothing is
# silently substituted with canned content.
# -----------------------------------------------------------------------------
class ManagerStepError(Exception):
    """Raised when one of the manager's small LLM steps returns unparseable
    output. Carries the step name and raw reply so the UI can show both."""
    def __init__(self, step: str, raw: str, message: str = ""):
        super().__init__(message or f"step '{step}' did not return parseable output")
        self.step = step
        self.raw = raw or ""


def llm_parse_negative(negative_prompt: str, server: dict, seed: int = 0,
                       ) -> tuple[list[str], list[str], str]:
    """Step 0 — split the raw negative-prompt text into structured bans.

    Returns (axis_bans, scene_bans, raw_reply). `axis_bans` is a list of
    snake_case attribute names that must NOT become wildcard placeholders
    (e.g. 'no age' → 'age'); `scene_bans` is a list of short phrases that
    must NOT appear in the image (e.g. 'no phone' → 'phone'). Empty input
    short-circuits to ([], [], "")."""
    text = (negative_prompt or "").strip()
    if not text:
        return [], [], ""
    user = (
        f"Negative prompt items:\n{text}\n\n"
        "Classify each item. Output the JSON object now."
    )
    raw = _server_call(server, PARSE_NEGATIVE_SYSTEM_PROMPT, user,
                       request_json=True, seed=seed,
                       json_schema=PARSE_NEGATIVE_JSON_SCHEMA)
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        # Fall back to treating everything as scene bans rather than failing
        # the whole pipeline — user can still correct via forbidden_placeholders.
        return [], [t for t in _negative_terms(text)], raw
    axis_raw = parsed.get("axis_bans") if isinstance(parsed, dict) else None
    scene_raw = parsed.get("scene_bans") if isinstance(parsed, dict) else None
    axis_bans: list[str] = []
    seen_axis: set[str] = set()
    if isinstance(axis_raw, list):
        for entry in axis_raw:
            n = _to_snake_case(entry if isinstance(entry, str) else "")
            if n and _KEY_RE.match(n) and n not in seen_axis:
                seen_axis.add(n)
                axis_bans.append(n)
    scene_bans: list[str] = []
    seen_scene: set[str] = set()
    if isinstance(scene_raw, list):
        for entry in scene_raw:
            s = str(entry or "").strip().strip('"').strip("'")
            if s and s.lower() not in seen_scene:
                seen_scene.add(s.lower())
                scene_bans.append(s)
    return axis_bans, scene_bans, raw


def _format_scene_bans(scene_bans: list[str]) -> str:
    """Render scene_bans as a bullet list for injection into user messages.
    Returns "" when the list is empty so callers can skip the section."""
    if not scene_bans:
        return ""
    return "\n".join(f"- {s}" for s in scene_bans)


def llm_draft_prompt(idea: str, direction_text: str, server: dict,
                     seed: int = 0, scene_bans: list[str] | None = None,
                     ) -> tuple[str, str]:
    """Step 1 — turn the user idea + direction into a single image-prompt
    sentence. `scene_bans` lists scene elements the sentence must avoid
    (already extracted from the raw negative prompt by llm_parse_negative).
    Returns (sentence, raw_reply)."""
    parts = [f"User idea:\n{(idea or '').strip() or '(no example provided)'}"]
    if direction_text and direction_text.strip():
        parts.append(f"Direction:\n{direction_text.strip()}")
    bans = _format_scene_bans(scene_bans or [])
    if bans:
        parts.append(f"Forbidden scene elements (must not appear):\n{bans}")
    if seed:
        parts.append(f"Variation token: {seed}")
    parts.append("Write the image-prompt sentence now.")
    user = "\n\n".join(parts)
    raw = _server_call(server, DRAFT_SYSTEM_PROMPT, user, seed=seed)
    sentence = (raw or "").strip()
    if len(sentence) >= 2 and sentence[0] in "\"'`" and sentence[-1] == sentence[0]:
        sentence = sentence[1:-1].strip()
    if not sentence:
        raise ManagerStepError("draft", raw, "empty draft prompt")
    return sentence, raw


def _build_wildcardify_cap_line(lo: int, hi: int) -> str:
    if lo and hi and lo == hi:
        return f"Use exactly {hi} placeholder{'' if hi == 1 else 's'}."
    if lo and hi:
        return f"Use between {lo} and {hi} placeholders."
    if hi:
        return f"Use at most {hi} placeholder{'' if hi == 1 else 's'}."
    if lo:
        return f"Use at least {lo} placeholder{'' if lo == 1 else 's'}."
    return ""


def _parse_wildcardify_reply(raw: str) -> tuple[str, list[str], list[str]]:
    """Parse one wildcardify LLM reply. Returns (template, names, in_template).
    Raises ManagerStepError if the reply is unparseable."""
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise ManagerStepError("wildcardify", raw)

    template = str(parsed.get("prompt") or "").strip()
    if not template:
        raise ManagerStepError("wildcardify", raw, "missing 'prompt' field")

    raw_categories = parsed.get("categories")
    declared: list[str] = []
    if isinstance(raw_categories, list):
        seen: set[str] = set()
        for entry in raw_categories:
            n = _to_snake_case(entry if isinstance(entry, str) else "")
            if n and _KEY_RE.match(n) and n not in seen:
                seen.add(n)
                declared.append(n)

    template = ensure_wildcard_format(template, declared)
    in_template = extract_wildcard_names(template)
    names: list[str] = []
    for n in declared + in_template:
        if n and n not in names:
            names.append(n)

    if not names:
        raise ManagerStepError(
            "wildcardify", raw,
            "no placeholders inserted and no categories listed",
        )
    return template, names, in_template


def llm_wildcardify_prompt(draft_prompt: str, server: dict,
                           seed: int = 0,
                           min_categories: int = 0,
                           max_categories: int = 0,
                           forbidden_names: list[str] | None = None,
                           ) -> tuple[str, list[str], str]:
    """Step 2 — rewrite the draft with __placeholders__ already inserted, plus
    the list of placeholder names. Returns (template, names, raw_reply).

    `forbidden_names` is the FULL deny-list of placeholder names (axis bans
    auto-derived from the negative prompt + any user-supplied bans). The LLM
    is told to avoid them; any that slip through are demoted to plain words
    deterministically after parsing — the LLM instruction alone is unreliable.

    `min_categories` is enforced via retry; `max_categories` via post-hoc
    demotion of surplus placeholders."""
    lo = max(0, int(min_categories or 0))
    hi = max(0, int(max_categories or 0))
    if lo and hi and lo > hi:
        lo = hi

    forbidden_set: set[str] = {n for n in (forbidden_names or []) if n}

    def _call(attempt: int, prev_template: str = "", prev_count: int = -1,
              prev_names: list[str] | None = None) -> tuple[str, list[str], list[str], str]:
        cap_line = _build_wildcardify_cap_line(lo, hi)
        parts = [f"Image prompt:\n{draft_prompt}"]
        if forbidden_set:
            listed = ", ".join(f"__{n}__" for n in sorted(forbidden_set))
            parts.append(
                "Forbidden placeholder names (must not appear):\n"
                f"{listed}"
            )
        if cap_line:
            parts.append(cap_line)
        if attempt > 0 and prev_count >= 0 and lo and prev_count < lo:
            parts.append(
                f"Previous attempt produced {prev_count} placeholder(s); "
                f"need at least {lo}. Keep them and add more.\n"
                f"Previous template:\n{prev_template}"
            )
        parts.append("Output the JSON object now.")
        user = "\n\n".join(parts)
        attempt_seed = seed + attempt * 9973 if seed else 0
        raw = _server_call(server, WILDCARDIFY_SYSTEM_PROMPT, user,
                           request_json=True, seed=attempt_seed,
                           json_schema=WILDCARDIFY_JSON_SCHEMA)
        template, names, in_template = _parse_wildcardify_reply(raw)
        if forbidden_set and any(n in forbidden_set for n in names):
            keep = [n for n in names if n not in forbidden_set]
            template = _trim_template_wildcards(template, keep)
            names = keep
            in_template = [n for n in in_template if n not in forbidden_set]
        return template, names, in_template, raw

    template, names, in_template, raw = _call(0)

    # Retry up to 2 times if the LLM falls short of the minimum. Beyond that,
    # accept what we have rather than spinning forever.
    retries = 0
    raw_log = [raw]
    while lo and len(names) < lo and retries < 2:
        retries += 1
        try:
            template, names, in_template, raw = _call(
                retries, prev_template=template, prev_count=len(names),
                prev_names=names,
            )
            raw_log.append(raw)
        except ManagerStepError as e:
            # Keep the last good result; surface the failed retry's raw.
            raw_log.append(e.raw or "")
            break

    # Hard enforcement of the cap when the LLM ignored the instruction. Keep
    # the first `max_categories` placeholders by appearance in the template
    # (then by declared order) and demote the rest to plain words so the
    # sentence still reads.
    if hi and len(names) > hi:
        ordered: list[str] = []
        for n in in_template + names:
            if n and n not in ordered:
                ordered.append(n)
        kept = ordered[:hi]
        template = _trim_template_wildcards(template, kept)
        names = [n for n in names if n in kept]

    combined_raw = raw_log[0] if len(raw_log) == 1 else "\n\n".join(
        f"--- attempt {i} ---\n{(r or '').strip()}"
        for i, r in enumerate(raw_log)
    )
    return template, names, combined_raw


def llm_describe_wildcards(names: list[str], server: dict,
                           seed: int = 0,
                           idea: str = "",
                           direction_text: str = "",
                           template: str = "",
                           scene_bans: list[str] | None = None,
                           ) -> tuple[dict[str, str], str]:
    """Step 3 — short shape-of-value description for each wildcard name.
    Returns (descriptions, raw_reply).

    `idea`, `direction_text`, and `template` anchor the descriptions to this
    specific prompt instead of producing generic 'an outfit' filler.

    `scene_bans` is the structured list of forbidden scene elements; passed
    so the descriptions don't accidentally invite values that reintroduce
    those elements."""
    if not names:
        return {}, ""
    parts: list[str] = []
    if idea and idea.strip():
        parts.append(f"User idea:\n{idea.strip()}")
    if direction_text and direction_text.strip():
        parts.append(f"Direction:\n{direction_text.strip()}")
    if template and template.strip():
        parts.append(f"Prompt template:\n{template.strip()}")
    bans = _format_scene_bans(scene_bans or [])
    if bans:
        parts.append(f"Forbidden scene elements (no description may invite them):\n{bans}")
    parts.append("Wildcard names:\n" + "\n".join(f"- {n}" for n in names))
    parts.append("Output the JSON object now.")
    user = "\n\n".join(parts)
    raw = _server_call(server, DESCRIBE_SYSTEM_PROMPT, user,
                       request_json=True, seed=seed,
                       json_schema=DESCRIBE_JSON_SCHEMA)
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise ManagerStepError("describe", raw)
    descs: dict[str, str] = {}
    for k, v in parsed.items():
        key = _to_snake_case(k)
        if key and isinstance(v, str) and v.strip():
            descs[key] = v.strip()
    return descs, raw


def llm_generate_value_list(category: str, description: str,
                            existing: list[str], server: dict,
                            count: int = 10, seed: int = 0,
                            template: str = "",
                            direction_text: str = "",
                            scene_bans: list[str] | None = None,
                            ) -> tuple[list[str], str]:
    """Resolver step — generate distinct values for one wildcard slot.

    `template`, `direction_text`, and `scene_bans` keep each value consistent
    with the overall image and away from forbidden scene elements. Returns
    (values, raw_reply)."""
    parts = [
        f"Category: {category}",
        f"Description: {description}",
    ]
    if template and template.strip():
        parts.append(f"Image prompt template:\n{template.strip()}")
    if direction_text and direction_text.strip():
        parts.append(f"Direction:\n{direction_text.strip()}")
    bans = _format_scene_bans(scene_bans or [])
    if bans:
        parts.append(f"Forbidden scene elements (no value may contain or imply them):\n{bans}")
    forbidden = ("\n".join(f"- {e}" for e in existing)
                 if existing else "(none yet)")
    parts.append(f"Already used (do not repeat):\n{forbidden}")
    parts.append(f"Produce {count} distinct new values. Output the JSON object now.")
    user = "\n\n".join(parts)
    raw = _server_call(server, LIST_SYSTEM_PROMPT, user,
                       request_json=True, seed=seed,
                       json_schema=LIST_JSON_SCHEMA)
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        return [], raw
    raw_values = parsed.get("values")
    if not isinstance(raw_values, list):
        return [], raw
    out: list[str] = []
    seen: set[str] = set()
    for v in raw_values:
        cleaned = _clean_llm_value(str(v) if v is not None else "")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out, raw


def llm_align_prompt(populated: str, server: dict, seed: int = 0
                     ) -> tuple[str, str]:
    """Resolver step — light grammar/article fix over the fully-populated
    sentence. Returns (aligned, raw_reply). Aligned text is verified by the
    caller to still contain every populated value verbatim before being used."""
    user = (
        f"Image prompt:\n{populated}\n\n"
        "Output the corrected sentence only."
    )
    raw = _server_call(server, ALIGN_SYSTEM_PROMPT, user, seed=seed)
    aligned = (raw or "").strip()
    if len(aligned) >= 2 and aligned[0] in "\"'`" and aligned[-1] == aligned[0]:
        aligned = aligned[1:-1].strip()
    return aligned, raw


# -----------------------------------------------------------------------------
# Report formatting + parsing
# -----------------------------------------------------------------------------
def format_report(records: list[dict], flair: str = "",
                  using_custom_prompt: bool = False) -> str:
    if not records:
        return "(no wildcards in template)"
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
    meta.append("system_prompt: CUSTOM" if using_custom_prompt else "system_prompt: default")
    if flair:
        meta.append(f"flair: {flair!r}")
    lines = [head, *meta, "=" * 64]
    for r in records:
        lines.append(f"[{r['name']}]  status={r['status']}  value={r.get('value', '')!r}")
        if "pool_size" in r:
            lines.append(f"    pool       : {r['pool_size']} known values on disk")
        if "blocked_by_negative" in r:
            lines.append(
                f"    neg-block  : {r['blocked_by_negative']} pool entries "
                "filtered out by negative prompt"
            )
        if "sent" in r:
            lines.append(f"    sent → LLM : {r['sent']}")
        if "raw" in r:
            lines.append(f"    LLM reply  : {r['raw']!r}")
        if "filtered_by_negative" in r:
            lines.append(
                f"    neg-drop   : {r['filtered_by_negative']} fresh values "
                "dropped for matching negative prompt"
            )
        if "retry_sent" in r:
            lines.append(f"    retry sent : {r['retry_sent']}")
        if "retry_raw" in r:
            lines.append(f"    retry reply: {r['retry_raw']!r}")
        if "err" in r:
            lines.append(f"    error      : {r['err']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def write_last_report(text: str, records: list[dict],
                      flair: str = "", using_custom_prompt: bool = False) -> None:
    try:
        LAST_REPORT_TXT.write_text(text or "", encoding="utf-8")
    except Exception as e:
        print(f"[LLMWildcard] Could not write last report text: {e}")
    payload = {
        "records": records,
        "flair": flair,
        "using_custom_prompt": using_custom_prompt,
        "tallies": _tally(records),
    }
    try:
        LAST_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[LLMWildcard] Could not write last report json: {e}")


def read_last_report_text() -> str:
    if not LAST_REPORT_TXT.exists():
        return ""
    try:
        return LAST_REPORT_TXT.read_text(encoding="utf-8")
    except Exception:
        return ""


def read_last_report_payload() -> dict | None:
    if not LAST_REPORT_JSON.exists():
        return None
    try:
        return json.loads(LAST_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tally(records: list[dict]) -> dict:
    t = {"generated": 0, "reused": 0, "errors": 0, "total": len(records)}
    for r in records:
        s = r.get("status", "error")
        if s.startswith("generated"):
            t["generated"] += 1
        elif s in ("reused", "cap_reached"):
            t["reused"] += 1
        elif s == "error":
            t["errors"] += 1
    return t


# -----------------------------------------------------------------------------
# Snapshot for the Manager UI
# -----------------------------------------------------------------------------
def build_manager_snapshot(effective_categories: dict, *,
                           direction: str = "none",
                           negative_prompt: str = "",
                           generated_prompt: str = "",
                           user_overrides: dict | None = None) -> dict:
    # The category list shown by the UI tracks the *current* template plus any
    # user override. We deliberately do NOT union in every disk file — that
    # would make the list look identical across runs regardless of the prompt.
    disk = list_disk_categories()
    user_overrides = user_overrides or {}
    names = sorted(set(effective_categories) | set(user_overrides))
    rows = []
    for name in names:
        rows.append({
            "name": name,
            "description": effective_categories.get(
                name, user_overrides.get(name, DEFAULT_CATEGORIES.get(name, ""))),
            "user_override": name in user_overrides,
            "entries": disk.get(name, []),
            "count": len(disk.get(name, [])),
            "on_disk": name in disk,
        })
    return {
        "wildcards_dir": str(WILDCARDS_DIR),
        "direction": direction,
        "direction_text": resolve_direction(direction),
        "direction_presets": DIRECTION_PRESETS,
        "negative_prompt": negative_prompt,
        "generated_prompt": generated_prompt,
        "rows": rows,
    }


# -----------------------------------------------------------------------------
# Optional ComfyUI server endpoint so the Manager UI can refresh disk state
# without re-queueing the workflow. Safe no-op if the import fails.
# -----------------------------------------------------------------------------
try:
    from server import PromptServer  # type: ignore
    from aiohttp import web as _aiohttp_web  # type: ignore

    @PromptServer.instance.routes.get("/llm_wildcard/state")
    async def _llm_wildcard_state(_request):
        all_cats = load_category_config()
        last_template = ""
        if LAST_TEMPLATE_PATH.exists():
            try:
                last_template = LAST_TEMPLATE_PATH.read_text(encoding="utf-8")
            except Exception:
                pass

        # Scope the rebuilt UI to wildcards in the last template, matching the
        # post-execute snapshot. Avoids the "same list every reload" bug.
        used = set(extract_wildcard_names(last_template))
        display_cats = {n: all_cats.get(n, DEFAULT_CATEGORIES.get(n, ""))
                        for n in sorted(used)}

        snap = build_manager_snapshot(display_cats, generated_prompt=last_template)

        if LAST_REPLY_PATH.exists():
            try:
                payload = json.loads(LAST_REPLY_PATH.read_text(encoding="utf-8"))
                snap["raw_reply"] = payload.get("raw_reply", "")
                snap["status"] = payload.get("status", "ok")
                snap["status_message"] = payload.get("status_message", "")
            except Exception:
                pass

        return _aiohttp_web.json_response(snap)

    @PromptServer.instance.routes.get("/llm_wildcard/last_report")
    async def _llm_wildcard_last_report(_request):
        payload = read_last_report_payload()
        if payload is None:
            payload = {"records": [], "tallies": _tally([]), "flair": "",
                       "using_custom_prompt": False}
        payload["text"] = read_last_report_text()
        return _aiohttp_web.json_response(payload)

    @PromptServer.instance.routes.get("/llm_wildcard/last_resolver")
    async def _llm_wildcard_last_resolver(_request):
        empty = {"template": "", "resolved": "", "records": [],
                 "tallies": _tally([])}
        if not LAST_RESOLVER_PATH.exists():
            return _aiohttp_web.json_response(empty)
        try:
            data = json.loads(LAST_RESOLVER_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = empty
        except Exception:
            data = empty
        return _aiohttp_web.json_response(data)
except Exception:  # pragma: no cover — running outside ComfyUI
    pass


# -----------------------------------------------------------------------------
# IS_CHANGED helper
# -----------------------------------------------------------------------------
def _seeded_is_changed(locked: bool, kwargs: dict):
    """If `locked` is True, return a stable hash of inputs (deterministic).
    If False, return NaN (always re-execute)."""
    if not locked:
        return float("nan")
    # The server bundle contains the temperature; serialize it too.
    return json.dumps(kwargs, sort_keys=True, default=str)


# =============================================================================
# Node 1: LLMServerConfig — single place to configure the LLM backend.
# =============================================================================
class LLMServerConfig:
    """ComfyUI node: bundle backend + endpoint + model + key + temperature.
    Wire `server` into both Manager and Resolver so settings live in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["ollama", "llamacpp", "openai_compatible"],
                            {"default": "ollama"}),
                # Endpoint examples by backend:
                #   ollama            -> http://localhost:11434
                #   llamacpp          -> http://localhost:8080/v1
                #   openai_compatible -> https://api.openai.com/v1
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),
                "model": ("STRING", {"default": "llama3.1"}),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0,
                                          "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LLM_SERVER",)
    RETURN_NAMES = ("server",)
    FUNCTION = "build"
    CATEGORY = "prompt/wildcards"

    def build(self, backend, endpoint, model, api_key, temperature):
        return ({
            "backend": backend,
            "endpoint": endpoint,
            "model": model,
            "api_key": api_key,
            "temperature": float(temperature),
        },)


# =============================================================================
# Node 2: LLMWildcardManager — designs the prompt template + suggests categories.
# =============================================================================
class LLMWildcardManager:
    """ComfyUI node: ask the LLM to turn the user's idea into a prompt template
    with __wildcard__ placeholders + a description for each placeholder.

    Outputs:
        prompt_template — STRING, wire into Resolver's `template`.
        prompts         — WILDCARD_PROMPTS bundle (system_prompt + flair +
                          merged category descriptions). Wire into Resolver.

    Seed semantics: seed=0 re-rolls every queue (new template, new categories);
    seed!=0 is reproducible (same inputs → same template + same categories)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLM_SERVER",),
                "example_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A portrait of a woman doing an outdoor activity, "
                        "photorealistic, masterpiece."
                    ),
                    "placeholder": (
                        "Your prompt idea. The Manager rewrites it as a template "
                        "with __wildcard__ placeholders for the variable parts."
                    ),
                }),
                # Skip the LLM entirely and reuse the last successful template +
                # categories. Lets the user re-queue to get fresh wildcard fills
                # from the Resolver without the Manager rewriting the prompt.
                "lock_template": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When ON, skip the LLM and reuse the last generated "
                        "prompt template + categories. Re-queue to get fresh "
                        "random wildcards from the Resolver without changing "
                        "the prompt."
                    ),
                }),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF}),
                "direction": ("STRING", {
                    "default": "none",
                    "placeholder": "preset key (e.g. 'cinematic') or any custom steering text",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Traits to AVOID — applied at every step.\n"
                        "1) The drafted sentence won't include them.\n"
                        "2) Aspects you pin here won't become wildcards "
                        "(e.g. 'no old or middle-aged people' stops "
                        "__age__ from being generated).\n"
                        "3) Each wildcard description gets an explicit "
                        "exclusion clause so the resolver can't drift into "
                        "them either.\n"
                        "One item per line or comma-separated."
                    ),
                    "tooltip": (
                        "What the LLM must avoid. Pinning an aspect here "
                        "(e.g. 'no old/middle-aged people' when the idea "
                        "is 'young woman') prevents the Manager from "
                        "creating a wildcard for that aspect AND keeps the "
                        "resolver's per-slot values from contradicting it."
                    ),
                }),
                # Hard deny-list of placeholder names. The LLM is told to
                # avoid them, and any that slip through are demoted to plain
                # words deterministically. Use this when the negative prompt
                # alone hasn't been enough to keep a specific dimension out
                # of the template.
                "forbidden_placeholders": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Placeholder names that must NEVER appear in the "
                        "template. Comma- or newline-separated.\n"
                        "Examples: age, ethnicity, body_type\n"
                        "(with or without surrounding underscores — "
                        "'__age__' and 'age' both work)."
                    ),
                    "tooltip": (
                        "Hard deny-list. Any placeholder name listed here "
                        "is stripped from the generated template and "
                        "demoted to concrete words, even if the LLM "
                        "ignored the instruction. Use this as a backstop "
                        "when the negative prompt isn't enough."
                    ),
                }),
                # Soft floor on how many `__wildcard__` placeholders the LLM
                # should introduce. Communicated to the LLM in the user
                # message; not enforced server-side because we can't promote
                # concrete words to placeholders without losing semantics.
                "min_categories": ("INT", {
                    "default": 3, "min": 1, "max": 30,
                    "display": "slider",
                    "tooltip": (
                        "Minimum number of __wildcard__ placeholders the "
                        "Manager will accept. Enforced by retrying the "
                        "wildcardify step if the LLM falls short. Higher = "
                        "more variation."
                    ),
                }),
                # Hard cap on how many `__wildcard__` placeholders the LLM may
                # introduce in the generated template. Keeps the prompt focused
                # — the model is told to pick the most impactful variables and
                # leave the rest as concrete words. If the model still exceeds
                # the cap, the surplus placeholders are demoted to plain words
                # deterministically.
                "max_categories": ("INT", {
                    "default": 8, "min": 1, "max": 30,
                    "display": "slider",
                    "tooltip": (
                        "Maximum number of __wildcard__ placeholders in the "
                        "generated template. Lower = more focused prompts."
                    ),
                }),
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "Leave empty to use the built-in template-design system prompt.\n"
                        "Fill to fully replace it (advanced)."
                    ),
                }),
                "categories": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "User overrides — edited via the table UI on the node.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "WILDCARD_PROMPTS")
    RETURN_NAMES = ("prompt_template", "prompts")
    FUNCTION = "manage"
    CATEGORY = "prompt/wildcards"
    OUTPUT_NODE = True

    def manage(self, server, example_prompt, lock_template, seed, direction,
               negative_prompt, forbidden_placeholders,
               min_categories, max_categories,
               system_prompt_override, categories):
        direction = (direction or "").strip() or "none"
        direction_text = resolve_direction(direction)
        negative = (negative_prompt or "").strip()
        forbidden_names = _parse_forbidden_names(forbidden_placeholders or "")

        # `system_prompt_override` is kept on the input for backward compat,
        # but the manager now drives four small calls — a single override
        # can't apply to all of them, so it's ignored here. The Resolver still
        # honours overrides via the bundle below.
        _ = system_prompt_override

        # User explicit overrides from the JSON widget (edited in the table UI)
        user_overrides: dict[str, str] = {}
        text = (categories or "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    user_overrides = {str(k): str(v) for k, v in parsed.items()
                                      if str(k).strip()}
                else:
                    print("[LLMWildcardManager] categories must be a JSON object")
            except Exception as e:
                print(f"[LLMWildcardManager] Bad categories JSON: {e}")

        # Effective sampling seed: 0 = fresh roll, non-zero = reproducible.
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = 0
        if seed_int == 0:
            effective_seed = random.SystemRandom().randrange(1, 2**31)
        else:
            effective_seed = seed_int

        try:
            max_cats = int(max_categories)
        except Exception:
            max_cats = 0
        if max_cats < 0:
            max_cats = 0
        try:
            min_cats = int(min_categories)
        except Exception:
            min_cats = 0
        if min_cats < 0:
            min_cats = 0
        if max_cats and min_cats > max_cats:
            min_cats = max_cats

        template = ""
        suggested_cats: dict[str, str] = {}
        used_names: list[str] = []
        scene_bans: list[str] = []
        status = "ok"
        status_message = ""
        raw_sections: list[tuple[str, str]] = []

        if lock_template:
            # Skip the LLM entirely. Reuse the last persisted template +
            # whatever category descriptions are already on disk. The Resolver
            # will still re-roll wildcard fills each queue (when fix_seed is
            # off), so the user gets a stable prompt with fresh randoms.
            cached = ""
            if LAST_TEMPLATE_PATH.exists():
                try:
                    cached = LAST_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
                except Exception as e:
                    print(f"[LLMWildcardManager] Could not read cached template: {e}")
            if cached:
                template = cached
                used_names = extract_wildcard_names(template)
                status = "locked"
                status_message = (
                    "Lock is ON — reusing the last generated template "
                    "(LLM not called). Toggle off to regenerate."
                )
                raw_sections.append((
                    "locked",
                    "(LLM calls skipped — template loaded from cache)",
                ))
            else:
                status = "no_locked_template"
                status_message = (
                    "Lock is ON but no cached template exists yet. "
                    "Toggle Lock off and queue once to generate one."
                )
                raw_sections.append((
                    "locked",
                    "(no cached template — toggle Lock off to generate)",
                ))
        else:
            axis_bans: list[str] = []
            try:
                # Step 0 — split the raw negative prompt into structured bans.
                # axis_bans names attribute axes that MUST NOT become
                # placeholders (e.g. 'no age' → 'age'); scene_bans names
                # things that must not appear in the image.
                if negative:
                    axis_bans, scene_bans, raw_parse_neg = llm_parse_negative(
                        negative, server, seed=effective_seed,
                    )
                    raw_sections.append(("parse_negative", raw_parse_neg))

                # Merge auto-derived axis bans into the user's explicit
                # forbidden_placeholders list. This is the actual fix for
                # 'no age, no gender' producing __age__/__gender__ anyway:
                # the user no longer has to repeat themselves in two fields.
                effective_forbidden_names: list[str] = []
                seen_forbidden: set[str] = set()
                for n in axis_bans + forbidden_names:
                    if n and n not in seen_forbidden:
                        seen_forbidden.add(n)
                        effective_forbidden_names.append(n)

                # Step 1 — draft the prompt sentence from idea + direction.
                draft, raw_draft = llm_draft_prompt(
                    example_prompt or "", direction_text, server,
                    seed=effective_seed,
                    scene_bans=scene_bans,
                )
                raw_sections.append(("draft", raw_draft))

                # Step 2 — wildcardify with the merged deny-list.
                template, used_names, raw_wildcardify = llm_wildcardify_prompt(
                    draft, server, seed=effective_seed,
                    min_categories=min_cats,
                    max_categories=max_cats,
                    forbidden_names=effective_forbidden_names,
                )
                raw_sections.append(("wildcardify", raw_wildcardify))

                # Step 3 — describe each wildcard.
                descs, raw_describe = llm_describe_wildcards(
                    used_names, server, seed=effective_seed,
                    idea=example_prompt or "",
                    direction_text=direction_text,
                    template=template,
                    scene_bans=scene_bans,
                )
                raw_sections.append(("describe", raw_describe))
                suggested_cats = descs
            except ManagerStepError as e:
                template = ""
                used_names = []
                status = f"failed_{e.step}"
                status_message = (
                    f"Step '{e.step}' did not return parseable output. "
                    "See the raw reply panel below."
                )
                raw_sections.append((e.step + " (failed)", e.raw))
            except Exception as e:
                print(f"[LLMWildcardManager] LLM call failed: {e}")
                template = ""
                used_names = []
                status = "llm_error"
                status_message = str(e)
                raw_sections.append(("error", f"(exception calling LLM: {e})"))

        # Stitch the per-step raw replies into one panel-friendly blob so the
        # existing UI's raw_reply pane stays useful for debugging all four steps.
        raw_reply = "\n\n".join(
            f"--- step: {step} ---\n{(body or '').strip()}"
            for step, body in raw_sections
        ) or "(no LLM output captured)"

        # Effective category descriptions: defaults < disk < LLM-suggested < user.
        merged_disk = load_category_config()
        effective: dict[str, str] = dict(DEFAULT_CATEGORIES)
        effective.update(merged_disk)
        effective.update(suggested_cats)
        effective.update(user_overrides)

        # Persist suggested + user-override categories so the Resolver-only
        # path picks them up too. Only on success — failed runs shouldn't poison
        # the disk config.
        if status == "ok":
            merged_disk.update(suggested_cats)
            merged_disk.update(user_overrides)
            save_category_config(merged_disk)

        # Persist the last successful template only on real success.
        if template and status == "ok":
            try:
                LAST_TEMPLATE_PATH.write_text(template, encoding="utf-8")
            except Exception as e:
                print(f"[LLMWildcardManager] Could not persist template: {e}")

        # Persist the raw reply + status so the UI can show it after reload.
        try:
            LAST_REPLY_PATH.write_text(json.dumps({
                "raw_reply": raw_reply,
                "status": status,
                "status_message": status_message,
                "template": template,
            }, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[LLMWildcardManager] Could not persist last reply: {e}")

        # Build the bundle handed to the Resolver. `scene_bans` is the
        # structured list of forbidden scene elements derived from the raw
        # negative prompt; the Resolver passes it to llm_generate_value_list
        # so per-slot values can't reintroduce them. Falls back to the raw
        # text via _negative_terms when the locked-template path skipped
        # the parse_negative step.
        bundle = {
            "system_prompt": "",
            "flair": direction_text,
            "negative": negative,
            "scene_bans": list(scene_bans),
            "category_overrides": dict(effective),
            "intended_names": list(used_names),
        }

        # Snapshot the categories the UI should display.
        used = set(used_names)
        display_cats = {n: effective.get(n, "") for n in
                        sorted(used | set(user_overrides))}

        snapshot = build_manager_snapshot(
            display_cats,
            direction=direction,
            negative_prompt=negative,
            generated_prompt=template,
            user_overrides=user_overrides,
        )
        snapshot["raw_reply"] = raw_reply
        snapshot["status"] = status
        snapshot["status_message"] = status_message

        return {
            "ui": {
                "manager_state": [json.dumps(snapshot)],
            },
            "result": (template, bundle),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # lock_template=True OR seed!=0 → reproducible (downstream Resolver
        # still re-rolls each queue when its fix_seed is off).
        # lock_template=False AND seed=0 → fresh roll every queue.
        if bool(kwargs.get("lock_template")):
            return _seeded_is_changed(True, kwargs)
        locked = int(kwargs.get("seed", 0) or 0) != 0
        return _seeded_is_changed(locked, kwargs)


# =============================================================================
# Node 3: LLMWildcardResolver — fills __wildcard__ slots in a template.
# =============================================================================
class LLMWildcardResolver:
    """ComfyUI node: resolves __wildcard__ slots via cache + LLM with anti-repetition.

    `fix_seed=False` (default): IS_CHANGED returns NaN — every queue re-rolls.
    `fix_seed=True`: deterministic — same template + same seed = same fills."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLM_SERVER",),
                "template": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Digital illustration of a __age__ __ethnicity__ woman with __hair__, "
                        "__activity__ at a __location__, __time__, wearing __outfit__, "
                        "__pose__, __style__, masterpiece, best quality, ultra-detailed"
                    ),
                }),
                "mode": (["reuse_existing", "force_new", "hybrid"],
                         {"default": "hybrid"}),
                "max_per_category": ("INT", {"default": 200, "min": 1, "max": 10000}),
                "min_pool_size": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "values_per_call": ("INT", {"default": 10, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF}),
                "fix_seed": ("BOOLEAN", {"default": False}),
                "trigger_words": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "trigger_position": (["prefix", "suffix"], {"default": "prefix"}),
            },
            "optional": {
                "prompts": ("WILDCARD_PROMPTS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("resolved_prompt", "report")
    FUNCTION = "resolve"
    CATEGORY = "prompt/wildcards"
    OUTPUT_NODE = True

    def resolve(self, server, template, mode, max_per_category,
                min_pool_size, values_per_call, seed, fix_seed,
                trigger_words="", trigger_position="prefix", prompts=None):
        rng = random.Random(seed if (fix_seed or seed != 0) else None)

        # Pool floor cannot exceed the hard cap.
        effective_min = max(1, min(int(min_pool_size), int(max_per_category)))
        per_call = max(1, int(values_per_call))

        categories = load_category_config()
        flair_text = ""
        negative_text = ""
        scene_bans: list[str] = []
        intended_names: list[str] = []
        if isinstance(prompts, dict):
            flair_text = prompts.get("flair") or ""
            negative_text = prompts.get("negative") or ""
            raw_scene = prompts.get("scene_bans") or []
            if isinstance(raw_scene, list):
                scene_bans = [str(s) for s in raw_scene if str(s).strip()]
            cfg_overrides = prompts.get("category_overrides") or {}
            if isinstance(cfg_overrides, dict):
                categories.update(cfg_overrides)
            raw_intended = prompts.get("intended_names") or []
            if isinstance(raw_intended, list):
                intended_names = [str(n) for n in raw_intended if str(n).strip()]

        # Fall back to the heuristic split when the manager didn't supply
        # structured scene_bans (locked-template path, or a stand-alone
        # Resolver wired without a Manager). Keeps the negative prompt
        # active in value generation either way.
        if not scene_bans and negative_text:
            scene_bans = list(_negative_terms(negative_text))

        # If the manager bundled a list of intended wildcard names, repair any
        # template tokens that lost their double underscores (e.g. `_subject_`
        # or bare `subject` left by a manual edit).
        if intended_names:
            template = ensure_wildcard_format(template, intended_names)

        # Effective sampling seed for any LLM calls. seed=0 + fix_seed=False
        # means fresh roll every queue.
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = 0
        if seed_int == 0:
            llm_seed = random.SystemRandom().randrange(1, 2**31)
        else:
            llm_seed = seed_int

        # Phase 1 — for each unique wildcard name, decide on a value.
        # Generation triggers when:
        #   * mode is force_new (always), OR
        #   * pool is empty (need at least one value), OR
        #   * pool is below `effective_min` (top-up to grow combinatoric breadth).
        # Otherwise we pick from disk without an LLM call.
        records: list[dict] = []
        picks: dict[str, str] = {}
        neg_terms = _negative_terms(negative_text)
        unique_names = extract_wildcard_names(template)
        for name in unique_names:
            existing = read_wildcard_file(name)
            # Filter the on-disk pool against the current negative prompt so
            # that values generated before the negative was set (or values
            # the LLM slipped through despite the instruction) cannot be
            # reused. Without this, the resolver happily picks forbidden
            # entries from the cached pool on every run.
            safe_existing = [
                e for e in existing
                if not _value_violates_negative(e, neg_terms)
            ]
            blocked_existing = len(existing) - len(safe_existing)
            description = categories.get(
                name, f"A value for the '{name}' wildcard category.")
            rec: dict = {"name": name, "pool_size": len(existing)}
            if blocked_existing:
                rec["blocked_by_negative"] = blocked_existing

            force_new = mode == "force_new"

            # Hard cap: only treat the pool as "full" when the *safe* subset
            # already meets the cap. If everything safe is exhausted we fall
            # through to generation so the user actually gets a compliant
            # value instead of a recycled forbidden one.
            if not force_new and len(safe_existing) >= max_per_category:
                value = rng.choice(safe_existing)
                rec.update({"status": "cap_reached", "value": value})
                records.append(rec)
                picks[name] = value
                continue

            below_floor = (mode != "force_new") and safe_existing and \
                len(safe_existing) < effective_min
            needs_generation = force_new or not safe_existing or below_floor

            if not needs_generation:
                value = rng.choice(safe_existing)
                rec.update({"status": "reused", "value": value})
                records.append(rec)
                picks[name] = value
                continue

            # How many to ask for: enough to reach the floor in one shot when
            # possible, but never less than `per_call` (avoid wasting a call on
            # one or two items) and never more than the cap allows.
            target_new = max(per_call, effective_min - len(safe_existing))
            if not force_new:
                cap_remaining = max(1, max_per_category - len(existing))
                target_new = min(target_new, cap_remaining)
            target_new = max(1, target_new)

            sent = (
                f'category="{name}" | desc={description!r} | '
                f'pool={len(existing)} (safe={len(safe_existing)}) items | '
                f'request={target_new} | '
                f'model={server.get("model", "")!r} | '
                f'temp={server.get("temperature", 0.9)}'
            )
            rec["sent"] = sent
            try:
                values, raw = llm_generate_value_list(
                    name, description, existing, server,
                    count=target_new, seed=llm_seed,
                    template=template or "",
                    direction_text=flair_text,
                    scene_bans=scene_bans,
                )
                rec["raw"] = raw
                # Drop any LLM-returned values that violate the negative
                # prompt before they hit disk. If the model returned nothing
                # safe, retry once with the forbidden traits glued onto the
                # description so the per-slot generator can't miss them.
                if neg_terms:
                    kept = [v for v in values
                            if not _value_violates_negative(v, neg_terms)]
                    dropped = len(values) - len(kept)
                    if dropped:
                        rec["filtered_by_negative"] = dropped
                    values = kept
                    if not values:
                        retry_desc = (
                            f"{description} CRITICAL: the value MUST NOT "
                            f"contain or imply any of: "
                            f"{negative_text.strip()}"
                        )
                        retry_sent = (
                            f'category="{name}" | retry after negative-'
                            f'prompt filter dropped all values'
                        )
                        rec["retry_sent"] = retry_sent
                        retry_values, retry_raw = llm_generate_value_list(
                            name, retry_desc, existing, server,
                            count=target_new, seed=llm_seed + 1,
                            template=template or "",
                            direction_text=flair_text,
                            scene_bans=scene_bans,
                        )
                        rec["retry_raw"] = retry_raw
                        values = [v for v in retry_values
                                  if not _value_violates_negative(v, neg_terms)]
                # Only keep values that aren't already in the pool.
                lower_existing = {e.lower() for e in existing}
                fresh = [v for v in values if v.lower() not in lower_existing]
                appended: list[str] = []
                for v in fresh:
                    if append_wildcard(name, v):
                        appended.append(v)
                rec["new_count"] = len(appended)

                if force_new:
                    pool = appended or fresh
                else:
                    pool = safe_existing + appended
                if not pool:
                    rec.update({"status": "error", "err": "no values produced",
                                "value": ""})
                    records.append(rec)
                    picks[name] = ""
                    continue

                value = rng.choice(pool)
                rec.update({
                    "status": "generated_new" if value in appended else "reused",
                    "value": value,
                })
                records.append(rec)
                picks[name] = value
            except Exception as e:
                rec.update({"status": "error", "err": str(e), "value": ""})
                records.append(rec)
                picks[name] = ""

        def _substitute(match: "re.Match") -> str:
            return picks.get(match.group(2), match.group(0))

        substituted = WILDCARD_RE.sub(_substitute, template or "")

        # Phase 2 — light grammar/article alignment pass. Skip if substitution
        # produced nothing useful or every value is empty.
        resolved = substituted
        align_raw = ""
        align_status = "skipped"
        non_empty_values = [v for v in picks.values() if v]
        if substituted and non_empty_values:
            try:
                aligned, align_raw = llm_align_prompt(
                    substituted, server, seed=llm_seed,
                )
                lower_aligned = aligned.lower()
                # Only accept the alignment if every populated value is still
                # present verbatim (case-insensitive). Otherwise keep the
                # literal substitution — losing a value is worse than a small
                # grammar slip.
                if aligned and all(v.lower() in lower_aligned
                                   for v in non_empty_values):
                    resolved = aligned
                    align_status = "applied"
                else:
                    align_status = "rejected"
            except Exception as e:
                align_status = f"error: {e}"

        # Phase 3 — splice trigger words onto the (already aligned) prompt.
        # Done after alignment so LoRA trigger tokens stay verbatim — the
        # alignment LLM might otherwise paraphrase or drop them.
        triggers = (trigger_words or "").strip().strip(",").strip()
        if triggers:
            if trigger_position == "suffix":
                resolved = f"{resolved}, {triggers}" if resolved else triggers
            else:
                resolved = f"{triggers}, {resolved}" if resolved else triggers

        report = format_report(records, flair=flair_text,
                               using_custom_prompt=False)
        write_last_report(report, records, flair=flair_text,
                          using_custom_prompt=False)

        snapshot = {
            "template": template or "",
            "resolved": resolved or "",
            "records": records,
            "tallies": _tally(records),
            "align_status": align_status,
            "align_raw": align_raw,
            "trigger_words": triggers,
            "trigger_position": trigger_position if triggers else "",
        }
        try:
            LAST_RESOLVER_PATH.write_text(
                json.dumps(snapshot, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[LLMWildcardResolver] Could not persist last resolver state: {e}")

        return {
            "ui": {"resolver_state": [json.dumps(snapshot)]},
            "result": (resolved, report),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _seeded_is_changed(bool(kwargs.get("fix_seed")), kwargs)


# =============================================================================
# Node 4: LLMWildcardReport — structured collapsible view of the report.
# =============================================================================
_REPORT_HEAD_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s+status=(?P<status>\S+)")


class LLMWildcardReport:
    """ComfyUI node: parse the resolver's report into stats and render a
    structured collapsible view inside the node body.

    The Resolver writes a JSON payload alongside the text report on every run;
    the JS frontend pulls that payload via /llm_wildcard/last_report so it can
    render per-slot rows with expand chevrons that reveal raw LLM replies."""

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
        # Prefer structured records if they line up with the incoming text.
        payload = read_last_report_payload() or {}
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            records = []

        if not records:
            # fall back to parsing the text report header counts
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
            tallies = {"generated": generated, "reused": reused,
                       "errors": errors, "total": total}
        else:
            tallies = _tally(records)

        summary = (report or "(no report)").strip()
        ui_payload = {
            "records": records,
            "tallies": tallies,
            "raw": summary,
            "flair": payload.get("flair", "") if isinstance(payload, dict) else "",
            "using_custom_prompt": (
                payload.get("using_custom_prompt", False)
                if isinstance(payload, dict) else False),
        }
        return {
            "ui": {"report_state": [json.dumps(ui_payload)]},
            "result": (summary, tallies["generated"], tallies["reused"],
                       tallies["errors"], tallies["total"]),
        }


NODE_CLASS_MAPPINGS = {
    "LLMServerConfig": LLMServerConfig,
    "LLMWildcardManager": LLMWildcardManager,
    "LLMWildcardResolver": LLMWildcardResolver,
    "LLMWildcardReport": LLMWildcardReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMServerConfig": "🎲 LLM Server Config",
    "LLMWildcardManager": "🎲 LLM Wildcard Manager",
    "LLMWildcardResolver": "🎲 LLM Wildcard Resolver",
    "LLMWildcardReport": "🎲 LLM Wildcard Report",
}
