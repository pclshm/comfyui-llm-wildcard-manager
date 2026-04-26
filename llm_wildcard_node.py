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
# System prompts
# -----------------------------------------------------------------------------
RESOLVER_SYSTEM_PROMPT = (
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


MANAGER_SYSTEM_PROMPT = (
    "TASK\n"
    "Rewrite the user's prompt idea by REPLACING noun phrases with\n"
    "__wildcard__ placeholders. Output a single JSON object — no markdown\n"
    "fences, no prose before or after, no commentary.\n\n"
    "CRITICAL CONSTRAINT (failure to follow = invalid output):\n"
    "  The `prompt` string MUST literally contain at least 3 placeholders\n"
    "  shaped like __snake_case_name__ (double underscore + name + double\n"
    "  underscore). If your output contains zero `__` substrings, you have\n"
    "  failed the task.\n\n"
    "EXACT OUTPUT SHAPE (only these two top-level keys are allowed):\n"
    "{\n"
    "  \"prompt\": \"<the user's idea, but with __wildcards__ inserted>\",\n"
    "  \"categories\": {\n"
    "    \"<wildcard_name>\": \"<short shape-of-value description>\"\n"
    "  }\n"
    "}\n\n"
    "EXAMPLE 1\n"
    "INPUT: A portrait of a woman doing an outdoor activity, photorealistic, masterpiece.\n"
    "OUTPUT:\n"
    "{\n"
    "  \"prompt\": \"A portrait of a __age__ __ethnicity__ woman with __hair__, "
    "__activity__ at a __location__, __time__, wearing __outfit__, __pose__, "
    "__style__, photorealistic, masterpiece\",\n"
    "  \"categories\": {\n"
    "    \"age\": \"An age descriptor.\",\n"
    "    \"ethnicity\": \"A heritage descriptor.\",\n"
    "    \"hair\": \"Hair style, length, and color.\",\n"
    "    \"activity\": \"A single active activity.\",\n"
    "    \"location\": \"An indoor or outdoor location.\",\n"
    "    \"time\": \"A time-of-day or lighting condition.\",\n"
    "    \"outfit\": \"An outfit description.\",\n"
    "    \"pose\": \"A pose or body-language description.\",\n"
    "    \"style\": \"Photographic or illustration style modifiers.\"\n"
    "  }\n"
    "}\n\n"
    "EXAMPLE 2\n"
    "INPUT: A cyberpunk samurai walking through neon-lit streets at night.\n"
    "OUTPUT:\n"
    "{\n"
    "  \"prompt\": \"A __era__ __character_class__ __action__ through __setting__ at __time__\",\n"
    "  \"categories\": {\n"
    "    \"era\": \"An era or genre descriptor.\",\n"
    "    \"character_class\": \"A character archetype.\",\n"
    "    \"action\": \"What the character is doing.\",\n"
    "    \"setting\": \"The environment or streetscape.\",\n"
    "    \"time\": \"Time-of-day or atmospheric phase.\"\n"
    "  }\n"
    "}\n\n"
    "WRONG OUTPUTS — do NOT do any of these:\n"
    "  WRONG: {\"prompt\": \"A polished version of the user's idea.\"}\n"
    "         (no __wildcards__ — failed the critical constraint)\n"
    "  WRONG: {\"subject\": \"woman\", \"style\": \"photorealistic\"}\n"
    "         (flat keys instead of {prompt, categories})\n"
    "  WRONG: {\"prompt\": \"...\", \"style\": \"...\", \"lighting\": \"...\"}\n"
    "         (extra top-level keys forbidden — those belong inside prompt or categories)\n"
    "  WRONG: {\"prompt\": \"...\", \"negative_prompt\": \"...\",\n"
    "          \"parameters\": {...}, \"wildcard_slots_used\": [\"subject_action\", ...]}\n"
    "         (this is an image-gen API payload, NOT a wildcard template. The\n"
    "          wildcards must be inserted INTO the prompt string as literal\n"
    "          __snake_case_name__ tokens — never listed in a separate array,\n"
    "          and never accompanied by negative_prompt / parameters / steps /\n"
    "          guidance_scale / aspect_ratio / seed / etc.)\n\n"
    "STRICT RULES\n"
    "1. The `prompt` string MUST contain `__name__` placeholders. At least 3.\n"
    "2. Do NOT just polish or rewrite the user's idea — you MUST insert\n"
    "   __wildcards__ where the variable parts are.\n"
    "3. Wildcards look like __snake_case_name__ (double underscore each side).\n"
    "4. Every wildcard in `prompt` MUST have a matching key in `categories`.\n"
    "5. Every key in `categories` MUST appear in `prompt` at least once.\n"
    "6. The ONLY top-level keys allowed are `prompt` and `categories`.\n"
    "7. Keep category descriptions short and shape-focused, not specifics.\n"
    "8. Reuse common category names (hair, age, outfit, location, time,\n"
    "   activity, style, pose, ...) to build up reusable wildcard libraries."
)


# Strict JSON schema for the Manager's reply. Sent to the backend so the
# server grammar-constrains output and the model can't invent extra top-level
# keys (negative_prompt, parameters, wildcard_slots_used, seed, ...) the way
# json_object mode allows.
#
# The `pattern` on `prompt` forces the grammar to require at least one
# __snake_case__ substring. Without this, weaker models exhibit "schema
# collapse" — they satisfy the {prompt, categories} shape but emit a plain
# polished sentence with zero placeholders, treating categories as filled-in
# values instead of placeholder descriptions.
MANAGER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            # llama.cpp's schema-to-grammar converter requires the pattern to
            # be anchored with ^ and $. The `[\s\S]*` (instead of `.*`) lets
            # the prompt span newlines, since `.` excludes \n in its regex.
            "pattern": "^[\\s\\S]*__[A-Za-z][A-Za-z0-9_-]*__[\\s\\S]*$",
        },
        "categories": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["prompt", "categories"],
    "additionalProperties": False,
}


# Raw GBNF grammar mirroring MANAGER_JSON_SCHEMA. Sent via llama.cpp's native
# `grammar` field — the only enforcement path that has worked reliably across
# llama.cpp builds for the user. The schema-to-grammar converter has historically
# produced permissive grammars (extra top-level keys leak through, the prompt's
# pattern requirement is dropped, ...). Hand-written GBNF bypasses that and
# guarantees:
#   1. exactly two top-level keys, in order: "prompt" and "categories"
#   2. the prompt string contains at least one __snake_case_name__ substring
#      (the prefix rule rejects any sequence containing a stray "__" before the
#      wildcard, forcing the model to emit a real placeholder)
#   3. categories is an object mapping snake_case keys to strings
MANAGER_GBNF = r'''
root ::= "{" ws "\"prompt\"" ws ":" ws prompt-string ws "," ws "\"categories\"" ws ":" ws cat-obj ws "}"

prompt-string ::= "\"" pre-segs "__" wname "__" post-segs "\""

pre-segs ::= ( pre-char | "_" non-und )*
pre-char ::= [^"\\_\x7F\x00-\x1F]
non-und ::= [^_"\\\x7F\x00-\x1F]

wname ::= [a-zA-Z] [a-zA-Z0-9_-]*

post-segs ::= post-char*
post-char ::= [^"\\\x7F\x00-\x1F] | escape

escape ::= "\\" ( ["\\bfnrt] | "u" [0-9a-fA-F]{4} )

cat-obj ::= "{" ws "}" | "{" ws cat-entry (ws "," ws cat-entry)* ws "}"
cat-entry ::= cat-key ws ":" ws cat-value
cat-key ::= "\"" [a-z] [a-z0-9_]* "\""
cat-value ::= "\"" val-char* "\""
val-char ::= [^"\\\x7F\x00-\x1F] | escape

ws ::= [ \t\n\r]*
'''.strip()


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


WILDCARD_RE = re.compile(r"__(!)?([A-Za-z0-9_\-]+)__")


def extract_wildcard_names(template: str) -> list[str]:
    seen = []
    for m in WILDCARD_RE.finditer(template or ""):
        n = m.group(2)
        if n not in seen:
            seen.append(n)
    return seen


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _salvage_flat_template(base_text: str, flat: dict
                           ) -> tuple[str, dict[str, str]] | None:
    """Reconstruct a template from a flat {key: value, ...} LLM reply.

    Many models ignore the {prompt, categories} schema and instead respond with
    a flat object that *correctly* names the variables — `subject: "woman"`,
    `activity: "outdoor activity"`, `style: "photorealistic"`. If those values
    appear in the user's idea, we can splice each one back in as `__key__` and
    end up with a perfectly usable template.

    Returns (template, categories) on success, None if nothing matched.
    """
    if not base_text or not isinstance(flat, dict):
        return None
    base_text = base_text.strip()
    if not base_text:
        return None

    candidates: list[tuple[str, str]] = []
    for raw_k, raw_v in flat.items():
        if not isinstance(raw_v, str):
            continue
        v = raw_v.strip()
        if len(v) < 2:
            continue
        k = re.sub(r"[^a-z0-9_]+", "_", str(raw_k).lower()).strip("_")
        if not k or not _KEY_RE.match(k):
            continue
        if k == "prompt":
            continue  # the literal sentence, not a wildcard slot
        candidates.append((k, v))
    if not candidates:
        return None
    # Longest value first so "outdoor activity" beats "activity" etc.
    candidates.sort(key=lambda kv: -len(kv[1]))

    # Segment-walk the base text. Each segment is either ("text", str) or
    # ("wild", key). We only search inside text segments, so once a span has
    # been wildcarded we won't re-match a substring of it.
    segments: list[tuple[str, str]] = [("text", base_text)]
    for key, value in candidates:
        new_segments: list[tuple[str, str]] = []
        replaced = False
        for kind, content in segments:
            if replaced or kind != "text":
                new_segments.append((kind, content))
                continue
            idx = content.lower().find(value.lower())
            if idx < 0:
                new_segments.append((kind, content))
                continue
            before = content[:idx]
            after = content[idx + len(value):]
            if before:
                new_segments.append(("text", before))
            new_segments.append(("wild", key))
            if after:
                new_segments.append(("text", after))
            replaced = True
        segments = new_segments

    used: list[str] = []
    parts: list[str] = []
    for kind, content in segments:
        if kind == "text":
            parts.append(content)
        else:
            parts.append(f"__{content}__")
            if content not in used:
                used.append(content)
    if not used:
        return None
    template = "".join(parts)
    cats = {k: f"A value for the '{k}' wildcard category." for k in used}
    return template, cats


# Heuristic patterns for the next salvage tier — when the LLM gives us a
# polished sentence with no flat keys, pattern-match common nouns and inject
# the matching default-category wildcard. Order matters: longer / more
# specific phrases first, broad subject words last.
_HEURISTIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\boutdoor activity\b", re.I), "activity"),
    (re.compile(r"\bindoor activity\b", re.I), "activity"),
    (re.compile(
        r"\b(?:jogging|running|swimming|climbing|hiking|cycling|dancing|"
        r"skiing|surfing|reading|painting|gardening|practicing\s+yoga)\b",
        re.I), "activity"),
    (re.compile(
        r"\b(?:photorealistic|hyperrealistic|cinematic|anime|cartoon|"
        r"illustration|watercolour|watercolor|oil\s+painting|sketch|noir|"
        r"dreamlike|vintage)\b", re.I), "style"),
    (re.compile(
        r"\b(?:sunrise|sunset|midnight|twilight|dawn|dusk|noon|morning|"
        r"afternoon|evening|night)\b", re.I), "time"),
    (re.compile(
        r"\b(?:beach|forest|mountain|park|garden|rooftop|cafe|library|"
        r"studio|street|alley|meadow|lake|river|desert|coast|harbour|harbor)\b",
        re.I), "location"),
    (re.compile(
        r"\b(?:rainy|sunny|cloudy|stormy|foggy|misty|snowy|windy)\b",
        re.I), "weather"),
    # subject is broad — keep it last so more specific matches win first.
    (re.compile(r"\b(?:woman|man|girl|boy|person|lady|guy|model)\b",
                re.I), "subject"),
]

_HEURISTIC_DESCRIPTIONS = {
    "subject": (
        "The portrait subject (e.g. 'woman', 'man', 'older gentleman'). "
        "One concise phrase."
    ),
}


def _heuristic_salvage(base_text: str
                       ) -> tuple[str, dict[str, str]] | None:
    """Pattern-match common nouns in `base_text` and replace each with the
    matching default-category wildcard. Returns (template, categories) on
    success, None if no patterns matched."""
    if not base_text or not base_text.strip():
        return None
    out = base_text
    used: list[str] = []
    for pattern, name in _HEURISTIC_PATTERNS:
        if name in used:
            continue
        m = pattern.search(out)
        if not m:
            continue
        # Don't overwrite an already-inserted wildcard.
        span = out[m.start():m.end()]
        if "__" in span:
            continue
        out = out[:m.start()] + f"__{name}__" + out[m.end():]
        used.append(name)
    if not used:
        return None
    cats = {}
    for n in used:
        cats[n] = (
            DEFAULT_CATEGORIES.get(n)
            or _HEURISTIC_DESCRIPTIONS.get(n)
            or f"A value for the '{n}' wildcard category."
        )
    return out, cats


_DEFAULT_FALLBACK_TEMPLATE = (
    "A portrait of a __age__ __ethnicity__ __subject__ with __hair__, "
    "__activity__ at a __location__, __time__, __weather__, wearing "
    "__outfit__, __pose__, __style__"
)


def _default_template_fallback() -> tuple[str, dict[str, str]]:
    """Last resort — return a generic template using DEFAULT_CATEGORIES so
    the downstream Resolver always has wildcards to fill, even when the LLM
    completely failed to produce a usable structure."""
    cats: dict[str, str] = {}
    for name in extract_wildcard_names(_DEFAULT_FALLBACK_TEMPLATE):
        cats[name] = (
            DEFAULT_CATEGORIES.get(name)
            or _HEURISTIC_DESCRIPTIONS.get(name)
            or f"A value for the '{name}' wildcard category."
        )
    return _DEFAULT_FALLBACK_TEMPLATE, cats


# -----------------------------------------------------------------------------
# LLM operations
# -----------------------------------------------------------------------------
def llm_generate_value(category: str, description: str, existing: list[str],
                       server: dict, system_prompt: str | None = None,
                       temperature_override: float | None = None) -> str:
    sys_p = system_prompt if (system_prompt and system_prompt.strip()) else RESOLVER_SYSTEM_PROMPT
    forbidden = "\n".join(f"- {e}" for e in existing) if existing else "(none yet)"
    user = (
        f"Category: {category}\n"
        f"Category meaning: {description}\n\n"
        f"Forbidden values (do NOT repeat or paraphrase any of these):\n{forbidden}\n\n"
        f"Output exactly one new value for the '{category}' category."
    )
    raw = _server_call(server, sys_p, user, temperature_override=temperature_override)
    return _clean_llm_value(raw)


def llm_design_template(example_prompt: str, direction_text: str, extra_flair: str,
                        server: dict, system_prompt_override: str | None = None,
                        seed: int = 0
                        ) -> tuple[str, dict[str, str], str, str]:
    """Ask the LLM to produce a prompt template + a description per wildcard.

    Returns (template, categories, raw_reply, status). status is one of:
      "ok"           — JSON parsed and `prompt` non-empty.
      "parse_failed" — could not extract a JSON object from the reply.
      "no_prompt"    — JSON parsed but `prompt` was missing/empty.
    No silent fallback to example_prompt — the Manager surfaces the failure
    so the user can see the raw reply and fix the LLM/system prompt.
    """
    base = (system_prompt_override or "").strip() or MANAGER_SYSTEM_PROMPT
    flair_lines = [s for s in (direction_text or "", extra_flair or "") if s.strip()]
    if flair_lines:
        base = base + "\n\nAdditional direction from the user:\n" + "\n".join(flair_lines)

    user_parts = [
        "User idea / example prompt:",
        (example_prompt or "").strip() or "(no example provided)",
    ]
    if seed:
        # Belt-and-suspenders: also nudge the model textually, since some local
        # backends ignore the sampling seed in their /chat options.
        # Avoid the literal word "seed" in the user-visible text — it cues some
        # models to emit an image-gen-style payload with a top-level "seed" key.
        user_parts.append(
            f"\nVariation token: {seed}. Treat this token as inspiration for "
            "fresh phrasing and a different mix of wildcard slots — produce a "
            "meaningfully different template than you would for any other token."
        )
    user_parts.append("\nNow produce the JSON object.")
    user = "\n".join(user_parts)

    raw = _server_call(server, base, user, request_json=True, seed=seed,
                       json_schema=MANAGER_JSON_SCHEMA, grammar=MANAGER_GBNF)
    parsed = _extract_json_object(raw)
    if not parsed:
        return ("", {}, raw, "parse_failed")

    template = str(parsed.get("prompt") or "").strip()
    categories_raw = parsed.get("categories")

    # Layered salvage. Order: (1) flat-key splice, (2) heuristic noun match,
    # (3) generic default template. Each tier runs only if the previous one
    # didn't produce a template with __wildcards__. This guarantees the
    # downstream Resolver always has slots to fill, even with a dumb LLM.
    salvage_kind = ""

    def _has_wildcards(t: str) -> bool:
        return bool(extract_wildcard_names(t or ""))

    # Tier 1 — flat-key salvage from the LLM reply.
    if not _has_wildcards(template) or not isinstance(categories_raw, dict):
        candidates = []
        for base_text in (template, (example_prompt or "").strip()):
            base_text = (base_text or "").strip()
            if not base_text:
                continue
            r = _salvage_flat_template(base_text, parsed)
            if r is None:
                continue
            candidates.append((len(extract_wildcard_names(r[0])), r))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            template, categories_raw = candidates[0][1]
            salvage_kind = "flat"

    # Tier 2 — heuristic noun-pattern salvage on the LLM prompt or user idea.
    if not _has_wildcards(template):
        candidates = []
        for base_text in (template, (example_prompt or "").strip()):
            base_text = (base_text or "").strip()
            if not base_text:
                continue
            r = _heuristic_salvage(base_text)
            if r is None:
                continue
            candidates.append((len(extract_wildcard_names(r[0])), r))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            template, categories_raw = candidates[0][1]
            salvage_kind = "heuristic"

    # Tier 3 — generic default template. Always succeeds.
    if not _has_wildcards(template):
        template, categories_raw = _default_template_fallback()
        salvage_kind = "default"

    categories: dict[str, str] = {}
    if isinstance(categories_raw, dict):
        for k, v in categories_raw.items():
            ks = str(k).strip()
            if ks:
                categories[ks] = str(v).strip()

    # final consistency pass — every wildcard in template must have a description.
    for name in extract_wildcard_names(template):
        categories.setdefault(name, f"A value for the '{name}' wildcard.")

    if salvage_kind == "default":
        status = "fallback_default"
    elif salvage_kind:
        status = "salvaged"
    else:
        status = "ok"
    return (template, categories, raw, status)


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
                           extra_flair: str = "",
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
        "extra_flair": extra_flair,
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
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF}),
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

    def manage(self, server, example_prompt, seed, direction, extra_flair,
               system_prompt_override, categories):
        direction = (direction or "").strip() or "none"
        direction_text = resolve_direction(direction)
        extra = (extra_flair or "").strip()
        flair = "\n".join(s for s in (direction_text, extra) if s)

        override = (system_prompt_override or "").strip()

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

        # Effective sampling seed for the LLM. seed=0 means "fresh roll every
        # queue", so pick a non-deterministic one — otherwise the model would
        # see the same input each queue and likely emit the same template.
        # seed!=0 is passed through verbatim so reproducibility is preserved.
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = 0
        if seed_int == 0:
            effective_seed = random.SystemRandom().randrange(1, 2**31)
        else:
            effective_seed = seed_int

        # Call the LLM to design the template + suggest descriptions.
        status = "ok"
        status_message = ""
        try:
            template, suggested_cats, raw_reply, status = llm_design_template(
                example_prompt=example_prompt or "",
                direction_text=direction_text,
                extra_flair=extra,
                server=server,
                system_prompt_override=override or None,
                seed=effective_seed,
            )
        except Exception as e:
            print(f"[LLMWildcardManager] LLM call failed: {e}")
            template = ""
            suggested_cats = {}
            raw_reply = f"(exception calling LLM: {e})"
            status = "llm_error"
            status_message = str(e)

        if status == "parse_failed":
            status_message = (
                "LLM did not return a parseable JSON object. "
                "See the raw reply panel below."
            )
        elif status == "salvaged":
            status_message = (
                "LLM didn't follow the schema, but I rebuilt a template by "
                "matching its output back into your idea. Edit the prompt "
                "above if the wildcards landed in the wrong place."
            )
        elif status == "fallback_default":
            status_message = (
                "LLM never produced a usable template (even after pattern "
                "salvage). Falling back to a generic default template using "
                "the built-in categories. For better results, switch to a "
                "stronger model (e.g. qwen2.5:7b, llama3.1:8b-instruct, "
                "gemma2:9b) in your LLM Server Config."
            )

        # Effective category descriptions: defaults < disk < LLM-suggested < user.
        merged_disk = load_category_config()
        effective: dict[str, str] = dict(DEFAULT_CATEGORIES)
        effective.update(merged_disk)
        effective.update(suggested_cats)
        effective.update(user_overrides)

        # Persist any newly-suggested or user-overridden category to disk so the
        # Resolver-only path also picks them up. This file accumulates over time.
        merged_disk.update(suggested_cats)
        merged_disk.update(user_overrides)
        save_category_config(merged_disk)

        # Persist the last successful template only on real success — never
        # mask a failure by stashing example_prompt or a malformed reply.
        if template and status in ("ok", "salvaged", "fallback_default"):
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

        # Build the bundle handed to the Resolver. Effective categories include
        # everything we know — that's what the Resolver should use as descriptions.
        bundle = {
            "system_prompt": (
                (override or RESOLVER_SYSTEM_PROMPT)
                + (f"\n\nAdditional direction from the user:\n{flair}" if flair else "")
            ),
            "flair": flair,
            "category_overrides": dict(effective),
        }

        # Snapshot the categories the UI should display: only the ones used by
        # the current template + any user override + any disk file. Everything
        # else is noise.
        used = set(extract_wildcard_names(template))
        display_cats = {n: effective.get(n, "") for n in
                        sorted(used | set(user_overrides))}

        snapshot = build_manager_snapshot(
            display_cats,
            direction=direction,
            extra_flair=extra,
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
        # seed=0 → fresh roll every queue; seed!=0 → reproducible.
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
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF}),
                "fix_seed": ("BOOLEAN", {"default": False}),
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

    def resolve(self, server, template, mode, max_per_category, seed, fix_seed,
                prompts=None):
        rng = random.Random(seed if (fix_seed or seed != 0) else None)

        categories = load_category_config()
        custom_system_prompt: str | None = None
        flair_text = ""
        if isinstance(prompts, dict):
            custom_system_prompt = prompts.get("system_prompt") or None
            flair_text = prompts.get("flair") or ""
            cfg_overrides = prompts.get("category_overrides") or {}
            if isinstance(cfg_overrides, dict):
                categories.update(cfg_overrides)

        records: list[dict] = []

        def resolve_slot(match: "re.Match") -> str:
            force_flag = match.group(1)
            name = match.group(2)
            force_new = (force_flag == "!") or (mode == "force_new")
            existing = read_wildcard_file(name)
            description = categories.get(
                name, f"A value for the '{name}' wildcard category.")

            rec: dict = {"name": name, "pool_size": len(existing)}

            should_reuse = (not force_new) and existing and (mode != "force_new")
            if should_reuse:
                value = rng.choice(existing)
                rec.update({"status": "reused", "value": value})
                records.append(rec)
                return value

            if len(existing) >= max_per_category:
                value = rng.choice(existing) if existing else f"[{name}]"
                rec.update({"status": "cap_reached", "value": value})
                records.append(rec)
                return value

            sent = (
                f'category="{name}" | desc={description!r} | '
                f'forbidden={len(existing)} items | '
                f'model={server.get("model", "")!r} | '
                f'temp={server.get("temperature", 0.9)}'
            )
            rec["sent"] = sent
            try:
                value = llm_generate_value(
                    name, description, existing, server,
                    system_prompt=custom_system_prompt,
                )
                rec["raw"] = value
                if not value:
                    raise RuntimeError("empty LLM response")

                # one retry if the model ignored the forbidden list
                if existing and any(e.lower() == value.lower() for e in existing):
                    bumped = min(2.0, float(server.get("temperature", 0.9)) + 0.3)
                    rec["retry_sent"] = f"temp={bumped} (after duplicate)"
                    retry = llm_generate_value(
                        name, description, existing + [value], server,
                        system_prompt=custom_system_prompt,
                        temperature_override=bumped,
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
        write_last_report(report, records, flair=flair_text,
                          using_custom_prompt=bool(custom_system_prompt))

        snapshot = {
            "template": template or "",
            "resolved": resolved or "",
            "records": records,
            "tallies": _tally(records),
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
