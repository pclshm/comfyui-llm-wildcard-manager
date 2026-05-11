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
LAST_BRIEF_PATH = WILDCARDS_DIR / ".last_brief.json"


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


def _sampling_from_server(server: dict) -> dict:
    """Pick sampling overrides off the server dict. Empty dict if the user has
    enabled `use_model_default_sampling` or no LLMSamplingOptions node was
    wired in."""
    if not server.get("use_model_default_sampling", False):
        keys = ("top_k", "top_p", "min_p", "repeat_penalty", "context_size")
        return {k: server[k] for k in keys if server.get(k) is not None}
    return {}


def _call_ollama(endpoint: str, model: str, system: str, user: str, temperature: float,
                 request_json: bool = False, seed: int = 0,
                 json_schema: dict | None = None,
                 sampling: dict | None = None) -> str:
    options = {"temperature": float(temperature)}
    if seed:
        options["seed"] = int(seed)
    if sampling:
        # Ollama option names: top_k, top_p, min_p, repeat_penalty, num_ctx.
        if "top_k" in sampling:
            options["top_k"] = int(sampling["top_k"])
        if "top_p" in sampling:
            options["top_p"] = float(sampling["top_p"])
        if "min_p" in sampling:
            options["min_p"] = float(sampling["min_p"])
        if "repeat_penalty" in sampling:
            options["repeat_penalty"] = float(sampling["repeat_penalty"])
        if "context_size" in sampling:
            options["num_ctx"] = int(sampling["context_size"])
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
                            backend: str = "openai_compatible",
                            sampling: dict | None = None) -> str:
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
    if sampling:
        # llama.cpp's OpenAI-compat server accepts these as top-level fields.
        # Real OpenAI ignores unknown fields, so this is safe to send unconditionally.
        if "top_k" in sampling:
            payload["top_k"] = int(sampling["top_k"])
        if "top_p" in sampling:
            payload["top_p"] = float(sampling["top_p"])
        if "min_p" in sampling:
            payload["min_p"] = float(sampling["min_p"])
        if "repeat_penalty" in sampling:
            payload["repeat_penalty"] = float(sampling["repeat_penalty"])
        if "context_size" in sampling:
            # llama.cpp server also accepts `n_ctx` on /v1/chat/completions.
            payload["n_ctx"] = int(sampling["context_size"])

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
    sampling = _sampling_from_server(server)
    if server.get("show_everything_in_console", False):
        print(
            f"[LLMWildcard DEBUG] system={system!r}\n"
            f"[LLMWildcard DEBUG] user={user!r}\n"
            f"[LLMWildcard DEBUG] temperature={temp} sampling={sampling}"
        )
    if backend == "ollama":
        # Ollama has no `grammar` field — schema is the only enforcement path.
        reply = _call_ollama(endpoint, model, system, user, temp,
                             request_json=request_json, seed=seed,
                             json_schema=json_schema, sampling=sampling)
    else:
        reply = _call_openai_compatible(endpoint, model, system, user, temp, api_key,
                                        request_json=request_json, seed=seed,
                                        json_schema=json_schema, grammar=grammar,
                                        backend=backend, sampling=sampling)
    if server.get("show_everything_in_console", False):
        print(f"[LLMWildcard DEBUG] reply={reply!r}")
    return reply


# -----------------------------------------------------------------------------
# Vision-aware chat helper for the LLMPromptGenerator node.
#
# Talks directly to whatever endpoint the user already configured in their
# LLM_SERVER. No subprocesses, no port management — purely a client.
# Returns (content, reasoning) so callers can split a model's "thinking"
# stream from its final answer.
# -----------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>([\s\S]*?)</think\s*>", re.IGNORECASE)


def _split_thinking(text: str) -> tuple[str, str]:
    """Pull <think>...</think> blocks out of `text` and return
    (clean_content, joined_thinking_text)."""
    if not text:
        return "", ""
    thinking_parts = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(text)]
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    return cleaned, "\n\n".join(p for p in thinking_parts if p)


def _encode_comfy_image_to_b64(image_tensor, max_pixels: int = 2_000_000) -> str:
    """Convert a ComfyUI IMAGE tensor (batch, H, W, 3) float 0..1 to PNG base64.
    Lazy imports PIL/numpy/torch — only the IMAGE-using code path pulls them in."""
    from PIL import Image
    from io import BytesIO
    import base64
    import numpy as np

    img = image_tensor[0]
    if hasattr(img, "cpu"):
        img = img.cpu()
    if hasattr(img, "numpy"):
        img = img.numpy()
    arr = (np.clip(np.asarray(img), 0.0, 1.0) * 255).astype("uint8")
    pil = Image.fromarray(arr)
    w, h = pil.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.Resampling.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _chat_with_vision(server: dict, system: str, user: str,
                      image_b64: str | None = None,
                      seed: int = 0,
                      enable_thinking: bool = False,
                      timeout: int = 180) -> tuple[str, str]:
    """Send a chat-completion to whichever endpoint the LLM_SERVER points at,
    optionally with one base64-encoded PNG image. Returns (content, thinking).

    Routing:
      backend == "ollama"  → POST {endpoint}/api/chat   (native, images: [b64])
      otherwise            → POST {endpoint}/chat/completions  (OpenAI-compat,
                              image_url content array)
    """
    backend = server.get("backend", "ollama")
    endpoint = server.get("endpoint", "")
    model = server.get("model", "")
    api_key = server.get("api_key", "")
    temperature = float(server.get("temperature", 0.7))
    sampling = _sampling_from_server(server)

    if backend == "ollama":
        options = {"temperature": temperature}
        if seed:
            options["seed"] = int(seed)
        if "top_k" in sampling:
            options["top_k"] = int(sampling["top_k"])
        if "top_p" in sampling:
            options["top_p"] = float(sampling["top_p"])
        if "min_p" in sampling:
            options["min_p"] = float(sampling["min_p"])
        if "repeat_penalty" in sampling:
            options["repeat_penalty"] = float(sampling["repeat_penalty"])
        if "context_size" in sampling:
            options["num_ctx"] = int(sampling["context_size"])

        user_msg: dict = {"role": "user", "content": user}
        if image_b64:
            user_msg["images"] = [image_b64]
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                user_msg,
            ],
            "stream": False,
            "options": options,
        }
        if enable_thinking:
            # Newer Ollama (0.4+) honors `think` for reasoning-enabled models;
            # older versions ignore unknown fields, so this is safe.
            payload["think"] = True
        url = endpoint.rstrip("/") + "/api/chat"
        body = _http_post_json(url, payload, {"Content-Type": "application/json"},
                               timeout=timeout)
        message = body.get("message", {}) or {}
        content_raw = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()
        content, inline_thinking = _split_thinking(content_raw)
        if inline_thinking and not thinking:
            thinking = inline_thinking
        return content, thinking

    # OpenAI-compatible path (llama.cpp server, real OpenAI, etc.)
    if image_b64:
        user_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": user},
        ]
    else:
        user_content = user
    payload = {
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if seed:
        payload["seed"] = int(seed)
    if "top_k" in sampling:
        payload["top_k"] = int(sampling["top_k"])
    if "top_p" in sampling:
        payload["top_p"] = float(sampling["top_p"])
    if "min_p" in sampling:
        payload["min_p"] = float(sampling["min_p"])
    if "repeat_penalty" in sampling:
        payload["repeat_penalty"] = float(sampling["repeat_penalty"])
    if "context_size" in sampling:
        payload["n_ctx"] = int(sampling["context_size"])
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = endpoint.rstrip("/") + "/chat/completions"
    body = _http_post_json(url, payload, headers, timeout=timeout)
    message = body["choices"][0].get("message", {}) or {}
    content_raw = (message.get("content") or "").strip()
    thinking = (message.get("reasoning_content") or "").strip()
    content, inline_thinking = _split_thinking(content_raw)
    if inline_thinking and not thinking:
        thinking = inline_thinking
    return content, thinking


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
BRIEF_SYSTEM_PROMPT = (
    "The user gives you a seed idea — usually a short concept, not a finished "
    "scene. Your job is to invent a vivid scenario around that seed and "
    "package it as a design brief. Downstream steps will turn the scenario "
    "into a wildcard template so MANY varied images of this same scenario "
    "can be generated.\n"
    "Return four fields:\n"
    " - refined_idea: two or three sentences describing a concrete imagined "
    "scenario built from the user's seed. INVENT the supporting details the "
    "user did not specify: setting, time of day, atmosphere, what the subject "
    "is doing, framing, mood, era, supporting elements. Be specific and "
    "evocative — picture an actual moment, not a category. Do not contradict "
    "anything the user explicitly said; expand around it.\n"
    " - fixed_traits: short literal phrases that the user explicitly named "
    "(or unambiguously implied) as mandatory. These will appear verbatim in "
    "every generated image and must NEVER become wildcards. Pull only what "
    "the user already said; do NOT include details you invented in "
    "refined_idea. Each phrase 1-5 words, no leading article. Examples of "
    "shape: a fabric pattern the user named, a named subject, a specific "
    "location, a named garment, a stated era.\n"
    " - forbidden_axes: snake_case names of attribute axes whose values are "
    "already pinned by fixed_traits and therefore must NOT become wildcard "
    "placeholders. Derive these directly from fixed_traits (e.g. a named "
    "fabric pattern → 'pattern'; a stated location → 'location'). "
    "One- or two-word snake_case only. Empty list if no fixed_traits.\n"
    " - scene_bans: things that must not appear in the image, inferred from "
    "negation language in the user's idea (\"no X\", \"without X\", \"never X\"). "
    "Leave empty if the user did not negate anything.\n"
    "Invent confidently in refined_idea; stay strict and literal in "
    "fixed_traits / forbidden_axes / scene_bans. The user can edit the "
    "brief afterwards.\n"
    'Output JSON: {"refined_idea": "...", "fixed_traits": ["..."], '
    '"forbidden_axes": ["..."], "scene_bans": ["..."]}'
)

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
    "Write one rich image-prompt sentence that fully realises the scenario "
    "described in the user message. Concrete, visual, present tense. No "
    "preamble, no quotes — sentence only.\n"
    "Flesh the scenario out with specific visual content: the subject and "
    "what they are doing, their clothing or appearance, the setting and "
    "props, the lighting and atmosphere, the framing or composition, the "
    "overall mood. Pack in concrete details — this single sentence is the "
    "blueprint for many generated images, so it must paint a vivid picture, "
    "not stay abstract. Aim for a long, descriptive sentence (comma-joined "
    "clauses are fine).\n"
    "If a list of fixed traits is provided, every phrase in that list MUST "
    "appear verbatim (or as a near-identical substring) in the sentence. Do "
    "not paraphrase, abstract, or substitute them. Build the rest of the "
    "scene around those locked phrases.\n"
    "If a list of forbidden scene elements is provided, the sentence MUST "
    "NOT depict or imply any of them (no clever rephrasings).\n"
    "Do not contradict the scenario the user gave you, but you SHOULD invent "
    "concrete supporting detail that fits it. The point is a fully imagined "
    "scene, not a restatement of the seed idea."
)

WILDCARDIFY_SYSTEM_PROMPT = (
    "Rewrite the image-prompt sentence by replacing variable elements with "
    "__snake_case__ placeholders (double underscores on each side). The "
    "rewritten template will be sampled many times to produce a diverse set "
    "of images that all depict the SAME scenario — so the placeholders you "
    "pick should be the dimensions that most usefully vary between those "
    "images.\n"
    "Pick the most impactful VARIABLE dimensions only: subject action, pose, "
    "lighting, mood, composition, props, materials, color palette, era "
    "markers — whatever fits the sentence and is not already fixed.\n"
    "Fixed traits (provided in the user message) are mandatory specifics of "
    "this concept. The phrases in that list MUST remain verbatim concrete "
    "words in the rewritten sentence. You must NOT turn any fixed trait — or "
    "any near-synonym of one — into a placeholder. The axes those fixed "
    "traits sit on are also off-limits for placeholders.\n"
    "Stay within the placeholder count range you are given. To reach the "
    "minimum, wildcardify additional VARIABLE dimensions; to stay under the "
    "maximum, leave the rest as concrete words. Do not invent placeholders "
    "for things the original sentence does not mention.\n"
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
    "If a list of fixed traits is provided, no description may invite "
    "values that contradict, replace, or restate any of those fixed traits.\n"
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
    "If a list of fixed traits is provided, no value may contradict, "
    "replace, or restate any of them.\n"
    "If a list of forbidden scene elements is provided, no value may "
    "contain or imply any of them.\n"
    'Output JSON: {"values": ["...", "...", ...]}'
)


# Light per-step JSON schemas. No `pattern` constraints, no GBNF — failures
# surface as parse errors rather than getting masked by salvage paths.
BRIEF_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "refined_idea": {"type": "string"},
        "fixed_traits": {"type": "array", "items": {"type": "string"}},
        "forbidden_axes": {"type": "array", "items": {"type": "string"}},
        "scene_bans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["refined_idea", "fixed_traits", "forbidden_axes", "scene_bans"],
}

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


def _normalize_brief_list(raw, *, snake_case: bool = False) -> list[str]:
    """Dedupe a list of strings from an LLM brief reply. When `snake_case` is
    set, normalise to snake_case axis names (used for `forbidden_axes`)."""
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return out
    for entry in raw:
        s = str(entry or "").strip().strip('"').strip("'")
        if not s:
            continue
        if snake_case:
            n = _to_snake_case(s)
            if not n or not _KEY_RE.match(n):
                continue
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        else:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def _empty_brief() -> dict:
    return {"refined_idea": "", "fixed_traits": [], "forbidden_axes": [],
            "scene_bans": []}


def normalize_brief(raw) -> dict:
    """Coerce an arbitrary JSON object (LLM reply or user-edited widget) into
    a clean brief dict with the four expected keys. Unknown keys are dropped,
    list entries are deduped, axes are normalised to snake_case."""
    if not isinstance(raw, dict):
        return _empty_brief()
    refined = str(raw.get("refined_idea") or "").strip()
    fixed = _normalize_brief_list(raw.get("fixed_traits"))
    axes = _normalize_brief_list(raw.get("forbidden_axes"), snake_case=True)
    scene = _normalize_brief_list(raw.get("scene_bans"))
    return {"refined_idea": refined, "fixed_traits": fixed,
            "forbidden_axes": axes, "scene_bans": scene}


def _format_fixed_traits(fixed_traits: list[str]) -> str:
    """Render fixed_traits as a bullet list for injection into user messages.
    Returns "" when the list is empty so callers can skip the section."""
    if not fixed_traits:
        return ""
    return "\n".join(f"- {s}" for s in fixed_traits)


def llm_design_brief(idea: str, direction_text: str, server: dict,
                     seed: int = 0) -> tuple[dict, str]:
    """Step 0 — turn the raw user idea into a structured design brief.

    The brief locks specific phrases the user named (fixed_traits) so they
    cannot drift downstream; the axes those phrases sit on (forbidden_axes)
    are blocked from becoming wildcard placeholders. Returns (brief, raw_reply).
    Empty idea short-circuits to an empty brief without calling the LLM."""
    text = (idea or "").strip()
    if not text:
        return _empty_brief(), ""
    parts = [f"User idea:\n{text}"]
    if direction_text and direction_text.strip():
        parts.append(f"Direction:\n{direction_text.strip()}")
    parts.append("Output the JSON object now.")
    user = "\n\n".join(parts)
    raw = _server_call(server, BRIEF_SYSTEM_PROMPT, user,
                       request_json=True, seed=seed,
                       json_schema=BRIEF_JSON_SCHEMA)
    parsed = _extract_json_object(raw)
    return normalize_brief(parsed), raw


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
                     fixed_traits: list[str] | None = None,
                     ) -> tuple[str, str]:
    """Step 1 — turn the user idea + direction into a single image-prompt
    sentence. `scene_bans` lists scene elements the sentence must avoid;
    `fixed_traits` lists literal phrases the sentence must contain (so the
    LLM can't paraphrase away the user's required specifics).
    Returns (sentence, raw_reply)."""
    parts = [f"User idea:\n{(idea or '').strip() or '(no example provided)'}"]
    if direction_text and direction_text.strip():
        parts.append(f"Direction:\n{direction_text.strip()}")
    fixed = _format_fixed_traits(fixed_traits or [])
    if fixed:
        parts.append(f"Fixed traits (must appear verbatim in the sentence):\n{fixed}")
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
                           fixed_traits: list[str] | None = None,
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
        fixed_block = _format_fixed_traits(fixed_traits or [])
        if fixed_block:
            parts.append(
                "Fixed traits (must remain verbatim concrete words; do NOT "
                "wildcardify these or any near-synonym):\n"
                f"{fixed_block}"
            )
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
                           fixed_traits: list[str] | None = None,
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
    fixed = _format_fixed_traits(fixed_traits or [])
    if fixed:
        parts.append(
            "Fixed traits (no description may invite values that contradict, "
            f"replace, or restate any of these):\n{fixed}"
        )
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
                            fixed_traits: list[str] | None = None,
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
    fixed = _format_fixed_traits(fixed_traits or [])
    if fixed:
        parts.append(
            "Fixed traits (no value may contradict, replace, or restate "
            f"any of these):\n{fixed}"
        )
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
                           user_overrides: dict | None = None,
                           brief: dict | None = None) -> dict:
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
        "brief": normalize_brief(brief) if brief is not None else _empty_brief(),
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

        last_brief: dict = _empty_brief()
        if LAST_BRIEF_PATH.exists():
            try:
                last_brief = normalize_brief(json.loads(
                    LAST_BRIEF_PATH.read_text(encoding="utf-8")))
            except Exception:
                last_brief = _empty_brief()
        snap = build_manager_snapshot(display_cats,
                                      generated_prompt=last_template,
                                      brief=last_brief)

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
# Node 1b: LLMSamplingOptions — optional override layer for an LLM_SERVER.
#
# Wire LLMServerConfig → LLMSamplingOptions → Manager/Resolver to add fine-grained
# sampling controls (top_k, top_p, min_p, repeat_penalty, context_size) on top of
# the base server. The output is still an LLM_SERVER so it slots in anywhere the
# base server does.
#
# Also emits an OPTIONS bundle on a second output, shaped to match the contract
# used by the "Prompt Generator" node from
# https://github.com/abdozmantar/ComfyUI-Prompt-Manager — wire it into that
# node's `options` input to drive sampling/model/system_prompt from one place.
# =============================================================================
class LLMSamplingOptions:
    """ComfyUI node: layer extra sampling controls onto an existing LLM_SERVER.
    Outputs both an enhanced LLM_SERVER (for this package's Manager/Resolver)
    and an OPTIONS dict compatible with the external Prompt Generator node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLM_SERVER",),
            },
            "optional": {
                "use_model_default_sampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If ON, drop all sampling overrides and let the model use its built-in defaults. Temperature from the base server is still applied.",
                }),
                "temperature": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Override the server's temperature (-1 = keep server value). 0.0 = deterministic, 2.0 = very random.",
                }),
                "top_k": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Sample from top K tokens (0 = disabled / not sent).",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold (0.0 = not sent).",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability relative to the top token (0.0 = not sent).",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Repetition penalty (1.0 = no penalty / not sent).",
                }),
                "context_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 131072,
                    "step": 512,
                    "tooltip": "Context window size in tokens (0 = use server default / not sent).",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional system prompt. Honored by the external Prompt Generator node; the wildcard Manager/Resolver use their own per-step prompts and ignore this field.",
                    "tooltip": "Custom system prompt. Used only when wired into a node that reads it (e.g. Prompt Generator). Leave empty to use the consumer's defaults.",
                }),
                "show_everything_in_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print system prompt, user prompt, sampling options, and raw replies to the ComfyUI console.",
                }),
            },
        }

    RETURN_TYPES = ("LLM_SERVER", "OPTIONS")
    RETURN_NAMES = ("server", "options")
    FUNCTION = "build"
    CATEGORY = "prompt/wildcards"

    def build(self, server, use_model_default_sampling=False, temperature=-1.0,
              top_k=0, top_p=0.0, min_p=0.0, repeat_penalty=1.0,
              context_size=0, system_prompt="", show_everything_in_console=False):
        # Shallow copy so we don't mutate the upstream server dict in place.
        merged = dict(server)
        if temperature is not None and float(temperature) >= 0.0:
            merged["temperature"] = float(temperature)
        merged["use_model_default_sampling"] = bool(use_model_default_sampling)
        if not use_model_default_sampling:
            if top_k and int(top_k) > 0:
                merged["top_k"] = int(top_k)
            if top_p and float(top_p) > 0.0:
                merged["top_p"] = float(top_p)
            if min_p and float(min_p) > 0.0:
                merged["min_p"] = float(min_p)
            if repeat_penalty and float(repeat_penalty) > 1.0:
                merged["repeat_penalty"] = float(repeat_penalty)
            if context_size and int(context_size) > 0:
                merged["context_size"] = int(context_size)
        merged["show_everything_in_console"] = bool(show_everything_in_console)
        if system_prompt and system_prompt.strip():
            # Stored on the server dict but ignored by _server_call — kept here
            # purely so a downstream node could pick it up if it wanted to.
            merged["system_prompt"] = system_prompt

        # OPTIONS bundle — shape matches the external Prompt Generator node's
        # contract so users can wire `options` straight into it.
        backend = server.get("backend", "ollama")
        options: dict = {
            "llm_backend": "ollama" if backend == "ollama" else "llama.cpp",
            "use_model_default_sampling": bool(use_model_default_sampling),
            "show_everything_in_console": bool(show_everything_in_console),
        }
        if server.get("model"):
            options["model"] = server["model"]
        if system_prompt and system_prompt.strip():
            options["system_prompt"] = system_prompt
        if temperature is not None and float(temperature) >= 0.0:
            options["temperature"] = float(temperature)
        elif "temperature" in server:
            options["temperature"] = float(server["temperature"])
        if not use_model_default_sampling:
            if top_k and int(top_k) > 0:
                options["top_k"] = int(top_k)
            if top_p and float(top_p) > 0.0:
                options["top_p"] = float(top_p)
            if min_p and float(min_p) > 0.0:
                options["min_p"] = float(min_p)
            if repeat_penalty and float(repeat_penalty) > 1.0:
                options["repeat_penalty"] = float(repeat_penalty)
        if context_size and int(context_size) > 0:
            options["context_size"] = int(context_size)

        return (merged, options)


# =============================================================================
# Node 1c: LLMPromptGenerator — text/image → enhanced prompt, using LLM_SERVER.
#
# Important: this node never starts or stops a llama.cpp server. It only sends
# HTTP requests to the endpoint already configured upstream in LLMServerConfig
# (optionally enhanced by LLMSamplingOptions). If you want a model loaded, load
# it once outside ComfyUI and point LLMServerConfig at the running endpoint.
# =============================================================================
_PG_DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "Enhance Prompt (Image)": (
        "You are an expert prompt engineer for text-to-image diffusion models. "
        "Take the user's idea and rewrite it as a single polished prompt: "
        "concrete subject, scene, lighting, composition, materials, style. "
        "Output the prompt only — no preamble, no quotes, no explanations."
    ),
    "Enhance Prompt (Video)": (
        "You are an expert prompt engineer for text-to-video models. "
        "Take the user's idea and rewrite it as a single polished prompt that "
        "covers subject, motion, camera movement, lighting, and style. "
        "Output the prompt only — no preamble, no quotes, no explanations."
    ),
    "Analyze Image": (
        "Describe the provided image as a prompt suitable for regenerating it "
        "with a text-to-image model. Concrete subject, scene, lighting, "
        "composition, style. Output the prompt only — no preamble."
    ),
    "Analyze Image with Prompt": (
        "Follow the user's instructions to describe or analyze the provided "
        "image. Be concrete and visual. Output only what the user asked for."
    ),
}

_PG_DEFAULT_IMAGE_ACTION = "Describe this image in vivid, concrete detail."

_PG_JSON_SUFFIX = (
    "\n\nReturn the result as a single JSON object with keys "
    '"subject", "scene", "style", "lighting", "composition". '
    "No prose outside the JSON."
)

_PG_VISION_MODES = {"Analyze Image", "Analyze Image with Prompt"}


class LLMPromptGenerator:
    """ComfyUI node: enhance a text prompt or analyze an image using the LLM
    endpoint configured in LLMServerConfig. Drop-in replacement for the external
    Prompt Generator node, but it never spawns its own llama.cpp subprocess —
    it just talks to whichever endpoint you've already set up."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLM_SERVER",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible generation. 0 lets the server pick.",
                    "control_after_generate": True,
                }),
            },
            "optional": {
                "mode": (list(_PG_DEFAULT_SYSTEM_PROMPTS.keys()), {
                    "default": "Enhance Prompt (Image)",
                    "tooltip": "Enhance text prompt | Analyze image | Analyze image with custom instructions",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Your prompt or instructions…",
                    "tooltip": "Required for Enhance Prompt modes; optional for Analyze Image modes.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Required for the Analyze Image modes.",
                }),
                "format_as_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append JSON-formatting instructions to the system prompt.",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Ask the model for explicit reasoning. Captured separately into the `thoughts` output. Ignored by models that don't support it.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Override the default system prompt for this mode (optional).",
                    "tooltip": "Custom system prompt. Leave empty to use the per-mode default.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 180,
                    "min": 10,
                    "max": 1800,
                    "step": 10,
                    "tooltip": "HTTP timeout for the LLM call.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "thoughts")
    FUNCTION = "generate"
    CATEGORY = "prompt/wildcards"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    def generate(self, server, seed, mode="Enhance Prompt (Image)", prompt="",
                 image=None, format_as_json=False, enable_thinking=False,
                 system_prompt="", timeout_seconds=180):
        is_vision = mode in _PG_VISION_MODES

        if not is_vision and not (prompt and prompt.strip()):
            raise RuntimeError(f"Mode '{mode}' requires a non-empty prompt.")
        if is_vision and image is None:
            raise RuntimeError(
                f"Mode '{mode}' requires an image. Connect an IMAGE input or "
                f"switch to an Enhance Prompt mode."
            )

        # System prompt: explicit user input wins, then any system_prompt that
        # arrived on the LLM_SERVER (e.g. from LLMSamplingOptions), then the
        # per-mode default baked in here.
        if system_prompt and system_prompt.strip():
            sys_prompt = system_prompt.strip()
        elif server.get("system_prompt"):
            sys_prompt = server["system_prompt"]
        else:
            sys_prompt = _PG_DEFAULT_SYSTEM_PROMPTS[mode]

        if format_as_json:
            sys_prompt = sys_prompt + _PG_JSON_SUFFIX

        # User content per mode.
        if mode == "Analyze Image":
            user_content = _PG_DEFAULT_IMAGE_ACTION
        elif mode == "Analyze Image with Prompt":
            user_content = (prompt.strip() if prompt and prompt.strip()
                            else _PG_DEFAULT_IMAGE_ACTION)
        else:
            user_content = prompt

        image_b64 = _encode_comfy_image_to_b64(image) if is_vision else None

        debug = bool(server.get("show_everything_in_console", False))
        if debug:
            print(f"[LLMPromptGenerator] backend={server.get('backend')} "
                  f"endpoint={server.get('endpoint')} model={server.get('model')} "
                  f"mode={mode} vision={'yes' if is_vision else 'no'} "
                  f"thinking={enable_thinking} json={format_as_json}")
            print(f"[LLMPromptGenerator] system: {sys_prompt}")
            print(f"[LLMPromptGenerator] user:   {user_content}")

        try:
            content, thinking = _chat_with_vision(
                server=server,
                system=sys_prompt,
                user=user_content,
                image_b64=image_b64,
                seed=int(seed) if seed else 0,
                enable_thinking=bool(enable_thinking),
                timeout=int(timeout_seconds),
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LLMPromptGenerator: could not reach {server.get('endpoint')!r} "
                f"({e}). Make sure your llama.cpp / Ollama server is running on "
                f"that endpoint."
            )

        if debug:
            print(f"[LLMPromptGenerator] thinking: {thinking!r}")
            print(f"[LLMPromptGenerator] output:   {content!r}")

        if not content:
            content = prompt if prompt else ""

        return (content, thinking)


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
                # Hidden JSON widget mirroring the design brief
                # (refined_idea, fixed_traits, forbidden_axes, scene_bans).
                # Empty/default → the Manager calls the LLM to fill it.
                # Non-empty → the Manager treats it as the user's edited brief
                # and skips the LLM brief call. The UI panel reads/writes here.
                "design_brief": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "Design brief — edited via the brief panel on the node.",
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
               system_prompt_override, categories,
               design_brief=""):
        direction = (direction or "").strip() or "none"
        direction_text = resolve_direction(direction)
        negative = (negative_prompt or "").strip()
        forbidden_names = _parse_forbidden_names(forbidden_placeholders or "")

        # Parse the user-edited brief (or default {}). Non-empty means the
        # user has reviewed/edited the brief and the LLM brief step is skipped;
        # empty means we run the LLM to generate one. The brief is the single
        # source of truth for fixed_traits / forbidden_axes / scene_bans that
        # the user can edit, on top of the implicit derivation from the idea.
        user_brief: dict | None = None
        brief_text = (design_brief or "").strip()
        if brief_text and brief_text not in ("{}", ""):
            try:
                parsed_brief = json.loads(brief_text)
                user_brief = normalize_brief(parsed_brief)
                # An object whose fields are all empty is treated as "no brief"
                # so the LLM still runs (avoids the UI default trapping users).
                if not (user_brief["refined_idea"]
                        or user_brief["fixed_traits"]
                        or user_brief["forbidden_axes"]
                        or user_brief["scene_bans"]):
                    user_brief = None
            except Exception as e:
                print(f"[LLMWildcardManager] Bad design_brief JSON: {e}")
                user_brief = None

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
        fixed_traits: list[str] = []
        brief: dict = _empty_brief()
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
            # Pull the cached brief so the Resolver still gets fixed_traits /
            # scene_bans even on the locked path.
            if user_brief is not None:
                brief = user_brief
            elif LAST_BRIEF_PATH.exists():
                try:
                    brief = normalize_brief(json.loads(
                        LAST_BRIEF_PATH.read_text(encoding="utf-8")))
                except Exception:
                    brief = _empty_brief()
            fixed_traits = list(brief.get("fixed_traits") or [])
            scene_bans = list(brief.get("scene_bans") or [])
        else:
            axis_bans: list[str] = []
            try:
                # Step 0 — design brief. Distils the user's idea into
                # refined_idea + fixed_traits (mandatory specifics that must
                # appear verbatim) + forbidden_axes (axis names already pinned
                # by fixed_traits, so they MUST NOT become wildcards) +
                # scene_bans (things to keep out, inferred from negation in
                # the idea). If the user already edited the brief on the node,
                # we skip this LLM call and use their version as-is.
                if user_brief is not None:
                    brief = user_brief
                    raw_sections.append((
                        "brief",
                        "(LLM brief skipped — using the edited brief from the node)",
                    ))
                else:
                    brief, raw_brief = llm_design_brief(
                        example_prompt or "", direction_text, server,
                        seed=effective_seed,
                    )
                    raw_sections.append(("brief", raw_brief))

                fixed_traits = list(brief.get("fixed_traits") or [])
                brief_forbidden_axes = list(brief.get("forbidden_axes") or [])
                brief_scene_bans = list(brief.get("scene_bans") or [])

                # Step 0b — split the raw negative prompt (user-supplied free
                # text) into structured bans. Skipped when the user left the
                # negative_prompt empty.
                raw_neg_axis_bans: list[str] = []
                raw_neg_scene_bans: list[str] = []
                if negative:
                    raw_neg_axis_bans, raw_neg_scene_bans, raw_parse_neg = (
                        llm_parse_negative(negative, server, seed=effective_seed)
                    )
                    raw_sections.append(("parse_negative", raw_parse_neg))
                axis_bans = raw_neg_axis_bans

                # Merge scene_bans from the brief + the parsed negative prompt.
                scene_bans = []
                seen_scene: set[str] = set()
                for s in brief_scene_bans + raw_neg_scene_bans:
                    key = s.lower()
                    if key in seen_scene:
                        continue
                    seen_scene.add(key)
                    scene_bans.append(s)

                # Merge auto-derived axis bans (from brief + negative prompt)
                # into the user's explicit forbidden_placeholders list. This
                # is the actual fix for required attributes leaking into
                # placeholders: 'a woman in a polkadot dress' → the brief
                # produces forbidden_axes=['pattern'] → __pattern__ is denied.
                effective_forbidden_names: list[str] = []
                seen_forbidden: set[str] = set()
                for n in brief_forbidden_axes + raw_neg_axis_bans + forbidden_names:
                    if n and n not in seen_forbidden:
                        seen_forbidden.add(n)
                        effective_forbidden_names.append(n)

                # Use the refined idea (when present) as the input for the
                # drafting step — it keeps the LLM tighter to what the user
                # actually asked for, while the raw idea remains available as
                # extra context. Falls back to the raw idea when the brief
                # produced nothing.
                idea_for_draft = (brief.get("refined_idea") or "").strip() \
                    or (example_prompt or "")

                # Step 1 — draft the prompt sentence from idea + direction.
                draft, raw_draft = llm_draft_prompt(
                    idea_for_draft, direction_text, server,
                    seed=effective_seed,
                    scene_bans=scene_bans,
                    fixed_traits=fixed_traits,
                )
                raw_sections.append(("draft", raw_draft))

                # Step 2 — wildcardify with the merged deny-list. Fixed traits
                # are passed so the LLM is told not to wildcardify any of
                # them; if it ignores the instruction, the forbidden_axes from
                # the brief still strip the offending placeholders post-hoc.
                template, used_names, raw_wildcardify = llm_wildcardify_prompt(
                    draft, server, seed=effective_seed,
                    min_categories=min_cats,
                    max_categories=max_cats,
                    forbidden_names=effective_forbidden_names,
                    fixed_traits=fixed_traits,
                )
                raw_sections.append(("wildcardify", raw_wildcardify))

                # Step 3 — describe each wildcard.
                descs, raw_describe = llm_describe_wildcards(
                    used_names, server, seed=effective_seed,
                    idea=idea_for_draft,
                    direction_text=direction_text,
                    template=template,
                    scene_bans=scene_bans,
                    fixed_traits=fixed_traits,
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

        # Persist the brief only when one was produced (success path or
        # user-edited brief on the locked path). Failed runs should not
        # poison the cache. The Resolver-only path picks this up too.
        if status == "ok" and (brief.get("refined_idea")
                               or brief.get("fixed_traits")
                               or brief.get("forbidden_axes")
                               or brief.get("scene_bans")):
            try:
                LAST_BRIEF_PATH.write_text(
                    json.dumps(brief, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[LLMWildcardManager] Could not persist brief: {e}")

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
            "fixed_traits": list(fixed_traits),
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
            brief=brief,
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
        fixed_traits: list[str] = []
        intended_names: list[str] = []
        if isinstance(prompts, dict):
            flair_text = prompts.get("flair") or ""
            negative_text = prompts.get("negative") or ""
            raw_scene = prompts.get("scene_bans") or []
            if isinstance(raw_scene, list):
                scene_bans = [str(s) for s in raw_scene if str(s).strip()]
            raw_fixed = prompts.get("fixed_traits") or []
            if isinstance(raw_fixed, list):
                fixed_traits = [str(s) for s in raw_fixed if str(s).strip()]
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
                    fixed_traits=fixed_traits,
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
                            fixed_traits=fixed_traits,
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
    "LLMSamplingOptions": LLMSamplingOptions,
    "LLMPromptGenerator": LLMPromptGenerator,
    "LLMWildcardManager": LLMWildcardManager,
    "LLMWildcardResolver": LLMWildcardResolver,
    "LLMWildcardReport": LLMWildcardReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMServerConfig": "🎲 LLM Server Config",
    "LLMSamplingOptions": "🎲 LLM Sampling Options",
    "LLMPromptGenerator": "🎲 LLM Prompt Generator",
    "LLMWildcardManager": "🎲 LLM Wildcard Manager",
    "LLMWildcardResolver": "🎲 LLM Wildcard Resolver",
    "LLMWildcardReport": "🎲 LLM Wildcard Report",
}
