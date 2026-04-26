# 🎲 LLM Wildcard Manager for ComfyUI

A small set of ComfyUI nodes that fill `__wildcard__` slots in a template by
either reusing values from disk or asking an LLM to invent new ones — with
explicit anti-repetition.

**Three nodes:**

| Node | Purpose |
|------|---------|
| 🎲 LLM Wildcard Resolver       | The core: resolves `__wildcard__` slots using cache + LLM. |
| 🎲 LLM Wildcard Prompt Config  | Override the LLM system prompt and define category descriptions in-graph. |
| 🎲 LLM Wildcard Report         | Parse the resolver's report into counts (`generated`, `reused`, `errors`, `total`) and a clean summary. |

## Why this exists

When you ask an LLM to "enhance this prompt," it sees the whole prompt and
anchors on it. Run it twice and you get two near-identical outputs. This node
flips the model: instead of enhancing the *prompt*, it generates one *wildcard
value at a time*, in isolation, with the existing values listed as forbidden.
The downstream prompt then has genuinely fresh variance to feed into your
prompt-enhancer.

## Install

### Via ComfyUI Manager (recommended once published)

Search for **LLM Wildcard Manager** in ComfyUI Manager and click Install.

### Manual

```
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_GITHUB_USER/comfyui-llm-wildcard-manager
```

No extra Python deps — uses stdlib `urllib`.

Restart ComfyUI. The node appears under **prompt → wildcards → 🎲 LLM Wildcard Resolver**.

An example workflow is included in [`example_workflows/`](example_workflows/llm_wildcard_basic.json).

## Wildcard files

Stored in `ComfyUI/wildcards/<name>.txt`, one value per line. Compatible with
the Impact Pack and Santodan Wildcard Manager file format. The node creates
files automatically as it generates.

## Template syntax

```
__hair__       reuse a stored value (or generate if file empty / mode=force_new)
__!hair__      force LLM to generate a new value, append to file
```

## Modes

| Mode             | Behavior                                                   |
|------------------|------------------------------------------------------------|
| `reuse_existing` | Always pick from file. Generates only if file is empty.    |
| `force_new`      | Every slot is regenerated and appended to its file.        |
| `hybrid`         | Reuse by default; only `__!name__` slots force generation. |

## Backends

- **ollama** — `endpoint = http://localhost:11434`, `model = llama3.1` (or any pulled model), `api_key` blank.
- **llamacpp** — `endpoint = http://localhost:8080/v1`, `model` can be left blank (llama.cpp serves whatever GGUF you loaded), `api_key` blank.
- **openai_compatible** — works with OpenAI, LM Studio, vLLM, OpenRouter, etc. Set `endpoint` to the base URL ending in `/v1` (e.g. `https://api.openai.com/v1`), set `model`, set `api_key`.

### Using llama.cpp in Docker

The official server image:

```
docker run --rm -p 8080:8080 \
  -v /path/to/models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/your-model.gguf -c 4096 --host 0.0.0.0 --port 8080
```

In the node, set `backend = llamacpp` and `endpoint = http://localhost:8080/v1`.

**Networking gotchas:**
- If **ComfyUI also runs in Docker**, `localhost` inside the ComfyUI container points to itself, not to the llama.cpp container. Use one of:
  - `http://host.docker.internal:8080/v1` (Docker Desktop on Windows/macOS, and Linux with `--add-host=host.docker.internal:host-gateway`)
  - The llama.cpp container's name on a shared user-defined network: `http://llamacpp:8080/v1`
- If **only llama.cpp is in Docker** and ComfyUI runs on the host, `http://localhost:8080/v1` works as long as you published the port with `-p 8080:8080`.

**Sanity check from your terminal:**

```
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"say hi"}]}'
```

If that returns JSON with a `choices[0].message.content`, the node will work.

## Category overrides

The optional `category_overrides` input takes a JSON object mapping wildcard
names to descriptions. Use this to teach the LLM what shape of value belongs
in a custom category without editing `wildcard_categories.json`:

```json
{
  "lens": "A camera lens descriptor like '85mm f/1.4' or 'wide-angle 24mm'",
  "mood": "A single mood/atmosphere word"
}
```

## Wiring it into your workflow

Minimal:

```
[LLM Wildcard Resolver]  --resolved_prompt-->  [CLIP Text Encode]
```

Full pipeline with prompt config + report:

```
[LLM Wildcard Prompt Config] --prompts-->  [LLM Wildcard Resolver] --report-->  [LLM Wildcard Report]
                                                  |
                                                  +--resolved_prompt-->  [CLIP Text Encode]
```

Your downstream prompt generator's "Change this prompt:" instruction now
operates on a freshly varied prompt every queue run, so it can't iterate on
the same input.

### LLM Wildcard Prompt Config

Three fields:

- **`flair`** — a short steering phrase *appended* to the default LLM system prompt. Use this 95% of the time. Examples:
  - `Lean cyberpunk neon noir, no clichés.`
  - `Subjects must always be European-ish.`
  - `Keep outputs strictly SFW.`
- **`system_prompt_override`** — full replacement of the built-in system prompt. Leave empty unless you really need to change the rules of the game.
- **`category overrides`** — a clickable add/remove table. Each row is *name* + *description sent to the LLM*. Examples:
  - `lens` → `A camera lens descriptor like '85mm f/1.4'`
  - `mood` → `A single mood/atmosphere word`

The table is backed by a JSON string under the hood, so headless / API workflows keep working — just edit the JSON directly if no UI is available.

Wire `prompts` into the resolver's optional `prompts` input. Anything left blank falls back to defaults.

### LLM Wildcard Report

Wire the resolver's `report` output into this node's `report` input. The full report is rendered **inside the node body** (read-only textarea, resizable) and re-emitted as outputs:

| Output      | Type   | Meaning                                                        |
|-------------|--------|----------------------------------------------------------------|
| `summary`   | STRING | The complete report text.                                      |
| `generated` | INT    | Count of slots that produced new values (and were appended).   |
| `reused`    | INT    | Count of slots that reused an existing value (incl. cap-hits). |
| `errors`    | INT    | Count of slots where the LLM call failed.                      |
| `total`     | INT    | Total wildcard slots resolved.                                 |

For each slot you'll see: status, final value, pool size, the prompt sent to the LLM (category, description, forbidden count, model, temperature), the LLM's raw reply, and any retry / error details. Useful for debugging a model that keeps producing the wrong shape of output.

Useful for routing: e.g. only save the workflow image when `errors == 0`.

## Anti-repetition guarantees

Each LLM call receives:
1. Only the category name and category description.
2. The full list of existing values flagged as **forbidden / do not paraphrase**.
3. No other context from the surrounding prompt.

If the model returns a duplicate anyway, the node retries once with a higher
temperature.

## Seed behavior

- `seed = 0` — fresh randomness every queue run.
- `seed != 0` — reproducible cache picks (the LLM call itself is still
  non-deterministic unless your backend supports a seed parameter).
