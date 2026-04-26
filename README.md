# 🎲 LLM Wildcard Manager for ComfyUI

A small set of ComfyUI nodes that let an LLM **design a prompt template
with `__wildcard__` placeholders** and then **fill those placeholders one at
a time, in isolation, with explicit anti-repetition** — backed by reusable
on-disk wildcard files.

**Four nodes:**

| Node                         | Purpose |
|------------------------------|---------|
| 🎲 LLM Server Config         | Single place to configure the LLM backend (endpoint, model, key, temperature). Wired into both Manager and Resolver. |
| 🎲 LLM Wildcard Manager      | Designs the prompt template. Asks the LLM to rewrite your idea as a template with `__wildcard__` placeholders, plus a description for each placeholder. |
| 🎲 LLM Wildcard Resolver     | Fills `__wildcard__` slots: reuses values from disk or asks the LLM for a fresh, anti-repetition value (one slot at a time). |
| 🎲 LLM Wildcard Report       | Renders the resolver's per-slot results as a structured collapsible view + raw text panel. Outputs counters for routing. |

## Why this exists

When you ask an LLM to "enhance this prompt," it sees the whole prompt and
anchors on it. Run it twice and you get two near-identical outputs.

This pack flips that: the **Manager** turns your idea into a template with
small variable parts (`__hair__`, `__location__`, …), and the **Resolver**
fills each variable part with a fresh value the LLM has never produced for
that category before. The downstream prompt has genuine variance instead of
being a re-skin of the same sentence.

## Install

### Manual

```
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_GITHUB_USER/comfyui-llm-wildcard-manager
```

No extra Python deps — uses stdlib `urllib`.

Restart ComfyUI. Nodes appear under **prompt → wildcards**.

A starter workflow is in [`example_workflows/llm_wildcard_basic.json`](example_workflows/llm_wildcard_basic.json).

> **Upgrading from 0.2.x:** the 0.3 release replaces `LLMWildcardPromptConfig`
> with the new Manager + Server Config split. Saved workflows that referenced
> the old PromptConfig node, or the Resolver's old backend/endpoint/model
> widgets, will need re-wiring. Open the new example workflow as a starting
> point.

## Recommended wiring

```
[LLM Server Config] --server--> [LLM Wildcard Manager] --prompt_template--> [LLM Wildcard Resolver] --resolved_prompt--> [CLIP Text Encode]
                  \--server-----------------------------/                  \--report------> [LLM Wildcard Report]
                                              \--prompts---------/
```

The **Server Config** is wired into both Manager and Resolver so endpoint
settings live in one node. The Manager hands the Resolver both:

- `prompt_template` — the template the Manager designed (wire to `template`).
- `prompts` — a bundle with the system prompt + flair + per-category
  descriptions (wire to the optional `prompts` socket).

## Wildcard files

Stored in `ComfyUI/wildcards/<name>.txt`, one value per line. Compatible with
the Impact Pack and Santodan Wildcard Manager file formats. The Resolver
creates files automatically as it generates new values.

## Template syntax

```
__hair__       reuse a stored value (or generate if file empty / mode=force_new)
__!hair__      force LLM to generate a new value, append to file
```

## Modes (Resolver)

| Mode             | Behavior                                                   |
|------------------|------------------------------------------------------------|
| `reuse_existing` | Always pick from file. Generates only if file is empty.    |
| `force_new`      | Every slot is regenerated and appended to its file.        |
| `hybrid`         | Reuse by default; only `__!name__` slots force generation. |

## Backends (Server Config)

- **ollama** — `endpoint = http://localhost:11434`, `model = llama3.1` (or any pulled model), `api_key` blank.
- **llamacpp** — `endpoint = http://localhost:8080/v1`, `model` can be left blank, `api_key` blank.
- **openai_compatible** — works with OpenAI, LM Studio, vLLM, OpenRouter, etc. Set `endpoint` to the base URL ending in `/v1`, set `model`, set `api_key`.

### Using llama.cpp in Docker

```
docker run --rm -p 8080:8080 \
  -v /path/to/models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/your-model.gguf -c 4096 --host 0.0.0.0 --port 8080
```

In the Server Config node, set `backend = llamacpp` and `endpoint = http://localhost:8080/v1`.

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

If that returns JSON with `choices[0].message.content`, the nodes will work.

## Node reference

### 🎲 LLM Server Config

The single source of truth for LLM settings. One output: `server` (a
`LLM_SERVER` bundle). Wire into both Manager and Resolver.

Fields: `backend`, `endpoint`, `model`, `api_key`, `temperature`.

### 🎲 LLM Wildcard Manager

Designs the prompt template. On each run it asks the LLM to rewrite your
`example_prompt` as a template with `__wildcard__` placeholders, and to
return a description for each placeholder it invented.

Inputs:

- **`server`** — wire from the Server Config.
- **`example_prompt`** — your idea. The Manager turns this into a template
  with variable parts replaced by `__wildcards__`.
- **`seed`** — `0` re-rolls the template every queue run (new template, new
  category set). Non-zero is reproducible: same inputs → same template.
- **`direction`** — *free-text with autocomplete*. Pick a built-in preset
  (`photoreal`, `cinematic`, `editorial`, `vintage_film`, `noir`, `cyberpunk`,
  `fantasy`, `anime`, `dreamlike`, `minimal`, `sfw_strict`) or type your own
  steering text directly.
- **`extra_flair`** — optional extra steering, appended after the direction.
- **`system_prompt_override`** — leave empty for the built-in template-design
  system prompt; fill to fully replace it (advanced).
- **`categories`** — JSON object of `{name: description}`. **User overrides
  only.** Edited via the table UI on the node. User overrides win over
  LLM-suggested descriptions.

Outputs:

- **`prompt_template`** — wire into the Resolver's `template` input.
- **`prompts`** — `WILDCARD_PROMPTS` bundle. Wire into the Resolver's
  optional `prompts` input. Carries the system prompt, flair, and the merged
  category descriptions.

UI:

- **Generated prompt** panel at the top shows the template the LLM produced
  (with wildcard tokens highlighted).
- **Categories** table beneath shows every category in the current template
  + every user override + every category that has entries on disk. Each row:
  expand chevron · name · description · entry-count badge · OVERRIDE tag if
  user-edited · remove button.
- **↻ Refresh disk** re-reads the wildcards folder without re-queuing.
- **+ Add category** appends a fresh override row.
- The disk path of the wildcards folder is shown so you always know where
  values are written.

### 🎲 LLM Wildcard Resolver

Fills `__wildcard__` slots in the template.

Inputs:

- **`server`** — wire from the Server Config.
- **`template`** — wire from the Manager's `prompt_template`, or type your
  own template directly into the widget.
- **`mode`** — `hybrid` (recommended), `reuse_existing`, or `force_new`.
- **`max_per_category`** — soft cap. Once a category file hits this many
  entries, the resolver stops appending and starts reusing.
- **`seed`** — random seed for the choice between existing values.
- **`fix_seed`** — when **off** (default), every queue run re-rolls the
  fills (regardless of seed). When **on**, the resolver is fully
  deterministic: same template + same seed = same final values.
- **`prompts`** *(optional input)* — wire from the Manager. Carries the
  system prompt + flair + category descriptions used per-slot.

Outputs:

- **`resolved_prompt`** — the final template with all wildcards replaced.
  Wire into your CLIP Text Encode positive input.
- **`report`** — full text report. Wire into the Report node.

### 🎲 LLM Wildcard Report

Renders the resolver's run results inside the node body and re-emits parsed
counters for routing.

Inputs:

- **`report`** — wire from the Resolver's `report` output.

Outputs:

| Output      | Type   | Meaning                                                        |
|-------------|--------|----------------------------------------------------------------|
| `summary`   | STRING | The complete report text.                                      |
| `generated` | INT    | Slots that produced new values (and were appended).            |
| `reused`    | INT    | Slots that reused an existing value (incl. cap-hits).          |
| `errors`    | INT    | Slots where the LLM call failed.                               |
| `total`     | INT    | Total wildcard slots resolved.                                 |

UI:

- Header bar with `total / generated / reused / errors` counters.
- One row per slot: status badge · name · final value (truncated). Click the
  expand chevron to reveal the prompt sent to the LLM, the LLM's raw reply,
  any retry, and any error.
- **Raw report** textarea at the bottom for copy-paste.

Useful for routing — e.g. only save the workflow image when `errors == 0`.

## Anti-repetition guarantees

Each per-slot LLM call receives:
1. Only the category name and category description.
2. The full list of existing values flagged as **forbidden / do not paraphrase**.
3. No other context from the surrounding prompt.

If the model returns a duplicate anyway, the Resolver retries once with a
higher temperature.

## Seed behavior at a glance

| Where         | Setting              | Effect                                                                |
|---------------|----------------------|-----------------------------------------------------------------------|
| Manager       | `seed = 0`           | Re-roll the prompt template + category set every queue run.           |
| Manager       | `seed != 0`          | Reproducible: same inputs + same seed → same template + categories.   |
| Resolver      | `fix_seed = false`   | Re-roll the per-slot fills every queue run (regardless of seed).      |
| Resolver      | `fix_seed = true`    | Fully deterministic: same template + same seed → same final values.   |
