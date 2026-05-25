# 🎲 LLM Wildcard Manager for ComfyUI

A small set of ComfyUI nodes that let an LLM **design a prompt template
with `__wildcard__` placeholders** and then **fill those placeholders one at
a time, in isolation, with explicit anti-repetition** — backed by reusable
on-disk wildcard files.

**Core nodes:**

| Node                          | Purpose |
|-------------------------------|---------|
| 🎲 LLM Server Config          | Single place to configure the LLM backend (endpoint, model, key, temperature). Wired into both Manager and Resolver. |
| 🎲 LLM Wildcard Manager       | Designs the prompt template. Asks the LLM to rewrite your idea as a template with `__wildcard__` placeholders, plus a description for each placeholder. |
| 🎲 LLM Wildcard Template Builder | *Structure-first alternative to the Manager.* You hand-author the prompt's **shape** in a block editor (sentences + wildcard groups with count sliders); the LLM only writes the sentence text and per-slot descriptions. Same outputs as the Manager. |
| 🎲 LLM Wildcard Resolver      | Fills `__wildcard__` slots: reuses values from disk or asks the LLM for a fresh, anti-repetition value (one slot at a time). |
| 🎲 LLM Wildcard Report        | Renders the resolver's per-slot results as a structured collapsible view + raw text panel. Outputs counters for routing. |

> **Manager vs. Template Builder:** the Manager turns a free-text *idea* into a
> template and decides the structure for you. The Template Builder flips that —
> *you* decide the structure (a sentence, then 3 character wildcards, an action
> sentence, 2 pose wildcards, …) and the LLM fills in the prose + descriptions.
> Use one or the other into the Resolver; they share the same output sockets.

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
[LLM Server Config] --server--> [LLM Wildcard Manager] --prompt_template--> [LLM Wildcard Resolver] --resolved_prompt--> [CLIP Text Encode (positive)]
                  \--server-----------------------------/                  \--report------> [LLM Wildcard Report]
                                              \--prompts---------/         \--negative_prompt-> [CLIP Text Encode (negative)]
                  \-----------------------------(Manager) --negative_prompt-> [CLIP Text Encode (negative)] (alternative, identical output)
```

Prefer to design the prompt's **structure** yourself? Swap the Manager for the
**Template Builder** — it has the same three output sockets:

```
[LLM Server Config] --server--> [LLM Wildcard Template Builder] --prompt_template--> [LLM Wildcard Resolver]
                              \--server-----------------------/  \--prompts--------/
                                                                 \--negative_prompt-> [CLIP Text Encode (negative)]
```

The `negative_prompt` output is a stable, deterministic comma-separated
deny-list — it is not LLM-rewritten — so the negative side of your CLIP
encode pair stays predictable regardless of how the positive prompt
re-rolls each queue.

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
- **`lock_template`** — when **on**, the Manager skips the LLM entirely and
  reuses the last generated template + categories. Re-queue to get fresh
  random wildcard fills from the Resolver without changing the prompt. When
  **off** (default), the Manager regenerates every queue (modulated by
  `seed`).
- **`seed`** — `0` re-rolls the template every queue run (new template, new
  category set). Non-zero is reproducible: same inputs → same template.
- **`direction`** — *free-text with autocomplete*. Pick a built-in preset
  (`photoreal`, `cinematic`, `editorial`, `vintage_film`, `noir`, `cyberpunk`,
  `fantasy`, `anime`, `dreamlike`, `minimal`, `sfw_strict`) or type your own
  steering text directly.
- **`negative_prompt`** — traits to avoid, applied at every step:
  1. The drafted sentence won't include them (with a retry + clause-strip
     backstop when the LLM ignores the instruction — `no phones` now
     actually keeps "holding a phone" out of the template).
  2. Aspects you pin here won't become wildcards (e.g. `no old or
     middle-aged people` stops `__age__` from being created when your idea
     already says "young woman").
  3. Each wildcard description gets an explicit exclusion clause so the
     resolver's per-slot LLM call can't drift into them either.
  4. The same list is compiled into a deterministic `negative_prompt`
     output you wire into CLIP Text Encode (negative).
- **`min_categories`** — minimum number of `__wildcard__` placeholders the
  Manager will accept. Enforced by retrying the wildcardify step (up to
  twice) with the previous result and an explicit "you produced X, need at
  least N — add more" instruction if the LLM falls short.
- **`max_categories`** — hard cap on how many `__wildcard__` placeholders the
  template may contain. The LLM is asked to pick the most impactful variables
  and leave the rest as concrete words; if it exceeds the cap, the surplus
  placeholders are demoted to plain words deterministically. Lower values =
  more focused prompts.
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
- **`negative_prompt`** — deterministic comma-separated string built from
  your `negative_prompt` widget + the parsed `scene_bans` + axis bans +
  `forbidden_placeholders`. Not LLM-rewritten — wire it straight into your
  CLIPTextEncode (negative) so the image model sees the same forbidden list
  every run. Also passed through to the Resolver in the `prompts` bundle.

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

### 🎲 LLM Wildcard Template Builder

A structure-first sibling of the Manager. Instead of letting the LLM decide the
prompt's shape, you compose it from an ordered list of **blocks** in the node's
editor, and the LLM is called only to (a) draft any sentence you leave blank and
(b) describe each wildcard slot. Counts and placement are resolved
deterministically in Python, so the requested shape is exact — the LLM can't
miscount your "3 character / 2 pose" structure.

Block types:

- **Sentence block** — a literal sentence (no wildcards). Type the text to use
  it verbatim, or leave it empty and the LLM writes one biased to the block's
  abstract **role** (scene / action / setting / …).
- **Wildcard group** — a **count slider** (1–12), an abstract **role** (pick
  from a predefined list, or leave *undefined* for pure structure), and a
  **new** checkbox (emits `__!name__` to force a fresh value every run). A group
  of count *N* emits *N* uniquely-numbered placeholders (`__character_1__`,
  `__character_2__`, …) so the Resolver fills each with a **different** value.

Roles are structural labels only — they name the *dimension* a slot covers,
never its content. The content comes from your `example_prompt` idea + the LLM.

Inputs:

- **`server`** — wire from the Server Config.
- **`example_prompt`** — your idea. Supplies the **content**; the structure
  editor controls the **shape**.
- **`lock_template`** — when **on**, skip the LLM and reuse the last built
  template (re-queue for fresh Resolver fills without rebuilding).
- **`seed`** — `0` re-rolls every queue; non-zero is reproducible.
- **`direction`** / **`production_tier`** / **`mood`** — same steering presets
  as the Manager; shape the AI-written sentences and per-slot descriptions.
- **`negative_prompt`** — traits to avoid; shapes the drafted sentences and is
  compiled into the `negative_prompt` output.
- **`structure`** — hidden JSON widget holding the block list. Edited via the
  block editor on the node; it's also the source of truth for headless/API runs.

Outputs (identical to the Manager — drop-in for the Resolver):

- **`prompt_template`** — the assembled template. Wire into the Resolver's
  `template`.
- **`prompts`** — `WILDCARD_PROMPTS` bundle (flair + per-slot descriptions +
  scene bans). Wire into the Resolver's `prompts`.
- **`negative_prompt`** — deterministic deny-list string for CLIP Text Encode
  (negative).

UI:

- **Generated prompt template** panel showing the last built template.
- **Structure** editor: a live skeleton preview, one row per block with its
  controls, `+ Sentence` / `+ Wildcard group` buttons, and ↑/↓/✕ to reorder or
  remove. After a queue, a slot summary shows each placeholder's description and
  on-disk entry count.

### 🎲 LLM Wildcard Resolver

Fills `__wildcard__` slots in the template.

Inputs:

- **`server`** — wire from the Server Config.
- **`template`** — wire from the Manager's `prompt_template`, or type your
  own template directly into the widget.
- **`mode`** — `hybrid` (recommended), `reuse_existing`, or `force_new`.
- **`max_per_category`** — soft cap. Once a category file hits this many
  entries, the resolver stops appending and starts reusing.
- **`min_pool_size`** — pool floor. If a category's on-disk pool is below
  this number when the resolver runs, it tops up the pool with fresh values
  before picking. Default `5`. Bump higher (e.g. `20`–`50`) when outputs feel
  same-y across runs — wider pool = more combinatoric variety. Tops up
  silently once each pool is seeded.
- **`values_per_call`** — how many values to request per LLM call when the
  resolver does generate or top up. Default `10`. Higher = fewer calls to
  reach the floor but each call is bigger.
- **`seed`** — random seed for the choice between existing values.
- **`fix_seed`** — when **off** (default), every queue run re-rolls the
  fills (regardless of seed). When **on**, the resolver is fully
  deterministic: same template + same seed = same final values.
- **`trigger_words`** — optional text spliced onto the final prompt (after
  the grammar-alignment pass, so LoRA trigger tokens stay verbatim). Empty
  means no triggers are added. Joined to the resolved prompt with `, `.
- **`trigger_position`** — `prefix` (default) puts the trigger words at the
  start of the prompt; `suffix` appends them at the end.
- **`prompts`** *(optional input)* — wire from the Manager. Carries the
  system prompt + flair + category descriptions used per-slot.

Outputs:

- **`resolved_prompt`** — the final template with all wildcards replaced.
  Wire into your CLIP Text Encode positive input. After substitution and the
  alignment pass, any comma-separated clause that still contains a forbidden
  term from the negative prompt is stripped — so "girl, holding a phone,
  on a bench" becomes "girl, on a bench" when `phone` is on the deny-list.
- **`report`** — full text report. Wire into the Report node.
- **`negative_prompt`** — the same deterministic deny-list string the
  Manager produced (or rebuilt from `prompts` if the Manager wasn't wired).
  Wire into your CLIP Text Encode (negative) input.

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

The system prompt also asks the LLM to identify the implicit dimensions of
the description (e.g. for hair: color × length × texture × style) and produce
values that combine choices across dimensions instead of varying along a
single axis. Anti-duplication and combinatoric breadth are different
problems; the Resolver enforces both.

If the model returns a duplicate anyway, the Resolver retries once with a
higher temperature.

## Seed behavior at a glance

| Where         | Setting                | Effect                                                                |
|---------------|------------------------|-----------------------------------------------------------------------|
| Manager       | `lock_template = true` | Skip the LLM. Reuse the last cached template + categories.            |
| Manager       | `seed = 0`             | Re-roll the prompt template + category set every queue run.           |
| Manager       | `seed != 0`            | Reproducible: same inputs + same seed → same template + categories.   |
| Resolver      | `fix_seed = false`     | Re-roll the per-slot fills every queue run (regardless of seed).      |
| Resolver      | `fix_seed = true`      | Fully deterministic: same template + same seed → same final values.   |

**Reuse the same prompt with fresh wildcards every queue:** turn on Manager
`lock_template` and leave Resolver `fix_seed` off. The Manager won't call
the LLM; the Resolver re-rolls each slot every run.
