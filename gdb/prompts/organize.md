# Organize Names

> **Goal: read every name and emit ONE clean record per real
> artifact.** Group names into families, collapse surface
> variants, pick a canonical `formal_name`, and break the
> identity into structured fields. Every emitted item must
> trace back to at least one real input name — no invented
> nodes.

Read `{{names_path}}` and write the artifact to
`{{artifact_path}}`.

## Inputs

- `{{names_path}}`: JSON `{"names": [{"type": "model"|"dataset", "name": "..."}, ...]}`.
  Already deduped on `(type, name)`. Surface variants of the
  same artifact (case / separator / accent / HF-org-prefix /
  parenthetical differences) are NOT deduped — that's your job.

## Filesystem scope

Read `{{names_path}}` and `{{input_path}}` (same file). Write
`{{artifact_path}}`. Web search is permitted for HF / GitHub
disambiguation when a name's resolvable form is unclear. Use it
sparingly.

## What you decide

For each input name:

1. **Family membership** — which other names refer to the same
   family of artifacts. You decide what counts as a family.
2. **Family name** — a short, recognizable label. You decide.
3. **Identity keys for the family** — the dimensions that vary
   inside it. Open vocabulary. You decide per family.
4. **Surface collapse** — names that differ only in case /
   separator / accent / HF-org prefix / trailing parenthetical
   merge into ONE item with multiple `aliases`. Names that
   differ in any identity dimension (size, stage, date,
   quantization) stay separate.
5. **Per item: `formal_name`, `identity` dict, `aliases` list,
   `kind`** — see schema below.

Every item must trace back to at least one real input name. Do
not invent items.

## Bucketing for parallelism

This is a hint for splitting the input across subagents — NOT a
definition of family membership.

For each name, take the substring before the first `/` or `-`
(whichever appears first):
- `Qwen/Qwen3-4B` → `Qwen`
- `Qwen3-7B-Instruct` → `Qwen3`
- `OLMo-3-1025-7B` → `OLMo`
- `MMLU-Pro` → `MMLU`

Group names whose prefix-tokens share ≥3 consecutive identical
characters into the same bucket. (`Qwen` ≈ `Qwen3` → same
bucket; `OLMo` ≈ `OLMo3` → same bucket.) Each bucket goes to
one subagent.

The 3-char rule is approximate — it gets coarse parallelism
right and is wrong at the margins. Two names in different
buckets may turn out to belong to the same family; the planner
reviews subagent outputs and merges where needed before writing
the final artifact.

## Per-item schema

```json
{
  "kind": "model",
  "formal_name": "Qwen/Qwen3-4B-Base",
  "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "Base"},
  "aliases": ["Qwen3-4B-Base", "qwen3-4b-base", "Qwen 3 4B Base"]
}
```

- `kind`: `"model"` or `"dataset"`, carried from the input.
- `formal_name`: the most-resolvable identifier-like form on HF
  or GitHub — what someone would paste into a URL bar to visit
  the artifact. Prefer the org-prefixed form. Examples:
  - `Qwen/Qwen3-4B-Base`
  - `allenai/Olmo-3-1025-7B`
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `HuggingFaceTB/finemath` (parent dataset)
  - `HuggingFaceTB/finemath::finemath-3plus` (config of a
    parent dataset)

  For artifacts with no HF/GitHub repo (e.g. OpenAI /
  Anthropic API models), construct a synthetic
  `vendor/identifier` form:
  - `OpenAI/gpt-4o-mini-2024-07-18`
  - `Anthropic/claude-3-5-sonnet-20240620`
- `identity`: dict whose keys are this family's
  `identity_keys`. Populate only the keys this item actually
  carries. (Don't write empty strings for keys that don't
  apply.)
- `aliases`: deduped list of every original input name that
  collapsed to this item. The `formal_name` itself goes in
  `aliases` only if a source emitted it verbatim.

A name that genuinely refers to two different artifacts (e.g.,
`WildGuard` the model vs `wildguardmix` the dataset, both
mentioned in the source as `WildGuard`) splits into two items
with different `kind`. The cross-kind extract overlap captured
8 such cases — review each and either merge under one item
(extractor mis-tagged) or split into two.

## Per-family schema

A family bundles items that share most identity dimensions —
they're the same product line at different points along its
varying axes.

```json
{
  "family": "Qwen3",
  "identity_keys": ["org", "collection", "size", "stage"],
  "items": [ ... ]
}
```

- `family`: a short label you choose. Pick the most recognizable
  short form (the collection name in most cases).
- `identity_keys`: the dimensions that vary across this
  family's items. Pick from open vocabulary; common keys
  include `org`, `collection`, `version`, `size`, `stage`,
  `date`, `quantization`, `subset`, `vendor`, `family`, etc.
  Don't force one schema across unrelated families. A
  family of evaluation-benchmark variants will look nothing
  like a family of model checkpoints.

## Output

```json
{
  "groups": [
    {
      "family": "Qwen3",
      "identity_keys": ["org", "collection", "size", "stage"],
      "items": [
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3-4B-Base",
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "Base"},
          "aliases": ["Qwen3-4B-Base", "qwen3-4b-base"]
        },
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3-4B-Instruct",
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B", "stage": "Instruct"},
          "aliases": ["Qwen3-4B-Instruct"]
        },
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3",
          "identity": {"org": "Qwen", "collection": "Qwen3"},
          "aliases": ["Qwen3", "Qwen 3", "the Qwen3 family"]
        }
      ]
    },
    {
      "family": "MMLU",
      "identity_keys": ["family", "subset"],
      "items": [
        {
          "kind": "dataset",
          "formal_name": "cais/mmlu",
          "identity": {"family": "MMLU"},
          "aliases": ["MMLU", "mmlu"]
        },
        {
          "kind": "dataset",
          "formal_name": "TIGER-Lab/MMLU-Pro",
          "identity": {"family": "MMLU", "subset": "Pro"},
          "aliases": ["MMLU-Pro", "Mmlu-pro", "mmlu_pro"]
        }
      ]
    }
  ]
}
```

## Subagent dispatch

The Task tool is available — subagents run as `{{subagent_model}}`.
One subagent per bucket (see "Bucketing for parallelism"
above). Right-size: 30-150 names per bucket is the sweet spot.
Don't go narrower (overhead) or wider (defeats parallelism).

When dispatching, transcribe the rules in this prompt verbatim
into the subagent's brief — it has none of your context. The
subagent decides its own family/identity-keys structure within
its bucket.

After all subagents return, review for cross-bucket merges:
two seed buckets that turned out to hold one family. Merge
those before writing the final artifact.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
