# Audit and Revise

> **Goal: read the lattice, fix what's wrong, write the
> revised lattice.** Same shape in, same shape out. You're a
> second pass over organize's work — make whatever edits you
> see needed, no predefined categories.

Read `{{organize_path}}` and write the revised artifact to
`{{artifact_path}}`.

## Filesystem scope

Read `{{organize_path}}` and `{{input_path}}` (same file).
Write `{{artifact_path}}`. Web search and HF/GitHub URL lookups
are permitted to verify ambiguous cases. Use them sparingly.

## What you do

Look at every family at once. Decide what's wrong. Fix it.
There is no catalog of issue types — make whatever edits you
think the lattice needs. Possible edits include:

- **Split an item** whose `aliases` actually refer to different
  artifacts (a thinking/no-thinking pair, a base/instruct
  pair, a quantized/unquantized pair collapsed into one).
- **Merge two families** that turn out to refer to the same
  product line (e.g., the heuristic put `Tulu 3` and `Tulu
  models` in different buckets).
- **Move an item** from one family to another when the
  original placement was wrong.
- **Fix a `formal_name`** that doesn't resolve on HF/GitHub.
  Spot-check unusual ones with:
  ```
  curl -sL -o /dev/null -w '%{http_code}' \
    https://huggingface.co/<repo>
  ```
- **Adjust `identity_keys`** when the keys are redundant,
  overlapping, or missing a dimension that varies inside the
  family.
- **Drop an item** that shouldn't exist (e.g., a synthetic
  bare-stage placeholder no source actually mentioned).
- **Add a missing alias** when an obvious surface variant is
  in another family by mistake.

If a family is already clean, leave it alone. The output is
the WHOLE revised lattice, not just the diff — every family
that should remain in the lattice must be in the output, even
if unchanged.

## When uncertain

Dispatch a sonnet subagent to investigate ONE specific case
(e.g., "is `Qwen/Qwen3-8B` resolvable? does it confuse
thinking vs no-thinking?"). Cap total investigations at ~10
per pass — this is a check-and-fix pass, not a full re-organize.

## Output schema

Same shape as the organize artifact: a list of `groups`, each
with a `family` label, `identity_keys`, and `items[]`. Every
item has `kind`, `formal_name`, `identity`, `aliases`. Every
item must trace back to at least one real input name (carried
through from organize via the alias list).

Optionally include a top-level `notes` field — a brief plain-
English summary of what you changed, for the audit log. Skip
it if you made no substantive edits.

```json
{
  "groups": [
    {
      "family": "Qwen3",
      "identity_keys": ["org", "collection", "size", "stage", "variant"],
      "items": [
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3-8B",
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "8B", "variant": "no-thinking"},
          "aliases": ["Qwen 3 8B", "Qwen 3 8B (no reasoning)", "Qwen/Qwen3-8B"]
        },
        {
          "kind": "model",
          "formal_name": "Qwen/Qwen3-8B-Thinking",
          "identity": {"org": "Qwen", "collection": "Qwen3", "size": "8B", "variant": "thinking"},
          "aliases": ["qwen3-reasoning-8b", "Qwen 3 8B thinking"]
        }
      ]
    }
  ],
  "notes": "Split Qwen3-8B into thinking and no-thinking variants; merged 'Tulu 3' and 'Tulu 3 (Llama-3.1 base)' under one 'Tulu 3' family with base_model in identity_keys."
}
```

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
