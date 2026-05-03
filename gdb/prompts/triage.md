# Triage Upstream Nodes for Recursive Expansion

> **Goal: read the lattice + relations, classify every
> upstream entity-leaf as `auto_expand`, `decline`, or
> `manual`.** The output is a queue the operator works
> through with `gdb run expand --node <formal_name>`. We do
> not auto-recurse — depth is operator-gated.

Read `{{lattice_path}}` and `{{relations_path}}`. Write the
classification artifact to `{{artifact_path}}`.

## Filesystem scope

Read `{{lattice_path}}` (groups+items+links from
linker / audit / organize) and `{{relations_path}}` (relations
artifact, possibly merged across batches). Write
`{{artifact_path}}`. Web search permitted for sanity-checking
borderline nodes; use sparingly.

## What "upstream" means

An entity-leaf is **upstream** of the original target if the
lattice contains an edge whose **subject** is the target (or a
descendant of the target) and whose **object** is that
entity-leaf. Concretely: walk the relations file. For every
edge whose `subject` is the run's target or transitively
trained-on/initialized-from/distilled-from/transformed-by/
filtered-by the target's lineage, the `object_ref` (when set)
is upstream and a candidate for triage.

Do not triage:
- Concept-level nodes (no resolvable link).
- The original target itself.
- Nodes that appear only as STRUCTURAL endpoints
  (`subset_of` parents, `supersedes` predecessors) — those
  are lineage notes, not expansion targets.

## Classification

For each upstream entity-leaf, emit ONE of three
classifications with a one-line `rationale`:

### `auto_expand`

Worth running the full pipeline against on its own. Typical
cases:

- A datamix or training corpus the target trains on
  (`allenai/dolma3-mix`, `tulu-3-sft-mixture`) — its sources
  matter for the target's lineage.
- A judge / filter model used at training time
  (`Qwen/Qwen3-32B` if used as DPO judge) — its training in
  turn determines what kind of preferences it imposed.
- A distillation source (the model that generated the target's
  synthetic training data) — its lineage propagates through.
- An OCR / rewriter model (`allenai/olmOCR-7B-0225`) — its
  training defines what content the target absorbed.

Rationale should name the relation that motivated expansion
(e.g., `"trained_on at stage 1; expanding to enumerate
sub-mix sources"`).

### `decline`

Should NOT be expanded. Typical cases:

- A comparison baseline cited in evaluation tables but not
  used in training (`Qwen/Qwen3-7B-Instruct` evaluated against
  but not trained from) — `used_for_evaluation` /
  `cited_as_baseline` only.
- A well-known foundation model whose lineage is itself the
  subject of dedicated tracing (the target should declare its
  base, not recursively explain Llama).
- An ablation-only artifact (`used_for_ablation`).
- A dataset on HF that has no card / has been deprecated /
  is irrecoverably obscure.

Rationale should explain why expansion adds no information
(e.g., `"used_for_evaluation only; expanding would conflate
eval and training lineage"`).

### `manual`

Operator should decide. Use sparingly — only when:
- The node has a real training-pipeline relation but the
  expansion budget is uncertain (e.g., a personal-namespace
  HF dataset with hundreds of mixed sources).
- The node could be expanded but only if combined with
  another run's output.

Rationale should describe what the operator needs to weigh.

## Output schema

```json
{
  "auto_expand": [
    {
      "formal_name": "allenai/dolma3-mix",
      "kind": "dataset",
      "primary_link": "https://huggingface.co/datasets/allenai/dolma3-mix",
      "rationale": "trained_on at stage 1; expanding to enumerate sub-mix sources",
      "motivating_relations": ["trained_on"]
    },
    {
      "formal_name": "Qwen/Qwen3-32B",
      "kind": "model",
      "primary_link": "https://huggingface.co/Qwen/Qwen3-32B",
      "rationale": "used as DPO judge; its training shapes preference distribution",
      "motivating_relations": ["filtered_by"]
    }
  ],
  "decline": [
    {
      "formal_name": "Qwen/Qwen3-7B-Instruct",
      "kind": "model",
      "primary_link": "https://huggingface.co/Qwen/Qwen3-7B-Instruct",
      "rationale": "used_for_evaluation only; expanding would conflate eval and training lineage",
      "motivating_relations": ["used_for_evaluation"]
    }
  ],
  "manual": [
    {
      "formal_name": "hamishivi/rlvr_general_mix",
      "kind": "dataset",
      "primary_link": null,
      "rationale": "personal-namespace mix; HF page exists but composition is undocumented; operator should decide if expansion budget is justified",
      "motivating_relations": ["trained_on"]
    }
  ]
}
```

Every entity-leaf upstream of the target must appear in
exactly one bucket. If you skipped a node (concept, target,
STRUCTURAL-only), include it in a top-level `skipped[]` array
with a one-line reason — that lets the operator audit
coverage.

## Subagent dispatch

The Task tool is available — subagents run as
`{{subagent_model}}`. For a small lattice (< 50 upstream
nodes) classify inline. For a larger one, bucket by family
and dispatch subagents. Each subagent classifies its slice
and returns the same shape; aggregate before writing.

When dispatching, transcribe the three classification
definitions and rationale-style guidance verbatim — subagents
have none of your context.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
