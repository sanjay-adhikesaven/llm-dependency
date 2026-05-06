# Evaluation

`pooled_eval.py` is the canonical pooled LLM-as-judge verifier used in the
paper. It works for any submission that follows the per-subject
`{nodes, edges}` JSON convention.

## How it works

For each target model, every emitted edge across all systems is pooled and
clustered by canonicalized `(subject, object)` pair. The cluster's longest
description is sent to a single Claude verifier instance equipped with
`web_search` (max 6 queries per call), which returns one of:

- `verified` — the relationship is real (cited evidence supports it, or
  the verifier independently corroborated it via web search)
- `refuted` — neither the cited evidence nor independent search supports
  the claim, or the verifier found direct contradiction
- `unclear` — genuinely ambiguous after honest search

A single verifier verdict cleanly attributes back to every system that
proposed an edge in that cluster. So the per-system Verified / Refuted
counts in the eval table are: how many clusters did this system contribute
to, broken down by verdict.

The verifier model in all reported runs is `claude-sonnet-4-6` with the
`web_search_20250305` server tool.

## Reproducing the reported numbers

```bash
pip install -r ../requirements.txt
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py
```

This pools the 4 baselines × 4 subjects, finds the unique relationships,
and verifies each one. With concurrency 12 and the cleaner pool, expect
~15 min wall time. New verifications append incrementally to
`outputs/verifications.jsonl`; on resume, completed clusters are skipped.

The pre-computed verdicts in `outputs/` are the exact ones used in the
paper. To rerun from scratch, delete `outputs/verifications.jsonl` first.

## Evaluating a new submission

If your submission follows the per-subject convention, drop your files
into `../baselines/outputs/` (or any other directory) named
`<slug>_<subject>.json`, then:

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py \
    --systems gpt55pro,gpt54pro,cc,o3dr,mysystem
```

Only clusters your system contributes that aren't already in
`outputs/verifications.jsonl` will be freshly verified. Cost scales with
the number of new clusters.

If your system is the same per-cluster verdict but a different system
slug — e.g., to attribute existing verdicts to your slug as well —
you can re-run with the new slug; the verifier won't be re-called for
matched clusters because the `cluster_key` resolves identically.

## Outputs

- `outputs/verifications.jsonl` — one JSON record per verified cluster
  with `cluster_key`, `target`, canonicalized `subject`/`object`,
  contributing `systems`, `verdict`, `confidence`, and `explanation`.
- `outputs/score.json` — aggregate verified/refuted/unclear/error counts
  per system, with computed precision over decisive verdicts.
- `outputs/score_per_target.json` — same breakdown per target.
- `outputs/table.txt` / `outputs/table.latex` — human-readable and
  paste-ready LaTeX renderings of the aggregate table.

## Schema

Each input edge needs at minimum:

```json
{
  "subject": "<canonical_id>",
  "object":  "<canonical_id>",
  "description": "<lossless prose>",
  "evidence": [
    {"source": "<URL>", "location": "<locator>",
     "excerpt": "<verbatim quote>", "explanation": "<how it supports>"}
  ]
}
```

`relation_type` is accepted but not graded — the verifier judges the
underlying relationship, not the bucket label. See
`prompts/investigator_prompt.md` for the full schema and admission criteria.
