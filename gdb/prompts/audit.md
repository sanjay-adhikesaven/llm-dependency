# Audit Clusters

> **Goal: cluster-aware decisions on Python-grouped mentions.**
> The check stage already deduped mentions into clusters and ran
> structural conflict detection. Your job is to look at each
> cluster with the surrounding sibling context and decide:
> identity (family / size / stage), aux (lossless facets that
> don't add lattice axes), aliases (alias collapse), and the
> exact public link when the cluster doesn't have one yet.

You are running as `{{planner_model}}`. **Single planner, fans out
subagents.** This is NOT a per-batch parallel stage — Python
spawns ONE planner per audit run. The planner reads the global
cluster packet, decides how to bucket the work, and dispatches
subagents that return their decisions; the planner aggregates
and writes one artifact.

{{subagent_prompt}}

Read `{{cluster_packet_path}}` and write decisions to
`{{artifact_path}}`.

## Inputs

- `{{cluster_packet_path}}`: JSON
  `{"clusters": [...], "violations": [...]}`. Each cluster has a
  stable `cluster_key`, the merged `surface` / `aliases` /
  `links` / `anchors` / `aux`, and the structural `violations`
  the check stage emitted against it.

## Filesystem scope

Read `{{cluster_packet_path}}`. Write `{{artifact_path}}`. Do not
read or write any other local path. Web search and HF API /
GitHub API page fetches are allowed for the sole purpose of
confirming a cluster's exact link (HF repo path, GitHub repo,
arxiv id, API model id) — not for general source discovery.

## Bucketing

Group clusters by namespace + leading atom prefix into buckets
of ~12. Dispatch one subagent per bucket. Brief each subagent
with the rules below verbatim — don't paraphrase. The subagent
sees only its bucket; the planner aggregates.

## Per-cluster decision rubric

For each cluster, the subagent decides and emits an update of
the shape:

```json
{
  "cluster_key": "...",
  "kind": "model" | "dataset",
  "identity": {"family": "...", "size": "...", "stage": "..."},
  "concept_path": ["family", "..."],
  "aux": {"date": "1025", "context_length": "8192", ...},
  "aliases": [
    {"surface": "Qwen3-7B-Instruct-FP8",
     "descriptors": {"quantization": "FP8"},
     "links": [{"type": "hf_model",
                "value": "Org/Qwen3-7B-Instruct-FP8",
                "exact": true}]}
  ],
  // alias[i].links carries an alias-specific public release
  // when the variant has one (e.g., a quantized mirror).
  "referent_scope": "entity" | "concept" | "ambiguous",
  "links": [
    {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct",
     "exact": true}
  ],
  "drop": false,
  "rationale": "one short line"
}
```

To split a cluster, emit ONE update per resulting entity,
keyed by `mention_id` (one update per mention belonging to
that split). Cluster-keyed updates apply to ALL members of
the cluster — never use `cluster_key` when you mean to split.

### Identity vs aux

- `identity` is the lattice-axis facets only: `family`
  (required), `size`, `stage`. Date tokens that distinguish
  snapshots (e.g., `OLMo-3-1025` vs `OLMo-3-1125`) belong in
  `identity.extra.date` so the snapshots cluster separately.
- `aux` is lossless info that should match across mentions of
  the same concept but does NOT add a lattice axis: release
  `date` (when not snapshot-distinguishing), `mix_size`,
  `context_length`, `version`, `organization`, source-local
  labels.
- Per-variant distribution facets (`quantization`, `precision`,
  `format`, `file_format`, `namespace`) belong on the alias's
  `descriptors`, NOT on `aux`.

### Family boundaries respect HF collection / repo path

The `family` is whatever the HF collection name or repo-path
namespace says. It is **not** a hyphen prefix.
`Qwen3`, `Qwen3-VL`, `Qwen3-Coder`, `Qwen3Guard` are PEER
families — none nested under another. Multi-token families like
`Qwen3-VL`, `Qwen3-Coder`, `Llama-3.1`, `InfiMM-WebMath`,
`HuggingFaceTB/finemath` stay intact in `identity.family` and
as a single `concept_path` element.

Only `size` (digit-bearing token like `7B`, `30B-A3B`) and
`stage` (Base / Instruct / Thinking / Reasoning / Stream /
Coder) peel off the right side.

### Link confirmation

If the cluster has no exact link, web-search HF / GitHub /
arxiv / vendor API docs to find one. Preferred order:

1. `hf_dataset_config` (`<repo>::<config>`) for HF dataset
   subsets (`HuggingFaceTB/finemath::finemath-3plus`).
2. `hf_model` / `hf_dataset` (`<org>/<repo>`).
3. `github_ref` (`<org>/<repo>@<ref>:<path>`) for code
   pinned to a commit.
4. `github_repo` for the bare repo when there's no commit pin.
5. `api_model_id` for closed-source vendor IDs
   (`gpt-4o-mini-2024-07-18`, `claude-opus-4-5`).
6. `official_release_url` for first-party release pages when
   nothing above applies.
7. `paper_release` for paper-only artifacts (a benchmark with
   no open dataset, etc.) — last resort.

Set `exact: true` only when the link uniquely identifies the
artifact. Mark `metadata.mirror: true` if the source explicitly
identifies a different first-party release (e.g., MASS lives at
`microsoft/MASS` on GitHub; an HF mirror should be flagged).

### Quantization / format / precision / mirrors

If a member surface ends in a quantization or runtime-format
suffix (`-FP8`, `-FP16`, `-BF16`, `-Q\d+(_\w+)*`, `-AWQ`,
`-GPTQ`, `-EXL2`, `-BNB-4bit`, `-INT4`, `-INT8`, `-GGUF`,
`-MLX`, `-SafeTensors`), it is the SAME identity as the
canonical (no quantization for identity); fold the variant as
an alias whose `descriptors` record the suffix and whose
`links` carry the variant's HF/GitHub link if it has one.

### Conflict resolution

For each violation flagged on the cluster:

- `aux_conflict`: decide whether the cluster should split into
  N entities (emit one `mention_id`-keyed update per group of
  mentions that share an aux value) or whether one of the
  values is wrong and the others are right (cite the
  source-side anchors).
- `surface_identity_conflict`: split — same surface mapping to
  two different identities means at least one entity is hidden
  inside that ambiguous name. Emit one `mention_id`-keyed
  update per identity.
- `link_identity_conflict`: same exact link points to two
  different identities. Either fix one identity or split via
  per-`mention_id` updates.
- `link_concept_conflict`: same exact link sits under two
  different concept paths. Pick one path; flag the rest for
  re-routing.

Use `drop: true` (with `mention_id`) to retire a noise mention.
Populate `rationale` with one short line citing the source-side
anchors.

### Vague references

If the cluster is a bare family or stage abstraction with no
exact public release at this tier (`Qwen3-Base`, plain `Qwen3`,
`FineMath` family), set `referent_scope: "concept"` and leave
`links: []`. Don't fabricate a slash path. The lattice will
mint the concept node; an entity child can attach later.

## Output

Write the artifact to `{{artifact_path}}` and exit 0:

```json
{
  "updates": [
    {
      "cluster_key": "...",
      ...per-cluster update...
    }
  ]
}
```

If no clusters need updates, emit `{"updates": []}` and exit 0.

Subagents you dispatch run as `{{subagent_model}}`.
