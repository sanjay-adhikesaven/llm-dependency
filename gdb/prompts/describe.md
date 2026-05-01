# Describe Entity Leaves

> **Goal: per-entity-leaf descriptions, grounded in source
> anchors + HF card text.** Concepts are NEVER described — only
> entity leaves (nodes with `node_type: "entity"` and at least
> one verified or candidate link).

You are running as `{{planner_model}}`. **Single planner, fans out
subagents.** Python spawns ONE Claude planner per describe run.
The planner reads the lattice, filters out concept nodes, buckets
the entity leaves (e.g., by family or by link type), and
dispatches subagents (e.g., `{{subagent_model}}`) via the Task
tool. Each subagent fetches HF metadata for its bucket's entities
and writes descriptions. The planner aggregates into one artifact.

Read `{{lattice_path}}` and write descriptions to
`{{artifact_path}}`.

## Inputs

- `{{lattice_path}}`: the lattice artifact emitted by
  `build-lattice` — `{"nodes": [...], "edges": [...],
  "forests": [...], "audit": {...}}`. Each node has `node_key`,
  `node_type`, `kind`, `display_name`, `concept_path`, `aliases`,
  `links`, `verified_links`, `aux`, and any source-side
  `anchors` accumulated from member mentions.

## Filesystem scope

Read `{{lattice_path}}`. Write `{{artifact_path}}`. HF README,
HF API, GitHub README, and arxiv abstract fetches for the
entity's link are allowed — use your native fetch / web tools
to read them directly. Do not read or write any other local
path.

## Bucketing

Filter the lattice's `nodes` to those with `node_type ==
"entity"`. Group the entities into buckets of ~10 by family or
by link namespace so subagents have coherent context. Dispatch
one subagent per bucket.

## Per-entity output

For each entity leaf, the subagent emits:

```json
{
  "entity_key": "<node_key>",
  "kind": "model" | "dataset",
  "display_name": "Qwen/Qwen3-4B",
  "links": [...],                 // typed identifier list, copied from the node
  "description": "Qwen3-4B chat model; pipeline=text-generation; base_model=Qwen/Qwen3-4B-Base; ...",
  "metadata": {"front_matter": {...}, "card_data": {...}},
  "source": {"repo_url": "...", "readme_url": "...", "api_url": "..."}
}
```

## Description rubric

The description is grounded in three things, in this order of
priority:

1. **HF card front-matter** (when the entity has an HF link):
   `pipeline_tag`, `base_model`, `datasets`, `library_name`,
   model summary text from the README body.
2. **Cluster `aux`** facets carried up from the mentions
   (`context_length`, `mix_size`, `date`, `version`, etc.).
3. **Source-side `anchors`** — short verbatim quote(s) from the
   batch sources that justify the entity's existence.

Compose them into 1–3 sentences. Lead with what the artifact
IS (model family + size + stage; or dataset family + subset).
Follow with one line of facets from card + aux. Close with a
source citation if the card was thin.

### Neutral framing

The description should read identically across investigations
and across sources. Do NOT write "used by `<target>`" or
"referenced in `<filename>`". Source citations live in the
node's source-side anchors, not in prose. The same entity-leaf
node should produce the same description regardless of which
investigation this run was about.

### License does NOT enter the description

If `metadata.front_matter.license` is set, copy it into
`metadata` raw but do NOT promote it to the description, the
identity, or the aux. The system's scope is models and
datasets only — license is metadata, not a graph node.

## Link types other than HF

- `github_repo` / `github_ref`: fetch the repo's README from
  GitHub if reachable; describe the artifact from there.
- `api_model_id` (`gpt-4o-mini-2024-07-18`, `claude-opus-4-5`):
  describe from the vendor's documentation. The description
  should make the version snapshot explicit.
- `paper_release`: describe from the paper abstract; flag that
  the entity is paper-only.
- `official_release_url`: fetch the release page; describe from
  there.

If no link is verifiable for the entity (`verified_links` is
empty AND no fallback fetch succeeded), emit a description
based on cluster aux + source anchors only, and set
`metadata.fetch_failed: true`.

## Output

Write the artifact to `{{artifact_path}}` and exit 0:

```json
{
  "descriptions": [
    {
      "entity_key": "...",
      "kind": "...",
      "display_name": "...",
      "links": [...],
      "description": "...",
      "metadata": {...},
      "source": {...}
    }
  ]
}
```

If no entity leaves need descriptions, emit
`{"descriptions": []}` and exit 0.

You are running as `{{planner_model}}`. Subagents you dispatch
run as `{{subagent_model}}`.
