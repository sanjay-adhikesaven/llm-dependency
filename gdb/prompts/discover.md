# Discover

Fetch sources for `{{target}}` and write a discovery artifact to
`{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `target` and `workspace_dir`.
- `{{workspace_dir}}`: writable directory for fetched files.

## Filesystem scope

Read `{{input_path}}`. Write fetched files into
`{{workspace_dir}}` (relative paths recorded in the discovery
artifact). Write the discovery artifact to `{{artifact_path}}`.
Network fetches land in `{{workspace_dir}}` — never to the
project tree, never to /tmp, never anywhere else.

## Scope

- Fetch the target model/dataset card, release blog, paper, and
  target-specific configs or recipes.
- Fetch direct model and dataset upstreams: base models, teacher or
  judge models, pretraining/SFT/preference datasets, named dataset
  subsets, and evaluation benchmark cards/papers.
- Prefer first-party HF cards, GitHub repos, arXiv/venue PDFs, and
  official project pages.

Output:

```json
{
  "batches": [
    {
      "label": "target cards",
      "summary": "why these sources belong together",
      "sources": [
        {"path": "relative/to/workspace.md", "url": "https://...", "commit_sha": null, "title": "..."}
      ]
    }
  ]
}
```

Batch related sources together: one stage chain, one dataset family,
or one paper/blog packet per batch. Enumerate variants by size, stage,
date, subset, quality cut, and mix variant when the target source names
them.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. Each
`sources[i].path` must be a path inside `{{workspace_dir}}`
that you actually wrote. An empty `batches[]` list usually
signals a misread or a fetch failure — surface the failure
mode rather than emitting nothing.

You are running as `{{planner_model}}`. Use subagents for
independent fetch packets. Subagents run as `{{subagent_model}}`.

{{subagent_prompt}}

