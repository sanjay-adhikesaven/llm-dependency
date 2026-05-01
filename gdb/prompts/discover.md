# Discover

Fetch sources for `{{target}}` and write a discovery artifact to
`{{artifact_path}}`.

Inputs:

- `{{input_path}}`: JSON with `target` and `workspace_dir`.
- `{{workspace_dir}}`: writable directory for fetched files.

Scope:

- Fetch the target model/dataset card, release blog, paper, and
  target-specific configs or recipes.
- Fetch direct model and dataset upstreams: base models, teacher or
  judge models, pretraining/SFT/preference datasets, named dataset
  subsets, and evaluation benchmark cards/papers.
- Do not fetch software-only or license-only sources.
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

You are running as `{{planner_model}}`. Use subagents for independent
fetch packets; subagents run as `{{subagent_model}}`.

