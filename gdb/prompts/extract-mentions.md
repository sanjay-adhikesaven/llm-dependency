# Extract Mentions

Read `{{batch_dir}}` and write model/dataset-only mentions to
`{{artifact_path}}`.

Inputs:

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: filename, source id, title.

Output:

```json
{
  "mentions": [
    {
      "surface": "Qwen3-7B-Instruct-FP8",
      "kind": "model",
      "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
      "descriptors": {"precision": "FP8", "context_roles": ["released_artifact"]},
      "aliases": [
        {"surface": "Qwen3-7B-Instruct-FP8", "descriptors": {"precision": "FP8"}}
      ],
      "links": {"hf_ids": ["Qwen/Qwen3-7B-Instruct-FP8"], "github_repos": [], "official_urls": [], "papers": []},
      "subsets": [],
      "context_roles": ["released_artifact"],
      "evidence": [
        {"file": "card.md", "source_id": "...", "location": "README", "excerpt": "verbatim sentence containing the surface"}
      ],
      "notes": "optional"
    }
  ]
}
```

Rules:

- Emit model and dataset mentions only.
- Benchmarks are datasets. Baseline, teacher, judge, generator, filter,
  and base artifacts are model/dataset mentions with role tags.
- Put identity-bearing uncertainty in `identity.extra`, not in
  descriptors.
- Quantization, precision, and file format are alias-local descriptors
  unless the source says they are separately trained weights.
- For datasets, use `subsets` for HF configs, named release subsets,
  quality cuts, and evidence.
- Every mention needs non-empty evidence with a verbatim excerpt.
- Do not deep-search the web in this stage. Use only obvious links in
  the source text or surface-derived IDs.

You are running as `{{planner_model}}`. Use subagents for independent
source packets; subagents run as `{{subagent_model}}`.

