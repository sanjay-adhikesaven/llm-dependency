# Link Unresolved

Read unresolved clusters from `{{unresolved_clusters_path}}` and write
targeted link suggestions to `{{artifact_path}}`.

Search only unresolved or suspicious clusters. Prefer first-party HF,
GitHub, arXiv, venue, or official URLs. Do not broaden into a general
source discovery pass.

Output:

```json
{
  "links": [
    {
      "cluster_key": "model:...",
      "links": {"hf_ids": ["org/repo"], "github_repos": [], "official_urls": [], "papers": []},
      "evidence": "short note on why the link identifies this cluster"
    }
  ]
}
```

Python will verify returned links after this pass.

You are running as `{{planner_model}}`.

