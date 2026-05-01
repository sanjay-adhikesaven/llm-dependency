# Link Unresolved

Read unresolved entity/name groups from `{{unresolved_clusters_path}}`
and write targeted anchor suggestions to `{{artifact_path}}`.

Search only unresolved or suspicious groups. Prefer exact first-party
anchors in this order: HF repo/config, GitHub repo/ref, API model id,
official release page, then exact paper-only release. Do not broaden
into a general source discovery pass.

Output:

```json
{
  "updates": [
    {
      "mention_id": "...",
      "anchors": [
        {"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus", "exact": true}
      ],
      "evidence": "short note on why the anchor identifies this mention"
    }
  ]
}
```

Only use `paper_release` when the paper is itself the exact release
record for that model or dataset, not merely a broad technical report.

Python will verify returned anchors after this pass.

You are running as `{{planner_model}}`.
