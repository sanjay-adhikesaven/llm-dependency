# Link Unresolved

Read unresolved entity/name groups from `{{unresolved_clusters_path}}`
and write targeted anchor suggestions to `{{artifact_path}}`.

Search only unresolved or suspicious groups. Prefer exact first-party
anchors in this order: HF repo/config, GitHub repo/ref, API model id,
official release page, then exact paper-only release. Do not broaden
into a general source discovery pass.

## Inputs

- `{{unresolved_clusters_path}}`: JSON with the cluster list.
  Each cluster has surface, identity, concept_path, atoms,
  current anchors (if any), and source-evidence excerpts. The
  packet is self-contained.

## Filesystem scope

Read `{{unresolved_clusters_path}}`. Write `{{artifact_path}}`.
Web search is allowed for finding HF / GitHub / arxiv / official
URLs; do not fetch and re-read full source files. Do not write
to any other local path.

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

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. Empty
`updates[]` is legal — it means none of the unresolved groups
have a confidently-discoverable anchor, and that's better than
fabricating one.

You are running as `{{planner_model}}`.
