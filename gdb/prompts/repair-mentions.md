# Repair Mentions

Read `{{repair_packet_path}}` and write a compact repair artifact to
`{{artifact_path}}`.

The packet contains only Python-detected violation summaries and local
mention evidence. Do not reread all sources. Patch labels and alias
decisions only where the evidence makes the repair obvious.

Output:

```json
{
  "updates": [
    {
      "mention_id": "...",
      "kind": "model",
      "identity": {"family": "..."},
      "descriptors": {},
      "aliases": [{"surface": "...", "descriptors": {}}],
      "links": {"hf_ids": [], "github_repos": [], "official_urls": [], "papers": []}
    }
  ]
}
```

Use `drop: true` only for software/license/noise mentions that cannot
be converted into a model or dataset mention.

You are running as `{{planner_model}}`.

