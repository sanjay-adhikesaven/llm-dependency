# Repair Mentions

Read `{{repair_packet_path}}` and write a compact repair artifact to
`{{artifact_path}}`.

The packet contains only Python-detected violation summaries and local
mention evidence. Do not reread all sources. Patch labels and alias
decisions only where the evidence makes the repair obvious.

## Filesystem scope

Read `{{repair_packet_path}}`. Write `{{artifact_path}}`. Do not
read or write any other path. Do not fetch sources or web pages
— the packet contains everything you need.

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

Use `drop: true` only for noise mentions that cannot be converted
into a valid model or dataset mention.

## Violation handling

- `aux_conflict`: two or more mentions in the same cluster carry
  different non-null values for the same `aux` key (e.g.,
  `aux.date = "1025"` vs `"1125"`). Decide whether (a) the cluster
  should split into N separate entities — emit identity updates that
  push the distinguishing field into `identity.extra` so the
  mentions stop sharing a cluster — or (b) one of the values is
  wrong and the others are right; correct the wrong mention's `aux`
  using source evidence.
- `should_be_alias`: a mention's surface is a quantization or format
  variant of a sibling canonical mention (e.g.,
  `Qwen3-7B-Instruct-FP8` next to `Qwen3-7B-Instruct`). Merge the
  variant into the canonical: drop the variant's standalone
  mention; add the variant's surface as an alias of the canonical
  with `descriptors` recording the suffix info; if the variant had
  its own anchor, attach it to the alias's `anchors` list.
- `surface_identity_conflict`: same surface name maps to multiple
  identity signatures. If the source uses the same name for
  different things (rare but real), keep them separate but
  disambiguate descriptors / surfaces with version markers. If
  one is wrong, fix the identity.
- `link_identity_conflict` / `anchor_concept_conflict`: one
  link or anchor maps to multiple identities or concept paths.
  Decide whether the link is wrong on one mention, or whether
  the cluster genuinely needs splitting via identity changes.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`updates[]` list is legal when no obvious repair fits — leave
ambiguous cases for the reviewer rather than guessing.

You are running as `{{planner_model}}`.

