# Review Entities

Read `{{review_packet_path}}` and write a compact review artifact to
`{{artifact_path}}`.

The packet groups extracted model/dataset names by likely family,
namespace, or anchor. Review each group and patch only fields that
need correction.

Output:

```json
{
  "updates": [
    {
      "mention_id": "...",
      "referent_scope": "entity",
      "concept_path": ["Qwen3", "4B"],
      "anchors": [{"type": "hf_model", "value": "Qwen/Qwen3-4B", "exact": true}],
      "aux": {"base_model": "Qwen/Qwen3-4B-Base"},
      "description": "Qwen3 4B post-trained chat/reasoning release."
    }
  ]
}
```

Review policy:

- Decide concept paths from source evidence, official collection
  boundaries, and repeated naming patterns. Do not rely on punctuation
  alone.
- Keep exact artifacts as entity leaves with exact anchors. HF anchors
  outrank GitHub, GitHub outranks official URLs, and paper anchors are
  only for exact paper-only releases.
- Use open `aux` for details that are useful but should not become
  lattice dimensions.
- Allow duplicate display names when one node is a concept and another
  is an entity.
- For Qwen-style releases, distinguish code references to exact HF ids
  from prose references to size/stage umbrellas.
- For FineMath/InfiWebMath and Dolma3, preserve distinct HF dataset
  repos/configs/mixes/pools; do not merge components, subsets, or
  upstream parents as aliases.

You are running as `{{planner_model}}`. Use subagents for independent
groups when the packet is large; subagents run as `{{subagent_model}}`.
