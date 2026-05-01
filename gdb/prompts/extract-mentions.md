# Extract Mentions

Read `{{batch_dir}}` and write model/dataset-only name mentions to
`{{artifact_path}}`.

Inputs:

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: filename, source id, title.

Output:

```json
{
  "mentions": [
    {
      "surface": "Qwen/Qwen3-4B",
      "kind": "model",
      "atoms": ["Qwen3", "4B"],
      "referent_scope": "entity",
      "concept_path": ["Qwen3", "4B"],
      "anchor_candidates": [
        {"type": "hf_model", "value": "Qwen/Qwen3-4B", "exact": true}
      ],
      "aux": {},
      "aliases": [{"surface": "Qwen3-4B", "descriptors": {}}],
      "context_roles": ["released_artifact"],
      "evidence": [
        {"file": "config.py", "source_id": "...", "location": "L10", "excerpt": "model_name = \"Qwen/Qwen3-4B\""}
      ],
      "description": "optional source-grounded description"
    }
  ]
}
```

Rules:

- Emit model and dataset mentions only.
- Extract names from prose, tables, model cards, dataset cards, YAML,
  JSON, and code-shaped calls such as `from_pretrained`,
  `load_dataset`, `model_name_or_path`, `model_name`,
  `tokenizer_name`, and `dataset_name`.
- Do not deep-search the web in this stage. Use only obvious links in
  the source text or surface-derived exact IDs.
- Do not use the target as an identity field. Role tags capture how an
  artifact is used by the target.
- Use `hf_dataset_config` for exact HF configs/subsets, for example
  `{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}`.
- Put uncertain or over-specific pieces in `aux`, not into
  `concept_path`. For example `dolma3_longmino_mix-100B-1125` should
  usually have atoms `["dolma3", "longmino", "mix", "100B", "1125"]`,
  concept path `["Dolma3", "longmino"]`, and aux carrying mix/date
  details unless the source/review evidence says those are reusable
  concept tiers.
- Quantization, precision, file format, and mirror/conversion details
  are entity aux or alias-local descriptors unless the source states
  separately trained weights.
- Every mention needs non-empty evidence with a verbatim excerpt.

You are running as `{{planner_model}}`. Use subagents for independent
source packets; subagents run as `{{subagent_model}}`.
