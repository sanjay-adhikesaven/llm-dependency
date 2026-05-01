# Shared Context

This prototype extracts only model and dataset mentions. Software,
licenses, frameworks, tokenizer packages, eval harnesses, and generic
URLs are out of mention scope unless they identify a model or dataset.

The first pass is name-first. Extract the surface name, ordered atoms,
obvious exact anchors, roles, and evidence. Do not force every token
into a predefined identity field.

- `atoms`: ordered name pieces as the source presents them. Preserve
  protected spans when punctuation is misleading, e.g. `Qwen3Guard`
  can be `["Qwen3", "Guard"]` if release evidence supports it.
- `concept_path`: reviewed lattice path from general to specific.
  Examples: `["Qwen3"]`, `["Qwen3", "VL"]`,
  `["Dolma3", "longmino"]`, `["FineMath", "3plus"]`.
- `anchor_candidates`: exact public release identifiers. Use
  `hf_model`, `hf_dataset`, `hf_dataset_config`, `github_repo`,
  `github_ref`, `api_model_id`, `official_release_url`, or
  `paper_release`.
- `aux`: open structured details that should not create lattice axes,
  such as release size, date, mix size, config names, quantization,
  precision, file format, token counts, or source-local labels.
- `context_roles`: open strings. Suggested roles include
  `training_data`, `pretraining_data`, `sft_data`,
  `preference_data`, `base_model`, `teacher_model`,
  `judge_model`, `generator_model`, `filter_or_classifier`,
  `evaluation_benchmark`, `comparison_baseline`,
  `released_artifact`, and `unknown`.

Every concrete entity leaf must have an exact anchor. A broad technical
report, family blog, or general project page is evidence, not an
entity anchor, unless it is the only exact release record for a
paper-only model or dataset.

Same display names can appear twice when they refer to different node
types. For example, `Qwen3-4B` may be a concept node covering all
Qwen3 4B releases, while `Qwen/Qwen3-4B` is a concrete HF model
entity under that concept path.
