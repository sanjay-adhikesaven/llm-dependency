# Shared Context

This prototype extracts only model and dataset mentions. Software,
licenses, frameworks, tokenizer packages, eval harnesses, and generic
URLs are out of mention scope unless they identify a model or dataset.

Mentions keep identity metadata separate from descriptors:

- `identity`: `family`, `size`, `stage`, `version`, `date`, `subset`,
  `quality_cut`, `mix_variant`, `modality`, `domain`,
  `context_length`, `checkpoint`, and `extra`.
- `descriptors`: `organization`, `namespace`, `context_roles`,
  `quantization`, `precision`, `format`, `adapter`, `language`,
  `token_count`, `sample_count`, and notes.
- `aliases`: explicit surface variants. Each alias may include
  alias-local descriptors.
- `links`: `hf_ids`, `github_repos`, `official_urls`, and `papers`.

Use `context_roles` for usage context:
`training_data`, `pretraining_data`, `sft_data`, `preference_data`,
`base_model`, `teacher_model`, `judge_model`, `generator_model`,
`filter_or_classifier`, `evaluation_benchmark`,
`comparison_baseline`, `released_artifact`, `unknown`.

