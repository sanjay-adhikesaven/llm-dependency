# Shared Context

This prototype extracts named models and named datasets only. Focus
on these two categories — every emitted mention must be either a
model (open or closed checkpoint) or a dataset (data corpus,
benchmark, or HF dataset config).

A mention has three conceptual layers:

1. **Identity** — the canonical fields that decide whether two
   mentions refer to the same concept. `family` is required;
   `size` and `stage` populate when surface evidence supports
   them. Anything else identity-relevant the LLM cannot place
   into a named field goes into `identity.extra` (catch-all
   defaults to identity, NOT aux — identity is the safer side:
   conservative merges).
2. **Auxiliary (aux)** — identity-LEVEL lossless info that
   should match across mentions of the same concept (release
   `date`, `mix_size`, `context_length`, `version`, source-local
   labels). Two mentions with the same identity but different
   non-null aux values for the same key are treated as a
   conflict, not merged.
3. **Aliases** — surface variants of the SAME concept. Each
   alias carries its own `descriptors` (lossless distribution
   facets that DO NOT affect identity: `quantization`,
   `precision`, `file_format`, `format`, mirror namespace). An
   alias may also carry its own `links` list when the variant
   has a distinct public release (e.g.,
   `Org/Qwen3-7B-Instruct-FP8`).

- `atoms`: ordered name pieces as the source presents them. Preserve
  protected spans when punctuation is misleading, e.g. `Qwen3Guard`
  can be `["Qwen3", "Guard"]` if release evidence supports it.
- `identity`: named-field identity object. Example
  `{"family": "Qwen3", "size": "7B", "stage": "Instruct"}`. Use
  `identity.extra` for identity-relevant info that doesn't fit
  a named field. Date tokens that distinguish snapshots
  (`OLMo-3-1025` vs `OLMo-3-1125`) belong in `identity.extra.date`.
- `concept_path`: reviewed lattice path from general to specific.
  Mirrors identity. Examples: `["Qwen3"]`, `["Qwen3", "VL"]`,
  `["Dolma3", "longmino"]`, `["FineMath", "3plus"]`.
- `links`: exact public release identifiers for the
  primary mention. Use `hf_model`, `hf_dataset`,
  `hf_dataset_config`, `github_repo`, `github_ref`,
  `api_model_id`, `official_release_url`, or `paper_release`.
- `aux`: lossless descriptive info that should match across
  mentions of the same concept (release `date`, `mix_size`,
  `context_length`, `version`, `organization`, source-local
  labels). Per-variant traits like `quantization`, `precision`,
  `format`, `file_format`, `namespace` go on the alias's
  `descriptors` instead, so they don't block alias-merging.
- `aliases[i].descriptors`: per-alias distribution facets like
  `quantization`, `precision`, `format`, `file_format`,
  `namespace` — info that distinguishes one alias from another
  without changing identity.
- `aliases[i].links` (optional): exact public links for the alias
  if the variant has its own public release (e.g., a quantized
  mirror under a different org).
- `context_roles`: open strings. Suggested roles include
  `training_data`, `pretraining_data`, `sft_data`,
  `preference_data`, `base_model`, `teacher_model`,
  `judge_model`, `generator_model`, `filter_or_classifier`,
  `evaluation_benchmark`, `comparison_baseline`,
  `released_artifact`, and `unknown`.

Every concrete entity leaf must have an exact link (HF repo path,
GitHub repo, API model id, official release URL, or paper release).
A broad technical report, family blog, or general project page is
a source anchor — not an entity-defining link — unless it is the
only exact release record for a paper-only model or dataset.

Same display names can appear twice when they refer to different node
types. For example, `Qwen3-4B` may be a concept node covering all
Qwen3 4B releases, while `Qwen/Qwen3-4B` is a concrete HF model
entity under that concept path.

## Aliases are NOT for these

Aliases are surface variants of THE SAME mention's referent ONLY
(different casing, hyphen/underscore variants, repo path forms,
mirrors that point at the same release, quantization/format
variants of the canonical). Aliases are NEVER:

- **Constituent members** of a composite. Wikipedia is not an
  alias of `Dolma3`; it's a separate dataset entity that
  `Dolma3`'s description claims as upstream.
- **Upstream parents.** If `Y` is derived from `X`, `X` is not
  an alias of `Y` — it's a separate entity. Emit it as its own
  top-level mention with its own anchors.
- **Sibling cuts / subsets.** `finemath-3plus` and
  `finemath-4plus` are not aliases of each other; both are
  configs/subsets of `HuggingFaceTB/finemath` and emit as
  distinct mentions with `hf_dataset_config` links.
- **Downstream consumers.** `Olmo-3-Instruct` is not an alias
  of the SFT dataset it consumed.

The diagnostic: ask "would a reader see this name and know it's
the same artifact?" If the answer is "no, that's a different
thing — they're related though", it's not an alias. Emit the
related artifact as its own top-level mention. Cross-entity
relationship modeling is a future stage; this prototype does
not capture relationship edges.

## Subsets / configs vs separate entities

A subset is an HF multi-config sibling under one repo
(`finemath-3plus` under `HuggingFaceTB/finemath`) or a named
quality cut of one release. Encode as a separate mention with
link type `hf_dataset_config` and value `repo::config`.

Separate-publication test: if a candidate-subset has its OWN HF
repo path, its OWN card and authors, its OWN paper — it is its
own entity, not a subset. Emit it as a top-level mention with
its own `hf_dataset` or `hf_model` link.

## Don't fabricate slash paths for vague references

Sources reference artifacts at any tier of the family hierarchy:
`Qwen3` (family), `Qwen3-Base` (stage), `Qwen/Qwen3-7B-Base`
(entity). When the source uses a non-leaf tier:

- If the abstraction has its own public artifact (a family
  paper, a GitHub org, a project homepage), emit the mention
  with that link (`paper_release` for the technical report,
  `github_repo` for the org). The bare name is the mention
  surface; do not fabricate a slash path.
- If the abstraction has no public link at that tier (common
  for stage names like `Qwen3-Base`), emit the mention as a
  concept-only referent (`referent_scope: "concept"`,
  `concept_path: ["Qwen3", "Base"]`, no `links`).
  Lattice mints the concept node; reviewer can attach entities
  later.

Never emit fabricated paths like `Qwen/Qwen3` or
`Qwen/Qwen3-Base` — these don't exist, fail link verification,
and drop information.

## Open vocabularies

Semantic fields (`context_roles`, `aux` keys, alias
`descriptors` keys) are open strings. Common values are listed
as suggestions; coin a new phrase if none fit. Do not gate on
closed enums. The schema only constrains structural fields
(`kind`, `node_type`, `link type`).

## Description conventions: neutral framing

Describe artifacts without anchoring on a particular target
investigation OR on the source you happened to read. Banned
phrasings:

- *target-framing*: "used by `<target>`", "in `<target>`'s
  training", possessive forms tying the artifact to a downstream
  consumer.
- *source-framing*: "referenced by `<source>`", "as named in
  `<filename>`", "mentioned by `<repo>`". The source is recorded
  in the mention's source-side anchors, not in prose.

The same mention should read identically across investigations
and across sources. If you're tempted to write "as referenced
by …", you're source-framing.

## Dispatching subagent work

You delegate when the work fans out cleanly — long sources,
large mention sets, topical chunks. The subagent runtime varies
(Claude task subagents now; codex / other runtimes possibly
later); these principles apply regardless.

- **Right-size the unit.** Group source material into topically
  coherent units before dispatching. Too narrow (one-per-file,
  one-per-record) duplicates context-loading and turns the
  planner into the worker. Too wide (the entire batch in one
  call) dilutes attention. Aim for the unit a careful reader
  would naturally treat as one pass.
- **Brief like a stranger.** Subagents work with weaker context
  than you have. When a subagent needs a rule from this prompt,
  transcribe it verbatim — don't paraphrase or summarize. Rule
  erosion at dispatch is the most common cause of subagent
  output drifting from the rules the orchestrator was given.

## Agent I/O conventions

All agent-facing I/O uses surface names, anchor values, and
`mention_id` strings — never database UUIDs. `{{artifact_path}}`
is always where you write your output. `{{input_path}}` is
always where you read your input. Filesystem scope sections in
each prompt list the only paths you may read or write.
