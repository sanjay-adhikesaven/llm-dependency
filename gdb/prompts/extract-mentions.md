# Extract Mentions

> **Goal: COVERAGE plus inline confirmation.** Find every named
> model and dataset this batch's sources mention, emit them
> with rich identity, aux, and alias structure, and confirm
> their HF identifier / publishing organization / API endpoint
> when the artifact has a public release. Identity refinement
> happens in later stages — your job is to describe what the
> source says (with confirmed anchors when available), not to
> pre-classify.

This is a **name-first first pass**: extract the surface name,
ordered atoms, obvious exact anchors, roles, and evidence; don't
force every token into a predefined identity field — uncertain
pieces go into `identity.extra` (catch-all) or `aux`, not into
fabricated structure.

Read `{{batch_dir}}` and write model/dataset-only name mentions to
`{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: tab-separated filename, source id,
  title. Cite evidence by the filename column.

## Filesystem scope

Read `{{batch_dir}}` and `{{input_path}}`. Write
`{{artifact_path}}`. Do not read or write any other local path.
Web search and HF API / page fetches are allowed for the sole
purpose of confirming a mention's HF identifier, organization,
and API endpoint — not for general source discovery.

Output:

```json
{
  "mentions": [
    {
      "surface": "Qwen/Qwen3-4B",
      "kind": "model",
      "atoms": ["Qwen3", "4B"],
      "referent_scope": "entity",
      "identity": {"family": "Qwen3", "size": "4B"},
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

## Rules

- Emit model and dataset mentions only.
- Extract from prose, tables, model/dataset cards, YAML, JSON,
  and code-shaped calls (`from_pretrained`, `load_dataset`,
  `model_name_or_path`, `tokenizer_name`, `dataset_name`).
- Every mention needs non-empty evidence with a verbatim excerpt.
- Don't use the target as an identity field. Role tags capture
  how an artifact is used by the target.

### Identity, aux, aliases

Populate `identity` as a named-field object: `family` is
required; add `size` and `stage` when the surface supports
them. Anything identity-relevant that doesn't fit a named
field goes in `identity.extra` (catch-all defaults to identity
for safer merges). `concept_path` mirrors identity from
general to specific. Identity tokens that distinguish
snapshots (`OLMo-3-1025` vs `OLMo-3-1125`) belong in
`identity.extra.date` — they make the mentions cluster
separately. Per-concept lossless info (`context_length`,
`mix_size`, `organization`) belongs in `aux`. Per-variant
distribution facets (`quantization`, `precision`, `format`,
`namespace`) belong on the alias's `descriptors`, not on
`aux`.

### Canonical surface

When the source uses multiple forms for one artifact, the
mention's `surface` is the most-canonical form available; all
other forms go in `aliases[]`. Preference order:

1. **HF repo path** — `Qwen/Qwen3-7B-Base`, `HuggingFaceTB/finemath`.
2. **GitHub repo path** — `microsoft/MASS`, `mlfoundations/open_lm`.
3. **Closed-source vendor form** — `OpenAI/GPT-4`,
   `Anthropic/Claude-3.5-Sonnet` (slash mirrors the org-or-vendor
   shape even when the vendor doesn't publish at that path).
4. **Bare canonical** for paper-only abstractions and benchmarks
   with no per-release identifier — `Common Crawl`, `Wikipedia`,
   `MATH`, `AIME-2024`.

Papers are LINKS, not names: when a paper introduces an
artifact, the artifact is the mention surface; the arxiv URL
is its `paper_release` anchor. Do not extract paper TITLES as
mention surfaces.

### Inline anchor confirmation

For each mention, attempt to confirm three pieces inline:
(1) the HF identifier (or paper / GitHub / API id),
(2) the publishing organization, and
(3) any documented API endpoint. Web search and HF API calls
are permitted for confirmation. Some mentions are inherently
vague (`Qwen3` without a stage, internal models without a
release) — emit them as concept-only and move on. Don't
fabricate anchors to avoid the empty case.

### Aliases, subsets, vague references

See shared-context for strict definitions. Subsets / configs
(`finemath-3plus`) emit as separate mentions with
`hf_dataset_config` anchors. Vague references emit as
concept-only mentions (`referent_scope: "concept"`).

## Dataset-specific anchor fields

For every dataset mention with a confirmable public release,
populate the appropriate anchor types (multiple are fine; the
lattice picks the primary by priority and keeps the others):

1. **HF identifier** — `hf_dataset` for whole datasets
   (`HuggingFaceTB/finemath`); `hf_dataset_config` with
   `<repo>::<config>` for named subsets / configs
   (`HuggingFaceTB/finemath::finemath-3plus`).
2. **GitHub repository** — `github_repo` (`<org>/<repo>`) when
   the dataset has a code repo (`microsoft/MASS`,
   `allenai/dolma`).
3. **Official release source** — `official_release_url` for the
   first-party release page when neither HF nor GitHub is
   first-party. If the source explicitly identifies the
   first-party release as elsewhere (e.g., "MASS, originally
   released at github.com/microsoft/MASS") AND a HF page exists
   that is not first-party, mark the HF anchor with
   `metadata: {"mirror": true}` so reviewers can prefer the
   official source.
4. **Subsets** — when the source describes a parent dataset
   with named children (FineMath has `finemath-3plus`,
   `finemath-4plus`; Dolma3 has named mixtures), emit each
   child as its own mention with the appropriate
   `hf_dataset_config` (or `hf_dataset` if separately
   published) anchor. Do NOT collapse children into the
   parent's `aliases[]`.

## Quantization, format, precision, mirrors

When a surface name ends in one of these suffix patterns, do NOT
mint a separate top-level mention. Strip the suffix to recover the
canonical surface; emit the canonical as the primary mention; emit
the original surface as an alias whose `descriptors` record the
suffix info. If the variant has its own HF or GitHub release,
attach those anchors to the alias's `anchors` list.

Suffix patterns:
- Quantization: `-FP8`, `-FP16`, `-BF16`, `-Q[0-9]+(_[A-Z0-9]+)*`
  (e.g., `-Q4_K_M`, `-Q8_0`), `-AWQ`, `-GPTQ`, `-EXL2`,
  `-BNB-4bit`, `-INT4`, `-INT8`.
- File format / runtime: `-GGUF`, `-MLX`, `-SafeTensors`.

If the canonical surface is not present anywhere in this batch and
only the variant is, still emit the canonical as the primary
mention and the variant as alias. The check-mentions / repair
stage will reconcile if a sibling batch carries the canonical.

## Worked examples

### 1. Quantization variant collapses to alias

Source mentions both `Qwen3-7B-Instruct` (HF `Qwen/Qwen3-7B-Instruct`)
and `Qwen3-7B-Instruct-FP8` (HF `Org/Qwen3-7B-Instruct-FP8`).

Emit ONE mention:

```json
{
  "surface": "Qwen3-7B-Instruct",
  "kind": "model",
  "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
  "concept_path": ["Qwen3", "7B", "Instruct"],
  "anchor_candidates": [
    {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": true}
  ],
  "aliases": [
    {"surface": "Qwen3-7B-Instruct", "descriptors": {}},
    {"surface": "Qwen3-7B-Instruct-FP8",
     "descriptors": {"quantization": "FP8"},
     "anchors": [{"type": "hf_model",
                  "value": "Org/Qwen3-7B-Instruct-FP8",
                  "exact": true}]}
  ]
}
```

### 2. Date-versioned snapshots are SEPARATE entities

`OLMo-3-1025-7B-Base` and `OLMo-3-1125-7B-Base` are two snapshots
(different release months). Emit TWO mentions; date is in
`identity.extra.date` so they don't merge:

```json
[
  {"surface": "Olmo-3-1025-7B-Base", "kind": "model",
   "identity": {"family": "Olmo-3", "size": "7B", "stage": "Base",
                "extra": {"date": "1025"}},
   "concept_path": ["Olmo-3", "7B", "Base"],
   "anchor_candidates": [{"type": "hf_model",
                          "value": "allenai/Olmo-3-1025-7B", "exact": true}]},
  {"surface": "Olmo-3-1125-7B-Base", "kind": "model",
   "identity": {"family": "Olmo-3", "size": "7B", "stage": "Base",
                "extra": {"date": "1125"}},
   "concept_path": ["Olmo-3", "7B", "Base"],
   "anchor_candidates": [{"type": "hf_model",
                          "value": "allenai/Olmo-3-1125-7B", "exact": true}]}
]
```

### 3. HF dataset config under parent repo

`HuggingFaceTB/finemath` parent + `finemath-3plus` config: emit TWO
mentions. The config's anchor type is `hf_dataset_config` with
value `<repo>::<config>`.

```json
[
  {"surface": "HuggingFaceTB/finemath", "kind": "dataset",
   "identity": {"family": "FineMath"},
   "concept_path": ["FineMath"],
   "anchor_candidates": [{"type": "hf_dataset",
                          "value": "HuggingFaceTB/finemath", "exact": true}]},
  {"surface": "finemath-3plus", "kind": "dataset",
   "identity": {"family": "FineMath", "stage": "3plus"},
   "concept_path": ["FineMath", "3plus"],
   "anchor_candidates": [{"type": "hf_dataset_config",
                          "value": "HuggingFaceTB/finemath::finemath-3plus",
                          "exact": true}]}
]
```

### 4. GitHub-canonical dataset (HF mirror)

If the source explicitly says the dataset's official release is on
GitHub (e.g., "MASS, originally released at github.com/microsoft/MASS"),
emit `github_repo` as the primary anchor candidate. If a HF mirror
also exists, include it but mark `mirror: true` in its candidate
metadata.

```json
{"surface": "MASS", "kind": "dataset",
 "identity": {"family": "MASS"},
 "concept_path": ["MASS"],
 "anchor_candidates": [
   {"type": "github_repo", "value": "microsoft/MASS", "exact": true},
   {"type": "hf_dataset", "value": "OtherOrg/MASS-mirror", "exact": true,
    "metadata": {"mirror": true}}
 ]}
```

## Within-batch merge

When the same artifact appears in multiple passages of this
batch, emit ONE mention with multiple `evidence[]` entries and
a unioned alias list. Do not duplicate. Aliases get merged by
casefolded surface; descriptors and per-alias anchors are
combined.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`mentions[]` list is legal only if the batch genuinely contains
no model or dataset names; if it does and you wrote none,
that's a misread.

You are running as `{{planner_model}}`. Use subagents for independent
source packets; subagents run as `{{subagent_model}}`.
