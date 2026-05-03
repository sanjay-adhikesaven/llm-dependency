# Extract Names

> **Goal: list every model and dataset this batch mentions, in
> the most specific form the source uses.** Output one line per
> mention, with `type` ('model' or 'dataset') and `name`.
> Nothing else.

Read `{{batch_dir}}` and write the artifact to
`{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: tab-separated filename, source
  id, title — for orientation only; you do not cite it.

## Filesystem scope

Read `{{batch_dir}}` and `{{input_path}}`. Write
`{{artifact_path}}`. Do not read or write any other local path.
No web browsing. No HF API calls. No classification beyond
choosing `model` vs `dataset`. No fuzzy matching, normalization,
or deduplication beyond removing literal duplicates of the same
(type, name) pair within this batch.

## Output schema

```json
{
  "mentions": [
    {"type": "model",   "name": "Qwen/Qwen3-7B-Instruct"},
    {"type": "model",   "name": "Qwen3-7B-Instruct-FP8"},
    {"type": "dataset", "name": "HuggingFaceTB/finemath"},
    {"type": "dataset", "name": "finemath-3plus"},
    {"type": "model",   "name": "OLMo-3-1025-7B-Base"},
    {"type": "dataset", "name": "MMLU-Pro"}
  ]
}
```

The artifact has exactly one key, `mentions`, holding a list of
records. Each record has exactly two fields: `type` (`"model"`
or `"dataset"`) and `name` (the verbatim name string from the
source). Do NOT emit any other field — no kind aliases, no
identity, no atoms, no anchors, no links, no descriptions, no
file references, no excerpts, no aliases, no aux, no
descriptors, no concept_path. Adding extra fields will be
ignored, but bloats the artifact.

## What counts

A noun-phrase the source uses to refer to a specific model or
dataset. Examples (each line is one valid name):

```
Qwen2.5-7B-Instruct
Qwen/Qwen3-4B
Llama 3.1
OLMo-3-1025-7B-Base
allenai/dolma3-fasttext-quality-classifier
HuggingFaceTB/finemath
finemath-3plus
FineMath-3+
gpt-4o-mini-2024-07-18
MMLU-Pro
AIME 2024
DeepSeek-R1
Qwen3-7B-Instruct-FP8
```

### Neural artifacts framed as tools

Some sources name a neural model with tool-style phrasing
("using olmOCR (Poznanski et al., 2025a,b)", "we judge with
GPT-4"). The model is still a model. **Emit it as `model`** when
the surface names a *neural artifact that processes data*
(generates, transforms, filters, judges, embeds), even when the
prose frames it as a tool. Cues to look for:

- it appears alongside a paper citation (`(Author et al., 20XX)`)
- it is named after a model card slug (`allenai/olmOCR-7B-0225`,
  `Qwen/Qwen3-32B`)
- it is described as performing a learned function — OCR,
  classification, judging, rewriting, embedding, ranking

When in doubt, emit. The organize / audit stages drop noise via
web verification; missing a real model is irrecoverable.

### Skip these (they aren't model/dataset names)

- license names (`Apache-2.0`, `MIT`, `CC-BY-4.0`)
- pure-software packages, frameworks, tokenizers as such
  (`transformers`, `vLLM`, `datatrove`, `tiktoken`,
  `pyarrow`) — non-neural libraries only. Neural artifacts
  framed as tools (above) are NOT skipped.
- author/organization names by themselves (`Anthropic`,
  `Meta`, `Allen Institute for AI`)
- paper titles
- task / capability names that are NOT a dataset
  (e.g., "math reasoning" the task vs. `MATH-500` the dataset)

## "Most specific form" means

Match what the source actually wrote. Don't generalize, don't
expand.

- Source says `Qwen2.5-7B-Instruct` → emit `Qwen2.5-7B-Instruct`.
  Don't generalize to `Qwen2.5`.
- Source says `the Qwen3 family of models` → emit `Qwen3`. The
  source said `Qwen3`. Don't invent a size.
- Source says `from_pretrained("Qwen/Qwen3-4B")` → emit
  `Qwen/Qwen3-4B` exactly (with the org prefix, because the
  source has it).
- Source says `Tülu 3` (with umlaut) → emit `Tülu 3`. Keep the
  original characters; canonicalization happens later.
- A surface like `Qwen3-7B-Instruct-FP8` → emit it as is. Do
  not collapse to its canonical.
- Source mentions a dataset config: `load_dataset("HuggingFaceTB/finemath", "finemath-3plus")` →
  emit two records: `{"type": "dataset", "name": "HuggingFaceTB/finemath"}`
  AND `{"type": "dataset", "name": "finemath-3plus"}`.

## Within-batch dedup

If the source mentions the same `(type, name)` pair multiple
times, emit it once. Don't emit a (type, name) pair twice.
Do NOT collapse name variants — `Qwen3-7B`, `Qwen 3 7B`, and
`qwen3-7b` are three separate records here. The organize stage
handles fuzzy matching across name variants.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`mentions[]` list is legal only if the batch genuinely contains
no model or dataset names.

You are running as `{{planner_model}}`. Use subagents for
independent source packets within the batch. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
