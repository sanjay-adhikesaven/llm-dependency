# Extract Mentions

> **Goal: COVERAGE of the names + their atomization + minimal
> identity + inline link confirmation.** Find every named model
> and dataset this batch's sources mention. Atomize each name
> into an ordered list of pieces. Tag each mention with its
> minimal identity (`family` is required; `size` and `stage`
> when the surface plainly carries them). Capture the typed
> link inline when the source states it explicitly (a
> `from_pretrained` call, a HuggingFace URL, a GitHub repo
> path). Anchor each mention to the spot in the source where it
> appears.

This is a **lightweight first pass**. Alias collapse, aux
facets, conflict resolution, and description writing all
happen later in the audit and describe stages — they need the
cluster context that this single-batch pass doesn't have.
Don't pre-classify beyond the minimal identity above.

Read `{{batch_dir}}` and write the artifact to
`{{artifact_path}}`.

## Inputs

- `{{input_path}}`: JSON with `batch_id` and `batch_dir`.
- `{{batch_dir}}/MANIFEST.txt`: tab-separated filename, source id,
  title. Use the filename column when citing source-side anchors.

## Filesystem scope

Read `{{batch_dir}}` and `{{input_path}}`. Write
`{{artifact_path}}`. Do not read or write any other local path.
Web search and HF API / page fetches are allowed for the sole
purpose of confirming a mention's HF identifier when the source
text doesn't already give the exact form (e.g., the source says
`Qwen3-4B` but the HF id is `Qwen/Qwen3-4B`).

## Output

```json
{
  "mentions": [
    {
      "surface": "Qwen/Qwen3-4B",
      "kind": "model",
      "identity": {"family": "Qwen3", "size": "4B"},
      "atoms": ["Qwen3", "4B"],
      "links": [
        {"type": "hf_model", "value": "Qwen/Qwen3-4B", "exact": true}
      ],
      "anchors": [
        {"file": "config.py", "source_id": "...", "location": "L10",
         "excerpt": "model_name = \"Qwen/Qwen3-4B\""}
      ]
    }
  ]
}
```

`identity.family` is REQUIRED on every mention. `size` and
`stage` populate when the surface carries them
(`Qwen3-7B-Instruct` → `{family: Qwen3, size: 7B, stage:
Instruct}`). Date snapshots that distinguish releases go in
`identity.extra.date` (`OLMo-3-1025` → `extra.date: "1025"`).
Multi-token families like `Qwen3-VL`, `Qwen3-Coder`,
`Llama-3.1`, `HuggingFaceTB/finemath` stay intact in `family`.

## Rules

- Emit model and dataset mentions only. License, software
  packages, frameworks, and tokenizers are out of scope and the
  storage layer rejects them.
- Extract from prose, tables, model/dataset cards, YAML, JSON,
  and code-shaped calls (`from_pretrained`, `load_dataset`,
  `model_name_or_path`, `tokenizer_name`, `dataset_name`,
  config files). Code-shaped calls almost always give the exact
  HF id — capture the link inline.
- Every mention needs at least one anchor with a verbatim
  excerpt and the file path.

## Atomization (HF-collection-aware)

Atoms are the ordered name pieces the source presents. The
**leftmost atoms are the most general**; size and stage tokens
sit on the right.

The default split is on `[-_/:\s]+`, BUT family-name punctuation
must be preserved. Hugging Face publishes peer collections at
the same tier (e.g. Qwen org publishes `Qwen3`, `Qwen3-VL`,
`Qwen3-Coder`, `Qwen3Guard`, `Qwen3-Embedding`,
`Qwen3-VL-Embedding` as **peers** — none nested under another).
Hyphens inside a family name are not separator hyphens.

Worked examples:

| Surface | atoms |
|---|---|
| `Qwen3-7B-Instruct` | `["Qwen3", "7B", "Instruct"]` |
| `Qwen3-VL-72B-Instruct` | `["Qwen3-VL", "72B", "Instruct"]` |
| `Qwen3Guard-Stream-7B` | `["Qwen3Guard", "Stream", "7B"]` |
| `Qwen3-Coder-30B-A3B-Instruct` | `["Qwen3-Coder", "30B-A3B", "Instruct"]` |
| `Llama-3.1-70B-Instruct` | `["Llama-3.1", "70B", "Instruct"]` |
| `OLMo-3-1025-7B-Base` | `["OLMo-3", "1025", "7B", "Base"]` |
| `dolma3_longmino_mix-100B-1125` | `["dolma3", "longmino", "mix", "100B", "1125"]` |
| `HuggingFaceTB/finemath::finemath-3plus` | `["HuggingFaceTB", "finemath", "finemath-3plus"]` |

When uncertain whether a hyphen is a family-internal punctuation
or a tier separator, check the org's HF collections page. If
`Qwen3-VL` appears as its own collection, it's one atom.

## Inline link confirmation

When the source explicitly names the link, emit it:

- `from_pretrained("Qwen/Qwen3-4B")` → `{type: "hf_model", value:
  "Qwen/Qwen3-4B", exact: true}`.
- `load_dataset("HuggingFaceTB/finemath", "finemath-3plus")` →
  `{type: "hf_dataset_config", value:
  "HuggingFaceTB/finemath::finemath-3plus", exact: true}`.
- A URL `https://huggingface.co/datasets/HuggingFaceTB/finemath`
  → `{type: "hf_dataset", value: "HuggingFaceTB/finemath",
  exact: true}`.
- An arxiv URL or `arXiv:2404.12345` → `{type: "paper_release",
  value: "https://arxiv.org/abs/2404.12345", exact: true}`.
- A vendor model id like `gpt-4o-mini-2024-07-18` →
  `{type: "api_model_id", value: "gpt-4o-mini-2024-07-18",
  exact: true}`. **NOT** `hf_model`.

When the source uses a display form (e.g., `FineMath-3+`,
`Qwen3-4B` in prose with no HF id nearby), it's fine to leave
`links: []`. The audit stage will web-search and add a link.

### Display vs config strings

HF dataset configs have programmatic ids (loadable via
`load_dataset(repo, "<config-id>")`) that can differ from the
human display string:

| Display in prose | Loadable config id |
|---|---|
| `FineMath-3+` | `finemath-3plus` |
| `FineMath-4+` | `finemath-4plus` |
| `InfiMM-WebMath-3+` | `infiwebmath-3plus` |
| `InfiMM-WebMath-4+` | `infiwebmath-4plus` |

When you confirm the link inline, use the loadable id (the form
that goes in `<repo>::<config>`).

## Quantization, format, precision

If a surface name ends in a quantization or format suffix
(`-FP8`, `-FP16`, `-BF16`, `-Q\d+(_\w+)*`, `-AWQ`, `-GPTQ`,
`-EXL2`, `-BNB-4bit`, `-INT4`, `-INT8`, `-GGUF`, `-MLX`,
`-SafeTensors`), emit it as a top-level mention as you found it.
Do NOT pre-collapse it as an alias of the canonical here — the
audit stage has cluster context and decides whether to fold it
into the canonical's aliases.

## Date-versioned snapshots

Date tokens that distinguish snapshots
(`OLMo-3-1025-7B-Base` vs `OLMo-3-1125-7B-Base`,
`gpt-4o-mini-2024-07-18` vs `gpt-4o-mini-2024-08-06`) come
through as their own atoms. Each surface is its own mention;
audit will keep them as separate clusters via
`identity.extra.date`.

## Subsets and configs

If the source names both a parent dataset and a config (e.g.,
`HuggingFaceTB/finemath` and `finemath-3plus`), emit BOTH as
separate mentions. The parent gets `links: [{"type":
"hf_dataset", "value": "HuggingFaceTB/finemath"}]`; the config
gets `links: [{"type": "hf_dataset_config", "value":
"HuggingFaceTB/finemath::finemath-3plus"}]`. Don't collapse the
config into the parent's aliases — they're sibling configs
under one repo.

## Within-batch merge

When the same artifact appears in multiple passages of THIS
batch, emit one mention with multiple `anchors[]` entries. Do
not duplicate. Cross-batch merge is handled by the check stage.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0. An empty
`mentions[]` list is legal only if the batch genuinely contains
no model or dataset names; if it does and you wrote none,
that's a misread.

You are running as `{{planner_model}}`. Use subagents for
independent source packets within the batch; subagents run as
`{{subagent_model}}`.
