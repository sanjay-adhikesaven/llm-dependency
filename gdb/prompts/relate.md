# Extract Operations and Edges Anchored to the Lattice

> **Goal: read this batch's source files, identify every
> training / filtering / generation / transformation EVENT
> (an `operation`), then emit one EDGE per participant with
> a closed-enum `relation` classification AND a per-edge
> description. Subjects are forced to be lattice
> `formal_name`s — that's the controllability lever.**

Read the lattice at `{{lattice_path}}` and the source files
under `{{batch_dir}}`. Write operations + edges to
`{{artifact_path}}`.

## Operations vs edges (load-bearing)

A real training event involves multiple participants playing
different roles. The Olmo-3 7B Think RLVR run, for example,
involves at least four participants:

- the resulting model (`Olmo-3-7B-Think`)
- the base checkpoint (`Olmo-3-7B-Think-DPO`)
- the training data (`Dolci-Think-RL-7B`)
- the judge model (`Qwen/Qwen3-32B`)

We capture this as **one `operation`** (a first-class record
with a lossless prose description of the whole event) and
**N `edges`** (lightweight participant pointers), all sharing
the same `operation_id`.

Why this shape: pairwise edges alone lose the structural fact
that these participants belong to the SAME event. A
downstream query like "what models participated in the same
training event as Qwen3-32B as judge?" needs operation
grouping. License inheritance and contamination tracing also
need it: license flows along the operation, not along
disjoint pairwise edges.

Every edge in `edges[]` MUST point at an operation in
`operations[]` via `operation_id` — except for STRUCTURAL
literal-value relations (size, training_tokens, etc.) and
INDIRECT relations like `used_for_evaluation` /
`cited_as_baseline` that aren't part of a training pipeline
event. Those may carry `operation_id: null`.

## Filesystem scope

Read `{{lattice_path}}` (groups+items+links artifact from
linker / audit / organize) and every file under
`{{batch_dir}}` recursively (PDFs, markdown, code, configs).
Skip `__pycache__`, `node_modules`, `.git`, `venv`. Do not
read anything outside `{{batch_dir}}` except the lattice.

Web search is **off** for this stage — we want grounded claims
from the source files, not synthesized knowledge.

## What "lattice-anchored" means

The lattice gives you a closed set of canonical entities.
Every `subject` you emit MUST be a `formal_name` taken
verbatim from the lattice. If a source mentions a thing that
maps to a lattice entity by alias, normalize the surface form
to the canonical `formal_name` before emitting.

If a source mentions something that is *not* in the lattice
and *cannot* be normalized to one (e.g., a researcher's
personal-namespace HF dataset, a one-off internal codename,
**a frontier API model like GPT-4.1 or o4-mini that has no
HF/GitHub link**), you have two valid moves:

1. If it appears as the **object** of a relation whose subject
   is in the lattice — emit the relation with `object_ref:
   null`, `object_text: "<the literal string from source>"`,
   `object_in_lattice: false`. **This includes well-known API
   judges and synthetic-data generators** — they are real
   entities and the edge to them is load-bearing for license
   inheritance and provenance. Don't drop the edge just
   because the object isn't a HF artifact.
2. If it appears alone (no relation to any lattice entity)
   — drop it. Don't invent relations.

`subject_in_lattice` is always `true`. We don't extract
relations between two off-lattice entities; both endpoints
need to anchor to canonical names for the system to compare
across batches.

### Global-policy edges (load-bearing)

When the source describes a model as the **global** judge,
generator, filter, or transformer for a **class** of training
stages, emit ONE edge per training-stage subject in the
lattice that participates in that class.

Example: the OLMo-3 paper says "Unless otherwise stated, for
an LM judge we host **Qwen3 32B** ... thinking mode turned
off". This is a global default for ALL OLMo-3 RLVR runs.
Emit `filtered_by Qwen/Qwen3-32B` for **every** OLMo-3
RLVR-trained model: `Olmo-3-7B-Think`, `Olmo-3-32B-Think`,
`Olmo-3.1-32B-Think`, `Olmo-3-7B-Instruct`, the 5+ RL-Zero
variants — that's ~10 edges from one global statement.

Same pattern for global generators: "we distill thinking
traces from GPT-4.1 and o4-mini" → emit `distilled_from`
edges to both objects (off-lattice via `object_text` if
needed) for every model whose midtraining/post-training
mixture incorporated those traces.

A global statement that you DO NOT expand to per-stage edges
will manifest downstream as missing license-inheritance and
provenance edges. Err on the side of expanding.

## Relation taxonomy (canonical labels + coining when needed)

`relation` is an **open string**, but the canonical labels
below cover the great majority of LLM-pipeline events. The 5
DIRECT and 3 INDIRECT buckets are designed to be orthogonal —
every direct dependency that fits one of them lands in exactly
one. STRUCTURAL is for artifact-lineage links that aren't
training-pipeline events.

**Use a canonical label when one fits.** Only coin a new
snake_case label when the source describes an event that none
of the canonical values capture. Examples of legitimate
coining:

- `merged_from` — model souping / weight averaging across
  multiple checkpoint variants (the canonical
  `initialized_from` is one-to-one).
- `deduplicated_by` — a deduplication pipeline that strips
  documents from a corpus (close to `transformed_by` but the
  reader benefits from the more specific label).
- `embedded_by` / `tokenized_by` / `decontaminated_by` — when
  these are the named operation in the source.

**Do NOT coin** when the canonical label fits. The planner
that emits `cited_as_baseline` instead of `used_for_evaluation`
is an example of bad coining: comparison baselines ARE
evaluation usage. The cost of frivolous coining is downstream
analysis splitting on cosmetic variants.

When you coin, the edge `description` MUST explain the role
clearly enough that a downstream reader understands what the
new label means without re-reading the source. Coining
substitutes prompt-level guidance for runtime LLM judgment;
make the description carry that weight.

### DIRECT relations (subject's training pipeline depended on object)

| relation | subject→object | when to use |
|---|---|---|
| `trained_on` | M→D | the dataset is pre-training or post-training data for the model |
| `initialized_from` | M→M | weight initialization (the subject's parameters are copied / continued from the object) |
| `distilled_from` | M→M | the subject's training data was **generated by** the object — original content comes *from the object model itself* (synthetic data, instruction generation, RL rollouts, distillation traces) |
| `transformed_by` | M→D, M→M | the subject's training data was **rewritten / OCR'd / reformatted** by the object — content originated *somewhere else* and the object only modified it |
| `filtered_by` | M→M | the object **decided inclusion** of training samples (judge in pairwise preference, quality-score classifier, reward model used as filter) — content unchanged, only included or excluded |

### INDIRECT relations (object shaped subject without entering training)

| relation | subject→object | when to use |
|---|---|---|
| `inspired_by` | M→M | methodology borrowed; no weight or data inheritance |
| `used_for_ablation` | M→M, M→D | used in an ablation experiment, not the production training run |
| `used_for_evaluation` | M→D, M→M | benchmark / eval set OR comparison baseline reported in the paper. **Use this for comparison baselines too** — do NOT invent `cited_as_baseline` or any other relation; baselines are evaluation usage. |

### STRUCTURAL relations (artifact-to-artifact lineage, not training-pipeline events)

| relation | subject→object | when to use |
|---|---|---|
| `subset_of` | D→D | the subject is a subset / filtered copy of the parent |
| `supersedes` | M/D → M/D | the subject replaces a predecessor |
| `released_with` | M↔M/D | tokenizer or companion artifact bundled with a release |
| `contains` | D→D | the subject dataset bundles the object dataset |

## Why no overlap among the 5 DIRECT buckets

Two orthogonal axes separate them:

- **Content origin** — where did the training content come from?
  - external data → `trained_on`, `transformed_by`, `filtered_by`
  - external weights → `initialized_from`
  - the object model itself → `distilled_from`

- **What the object did to the content**:
  - the object IS the content (weights or generated text) →
    `initialized_from`, `distilled_from`, `trained_on`
  - the object **rewrote** existing content → `transformed_by`
  - the object **only decided inclusion** → `filtered_by`

```
                  | content from object | content from elsewhere, unchanged | content from elsewhere, modified
weights           | initialized_from    | —                                  | —
content (data)    | distilled_from      | trained_on (data) /                | transformed_by
                  |                     | filtered_by (decision-only model)  | (rewriter model)
```

Every M-to-M-or-M-to-D direct dependency lands in exactly one
cell. If you can't decide between two cells, the edge probably
needs to be split into two edges along different axes — emit
both.

### Worked examples

- `allenai/Olmo-3-7B-Instruct` is initialized from
  `allenai/Olmo-3-7B-Base` → `initialized_from`.
- `allenai/Olmo-3-7B-Base` was pretrained on
  `allenai/dolma3-mix` → `trained_on`.
- A paragraph in Dolma-3 was OCR'd from a PDF by
  `allenai/olmOCR-7B-0225`. Subject is the dataset that
  consumed the OCR'd content (or the model that consumed the
  dataset); object is the OCR model. → `transformed_by`. The
  *content* came from the PDF, not the OCR model.
- An RLHF DPO run for `allenai/Olmo-3-7B-Instruct` used
  `Qwen/Qwen3-32B` as the judge that picked preferred
  responses. → `filtered_by` (judge decided inclusion; it
  didn't generate the responses, it ranked them).
- `CraneMath` was generated by `Qwen/Qwen3-32B` rewriting math
  problems. The CraneMath *content* came from
  Qwen3-32B. From the model that trained on CraneMath:
  `distilled_from` Qwen3-32B. (If the rewriting was over
  pre-existing math problems, this could also be argued as
  `transformed_by` — the rule is: if the object model
  *substantively rewrote*, it's `transformed_by`; if the
  object model produced the content from scratch, it's
  `distilled_from`. Source phrasing decides.)
- The Olmo-3 paper reports MATH-500 scores for `allenai/Olmo-3-7B-Instruct`.
  → `used_for_evaluation`.
- Olmo-3 ablations include a configuration without DPO,
  measured on AlpacaEval2. AlpacaEval2 here is
  `used_for_ablation`.
- The Olmo-3 paper credits Tülu 3's recipe as the basis for
  its post-training mix. → `inspired_by`.

## Entity → literal value (numeric facts)

Numeric facts about lattice entities are stored as edges with
literal objects. `relation` for these is the property name (a
small closed set):

| relation | example value/unit | what it captures |
|---|---|---|
| `size` | 102014 / "prompts" | data size or sample count |
| `training_tokens` | 5.93e12 / "tokens" | total tokens model was trained on |
| `context_length` | 65536 / "tokens" | max context window |
| `release_date` | "2025-10-25" / "iso" | release date |
| `parameter_count` | 7e9 / "params" | model parameter count |
| `composition_count` | 29813 / "prompts" | size of one named subsource (use with `object_text` for the sub-source name) |

For these, set `direction: "STRUCTURAL"`, `object_ref: null`,
`object_in_lattice: false`, and put the value/unit in
`object_value` / `object_unit`. For `composition_count`, also
put the sub-source name in `object_text`.

If you observe a numeric fact that doesn't fit any of the
above, prefer not emitting over inventing. The closed
vocabulary is the point.

## Provenance kind (canonical labels + coining when needed)

Tag every relation with where exactly it came from. Use one
of the canonical labels below when it fits; coin a new
snake_case label when the source class is genuinely new.

**Canonical labels** (cover ~95% of cases):

- `paper_prose` — body text in a PDF / blog
- `paper_table` — a numbered table inside a PDF / blog
- `paper_figure` — a figure caption or in-figure label
- `hf_frontmatter` — YAML frontmatter of an HF README (the
  `base_model:` / `datasets:` / `license:` block)
- `hf_card_body` — the prose / tables under the YAML
- `script_flag` — a CLI flag in a `.sh` / launcher (e.g.
  `--dataset_mixer_list X 10000`)
- `code_constant` — a Python / YAML constant assignment
  (e.g. `MODEL = "o3"`, `DataMix.OLMo_midtraining_mix_0925`)
- `code_comment` — a `#` comment line near training code
- `config_yaml` — a non-script YAML / JSON config
- `markdown_doc` — internal doc markdown (e.g.
  `docs/olmo3.md`) that isn't an HF card

**Legitimate coining** when the source is a class not
covered above:

- `notebook_cell` — Jupyter notebook output / cell
- `wandb_log` — Weights & Biases run / artifact log
- `tensorboard_event` — TensorBoard event file
- `release_notes` — GitHub Releases body / CHANGELOG entry
- `slack_thread` — internal Slack discussion (rare; only if
  exported into the source set)

The `provenance_kind` lets downstream comparison weight
sources differently when adjudicating conflicts. Use the
most specific label — frivolous coining (e.g.,
`hf_card_body_first_paragraph` when `hf_card_body` fits)
fragments the analysis surface.

## Output schema

Write a single JSON object to `{{artifact_path}}` containing
TWO arrays, `operations` and `relations`. Operation IDs are
batch-local strings (e.g., `op-001`, `op-002`); they don't
need to be globally unique.

```json
{
  "batch_id": "{{batch_id}}",
  "batch_label": "<the batch label, copied from input.json>",
  "operations": [
    {
      "id": "op-001",
      "description": "OLMo-3 7B Think RLVR (Stage 3 post-training): RL with verifiable + LM-judge rewards across math, code, IF, and chat domains. Initialized from the Think-DPO checkpoint, trained on Dolci-Think-RL-7B prompts, judged by Qwen3-32B (no thinking).",
      "evidence": "Stage 3 of post-training is reinforcement learning with a mixture of verifiable and LM-judge rewards across a variety of domains.",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "id": "op-002",
      "description": "Olmo-3 7B Base pretraining (Stage 1): Dolma 3 web mix tokenized via DataMix.OLMo_mix_0625_official, with academic PDFs OCR'd by olmOCR before inclusion.",
      "evidence": "We pretrain Olmo-3-1025-7B on dolma3_mix... We use olmOCR (Poznanski et al., 2025a,b) for PDF text extraction.",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    }
  ],
  "relations": [
    {
      "operation_id": "op-001",
      "subject": "allenai/Olmo-3-7B-Think",
      "subject_in_lattice": true,
      "relation": "initialized_from",
      "direction": "DIRECT",
      "object_ref": "allenai/Olmo-3-7B-Think-DPO",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "description": "RL stage starts from the DPO checkpoint",
      "evidence": "RL stage initialized from the DPO model",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "operation_id": "op-001",
      "subject": "allenai/Olmo-3-7B-Think",
      "subject_in_lattice": true,
      "relation": "trained_on",
      "direction": "DIRECT",
      "object_ref": "allenai/Dolci-Think-RL-7B",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "description": "RL prompt mixture for the Think model (102,014 prompts spanning math, code, IF, chat)",
      "evidence": "trained_on:\n- allenai/Dolci-Think-RL-7B",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "operation_id": "op-001",
      "subject": "allenai/Olmo-3-7B-Think",
      "subject_in_lattice": true,
      "relation": "filtered_by",
      "direction": "DIRECT",
      "object_ref": "Qwen/Qwen3-32B",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "description": "Qwen3-32B (thinking off) is the LM-judge in the chat / open-ended reward; assigns a [0,1] quality score per response",
      "evidence": "Unless otherwise stated, for an LM judge we host Qwen3 32B with thinking mode turned off",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "operation_id": "op-002",
      "subject": "allenai/Olmo-3-1025-7B",
      "subject_in_lattice": true,
      "relation": "trained_on",
      "direction": "DIRECT",
      "object_ref": "allenai/dolma3_mix",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "description": "Stage-1 pretraining mixture (~6T tokens)",
      "evidence": "Pretrained on dolma3_mix",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "operation_id": "op-002",
      "subject": "allenai/dolma3_mix",
      "subject_in_lattice": true,
      "relation": "transformed_by",
      "direction": "DIRECT",
      "object_ref": "allenai/olmOCR",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "description": "Academic PDFs in dolma3_mix were OCR'd to plain text by olmOCR before tokenization",
      "evidence": "We use olmOCR (Poznanski et al., 2025a,b) to convert PDF pages",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_prose"
    },
    {
      "operation_id": null,
      "subject": "allenai/Dolci-Think-RL-7B",
      "subject_in_lattice": true,
      "relation": "size",
      "direction": "STRUCTURAL",
      "object_ref": null,
      "object_in_lattice": false,
      "object_text": null,
      "object_value": 102014,
      "object_unit": "prompts",
      "description": "total prompt count reported on the dataset card",
      "evidence": "Total Samples: 102,014",
      "source_path": "dolci-think-rl-7b.md",
      "source_line": 67,
      "provenance_kind": "hf_card_body"
    },
    {
      "operation_id": null,
      "subject": "allenai/Olmo-3-7B-Think",
      "subject_in_lattice": true,
      "relation": "used_for_evaluation",
      "direction": "INDIRECT",
      "object_ref": null,
      "object_in_lattice": false,
      "object_text": "MATH-500",
      "object_value": null,
      "object_unit": null,
      "description": "Olmo-3-7B-Think is reported on the MATH-500 benchmark in Table 4",
      "evidence": "MATH-500 78.3",
      "source_path": "olmo-3-tech-report.pdf",
      "source_line": null,
      "provenance_kind": "paper_table"
    }
  ]
}
```

### Field semantics

**Operations:**
- `id` — batch-local string (`op-001`, `op-002`, ...). Just unique within this artifact.
- `description` — lossless prose narrative of the event: what was trained, what data, what judge/filter/generator participated, key hyperparameters when present. The description is where mixture weights, learning rates, judge templates, and stage labels live.
- `evidence` — verbatim excerpt grounding the operation, ≤200 chars.
- `source_path`, `source_line`, `provenance_kind` — where the event description came from.

**Relations:**
- `operation_id` — the operation this edge belongs to, OR `null` for STRUCTURAL literals (`size`, `release_date`, etc.) and INDIRECT relations (`used_for_evaluation`, `cited_as_baseline`) that aren't tied to a training event.
- `subject` — must be a lattice `formal_name`.
- `relation` — the closed-enum classification (the planner picks which of the 8 buckets fits).
- `description` — open prose explaining **this participant's role in the operation**. Different from the operation description, which describes the whole event. This edge description is per-edge: what role did this object play?
- All other fields as before.

### Why both operation description AND edge description

The operation description tells you what happened. The edge description tells you what THIS participant contributed. They're both necessary:

- Operation: `"OLMo-3 7B Think RLVR — judge=Qwen3-32B, base=Think-DPO, data=Dolci-Think-RL-7B"`
- Edge `filtered_by`: `"Qwen3-32B (thinking off) is the LM-judge in the chat / open-ended reward; assigns a [0,1] quality score per response"`
- Edge `trained_on`: `"RL prompt mixture for the Think model (102,014 prompts spanning math, code, IF, chat)"`

Same event, three different participant roles, each captured.

`source_path` is the path *relative to* `{{batch_dir}}`.
`source_line` is best-effort (the line where the evidence
quote starts); leave it null if PDFs or other unpaginated
sources make this awkward.

`evidence` is a verbatim excerpt — at most ~200 chars, just
enough to ground the claim. Don't paraphrase. If the source
is binary (PDF), excerpt the extracted text.

`description` is open prose: the prov-system `role` style. It
carries the specifics that the closed `relation` enum can't —
mixture weights, thresholds, judge templates, stage labels.
Examples: `"60% of stage-1 pretraining mixture; FineWeb-Edu
filter at score≥3"`, `"DPO judge model with temperature 0.0"`.

## Coverage expectation

For each lattice entity that this batch's sources mention:
emit at least one relation if anything substantive is said
about it. A batch that mentions `Dolci-Think-RL-7B` 30 times
in passing but never says where it came from, what's in it,
or what model trained on it — that batch may correctly emit
zero relations for it. Quality over coverage.

A reasonable batch yields tens to a few hundred relations.
If you're heading past 1000, you're probably emitting
restatements of the same fact from different sentences; pick
the cleanest evidence and skip the rest.

## Subagent dispatch

The Task tool is available — subagents run as
`{{subagent_model}}`. If this batch has many sources (>5)
or large code repos, bucket them and dispatch one subagent
per bucket. Each subagent reads its slice and returns its
relations. Aggregate before writing.

When dispatching, transcribe verbatim into the subagent's
brief: (a) the closed relation taxonomy table above,
(b) the no-overlap matrix, (c) the provenance-kind list,
(d) the rule that subjects must be lattice `formal_name`s,
(e) the off-lattice-object channel. Subagents have none of
your context — rule erosion at dispatch is the main failure
mode.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
