# Extract Relations Anchored to the Lattice

> **Goal: read this batch's source files, emit every relation
> a lattice entity participates in. Subjects are forced to be
> lattice `formal_name`s — that's the controllability lever.**

Read the lattice at `{{lattice_path}}` and the source files
under `{{batch_dir}}`. Write the relation list to
`{{artifact_path}}`.

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
personal-namespace HF dataset, a one-off internal codename),
you have two valid moves:

1. If it appears as the **object** of a relation whose subject
   is in the lattice — emit the relation with `object_ref:
   null`, `object_text: "<the literal string from source>"`,
   `object_in_lattice: false`. Off-lattice objects are
   themselves a finding signal.
2. If it appears alone (no relation to any lattice entity)
   — drop it. Don't invent relations.

`subject_in_lattice` is always `true`. We don't extract
relations between two off-lattice entities; both endpoints
need to anchor to canonical names for the system to compare
across batches.

## Relation vocabulary (closed)

Use these strings verbatim. Don't invent new relations.

**Entity → entity** (`object_ref` is a lattice `formal_name`):

| relation | subject | object | what it captures |
|---|---|---|---|
| `trained_on` | model | dataset | training data used |
| `derived_from` | model | model | base / parent checkpoint |
| `evaluates_on` | model | dataset | benchmark evaluated on |
| `cited_as_baseline` | model | model | comparison model in eval table |
| `subset_of` | dataset | dataset | this is a subset/filtered copy of that |
| `supersedes` | model/dataset | model/dataset | replaces a predecessor |
| `judged_by` | model | model | LLM judge in RL / eval |
| `generated_by` | dataset | model | synthetic data generator |
| `released_with` | model | model/dataset | tokenizer or companion artifact |
| `contains` | dataset | dataset | this dataset bundles that one |

**Entity → literal value** (`object_value` plus `object_unit`):

| relation | example value/unit | what it captures |
|---|---|---|
| `size` | 102014 / "prompts" | data size or model parameter count |
| `training_tokens` | 5.93e12 / "tokens" | total tokens model was trained on |
| `context_length` | 65536 / "tokens" | max context window |
| `release_date` | "2025-10-25" / "iso" | release date |
| `parameter_count` | 7e9 / "params" | model parameter count |
| `composition_count` | 29813 / "prompts" | size of one named subsource (use with `composition_source` text) |

For `composition_count`, `object_text` should be the
sub-source name (e.g., `"IF Multi-Constraint"`) and
`object_value` its count. This is how Table-20-style
breakdowns become structured triples.

If you observe a fact you want to capture that doesn't fit
any of the above, prefer not emitting over inventing a new
relation. The closed vocabulary is the point.

## Provenance kind (closed)

Tag every relation with where exactly it came from. Pick
the most specific:

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

The `provenance_kind` lets downstream comparison weight
sources differently when adjudicating conflicts.

## Output schema

Write a single JSON object to `{{artifact_path}}`:

```json
{
  "batch_id": "{{batch_id}}",
  "batch_label": "<the batch label, copied from input.json>",
  "relations": [
    {
      "subject": "allenai/Olmo-3-7B-Instruct-DPO",
      "subject_in_lattice": true,
      "relation": "trained_on",
      "object_ref": "allenai/Dolci-Think-DPO-7B",
      "object_in_lattice": true,
      "object_text": null,
      "object_value": null,
      "object_unit": null,
      "evidence": "datasets:\n- allenai/Dolci-Think-DPO-7B",
      "source_path": "olmo-3-7b-instruct-dpo.md",
      "source_line": 8,
      "provenance_kind": "hf_frontmatter"
    },
    {
      "subject": "allenai/Dolci-Think-RL-7B",
      "subject_in_lattice": true,
      "relation": "size",
      "object_ref": null,
      "object_in_lattice": false,
      "object_text": null,
      "object_value": 102014,
      "object_unit": "prompts",
      "evidence": "Total Samples: 102,014",
      "source_path": "dolci-think-rl-7b.md",
      "source_line": 67,
      "provenance_kind": "hf_card_body"
    },
    {
      "subject": "allenai/Olmo-3-7B-Instruct",
      "subject_in_lattice": true,
      "relation": "trained_on",
      "object_ref": null,
      "object_in_lattice": false,
      "object_text": "hamishivi/rlvr_general_mix",
      "object_value": null,
      "object_unit": null,
      "evidence": "--mixer_list ... hamishivi/rlvr_general_mix 13314",
      "source_path": "scripts/train/olmo3/7b_instruct_rl.sh",
      "source_line": 47,
      "provenance_kind": "script_flag"
    },
    {
      "subject": "allenai/Dolci-Think-RL-7B",
      "subject_in_lattice": true,
      "relation": "composition_count",
      "object_ref": null,
      "object_in_lattice": false,
      "object_text": "KlearReasoner-Code",
      "object_value": 6272,
      "object_unit": "prompts",
      "evidence": "| KlearReasoner Code | 6,272 |",
      "source_path": "dolci-think-rl-7b.md",
      "source_line": 84,
      "provenance_kind": "hf_card_body"
    }
  ]
}
```

`source_path` is the path *relative to* `{{batch_dir}}`.
`source_line` is best-effort (the line where the evidence
quote starts); leave it null if PDFs or other unpaginated
sources make this awkward.

`evidence` is a verbatim excerpt — at most ~200 chars, just
enough to ground the claim. Don't paraphrase. If the source
is binary (PDF), excerpt the extracted text.

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
brief: (a) the relation vocabulary table above, (b) the
provenance-kind list, (c) the rule that subjects must be
lattice `formal_name`s, (d) the off-lattice-object channel.
Subagents have none of your context — rule erosion at
dispatch is the main failure mode.

## Completion

Write the artifact to `{{artifact_path}}` and exit 0.

You are running as `{{planner_model}}`. Subagents (when
dispatched) run as `{{subagent_model}}`.

{{subagent_prompt}}
