# Extract Operations and Edges from this Batch's Sources

> **Goal: read this batch's source files, identify every
> training / data-construction / evaluation EVENT, and append
> ONE JSON line per event to `{{artifact_path}}`. Each event
> wraps its participating edges. Subjects are forced to lattice
> formal_names — the controllability lever.**

Read the lattice at `{{lattice_path}}` and the source files
under `{{batch_dir}}`. Append events as JSON-Lines records to
`{{artifact_path}}`.

## Append-as-you-go (load-bearing)

You are NOT producing a single giant JSON object at the end.
Instead, **after you finish identifying each event, append one
JSON object as a single line** to `{{artifact_path}}`. The file
is JSONL — one record per line, each a self-contained event.
Use:

```
cat >> {{artifact_path}} <<'EOF'
{"description": "...", "anchor_list": [...], "edges": [...]}
EOF
```

(or use the Edit tool to append; either works). Don't keep
prior events in your working memory — they're persisted on
disk. After your turn, the pipeline reads this JSONL,
validates it, and assembles the per-batch artifact.

This means: NO operation IDs to maintain across edges. Edges
live INSIDE their operation as a nested array. If you forget
op-ids you've assigned, that's fine — there are no op-ids.

## Filesystem scope

Read `{{lattice_path}}` (the post-audit lattice — every dataset
node has `subsets[]` populated) and every file under
`{{batch_dir}}` recursively (PDFs, markdown, code, configs).
Skip `__pycache__`, `node_modules`, `.git`, `venv`. Do not
read anything outside `{{batch_dir}}` except the lattice.

Web search is **off** for this stage — we want grounded claims
from the source files, not synthesized knowledge.

## What is an event (operation)

A real training or data-construction event involves multiple
participants playing different roles. The OLMo-3 7B Think DPO
event, for example, involves at least four participants:

- the resulting model (`allenai/Olmo-3-7B-Think-DPO`)
- the base checkpoint (`allenai/Olmo-3-7B-Think-SFT`)
- the chosen-completion generator (`Qwen/Qwen3-32B`, thinking)
- the rejected-completion generator (`Qwen/Qwen3-0.6B`, thinking)

Capture this as ONE event whose `description` names what
happened, with N edges nested inside (one per participant
role).

Pair-only facts (e.g., "OLMo-3 inspired_by SwallowMath
recipe") wrap as a singleton-edge event — same shape, just
one edge in the array. Uniform schema.

## Lattice anchoring — preserve source specificity

The lattice is a partial order: every family has a **family
root** (concept node, identity `{family: X}` only, no production
link) and zero-or-more **entity leaves** (full identity, with HF
/ GitHub / vendor docs link). **Vague mentions land on the
family root or on intermediate concept nodes; precise mentions
land on leaves.** Match the source's specificity exactly — don't
upgrade vague mentions to specific leaves, and don't downgrade
specific mentions to roots.

### Resolving a mention string to a lattice address

For each mention you'd put in `subject` or `object`:

1. **Try literal alias lookup.** Walk the lattice items. If the
   normalized mention (lowercase, alphanum-only) equals the
   normalized form of any item's `formal_name` or `aliases[]`
   entry, that item is the address. Done.

2. **Otherwise parse the mention into facet hints.** Common
   patterns:
   - size: `\d+(\.\d+)?(B|M)` → `size: "7B"` etc.
   - stage: `Base|Instruct|Chat|Think|SFT|DPO|RL|RL-Zero|Preview` → `stage: <token>`
   - variant: `thinking|no-thinking|FP8|AWQ|Distill` → `variant: <token>`
   - date: `\d{4}` or `\d{4}-\d{2}-\d{2}` → `date: <token>`

3. **Find the family** — match the family root whose
   `identity.family` value (or alias of that root) appears as a
   prefix or token in the mention. This is your family pivot.

4. **Build the lattice address.** Combine `{family: X}` with the
   parsed facets. Find the lattice item whose `identity` exactly
   equals that address:
   - If a leaf matches exactly → use the leaf's `formal_name`.
   - If only the family root matches (your facets are empty
     beyond `family`) → use the root's `formal_name`.
   - If your facet set lies between the root and some leaves
     (e.g., paper says "OLMo 3 Base" → `{family: OLMo 3, stage:
     Base}` — multiple Base leaves exist but no exact-match item)
     → emit a **virtual concept address** in this notation:
     ```
     <family> [<facet1>=<value1>, <facet2>=<value2>, ...]
     ```
     Examples:
     - `OLMo 3 [stage=Base]`
     - `Qwen3 [size=4B]`
     - `olmOCR [version=v1]`

   Reconcile (the next stage) merges edges across specificity
   levels via dict-subset comparison — any virtual address that
   subsumes some leaves will be folded into a leaf-anchored edge
   if other sources provided the missing facets.

5. **Off-lattice fallback** — if you can't even find a family
   pivot (e.g., a personal-namespace HF dataset that wasn't
   extracted; an internal codename; a name audit dropped), emit
   the literal source mention as a free-text string. The query
   layer distinguishes "address resolves to lattice item" vs
   "free text" at read time, no flag needed on the edge.

### Subject vs object specificity

- **Subjects** are usually leaves (the target's own checkpoints
  are pinned by `from_pretrained()` calls in configs). Emit
  leaf-level when the source pins the specific checkpoint.
- **Subjects can also be virtual concept addresses** when the
  source describes an event at the family or stage level (e.g.,
  "for the OLMo 3 Think family we use Qwen3 32B as judge" — the
  subject is `OLMo 3 [stage=Think]`, not a specific checkpoint).
  Emit one edge with the concept-level subject, OR emit one
  per-leaf edge — the global-policy-edges section below
  describes when to fan out.
- **Objects** can be leaves, family roots, virtual concept
  addresses, or free-text. Same matching rules.

If an artifact is mentioned only as a side-comment with no
relation to anything in the lattice, drop it; don't invent
relations.

## Aggregator + leaf rule (load-bearing)

Modern training pipelines structure data as **aggregator
mixes** that compose named **leaf sub-corpora**. The lattice
encodes this via the dataset node's `subsets[]` field:

```json
{
  "kind": "dataset",
  "formal_name": "allenai/dolma3_dolmino_mix-100B-1025",
  "subsets": ["cranemath", "cranecode", "finemath4plus", ...]
}
```

When the subject `trained_on` an aggregator that has populated
`subsets[]`, emit `trained_on` edges at BOTH granularities:

- One aggregator-level edge: `subject → trained_on →
  <aggregator-formal-name>`
- One leaf-level edge per subset: `subject → trained_on →
  <aggregator-formal-name>/<subset-slug>` for each entry in
  the parent's `subsets[]`

The leaf-level edges look redundant with the aggregator-level
one but they capture the dependency at the granularity
reference graphs use. Do NOT drop them.

If the source explicitly says only some sub-corpora were used
(e.g., "Olmo-3-7B-Base used CraneMath and FineMath4+ but not
the OMR-Rewrite subset of dolmino"), emit leaf edges only for
the ones the source names.

## Global-policy edges

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
variants. Each lands in its own event (no shared op).

Same pattern for global generators: "we distill thinking
traces from GPT-4.1 and o4-mini" → emit `generated_by` edges
to both objects (off-lattice via free-text `object` if the
lattice doesn't have them) for every model whose midtraining
or post-training mixture incorporated those traces.

A global statement that you DO NOT expand to per-stage edges
will manifest downstream as missing license-inheritance and
provenance edges. Err on the side of expanding.

## Relation taxonomy (canonical labels — coining allowed when none fits)

`relation` is an open string, but the canonical labels below
cover the vast majority of LLM-pipeline events. The 5 DIRECT
and 3 INDIRECT buckets are designed to be orthogonal — every
direct dependency that fits one of them lands in exactly one.

**Use a canonical label when one fits.** Only coin a new
snake_case label when the source describes an event that none
of the canonical values capture. Examples of legitimate
coining: `merged_from` (model souping), `deduplicated_by`,
`embedded_by`, `tokenized_by`, `decontaminated_by`.

**Do NOT coin** when the canonical label fits. The planner
that emits `cited_as_baseline` instead of skipping the edge
entirely (baseline comparisons are out-of-scope, see below)
is bad reasoning, not bad coining.

When you coin, the edge `description` MUST explain the role
clearly enough that a downstream reader understands what the
new label means without re-reading the source.

### DIRECT relations (`dependency_kind: "direct"` — object enters training)

| relation | typical direction | when to use |
|---|---|---|
| `trained_on` | M → D | the dataset is pretraining or post-training data for the model |
| `trained_from` | M → M | weight initialization (subject's parameters copied / continued from object) |
| `generated_by` | M → M | object model generated content used as subject's training data (synthetic data, distillation traces, RL rollouts) |
| `transformed_by` | M → M / M → D | object rewrote / OCR'd / reformatted content used in training (content originated elsewhere; object only modified) |
| `filtered_by` | M → M | object decided inclusion (judge in pairwise preference, quality classifier, RM-as-filter) — content unchanged |

### INDIRECT relations (`dependency_kind: "indirect"` — object shaped subject without entering training)

| relation | typical direction | when to use |
|---|---|---|
| `inspired_by` | M → M / M → D | methodology borrowed; recipe explicitly cited; no weight or data inheritance |
| `used_for_ablation` | M → M / M → D | object was a design-space variant in the subject team's OWN ablation studies (not a baseline they compared against) |
| `used_for_evaluation` | M → M / M → D | benchmark / eval set OR LLM judge model — used to evaluate the release |

### Out-of-scope — do NOT emit

- **Baseline comparisons** ("we report scores against Llama-3
  in Table 7", "our model outperforms GPT-4 on GSM8K"). These
  are lateral comparisons, not provenance. Emit nothing — no
  node, no edge. The test: was this artifact part of the
  team's OWN development pipeline (ablation, eval, methodology
  borrowing), or just a published number to compare against?
  Only the former is in scope.
- **Generic architecture / algorithm** (Transformer, RoPE,
  RMSNorm, AdamW, MoE, GQA, SwiGLU). Never edges.
- **Tokenizers, frameworks, infrastructure** (PyTorch, vLLM,
  Transformers, datatrove, tiktoken). Never edges.
- **Vague inspiration** ("inspired by the broader RL
  literature", "following common practice"). Never edges.
- **Numeric facts about a node** (size, training_tokens,
  release_date, parameter_count, context_length). These are
  NOT edges. They live in node descriptions. The lattice
  validator will reject them as relations.

### No STRUCTURAL bucket

Subset / contains / supersedes / released_with relationships
are NOT emitted as edges. Their information lives in:
- The dataset node's `subsets[]` field (composition).
- The aggregator+leaf rule (subject emits leaf-level edges).
- Node descriptions (release ordering, supersession,
  bundling).

This avoids dataset-as-subject edges and matches the way
reference graphs encode lineage.

## Why no overlap among the 5 DIRECT buckets

Two orthogonal axes separate them:

- **Content origin** — where did the training content come from?
  - external data → `trained_on`, `transformed_by`, `filtered_by`
  - external weights → `trained_from`
  - the object model itself → `generated_by`

- **What the object did to the content**:
  - the object IS the content (weights or generated text) →
    `trained_from`, `generated_by`, `trained_on`
  - the object **rewrote** existing content → `transformed_by`
  - the object **only decided inclusion** → `filtered_by`

```
                  | content from object | content from elsewhere, unchanged | content from elsewhere, modified
weights           | trained_from        | —                                  | —
content (data)    | generated_by        | trained_on (data) /                | transformed_by
                  |                     | filtered_by (decision-only model)  | (rewriter model)
```

Every M-to-M-or-M-to-D direct dependency lands in exactly one
cell. If you can't decide between two cells, the edge probably
needs to be split into two edges along different axes — emit
both.

### Worked examples

- `allenai/Olmo-3-7B-Instruct` is initialized from
  `allenai/Olmo-3-7B-Base` → `trained_from`.
- `allenai/Olmo-3-7B-Base` was pretrained on
  `allenai/dolma3_mix-6T-1025-7B` → `trained_on` (aggregator);
  also emit per-leaf `trained_on` edges for every subset slug
  in that aggregator's `subsets[]`.
- A paragraph in Dolma-3 was OCR'd from a PDF by
  `allenai/olmOCR-7B-0225`. Subject is the OLMo-3 model that
  trained on the OCR'd content; object is the OCR model. →
  `transformed_by`. Content came from the PDF, not from
  olmOCR.
- An RLHF DPO run for `allenai/Olmo-3-7B-Instruct-DPO` used
  `Qwen/Qwen3-32B` as the judge that picked preferred
  responses. → `filtered_by` (judge decided inclusion; it
  didn't generate the responses, it ranked them).
- `CraneMath` was generated by `Qwen/Qwen3-32B` rewriting
  math problems. From the model that trained on the parent
  Dolmino mix: `generated_by` Qwen3-32B (the content was
  produced by Qwen3-32B). The CraneMath dataset itself shows
  up as a leaf-level `trained_on` edge per the
  aggregator+leaf rule.

## Edge schema (nested inside an event)

```json
{
  "subject":         "<lattice formal_name, verbatim>",
  "relation":        "trained_on",
  "dependency_kind": "direct",
  "object":          "<lattice formal_name OR free-text descriptor>",
  "description":     "<lossless 1-3 sentences, ≤ ~500 chars>",
  "anchor_list":     [
    {
      "source":      "<URL or local path>",
      "position":    "<locator within source: section, page, table, line range, YAML field>",
      "explanation": "<how the cited source supports this edge; verbatim quote inline>"
    }
  ]
}
```

- `subject`: MUST be one of:
  - a leaf `formal_name` (full-identity item),
  - a family-root `formal_name` (item with identity `{family: X}`),
  - a virtual concept address `<family> [<k>=<v>, ...]` notation
    when the source's specificity falls between root and leaf.
- `relation`: canonical from the table above when one fits;
  otherwise a coined snake_case label.
- `dependency_kind`: `"direct"` or `"indirect"`. Closed
  vocabulary; mismatch with the relation's bucket is a
  validation error.
- `object`: same shape as subject (leaf / root / virtual concept
  address) OR a free-text descriptor when no family pivot exists.
- `description`: lossless prose. MUST capture every
  structurally relevant fact that the relation, subject,
  object, and event description don't already express:
  training stage (sft/dpo/rl/midtraining/long_context),
  role sub-variants (Think-SFT vs Instruct-SFT, math vs
  code), quantities (prompt counts, token counts), specific
  subsets / filters, ordering / compositional context,
  caveats. ≤ ~500 chars.
- `anchor_list`: NON-EMPTY array. Each entry has REQUIRED
  `source` and `explanation`; RECOMMENDED `position` (the
  verifier uses position to navigate; absence triggers
  `external_support_only` downstream).

## Event schema (one JSONL line)

```json
{
  "description": "<lossless prose describing the event itself: what happened, who participated in what role, in which training stage>",
  "anchor_list": [
    {"source": "...", "position": "...", "explanation": "..."}
  ],
  "edges": [
    { ... edge 1 ... },
    { ... edge 2 ... },
    { ... edge 3 ... }
  ]
}
```

- `description`: event-level prose. Distinct from per-edge
  descriptions: this captures the EVENT (the training run, the
  dataset construction, the eval pass), while edges describe
  individual participant roles.
- `anchor_list`: same shape as edge anchors; supports the
  event-level claim. Often the same primary source as
  per-edge anchors but can include event-wide citations
  (e.g., a recipe section that lists all participants).
- `edges`: NON-EMPTY array. Each edge follows the schema above.

## Worked example — full event

```json
{
  "description": "OLMo-3-7B Think DPO post-training event: continued from Olmo-3-7B-Think-SFT, with Qwen3-32B (thinking mode) generating chosen-completion candidates and Qwen3-0.6B (thinking mode) generating rejected candidates per the Delta-Learning recipe (Section 4.3.1).",
  "anchor_list": [
    {
      "source": "https://arxiv.org/abs/2512.13961",
      "position": "Section 4.3.1, paragraph beginning 'For Think-DPO'",
      "explanation": "Paper describes the Think-DPO event with named generators."
    }
  ],
  "edges": [
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "trained_from",
      "dependency_kind": "direct",
      "object": "allenai/Olmo-3-7B-Think-SFT",
      "description": "Olmo-3-7B-Think-DPO is initialized from the Think-SFT checkpoint and continues training with DPO preference optimization.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1",
          "explanation": "States that Think-DPO continues from Think-SFT."
        }
      ]
    },
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "generated_by",
      "dependency_kind": "direct",
      "object": "Qwen/Qwen3-32B",
      "description": "Qwen3-32B (thinking mode on) generated chosen-completion candidates for the Think-DPO preference pairs per the Delta-Learning recipe.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1, Delta-Learning paragraph",
          "explanation": "Paper says 'Qwen3 32B thinking generates chosen completions'."
        }
      ]
    },
    {
      "subject": "allenai/Olmo-3-7B-Think-DPO",
      "relation": "generated_by",
      "dependency_kind": "direct",
      "object": "Qwen/Qwen3-0.6B",
      "description": "Qwen3-0.6B (thinking mode on) generated rejected-completion candidates for the Think-DPO preference pairs per the Delta-Learning recipe.",
      "anchor_list": [
        {
          "source": "https://arxiv.org/abs/2512.13961",
          "position": "Section 4.3.1, Delta-Learning paragraph",
          "explanation": "Paper says 'Qwen3 0.6B thinking generates rejected completions'."
        }
      ]
    }
  ]
}
```

That's ONE line in `{{artifact_path}}`. The next event you
identify is the next line. No cross-event references.

## Coverage expectation

There is no fixed target count. The number of events depends
on how richly the source describes the pipeline. Use these
qualitative checks:

- For every training stage the source names (pretraining,
  midtraining, long-context, SFT, DPO, RL, RL-Zero), there
  should be at least one event covering it.
- For every aggregator mix the subject `trained_on`, you
  should have leaf-level edges for every entry in the
  parent's `subsets[]` (per the aggregator+leaf rule).
- For every named generator / judge / rewriter / classifier
  the source mentions, there should be at least one
  `generated_by` / `filtered_by` / `transformed_by` edge.
- Every benchmark / eval-judge in the release's eval section
  becomes a `used_for_evaluation` indirect edge.
- The team's own ablation tables become `used_for_ablation`
  indirect edges (only their own design-space variants — not
  external baselines they compared against).

If you only emit `trained_on` and `trained_from` edges with
no `generated_by` / `transformed_by` / `filtered_by` / eval
edges, you've under-covered.

## Subagent dispatch

Subagents run as `{{subagent_model}}`. Bucket the source files
into topical packets (one packet per training stage, per
dataset family, per eval section) and dispatch one subagent per
packet. Each subagent appends its events directly to
`{{artifact_path}}` as it processes.

When dispatching, transcribe verbatim into each subagent's brief:
- The "Append-as-you-go" instruction (file path + how to append).
- The "Lattice anchoring" rule.
- The "Aggregator + leaf rule".
- The "Out-of-scope" list (especially baseline comparisons).
- The relation taxonomy table.
- The schemas (edge + event).

Subagents have none of your context. Without these
transcriptions they will silently revert to old patterns
(emitting baseline comparisons, missing leaf edges, fabricating
op-ids).

## Completion

When you've processed all sources and appended all events to
`{{artifact_path}}`, exit 0. Do NOT emit a final wrapping JSON
object — the file is JSONL, one event per line, and the
pipeline reads it as such.

You are running as `{{planner_model}}`. Subagents run as
`{{subagent_model}}`.

{{subagent_prompt}}
