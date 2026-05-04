# Baseline prompt vs. our pipeline — rule-by-rule comparison

This document compares the **single-shot baseline prompt** (the
zero-shot policy competing systems will receive in the paper's
evaluation) against the rules our **multi-stage pipeline** enforces
across discover → extract → organize → audit → relate → triage →
merge.

The eval is LLM-as-judge throughout — an alignment judge matches
edges across systems by `subject + object + description + evidence`.
Wherever the two policies disagree, our system's edges either
fail to match (lost recall in headline metrics) or get a
`bucket_concern` / `description_concern` annotation. Sections 2
and 5 below highlight the conflicts that matter most for the
comparison's fairness.

---

## 1. What is the same

Both policies share the core ontology and most of the structural
rules. An LLM aligner reading both will see the same picture for
the majority of edges.

| Concept | Baseline | Ours | Match |
|---|---|---|---|
| **Two node types** | `model` and `dataset` only | `kind: model \| dataset` (extract → audit) | exact |
| **Direct vs indirect taxonomy** | `dependency_kind: direct/indirect` is the load-bearing axis | `direction: DIRECT/INDIRECT/STRUCTURAL` covers both + structural lineage | mostly — see §2 for STRUCTURAL conflict |
| **Indirect categories** | methodology (`inspired_by`), ablations/baselines (`used_for_ablation`), evaluation (`used_for_evaluation`) | same three labels in our `relate.md` INDIRECT taxonomy | exact |
| **Out-of-scope artifacts** | tokenizers, frameworks, infra, hardware, compute, licenses, generic architecture (Transformer, RoPE, RMSNorm, AdamW, GELU, SwiGLU, MoE, GQA) | same skip-list in `extract.md` ("license names, pure-software packages, frameworks, tokenizers as such") | exact |
| **Direct relation labels (most)** | `trained_on`, `transformed_by`, `filtered_by`, `generated_by` | `trained_on`, `transformed_by`, `filtered_by`, `distilled_from` | partial — `generated_by` ↔ `distilled_from` is a **rename conflict** (§2.4) |
| **Indirect relation labels** | `inspired_by`, `used_for_ablation`, `used_for_evaluation` | identical | exact |
| **Authoritative releases only** | community re-uploads (`TheBloke/Llama-3-70B-GGUF`) NOT nodes; canonical = owning org | `organize.md` 3-form criterion: HF identifier of owning org, vendor URL, paper anchor; community mirrors fold into `aliases` | exact |
| **HF canonical ID format** | lowercased `owner/repo` for HF artifacts | `formal_name = <org>/<repo>` from HF | partial — **case conflict** (§2.3) |
| **Aliases** | populate generously: HF original-case, paper attribs, dated snapshots, short forms | every input surface form goes into `aliases[]` | exact |
| **Evidence: verbatim quote ≤200 chars** | `excerpt` recommended ≤200 chars | `evidence: <verbatim ≤200 chars>` field in relate edges | exact |
| **Description as lossless prose** | `description` MUST capture every structurally relevant fact not already encoded in (subject, object, relation_type) | per-edge `description` field, 5–15 word role label | partial — **length differs** (§2.10) |
| **Atomicity** | one edge per `(subject, object, relation_type)` triple | same | exact |
| **Subject is always a Model** | hard validator rule | most edges have model subjects; STRUCTURAL `subset_of`/`contains` allow dataset subjects (§2.6 conflict) | partial |
| **Facets / identity decomposition** | `facets` object: `family`, `size`, `stage`, `date`, `modality`, `source` | `identity` object with `org`, `collection`, `size`, `stage`, `date`, `quantization`, `subset`, `harness`, `variant`, `vendor` | exact concept; vocab overlaps |
| **Family-concept handling** | family-concept node valid iff has HF collection URL or own paper | identical rule in `organize.md` "Family-concept exception" | exact |
| **No invented nodes** | every canonical_id must resolve | `organize.md` 3-form criterion drops unresolvable | exact |
| **Multiple evidence entries** | `evidence` is an array | we currently emit single `evidence` string per edge (§4.16 — gap) | conflict-by-omission |
| **Mention without concrete edge ≠ node** | "node-worthy iff at least one concrete edge" | implicit (relate.md only emits subjects from the lattice; lattice items without edges remain) | partial |

---

## 2. What is different (CONFLICT — affects alignment)

These are the rule disagreements that will produce different
outputs for the same source material, and so will be visible to
the alignment judge / verifier.

### 2.1 Architecture: single-shot vs. multi-stage

| | Baseline | Ours |
|---|---|---|
| Topology | one prompt, one agent, web access | 6 stages, per-batch fanout, audit-then-relate |
| Output unit | one JSON `{subject, nodes, edges}` per investigation | one organize artifact + N per-batch relate artifacts + merge |
| Recursion | BFS baked into one prompt run, controlled by `max_depth` | each new node is a separate `expand` pipeline invocation |

**Implication:** the baseline expects everything in one document;
our `merge` stage collapses N artifacts into one. The schema we
hand the alignment judge MUST match `{subject, nodes, edges}`
even though our internal artifacts don't have that shape.

### 2.2 Top-level schema

| | Baseline | Ours |
|---|---|---|
| top-level | `{subject, nodes[], edges[]}` | `{batch_id, batch_label, operations[], relations[]}` per batch + lattice `{groups: [{family, identity_keys, items}]}` |
| `subject` field | REQUIRED top-level: the investigated model's `canonical_id` | implicit (the run target); not surfaced in any artifact |
| `nodes` array | flat list with required `node_type, canonical_id, name, provider`, optional `facets, aliases` | nested in `groups[].items[]`; required `kind, formal_name, identity, aliases, links, description` |
| `edges` array | flat list with required `edge_id, subject, object, relation_type, dependency_kind, description, evidence[]` | per-batch `relations[]` with `operation_id, subject, subject_in_lattice, relation, direction, object_ref, object_text, object_in_lattice, description, evidence, source_path, source_line, provenance_kind` |

**Implication:** we need an export adapter to transform our
artifact into the baseline schema for the alignment judge.

### 2.3 canonical_id case

- **Baseline:** lowercase EVERYTHING — `qwen/qwen3-32b`,
  `meta-llama/llama-3.1-70b-instruct`,
  `huggingfacetb/finemath`. Original-case form goes in `aliases`.
- **Ours:** `formal_name = Qwen/Qwen3-32B` (preserve HF
  original case). Lowercased variants go in `aliases`.

**Implication:** the alignment judge uses canonical_ids as the
primary signal and aliases as fallback; the case mismatch should
be recoverable but introduces noise. **For the baseline run we
should normalize our exports to lowercase**, or update the
baseline prompt to allow either case.

### 2.4 `generated_by` vs `distilled_from`

- **Baseline:** uses **`generated_by`** for "object model
  generated content that became subject's training data
  (distillation traces, synthetic data, rollouts)".
- **Ours:** uses **`distilled_from`** for the same role. We
  reserve `generated_by` (not in our taxonomy) for nothing — we
  collapse it.

**Implication:** every distillation/synthesis edge will
mismatch on label. The alignment judge is supposed to ignore
label and match by `subject + object + description`, but the
verifier records `bucket_concern` when labels disagree.
**Recommend renaming our `distilled_from` → `generated_by`** to
align (or adding `generated_by` as canonical alias).

### 2.5 `trained_from` vs `initialized_from`

- **Baseline:** **`trained_from`** = "subject's weights
  initialized from object's checkpoint" (continual pretraining,
  base→post-train chain).
- **Ours:** **`initialized_from`** = same role. `trained_from`
  is not in our taxonomy.

**Implication:** another rename-only conflict. Same
recommendation — add `trained_from` as alias or rename.

### 2.6 STRUCTURAL category — we have it, baseline doesn't

- **Baseline:** edges are EITHER `direct` OR `indirect`. Period.
  Dataset composition (mix → leaf) is captured indirectly via
  the AGGREGATOR rule (§AGGREGATOR): emit `subject (Model) →
  trained_on → leaf-dataset` for each named sub-corpus, NOT a
  `leaf → subset_of → mix` edge.
- **Ours:** STRUCTURAL bucket holds `subset_of`, `supersedes`,
  `released_with`, `contains` plus numeric properties (size,
  training_tokens, …). Subjects can be datasets here.

**Implication:** all our STRUCTURAL edges with dataset subjects
will be **invalid in the baseline schema**. The aligner will
either drop them or try to interpret them, costing matches.
**Recommend:**
1. Reframe `subset_of` and `contains` so the SUBJECT is the
   investigated model (`Olmo-3-7B-Base trained_on FineMath4+`
   PLUS `Olmo-3-7B-Base trained_on FineMath` — both at the
   model granularity, per the AGGREGATOR rule). Drop the
   dataset→dataset structural edge.
2. Move numeric properties (size, training_tokens, etc.)
   off edges and into node attributes — already in the planned
   relate redesign.

### 2.7 Aggregator + leaf dual edges (HARD RULE in baseline)

- **Baseline:** when an aggregator mix names sub-corpora,
  emit BOTH:
  - `subject → trained_on → mix` (aggregator-level)
  - `subject → trained_on → leaf₁`, `→ leaf₂`, … (one per
    named sub-corpus)
  This **looks redundant** but is required — the leaf-level
  edges are what the grader uses.
- **Ours:** typically emits `subject → trained_on → mix`
  PLUS `leaf → subset_of → mix` (STRUCTURAL). Does NOT
  emit `subject → trained_on → leaf`.

**Implication:** We will appear to **under-cover real
training-data dependencies** by ~10–50× depending on the mix's
fanout (Dolmino, Tulu 3 SFT, Dolci-Think-SFT, etc. each have
dozens of leaves). This is the **single biggest recall gap**
introduced by the schema mismatch.
**Recommend:** add a relate-stage rule that, for every
aggregator mix the subject `trained_on`, emit additional
`trained_on` edges to each known leaf in the lattice.

### 2.8 `relation_type` enforcement

- **Baseline:** `relation_type` is a **free-form short label**.
  Suggested vocabulary in §7 but **NOT enforced, NOT scored**.
  Picking a wrong bucket only triggers a `bucket_concern`
  diagnostic, not a refute.
- **Ours:** `relate.md` has a **closed taxonomy** (5 DIRECT, 3
  INDIRECT, plus STRUCTURAL); validator rejects unknown labels;
  coining `cited_as_baseline` is explicitly forbidden in our
  prompt.

**Implication:** our system is more conservative — every edge
that doesn't fit our 8-label set either gets coined (we allow
careful coining) or pushed into the closest fit. The baseline
allows free labels, so the baseline's outputs will have
diverse `relation_type` strings. The alignment judge handles
this correctly (label is metadata), but our `bucket_concern`
rate will be lower than the baseline's.

### 2.9 `dependency_kind` field

- **Baseline:** REQUIRED on every edge. Closed vocab:
  `"direct"` or `"indirect"`. Hard validator error if missing.
- **Ours:** field is named `direction`, vocab is `DIRECT |
  INDIRECT | STRUCTURAL`. Required.

**Implication:** rename for export (`direction` → `dependency_kind`)
and drop or remap STRUCTURAL.

### 2.10 `description` length / role

- **Baseline:** soft cap **~500 chars**; lossless; captures stage,
  role sub-variants, quantities, subsets, ordering, caveats.
- **Ours:** "5–15 word role" — much shorter. Sub-type prefixes
  (`benchmark:` / `comparison_baseline:` / `ablation_target:`)
  carry classification; numeric facts move to `node_attributes`
  in the planned redesign.

**Implication:** our descriptions are too terse for the
baseline's lossless-prose grading. The verifier reads
`description + evidence`, and a 5–15 word description loses
detail that the baseline expects. **Recommend:** loosen our
description cap to ~500 chars and require role/stage/quantity
detail (matching baseline §6.3).

### 2.11 Operations — we have, baseline doesn't

- **Baseline:** no notion of multi-participant operations.
  All edges are pairwise.
- **Ours:** `operations[]` array groups multiple edges sharing
  one training event (`Olmo-3-7B-Think RLVR run` with 4
  participants).

**Implication:** when exporting to baseline format, we must
flatten — drop `operation_id`, drop `operations[]`. We lose the
multi-participant grouping in the baseline view. Downstream
operation-aware queries are our system's value-add but not
visible to the baseline grader.

### 2.12 Evidence schema

| Field | Baseline | Ours |
|---|---|---|
| `source` (URL) | REQUIRED | `source_path` REQUIRED |
| `explanation` (free prose) | REQUIRED | not present (we have edge-level `description` instead) |
| `location` (section/page/line) | RECOMMENDED warning | `source_line` (int, line number) |
| `excerpt` (verbatim ≤200 chars) | RECOMMENDED warning | `evidence` (verbatim ≤200 chars) REQUIRED |
| Multiple entries | `evidence` is array; supports transitive grounding | single per edge |
| Transitive grounding (§8.5) | SHOULD: at least one entry cites subject's own docs | not specified |

**Implication:** our evidence is similar in spirit but the
baseline's array shape supports stacked sources for one edge,
which we don't. Edges that span paper + GitHub + HF card will
need either multiple edges (we already do this for atomicity)
or a schema upgrade to array.

### 2.13 `provider` and `name` fields on nodes

- **Baseline:** every node REQUIRES `node_type, canonical_id,
  name, provider`. `name` is a display name; `provider` is the
  owning-org slug.
- **Ours:** nodes have `kind, formal_name, identity{org, ...},
  aliases, links, description`. No separate `name` (we use
  `formal_name`); no separate `provider` (we have `identity.org`
  inside the identity dict).

**Implication:** export adapter must derive `name` and `provider`
from our existing fields (`name` from the most-canonical alias
or the formal_name post-org-slash; `provider` from `identity.org`
or the part before the slash in `formal_name`).

### 2.14 `edge_id`

- **Baseline:** every edge has a unique `edge_id` (short readable
  slug like `e_olmo3_qwen3_32b_gen_01`).
- **Ours:** no edge_id field. Edges are identified by index in
  the relations array.

**Implication:** export adapter must synthesize edge IDs.

### 2.15 Date-snapshot canonical_id rule

- **Baseline:** drop dated snapshots from canonical_id.
  `openai/gpt-4.1` is canonical, `gpt-4.1-2025-04-14` goes in
  `aliases`. Version periods preserved.
- **Ours:** `organize.md` example shows
  `OpenAI/gpt-4o-mini-2024-07-18` as a synthetic canonical
  (date in the canonical). But our `audit.md` fold rule **does
  fold date-snapshots** into the family canonical with `date`
  in identity_keys.

**Implication:** the two prompts in our pipeline are
**internally inconsistent** here. Organize emits dated
snapshots as canonical; audit folds them. Audit's fold should
win, so post-audit the canonical is undated — but if audit
isn't perfect, residual dated canonicals will exist.
**Recommend:** update `organize.md`'s API-only example to use
the undated canonical (`OpenAI/gpt-4o-mini`) so the fold and
the example agree.

### 2.16 Subject in relate edges

- **Baseline:** subject is an arbitrary canonical_id of any
  node in `nodes[]`. Validator checks the resolved node has
  `node_type == "model"`.
- **Ours:** subject MUST be a `formal_name` taken **verbatim
  from the lattice** (the audit-emitted lattice). This is
  stricter — we won't emit edges whose subject is a name we
  couldn't resolve to a lattice item.

**Implication:** more conservative; lower fabrication rate but
also lower coverage on borderline cases.

### 2.17 Link kinds — we have, baseline doesn't structure

- **Baseline:** `evidence.source` is just a URL string. No
  classification of what KIND of source.
- **Ours:** `links[].kind` per node uses closed vocab
  (`hf_model`, `hf_dataset`, `hf_collection`, `github`,
  `paper`, `vendor_docs`, `blog`, `hf_dataset_config`).
  `provenance_kind` per edge classifies evidence source type
  (hf_card_body, hf_frontmatter, paper_prose, github_yaml, …).

**Implication:** our richer source typing isn't visible to the
baseline. Lossless if we preserve as alias-style metadata in the
description; otherwise dropped.

---

## 3. What we have mentioned but the baseline does NOT

These are the dimensions where our system goes beyond the
baseline policy. Some are net-positive (reduce noise, surface
conflicts); some create export friction.

### 3.1 Lattice and identity decomposition (organize stage)
1. **3-form node criterion (DROP-IF-UNRESOLVABLE)** — every node
   MUST resolve to (a) HF model identifier, (b) closed-source
   model with vendor URL, or (c) dataset with HF identifier or
   paper anchor. Otherwise dropped.
2. **HEAD-verification of URLs** — every link `curl`-checked
   for 200 before adding.
3. **Disambiguating-facets rule** — within a family, every
   item's identity dict MUST distinguish from siblings.
4. **Subset notes in description** — when a canonical form is a
   subset/config of a parent, mention that explicitly in
   `description`.

### 3.2 Audit pass
5. **Fuzzy-match pass with Jaccard ≥ 0.7** — tokenize formal_names
   on `/`, `::`, `_`, `-`, `.`, ` `; compute Jaccard pairwise;
   flag merge candidates. Algorithmic.
6. **Three explicit fold rules** — date-snapshot variants,
   eval-harness reformulations (`BBH::cot::hamish_zs_reasoning`
   → `BBH`), bare-lowercase informal aliases.

### 3.3 Operations and structural edges (relate stage)
7. **Operations as multi-participant event groupings** —
   `operations[]` with `description, evidence, provenance_kind,
   confidence`; edges share `operation_id`.
8. **Closed `relation` taxonomy with validator enforcement** —
   8 canonical labels, careful coining allowed, validator
   rejects malformed labels.
9. **STRUCTURAL relation category** — `subset_of`, `supersedes`,
   `released_with`, `contains` (currently with dataset
   subjects, conflicting with baseline).
10. **`provenance_kind` taxonomy** —
    `hf_card_body`, `hf_frontmatter`, `paper_prose`, `github_yaml`,
    `github_python`, `github_shell`, `release_blog`, …
    Per-edge classification of evidence source type.
11. **Description sub-type prefix** for `used_for_evaluation`
    edges: `benchmark:`, `comparison_baseline:`, `ablation_target:`,
    `leaderboard_reference:`. (Planned; not yet in the prompt.)

### 3.4 Off-lattice + reconciliation features
12. **`object_text` channel** — entities not in lattice still get
    edges; subject must be in lattice but object can be off-lattice.
13. **`subject_in_lattice` / `object_in_lattice` flags** —
    explicit handling of off-lattice anchors.
14. **`global_policy_expansion`** — single edge with expansion
    list for "default" policy edges (e.g., "we use Qwen3-32B
    as judge for all RLVR runs" → one edge with expansion list,
    not 10 duplicates). (Planned redesign.)
15. **`corroborating_sources`** — multiple sources stacked under
    one edge (paper §3.3 corroboration mode). (Planned redesign.)
16. **`conflicts_with`** — surface conflicting claims (paper
    §3.3 conflict mode). (Planned redesign.)
17. **`ambiguous_resolution`** — flag when lattice resolution is
    ambiguous between multiple candidates. (Planned redesign.)
18. **Confidence per edge / operation** with calibration guidance
    and review.json sidecar routing. (Planned redesign.)

### 3.5 Source-position drops in extract
19. **Bibliography-only refusal** — names appearing only in
    References do not produce nodes.
20. **Comparison-baseline-only refusal** — names appearing only
    as bare table rows do not produce nodes.
    *(Baseline policy DOES emit these as `used_for_evaluation`
    indirect edges, so this is a net coverage difference — see §5.6.)*

### 3.6 Pipeline-level
21. **Bucketed subagent dispatch** — explicit 30–100 names per
    bucket; planner reviews cross-bucket merges.
22. **Multi-stage refinement architecture** — discover narrow,
    extract permissive, organize gate, audit fold; baseline does
    everything in one shot.
23. **`expand` recursion as a separate pipeline run** — recursing
    onto an upstream node spawns a new full pipeline instead of
    a deeper BFS in the same context.

---

## 4. What the baseline mentions but WE do NOT

These are concepts in the baseline policy that our pipeline
either lacks entirely or models implicitly. Some are gaps; some
are intentional simplifications.

### 4.1 Subject-block parameters (orchestrator inputs)
1. **`max_depth` parameter** — explicit BFS depth cap. We have
   no depth control; `expand` is manual.
2. **`scope_note: "single variant" vs "family handle"`** —
   determines whether post-training mixes attach to base
   (single variant) or to the family handle. We always operate
   at single-checkpoint granularity; no family-handle mode.
3. **`authoritative_paper / authoritative_repo /
   authoritative_card` URLs as inputs** — given to the agent.
   Our `discover` finds these but doesn't surface them as
   pipeline inputs to relate.
4. **`subject_glossary` block** — orchestrator can pre-populate
   canonical_ids; we don't have this.

### 4.2 Schema / required fields
5. **`name` field on nodes (display name)** — REQUIRED. We use
   `formal_name` for both canonical and display.
6. **`provider` field on nodes** — REQUIRED, separate from
   canonical_id. We have `identity.org` inside the identity dict.
7. **`edge_id`** — REQUIRED, unique short slug. We rely on
   array index.
8. **`facets` as a recommended field on every node** — explicit
   open-vocab key-value pairs at node level. We have `identity`
   on items inside groups; the contents overlap but aren't called
   `facets`.
9. **`evidence` as ARRAY** — multiple entries per edge for
   transitive grounding. We have a single evidence string.
10. **`evidence.explanation`** — REQUIRED free prose articulating
    HOW the excerpt supports the claim. We have edge-level
    `description` but no per-evidence explanation.

### 4.3 Recursion semantics
11. **BFS with depth labels** — depth 0 = subject, depth 1 = direct
    deps, depth 2 = upstream of those, etc. We don't track depth.
12. **Recursion patterns by upstream type** — explicit guidance
    for: external instruction-tuned generators, distilled
    reasoning teachers, curated synthetic datasets, web-derived
    datasets, indirect deps. We don't have type-specific recursion
    advice in any prompt.
13. **Stop-recursion conditions** — closed/API-only model with no
    public docs; vague pretraining mix; node already fully captured.
    Our equivalent is the lattice's drop-if-unresolvable rule, but
    that's a coverage gate not a recursion-stop rule.
14. **"Practical pacing" — recursion converges in 2–4 hops** —
    explicit advice. We have nothing.

### 4.4 Aggregator + leaf dual edges (HARD RULE)
15. **For every aggregator mix, ALSO emit
    `subject → trained_on → leaf` for each named sub-corpus.**
    This is the single most consequential gap (see §2.7).

### 4.5 Identity rules
16. **Released LoRA adapter = distinct node; private LoRA = not
    a node.** We don't address.
17. **Model-soup / SLERP / TIES merges = new distinct node.**
    We don't address (can be covered with a `merged_from` coined
    relation, but it's not in any prompt's guidance).
18. **Quantization (same training identity) = same node.** Our
    audit.md folds quantization with `quantization` facet; we
    keep the variant as a separate item but they collapse to one
    node. **Conflict** — baseline says quantization is the same
    node; we keep separate items with shared identity except for
    quantization facet.
19. **Thinking-mode vs non-thinking-mode** — same weights, same
    node in baseline. Our `variant: thinking/no-thinking` keeps
    them separate. **Conflict** — same direction as quantization.
20. **Intermediate training checkpoints** — only released
    checkpoints are nodes. We don't address.

### 4.6 Canonical-ID provider conventions
21. **`<provider>/<slug>` for paper-only / internal datasets** —
    if no separate release, slug like `commoncrawl/cc-main-2024-30`.
    We tend to use paper-derived labels but no fixed convention.
22. **Avoid bare unprefixed slugs** (`wikipedia`, `arxiv`,
    `common-crawl`). Use `wikimedia/wikipedia` etc.
23. **Resolve short forms to specific variants** — paper says
    "Qwen3" → resolve to `qwen/qwen3-32b`, not generic family.
    Our organize/audit should do this via web search; not stated
    as explicit rule.

### 4.7 Evidence quality
24. **Transitive grounding (§8.5)** — when the object is absent
    from the subject's own docs, at least one evidence entry
    SHOULD cite the subject's own paper / HF card / repo. Grader
    flags edges whose evidence never touches subject's docs.
25. **Grader contract (§8.6)** — explicit description of what an
    agentic grader can do (fetch source, locate claim, judge
    support). We have no analogue.

### 4.8 Scoring model
26. **`relationship_verdict` (verified/refuted/unclear)** — the
    LLM-judge axis we'll be measured on.
27. **`evidence_support` (cited_evidence_supports /
    external_support_only / insufficient_evidence /
    not_applicable)** — second axis; "cited" is the win
    condition.
28. **`bucket_concern` and `description_concern` diagnostics** —
    we should know about these to optimize for them.
29. **Fabrication is fatal** — refute rate goes into headline
    metrics. We have no equivalent guard beyond the lattice gate.

### 4.9 Self-verification checklist
30. **8-point pre-emit checklist** in baseline (§SELF-VERIFICATION).
    Includes validator conformance, scope discipline, recursion
    coverage, aggregator+leaf coverage, canonicalization,
    indirect-dep sanity, evidence quality, completeness. Our
    self-verification is implicit per-stage and far less
    exhaustive.

### 4.10 Common failure modes warnings
31. **15 explicit failure modes** with names (under-recursion,
    skipping leaf-level, inconsistent canonical_ids, confusing
    benchmark roles, fabricating canonical IDs, collapsing
    different training identities, …). We have similar guidance
    scattered across our prompts but no consolidated checklist.

---

## 5. Implications for the paper baseline (think section)

The point of this comparison is that **the baseline prompt
defines what competing systems are graded against, and the
LLM-judge alignment matches our outputs against theirs**. Every
schema or rule disagreement either:
- mismatches and lowers our recall (we look worse than we are),
- gets papered over by alias matching (works most of the time),
  or
- creates `bucket_concern` / `description_concern` warnings
  that show up in diagnostics.

For a fair, paper-quality comparison, the recommended changes —
roughly in priority order:

### 5.1 P0 — schema-mapping changes (export adapter)

These can live in a Python adapter that reads our merge
artifact and emits baseline-format JSON. No prompt changes
needed.

- **Lowercase canonical_ids** for the baseline export
  (`Qwen/Qwen3-32B` → `qwen/qwen3-32b`); keep original-case in
  `aliases`. (§2.3)
- **Rename labels** in the baseline export:
  `distilled_from` → `generated_by` (§2.4),
  `initialized_from` → `trained_from` (§2.5).
  In our internal artifacts, keep our names; only the export
  adapter renames.
- **Synthesize `edge_id`** as a slug like
  `e_<subj_slug>_<obj_slug>_<relation>_<index>`. (§2.14)
- **Derive `name` and `provider`** from `formal_name` and
  `identity.org`. (§2.13)
- **Wrap `evidence` in array** of one entry per edge in the
  baseline export. (§2.12)
- **Drop `operation_id` and `operations[]`** in the baseline
  export. (§2.11)
- **Map `direction: STRUCTURAL`** edges:
  - `subset_of` / `contains`: re-emit as model→leaf
    `trained_on` edges (§2.7's aggregator rule); drop the
    structural edge.
  - `released_with` / `supersedes`: emit as `inspired_by`
    indirect or describe in the related edge's description;
    drop standalone.
  - Numeric properties (size, training_tokens, …): drop the
    edges; keep the values in node metadata. (Planned in our
    relate redesign anyway.)

### 5.2 P0 — relate-stage rule changes

These require prompt + validator changes in our pipeline.

- **Aggregator + leaf dual edges**: for every aggregator mix
  the subject `trained_on`, ALSO emit `subject → trained_on →
  leaf` for each named sub-corpus that exists in the lattice.
  This is the single biggest recall-bridging change. (§2.7)
- **Allow lossless `description` ~500 chars** with explicit
  guidance to capture stage, role sub-variants, quantities,
  subsets, ordering, caveats. (§2.10)

### 5.3 P1 — extract-stage rule changes

The bibliography-only and comparison-baseline-only drops
filter out names that the baseline DOES want to capture as
indirect edges (`used_for_evaluation`).

- **Reverse the comparison-baseline-only drop** (or qualify
  it): a name appearing as a bare row in an eval/leaderboard
  table SHOULD still be extracted, since the baseline emits
  `used_for_evaluation` for benchmark scores. The drop's
  intent (cut noise) is sound for our internal analysis but
  hurts coverage in the comparison run. (§5.6)
- Keep the bibliography-only drop — both policies treat
  citation-only mentions as not-a-node.

### 5.4 P1 — organize-stage consistency

- **API-only canonical IDs**: update the worked example in
  `organize.md` to use the **undated** form
  (`OpenAI/gpt-4o-mini`) so it matches the audit fold. (§2.15)
- **Quantization / thinking-mode**: decide whether quantized
  variants and thinking-mode variants are separate items or
  separate facets-of-one-item. The baseline says "same
  weights = same node"; our current convention keeps them
  separate. For the comparison, **collapse them per baseline**
  (one node, quantization/variant in facets only). (§4.5
  items 18, 19)

### 5.5 P2 — facets renaming

- Rename `identity` → `facets` in node export to match
  baseline vocabulary. Keep our internal name.

### 5.6 What to update in the BASELINE prompt itself

A symmetrical question: should the baseline prompt move toward
us on any axis? Three candidates:

- **Closed-set check on `relation_type`** at validation time
  (matching our taxonomy). The baseline currently lets each
  competing system invent labels freely. A grader might want
  to encourage convergence; our 8-label set is a reasonable
  starting point.
- **Operation grouping**: optional — the baseline prompt
  could allow (not require) an `operations[]` field for
  multi-participant event grouping. Otherwise our distinguishing
  feature is invisible.
- **STRUCTURAL category**: NOT recommended to add to baseline.
  The aggregator + leaf duplication rule is cleaner.

### 5.7 What stays divergent (intentional)

These pipeline features should NOT be exported to the
baseline view; they are our internal value-add and would
confuse the comparison.

- The 6-stage architecture (discover → extract → organize →
  audit → relate → triage → merge).
- The off-lattice handling via `object_text`.
- The fuzzy-match Jaccard pass in audit.
- The three identity-fold rules in audit.
- Confidence + corroboration + conflicts + global-policy
  expansion (planned redesign).

---

## TL;DR

| Axis | Status |
|---|---|
| Node ontology (model/dataset, two types) | **same** |
| Direct vs indirect dependency split | **same** |
| Indirect categories (eval, ablation, methodology) | **same** |
| Direct relation labels | **2/5 rename** (`generated_by`, `trained_from`) |
| Canonical_id format (lowercase) | **conflict** — fix in export adapter |
| STRUCTURAL category (subset_of, contains) | **conflict** — drop in export, emit aggregator+leaf instead |
| Aggregator + leaf dual edges | **GAP** — biggest recall hit; add as relate rule |
| `description` length & lossless detail | **conflict** — loosen our cap |
| Operations grouping | **only ours** — strip in export |
| Facets / identity decomposition | **same concept**, rename for export |
| Evidence array vs single entry | **conflict** — wrap in array on export |
| `dependency_kind` field name | **conflict** — rename `direction` → `dependency_kind` |
| Recursion / max_depth | **only baseline** — gap; we use `expand` per-run |
| Quantization / thinking-mode = same node | **conflict** — baseline collapses, we split |
| Comparison-baseline-only refusal in extract | **only ours** — recall regression vs baseline |
| Bibliography-only refusal | **same** |
| `edge_id`, `name`, `provider`, `facets` field names | **only baseline** — synthesize in export |
| Transitive grounding (§8.5) | **only baseline** — should add as soft rule |
| Fuzzy match, fold rules, lattice gate | **only ours** — value-add, keep |

The export adapter (P0) closes most of the schema gap. The
aggregator-leaf rule (P0 prompt change) closes the biggest
recall gap. The description-length and comparison-baseline
rules (P1) close the remaining diagnostic gaps.
