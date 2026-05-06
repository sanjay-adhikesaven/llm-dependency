# llm-dependency

Reconstruct recursive, evidence-grounded dependency graphs of LLM releases
from public artifacts (technical reports, model and dataset cards, code
repositories, release blogs).

The system reads heterogeneous public release artifacts, identifies and
resolves artifact mentions across sources, builds operation-level dependency
claims anchored in source excerpts, reconciles overlapping or conflicting
evidence, and produces a self-contained JSON graph with nodes for models /
datasets and edges for relationships such as `trained_from`, `trained_on`,
`generated_by`, `filtered_by`, `transformed_by`, `used_for_evaluation`,
`inspired_by`, and `decontaminated_against`.

## Install

```bash
git clone git@github.com:sanjay-adhikesaven/llm-dependency.git
cd llm-dependency
python -m pip install -e .
```

The package depends on the `claude` CLI being available in `PATH` (used by
the LLM-driven dedup stages). Set `ANTHROPIC_API_KEY` before running.

## Quick start

A full run on a target model has two phases: build the graph with `lineage`,
then clean it with `dedup.py`.

```bash
# 1. Build the raw graph for a target model.
lineage init
lineage run discover --target HuggingFaceTB/SmolLM3-3B
lineage run extract
lineage run organize
lineage run audit
lineage run relate
lineage run reconcile
lineage run triage
lineage run merge          # writes storage/runs/<id>/merge_artifact.json

# 2. Run the dedup + filter pipeline on the merge artifact.
python dedup.py \
    --source storage/runs/<id>/merge_artifact.json \
    --dest   storage/runs/<id>/graph.json \
    --stages all

# 3. Browse the cleaned graph.
python viz.py --source storage/runs/<id>/graph.json --port 8102
# open http://127.0.0.1:8102/
```

Runtime paths are controlled by `LINEAGE_STORAGE` (state and artifacts) and
`LINEAGE_PATH` (lattice / cache).

## Pipeline

The base `lineage` pipeline has eight stages.

| Stage | Runtime | Job |
|---|---|---|
| `discover` | one Claude planner | fetch the target's official artifacts (paper, model and dataset cards, repo, release blog) into topical batches |
| `extract` | one Claude planner per batch (parallel) | per source: surface every model/dataset mention with kind, atoms, inline links, and source-side anchors |
| `organize` | Python | dedup mentions into clusters; detect identity conflicts |
| `audit` | one Claude planner → fans out subagents | per cluster: identity, aliases, concept_path, confirmed link, conflict resolution |
| `relate` | one Claude planner per batch | extract operation-level dependency claims (operations + edges + anchors) against the resolved lattice |
| `reconcile` | Python | merge overlapping claims via refinement, consolidation, and conflict detection |
| `triage` | Python | flag candidate ghosts for further review |
| `merge` | Python | merge per-batch artifacts into a single graph JSON |

The CLI lives in `lineage/cli.py`; stage implementations are in `lineage/pipeline.py`.
Stage prompts (used by the Claude planners) are markdown files in
`lineage/prompts/`.

### Recursive expansion

The base pipeline produces the immediate (one-hop) dependencies of a single
target. To recover a multi-hop graph across several seed releases, run the
pipeline once per seed (re-using `LINEAGE_STORAGE` per seed), then re-run it on
each upstream artifact `lineage run merge` discovers, until a chosen depth is
reached. A reference implementation of this beam-search-style expansion is
not bundled in this repo because its useful parameters (which seeds, which
high-betweenness bridge artifacts to expand first, per-seed ranking) are
target-specific. The algorithm we used:

1. Run the base pipeline against each seed; merge into a per-seed graph.
2. From each per-seed graph, take the top-K newly-discovered upstream
   artifacts at depth $d$, ranked by *parent count* (how many edges from the
   already-expanded subgraph point at them).
3. Run the base pipeline against each of those upstream artifacts; merge.
4. Repeat for $d \in \{2, 3, 4\}$.
5. Optionally pre-seed a small set of high-betweenness bridge artifacts
   (popular benchmarks, common base models) before any per-seed expansion
   so the seed families share an upstream backbone.

## Dedup pipeline

`dedup.py` runs four stages over the merged JSON graph; each stage can also
be run in isolation.

| Stage | Mechanism | What it does |
|---|---|---|
| `heuristic` | no LLM | Signature-based clustering with hard separators on org × bare_norm × versions × sizes × stages × dates × parens × bracket attrs. Folds bare names into the highest-degree compatible prefixed cluster. Drops internal paths and free-text descriptive nodes. Filters low-signal concept names with degree < 3. |
| `hub-audit` | Opus 4.7 + max thinking | For each top out-hub and in-hub, asks the LLM to drop edges that are duplicates, hallucinations, vacuous concepts, or wrong-relation. Tagged drop categories. |
| `node-dedup` | Opus 4.7 + max thinking | Builds candidate dedup clusters across the whole graph from five high-precision signals (lex-collapse, token-Jaccard ≥ 0.6, substring containment, cross-org bare-lex match, suffix stripping). Verifies each cluster with the LLM. Applies decisions via a conflict-guarded union-find that refuses to merge components with mutually conflicting versions / sizes / stages / dates. |
| `release` | Opus 4.7 + high effort | Classifies every node as KEEP (officially released artifact / standard benchmark) or DROP (intermediate research checkpoint, internal training-data variant, prose alias). For each dropped node, transitively rewires `A → DROP → B` chains along compatible relation pairs (`trained_from`+`trained_from`, `trained_on`+`trained_on`, etc.) so released-to-released ancestry stays connected. |

Each stage reads JSON, applies its operation, writes JSON. Sanity invariants
(distinct version numbers, distinct year-stamped releases, all seed targets
present, conflict-free dates / sizes / stages) are asserted before every
write.

```bash
# Run all four stages end-to-end (default).
python dedup.py --source merge.json --dest graph.json --stages all

# Or run a single stage.
python dedup.py --source merge.json    --dest after_heuristic.json --stages heuristic
python dedup.py --source after_heuristic.json --dest after_hub.json --stages hub-audit
python dedup.py --source after_hub.json --dest after_node.json     --stages node-dedup
python dedup.py --source after_node.json --dest graph.json         --stages release
```

The hub-audit and node-dedup stages take ~25 min each on a 15k-edge graph
with 24 parallel `claude` workers; release-filter takes under 2 min.

## Visualizer

```bash
python viz.py --source path/to/graph.json --port 8102
```

A self-contained HTTP server with a vis-network frontend. Default view caps
at the top 200 nodes by degree (min-degree ≥ 10) with physics off; raise
the slider or trigger an ego mode (1-hop or 2-hop) on any selected node to
explore further. Search auto-pivots to ego mode on the matching node.

## Edge audit

```bash
python edge_audit.py --source path/to/graph.json
```

Static edge-pattern analyzer: top hubs, anchor-coverage distribution,
near-duplicate object/subject names that survived dedup, weak-evidence
claims, and (subject, object) pairs with multiple distinct relations. Useful
as a quick sanity-check on any graph version.

## Tests

```bash
python -m pytest tests/ -q
```

## Repository layout

```
.
├── lineage/        # base pipeline package
│   ├── cli.py          # `lineage` CLI entry point
│   ├── config.py
│   ├── pipeline.py     # stage implementations
│   ├── prompts/        # stage-level markdown prompts
│   ├── resolve.py      # identity-lattice resolver
│   ├── store.py        # SQLite-backed pipeline state
│   └── subsets.py      # HF metadata subset population
├── dedup.py            # 4-stage dedup pipeline (CLI)
├── dedup_lib.py        # shared dedup helpers (signatures, union-find, LLM, …)
├── viz.py              # HTTP visualizer for a merged JSON graph
├── edge_audit.py       # static noise-pattern analyzer
├── tests/
├── prompts/                # investigator prompt (master + per-subject copies, used by baselines)
├── baselines/              # baseline runs against the same subjects
│   ├── launch_baselines.py
│   └── outputs/            # baseline JSON graphs
├── eval/                   # pooled LLM-as-judge verifier and verdict outputs
│   ├── pooled_eval.py
│   └── outputs/            # verifications.jsonl, score.json, score_per_target.json
├── pyproject.toml
├── schema.sql
└── README.md
```

## Baselines and evaluation

The `baselines/` and `eval/` directories contain everything needed to
reproduce the comparative evaluation reported in the paper, and to
evaluate any new submission against the same pool.

### Baseline systems

Four single-pass baselines are run against the same four subjects
(OLMo 3, Nemotron 3 Super, DR-Tulu, SmolLM3). Each gets the same
investigator prompt as our system; the difference is the absence of
the multi-stage harness.

| Slug | System | Configuration |
|---|---|---|
| `gpt55pro` | GPT-5.5-Pro | OpenAI Responses API, `web_search_preview`, background mode |
| `gpt54pro` | GPT-5.4-Pro | OpenAI Responses API, `web_search_preview`, background mode |
| `o3dr` | OpenAI Deep Research (`o3-deep-research`) | OpenAI Responses API, `web_search_preview`, background mode |
| `cc` | Single-prompt Claude Code | `claude -p` headless (Opus 4.7 1M, default effort) |

Reproduce:

```bash
cd baselines
OPENAI_API_KEY=sk-... python3 launch_baselines.py
```

Each (system, subject) pair writes to `baselines/outputs/<slug>_<subject>.json`.
The 16 outputs from the run reported in the paper are committed.

### Pooled evaluation

`eval/pooled_eval.py` is the canonical pooled LLM-as-judge verifier. For
each target, every emitted edge across all systems is pooled and
clustered by canonicalized `(subject, object)` pair. Each cluster's
representative claim (longest description) is sent to a single
`claude-sonnet-4-6` verifier instance equipped with `web_search`, which
returns one of `verified` / `refuted` / `unclear`. A single verifier
verdict cleanly attributes back to every system that proposed an edge in
that cluster, so per-system Verified / Refuted counts mean: how many
clusters did this system contribute to, broken down by verdict.

Reproduce:

```bash
cd eval
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py
```

The verdicts in `eval/outputs/verifications.jsonl` (and the aggregated
`score.json` / `score_per_target.json`) are the exact records used in
the paper. Verifications append incrementally and resume on kill.

### Adding a new submission

If your submission follows the same per-subject `{nodes, edges}` JSON
convention as the baselines, drop your files into `baselines/outputs/`
named `<slug>_<subject>.json`, then re-run `pooled_eval.py` with your
slug appended:

```bash
cd eval
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py \
    --systems gpt55pro,gpt54pro,cc,o3dr,mysystem
```

Only clusters your system contributes that aren't already covered get
fresh verifier calls. Existing verdicts are preserved.

## Concepts

- **operation** — a structured group of edges that jointly describes one
  pipeline event (e.g., a DPO step). Edges within an operation share an
  anchor list and description, preserving the event structure that a flat
  pairwise edge list would erase.
- **anchor** — a source-side citation: file path, position, and verbatim
  excerpt grounding a claim to a specific spot in the source corpus.
- **identity lattice** — partial-order structure for artifact identity with
  vague-mention roots, partial-spec intermediate nodes, and pinned-link
  entity leaves. Each node is an open-vocabulary set of facets (`family`,
  `size`, `stage`, …); subset ordering on facet sets defines the hierarchy.
- **dependency-kind** — coarse type label (`direct` / `indirect`) on every
  edge, distinguishing artifacts that materially enter weights or training
  data from those that merely influence development decisions.

## License

TBD (add `LICENSE` before public release).
