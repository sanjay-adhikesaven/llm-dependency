# ModSleuth

Reconstructing recursive, evidence-grounded dependency graphs of LLM
releases from public artifacts (technical reports, model and dataset
cards, code repositories, release blogs).

This repository accompanies the NeurIPS 2026 submission *"Which Models Are
Our Models Built On? Auditing Invisible Dependencies in Modern LLMs"*
and provides:

- **The ModSleuth pipeline** (paper §3.2): five stages — Gather → Extract →
  Resolve → Relate → Reconcile — implemented on top of Claude Code.
- **A recursive-expansion driver** (paper §3.2 / §A) that BFS-expands
  the top-K upstream parents per seed up to a chosen depth.
- **A post-merge dedup + filter pipeline** that produces the audit-ready
  graph used in our evaluation.
- **The four LLM-based baselines** evaluated against ModSleuth (paper
  §3.3), the **pooled LLM-as-judge verifier** (paper §B), and the
  per-target output JSONs for the run reported in Table 1.
- **An interactive web visualizer** for browsing the resulting graph.

## How the paper maps to the code

| Paper stage (§3.2) | Code |
|---|---|
| Stage 1 — Gather | `modsleuth run discover` |
| Stage 2 — Extract | `modsleuth run extract` |
| Stage 3 — Resolve | `modsleuth run organize` then `modsleuth run audit` |
| Stage 4 — Relate | `modsleuth run relate` |
| Stage 5 — Reconcile | `modsleuth run reconcile` then `modsleuth run triage`, plus `modsleuth run merge` for cross-batch / cross-run merge |
| Recursive expansion (§3.2 / §A) | `modsleuth recursive` |
| Post-merge cleanup (§D) | `modsleuth dedup` (4 sub-stages) |
| Pooled evaluation (§B) | `eval/pooled_eval.py` |
| Baseline runs (§3.3, §C) | `baselines/launch_baselines.py` |

| Paper artifact | File |
|---|---|
| Stage prompts (§A) | `modsleuth/prompts/*.md` |
| Baseline prompt (§C) | `baselines/prompts/baseline_prompt.md` |
| Verifier prompt (§B) | `eval/verifier_prompt.md` |
| Per-system × per-target baseline outputs | `baselines/outputs/<slug>_<subject>.json` |
| Per-target ModSleuth attribution outputs (§B) | `baselines/outputs/{prov,prov_unbounded}_<subject>.json` |
| ModSleuth attribution-rule builder (§B) | `eval/build_modsleuth_inputs.py` |
| Pooled verdicts (Table 1) | `eval/outputs/{verifications.jsonl, score.json, score_per_target.json}` |
| Full-graph audit script (Table 6, §D.2) | `eval/full_graph_audit.py` |
| Graph-stats reproducer (Tables 2 / 4 / 5) | `eval/compute_graph_stats.py` |
| Merged ModSleuth graph (14,769 edges, drives Tables 2 / 4 / 5 / 6 + §B inputs) | `data/merge_artifact.json` (git-lfs) |

## Install

```bash
git clone git@github.com:sanjay-adhikesaven/llm-dependency.git
cd llm-dependency
python -m pip install -e .
```

The pipeline depends on the `claude` CLI being available in `PATH`
(used by every Claude planner / subagent). Set `ANTHROPIC_API_KEY`
before running.

## Quick start

A full target-model run goes through three layers (Figure 2 in
the paper):

```bash
# 1. Base pipeline (Stages 1–5 of paper §3.2): single-target, depth-1.
modsleuth init
modsleuth run discover --target HuggingFaceTB/SmolLM3-3B   # Gather
modsleuth run extract                                       # Extract
modsleuth run organize                                      # Resolve (build lattice)
modsleuth run audit                                         #   ↳ revise
modsleuth run relate                                        # Relate
modsleuth run reconcile                                     # Reconcile
modsleuth run triage                                        #   ↳ flag for expand
modsleuth run merge                                         # combine per-batch
# → writes storage/runs/<id>/merge_artifact.json

# 2. Recursive expansion (paper §3.2 / §A): multi-hop, top-K BFS.
modsleuth recursive --seed HuggingFaceTB/SmolLM3-3B --depth 3 --top-k 5

# 3. Post-merge cleanup (§D / appendix).
modsleuth dedup \
    --source storage/runs/<id>/merge_artifact.json \
    --dest   storage/runs/<id>/graph.json \
    --stages all

# 4. Browse the cleaned graph.
modsleuth viz --source storage/runs/<id>/graph.json --port 8102
# open http://127.0.0.1:8102/
```

Storage paths are controlled by `MODSLEUTH_STORAGE` (state and
artifacts) and `MODSLEUTH_PATH` (SQLite database).

## The base pipeline

The base pipeline runs eight stages over the artifacts of a single
target release. Each stage maps to a stage in the paper's five-stage
figure (Figure 2):

| Stage | Paper (§3.2) | Runtime | Job |
|---|---|---|---|
| `discover` | **Gather** | Claude planner | Fetch the target's official artifacts (paper, model and dataset cards, repo, release blog) into topical batches |
| `extract` | **Extract** | Claude planner per batch (parallel) | Per source: surface every model/dataset mention with kind, atoms, inline links, and source-side anchors |
| `organize` | **Resolve** (build) | Python | Dedup mentions into clusters; detect identity conflicts; build the identity lattice |
| `audit` | **Resolve** (revise) | Claude planner → fans out subagents | Per cluster: identity, aliases, concept_path, confirmed link, conflict resolution. A pure-Python pre-pass in `modsleuth.subsets` populates HF subset / parent metadata before the LLM step. |
| `relate` | **Relate** | Claude planner per batch | Extract operation-level dependency claims (operations + edges + anchors) against the resolved lattice |
| `reconcile` | **Reconcile** (refinement / consolidation / conflict-detection) | Python | Merge overlapping claims into the lattice; surface conflicts |
| `triage` | **Reconcile** (audit step) | Python | Flag candidate ghosts and upstream-node expansion candidates for further review |
| `merge` | (cross-batch / cross-run) | Python | Merge per-batch and per-seed artifacts into a single graph JSON |

The CLI lives at `modsleuth/cli.py`; stage implementations are in
`modsleuth/pipeline.py`. Stage prompts (used by the Claude planners)
are markdown files in `modsleuth/prompts/`.

### Inspecting state mid-run

The `debug` subcommands surface intermediate artifacts:

```bash
modsleuth debug names              # extracted mentions
modsleuth debug names-packet       # what organize will see
modsleuth debug organize           # lattice JSON
modsleuth debug audit              # revised lattice JSON
modsleuth debug lattice -q smollm3 # search the lattice
modsleuth debug relate             # per-batch relate edges
modsleuth debug triage             # auto_expand / decline / manual buckets
modsleuth debug merge              # cross-run merged graph
```

## Recursive expansion

The base pipeline produces only the immediate (one-hop) dependencies of
a single target. The paper's recursive-tracing claim (Figure 1, §3.2,
§4) requires expanding upstream artifacts as fresh targets and re-merging.

`modsleuth recursive` is a reference BFS driver:

```bash
modsleuth recursive \
    --seed allenai/OLMo-3-1125-32B \
    --seed nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --seed rl-research_DR-Tulu-8B \
    --seed HuggingFaceTB/SmolLM3-3B \
    --depth 3 --top-k 5
```

For each seed it runs the full base pipeline once (in its own per-seed
`MODSLEUTH_STORAGE` directory), then iteratively expands the top-K
newly-discovered upstream artifacts at each depth, ranked by parent
count in the merged graph. After each expansion round it re-runs
`merge` so the per-seed graph reflects the latest discoveries. Per-node
expansion uses the existing `modsleuth run expand --node <name>` step,
which re-runs `discover` → `reconcile` against the named upstream
artifact within the same storage.

The exact strategy used in the paper is target-specific (seed list,
per-seed K, optional pre-seeded high-betweenness bridge artifacts so
seeds share an upstream backbone). `modsleuth/recursive.py` is a
working reference; tune the strategy in that file to reproduce the
paper's exact recursion or to match a different audit budget.

To merge across seeds into a single graph, pass each per-seed
`merge_artifact.json` to `modsleuth run merge --source <path>`.

## Post-merge cleanup (`modsleuth dedup`)

Four stages run over the merged JSON graph. Each can be invoked alone:

| Stage | Mechanism | What it does |
|---|---|---|
| `heuristic` | no LLM | Signature-based clustering with hard separators on org × bare_norm × versions × sizes × stages × dates × parens × bracket attrs. Folds bare names into the highest-degree compatible prefixed cluster. Drops internal paths and free-text descriptive nodes. Filters low-signal concept names with degree < 3. |
| `hub-audit` | Opus 4.7 + max thinking | For each top out-hub and in-hub, asks the LLM to drop edges that are duplicates, hallucinations, vacuous concepts, or wrong-relation. Tagged drop categories. |
| `node-dedup` | Opus 4.7 + max thinking | Builds candidate dedup clusters across the whole graph from five high-precision signals (lex-collapse, token-Jaccard ≥ 0.6, substring containment, cross-org bare-lex match, suffix stripping). Verifies each cluster with the LLM. Applies decisions via a conflict-guarded union-find that refuses to merge components with mutually conflicting versions / sizes / stages / dates. |
| `release` | Opus 4.7 + high effort | Classifies every node as KEEP (officially released artifact / standard benchmark) or DROP (intermediate research checkpoint, internal training-data variant, prose alias). For each dropped node, transitively rewires `A → DROP → B` chains along compatible relation pairs (`trained_from`+`trained_from`, `trained_on`+`trained_on`, etc.) so released-to-released ancestry stays connected. |

Each stage reads JSON, applies its operation, writes JSON. Sanity
invariants (distinct version numbers, distinct year-stamped releases,
all seed targets present, conflict-free dates / sizes / stages) are
asserted before every write.

```bash
# Run all four stages end-to-end (default).
modsleuth dedup --source merge.json --dest graph.json --stages all

# Or run a single stage.
modsleuth dedup --source merge.json           --dest after_heuristic.json --stages heuristic
modsleuth dedup --source after_heuristic.json --dest after_hub.json       --stages hub-audit
modsleuth dedup --source after_hub.json       --dest after_node.json      --stages node-dedup
modsleuth dedup --source after_node.json      --dest graph.json           --stages release
```

The hub-audit and node-dedup stages take ~25 min each on a 15k-edge
graph with 24 parallel `claude` workers; release-filter takes under 2 min.

## Visualizer

```bash
modsleuth viz --source path/to/graph.json --port 8102
```

A self-contained HTTP server with a vis-network frontend. Default view
caps at the top 200 nodes by degree (min-degree ≥ 10) with physics off;
raise the slider or trigger an ego mode (1-hop or 2-hop) on any selected
node to explore further. Search auto-pivots to ego mode on the matching
node.

## Edge audit (sanity analyzer)

```bash
python edge_audit.py --source path/to/graph.json
```

Static edge-pattern analyzer: top hubs, anchor-coverage distribution,
near-duplicate object/subject names that survived dedup, weak-evidence
claims, and (subject, object) pairs with multiple distinct relations.
Useful as a quick sanity-check on any graph version.

## Baselines and pooled evaluation (paper §3.3, §B, §C)

`baselines/` and `eval/` contain everything needed to reproduce the
comparative evaluation reported in Table 1 of the paper, and to evaluate
any new submission against the same pool.

### Systems

Six systems are evaluated against the same four targets (OLMo 3,
Nemotron 3 Super, DR-Tulu, SmolLM3). The four single-pass baselines get
the same baseline prompt template (paper §C; full text at
`baselines/prompts/baseline_prompt.md`); the two ModSleuth scopes are
attribution variants of a single ModSleuth run (paper §B).

| Slug | Paper label | Configuration |
|---|---|---|
| `gpt55pro` | GPT-5.5 Pro | OpenAI Responses API, `web_search_preview`, background mode |
| `gpt54pro` | GPT-5.4 Pro | OpenAI Responses API, `web_search_preview`, background mode |
| `o3dr` | ChatGPT Deep Research (`o3-deep-research`) | OpenAI Responses API, `web_search_preview`, background mode |
| `cc` | CC-single (single-prompt Claude Code) | `claude -p` headless, Opus 4.7 1M context, default effort |
| `prov` | ModSleuth (depth-1) | Subject canonical-form == target's canonical id |
| `prov_unbounded` | ModSleuth (unbounded) | Depth-1 ∪ seed-tagged anchor ∪ uniquely-tied worker (§B) |

The internal slug → paper label mapping is also encoded in
`eval/pooled_eval.py:SLUG_TO_LABEL` and is what the rendered table
prints.

Reproduce the four baselines:

```bash
cd baselines
OPENAI_API_KEY=sk-... python3 launch_baselines.py
```

Each (system, target) pair writes to `baselines/outputs/<slug>_<subject>.json`.
The 16 baseline outputs and the 8 ModSleuth attribution outputs
(`prov_<target>.json`, `prov_unbounded_<target>.json`) from the run
reported in the paper are committed.

### Building the ModSleuth attribution outputs

The two ModSleuth rows in Table 1 are derived from a single merged
graph (`merge_artifact.json`, the 14,769-edge ModSleuth artifact)
under the two attribution scopes defined in paper §B. To rebuild
`prov_<target>.json` and `prov_unbounded_<target>.json` from a fresh
merge:

```bash
cd eval
python3 build_modsleuth_inputs.py \
    --merge-artifact path/to/merge_artifact.json \
    --out-dir ../baselines/outputs
```

The script implements the §B rules verbatim:

- **depth-1**: subject's canonical form (lowercased, non-alphanumeric
  collapsed to `-`, HF org prefix preserved) exactly matches the
  target's canonical identifier.
- **unbounded**: depth-1 ∪ at least one anchor source path containing
  `/seeds/<T's seed dir>/`, ∪ at least one anchor source path
  containing `/workers/<w>/` where worker `w` co-occurs (across the
  whole merge artifact) only with `T`'s seed directory.

The merge artifact itself (`data/merge_artifact.json`, the 14,769-edge
post-dedup ModSleuth graph) is shipped via **git-lfs** because it is
too large for a regular git checkout (~86 MB). To fetch it after
cloning:

```bash
git lfs install
git lfs pull
```

The same file is also the input to `full_graph_audit.py` and to the
`compute_graph_stats.py` reproducer for Tables 2/4/5.

### Pooled evaluation

`eval/pooled_eval.py` is the canonical pooled LLM-as-judge verifier
described in paper §B. For each target, every emitted edge across all
systems is pooled and clustered by canonicalized `(subject, object)`
pair. Each cluster's representative claim (longest description) is sent
to a single `claude-sonnet-4-6` verifier instance equipped with
`web_search`, which returns one of `verified` / `refuted` / `unclear`.
A single verifier verdict cleanly attributes back to every system that
proposed an edge in that cluster, so per-system Verified / Refuted
counts mean: how many clusters did this system contribute to, broken
down by verdict.

The verifier's system prompt is at `eval/verifier_prompt.md`.

Reproduce:

```bash
cd eval
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py
```

The verdicts in `eval/outputs/verifications.jsonl` (and the aggregated
`score.json` / `score_per_target.json`) are the exact records used in
the paper. Verifications append incrementally and resume on kill.

### Full-graph audit (Table 6, paper §D.2)

Table 1 measures comparative recall across systems via cluster-level
pooling. Table 6 measures *full-graph precision*: each of the 14,769
relations in the merged ModSleuth graph is verified individually.

```bash
cd eval
ANTHROPIC_API_KEY=sk-ant-... python3 full_graph_audit.py \
    --merge-artifact ../data/merge_artifact.json \
    --out outputs/full_graph_verifications.jsonl
```

The script appends one verdict per line and resumes on kill. It uses
the same `claude-sonnet-4-6` + `web_search` verifier as `pooled_eval.py`
and the same `verifier_prompt.md` system prompt. At completion it
prints (and writes to `outputs/full_graph_verifications.score.json`)
the totals reported in Table 6.

### Graph-level statistics (Tables 2 / 4 / 5, paper §4.1, §D.1)

`eval/compute_graph_stats.py` reproduces the three descriptive tables
that summarize the recovered graph:

```bash
python3 eval/compute_graph_stats.py \
    --merge-artifact data/merge_artifact.json
```

* **Table 2** (edges grouped by audit role × dependency-kind) is
  recovered exactly from the relation/dependency-kind fields.
* **Table 4** (per-target ancestor counts and max depth) is computed
  by BFS from per-target seed lists. Max-depth values reproduce the
  paper exactly; ancestor counts depend on the exact seed configuration
  used during the per-investigation expansion (see `TABLE4_TARGETS` in
  the script — adjust to match your run if you've used a different
  seed mix).
* **Table 5** (source-type distribution of operations) is computed by
  classifying each relation's `anchor_list` source paths. The total
  (14,701) reproduces exactly; the per-bucket counts depend on the
  source-classifier regex (see `_RX_*` in the script — adjust to match
  your local conventions).

### Adding a new submission

If your submission follows the same per-target `{nodes, edges}` JSON
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

## Tests

```bash
python -m pytest tests/ -q
```

## Repository layout

```
.
├── modsleuth/                  # ModSleuth pipeline package
│   ├── cli.py                  # `modsleuth` CLI entry point
│   ├── config.py               # storage / model / env defaults
│   ├── pipeline.py             # all stage implementations
│   ├── prompts/                # stage-level markdown prompts (paper §A)
│   │   ├── discover.md         #   Gather (Stage 1)
│   │   ├── extract.md          #   Extract (Stage 2)
│   │   ├── organize.md         #   Resolve build (Stage 3)
│   │   ├── audit.md            #   Resolve revise (Stage 3)
│   │   ├── relate.md           #   Relate (Stage 4)
│   │   └── triage.md           #   Reconcile audit step (Stage 5)
│   ├── resolve.py              # identity-lattice resolver
│   ├── store.py                # SQLite-backed pipeline state
│   ├── subsets.py              # HF metadata pre-pass for the audit stage
│   ├── viz.py                  # interactive HTTP graph viewer
│   ├── recursive.py            # multi-hop expansion driver (§A)
│   └── dedup/                  # post-merge dedup pipeline (4 sub-stages)
│       ├── __main__.py         #   `python -m modsleuth.dedup`
│       └── lib.py              #   shared helpers (signatures, union-find, …)
├── baselines/                  # baseline runs (paper §3.3 / §C)
│   ├── launch_baselines.py     #   fires all 16 (4 × 4) runs in parallel
│   ├── prompts/                #   template + per-target baseline prompts
│   │   ├── baseline_prompt.md
│   │   └── baseline_prompt_<subject>.md
│   ├── outputs/                #   committed per-system per-target JSON graphs
│   │                           #   (4 baselines + prov / prov_unbounded ModSleuth, §B)
│   └── README.md
├── eval/                       # pooled LLM-as-judge verifier (paper §B)
│   ├── pooled_eval.py          #   Sonnet 4.6 + web_search per cluster (Table 1)
│   ├── full_graph_audit.py     #   per-edge audit across the full graph (Table 6, §D.2)
│   ├── compute_graph_stats.py  #   reproduces Tables 2, 4, 5 (paper §4.1, §D.1)
│   ├── build_modsleuth_inputs.py  # builds prov_<target>.json + prov_unbounded_<target>.json
│   │                           #   from a merge_artifact.json (§B attribution rules)
│   ├── verifier_prompt.md      #   verifier system prompt
│   ├── outputs/                #   verifications.jsonl + score{,_per_target}.json
│   └── README.md
├── data/                       # release-only data artifact (git-lfs)
│   └── merge_artifact.json     #   the 14,769-edge ModSleuth merged graph
├── edge_audit.py               # static noise-pattern analyzer
├── tests/
├── pyproject.toml
├── requirements.txt
├── schema.sql
└── README.md
```

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

## Citation

If you use ModSleuth, please cite the NeurIPS 2026 paper:

```bibtex
@inproceedings{modsleuth2026,
  title     = {Which Models Are Our Models Built On? Auditing Invisible Dependencies in Modern LLMs},
  author    = {Anonymous Author(s)},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

TBD (add `LICENSE` before public release).
