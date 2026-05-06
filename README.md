# llm-dependency

Reconstructs recursive, evidence-grounded dependency graphs of LLM releases
from public artifacts (technical reports, model and dataset cards, code
repositories, release blogs).

The system reads heterogeneous public release artifacts, identifies and
resolves artifact mentions across sources, builds operation-level
dependency claims anchored in source excerpts, reconciles overlapping or
conflicting evidence, and recursively expands upstream artifacts to trace
multi-hop chains. The output is a single self-contained JSON graph
(`merge_artifact_deduped_v8.json`) with nodes for models / datasets and
edges for relationships such as `trained_from`, `trained_on`,
`generated_by`, `filtered_by`, `transformed_by`, `used_for_evaluation`,
`inspired_by`, `decontaminated_against`, etc.

Reproduction details for the hero run (4 seed releases — OLMo 3, DR-Tulu,
Nemotron-3, SmolLM3 — yielding 2,844 nodes / 14,769 edges / 51,456
anchors) are in [`REPRODUCE.md`](REPRODUCE.md).

## Quick start

```bash
git clone git@github.com:sanjay-adhikesaven/llm-dependency.git
cd llm-dependency
python -m pip install -e .

# Initialize storage (./storage/graph.db is the pipeline state DB)
gdb init

# Run the base pipeline against a target model / dataset
gdb run discover --target HuggingFaceTB/SmolLM3-3B
gdb run extract
gdb run organize
gdb run audit
gdb run relate
gdb run reconcile
gdb run triage
gdb run merge
```

Runtime paths are controlled by `GDB_STORAGE` (state and artifacts) and
`GDB_PATH` (lattice / cache). The CLI is in `gdb/cli.py`; pipeline stages
live in `gdb/pipeline.py`.

## Pipeline

The base pipeline has eight stages:

| Stage | Runtime | Job |
|---|---|---|
| `discover` | one Claude planner | fetch sources for the target into batches |
| `extract` | one Claude planner per batch (Python parallel) | per source: surface mentions, kind, atoms, inline links, source-side anchors |
| `organize` | Python | dedup mentions into clusters, detect identity conflicts |
| `audit` | one Claude planner → fans out subagents | per cluster: identity, aux, aliases, concept_path, confirmed link, conflict resolution |
| `relate` | one Claude planner per batch | extract dependency claims (operations + edges + anchors) against the resolved lattice |
| `reconcile` | Python | merge overlapping claims; refinement, consolidation, conflict detection |
| `triage` | Python | flag candidate ghosts for further expansion |
| `merge` | Python | merge per-batch artifacts into a single graph |

## Recursive expansion (beam search)

The base pipeline recovers one-hop dependencies for a single target. To
recover deep recursive ancestry across the seed corpus we layer a beam
expansion on top.

| Script | Role |
|---|---|
| `beam_v5.py` | 4-phase beam search: bridge → d=2 → d=3 → d=4. Per-seed per-level top-K = 5 by parent-count ranking; pre-seeded set of 12 high-betweenness bridge nodes; worker cap 8. |
| `recurse_v4.py` | Focused depth+0 expansion of the top-N ghost nodes that triage flagged for `auto_expand` but never received worker capacity. |
| `resume_bfs.py` | BFS expansion across all seeds with reduced concurrency. State is recovered from existing artifacts. |
| `retry_v3.py` | Retry pass for nodes whose expansion didn't complete (in-flight when an orchestrator was killed, or marked `ERROR`). |
| `auto_merge_combined.py` | Polls for `recurse_v4` and `beam_v5` to exit, then triggers a final merge. |
| `snapshot_merge.py` | Out-of-band merge into an isolated `GDB_STORAGE` so mid-run snapshots don't interfere with the live merge DB. |
| `watch_and_merge.py` | Polls for `retry_v3.py` to exit, waits 60 s for final disk writes, then runs `gdb run merge`. |
| `final_orchestrator.py` | Wakeup-to-done orchestrator: chains retries until convergence, then merges. Idempotent (writes `run-logs/RUN_DONE.txt` at completion); refuses to start a second merge if one is running; aggressive disk pre-flight cleanup. |
| `keep_alive_orchestrator.sh` | Bash wrapper that re-runs `final_orchestrator.py` if it dies prematurely (OOM, system blip), up to `MAX_LOOPS=20`. |

## Post-merge dedup pipeline (V1 → V8)

The base merge produces a graph (V0) with substantial cross-source
surface-form duplication. We layer eight sequential dedup stages on top
of the merge artifact, each preserving prior outputs on disk.

| Stage | Script | Role |
|---|---|---|
| **V1** | `dedup_graph.py` | Heuristic dedup: categorical drops (internal paths), signature-based clustering with hard separators (org / bare_norm / versions / sizes / stages / dates), no-org name folding, canonical pick, low-signal filter, edge rewrite + anchor merge. |
| **V2** | `dedup_apply_splits.py` | NO\_SPLIT preservation: re-runs V1 clustering deterministically, parses Opus verdicts on borderline clusters, reverts over-merges (e.g., MMLU subjects, AIME 2024 vs 2025, OLMo-2 stage checkpoints, FLAN community re-releases). |
| **V3** | `dedup_v3.py` | Fuzzy surface-form merge: hyphen-collapsed alt-key (`rl-zero` ↔ `rlzero`), version normalization (`3_3` ↔ `3.3`), bare-vs-prefixed merge with most-popular-target tiebreak. Adds `paren_specifier` and `bracket_specifier` to the signature so subset distinctions (e.g., MMLU STEM vs humanities) survive. |
| **V4** | (superseded by V5; not in repo) | Hub audit, free-form drop reasons, batch=80, 8-way parallel. Used in early development; superseded once V5's tagged outputs were available. |
| **V5** | `opus_verify_v5.py` | Deep hub audit with Opus 4.7 + `--effort max`. Top 75 OUT-hubs + top 30 IN-hubs, 40-edge batches sorted by anchor count, 12 parallel workers. Tagged drop categories: `DUPLICATE / HALLUCINATED / VACUOUS / WRONG_RELATION`. |
| **V6** | `dedup_v6.py` | Whole-graph node dedup. Three high-precision signals: lex-collapse, token-Jaccard ≥ 0.60 (top-3 candidates per node), substring containment. **No connected-components clustering** (initial design with anchor-co-citation + co-neighborhood signals fused unrelated hubs into mega-clusters; see `REPRODUCE.md`). 24-way parallel. |
| **V7** | `release_filter_v7.py` | Release-only filter. Opus classifies every node as KEEP (released artifact) or DROP (intermediate research checkpoint, internal training-data variant, prose alias). Transitive edge rewiring through DROP nodes along compatible relation pairs. |
| **V8** | `dedup_v8.py` + `dedup_v8_fix.py` | Cross-org / suffix dedup with conflict-guarded union. Cross-org bare-lex match (catches `OpenAI/GPT-2` ↔ `openai-community/gpt2`), suffix-stripping (`-turbo`, `-Instruct`, `-hf`), bare-no-slash → prefixed family. The `_fix` step adds a conflict guard that skips any union that would combine components with mutually-conflicting specifier sets (different dates, versions, sizes, or stages). |

Each script writes its output to a new file alongside V0:
`merge_artifact_deduped.json` (V2), `_v3.json`, `_v5.json`, `_v6.json`,
`_v7.json`, `_v8.json`. Prior versions are preserved for rollback /
comparison.

Sanity invariants are asserted before each write:

- Olmo-3 nodes > 0 **and** Olmo-3.1 nodes > 0 (different version numbers
  must stay distinct)
- All four seed prefixes still present
- AIME 2024 ≠ AIME 2025
- Some `cais/mmlu` subject splits preserved
- (V8-fix) `OLMo-2-0325-32B-Instruct` ≠ `OLMo-2-1124-32B-Instruct`

## Visualizer

```bash
python viz_v4.py --port 8102 --source v8
# open http://127.0.0.1:8102/
```

A self-contained HTTP server that loads any version (V0 / V2–V8) of the
merged artifact directly. Default view caps at top 200 nodes by degree
with min-degree ≥ 10 and physics off (otherwise vis-network locks the
browser at ~15k edges). Search auto-pivots to ego mode on the selected
node; ego (1-hop) and ego (2-hop) buttons in the toolbar.

## Edge auditing

```bash
python edge_audit.py
```

Static edge-pattern analysis for spotting residual noise after dedup
(top-degree hubs, near-duplicate object names, weak-evidence patterns,
zero-anchor edges). Used during development; useful as a quick
sanity-check on any new graph version.

## Tests

```bash
python -m pytest tests/ -q
```

## Repository layout

```
.
├── gdb/                    # base pipeline package (CLI + 8 stages)
│   ├── cli.py
│   ├── pipeline.py
│   ├── prompts/            # stage-level prompts (Markdown)
│   ├── resolve.py          # identity-lattice resolver
│   ├── store.py            # SQLite-backed pipeline state
│   ├── subsets.py
│   └── viz.py              # legacy viewer (reads SQLite)
├── beam_v5.py              # recursive beam expansion
├── recurse_v4.py           # focused ghost-node expansion
├── resume_bfs.py           # BFS expansion orchestrator
├── retry_v3.py             # retry pass for unfinished nodes
├── auto_merge_combined.py  # polls for completion, triggers final merge
├── snapshot_merge.py       # isolated out-of-band merge
├── watch_and_merge.py      # watch retry_v3, then merge
├── final_orchestrator.py   # wakeup-to-done orchestrator
├── keep_alive_orchestrator.sh  # bash wrapper around final_orchestrator
├── dedup_graph.py          # V1 heuristic dedup
├── dedup_apply_splits.py   # V2 NO_SPLIT reverts
├── dedup_v3.py             # V3 fuzzy surface-form merge
├── opus_verify_v5.py       # V5 deep hub audit (Opus 4.7 + max thinking)
├── dedup_v6.py             # V6 whole-graph node dedup
├── release_filter_v7.py    # V7 release-only filter
├── dedup_v8.py             # V8 cross-org / suffix dedup
├── dedup_v8_fix.py         # V8 conflict-guarded union fix
├── viz_v4.py               # custom visualizer (reads merged JSON)
├── edge_audit.py           # post-dedup noise pattern analyzer
├── tests/
├── paper/                  # paper artifacts (tex)
├── prompts/                # investigator prompt (master + per-subject copies, used by baselines)
├── baselines/              # baseline runs against the same 4 subjects
│   ├── launch_baselines.py
│   └── outputs/            # 16 baseline JSON graphs (4 systems × 4 subjects)
├── eval/                   # pooled LLM-as-judge verifier and verdict outputs
│   ├── pooled_eval.py
│   └── outputs/            # verifications.jsonl, score.json, score_per_target.json
├── pyproject.toml
├── schema.sql
├── README.md               # this file
└── REPRODUCE.md            # full hero-run reproduction notes
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

## Concept terminology

- **operation** — a structured group of edges that jointly describes one
  pipeline event (e.g., a DPO step). Edges within an operation share an
  anchor list and description. This preserves event structure where a
  flat pairwise edge list would erase it.
- **anchor** — a source-side citation: file path, position, and verbatim
  excerpt grounding a claim to a specific spot in the source corpus.
- **identity lattice** — a partial-order structure for artifact identity
  with vague-mention roots, partial-spec intermediate nodes, and
  pinned-link entity leaves. Each node is an open-vocabulary set of
  facets (`family`, `size`, `stage`, …); subset ordering on facet sets
  defines the hierarchy.
- **dependency-kind** — coarse type label (`direct` / `indirect`) on
  every edge, distinguishing artifacts that materially enter weights /
  training data from those that merely influence development decisions.

## License

TBD (add `LICENSE` file before public release).

## Citation

```bibtex
% TBD: add bibtex entry once the paper is public
```
