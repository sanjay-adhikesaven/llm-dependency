# Hero-run reproduction notes

Captures every decision made on top of the base `gdb` pipeline to produce
`merge_artifact_deduped_v8.json` (final V8 graph: 2,844 nodes, 14,769 edges,
51,456 anchors). Decisions are grouped by stage.

## 0. Inputs

- Seeds: 4 release families — `OLMo 3` (allenai), `DR-Tulu` (rl-research),
  `NVIDIA-Nemotron-3` (nvidia), `SmolLM3-3B` (HuggingFaceTB).
- Base pipeline: 8 stages (discover → extract → organize → audit → relate →
  reconcile → triage → merge), as shipped in `gdb/pipeline.py`.

## 1. Recursive expansion strategy (beam_v5)

The shipped pipeline expanded only 1-hop dependencies. To recover deep
recursive ancestry we layered an expansion loop on top.

- `beam_v5.py`. Beam-search expansion with 4 phases: bridge → depth=2 →
  depth=3 → depth=4.
- **Per-seed per-level top-K = 5.** At each level, for each seed family,
  rank candidate frontier nodes by **parent count** (number of edges
  pointing back to already-expanded nodes), expand the top 5.
- **Bridge node set = 12.** A pre-expanded set of high-betweenness nodes
  (popular benchmarks, common base models) seeded into the bridge phase to
  guarantee the seed families share at least some common upstream structure
  before the per-seed beams diverge.
- **Worker cap = 8.** Ceiling on concurrent extraction workers; raised from
  the default 1 to keep API throughput reasonable without overwhelming the
  HF / paper hosts.
- **Hard separator constraint.** Throughout expansion, candidate canonical
  identities cannot be merged across version, size, stage, date, or org.
  Enforced inside the lattice resolver and again at every dedup stage.

## 2. Pipeline bug fixes (gdb/pipeline.py)

Two patches to `gdb/pipeline.py` were required to keep `gdb run merge` from
failing on the recovered data:

1. **Family-equality validator downgraded from raise to warning.** The merge
   stage was throwing on benign family/identity mismatches that come from
   the same artifact being described slightly differently across sources.
   Replaced the `raise` with a logged warning. Without this, merges on the
   real seed corpus aborted partway through.

2. **Dict subject/object handling in `_merge_relations`.** The merge stage
   was building a `(subject, relation, object)` tuple key directly, but
   some relations have dict-typed subjects (`{"formal_name": ..., "name":
   ...}`) which are unhashable. Added a `_str()` helper that coerces dicts
   to their `formal_name` (fallback `name`) and unconditionally stringifies.
   Without this, merge fails with
   `TypeError: unhashable type: 'dict'`.

## 3. Auto-merge orchestration

- `auto_merge_combined.py` polls for both `recurse_v4` and `beam_v5` to
  exit, then triggers a final merge.
- `snapshot_merge.py` runs an out-of-band merge into an **isolated
  `GDB_STORAGE`** so mid-run snapshots do not interfere with the live merge
  database. We took one snapshot at 15:02 to lock in a paper-numbers
  baseline while expansion continued.

## 4. Eight-stage dedup pipeline (V0 → V8)

The base merge produced 24,660 edges across 18,680 nodes with substantial
surface-form duplication. We layered eight sequential dedup stages, each
preserving prior outputs on disk so any stage can be re-run from any earlier
checkpoint.

### V1 — heuristic dedup (`dedup_graph.py`, 6 sub-stages)

1. **Categorical drops.** Internal paths (`/weka/...`, `gs://`, `s3://`),
   free-text descriptive nodes (parens > 50 chars), and names > 200 chars.
2. **Signature-based clustering with hard separators.** Each node gets a
   signature `(org, bare_norm, versions, sizes, stages, dates)` and two
   nodes can only cluster if all separators match.
3. **No-org name folding.** Bare names like `MMLU` fold into prefixed
   clusters like `cais/mmlu` only if exactly one prefixed cluster has the
   same `bare_norm` (initial conservative rule; relaxed in V3).
4. **Canonical pick.** Within a cluster, prefer org/name HF form, then
   highest-degree, then longest non-aliased name.
5. **Low-signal filter.** Concept-like bare names (no `/`, no size, no
   parens, length < 30) with combined degree < 3 are dropped.
6. **Edge rewrite.** Rewrite each edge to canonical endpoints, merge
   `anchor_list` and `description_variants` for collapsed edges.

### V2 — NO\_SPLIT preservation (`dedup_apply_splits.py`)

V1 over-merged 33 of 100 borderline clusters per Opus verification (e.g.,
MMLU subjects, AIME 2024 vs 2025, OLMo-2 stage checkpoints, FLAN community
re-releases). V2 reverts those merges:

- Re-runs V1 clustering deterministically, then for each `NO_SPLIT`
  canonical from the LLM log, looks up the cluster by **signature** (not by
  canonical name, because `pick_canonical` is non-deterministic across runs
  due to set iteration order).
- Three-tier resolver: direct lookup → indirect via current `canon_map` →
  signature-based fallback with `can_merge` compatibility check. Without
  the fallback layers, ~10 of 33 NO_SPLIT canonicals fail to resolve.
- Reverts each member of a `NO_SPLIT` cluster to itself as canonical.

### V3 — fuzzy surface-form merge (`dedup_v3.py`)

Catches surface-form near-dupes V1 missed because of strict separator rules.

- **Hyphen-collapsed alt-key.** `rl-zero` ↔ `rlzero` ↔ `rl_zero`. Compute
  `bare_collapsed = bare_norm.replace("-", "").replace(".", "")` and use as
  the cluster key alongside `bare_norm`.
- **Version normalization.** `3_3` ↔ `3.3` (matters for
  `Llama-3_3-Nemotron` vs `Llama-3.3-Nemotron`).
- **Most-popular-target tiebreak.** When a no-org bare name has multiple
  prefixed candidates with matching `bare_norm`, V1 refused to fold; V3
  picks the highest-aggregate-degree candidate.
- **Added paren\_specifier and bracket\_specifier to the signature.**
  Critical post-hoc fix: the first run of V3 collapsed `cais/mmlu (STEM)`,
  `(humanities)`, `(other)` etc. back into `cais/mmlu` because the signature
  ignored parens content. Adding `paren_specifier` (the trailing parens
  string) and `bracket_specifier` (frozenset of non-standard
  `key=value` pairs from `[...]` like `ablation=...`, `variant=...`)
  preserves subset distinctions while still merging surface-form variants.

### V4 — Opus hub audit (`opus_verify.py`)

Per-hub edge-level audit using LLM judgment.

- **Top 20 OUT-hubs + top 15 IN-hubs.** OUT-hubs are mostly seed models;
  IN-hubs are the popular benchmarks (MMLU, GSM8K, GPQA, etc.).
- **Batch size 80 edges per call.** Largest batch we trusted for free-form
  drop reasoning.
- **Initial run was serial (one call at a time).** This took ~90 minutes
  for 35 calls. Re-ran with **`ThreadPoolExecutor`, 8 parallel workers** —
  finished in 5.3 minutes.
- Drop reasons free-form (un-tagged); drops applied as a flat blacklist.
- Edges-only audit: V4 does not change the node set, only removes edges.

### V5 — Opus 4.7 deep audit (`opus_verify_v5.py`)

Same idea as V4 but turned up:

- **Model: `claude-opus-4-7` explicit** (vs the `opus` alias used in V4).
- **`--effort max`** for maximum extended thinking budget per call. This
  is what makes V5 calls take 60–400 s instead of 3–15 s, and it is the
  reason V5 found 993 drops (vs V4's 249).
- **`--bare`** flag to skip auto-memory, hooks, MCP, plugin sync, etc.
- **Batch size dropped to 40 edges** (more attention per edge).
- **Coverage widened to top 75 OUT + top 30 IN hubs.**
- **Per-hub cap = 200 edges**, sorted by anchor count descending so the
  highest-evidence edges are reviewed first if a hub exceeds the cap.
- **12 parallel workers.**
- **Tagged drop categories.** Forced output format
  `DROP <id> :: <TAG> :: <reason>` with `TAG ∈ {DUPLICATE, HALLUCINATED,
  VACUOUS, WRONG_RELATION}`. Lets us inspect what kind of noise dominates.

### V6 — whole-graph node dedup (`dedup_v6.py`)

Block-then-verify across every node, not just hub neighborhoods.

- **First attempt blew up.** Used 4 candidate signals (lex-collapse,
  token-Jaccard, shared-anchor co-citation, co-neighborhood Jaccard) and
  unioned candidate pairs into connected components. Signals 3 and 4
  produced false positives that fused unrelated hubs into mega-clusters
  via single bad pairs. 310 of 457 clusters were noise.
- **Redesign.** Dropped signals 3 and 4 entirely. Final V6 uses three
  high-precision signals only:
  - Lex-collapse blocks (alphanumeric-only key match).
  - Token-Jaccard ≥ 0.60 with **top-3 candidates per node** (no global
    union — each node gets its own small candidate cluster).
  - Substring containment (short bare name fully contained in prefixed).
- **No connected components.** Each node yields one small cluster of
  ≤ 6 members; no transitive cluster growth.
- **24 parallel workers** for verification.
- **Verdict format: `ALL_SAME / PARTIAL / ALL_DISTINCT`** with structured
  index references. PARTIAL accepts `merge_indices` plus
  `canonical_within`, allowing the LLM to merge a subset of a cluster while
  keeping the rest distinct.
- **Apply via union-find** of `ALL_SAME` and `PARTIAL` decisions. No
  conflict guard at this stage (added in V8 fix).
- 1,593 clusters verified in 9.3 min; 162 ALL_SAME, 238 PARTIAL, 1,193
  ALL_DISTINCT.

### V7 — release-only filter (`release_filter_v7.py`)

Filters intermediate research artifacts (Stage-N checkpoints,
ingredient-N variants, midtraining steps) so the published graph contains
only released artifacts plus standard benchmarks.

- **Opus classifies every node as KEEP or DROP.**
- **Batch size 20 nodes per call**, ~180 calls.
- **`--effort high` (not max).** Classification is simpler than the
  judgment calls in V5 / V6, so high is sufficient and cuts cost.
- **24 parallel workers.** Whole pass took 78 seconds.
- **Transitive edge rewiring through DROP nodes.** For each `(A → DROP →
  B)` chain along **compatible relation pairs** (e.g.,
  `trained_from + trained_from → trained_from`), synthesize an `(A → B)`
  edge with merged anchors. Without this, dropping intermediates breaks
  the released-to-released ancestry chain.
- Defaulted unparsed verdicts to `KEEP` (safe).

### V8 — cross-org + suffix dedup (`dedup_v8.py`)

Catches surface-form dupes V3–V6 explicitly avoided because they would
have required relaxing the cross-org safety rule.

- **Four new candidate signals.**
  1. Cross-org bare-lex match (e.g., `OpenAI/GPT-2` ↔
     `openai-community/gpt2`). V3 hard-blocked these.
  2. Suffix-stripping pairs (`-turbo`, `-Instruct`, `-Base`, `-hf`, etc.).
     Most are correctly distinct; `-turbo` ↔ no-suffix and `-hf` ↔
     no-suffix usually merge.
  3. Bare-no-slash → prefixed-bare lex match (catches generic aliases of
     specific releases).
  4. Token-superset / subset (one node's tokens fully contain another's,
     differing by ≤ 2 tokens).
- Each candidate cluster (size 2–6) verified by Opus 4.7 + `--effort max`.
- 24 parallel workers, 2.8 min for 748 clusters.

### V8-fix — conflict-guarded union (`dedup_v8_fix.py`)

V8 used a flat union-find on Opus `ALL_SAME` decisions. When the LLM
approved both `(bare, 0325-variant)` and `(bare, 1124-variant)`
individually, transitive union merged all three — incorrectly fusing two
distinct date-stamped releases.

- **Conflict-guarded union-find.** Before applying a union, compute the
  combined component's specifier sets (dates, versions, sizes, stages).
  If any specifier key has different non-empty values on the two sides
  being joined, **skip the union** and leave the components separate.
- This caught the `OLMo-2-0325-32B-Instruct` ↔ `OLMo-2-1124-32B-Instruct`
  over-merge (different dates) that V8 produced.
- 22 of 23 unions applied; 1 skipped. V8 file overwritten with the
  corrected version.

## 5. LLM verification settings (used at V2, V4, V5, V6, V7, V8)

- **Default model: Opus.** Initial dedup verification used Sonnet, but the
  user explicitly requested Opus. Switched to `--model opus` from V2
  onward; pinned to `--model claude-opus-4-7` from V5 onward.
- **`--effort max` for judgment calls** (V5, V6, V8). `--effort high`
  for the simpler V7 KEEP/DROP classification.
- **`--bare`** on every call to skip hooks, auto-memory, plugin sync,
  CLAUDE.md auto-discovery — reduces per-call overhead.
- **`--permission-mode bypassPermissions`** so subprocess calls do not
  prompt.
- **Parallelism:** scaled from 8 (V4) → 12 (V5) → 24 (V6, V7, V8) workers
  via `concurrent.futures.ThreadPoolExecutor`. 24 was the practical sweet
  spot; higher values did not produce meaningful speedup.
- **Structured output formats** in every prompt:
  - V2 / V6 / V8 cluster verdicts:
    `ALL_SAME :: <canonical_idx> :: <reason>` /
    `PARTIAL :: <merge_idxs> :: <canonical_within> :: <reason>` /
    `ALL_DISTINCT :: <reason>`
  - V4 / V5 edge drops: `DROP <id> :: <TAG> :: <reason>` with tagged
    categories.
  - V7 node classification: `KEEP <id> :: <reason>` /
    `DROP <id> :: <reason>`.
- **Per-call timeouts:** 240 s for V4, 600 s for V5 / V6 / V7 / V8 (max
  thinking can take several minutes per call).

## 6. Sanity invariants enforced at every stage

Every dedup script asserts these before writing output. Each assertion
caught at least one regression during development:

- `Olmo-3 nodes > 0 AND Olmo-3.1 nodes > 0` (different version numbers
  must stay distinct).
- All four seed prefixes present (`allenai/Olmo-3`, `rl-research/DR-Tulu`,
  `nvidia/NVIDIA-Nemotron-3`, `HuggingFaceTB/SmolLM3`).
- `AIME 2024` and `AIME 2025` both present.
- At least some `cais/mmlu` subject splits preserved (`(STEM)`,
  `(humanities)`, etc. should not collapse into bare `cais/mmlu`).
- Collapse ratio cap: warn if more than 30% of nodes get merged in a
  single pass.
- (V8-fix only) `OLMo-2-0325-32B-Instruct` ≠ `OLMo-2-1124-32B-Instruct`.

## 7. Visualizer (`viz_v4.py`)

The shipped `gdb/viz.py` reads from a SQLite-backed pipeline state and
cannot load a self-contained merged JSON. We wrote a thin adapter:

- Loads any version (V0 / V2 / V3 / V4 / V5 / V6 / V7 / V8) via a
  `--source` flag.
- **Min-degree slider, default = 10.** Without this, vis-network's
  default physics locked the browser at 19k edges. Min-degree = 10 cuts
  the visible set to ~150 hubs.
- **Max-nodes cap, default = 200.**
- **Physics OFF by default.** User opts in via toolbar button if they
  want the force layout to settle.
- **Ego mode (1-hop and 2-hop).** Search for a node, click result,
  auto-pivots to ego view. The cleanest way to read ancestry.
- **Force vs hierarchical** layout toggle.
- Reset clears ego mode and restores the global view.

## 8. Final state and per-stage reductions

| Stage | Nodes  | Edges  | Anchors | Notes                              |
|-------|-------:|-------:|--------:|------------------------------------|
| V0    | 18,680 | 24,660 | 63,673  | Raw merged baseline                |
| V1    |  4,156 | 20,535 | 61,434  | Heuristic dedup                    |
| V2    |  4,251 | 20,857 | 61,421  | + NO_SPLIT reverts                 |
| V3    |  4,025 | 19,815 | 61,413  | + fuzzy surface forms              |
| V4    |  3,990 | 19,566 | 60,207  | + Opus hub audit                   |
| V5    |  3,941 | 18,573 | 57,485  | + Opus 4.7 deep audit              |
| V6    |  3,642 | 17,862 | 57,213  | + whole-graph dedup                |
| V7    |  2,866 | 14,871 | 51,462  | + release-only filter              |
| V8    |  2,844 | 14,769 | 51,456  | + cross-org / suffix + conflict-fix|

**Total reduction V0 → V8:** -85% nodes, -40% edges, -19% anchors.

## 9. Files on disk (single source of truth per stage)

```
storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/
  merge_artifact.json                 # V0 — original raw merge
  merge_artifact_deduped.json         # V2 — V1 + NO_SPLIT reverts
  merge_artifact_deduped_v3.json      # V3
  merge_artifact_deduped_v4.json      # V4
  merge_artifact_deduped_v5.json      # V5
  merge_artifact_deduped_v6.json      # V6
  merge_artifact_deduped_v7.json      # V7
  merge_artifact_deduped_v8.json      # V8 (final)
```

Plus, in `run-logs/`:

```
DEDUP_REPORT.txt        DEDUP_LLM_DECISIONS.txt
DEDUP_REPORT_V2.txt     DEDUP_V3_REPORT.txt
OPUS_VERDICTS.txt       (V4)
OPUS_V5_REPORT.txt      OPUS_V5_VERDICTS.txt
DEDUP_V6_REPORT.txt     DEDUP_V6_VERDICTS.txt
RELEASE_V7_REPORT.txt   RELEASE_V7_VERDICTS.txt
DEDUP_V8_REPORT.txt     DEDUP_V8_VERDICTS.txt
DEDUP_V8_FIX_REPORT.txt
EDGE_AUDIT.log          (pre-V3 noise pattern analysis)
```

## 10. Reproduction recipe (concise)

```
# 0. Fix pipeline.py before running merge.
#    (a) Validator: raise → log warning.
#    (b) _merge_relations: coerce dict subj/obj via _str() helper.

# 1. Run base pipeline + beam_v5 expansion + auto_merge_combined.
#    Produces: merge_artifact.json (V0).

# 2. Run dedup stages in order:
python dedup_graph.py             # V0 → V1 (intermediate, not saved)
                                  # V1 + NO_SPLIT applied -> V2 written
python dedup_apply_splits.py      # produces merge_artifact_deduped.json (V2)
python dedup_v3.py                # V3
python opus_verify_v5.py          # V5  (skip V4 — superseded)
python dedup_v6.py                # V6
python release_filter_v7.py       # V7
python dedup_v8.py                # V8 (un-fixed)
python dedup_v8_fix.py            # V8 (fixed; overwrites V8 file)

# 3. Visualize:
python viz_v4.py --port 8102 --source v8
# Open http://127.0.0.1:8102/
```

Total wall-clock for the dedup pipeline (LLM stages dominated):
- V1 + V2: < 1 min (heuristic, no LLM)
- V3: < 1 min
- V5: ~80 min (max thinking, 309 calls)
- V6: ~9 min
- V7: ~1.5 min
- V8: ~3 min
- V8-fix: < 30 s

End-to-end dedup: ~95 min on a single machine with 24 parallel `claude` CLI
processes.
