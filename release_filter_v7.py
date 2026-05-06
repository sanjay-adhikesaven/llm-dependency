#!/usr/bin/env python3
"""V7 release-only filter — Opus classifies every node.

Each node → KEEP (officially released artifact / standard benchmark) or DROP
(intermediate research metadata / unreleased checkpoint / concept alias).

After classification:
  - All DROP nodes' edges are removed
  - Transitive rewiring: for each (A → DROP → B) chain along compatible relations
    (trained_from chains stay connected), synthesize A → B with merged anchors

Output: merge_artifact_deduped_v7.json
"""
from __future__ import annotations
import json, re, subprocess, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PARALLEL_WORKERS = 24
BATCH_SIZE = 20             # nodes per Opus call
MODEL = "claude-opus-4-7"
EFFORT = "high"             # high (not max) — classification is simpler than V5/V6 reasoning

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
RUN_DIR = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d"
V6_IN = RUN_DIR / "merge_artifact_deduped_v6.json"
V7_OUT = RUN_DIR / "merge_artifact_deduped_v7.json"
REPORT = REPO / "run-logs/RELEASE_V7_REPORT.txt"
VERDICTS = REPO / "run-logs/RELEASE_V7_VERDICTS.txt"

REPORT.write_text(""); VERDICTS.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

# ============================================================
G = json.loads(V6_IN.read_text())
edges = G["relations"]
groups = G["lattice"]["groups"]

all_nodes = set()
for e in edges:
    if e.get("subject"): all_nodes.add(e["subject"])
    if e.get("object"):  all_nodes.add(e["object"])
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name");  fn and all_nodes.add(fn)
all_nodes = sorted({n for n in all_nodes if isinstance(n, str) and n})

degree = Counter()
for e in edges: degree[e["subject"]] += 1; degree[e["object"]] += 1

# Pre-index a sample anchor URL per node for context
def anchor_key(a): return a.get("source") or a.get("url") or a.get("path") or ""
sample_anchor = {}
for e in edges:
    for n in (e.get("subject"), e.get("object")):
        if n and n not in sample_anchor:
            anchors = e.get("anchor_list") or []
            if anchors:
                sample_anchor[n] = anchor_key(anchors[0])[:120]

report("="*70)
report(f"V7 RELEASE-ONLY FILTER  (model: {MODEL} effort: {EFFORT})")
report("="*70)
report(f"Source: {V6_IN.name}")
report(f"Nodes: {len(all_nodes):,}  ·  Edges: {len(edges):,}\n")

# ============================================================
# Build batches
# ============================================================
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

batches = list(chunked(all_nodes, BATCH_SIZE))
report(f"Batches: {len(batches)} ({BATCH_SIZE} nodes each)\n")

# ============================================================
# Prompt
# ============================================================
PROMPT = """You are filtering a graph of LLM dependencies down to OFFICIALLY RELEASED artifacts only. For each node listed below, classify it.

KEEP — the node is one of:
  - An officially released model checkpoint (HuggingFace org/name format with public weights, e.g., "allenai/Olmo-3-7B-Instruct", "openai/gpt-4")
  - An officially released dataset (e.g., "cais/mmlu", "openai/gsm8k", "Common Crawl")
  - A standard benchmark/evaluation suite (e.g., "MMLU", "GSM8K", "AIME 2024", "HumanEval", "BBH", "Hellaswag")
  - A well-known third-party API model ("openai/gpt-4o", "anthropic/claude-sonnet-4")
  - An organization name ONLY if it's a well-known concrete artifact provider used as a model identifier

DROP — the node is one of:
  - Training-stage checkpoint references ("Olmo 3 32B Stage 2", "Stage 2 Ingredient 1", "Stage 2 Soup", "midtraining run 2")
  - Long-context extension or pretraining checkpoints at specific steps ("Olmo 3 32B long-context extension checkpoint at step 10000")
  - Internal research data variants / preference-mix deltas ("allenai/olmo-3-pref-mix-deltas-...")
  - Bracket-tagged research metadata when a released sibling exists ("OLMo 3 [size=7B, stage=RL-Zero, domain=mix]" — drop, since "allenai/Olmo-3-7B-RL-Zero-Mix" is the released form)
  - Generic concept aliases ("Safety", "GPT", "olmo 3", "Llama" without size/version)
  - Experimental/preview/distill-student variants when not actually released
  - Off-lattice prose descriptions ("Stem-heavy crawl", "synthetic agentic data")

For each item, output exactly one line:
  KEEP <id> :: <one-phrase reason>
  DROP <id> :: <one-phrase reason>

NODES TO CLASSIFY ({n_items}):
{node_list}
"""

def build_node_list(start_id, members):
    lines = []
    for i, n in enumerate(members):
        global_id = start_id + i
        d = degree.get(n, 0)
        a = sample_anchor.get(n, "(no anchor)")
        lines.append(f"  [{global_id}] {n}    (degree={d}, sample_anchor={a})")
    return "\n".join(lines)

def call_opus(prompt):
    try:
        r = subprocess.run(
            ["claude", "-p", prompt,
             "--model", MODEL, "--effort", EFFORT, "--bare",
             "--output-format", "text", "--permission-mode", "bypassPermissions"],
            capture_output=True, text=True, timeout=600,
        )
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return f"ERR: {e!r}", -1

# ============================================================
# Run classification in parallel
# ============================================================
classifications = {}     # node_name → (verdict, reason)
results_lock = Lock()
log_lock = Lock()

def parse_classifications(text, batch_start, batch_members):
    decisions = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(KEEP|DROP)\s+(\d+)\s*::\s*(.*)$", line)
        if not m: continue
        verdict = m.group(1)
        gid = int(m.group(2))
        reason = m.group(3).strip()
        # Map global id back to node name
        idx = gid - batch_start
        if 0 <= idx < len(batch_members):
            decisions[batch_members[idx]] = (verdict, reason)
    return decisions

def classify_batch(batch_idx, members, start_id):
    prompt = PROMPT.format(n_items=len(members), node_list=build_node_list(start_id, members))
    t0 = time.time()
    out, rc = call_opus(prompt)
    dt = time.time() - t0
    decisions = parse_classifications(out, start_id, members)
    # Default any unparsed nodes to KEEP (safe default)
    for n in members:
        if n not in decisions:
            decisions[n] = ("KEEP", "(no verdict — default keep for safety)")
    with results_lock:
        classifications.update(decisions)
    with log_lock:
        with open(VERDICTS, "a") as f:
            f.write(f"\n{'='*70}\n[batch {batch_idx}] start_id={start_id} n={len(members)} rc={rc} dt={dt:.1f}s\n")
            f.write(f"{'='*70}\n{out}\n")
    return batch_idx, len(decisions), dt

print(f"\nLaunching {len(batches)} classification calls with {PARALLEL_WORKERS} workers...\n")
t_start = time.time()
done = 0
with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futs = {}
    for bi, batch in enumerate(batches):
        start_id = bi * BATCH_SIZE
        futs[ex.submit(classify_batch, bi, batch, start_id)] = bi
    for fut in as_completed(futs):
        try:
            bi, n_dec, dt = fut.result()
            done += 1
            if done % 10 == 0 or done == len(batches):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(batches) - done) / rate if rate > 0 else 0
                print(f"  [{done:3d}/{len(batches)}] dt={dt:>5.0f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
        except Exception as e:
            print(f"  batch {futs[fut]} FAILED: {e!r}")

t_total = time.time() - t_start
report(f"\nClassification done in {t_total:.0f}s ({t_total/60:.1f} min)")
report(f"Nodes classified: {len(classifications):,} / {len(all_nodes):,}")

verdict_counts = Counter(v for v, _ in classifications.values())
report(f"\nVerdict counts:")
for k, v in verdict_counts.most_common():
    report(f"  {k}: {v}")

# ============================================================
# Sample drops for sanity
# ============================================================
report("\nSample DROP verdicts (10 random):")
import random; random.seed(7)
drops = [(n, classifications[n][1]) for n in all_nodes if classifications[n][0] == "DROP"]
for n, reason in random.sample(drops, min(15, len(drops))):
    report(f"  [d={degree[n]:>4}] {n[:80]}")
    report(f"          → {reason[:120]}")

report("\nSample KEEP verdicts on potentially-borderline (small degree):")
keep_lowdeg = [(n, classifications[n][1]) for n in all_nodes
               if classifications[n][0] == "KEEP" and degree[n] <= 3]
for n, reason in random.sample(keep_lowdeg, min(10, len(keep_lowdeg))):
    report(f"  [d={degree[n]:>4}] {n[:80]}")
    report(f"          → {reason[:120]}")

# ============================================================
# Apply: drop nodes + rewire compatible chains
# ============================================================
keep_nodes = {n for n in all_nodes if classifications.get(n, ("KEEP",))[0] == "KEEP"}
drop_nodes = set(all_nodes) - keep_nodes
report(f"\nKEEP: {len(keep_nodes):,}  ·  DROP: {len(drop_nodes):,}")

# Build adjacency from DROP nodes for rewiring
edges_in  = defaultdict(list)   # node → list of (subject, edge_idx) in
edges_out = defaultdict(list)   # node → list of (object,  edge_idx) out
for ei, e in enumerate(edges):
    s, o = e.get("subject"), e.get("object")
    if s and o:
        edges_in[o].append((s, ei))
        edges_out[s].append((o, ei))

# Compatible relations for transitive rewiring
COMPATIBLE_PAIRS = {
    ("trained_from", "trained_from"): "trained_from",
    ("trained_from", "merged_from"):  "trained_from",
    ("merged_from",  "trained_from"): "trained_from",
    ("trained_on",   "trained_on"):   "trained_on",      # rare but valid (e.g. dataset chain)
    ("filtered_by",  "filtered_by"):  "filtered_by",
}

# Rewire: for each DROP node X, for every (A→X) and (X→B) where both A and B are KEEP,
# add (A→B) if relations are compatible.
rewired_edges = []   # list of new edge dicts to add
seen_rewire_keys = set()
for x in drop_nodes:
    for s, ei_in in edges_in.get(x, []):
        if s in drop_nodes: continue
        e_in = edges[ei_in]
        for o, ei_out in edges_out.get(x, []):
            if o in drop_nodes: continue
            e_out = edges[ei_out]
            r_in = e_in.get("relation"); r_out = e_out.get("relation")
            new_rel = COMPATIBLE_PAIRS.get((r_in, r_out))
            if not new_rel: continue
            key = (s, new_rel, o)
            if key in seen_rewire_keys: continue
            seen_rewire_keys.add(key)
            merged_anchors = (e_in.get("anchor_list") or []) + (e_out.get("anchor_list") or [])
            rewired_edges.append({
                "subject": s, "relation": new_rel, "object": o,
                "dependency_kind": "indirect",
                "description": f"(rewired through {x})",
                "anchor_list": merged_anchors[:6],   # cap to keep size sane
                "description_variants": [],
            })

report(f"Rewired edges synthesized (transitive through DROPs): {len(rewired_edges):,}")

# ============================================================
# Build new edge set
# ============================================================
new_edges = {}
for e in edges:
    s, o = e.get("subject"), e.get("object")
    if s in keep_nodes and o in keep_nodes:
        rel = e.get("relation", "")
        if not rel: continue
        key = (s, rel, o)
        if key not in new_edges:
            new_edges[key] = {**e, "anchor_list": list(e.get("anchor_list") or []),
                              "description_variants": list(e.get("description_variants") or [])}
        else:
            new_edges[key]["anchor_list"].extend(e.get("anchor_list") or [])
            for v in (e.get("description_variants") or []):
                if v not in new_edges[key]["description_variants"]:
                    new_edges[key]["description_variants"].append(v)

# Add rewired edges only if not already present
for re_edge in rewired_edges:
    key = (re_edge["subject"], re_edge["relation"], re_edge["object"])
    if key not in new_edges:
        new_edges[key] = re_edge

report(f"Edges before V7: {len(edges):,}")
report(f"Edges after V7:  {len(new_edges):,}")

# ============================================================
# Sanity invariants
# ============================================================
final_node_set = set()
for k in new_edges:
    final_node_set.add(k[0]); final_node_set.add(k[2])

olmo3 = [n for n in final_node_set if "olmo-3-" in n.lower().replace(" ","-") and "3.1" not in n.lower()]
olmo31 = [n for n in final_node_set if "olmo-3.1" in n.lower()]
report(f"\nSanity:")
report(f"  Olmo-3 nodes: {len(olmo3)}")
report(f"  Olmo-3.1 nodes: {len(olmo31)}")
assert len(olmo3) > 0 and len(olmo31) > 0, "ABORT: Olmo-3/3.1 distinction collapsed"

SEEDS = ["allenai/Olmo-3", "rl-research/DR-Tulu", "nvidia/NVIDIA-Nemotron-3", "HuggingFaceTB/SmolLM3"]
for s in SEEDS:
    found = any(s in n for n in final_node_set)
    report(f"  Seed '{s}': {'present' if found else 'MISSING'}")
    assert found, f"ABORT: seed {s} missing"

aime_2024 = any("aime" in n.lower() and "2024" in n for n in final_node_set)
aime_2025 = any("aime" in n.lower() and "2025" in n for n in final_node_set)
report(f"  AIME 2024 present: {aime_2024}")
report(f"  AIME 2025 present: {aime_2025}")

# ============================================================
# Build groups, write output
# ============================================================
new_groups = defaultdict(list)
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name", "")
        if not fn: continue
        if fn in keep_nodes and fn in final_node_set:
            new_groups[fn].append(it)
out_groups = []
for canon, its in new_groups.items():
    primary = next((i for i in its if i.get("formal_name") == canon), its[0])
    primary = dict(primary); primary["formal_name"] = canon; primary["alias_count"] = len(its)
    out_groups.append({"items": [primary], "id": canon})

OUT = {
    "lattice": {"groups": out_groups},
    "relations": list(new_edges.values()),
    "conflicts": G.get("conflicts", []),
    "sources": G.get("sources", []),
    "relations_sources": G.get("relations_sources", []),
    "dedup_metadata": {
        **G.get("dedup_metadata", {}),
        "v7_classified": len(classifications),
        "v7_kept": len(keep_nodes),
        "v7_dropped": len(drop_nodes),
        "v7_rewired_edges": len(rewired_edges),
        "v7_model": MODEL, "v7_effort": EFFORT,
    },
}
V7_OUT.write_text(json.dumps(OUT))

final_anchors = sum(len(e.get("anchor_list", []) or []) for e in new_edges.values())
report(f"\n{'='*70}\nV7 FINAL\n{'='*70}")
report(f"  Distinct nodes: {len(final_node_set):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(new_edges):,}")
report(f"  Anchors: {final_anchors:,}")
final_rels = Counter(e.get("relation","") for e in new_edges.values())
report(f"\n  Top relations:")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")

# Olmo-3 audit list
report(f"\nOlmo-3 nodes after V7 (sorted by degree):")
deg_v7 = Counter()
for e in new_edges.values():
    deg_v7[e["subject"]] += 1; deg_v7[e["object"]] += 1
olmo3_sorted = sorted(olmo3, key=lambda n: -deg_v7[n])
for n in olmo3_sorted:
    report(f"  [d={deg_v7[n]:>4}] {n}")

report(f"\n✓ Wrote {V7_OUT} ({V7_OUT.stat().st_size:,} bytes)")
