#!/usr/bin/env python3
"""V5 deep audit — Opus 4.7 with max thinking, smaller batches, wider coverage.

Differences from V4:
  - Model: claude-opus-4-7 explicit (vs alias 'opus')
  - --effort max → maximum extended thinking budget per call
  - Batch size: 40 edges per call (vs 80) → more attention per edge
  - Coverage: top 75 OUT-hubs + top 30 IN-hubs (vs 20+15)
  - Hubs with >40 edges are SPLIT into multiple batches
  - 12 parallel workers (vs 8)
  - --bare flag reduces per-call overhead

Output: merge_artifact_deduped_v5.json (preserves v4)
"""
from __future__ import annotations
import json, re, subprocess, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PARALLEL_WORKERS = 12
BATCH_SIZE = 40
TOP_OUT = 75
TOP_IN = 30
PER_HUB_CAP = 200   # don't audit more than this many edges per single hub
MODEL = "claude-opus-4-7"

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
V4_IN = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact_deduped_v4.json"
V5_OUT = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact_deduped_v5.json"
VERDICTS_LOG = REPO / "run-logs/OPUS_V5_VERDICTS.txt"
REPORT = REPO / "run-logs/OPUS_V5_REPORT.txt"

REPORT.write_text(""); VERDICTS_LOG.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

# ============================================================
G = json.loads(V4_IN.read_text())
edges = G["relations"]
groups = G["lattice"]["groups"]

report("="*70)
report(f"OPUS V5 DEEP AUDIT — {MODEL} --effort max")
report("="*70)
report(f"V4 input: {V4_IN.name}  ({len(edges):,} edges)")
report(f"Coverage: top {TOP_OUT} OUT-hubs + top {TOP_IN} IN-hubs")
report(f"Batch size: {BATCH_SIZE} edges  ·  workers: {PARALLEL_WORKERS}")
report(f"Per-hub cap: {PER_HUB_CAP} edges (audit highest-anchor first)\n")

out_deg, in_deg = Counter(), Counter()
for e in edges:
    out_deg[e["subject"]] += 1
    in_deg[e["object"]] += 1

edges_by_subject = defaultdict(list)
edges_by_object = defaultdict(list)
for i, e in enumerate(edges):
    edges_by_subject[e["subject"]].append(i)
    edges_by_object[e["object"]].append(i)

# Sort each hub's edges by anchor count (descending) so highest-evidence edges
# are audited first if we cap.
def sort_by_anchors(idx_list):
    return sorted(idx_list, key=lambda i: -len(edges[i].get("anchor_list", []) or []))

out_hubs = [(n, d) for n, d in out_deg.most_common(TOP_OUT)]
in_hubs  = [(n, d) for n, d in in_deg.most_common(TOP_IN)]

# ============================================================
# Build batched jobs
# ============================================================
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

jobs = []  # (hub, role, idx_list, label)
for hub_i, (hub, deg) in enumerate(out_hubs):
    idxs = sort_by_anchors(edges_by_subject[hub])[:PER_HUB_CAP]
    for batch_i, batch in enumerate(chunked(idxs, BATCH_SIZE)):
        label = f"OUT[{hub_i+1:2d}/{TOP_OUT}].b{batch_i+1}"
        jobs.append((hub, "subject", batch, label))
for hub_i, (hub, deg) in enumerate(in_hubs):
    idxs = sort_by_anchors(edges_by_object[hub])[:PER_HUB_CAP]
    for batch_i, batch in enumerate(chunked(idxs, BATCH_SIZE)):
        label = f"IN[{hub_i+1:2d}/{TOP_IN}].b{batch_i+1}"
        jobs.append((hub, "object", batch, label))

report(f"Total Opus calls planned: {len(jobs)}")
out_calls = sum(1 for _, r, _, _ in jobs if r == "subject")
in_calls = len(jobs) - out_calls
report(f"  OUT-hub calls: {out_calls}")
report(f"  IN-hub calls:  {in_calls}\n")

# ============================================================
# Subgraph text + prompt
# ============================================================
def edge_summary(idx):
    e = edges[idx]
    anchors = e.get("anchor_list", []) or []
    first_anchor = ""
    if anchors:
        a = anchors[0]
        first_anchor = (a.get("path") or a.get("source") or a.get("url") or "")[:140]
    return {
        "id": idx,
        "rel": e["relation"],
        "obj": e["object"], "subj": e["subject"],
        "n_anchors": len(anchors),
        "first_anchor": first_anchor,
        "desc": (e.get("description") or "")[:180],
    }

def build_subgraph_text(hub, role, idx_list, batch_label):
    direction = "outgoing" if role == "subject" else "incoming"
    lines = [f"HUB ({role}-side): {hub}", f"BATCH: {batch_label}", f"EDGES_IN_BATCH: {len(idx_list)}", "",
             f"This is one batch of edges where {hub} appears as {role}."]
    lines.append("\nEDGES (id is the line ID, use it in your output):")
    for idx in idx_list:
        es = edge_summary(idx)
        if role == "subject":
            line = f"  [{idx:6d}] --[{es['rel']}]--> {es['obj']}"
        else:
            line = f"  [{idx:6d}] {es['subj']} --[{es['rel']}]-->"
        line += f"   anchors={es['n_anchors']}"
        if es["first_anchor"]:
            line += f"  src≈{es['first_anchor']}"
        if es["desc"]:
            line += f"\n           desc: {es['desc']}"
        lines.append(line)
    return "\n".join(lines)

PROMPT = """You are a careful graph quality reviewer. Below is one HUB node and a batch of {direction} edges.

Think carefully. Then identify edges that should be DROPPED for one of these reasons:

1. DUPLICATE — the same fact is asserted by another edge in this same batch via a different surface form (e.g., "MMLU" vs "cais/mmlu", "stack-edu" vs "HuggingFaceTB/stack-edu", aggregator-path vs canonical-leaf for the same dataset).

2. HALLUCINATED — the relationship is implausible or impossible given the artifacts involved (e.g., a model "trained_on" something released after it; a benchmark "trained_from" a model; a model "merged_from" datasets).

3. VACUOUS — implausibly generic objects with no information value: "data", "text corpus", "training data", "internet text", or unbracketed bare concepts that match nothing real.

4. WRONG_RELATION — the edge would be valid with a different relation but the relation chosen is wrong (e.g., a benchmark labeled `trained_on` when it's actually `used_for_evaluation`; a model labeled `generated_by` another model when it's actually `trained_from`).

DO NOT drop edges merely because:
- they have only 1 anchor (1 anchor is normal)
- they describe ablations / distinct stages / subset variants — those are valid distinct artifacts
- the org/name has an unfamiliar prefix

OUTPUT FORMAT — strict, one decision per line, nothing else:
  DROP {{id}} :: {{tag}} :: {{one-sentence reason}}
  KEEP_ALL  (only if every edge looks fine)

Where {{tag}} is one of: DUPLICATE, HALLUCINATED, VACUOUS, WRONG_RELATION.

SUBGRAPH:
{subgraph}
"""

# ============================================================
# Opus call (with thinking budget)
# ============================================================
def call_opus(prompt_text):
    try:
        result = subprocess.run(
            ["claude", "-p", prompt_text,
             "--model", MODEL,
             "--effort", "max",
             "--bare",
             "--output-format", "text",
             "--permission-mode", "bypassPermissions"],
            capture_output=True, text=True, timeout=600,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return f"ERR: {e!r}", "", -1

all_drops = {}     # edge_id → (tag, reason, hub_label)
drop_lock = Lock()
log_lock = Lock()

def parse_drops(text):
    drops = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^DROP\s+(\d+)\s*::\s*(\w+)\s*::\s*(.+)$", line)
        if m:
            drops.append((int(m.group(1)), m.group(2), m.group(3).strip()))
            continue
        # tolerate older format without tag
        m2 = re.match(r"^DROP\s+(\d+)\s*::\s*(.+)$", line)
        if m2:
            drops.append((int(m2.group(1)), "UNTAGGED", m2.group(2).strip()))
    return drops

def audit_batch(hub, role, idx_list, label):
    if not idx_list:
        return label, hub, role, [], "", 0.0
    direction = "outgoing" if role == "subject" else "incoming"
    sg = build_subgraph_text(hub, role, idx_list, label)
    prompt = PROMPT.format(direction=direction, subgraph=sg)
    t0 = time.time()
    out, err, rc = call_opus(prompt)
    dt = time.time() - t0
    drops = parse_drops(out)
    with drop_lock:
        for eid, tag, reason in drops:
            if 0 <= eid < len(edges):
                all_drops[eid] = (tag, reason, label)
    with log_lock:
        with open(VERDICTS_LOG, "a") as f:
            f.write(f"\n{'='*72}\n[{label}] {role}={hub}  n={len(idx_list)}  rc={rc}  {dt:.1f}s\n")
            f.write(f"{'='*72}\n{out}\n")
            if err:
                f.write(f"\n--STDERR--\n{err[:500]}\n")
    return label, hub, role, drops, out, dt

# ============================================================
# Run all jobs in parallel
# ============================================================
print(f"\nLaunching {len(jobs)} calls with {PARALLEL_WORKERS} workers...\n")
t_start = time.time()
completed = 0
with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futs = {ex.submit(audit_batch, h, r, i, l): l for h, r, i, l in jobs}
    for fut in as_completed(futs):
        try:
            label, hub, role, drops, _, dt = fut.result()
            completed += 1
            print(f"[{completed:3d}/{len(jobs)}] {label:<22} {hub[:42]:<42} {dt:>6.1f}s  → {len(drops)} drops")
        except Exception as e:
            print(f"FAILED {futs[fut]}: {e!r}")

t_total = time.time() - t_start
report(f"\nAll calls done in {t_total:.0f}s ({t_total/60:.1f} min)")
report(f"Total edges flagged for drop: {len(all_drops):,}\n")

# Drop tag distribution
tag_counts = Counter(t for t, _, _ in all_drops.values())
report("Drop categories:")
for t, n in tag_counts.most_common():
    report(f"  {n:>4}  {t}")

# Sample reasons per category
report("\nSample reasons by category:")
for tag in tag_counts:
    sample = [(eid, r) for eid, (t, r, _) in all_drops.items() if t == tag][:5]
    report(f"\n  -- {tag} --")
    for eid, r in sample:
        e = edges[eid]
        report(f"    [{eid}] {e['subject'][:35]} -[{e['relation']}]-> {e['object'][:35]}")
        report(f"          → {r[:120]}")

# ============================================================
# Apply drops, write V5
# ============================================================
kept_edges = [e for i, e in enumerate(edges) if i not in all_drops]
final_nodes = set()
for e in kept_edges:
    final_nodes.add(e["subject"]); final_nodes.add(e["object"])

out_groups = []
for g in groups:
    new_items = [it for it in g["items"] if it.get("formal_name") in final_nodes]
    if new_items:
        out_groups.append({**g, "items": new_items})

OUT = {
    "lattice": {"groups": out_groups},
    "relations": kept_edges,
    "conflicts": G.get("conflicts", []),
    "sources": G.get("sources", []),
    "relations_sources": G.get("relations_sources", []),
    "dedup_metadata": {
        **G.get("dedup_metadata", {}),
        "v5_opus_drops": len(all_drops),
        "v5_drop_tags": dict(tag_counts),
        "v5_model": MODEL,
        "v5_effort": "max",
    },
}
V5_OUT.write_text(json.dumps(OUT))

final_anchors = sum(len(e.get("anchor_list",[]) or []) for e in kept_edges)
report(f"\n{'='*70}\nV5 FINAL\n{'='*70}")
report(f"  Distinct nodes: {len(final_nodes):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(kept_edges):,}")
report(f"  Anchors: {final_anchors:,}")

final_rels = Counter(e["relation"] for e in kept_edges)
report(f"\n  Top relations:")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")

# Sanity
v5_olmo3 = [n for n in final_nodes if "olmo-3-" in n.lower().replace(" ","-") and "3.1" not in n.lower()]
v5_olmo31 = [n for n in final_nodes if "olmo-3.1" in n.lower()]
report(f"\nSanity:")
report(f"  Olmo-3 nodes: {len(v5_olmo3)}")
report(f"  Olmo-3.1 nodes: {len(v5_olmo31)}")
SEEDS = ["allenai/Olmo-3", "rl-research/DR-Tulu", "nvidia/NVIDIA-Nemotron-3", "HuggingFaceTB/SmolLM3"]
for s in SEEDS:
    found = any(s in n for n in final_nodes)
    report(f"  Seed '{s}': {'present' if found else 'MISSING'}")

report(f"\n✓ Wrote {V5_OUT} ({V5_OUT.stat().st_size:,} bytes)")
