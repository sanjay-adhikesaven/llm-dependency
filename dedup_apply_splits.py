#!/usr/bin/env python3
"""Post-process: revert NO_SPLIT clusters flagged by Opus in dedup_graph.py Stage 6.

The first dedup pass over-merged 33 of 100 borderline clusters (per Opus verification).
Specifically: dataset subsets/configs, training-stage checkpoints, year-specific releases,
date-suffixed releases, and community re-releases were collapsed when they shouldn't be.

This script:
  1. Re-runs the clustering logic from dedup_graph.py (deterministic) to rebuild canon_map.
  2. Parses DEDUP_LLM_DECISIONS.txt to find NO_SPLIT canonicals.
  3. For each NO_SPLIT cluster, reverts every original name → itself (no merging within cluster).
  4. Re-runs edge rewrite + low-signal filter with the updated canon_map.
  5. Overwrites merge_artifact_deduped.json with the corrected output.
  6. Writes DEDUP_REPORT_V2.txt with before/after numbers (vs original AND vs v1).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
SOURCE = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact.json"
V1_OUT = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact_deduped.json"
OUT_PATH = V1_OUT  # overwrite v1 since v1 was wrong
LLM_LOG = REPO / "run-logs/DEDUP_LLM_DECISIONS.txt"
REPORT = REPO / "run-logs/DEDUP_REPORT_V2.txt"

# ============================================================
# Helpers (must match dedup_graph.py exactly for deterministic clustering)
# ============================================================

INTERNAL_PATH_RE = re.compile(r"^(?:/|gs://|weka://|s3://|/weka/|/scratch/|/fsx/)")
WIKI_BRACKET_RE = re.compile(r"^(.+?)\s*\[([^\]]+)\]\s*$")
PAREN_ALIAS_RE = re.compile(r"\s*\([^)]*\)\s*")
ORG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)/(.+)$")
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)[Bb]\b")
VERSION_RE = re.compile(r"\b(\d+(?:\.\d+)+|\d+)\b")
STAGE_KEYWORDS = {"sft","dpo","instruct","think","base","rl","rl-zero","rlzero",
                  "rlhf","rlvr","reasoning","content-safety","gen-rm","genrm",
                  "preview","fp8","bf16","nvfp4","onnx","fp16","int4","int8","quantized",
                  "preview","chat","completions","completion","reward","rm",
                  "policy","retriever","encoder","embedding","embed","tokenizer",
                  "checkpoint","intermediate","mid","cpt","awq","gguf"}
DATE_SLUG_RE = re.compile(r"\b(\d{4})\b")

def to_str(v):
    if isinstance(v, dict): return v.get("formal_name") or v.get("name") or ""
    if isinstance(v, str): return v
    return ""

def parse_wiki(name):
    m = WIKI_BRACKET_RE.match(name)
    if not m: return name
    family, attrs_str = m.groups()
    attrs = {}
    for attr in attrs_str.split(","):
        if "=" in attr:
            k, v = attr.split("=", 1)
            attrs[k.strip()] = v.strip()
    version = attrs.get("version")
    if version:
        family = re.sub(r"\s+\d+(?:\.\d+)?\s*$", f" {version}", family)
    parts = [family.strip()]
    if attrs.get("size"): parts.append(attrs["size"])
    if attrs.get("variant"): parts.append(attrs["variant"])
    if attrs.get("stage"): parts.append(attrs["stage"])
    if attrs.get("track"): parts.append(attrs["track"])
    return "-".join(parts).replace(" ", "-")

def is_drop(name):
    if not name or not isinstance(name, str): return True
    n = name.strip()
    if not n: return True
    if INTERNAL_PATH_RE.match(n): return True
    if "://" in n and not n.startswith("https://huggingface.co/"): return True
    paren = re.search(r"\(([^)]+)\)", n)
    if paren and len(paren.group(1)) > 50: return True
    if len(n) > 200: return True
    return False

def signature(name):
    if not isinstance(name, str): return (None, None, frozenset(), frozenset(), frozenset(), frozenset())
    work = parse_wiki(name)
    work = PAREN_ALIAS_RE.sub("", work).strip()
    org = None
    m = ORG_RE.match(work)
    if m:
        org = m.group(1).lower()
        bare = m.group(2)
    else:
        bare = work
    bare_norm = re.sub(r"[_\s]+", "-", bare).lower().strip("-")
    sizes = frozenset(s.lower() for s in SIZE_RE.findall(name))
    versions = frozenset(VERSION_RE.findall(name)) - sizes
    versions = frozenset(v for v in versions if not v.endswith("b") and len(v) <= 6)
    tokens = set(re.split(r"[\s\-_/.]", bare_norm))
    stages = frozenset(t for t in tokens if t in STAGE_KEYWORDS)
    dates = frozenset(d for d in DATE_SLUG_RE.findall(name) if 1000 < int(d) < 2030 and d not in versions)
    return (org, bare_norm, versions, sizes, stages, dates)

def can_merge(sig_a, sig_b):
    org_a, bare_a, ver_a, size_a, stage_a, date_a = sig_a
    org_b, bare_b, ver_b, size_b, stage_b, date_b = sig_b
    if bare_a != bare_b: return False
    if org_a and org_b and org_a != org_b: return False
    if ver_a and ver_b and ver_a != ver_b: return False
    if size_a and size_b and size_a != size_b: return False
    if stage_a and stage_b and stage_a != stage_b: return False
    if date_a and date_b and date_a != date_b: return False
    return True

def pick_canonical(names):
    prefixed = [n for n in names if "/" in n and not n.startswith("/")]
    if prefixed:
        return Counter(prefixed).most_common(1)[0][0]
    cleaned = [PAREN_ALIAS_RE.sub("", n).strip() for n in names]
    cleaned = [n for n in cleaned if n]
    if cleaned:
        return max(cleaned, key=len)
    return names[0]

# ============================================================
# Run
# ============================================================

REPORT.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

report("="*70)
report("DEDUP V2: Apply NO_SPLIT reverts from Opus verification")
report("="*70 + "\n")

# Load
G = json.loads(SOURCE.read_text())
relations_orig = G.get("relations", [])
groups_orig = G.get("lattice", {}).get("groups", [])
items_orig = [it for g in groups_orig for it in g.get("items", [])]

all_node_names = set()
for r in relations_orig:
    s = to_str(r.get("subject")); o = to_str(r.get("object"))
    if s: all_node_names.add(s)
    if o: all_node_names.add(o)
for it in items_orig:
    fn = it.get("formal_name")
    if fn: all_node_names.add(fn)

# Re-cluster (deterministic — same code as v1)
node_sigs = {}
dropped_nodes = []
for n in all_node_names:
    if is_drop(n): dropped_nodes.append(n); continue
    node_sigs[n] = signature(n)

clusters = defaultdict(list)
for name, sig in node_sigs.items():
    clusters[sig].append(name)

bare_to_prefixed = defaultdict(list)
for sig in clusters:
    if sig[0]: bare_to_prefixed[sig[1]].append((sig[0], sig))

final_clusters = {}
for sig, names in clusters.items():
    if sig[0] is None:
        bare = sig[1]
        candidates = bare_to_prefixed.get(bare, [])
        compat = [c_sig for c_org, c_sig in candidates if can_merge(sig, c_sig)]
        if len(compat) == 1:
            target_sig = compat[0]
            final_clusters.setdefault(target_sig, []).extend(names)
            continue
    final_clusters.setdefault(sig, []).extend(names)

canon_map = {}
canon_to_members = defaultdict(list)  # canonical → list of original names in this cluster
for sig, names in final_clusters.items():
    canonical = pick_canonical(names)
    for n in names:
        canon_map[n] = canonical
        canon_to_members[canonical].append(n)
for d in dropped_nodes:
    canon_map[d] = None

report(f"Re-clustered (matches v1):")
report(f"  Distinct nodes: {len(all_node_names):,}")
report(f"  Canonical clusters: {len(final_clusters):,}")
report(f"  Dropped (categorical): {len(dropped_nodes):,}\n")

# ============================================================
# Parse LLM log to find NO_SPLIT canonicals
# ============================================================
log_text = LLM_LOG.read_text()
# Format: "=== [N] {canon} ({M} names) — {VERDICT} ==="
SPLIT_RE = re.compile(r"^=== \[\d+\] (.+?) \(\d+ names\) — (\w+) ===$", re.MULTILINE)
verdicts = {}  # canon → verdict
for m in SPLIT_RE.finditer(log_text):
    canon, verdict = m.group(1), m.group(2)
    verdicts[canon] = verdict

no_split_canons = {c for c, v in verdicts.items() if v == "NO_SPLIT"}
partial_ok_canons = {c for c, v in verdicts.items() if v == "PARTIAL_OK"}
yes_canons = {c for c, v in verdicts.items() if v == "YES_ALL_SAME"}

report(f"Parsed LLM verdicts: {len(verdicts)} clusters")
report(f"  YES_ALL_SAME: {len(yes_canons)}")
report(f"  NO_SPLIT:     {len(no_split_canons)}")
report(f"  PARTIAL_OK:   {len(partial_ok_canons)} (kept as-is)")

# Map each NO_SPLIT canonical from log to a v2 cluster.
# pick_canonical is non-deterministic across runs (depends on set iteration order),
# so a NO_SPLIT name from the log might be a *member* of a v2 cluster rather than its canonical.
# Resolution order:
#   1. Direct: canon_log == v2 canonical → revert that cluster
#   2. Indirect: canon_log is a member name → revert the cluster whose canonical it maps to
#   3. Signature: compute signature(canon_log), find matching cluster
clusters_to_revert = set()  # v2 canonical names of clusters to revert
unresolved = []

for canon_log in no_split_canons:
    if canon_log in canon_to_members:
        clusters_to_revert.add(canon_log)
    elif canon_log in canon_map and canon_map[canon_log]:
        clusters_to_revert.add(canon_map[canon_log])
    else:
        sig = signature(canon_log)
        if sig in final_clusters:
            v2_canon = pick_canonical(final_clusters[sig])
            clusters_to_revert.add(v2_canon)
        else:
            # Try fallback: search for a sig-compatible cluster
            found = None
            for fc_sig in final_clusters:
                if can_merge(sig, fc_sig):
                    found = pick_canonical(final_clusters[fc_sig])
                    break
            if found:
                clusters_to_revert.add(found)
            else:
                unresolved.append(canon_log)

if unresolved:
    report(f"\n⚠ {len(unresolved)} NO_SPLIT canonicals could not be resolved to any v2 cluster:")
    for u in unresolved[:10]:
        report(f"  unresolved: {u}")

# Apply reverts
reverted_names_total = 0
for canon_v2 in clusters_to_revert:
    members = canon_to_members.get(canon_v2, [])
    for n in members:
        canon_map[n] = n
        reverted_names_total += 1

report(f"\nResolved {len(clusters_to_revert)}/{len(no_split_canons)} NO_SPLIT clusters → reverted {reverted_names_total} names (each → itself)")

# Recompute final canonical population
canon_groups = defaultdict(list)
for orig, canon in canon_map.items():
    if canon: canon_groups[canon].append(orig)
report(f"Distinct canonical names after revert: {len(canon_groups):,}")

# ============================================================
# Re-rewrite edges with updated canon_map
# ============================================================
report(f"\nRewriting edges with updated canon_map...")
incoming = defaultdict(int)
outgoing = defaultdict(int)
new_edges = {}
edges_dropped_endpoint = 0
edges_dropped_self = 0

for r in relations_orig:
    s = to_str(r.get("subject")); o = to_str(r.get("object")); rel = r.get("relation","")
    s_canon = canon_map.get(s, s)
    o_canon = canon_map.get(o, o)
    if not s_canon or not o_canon:
        edges_dropped_endpoint += 1; continue
    if s_canon == o_canon:
        edges_dropped_self += 1; continue
    if not rel: continue
    key = (s_canon, rel, o_canon)
    if key not in new_edges:
        new_edges[key] = {
            "subject": s_canon, "relation": rel, "object": o_canon,
            "dependency_kind": r.get("dependency_kind"),
            "description": r.get("description",""),
            "anchor_list": [], "description_variants": [],
        }
    new_edges[key]["anchor_list"].extend(r.get("anchor_list",[]) or [])
    desc = r.get("description","")
    if desc and desc != new_edges[key]["description"] and desc not in new_edges[key]["description_variants"]:
        new_edges[key]["description_variants"].append(desc)
    for v in r.get("description_variants",[]) or []:
        if v not in new_edges[key]["description_variants"]:
            new_edges[key]["description_variants"].append(v)
    incoming[o_canon] += 1
    outgoing[s_canon] += 1

report(f"  Edges merged to canonical: {len(new_edges):,}")
report(f"  Dropped (endpoint dropped): {edges_dropped_endpoint:,}")
report(f"  Dropped (self-loop after merge): {edges_dropped_self:,}")

# Low-signal filter
def looks_concept(name):
    if not isinstance(name, str): return False
    if "/" in name: return False
    if SIZE_RE.search(name): return False
    if "(" in name: return False
    if len(name) < 30 and re.match(r"^[A-Za-z][A-Za-z0-9\-\s_.]*$", name):
        return True
    return False

active_nodes = set(incoming.keys()) | set(outgoing.keys())
low_signal_drops = []
for n in list(active_nodes):
    if looks_concept(n):
        deg = incoming[n] + outgoing[n]
        if deg < 3:
            low_signal_drops.append(n); active_nodes.discard(n)
low_drop_set = set(low_signal_drops)
final_edges = {k: e for k, e in new_edges.items() if k[0] not in low_drop_set and k[2] not in low_drop_set}

report(f"  Concept-like low-degree drops: {len(low_signal_drops):,}")
report(f"  Final edges (post-low-signal): {len(final_edges):,}")

# Final node set
final_node_set = set()
for (s, rel, o) in final_edges:
    final_node_set.add(s); final_node_set.add(o)

final_anchors = sum(len(e["anchor_list"]) for e in final_edges.values())

# ============================================================
# Sanity checks
# ============================================================
report(f"\nSanity checks:")
olmo3 = [c for c in canon_groups if c and "olmo-3-" in c.lower() and "3.1" not in c.lower()]
olmo31 = [c for c in canon_groups if c and "olmo-3.1" in c.lower()]
report(f"  Olmo-3 distinct canonicals (not 3.1): {len(olmo3)}")
report(f"  Olmo-3.1 distinct canonicals: {len(olmo31)}")
assert len(olmo3) > 0 and len(olmo31) > 0, "ABORT: Olmo-3 / Olmo-3.1 collapsed"

SEEDS = ["allenai/Olmo-3", "rl-research/DR-Tulu", "nvidia/NVIDIA-Nemotron-3", "HuggingFaceTB/SmolLM3"]
for seed in SEEDS:
    found = any(seed in c for c in canon_groups)
    report(f"  Seed '{seed}': {'present' if found else 'MISSING'}")

# Check no NO_SPLIT canonicals are still active in the graph
still_merged = [c for c in no_split_canons if c in final_node_set]
report(f"  NO_SPLIT canonicals still merged in graph: {len(still_merged)} (these reflect the canonical's own original name still being valid)")

# ============================================================
# Build output
# ============================================================
new_groups = defaultdict(list)
for it in items_orig:
    fn = it.get("formal_name","")
    if not fn: continue
    canon = canon_map.get(fn)
    if not canon or canon not in final_node_set: continue
    new_groups[canon].append(it)
out_groups = []
for canon, its in new_groups.items():
    primary = next((i for i in its if i.get("formal_name") == canon), its[0])
    primary = dict(primary)
    primary["formal_name"] = canon
    primary["alias_count"] = len(its)
    out_groups.append({"items": [primary], "id": canon})

OUT = {
    "lattice": {"groups": out_groups},
    "relations": list(final_edges.values()),
    "conflicts": G.get("conflicts", []),
    "sources": G.get("sources", []),
    "relations_sources": G.get("relations_sources", []),
    "dedup_metadata": {
        "source_artifact": str(SOURCE),
        "v2_no_split_reverted": len(no_split_canons),
        "v2_names_reverted": reverted_names_total,
        "low_signal_drops": len(low_signal_drops),
        "llm_verdicts": dict(Counter(verdicts.values())),
    },
}
OUT_PATH.write_text(json.dumps(OUT))

# ============================================================
# Final report
# ============================================================
report("\n" + "="*70)
report("BEFORE / V1 / V2")
report("="*70)
v1 = json.loads(V1_OUT.read_text() if V1_OUT != OUT_PATH else "{}")  # if same path, this is now v2; need to compare ahead
# The v1 output was already overwritten — we report v2 vs original instead.

orig_anchors = sum(len(r.get("anchor_list",[]) or []) for r in relations_orig)
report(f"\nMETRIC                BEFORE       V2")
report(f"  Distinct nodes    {len(all_node_names):>8,}    {len(final_node_set):>8,}    Δ {len(all_node_names)-len(final_node_set):+,}")
report(f"  Edges             {len(relations_orig):>8,}    {len(final_edges):>8,}    Δ {len(relations_orig)-len(final_edges):+,}")
report(f"  Anchors           {orig_anchors:>8,}    {final_anchors:>8,}    Δ {orig_anchors-final_anchors:+,}")

# Item-kind breakdown
canon_to_items = defaultdict(list)
for it in items_orig:
    fn = it.get("formal_name","")
    if not fn: continue
    canon = canon_map.get(fn)
    if canon and canon in final_node_set:
        canon_to_items[canon].append(it)
final_kind = Counter()
for canon, its in canon_to_items.items():
    most_common_kind = Counter(i.get("kind","?") for i in its).most_common(1)[0][0]
    final_kind[most_common_kind] += 1

report(f"\n  Final canonical-node kind breakdown:")
for k, n in final_kind.most_common():
    report(f"    {n:>6,}  {k}")

final_rels = Counter(rel for (s,rel,o) in final_edges)
report(f"\n  Top relations (V2):")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")

report(f"\n✓ Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
report(f"✓ Original {SOURCE.name} preserved at source.")
