#!/usr/bin/env python3
"""V8 fix: re-apply V8 verdicts with conflict-guarded union-find.

Problem: V8 used union-find on Opus-approved merges. When Opus approved both
  - {bare, 0325-variant} → ALL_SAME
  - {bare, 1124-variant} → ALL_SAME
the union-find chained all three into one component, even though the two
date-specific variants are distinct artifacts.

Fix: when uniting two components, check if combining them would create a
component with mutually-conflicting specifiers (different dates / versions /
sizes / stages). If yes, skip the union — leave them separate.

The bare alias becomes attached to whichever candidate it was first paired with
(highest-degree wins via deterministic ordering), preventing the chain.
"""
from __future__ import annotations
import json, re
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
RUN_DIR = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d"
V7_IN = RUN_DIR / "merge_artifact_deduped_v7.json"
V8_OUT = RUN_DIR / "merge_artifact_deduped_v8.json"  # OVERWRITE
VERDICTS_LOG = REPO / "run-logs/DEDUP_V8_VERDICTS.txt"
REPORT = REPO / "run-logs/DEDUP_V8_FIX_REPORT.txt"

REPORT.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

# ============================================================
# Parse V8 verdicts to recover original cluster decisions
# ============================================================
log = VERDICTS_LOG.read_text()
# Block format:
#   ====...
#   [cluster N] n=M rc=X dt=Y tag=TAG
#   ====...
#     [0] name1
#     [1] name2
#     ...
#   VERDICT: <line>

block_re = re.compile(
    r"\[cluster (\d+)\] n=(\d+)[^\n]*?tag=(\w+)\n(.*?)VERDICT:\s*(.+?)\n",
    re.DOTALL,
)
verdicts = []
for m in block_re.finditer(log):
    cid = int(m.group(1)); tag = m.group(3)
    body = m.group(4); vline = m.group(5).strip()
    members = []
    for line in body.split("\n"):
        mm = re.match(r"\s*\[(\d+)\]\s+(.+)", line)
        if mm:
            members.append(mm.group(2).strip())
    verdicts.append((cid, tag, members, vline))

report(f"Parsed {len(verdicts)} verdict blocks from log")
all_same = sum(1 for _, t, _, _ in verdicts if t == "ALL_SAME")
partial  = sum(1 for _, t, _, _ in verdicts if t == "PARTIAL")
distinct = sum(1 for _, t, _, _ in verdicts if t == "ALL_DISTINCT")
report(f"  ALL_SAME: {all_same}  PARTIAL: {partial}  ALL_DISTINCT: {distinct}")

# Re-parse the verdict line to get canonical_idx / merge_indices
def parse_payload(tag, vline, n_members):
    if tag == "ALL_SAME":
        m = re.match(r"ALL_SAME\s*::\s*(\d+)\s*::", vline)
        if m and 0 <= int(m.group(1)) < n_members:
            return {"canonical_idx": int(m.group(1))}
    elif tag == "PARTIAL":
        m = re.match(r"PARTIAL\s*::\s*([\d,\s]+)\s*::\s*(\d+)\s*::", vline)
        if m:
            try:
                idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
                idxs = [i for i in idxs if 0 <= i < n_members]
                ci = int(m.group(2))
                if len(idxs) >= 2 and 0 <= ci < len(idxs):
                    return {"merge_indices": idxs, "canonical_within": ci}
            except Exception:
                pass
    return None

# ============================================================
# Load V7 source
# ============================================================
G = json.loads(V7_IN.read_text())
edges = G["relations"]
groups = G["lattice"]["groups"]

all_nodes = set()
for e in edges:
    if e.get("subject"): all_nodes.add(e["subject"])
    if e.get("object"):  all_nodes.add(e["object"])
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name");  fn and all_nodes.add(fn)
all_nodes = {n for n in all_nodes if isinstance(n, str) and n}

degree = Counter()
for e in edges: degree[e["subject"]] += 1; degree[e["object"]] += 1

report(f"\nV7 source: {len(all_nodes):,} nodes, {len(edges):,} edges")

# ============================================================
# Specifier extraction (for conflict detection)
# ============================================================
DATE_RE   = re.compile(r"(?<!\d)(0[1-9]\d{2}|1[0-2]\d{2})(?!\d)")  # MMDD with leading zero allowed
SIZE_RE   = re.compile(r"\b(\d+(?:\.\d+)?)[Bb]\b")
VERSION_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")
STAGE_TOKENS = {"sft", "dpo", "instruct", "think", "base", "chat", "rl",
                "reward", "rm", "rlhf", "rlvr", "rlzero", "preview", "turbo"}

def specs(name):
    nl = name.lower()
    dates = frozenset(DATE_RE.findall(name))
    sizes = frozenset(s.lower() for s in SIZE_RE.findall(name))
    # versions: dotted (3.1, 2.5, 1.5), exclude size-numbers
    raw_versions = VERSION_RE.findall(name)
    versions = frozenset(raw_versions) - sizes
    # stages: token-level match
    tokens = re.split(r"[\s\-_/.\[\](),=:]+", nl)
    stages = frozenset(t for t in tokens if t in STAGE_TOKENS)
    return {"dates": dates, "sizes": sizes, "versions": versions, "stages": stages}

def specs_conflict(s_a, s_b):
    """Two specifier sets conflict if BOTH have non-empty entries for the same key
    AND those entries differ."""
    for k in ("dates", "versions", "sizes", "stages"):
        a = s_a[k]; b = s_b[k]
        if a and b and a != b:
            return True
    return False

def merged_specs(s_a, s_b):
    return {k: s_a[k] | s_b[k] for k in s_a}

# Pre-compute specs for every node
node_specs = {n: specs(n) for n in all_nodes}

# ============================================================
# Conflict-guarded union-find
# ============================================================
parent = {n: n for n in all_nodes}
component_specs = {n: dict(node_specs[n]) for n in all_nodes}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

skipped_unions = []
applied_unions = 0

def try_unite(a, b, source_label):
    global applied_unions
    ra, rb = find(a), find(b)
    if ra == rb: return True
    s_a = component_specs[ra]; s_b = component_specs[rb]
    if specs_conflict(s_a, s_b):
        skipped_unions.append({
            "node_a": a, "node_b": b,
            "comp_a": [n for n in all_nodes if find(n) == ra],
            "comp_b": [n for n in all_nodes if find(n) == rb],
            "source": source_label,
            "spec_a": s_a, "spec_b": s_b,
        })
        return False
    parent[ra] = rb
    component_specs[rb] = merged_specs(s_a, s_b)
    applied_unions += 1
    return True

# Apply each Opus verdict, but guarded
for cid, tag, members, vline in verdicts:
    payload = parse_payload(tag, vline, len(members))
    if not payload: continue
    if tag == "ALL_SAME":
        canonical = members[payload["canonical_idx"]]
        if canonical not in all_nodes: continue
        for m in members:
            if m in all_nodes and m != canonical:
                try_unite(m, canonical, f"cluster-{cid}")
    elif tag == "PARTIAL":
        merge_subset = [members[i] for i in payload["merge_indices"]]
        canonical = merge_subset[payload["canonical_within"]]
        if canonical not in all_nodes: continue
        for m in merge_subset:
            if m in all_nodes and m != canonical:
                try_unite(m, canonical, f"cluster-{cid}")

report(f"\nUnions applied:  {applied_unions}")
report(f"Unions skipped (conflict-guard): {len(skipped_unions)}")

# Show a few skipped (these are the saves)
report("\nSample SKIPPED unions (conflict-guard caught these):")
for su in skipped_unions[:10]:
    report(f"  Block {su['source']}: would have merged")
    report(f"    A: {su['node_a']}  (component: {[n for n in su['comp_a']][:4]})")
    report(f"    B: {su['node_b']}  (component: {[n for n in su['comp_b']][:4]})")
    keys_diff = []
    for k in ("dates", "versions", "sizes", "stages"):
        if su['spec_a'][k] != su['spec_b'][k] and su['spec_a'][k] and su['spec_b'][k]:
            keys_diff.append(f"{k}: {sorted(su['spec_a'][k])} vs {sorted(su['spec_b'][k])}")
    report(f"    conflict: {'; '.join(keys_diff)}")

# ============================================================
# Build canon_map from final components
# ============================================================
component_members = defaultdict(list)
for n in all_nodes:
    component_members[find(n)].append(n)

canon_map = {}
nodes_merged = 0
for root, members in component_members.items():
    if len(members) <= 1:
        canon_map[members[0]] = members[0]; continue
    prefixed = [m for m in members if "/" in m and not m.startswith("/")]
    pool = prefixed if prefixed else members
    canonical = max(pool, key=lambda m: (degree.get(m, 0), len(m)))
    for m in members:
        canon_map[m] = canonical
        if m != canonical: nodes_merged += 1

report(f"\nNodes merged after conflict-guard: {nodes_merged}")

# ============================================================
# Re-rewrite edges
# ============================================================
new_edges = {}
self_loops = 0
for e in edges:
    s = canon_map.get(e["subject"], e["subject"])
    o = canon_map.get(e["object"], e["object"])
    rel = e.get("relation", "")
    if s == o or not rel:
        self_loops += 1; continue
    key = (s, rel, o)
    if key not in new_edges:
        new_edges[key] = {**e,
                          "subject": s, "object": o,
                          "anchor_list": list(e.get("anchor_list") or []),
                          "description_variants": list(e.get("description_variants") or [])}
    else:
        new_edges[key]["anchor_list"].extend(e.get("anchor_list") or [])
        for v in (e.get("description_variants") or []):
            if v not in new_edges[key]["description_variants"]:
                new_edges[key]["description_variants"].append(v)

report(f"\nEdges before fix: {len(edges):,}")
report(f"Edges after fix:  {len(new_edges):,}")
report(f"Self-loops collapsed: {self_loops}")

# ============================================================
# Sanity — specifically check the OLMo-2 case
# ============================================================
final_node_set = set()
for k in new_edges:
    final_node_set.add(k[0]); final_node_set.add(k[2])

assert "allenai/OLMo-2-0325-32B-Instruct" in final_node_set, "0325 missing"
assert "allenai/OLMo-2-1124-32B-Instruct" in final_node_set, "1124 missing"
report(f"\n✓ Both OLMo-2-0325 and OLMo-2-1124 32B-Instruct preserved as distinct nodes")

olmo3 = [n for n in final_node_set if "olmo-3-" in n.lower().replace(" ","-") and "3.1" not in n.lower()]
olmo31 = [n for n in final_node_set if "olmo-3.1" in n.lower()]
report(f"  Olmo-3 nodes: {len(olmo3)}")
report(f"  Olmo-3.1 nodes: {len(olmo31)}")
assert len(olmo3) > 0 and len(olmo31) > 0

SEEDS = ["allenai/Olmo-3", "rl-research/DR-Tulu", "nvidia/NVIDIA-Nemotron-3", "HuggingFaceTB/SmolLM3"]
for s in SEEDS:
    found = any(s in n for n in final_node_set)
    report(f"  Seed '{s}': {'present' if found else 'MISSING'}")
    assert found

# ============================================================
# Build groups, write
# ============================================================
new_groups = defaultdict(list)
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name", "")
        if not fn: continue
        canon = canon_map.get(fn, fn)
        if canon in final_node_set:
            new_groups[canon].append(it)
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
        "v8_fix_applied_unions": applied_unions,
        "v8_fix_skipped_unions": len(skipped_unions),
        "v8_fix_nodes_merged": nodes_merged,
    },
}
V8_OUT.write_text(json.dumps(OUT))

report(f"\n{'='*70}\nV8 (FIXED) FINAL\n{'='*70}")
report(f"  Distinct nodes: {len(final_node_set):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(new_edges):,}")
report(f"  Anchors: {sum(len(e.get('anchor_list', []) or []) for e in new_edges.values()):,}")
report(f"\n✓ Wrote {V8_OUT} ({V8_OUT.stat().st_size:,} bytes)")
