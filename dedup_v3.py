#!/usr/bin/env python3
"""V3 dedup pass — operates on V2 output, applies aggressive surface-form normalization.

Specific wins targeted (per edge_audit.py analysis):
  1. Hyphen-collapsed match: rl-zero ↔ rlzero ↔ rl_zero
  2. Underscore-vs-period in versions: Llama-3_3 ↔ Llama-3.3
  3. Bare-vs-prefixed merge with MOST-POPULAR tiebreak (instead of refusing on ambiguity):
     MMLU + cais/mmlu → cais/mmlu
     GSM8K + openai/gsm8k → openai/gsm8k
     GPQA + Idavidrein/gpqa → Idavidrein/gpqa
  4. Casing variants (already lowercased)

Output: merge_artifact_deduped_v3.json (preserves v1, v2 — both still on disk).
"""
from __future__ import annotations
import json, re
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
V2_IN = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact_deduped.json"
V3_OUT = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact_deduped_v3.json"
REPORT = REPO / "run-logs/DEDUP_V3_REPORT.txt"

REPORT.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

# ============================================================
# Load V2
# ============================================================
G = json.loads(V2_IN.read_text())
edges = G["relations"]
groups = G["lattice"]["groups"]

report("="*70)
report("DEDUP V3: aggressive surface-form normalization on V2 graph")
report("="*70 + "\n")
report(f"V2 input: {V2_IN.name}")
report(f"  Edges: {len(edges):,}")
report(f"  Lattice groups: {len(groups):,}\n")

# ============================================================
# Build V3 signature for every node in v2
# ============================================================

ORG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)/(.+)$")
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)[Bb]\b")
VERSION_RE = re.compile(r"\b(\d+(?:[\._]\d+)+|\d+)\b")  # allow _ as separator
PAREN_ALIAS_RE = re.compile(r"\s*\([^)]*\)\s*")
BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*")
STAGE_KEYWORDS = {"sft","dpo","instruct","think","base","rl","rl-zero","rlzero",
                  "rlhf","rlvr","reasoning","content-safety","gen-rm","genrm",
                  "preview","fp8","bf16","nvfp4","onnx","fp16","int4","int8","quantized",
                  "preview","chat","completions","completion","reward","rm",
                  "policy","retriever","encoder","embedding","embed","tokenizer",
                  "checkpoint","intermediate","mid","cpt","awq","gguf"}

def collect_v2_nodes():
    nodes = set()
    for e in edges:
        nodes.add(e["subject"]); nodes.add(e["object"])
    for g in groups:
        for it in g["items"]: nodes.add(it["formal_name"])
    return {n for n in nodes if isinstance(n, str) and n}

def normalize_version(v):
    """Normalize version string: convert _ → . for matching."""
    return v.replace("_", ".")

def signature_v3(name):
    """Stricter signature: hyphen-collapsed bare_norm, normalized versions, AND
    preserves parens content + non-standard bracket attrs as distinguishing specifiers
    (so e.g. cais/mmlu (STEM) does NOT merge with cais/mmlu)."""
    if not isinstance(name, str): return None
    work = name
    # Capture parens content as a distinguishing specifier.
    # Strip leading-org-style parens (e.g. "Open-Orca/FLAN" inside parens) but keep meaningful tags.
    paren_match = re.search(r"\(([^)]+)\)\s*$", name.strip())
    paren_specifier = None
    if paren_match:
        ps = paren_match.group(1).strip().lower()
        # Skip parens that look like an org-prefixed alias (e.g., "openai/gsm8k") — those are info-equivalent
        if "/" in ps and not any(k in ps for k in ("split", "subset", "variant", "ablation", "config", "version", "diamond")):
            paren_specifier = None
        else:
            paren_specifier = ps
    # Capture bracket attrs that aren't already in dedicated buckets.
    bracket_specs = []
    bracket_match = re.search(r"\[([^\]]+)\]", name)
    if bracket_match:
        for kv in bracket_match.group(1).split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip().lower()
                v = v.strip().lower()
                if k in ("size", "version", "stage", "track"):
                    continue  # already in dedicated buckets
                bracket_specs.append((k, v))
    bracket_specifier = frozenset(bracket_specs)
    # Strip parens and brackets to get the canonical core for org/bare extraction
    work_core = BRACKET_RE.sub("", PAREN_ALIAS_RE.sub("", work)).strip()
    org = None
    m = ORG_RE.match(work_core)
    if m:
        org = m.group(1).lower()
        bare = m.group(2)
    else:
        bare = work_core
    bare_norm = re.sub(r"[\s_]+", "-", bare.lower().strip())
    bare_norm = re.sub(r"-+", "-", bare_norm).strip("-")
    bare_collapsed = bare_norm.replace("-", "").replace(".", "")
    sizes = frozenset(s.lower() for s in SIZE_RE.findall(name))
    raw_versions = VERSION_RE.findall(name)
    versions = frozenset(normalize_version(v) for v in raw_versions
                          if not v.endswith("b") and not v.lower().endswith("b") and len(v) <= 8)
    versions = versions - {s.lower() for s in sizes}
    name_lower = name.lower()
    stages = frozenset(t for t in STAGE_KEYWORDS if t in name_lower)
    dates = frozenset(d for d in re.findall(r"\b(\d{4})\b", name)
                      if 1000 < int(d) < 2030 and d not in versions and d not in {s.lower() for s in sizes})
    return (org, bare_norm, bare_collapsed, versions, sizes, stages, dates, paren_specifier, bracket_specifier)

def can_merge_v3(sig_a, sig_b):
    if sig_a is None or sig_b is None: return False
    org_a, bare_a, coll_a, ver_a, size_a, stage_a, date_a, paren_a, spec_a = sig_a
    org_b, bare_b, coll_b, ver_b, size_b, stage_b, date_b, paren_b, spec_b = sig_b
    if coll_a != coll_b: return False
    if org_a and org_b and org_a != org_b: return False
    if ver_a and ver_b and ver_a != ver_b: return False
    if size_a and size_b and size_a != size_b: return False
    if stage_a and stage_b and stage_a != stage_b: return False
    if date_a and date_b and date_a != date_b: return False
    # Parens specifier: BOTH must agree exactly. Either both None, or both equal string.
    if paren_a != paren_b: return False
    # Bracket non-standard specifiers: must match exactly (frozenset equality).
    if spec_a != spec_b: return False
    return True

# ============================================================
# Build initial clusters from V2 nodes by signature
# ============================================================
v2_nodes = collect_v2_nodes()
report(f"Distinct V2 nodes: {len(v2_nodes):,}")

# Compute degree (in + out) for popularity tiebreak
degree = defaultdict(int)
for e in edges:
    degree[e["subject"]] += 1
    degree[e["object"]] += 1

# Build sigs
node_to_sig = {}
for n in v2_nodes:
    sig = signature_v3(n)
    if sig: node_to_sig[n] = sig

# Cluster by full sig (with org)
exact_clusters = defaultdict(list)
for n, sig in node_to_sig.items():
    exact_clusters[sig].append(n)

# Now build a "fuzzy" cluster index: bare_collapsed → list of (sig, names)
# Two sigs that differ only in (a) hyphen layout or (b) presence/absence of org
# get pulled into the same fuzzy cluster.
fuzzy_clusters = defaultdict(list)
for sig, names in exact_clusters.items():
    org, bare_norm, bare_coll, ver, size, stage, date, paren, spec = sig
    fuzzy_key = (bare_coll, ver, size, stage, date, paren, spec)
    fuzzy_clusters[fuzzy_key].append((sig, names))

# Decide canonical for each fuzzy cluster:
#   Among all (sig, names) in the cluster:
#     prefer prefixed sigs (have org) over bare;
#     among prefixed sigs, pick the one with highest aggregate degree;
#     pick canonical name = highest-degree prefixed name in that sig's names
canon_map = {}  # v2_node_name → v3_canonical_name
fuzzy_merge_count = 0
fuzzy_sample = []

for fuzzy_key, sig_groups in fuzzy_clusters.items():
    if len(sig_groups) == 1:
        # No fuzzy merge — just pick canonical for this single sig group
        sig, names = sig_groups[0]
        canonical = max(names, key=lambda n: (degree.get(n, 0), n.startswith("/") is False, "/" in n, len(n)))
        for n in names:
            canon_map[n] = canonical
    else:
        # Multiple sigs in the same fuzzy cluster — merge them
        all_names = []
        prefixed_sigs = [(s, ns) for s, ns in sig_groups if s[0]]
        bare_sigs = [(s, ns) for s, ns in sig_groups if not s[0]]
        # Compute aggregate degree per sig
        sig_degree = []
        for sig, names in sig_groups:
            d = sum(degree.get(n, 0) for n in names)
            sig_degree.append((sig, names, d))
            all_names.extend(names)
        # Prefer prefixed; among prefixed, pick highest-degree
        if prefixed_sigs:
            best = max([(s, ns, sum(degree.get(n, 0) for n in ns)) for s, ns in prefixed_sigs],
                       key=lambda x: x[2])
        else:
            best = max(sig_degree, key=lambda x: x[2])
        target_sig, target_names, _ = best
        # Canonical: highest-degree name in target_names that is "prefixed" if any
        prefixed_in_target = [n for n in target_names if "/" in n and not n.startswith("/")]
        if prefixed_in_target:
            canonical = max(prefixed_in_target, key=lambda n: degree.get(n, 0))
        else:
            canonical = max(target_names, key=lambda n: degree.get(n, 0))
        for n in all_names:
            canon_map[n] = canonical
        fuzzy_merge_count += 1
        if len(fuzzy_sample) < 25:
            samples = sorted(all_names, key=lambda n: -degree.get(n, 0))[:6]
            fuzzy_sample.append((canonical, samples, sum(degree.get(n,0) for n in all_names)))

report(f"Fuzzy clusters merging multiple sigs: {fuzzy_merge_count}")
report(f"Total V3 canonicals: {len(set(canon_map.values())):,}\n")

report("Sample of fuzzy merges (by aggregate degree):")
for canon, samples, agg_deg in sorted(fuzzy_sample, key=lambda x: -x[2])[:20]:
    report(f"\n  CANONICAL [agg_degree={agg_deg}]: {canon}")
    for n in samples:
        marker = "  *" if n == canon else "   "
        report(f"  {marker}  [d={degree.get(n,0):>3}] {n[:90]}")

# ============================================================
# Re-rewrite edges with v3 canonicals
# ============================================================
report(f"\nRewriting edges with V3 canonicals...")
new_edges = {}
edges_collapsed = 0
for e in edges:
    s = canon_map.get(e["subject"], e["subject"])
    o = canon_map.get(e["object"], e["object"])
    rel = e["relation"]
    if s == o or not rel:
        edges_collapsed += 1
        continue
    key = (s, rel, o)
    if key not in new_edges:
        new_edges[key] = {
            "subject": s, "relation": rel, "object": o,
            "dependency_kind": e.get("dependency_kind"),
            "description": e.get("description", ""),
            "anchor_list": [],
            "description_variants": [],
        }
    new_edges[key]["anchor_list"].extend(e.get("anchor_list", []) or [])
    desc = e.get("description", "")
    if desc and desc != new_edges[key]["description"] and desc not in new_edges[key]["description_variants"]:
        new_edges[key]["description_variants"].append(desc)
    for v in e.get("description_variants", []) or []:
        if v not in new_edges[key]["description_variants"]:
            new_edges[key]["description_variants"].append(v)

report(f"  Edges before: {len(edges):,}")
report(f"  Edges after V3 merge: {len(new_edges):,}")
report(f"  Collapsed to self-loops: {edges_collapsed:,}")
report(f"  Net edge reduction: {len(edges) - len(new_edges):,}")

# Sanity checks
canon_set = set(canon_map.values())
olmo3 = [c for c in canon_set if "olmo-3-" in c.lower() and "3.1" not in c.lower()]
olmo31 = [c for c in canon_set if "olmo-3.1" in c.lower()]
report(f"\nSanity:")
report(f"  Olmo-3 canonicals (not 3.1): {len(olmo3)}")
report(f"  Olmo-3.1 canonicals: {len(olmo31)}")
assert len(olmo3) > 0 and len(olmo31) > 0, "ABORT: version distinction collapsed"

# Check key benchmarks merged correctly
def find_canon_for(query):
    matches = [n for n in canon_map if query.lower() in n.lower()]
    canons = {canon_map[n] for n in matches}
    return canons

for q in ["MMLU", "GSM8K", "GPQA", "Common Crawl", "Stack Exchange"]:
    canons = find_canon_for(q)
    report(f"  '{q}' merges to: {sorted(canons)[:3]}")

# ============================================================
# Build output groups
# ============================================================
new_groups_d = defaultdict(list)
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name", "")
        if not fn: continue
        canon = canon_map.get(fn, fn)
        new_groups_d[canon].append(it)

# Filter to canonicals that appear in final edges
final_node_set = set()
for (s, _, o) in new_edges:
    final_node_set.add(s); final_node_set.add(o)

out_groups = []
for canon, its in new_groups_d.items():
    if canon not in final_node_set: continue
    primary = next((i for i in its if i.get("formal_name") == canon), its[0])
    primary = dict(primary)
    primary["formal_name"] = canon
    primary["alias_count"] = len(its)
    out_groups.append({"items": [primary], "id": canon})

OUT = {
    "lattice": {"groups": out_groups},
    "relations": list(new_edges.values()),
    "conflicts": G.get("conflicts", []),
    "sources": G.get("sources", []),
    "relations_sources": G.get("relations_sources", []),
    "dedup_metadata": {
        **G.get("dedup_metadata", {}),
        "v3_fuzzy_merges": fuzzy_merge_count,
    },
}
V3_OUT.write_text(json.dumps(OUT))

# Final stats
final_anchors = sum(len(e["anchor_list"]) for e in new_edges.values())
report(f"\n" + "="*70)
report(f"V3 FINAL")
report(f"="*70)
report(f"  Distinct nodes: {len(final_node_set):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(new_edges):,}")
report(f"  Anchors: {final_anchors:,}")
final_rels = Counter(rel for (s,rel,o) in new_edges)
report(f"\n  Top relations:")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")
report(f"\n✓ Wrote {V3_OUT} ({V3_OUT.stat().st_size:,} bytes)")
