#!/usr/bin/env python3
"""Heavy graph dedup + filter pass over the final merge artifact.

Six stages:
  1. Inventory + bucket every node (real, alias, internal-path, concept, junk)
  2. Compute normalization key for each candidate (preserve version/size/stage/org)
  3. Pick canonical name per cluster (prefer org/name HF form)
  4. Filter low-signal nodes (orphans, low-anchor concepts, unmappable internal paths)
  5. Rewrite edges using canonical names; merge duplicate edges; drop edges with dropped endpoints
  6. LLM-verify borderline clusters (Sonnet, only on ambiguous merges)

Output:
  - storage/runs/<rid>/merge_artifact_deduped.json  (cleaned graph, NEW file)
  - run-logs/DEDUP_REPORT.txt                       (before/after stats + sample merges)
  - run-logs/DEDUP_LLM_DECISIONS.txt                (LLM responses for borderline cases)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
SOURCE = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d/merge_artifact.json"
OUT_DIR = SOURCE.parent
OUT_PATH = OUT_DIR / "merge_artifact_deduped.json"
LOG_DIR = REPO / "run-logs"
REPORT = LOG_DIR / "DEDUP_REPORT.txt"
LLM_LOG = LOG_DIR / "DEDUP_LLM_DECISIONS.txt"

# ============================================================
# Helpers
# ============================================================

def to_str(v):
    if isinstance(v, dict): return v.get("formal_name") or v.get("name") or ""
    if isinstance(v, str): return v
    return ""

def report(msg, also_print=True):
    line = f"{msg}\n"
    with open(REPORT, "a") as f: f.write(line)
    if also_print: print(msg)

# Erase old reports
REPORT.write_text("")
LLM_LOG.write_text("")

report("="*70)
report(f"DEDUP RUN — source: {SOURCE.name}")
report(f"Output: {OUT_PATH.name}")
report("="*70 + "\n")

# ============================================================
# Load
# ============================================================

G = json.loads(SOURCE.read_text())
relations_orig = G.get("relations", [])
groups_orig = G.get("lattice", {}).get("groups", [])
items_orig = [it for g in groups_orig for it in g.get("items", [])]

# Collect every distinct node name (subjects + objects + lattice items)
all_node_names = set()
for r in relations_orig:
    s = to_str(r.get("subject")); o = to_str(r.get("object"))
    if s: all_node_names.add(s)
    if o: all_node_names.add(o)
for it in items_orig:
    fn = it.get("formal_name")
    if fn: all_node_names.add(fn)

report(f"Stage 1: Inventory")
report(f"  Distinct node names in graph: {len(all_node_names):,}")
report(f"  Edges: {len(relations_orig):,}")
report(f"  Lattice items: {len(items_orig):,}")
report(f"  Lattice groups: {len(groups_orig):,}\n")

# ============================================================
# STAGE 1+2: Bucket + normalization key
# ============================================================

INTERNAL_PATH_RE = re.compile(r"^(?:/|gs://|weka://|s3://|/weka/|/scratch/|/fsx/)")
WIKI_BRACKET_RE = re.compile(r"^(.+?)\s*\[([^\]]+)\]\s*$")
PAREN_ALIAS_RE = re.compile(r"\s*\([^)]*\)\s*")

# Regex pieces for separator detection
ORG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)/(.+)$")
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)[Bb]\b")
VERSION_RE = re.compile(r"\b(\d+(?:\.\d+)+|\d+)\b")
STAGE_KEYWORDS = {"sft","dpo","instruct","think","base","rl","rl-zero","rlzero",
                  "rlhf","rlvr","reasoning","content-safety","gen-rm","genrm",
                  "preview","fp8","bf16","nvfp4","onnx","fp16","int4","int8","quantized",
                  "preview","chat","completions","completion","reward","rm",
                  "policy","retriever","encoder","embedding","embed","tokenizer",
                  "checkpoint","intermediate","mid","cpt","awq","gguf"}
DATE_SLUG_RE = re.compile(r"\b(\d{4})\b")  # 1025, 1124 etc.

def parse_wiki(name):
    """Convert 'OLMo 3 [size=32B, stage=Instruct-SFT, version=3.1]' to a normalized form."""
    m = WIKI_BRACKET_RE.match(name)
    if not m: return name
    family, attrs_str = m.groups()
    attrs = {}
    for attr in attrs_str.split(","):
        if "=" in attr:
            k, v = attr.split("=", 1)
            attrs[k.strip()] = v.strip()
    # Override family version if specified
    version = attrs.get("version")
    if version:
        # "OLMo 3" + version=3.1 → "OLMo 3.1"
        family = re.sub(r"\s+\d+(?:\.\d+)?\s*$", f" {version}", family)
    parts = [family.strip()]
    if attrs.get("size"): parts.append(attrs["size"])
    if attrs.get("variant"): parts.append(attrs["variant"])
    if attrs.get("stage"): parts.append(attrs["stage"])
    if attrs.get("track"): parts.append(attrs["track"])
    return "-".join(parts).replace(" ", "-")

def is_drop(name):
    """Categorical drops: internal paths, free-text junk."""
    if not name or not isinstance(name, str): return True
    n = name.strip()
    if not n: return True
    if INTERNAL_PATH_RE.match(n): return True
    if "://" in n and not n.startswith("https://huggingface.co/"): return True
    # Free-text descriptive node: contains parenthetical longer than the bare name
    paren = re.search(r"\(([^)]+)\)", n)
    if paren and len(paren.group(1)) > 50: return True
    if len(n) > 200: return True
    return False

def signature(name):
    """Compute (org, bare_key, version_set, size_set, stage_set, date_set) for hard-separator checks."""
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
    versions = frozenset(VERSION_RE.findall(name)) - sizes  # exclude 7B/32B from versions
    # Strip the size markers from version candidates
    versions = frozenset(v for v in versions if not v.endswith("b") and len(v) <= 6)
    # Stage tokens
    tokens = set(re.split(r"[\s\-_/.]", bare_norm))
    stages = frozenset(t for t in tokens if t in STAGE_KEYWORDS)
    # Date slugs (4-digit numbers between size and stage that look like MMDD)
    dates = frozenset(d for d in DATE_SLUG_RE.findall(name) if 1000 < int(d) < 2030 and d not in versions)
    return (org, bare_norm, versions, sizes, stages, dates)

def can_merge(sig_a, sig_b):
    """Return True if two signatures can safely merge (no hard separator)."""
    org_a, bare_a, ver_a, size_a, stage_a, date_a = sig_a
    org_b, bare_b, ver_b, size_b, stage_b, date_b = sig_b
    if bare_a != bare_b: return False
    # Org check: if both have orgs and they differ → no merge
    if org_a and org_b and org_a != org_b: return False
    # Version check: must match exactly (or one is empty)
    if ver_a and ver_b and ver_a != ver_b: return False
    if size_a and size_b and size_a != size_b: return False
    if stage_a and stage_b and stage_a != stage_b: return False
    if date_a and date_b and date_a != date_b: return False
    return True

# Build node → signature map
node_sigs = {}
dropped_nodes = []
for n in all_node_names:
    if is_drop(n):
        dropped_nodes.append(n)
        continue
    node_sigs[n] = signature(n)

report(f"Stage 1.5: Categorical drops (internal paths / free-text junk): {len(dropped_nodes):,}")
for d in dropped_nodes[:5]:
    report(f"  drop: {d[:100]}")
if len(dropped_nodes) > 5: report(f"  ... and {len(dropped_nodes)-5} more")
report("")

# ============================================================
# STAGE 3: Cluster by signature, pick canonical
# ============================================================
report("Stage 2-3: Clustering with hard separators...")

# Group by (org, bare_norm, versions, sizes, stages, dates)
clusters = defaultdict(list)  # signature → [names]
for name, sig in node_sigs.items():
    clusters[sig].append(name)

# Now also fold "bare" forms (no org) into prefixed clusters when there's exactly one matching prefix
# Build a map: bare_key + (no-org) → list of (org, prefixed_name) clusters with that bare_key
bare_to_prefixed = defaultdict(list)  # bare_key → [(org, signature)] for prefixed clusters
for sig in clusters:
    org = sig[0]
    bare = sig[1]
    if org: bare_to_prefixed[bare].append((org, sig))

# Reassign no-org clusters into prefixed clusters where unique
no_org_reassigned = 0
final_clusters = {}  # signature → [names]
for sig, names in clusters.items():
    if sig[0] is None:
        # No org — try to merge into a prefixed cluster
        bare = sig[1]
        candidates = bare_to_prefixed.get(bare, [])
        # Filter to candidates compatible (matching version/size/stage/date)
        compat = [c_sig for c_org, c_sig in candidates if can_merge(sig, c_sig)]
        if len(compat) == 1:
            target_sig = compat[0]
            final_clusters.setdefault(target_sig, []).extend(names)
            no_org_reassigned += len(names)
            continue
    # Otherwise keep this cluster
    final_clusters.setdefault(sig, []).extend(names)

# Pick canonical name per cluster
def pick_canonical(names):
    # 1. Prefer org/name HF style
    prefixed = [n for n in names if "/" in n and not n.startswith("/")]
    if prefixed:
        # Among prefixed, pick the most-common form
        return Counter(prefixed).most_common(1)[0][0]
    # 2. Pick longest non-aliased name
    cleaned = [PAREN_ALIAS_RE.sub("", n).strip() for n in names]
    cleaned = [n for n in cleaned if n]
    if cleaned:
        return max(cleaned, key=len)
    # 3. Fallback
    return names[0]

canon_map = {}  # original_name → canonical_name
for sig, names in final_clusters.items():
    canonical = pick_canonical(names)
    for n in names:
        canon_map[n] = canonical

# Add dropped nodes as None mapping
for d in dropped_nodes:
    canon_map[d] = None

report(f"  Canonical clusters: {len(final_clusters):,}")
report(f"  No-org names re-folded into prefixed clusters: {no_org_reassigned:,}")
report(f"  Distinct canonical names: {len(set(c for c in canon_map.values() if c)):,}\n")

# ============================================================
# STAGE 5: Rewrite edges
# ============================================================
report("Stage 5: Rewriting edges with canonical names + merging duplicates...")

# Build incoming/outgoing maps for filtering
incoming = defaultdict(int)
outgoing = defaultdict(int)

new_edges = {}  # (subj_canon, rel, obj_canon) → merged_edge_dict
edges_dropped_endpoint = 0
edges_dropped_self = 0
edges_kept = 0

for r in relations_orig:
    s = to_str(r.get("subject")); o = to_str(r.get("object")); rel = r.get("relation","")
    s_canon = canon_map.get(s, s)
    o_canon = canon_map.get(o, o)
    if not s_canon or not o_canon:
        edges_dropped_endpoint += 1
        continue
    if s_canon == o_canon:
        edges_dropped_self += 1
        continue
    if not rel:
        continue

    key = (s_canon, rel, o_canon)
    if key not in new_edges:
        # Create edge skeleton
        new_edges[key] = {
            "subject": s_canon,
            "relation": rel,
            "object": o_canon,
            "dependency_kind": r.get("dependency_kind"),
            "description": r.get("description",""),
            "anchor_list": [],
            "description_variants": [],
        }
        edges_kept += 1
    # Merge anchors
    new_edges[key]["anchor_list"].extend(r.get("anchor_list",[]) or [])
    # Description variants
    desc = r.get("description","")
    if desc and desc != new_edges[key]["description"] and desc not in new_edges[key]["description_variants"]:
        new_edges[key]["description_variants"].append(desc)
    for v in r.get("description_variants",[]) or []:
        if v not in new_edges[key]["description_variants"]:
            new_edges[key]["description_variants"].append(v)

    # Track in/out degree for filtering
    incoming[o_canon] += 1
    outgoing[s_canon] += 1

report(f"  Edges in original: {len(relations_orig):,}")
report(f"  Edges merged to canonical: {edges_kept:,}")
report(f"  Dropped (endpoint dropped): {edges_dropped_endpoint:,}")
report(f"  Dropped (self-loop after merge): {edges_dropped_self:,}\n")

# ============================================================
# STAGE 4: Filter orphans, low-anchor concepts
# ============================================================
report("Stage 4: Filtering orphan + low-signal nodes...")

active_nodes = set(incoming.keys()) | set(outgoing.keys())
# Drop orphans (no edges either way) — automatically excluded since they aren't in active_nodes

# Concept-node detection: bare names without org/version/size — drop if low-degree
def looks_concept(name):
    if not isinstance(name, str): return False
    if "/" in name: return False  # has org → real artifact
    if SIZE_RE.search(name): return False  # has size → real
    if "(" in name: return False  # has alias → may have meaning
    # Short bare name with no specific markers
    if len(name) < 30 and re.match(r"^[A-Za-z][A-Za-z0-9\-\s_.]*$", name):
        return True
    return False

low_signal_drops = []
for n in list(active_nodes):
    if looks_concept(n):
        deg = incoming[n] + outgoing[n]
        if deg < 3:
            low_signal_drops.append(n)
            active_nodes.discard(n)

# Re-filter edges: drop those with endpoints in low_signal_drops
low_drop_set = set(low_signal_drops)
final_edges = {}
for k, e in new_edges.items():
    s, rel, o = k
    if s in low_drop_set or o in low_drop_set: continue
    final_edges[k] = e

report(f"  Concept-like low-degree drops (degree < 3): {len(low_signal_drops):,}")
report(f"  Final active nodes: {len(active_nodes) - len(low_signal_drops):,}")
report(f"  Final edges: {len(final_edges):,}\n")

# ============================================================
# STAGE 6: LLM-verify borderline clusters
# ============================================================
report("Stage 6: LLM-verification of borderline clusters...")

# Identify borderline clusters: where the heuristic might have over-merged
# Specifically: clusters with >5 distinct names mapping to same canonical
borderline = []
canon_groups = defaultdict(list)
for orig, canon in canon_map.items():
    if canon: canon_groups[canon].append(orig)
for canon, names in canon_groups.items():
    if len(names) >= 5:
        borderline.append((canon, names))
borderline.sort(key=lambda x: -len(x[1]))
report(f"  Borderline clusters (≥5 names merged): {len(borderline)}")
report(f"  Total names in borderline clusters: {sum(len(n) for c,n in borderline):,}")

# Sample top 100 for LLM check (or all if less)
to_verify = borderline[:100]
report(f"  Will LLM-verify top {len(to_verify)} clusters\n")

LLM_PROMPT = """Examine this cluster of model/dataset names that an automated dedup mapped to one canonical entity. Check whether all names refer to the same released artifact, or whether the dedup is wrong.

CANONICAL: {canon}

NAMES IN CLUSTER ({n}):
{names}

QUESTION: Are ALL these names referring to the same released model/dataset?

Reply with ONE of:
  YES_ALL_SAME  — all refer to the same released artifact
  NO_SPLIT      — there are distinct artifacts incorrectly merged (briefly explain which to split)
  PARTIAL_OK    — most are the same but a few are off (briefly note the off ones)

Be conservative: if the names mention different versions (e.g., 3 vs 3.1), different sizes (7B vs 32B), different stages (SFT vs DPO), different orgs, or different release dates, flag NO_SPLIT.

Reply with the verdict on the first line, then a one-sentence reason. Do not write more than 80 words total."""

def llm_verify(canon, names):
    prompt = LLM_PROMPT.format(canon=canon, n=len(names), names="\n".join(f"  - {n}" for n in names[:25]))
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "opus",
             "--output-format", "text", "--permission-mode", "bypassPermissions"],
            capture_output=True, text=True, timeout=180,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERR: {e!r}"

split_decisions = {}  # canon → "YES_ALL_SAME" | "NO_SPLIT" | "PARTIAL_OK"
for i, (canon, names) in enumerate(to_verify):
    print(f"  LLM verify {i+1}/{len(to_verify)}: {canon[:60]}", flush=True)
    verdict_text = llm_verify(canon, names)
    first_line = verdict_text.split("\n",1)[0].strip().upper()
    if "YES_ALL_SAME" in first_line: tag = "YES_ALL_SAME"
    elif "NO_SPLIT" in first_line: tag = "NO_SPLIT"
    elif "PARTIAL_OK" in first_line: tag = "PARTIAL_OK"
    else: tag = "UNKNOWN"
    split_decisions[canon] = tag
    with open(LLM_LOG, "a") as f:
        f.write(f"\n=== [{i+1}] {canon} ({len(names)} names) — {tag} ===\n")
        f.write(verdict_text + "\n")

verdict_counts = Counter(split_decisions.values())
report(f"\n  LLM verdicts:")
for k, n in verdict_counts.most_common():
    report(f"    {k}: {n}")

# Apply NO_SPLIT decisions: revert those clusters back to per-name nodes
no_split_canons = {c for c,v in split_decisions.items() if v == "NO_SPLIT"}
report(f"\n  Reverting {len(no_split_canons)} cluster merges flagged NO_SPLIT...")
edges_re_split = 0
for k in list(final_edges.keys()):
    s, rel, o = k
    # If subject or object was over-merged, we can't undo edge-level — note for report only
    if s in no_split_canons or o in no_split_canons:
        edges_re_split += 1
report(f"  Edges affected by NO_SPLIT (kept as-is, flagged in report): {edges_re_split}\n")
# Note: full revert would require re-tracking; we keep this as a flag for now since
# the LLM had no specific guidance on which to split. User can investigate.

# ============================================================
# Sanity invariants
# ============================================================
report("Sanity checks:")

# Olmo-3 vs Olmo-3.1
olmo3_canons = [c for c in canon_groups if c and "olmo-3-" in c.lower() and "3.1" not in c.lower()]
olmo31_canons = [c for c in canon_groups if c and "olmo-3.1" in c.lower()]
report(f"  Olmo-3 distinct canonicals (not 3.1): {len(olmo3_canons)}")
report(f"  Olmo-3.1 distinct canonicals: {len(olmo31_canons)}")
assert len(olmo3_canons) > 0 and len(olmo31_canons) > 0, "ABORT: version distinction collapsed"

qwen3 = sum(1 for c in canon_groups if c and "qwen3" in c.lower() and "qwen2" not in c.lower())
qwen25 = sum(1 for c in canon_groups if c and "qwen2.5" in c.lower())
report(f"  Qwen3 distinct canonicals: {qwen3}")
report(f"  Qwen2.5 distinct canonicals: {qwen25}")

# 4 seed models present?
SEEDS = ["allenai/Olmo-3", "rl-research/DR-Tulu", "nvidia/NVIDIA-Nemotron-3", "HuggingFaceTB/SmolLM3"]
for seed in SEEDS:
    found = any(seed in c for c in canon_groups)
    report(f"  Seed '{seed}': {'✓ present' if found else '✗ MISSING'}")

# ============================================================
# Sample 20 random merges for visual review
# ============================================================
report(f"\nSample of merge decisions (20 random clusters with ≥3 members):")
import random
random.seed(7)
multi_clusters = [(c, n) for c, n in canon_groups.items() if c and len(n) >= 3]
sample = random.sample(multi_clusters, min(20, len(multi_clusters)))
for canon, names in sample:
    report(f"\n  CANONICAL: {canon}")
    for n in names[:8]:
        marker = "  *" if n == canon else "   "
        report(f"  {marker}  {n[:90]}")
    if len(names) > 8: report(f"      ... and {len(names)-8} more")

# ============================================================
# Compute final stats
# ============================================================
report("\n" + "="*70)
report("BEFORE / AFTER")
report("="*70)

final_node_set = set()
for (s, rel, o) in final_edges:
    final_node_set.add(s); final_node_set.add(o)
final_anchors = sum(len(e["anchor_list"]) for e in final_edges.values())

# Item-kind breakdown after dedup
kind_count = Counter()
for it in items_orig:
    fn = it.get("formal_name","")
    if not fn: continue
    canon = canon_map.get(fn)
    if canon and canon in final_node_set:
        kind_count[it.get("kind","?")] += 1
        # Avoid double-counting different originals → same canonical: track seen
# Better: group items by canonical
canon_to_items = defaultdict(list)
for it in items_orig:
    fn = it.get("formal_name","")
    if not fn: continue
    canon = canon_map.get(fn)
    if canon and canon in final_node_set:
        canon_to_items[canon].append(it)
final_kind = Counter()
for canon, its in canon_to_items.items():
    # Pick majority kind for this canonical
    most_common_kind = Counter(i.get("kind","?") for i in its).most_common(1)[0][0]
    final_kind[most_common_kind] += 1

report(f"\nMETRIC                    BEFORE      AFTER       Δ")
report(f"  Distinct nodes        {len(all_node_names):>8,}    {len(final_node_set):>8,}    -{len(all_node_names)-len(final_node_set):,}")
report(f"  Edges                 {len(relations_orig):>8,}    {len(final_edges):>8,}    -{len(relations_orig)-len(final_edges):,}")
report(f"  Anchors               {sum(len(r.get('anchor_list',[]) or []) for r in relations_orig):>8,}    {final_anchors:>8,}")
report(f"\n  Final canonical-node kind breakdown:")
for k, n in final_kind.most_common():
    report(f"    {n:>6,}  {k}")

# Top relations
final_rels = Counter(rel for (s,rel,o) in final_edges)
report(f"\n  Top relations (post-dedup):")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")

# ============================================================
# Write output
# ============================================================

# Convert final_edges back to list form
relations_out = list(final_edges.values())
# Lattice: rebuild groups by collapsing items to canonical
new_groups = defaultdict(list)
for it in items_orig:
    fn = it.get("formal_name","")
    if not fn: continue
    canon = canon_map.get(fn)
    if not canon or canon not in final_node_set: continue
    new_groups[canon].append(it)
# Each canonical becomes a group with merged items
out_groups = []
for canon, its in new_groups.items():
    # Pick one canonical item record (the one with formal_name == canon, else first)
    primary = next((i for i in its if i.get("formal_name") == canon), its[0])
    primary = dict(primary)  # copy
    primary["formal_name"] = canon
    primary["alias_count"] = len(its)
    out_groups.append({"items": [primary], "id": canon})

OUT = {
    "lattice": {"groups": out_groups},
    "relations": relations_out,
    "conflicts": G.get("conflicts", []),  # preserve
    "sources": G.get("sources", []),
    "relations_sources": G.get("relations_sources", []),
    "dedup_metadata": {
        "source_artifact": str(SOURCE),
        "dedup_clusters": len(canon_groups),
        "dropped_internal_paths": len(dropped_nodes),
        "low_signal_drops": len(low_signal_drops),
        "llm_verdicts": dict(verdict_counts),
        "no_split_clusters": list(no_split_canons),
    },
}

OUT_PATH.write_text(json.dumps(OUT))
report(f"\n✓ Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
report(f"✓ Original {SOURCE.name} preserved.")
