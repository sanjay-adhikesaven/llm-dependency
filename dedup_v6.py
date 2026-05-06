#!/usr/bin/env python3
"""V6 whole-graph node dedup — block-then-verify (REVISED).

Lessons learned from dry-run: connected-components clustering with noisy
signals (anchor co-citation, neighbor Jaccard) explodes — single false-positive
pairs fuse unrelated hubs into mega-clusters. Fix: generate SMALL candidate
clusters per-node (not connected components), and use ONLY high-precision
signals.

Candidate generation (per node):
  1. Lex-collapse blocking: nodes sharing alpha-only collapsed key
  2. Token-Jaccard ≥0.6 (per node, top-3 best candidates)
  3. Substring containment: short bare name fully contained in prefixed name

For each node, we form ONE candidate cluster of {node + its candidates},
clip to ≤6 members. Each cluster → 1 Opus call.

This deliberately runs MORE LLM calls but each on a tightly-scoped cluster,
giving much higher precision per dollar than mega-cluster verification.
"""
from __future__ import annotations
import json, re, subprocess, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PARALLEL_WORKERS = 24
MAX_CLUSTER_SIZE = 6
MIN_TOKEN_JACCARD = 0.60
TOP_K_TOKEN_CAND = 3
MODEL = "claude-opus-4-7"
EFFORT = "max"

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
RUN_DIR = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d"
V5_IN = RUN_DIR / "merge_artifact_deduped_v5.json"
V4_IN = RUN_DIR / "merge_artifact_deduped_v4.json"
V6_OUT = RUN_DIR / "merge_artifact_deduped_v6.json"
REPORT = REPO / "run-logs/DEDUP_V6_REPORT.txt"
VERDICTS = REPO / "run-logs/DEDUP_V6_VERDICTS.txt"

REPORT.write_text(""); VERDICTS.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

SOURCE = V5_IN if V5_IN.exists() else V4_IN
report("="*70)
report(f"V6 WHOLE-GRAPH NODE DEDUP (revised)")
report("="*70)
report(f"Source: {SOURCE.name}\n")
G = json.loads(SOURCE.read_text())
edges = G["relations"]
groups = G["lattice"]["groups"]

all_nodes = set()
for e in edges:
    if e.get("subject"): all_nodes.add(e["subject"])
    if e.get("object"):  all_nodes.add(e["object"])
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name")
        if fn: all_nodes.add(fn)
all_nodes = {n for n in all_nodes if isinstance(n, str) and n}

degree = Counter()
for e in edges:
    degree[e["subject"]] += 1; degree[e["object"]] += 1

report(f"Nodes: {len(all_nodes):,}")
report(f"Edges: {len(edges):,}\n")

# ============================================================
# Build cheap features per node
# ============================================================
TOKEN_SPLIT_RE = re.compile(r"[\s\-_/.\[\](),=:]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
def tokenize(name):
    return frozenset(t for t in TOKEN_SPLIT_RE.split(name.lower()) if t and len(t) > 1)
def lex_collapse(name):
    return NON_ALNUM_RE.sub("", name.lower())

node_tokens = {n: tokenize(n) for n in all_nodes}
node_tokens = {n: t for n, t in node_tokens.items() if t}
node_lex = {n: lex_collapse(n) for n in all_nodes}

# ============================================================
# Signal 1: Lex-collapse blocks (high precision)
# ============================================================
lex_groups = defaultdict(list)
for n, k in node_lex.items():
    if k: lex_groups[k].append(n)
lex_clusters = [sorted(v, key=lambda x: -degree[x])
                for v in lex_groups.values() if 2 <= len(v) <= MAX_CLUSTER_SIZE]
# Bigger lex groups: split into chunks of MAX_CLUSTER_SIZE keeping highest-degree as anchor
big_lex_clusters = []
for v in lex_groups.values():
    if len(v) > MAX_CLUSTER_SIZE:
        v_sorted = sorted(v, key=lambda x: -degree[x])
        anchor = v_sorted[0]
        rest = v_sorted[1:]
        for i in range(0, len(rest), MAX_CLUSTER_SIZE-1):
            big_lex_clusters.append([anchor] + rest[i:i+MAX_CLUSTER_SIZE-1])
report(f"Signal 1 (lex-collapse) clusters: {len(lex_clusters) + len(big_lex_clusters)}")
report(f"  small (size 2-{MAX_CLUSTER_SIZE}): {len(lex_clusters)}")
report(f"  big (split into chunks): {len(big_lex_clusters)}")

# Build set of lex-key for nodes
node_to_lex_cluster = {}
for ci, cl in enumerate(lex_clusters + big_lex_clusters):
    for n in cl:
        node_to_lex_cluster.setdefault(n, set()).add(ci)

# ============================================================
# Signal 2: Token-Jaccard top-K per node (high precision)
# ============================================================
token_to_nodes = defaultdict(list)
for n, toks in node_tokens.items():
    for t in toks:
        token_to_nodes[t].append(n)

def jaccard(a, b):
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

# For each node, find top-K candidates via token-Jaccard (not in same lex cluster already)
token_candidate_clusters = []
seen_pairs_in_token = set()
nodes_processed = 0
for n, toks in node_tokens.items():
    nodes_processed += 1
    # Find candidate set via token postings
    cnt = Counter()
    for t in toks:
        post = token_to_nodes[t]
        if len(post) > 200: continue
        for n2 in post:
            if n2 != n: cnt[n2] += 1
    # Compute Jaccard for top-Y candidates
    cand_jaccards = []
    for n2, shared in cnt.most_common(20):
        if shared < 2: break
        j = jaccard(toks, node_tokens.get(n2, frozenset()))
        if j >= MIN_TOKEN_JACCARD:
            cand_jaccards.append((n2, j))
    cand_jaccards.sort(key=lambda x: -x[1])
    cand_jaccards = cand_jaccards[:TOP_K_TOKEN_CAND]
    if not cand_jaccards: continue
    # Build a cluster: {n} ∪ candidates, capped
    members = [n] + [c for c, _ in cand_jaccards]
    # If all members already in a single lex cluster, skip (already covered)
    lex_overlap = set.intersection(*[node_to_lex_cluster.get(m, set()) for m in members]) if all(m in node_to_lex_cluster for m in members) else set()
    if lex_overlap:
        continue
    # Dedupe by member-set
    key = tuple(sorted(members))
    if key in seen_pairs_in_token: continue
    seen_pairs_in_token.add(key)
    token_candidate_clusters.append(members[:MAX_CLUSTER_SIZE])

report(f"\nSignal 2 (token-Jaccard ≥{MIN_TOKEN_JACCARD}) per-node clusters: {len(token_candidate_clusters)}")

# ============================================================
# Signal 3: Substring containment (bare contained in prefixed)
# ============================================================
substring_clusters = []
seen_substring_keys = set()
# For every short, no-slash node, find prefixed nodes whose tail contains its lex-form.
short_bare_nodes = [n for n in all_nodes
                    if "/" not in n and 2 < len(node_lex[n]) < 30]
prefixed_nodes = [n for n in all_nodes if "/" in n and not n.startswith("/")]
prefixed_lex = {n: lex_collapse(n.split("/", 1)[1]) for n in prefixed_nodes}

for short in short_bare_nodes:
    sl = node_lex[short]
    if len(sl) < 3: continue
    # find prefixed nodes whose tail-lex equals or starts/ends with sl
    matches = []
    for p, pl in prefixed_lex.items():
        if pl == sl or pl.startswith(sl) or pl.endswith(sl):
            matches.append(p)
    if not matches: continue
    # Build cluster: short bare + its prefixed match candidates (highest-degree)
    matches.sort(key=lambda x: -degree[x])
    members = [short] + matches[:MAX_CLUSTER_SIZE-1]
    # Skip if already in lex cluster covering all
    lex_overlap = set.intersection(*[node_to_lex_cluster.get(m, set()) for m in members]) if all(m in node_to_lex_cluster for m in members) else set()
    if lex_overlap: continue
    key = tuple(sorted(members))
    if key in seen_substring_keys: continue
    seen_substring_keys.add(key)
    substring_clusters.append(members)

report(f"\nSignal 3 (substring-containment) clusters: {len(substring_clusters)}")

# ============================================================
# Combine + dedupe
# ============================================================
all_clusters = lex_clusters + big_lex_clusters + token_candidate_clusters + substring_clusters
seen_keys = set()
final_clusters = []
for cl in all_clusters:
    key = tuple(sorted(cl))
    if key in seen_keys: continue
    seen_keys.add(key)
    final_clusters.append(cl)

report(f"\nTOTAL deduped candidate clusters: {len(final_clusters)}")
size_dist = Counter(len(c) for c in final_clusters)
for sz in sorted(size_dist):
    report(f"  size={sz}: {size_dist[sz]}")

# ============================================================
# Helpers for prompt
# ============================================================
def anchor_key(a): return a.get("source") or a.get("url") or a.get("path") or ""

# Pre-index sample anchors per node
sample_anchor_per_node = {}
for e in edges:
    for n in (e.get("subject"), e.get("object")):
        if n and n not in sample_anchor_per_node:
            anchors = e.get("anchor_list") or []
            if anchors:
                sample_anchor_per_node[n] = anchor_key(anchors[0])[:140]

PROMPT = """You are a careful graph dedup verifier. Below is one CANDIDATE CLUSTER of node names that automated blocking flagged as possibly the same released artifact.

Decide ONE of (output exactly one line, no preamble):

ALL_SAME :: {{canonical_index}} :: {{brief reason}}
  → All N items refer to the same released artifact. {{canonical_index}} is the 0-based index of the best canonical name.

PARTIAL :: {{merge_indices}} :: {{canonical_within}} :: {{brief reason}}
  → A subset of items refer to the same artifact. {{merge_indices}} is comma-separated 0-based indices to merge (e.g. "0,2,3"). {{canonical_within}} is the index *within those merge_indices* (0..len-1) — e.g. "0" means first in your merge list. Other listed items remain distinct.

ALL_DISTINCT :: {{brief reason}}
  → Every item is its own released artifact (different versions, sizes, stages, dates, subsets, derivative orgs).

CRITICAL — NEVER merge across:
  - Different version numbers (3 vs 3.1 vs 3.2)
  - Different sizes (7B vs 13B vs 32B)
  - Different stages (SFT vs DPO vs RL vs Base vs Instruct)
  - Different dates / release tags (0625 vs 0925 vs 1025)
  - Parens-suffixes indicating subsets (e.g., MMLU "(STEM)" vs "(humanities)")
  - Different orgs hosting derivative versions (Open-Orca/FLAN vs SirNeural/flan_v2 — separate community re-releases)

DO merge across:
  - Pure surface-form variants (casing, hyphenation, underscores)
  - Bare name vs prefixed HF form (e.g., "MMLU" vs "cais/mmlu" — same dataset)
  - Aggregator-path vs canonical leaf for the same dataset

CANDIDATE CLUSTER ({n_items} items):
{cluster_text}
"""

def build_cluster_text(members):
    lines = []
    for i, n in enumerate(members):
        d = degree.get(n, 0)
        a = sample_anchor_per_node.get(n, "(none)")
        lines.append(f"  [{i}] {n}   degree={d}   sample_anchor={a}")
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

def parse_verdict(text, n_items):
    line = text.splitlines()[0].strip() if text else ""
    if line.startswith("ALL_SAME"):
        m = re.match(r"ALL_SAME\s*::\s*(\d+)\s*::\s*(.*)", line)
        if m and 0 <= int(m.group(1)) < n_items:
            return ("ALL_SAME", {"canonical_idx": int(m.group(1)), "reason": m.group(2)})
    elif line.startswith("PARTIAL"):
        m = re.match(r"PARTIAL\s*::\s*([\d,\s]+)\s*::\s*(\d+)\s*::\s*(.*)", line)
        if m:
            try:
                idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
                idxs = [i for i in idxs if 0 <= i < n_items]
                if len(idxs) >= 2:
                    ci = int(m.group(2))
                    if 0 <= ci < len(idxs):
                        return ("PARTIAL", {"merge_indices": idxs, "canonical_within": ci, "reason": m.group(3)})
            except Exception:
                pass
    elif line.startswith("ALL_DISTINCT"):
        m = re.match(r"ALL_DISTINCT\s*::\s*(.*)", line)
        return ("ALL_DISTINCT", {"reason": m.group(1) if m else ""})
    return ("UNPARSED", {"raw": text[:300]})

results_lock = Lock(); log_lock = Lock()
results = {}

def verify_cluster(idx, members):
    n = len(members)
    cluster_text = build_cluster_text(members)
    prompt = PROMPT.format(n_items=n, cluster_text=cluster_text)
    t0 = time.time()
    out, rc = call_opus(prompt)
    dt = time.time() - t0
    tag, payload = parse_verdict(out, n)
    with results_lock:
        results[idx] = (tag, payload, members)
    with log_lock:
        with open(VERDICTS, "a") as f:
            f.write(f"\n{'='*70}\n[cluster {idx}] n={n} rc={rc} dt={dt:.1f}s tag={tag}\n")
            for i, m in enumerate(members):
                f.write(f"  [{i}] {m}\n")
            f.write(f"VERDICT: {out}\n")
    return idx, tag, dt

# ============================================================
# Run all clusters in parallel
# ============================================================
print(f"\nLaunching {len(final_clusters)} LLM verifications with {PARALLEL_WORKERS} workers...\n")
t_start = time.time()
done = 0
tag_counts = Counter()
with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futs = {ex.submit(verify_cluster, i, m): i for i, m in enumerate(final_clusters)}
    for fut in as_completed(futs):
        try:
            idx, tag, dt = fut.result()
            done += 1
            tag_counts[tag] += 1
            if done % 25 == 0 or done == len(final_clusters):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(final_clusters) - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{len(final_clusters)}] {tag:<14} dt={dt:>5.0f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
        except Exception as e:
            print(f"  cluster {futs[fut]} FAILED: {e!r}")

t_total = time.time() - t_start
report(f"\nVerification done in {t_total:.0f}s ({t_total/60:.1f} min)")
report(f"Tag counts:")
for t, n in tag_counts.most_common():
    report(f"  {n:>4}  {t}")

# ============================================================
# Apply decisions (with union-find for transitive merges)
# ============================================================
# For each ALL_SAME cluster → union all members under canonical
# For each PARTIAL cluster → union only the merge_indices subset

parent = {n: n for n in all_nodes}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def unite(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for idx, (tag, payload, members) in results.items():
    if tag == "ALL_SAME":
        canonical = members[payload["canonical_idx"]]
        for m in members:
            unite(m, canonical)
    elif tag == "PARTIAL":
        merge_subset = [members[i] for i in payload["merge_indices"]]
        canonical = merge_subset[payload["canonical_within"]]
        for m in merge_subset:
            unite(m, canonical)

# Build canon_map: each node → its component representative,
# choosing the highest-degree (then prefixed-form preference) node as canonical
component_members = defaultdict(list)
for n in all_nodes:
    component_members[find(n)].append(n)

canon_map = {}
nodes_merged = 0
for root, members in component_members.items():
    if len(members) <= 1:
        canon_map[members[0]] = members[0]
        continue
    # Pick canonical: prefer prefixed (org/name) form, then highest degree
    prefixed = [m for m in members if "/" in m and not m.startswith("/")]
    pool = prefixed if prefixed else members
    canonical = max(pool, key=lambda m: (degree.get(m, 0), len(m)))
    for m in members:
        canon_map[m] = canonical
        if m != canonical: nodes_merged += 1

report(f"\nNodes merged into a different canonical: {nodes_merged}")

# ============================================================
# Re-rewrite edges with V6 canonicals
# ============================================================
report("Rewriting edges...")
new_edges = {}
self_loops = 0
for e in edges:
    s = canon_map.get(e["subject"], e["subject"])
    o = canon_map.get(e["object"], e["object"])
    rel = e["relation"]
    if s == o or not rel:
        self_loops += 1; continue
    key = (s, rel, o)
    if key not in new_edges:
        new_edges[key] = {
            "subject": s, "relation": rel, "object": o,
            "dependency_kind": e.get("dependency_kind"),
            "description": e.get("description", ""),
            "anchor_list": [], "description_variants": [],
        }
    new_edges[key]["anchor_list"].extend(e.get("anchor_list", []) or [])
    desc = e.get("description", "")
    if desc and desc != new_edges[key]["description"] and desc not in new_edges[key]["description_variants"]:
        new_edges[key]["description_variants"].append(desc)
    for v in e.get("description_variants", []) or []:
        if v not in new_edges[key]["description_variants"]:
            new_edges[key]["description_variants"].append(v)

report(f"  Edges before: {len(edges):,}")
report(f"  Edges after V6: {len(new_edges):,}  ({len(edges)-len(new_edges):+,} change)")
report(f"  Self-loops collapsed: {self_loops}")

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
assert aime_2024 and aime_2025, "ABORT: AIME year split collapsed"

mmlu_subj = [n for n in final_node_set if "cais/mmlu" in n.lower() and ("(" in n or "[" in n)]
report(f"  cais/mmlu subjects preserved: {len(mmlu_subj)}")

collapse_ratio = nodes_merged / max(len(all_nodes), 1)
report(f"  Collapse ratio: {collapse_ratio:.1%} ({nodes_merged}/{len(all_nodes)})")

# ============================================================
# Build groups, write output
# ============================================================
new_groups_d = defaultdict(list)
for g in groups:
    for it in g["items"]:
        fn = it.get("formal_name", "")
        if not fn: continue
        canon = canon_map.get(fn, fn)
        if canon in final_node_set:
            new_groups_d[canon].append(it)
out_groups = []
for canon, its in new_groups_d.items():
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
        "v6_clusters_verified": len(final_clusters),
        "v6_verdict_tags": dict(tag_counts),
        "v6_nodes_merged": nodes_merged,
        "v6_model": MODEL, "v6_effort": EFFORT,
        "v6_signals": {
            "lex_clusters_small": len(lex_clusters),
            "lex_clusters_big_split": len(big_lex_clusters),
            "token_clusters": len(token_candidate_clusters),
            "substring_clusters": len(substring_clusters),
        },
    },
}
V6_OUT.write_text(json.dumps(OUT))

final_anchors = sum(len(e["anchor_list"]) for e in new_edges.values())
report(f"\n{'='*70}\nV6 FINAL\n{'='*70}")
report(f"  Distinct nodes: {len(final_node_set):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(new_edges):,}")
report(f"  Anchors: {final_anchors:,}")
final_rels = Counter(rel for (s,rel,o) in new_edges)
report(f"\n  Top relations:")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")
report(f"\n✓ Wrote {V6_OUT} ({V6_OUT.stat().st_size:,} bytes)")
