#!/usr/bin/env python3
"""V8 cross-org + suffix-aware dedup — patterns V1-V7 explicitly avoided.

Candidate signals (graph-wide):
  1. Cross-org bare-lex match: same bare-name lex across different orgs
       e.g., OpenAI/GPT-2  ↔  openai-community/gpt2
  2. Suffix-stripping pairs: X and X-{turbo, Instruct, Base, ...} that may alias
       (Opus decides; most -Instruct/-Base pairs are correctly distinct)
  3. Bare-no-slash vs prefixed family: short bare name with same lex as
     bare-part of one or more prefixed nodes
       e.g., "OLMo 7B" ↔ "allenai/Olmo-3-1025-7B"
  4. Bare descriptive vs prefixed leaf: lex-collapse match
       e.g., "Dolma 3 FastText Quality Classifier" ↔ "allenai/dolma3-fasttext-quality-classifier"

Each candidate cluster → Opus 4.7 + --effort max with strict don't-merge guidance:
NEVER merge across versions/sizes/stages/dates/distinct-orgs.

Output: merge_artifact_deduped_v8.json
"""
from __future__ import annotations
import json, re, subprocess, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PARALLEL_WORKERS = 24
MODEL = "claude-opus-4-7"
EFFORT = "max"

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
RUN_DIR = REPO / "storage/runs/c6c6dfd9-0fb3-4d87-a5a7-01533c3af16d"
V7_IN = RUN_DIR / "merge_artifact_deduped_v7.json"
V8_OUT = RUN_DIR / "merge_artifact_deduped_v8.json"
REPORT = REPO / "run-logs/DEDUP_V8_REPORT.txt"
VERDICTS = REPO / "run-logs/DEDUP_V8_VERDICTS.txt"

REPORT.write_text(""); VERDICTS.write_text("")
def report(msg):
    with open(REPORT, "a") as f: f.write(msg + "\n")
    print(msg)

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

def anchor_key(a): return a.get("source") or a.get("url") or a.get("path") or ""
sample_anchor = {}
for e in edges:
    for n in (e.get("subject"), e.get("object")):
        if n and n not in sample_anchor:
            anchors = e.get("anchor_list") or []
            if anchors:
                sample_anchor[n] = anchor_key(anchors[0])[:120]

report("="*70)
report(f"V8 CROSS-ORG + SUFFIX DEDUP  (model: {MODEL} effort: {EFFORT})")
report("="*70)
report(f"Source: {V7_IN.name}")
report(f"Nodes: {len(all_nodes):,}  ·  Edges: {len(edges):,}\n")

# ============================================================
# Helpers
# ============================================================
ORG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)/(.+)$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
def lex_collapse(s): return NON_ALNUM_RE.sub("", s.lower())
def split_org(n):
    m = ORG_RE.match(n)
    if m: return m.group(1).lower(), m.group(2)
    return None, n

# Cap candidates per cluster
MAX_CLUSTER_SIZE = 6

# ============================================================
# Signal 1: Cross-org bare-lex match
# ============================================================
report("Signal 1: cross-org bare-lex match...")
by_bare_lex = defaultdict(list)  # lex(bare) → list of (org, full_name)
for n in all_nodes:
    org, bare = split_org(n)
    bl = lex_collapse(bare)
    if bl: by_bare_lex[bl].append((org, n))

cross_org_clusters = []
for bl, members in by_bare_lex.items():
    orgs = {org for org, _ in members if org}
    bare_only = [n for org, n in members if org is None]
    if len(orgs) >= 2 or (len(orgs) >= 1 and bare_only):
        # Different orgs (or org+bare-only) sharing same bare lex
        names = sorted({n for _, n in members}, key=lambda x: -degree[x])
        if 2 <= len(names) <= MAX_CLUSTER_SIZE:
            cross_org_clusters.append(names)
        elif len(names) > MAX_CLUSTER_SIZE:
            anchor = names[0]
            for i in range(1, len(names), MAX_CLUSTER_SIZE-1):
                cross_org_clusters.append([anchor] + names[i:i+MAX_CLUSTER_SIZE-1])
report(f"  Cross-org clusters: {len(cross_org_clusters)}")

# ============================================================
# Signal 2: Suffix-stripping pairs
# ============================================================
report("\nSignal 2: suffix-stripping pairs...")
SUFFIXES = ["-turbo", "-Turbo", "-Instruct", "-instruct", "-Base", "-base",
            "-Chat", "-chat", "-it", "-IT", "-hf", "-HF", "-preview", "-Preview"]
suffix_pairs = []
seen_pairs = set()
for n in all_nodes:
    for suf in SUFFIXES:
        if n.endswith(suf):
            stripped = n[:-len(suf)]
            if stripped in all_nodes:
                pair = tuple(sorted([n, stripped], key=lambda x: -degree[x]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    suffix_pairs.append(list(pair))
report(f"  Suffix-stripping pairs: {len(suffix_pairs)}")

# ============================================================
# Signal 3: Bare-no-slash containing/matching prefixed (loose)
# ============================================================
report("\nSignal 3: bare-no-slash → prefixed-bare lex match...")
prefixed_bare_lex = defaultdict(list)  # lex(bare-of-prefixed) → [prefixed_names]
for n in all_nodes:
    if "/" in n and not n.startswith("/"):
        _, bare = split_org(n)
        bl = lex_collapse(bare)
        if bl: prefixed_bare_lex[bl].append(n)

bare_prefixed_clusters = []
seen_bare = set()
for n in all_nodes:
    if "/" in n: continue
    if not n.strip(): continue
    nl = lex_collapse(n)
    if len(nl) < 4: continue
    matches = prefixed_bare_lex.get(nl, [])
    if matches:
        members = sorted({n} | set(matches), key=lambda x: -degree[x])[:MAX_CLUSTER_SIZE]
        key = tuple(sorted(members))
        if key not in seen_bare:
            seen_bare.add(key)
            bare_prefixed_clusters.append(members)
report(f"  Bare↔prefixed clusters: {len(bare_prefixed_clusters)}")

# ============================================================
# Signal 4: Token-superset relations (one node's tokens fully contain another's)
# ============================================================
report("\nSignal 4: token-superset (loose, lex-overlap)...")
TOKEN_SPLIT_RE = re.compile(r"[\s\-_/.\[\](),=:]+")
def tokenize(name):
    return frozenset(t for t in TOKEN_SPLIT_RE.split(name.lower()) if t and len(t) > 1)
node_tokens = {n: tokenize(n) for n in all_nodes}
node_tokens = {n: t for n, t in node_tokens.items() if t}

token_to_nodes = defaultdict(list)
for n, toks in node_tokens.items():
    for t in toks: token_to_nodes[t].append(n)

# For each pair where one is a strict subset of the other AND superset has ≤2 extra tokens
superset_clusters = []
seen_super = set()
for n, toks in node_tokens.items():
    cnt = Counter()
    for t in toks:
        post = token_to_nodes[t]
        if len(post) > 200: continue
        for n2 in post:
            if n2 != n: cnt[n2] += 1
    for n2, shared in cnt.most_common(15):
        if shared < 3: break
        toks2 = node_tokens.get(n2, frozenset())
        # Is one a subset of other with ≤2 extra tokens?
        diff_a = len(toks - toks2); diff_b = len(toks2 - toks)
        smaller_tokens_count = min(len(toks), len(toks2))
        if smaller_tokens_count >= 3 and ((diff_a == 0 and diff_b <= 2) or (diff_b == 0 and diff_a <= 2)):
            pair = tuple(sorted([n, n2], key=lambda x: -degree[x]))
            if pair not in seen_super:
                seen_super.add(pair)
                superset_clusters.append(list(pair))
report(f"  Token-superset clusters: {len(superset_clusters)}")

# ============================================================
# Combine all candidates, dedupe
# ============================================================
all_cl = cross_org_clusters + suffix_pairs + bare_prefixed_clusters + superset_clusters
seen_total = set()
clusters = []
for cl in all_cl:
    if len(cl) < 2 or len(cl) > MAX_CLUSTER_SIZE: continue
    key = tuple(sorted(cl))
    if key in seen_total: continue
    seen_total.add(key)
    clusters.append(cl)

report(f"\nTOTAL deduped candidate clusters: {len(clusters)}")
size_dist = Counter(len(c) for c in clusters)
for sz in sorted(size_dist):
    report(f"  size={sz}: {size_dist[sz]}")

# Sample 8 clusters
report("\nSample clusters (8 random):")
import random; random.seed(7)
for cl in random.sample(clusters, min(8, len(clusters))):
    report(f"\n  CLUSTER (n={len(cl)}):")
    for n in cl:
        report(f"    [d={degree[n]:>4}] {n[:90]}")

# ============================================================
# LLM verification — parallel
# ============================================================
PROMPT = """You are a strict graph dedup verifier. Below is a candidate cluster of node names that loose blocking flagged as possibly the same released artifact.

Decide ONE of (output exactly one line, no preamble):

ALL_SAME :: {{canonical_index}} :: {{brief reason}}
  → All N items refer to the same released artifact. {{canonical_index}} = 0-based index of canonical name (prefer most-canonical/most-popular HF org/name form).

PARTIAL :: {{merge_indices}} :: {{canonical_within}} :: {{brief reason}}
  → A subset is the same artifact. {{merge_indices}} = comma-separated 0-based indices that should merge. {{canonical_within}} = index within those merge_indices.

ALL_DISTINCT :: {{brief reason}}
  → Every item is its own released artifact.

CRITICAL — be strict. NEVER merge across:
  - Different version numbers (3 vs 3.1, gpt-3.5 vs gpt-4)
  - Different sizes (7B vs 13B vs 32B)
  - Different stages (Base vs Instruct vs Chat — usually distinct released checkpoints with different weights)
  - Different release dates / API snapshots (gpt-4o-2024-08-06 ≠ gpt-4o)
  - Subsets vs parent (gpqa_diamond ≠ gpqa)
  - Distinct community re-releases (Open-Orca/FLAN ≠ SirNeural/flan_v2)

DO merge when:
  - Same artifact, different surface forms (casing, hyphenation): "MMLU" ↔ "cais/mmlu"
  - Same artifact across orgs that turn out to be the SAME: "OpenAI/GPT-2" ↔ "openai-community/gpt2" (the actual HF org for GPT-2 is openai-community)
  - Bare descriptive name ↔ HF leaf for same artifact: "Dolma 3 FastText Quality Classifier" ↔ "allenai/dolma3-fasttext-quality-classifier"
  - API name ↔ -turbo suffix when research papers use them interchangeably: "gpt-3.5" ↔ "gpt-3.5-turbo" → MERGE (both refer to same chat API in research context)

CANDIDATE CLUSTER ({n_items} items):
{cluster_text}
"""

def build_cluster_text(members):
    lines = []
    for i, n in enumerate(members):
        d = degree.get(n, 0)
        a = sample_anchor.get(n, "(no anchor)")
        lines.append(f"  [{i}] {n}    degree={d}    sample_anchor={a}")
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

print(f"\nLaunching {len(clusters)} verifications with {PARALLEL_WORKERS} workers...\n")
t_start = time.time()
done = 0
tag_counts = Counter()
with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futs = {ex.submit(verify_cluster, i, m): i for i, m in enumerate(clusters)}
    for fut in as_completed(futs):
        try:
            idx, tag, dt = fut.result()
            done += 1
            tag_counts[tag] += 1
            if done % 25 == 0 or done == len(clusters):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(clusters) - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{len(clusters)}] {tag:<14} dt={dt:>5.0f}s  elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        except Exception as e:
            print(f"  cluster {futs[fut]} FAILED: {e!r}")

t_total = time.time() - t_start
report(f"\nDone in {t_total:.0f}s ({t_total/60:.1f} min)")
report(f"Tag counts:")
for t, n in tag_counts.most_common():
    report(f"  {n:>4}  {t}")

# ============================================================
# Apply via union-find
# ============================================================
parent = {n: n for n in all_nodes}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def unite(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for idx, (tag, payload, members) in results.items():
    if tag == "ALL_SAME":
        canonical = members[payload["canonical_idx"]]
        for m in members: unite(m, canonical)
    elif tag == "PARTIAL":
        merge_subset = [members[i] for i in payload["merge_indices"]]
        canonical = merge_subset[payload["canonical_within"]]
        for m in merge_subset: unite(m, canonical)

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

report(f"\nNodes merged: {nodes_merged}")

# Show merge groups (just ones with >1 member)
report("\nMerge groups applied (sample of largest by total degree):")
merge_groups = [(c, members) for c, members in
                ((canon_map[m], component_members[find(m)])
                 for m in {find(n) for n in canon_map.values()}.union())
                if len(members) > 1]
seen_canon = set()
for canon, members in merge_groups[:20]:
    if canon in seen_canon: continue
    seen_canon.add(canon)
    report(f"  → {canon}")
    for m in sorted(members, key=lambda x: -degree[x]):
        report(f"    [d={degree[m]:>4}] {m[:80]}")

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

report(f"\nEdges before V8: {len(edges):,}")
report(f"Edges after V8:  {len(new_edges):,}  (Δ {len(edges)-len(new_edges):+,})")
report(f"Self-loops collapsed: {self_loops}")

# ============================================================
# Sanity
# ============================================================
final_node_set = set()
for k in new_edges:
    final_node_set.add(k[0]); final_node_set.add(k[2])

olmo3 = [n for n in final_node_set if "olmo-3-" in n.lower().replace(" ","-") and "3.1" not in n.lower()]
olmo31 = [n for n in final_node_set if "olmo-3.1" in n.lower()]
report(f"\nSanity:")
report(f"  Olmo-3 nodes: {len(olmo3)}")
report(f"  Olmo-3.1 nodes: {len(olmo31)}")
assert len(olmo3) > 0 and len(olmo31) > 0, "ABORT: Olmo-3/3.1 collapsed"

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
        "v8_clusters_verified": len(clusters),
        "v8_verdict_tags": dict(tag_counts),
        "v8_nodes_merged": nodes_merged,
        "v8_signals": {
            "cross_org": len(cross_org_clusters),
            "suffix": len(suffix_pairs),
            "bare_prefixed": len(bare_prefixed_clusters),
            "token_superset": len(superset_clusters),
        },
        "v8_model": MODEL, "v8_effort": EFFORT,
    },
}
V8_OUT.write_text(json.dumps(OUT))

final_anchors = sum(len(e.get("anchor_list", []) or []) for e in new_edges.values())
report(f"\n{'='*70}\nV8 FINAL\n{'='*70}")
report(f"  Distinct nodes: {len(final_node_set):,}")
report(f"  Lattice groups: {len(out_groups):,}")
report(f"  Edges: {len(new_edges):,}")
report(f"  Anchors: {final_anchors:,}")
final_rels = Counter(e.get("relation","") for e in new_edges.values())
report(f"\n  Top relations:")
for r, n in final_rels.most_common(12):
    report(f"    {n:>6,}  {r}")
report(f"\n✓ Wrote {V8_OUT} ({V8_OUT.stat().st_size:,} bytes)")
