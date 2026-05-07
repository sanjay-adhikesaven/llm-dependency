#!/usr/bin/env python3
"""Reproduce paper Tables 2, 4, and 5 from the merged ModSleuth graph.

Tables:

  * **Table 2** (Recovered dependency edges grouped by audit role)
        Direct/Indirect counts per audit-role group, computed verbatim
        from each relation's ``relation`` and ``dependency_kind`` fields.
  * **Table 4** (Per-target ancestor counts and max depths)
        BFS over the full edge set from each target's seeds; ancestors
        are unique objects reachable from the seed set, max-depth is the
        longest path. Seed selection per target is configurable below;
        the defaults reproduce the seven rows in the paper.
  * **Table 5** (Source-type distribution of recovered operations)
        Each relation's ``anchor_list`` is classified into source-type
        categories. Operations supported by exactly one category are
        counted in that category; operations supported by anchors from
        multiple categories fall in *Multiple source types*.

Usage:

    python compute_graph_stats.py --merge-artifact ../data/merge_artifact.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


# ─── Table 2 grouping ─────────────────────────────────────────────────

AUDIT_ROLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Training / evaluation inputs",     ["trained_on", "used_for_evaluation", "used_for_ablation"]),
    ("Upstream model operations on data", ["generated_by", "filtered_by", "transformed_by", "embedded_by"]),
    ("Weight-level model lineage",        ["trained_from", "merged_from", "quantized_from"]),
    ("Methodology / audit influence",     ["inspired_by", "decontaminated_against"]),
]


def table2(relations: list[dict]) -> str:
    """Return a printable Table 2."""
    total = len(relations)
    by_pair = Counter(
        (r["relation"], r.get("dependency_kind", "?")) for r in relations
    )

    lines = [
        f"Table 2 — Recovered dependency edges by audit role  (over {total} edges)",
        "",
        f"{'Audit role':<38} {'Direct':>7} {'Indirect':>9} {'Count':>6} {'Share':>7}",
        "-" * 72,
    ]
    grand_d = grand_i = 0
    for role, rel_types in AUDIT_ROLE_GROUPS:
        d = sum(by_pair.get((rt, "direct"),   0) for rt in rel_types)
        i = sum(by_pair.get((rt, "indirect"), 0) for rt in rel_types)
        n = d + i
        share = n / total if total else 0.0
        grand_d += d; grand_i += i
        lines.append(f"{role:<38} {d:>7} {i:>9} {n:>6} {share*100:>6.1f}%")
    grand_total = grand_d + grand_i
    lines.append("-" * 72)
    lines.append(f"{'Total':<38} {grand_d:>7} {grand_i:>9} {grand_total:>6} {'100.0%':>7}")
    return "\n".join(lines)


# ─── Table 4 BFS configuration ────────────────────────────────────────

# Per-target seed-node lists. The defaults reproduce the seven rows in
# Table 4 of the paper using the canonical-id convention used by the
# lattice. If your reproduced numbers diverge, adjust the seed list:
# the seven rows in the paper are computed from per-investigation seed
# expansions, so the precise seed set is target-specific.
TABLE4_TARGETS: list[tuple[str, list[str]]] = [
    ("Olmo-3-Instruct",      ["allenai/Olmo-3-7B-Instruct",
                              "allenai/Olmo-3-32B-Instruct"]),
    ("Olmo-3-Think",         ["allenai/Olmo-3-7B-Think",
                              "allenai/Olmo-3-32B-Think"]),
    ("Olmo-3-Base",          ["allenai/Olmo-3-1025-7B",
                              "allenai/Olmo-3-1125-32B"]),
    ("Nemotron-3-Super",     ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"]),
    ("Nemotron-3-Nano-Base", ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"]),
    ("DR-Tulu",              ["rl-research/dr-tulu-8b"]),
    ("SmolLM3-Base",         ["HuggingFaceTB/SmolLM3-3B-Base"]),
]


def table4(relations: list[dict], lattice_groups: list[dict]) -> str:
    """Return a printable Table 4."""
    adj: dict[str, set[str]] = defaultdict(set)
    for r in relations:
        adj[r["subject"]].add(r["object"])
    valid_ids = {g["id"] for g in lattice_groups}

    def bfs(seeds: list[str]) -> tuple[int, int]:
        seeds = [s for s in seeds if s in valid_ids]
        if not seeds:
            return 0, 0
        depth = {s: 0 for s in seeds}
        q = deque(seeds)
        while q:
            n = q.popleft()
            for nb in adj[n]:
                if nb not in depth:
                    depth[nb] = depth[n] + 1
                    q.append(nb)
        return len(depth) - len(seeds), max(depth.values()) if depth else 0

    lines = [
        "Table 4 — Per-target ancestor counts and max depths",
        "",
        f"{'Target':<25} {'Ancestors':>10} {'Max depth':>10}",
        "-" * 50,
    ]
    for name, seeds in TABLE4_TARGETS:
        n_anc, md = bfs(seeds)
        lines.append(f"{name:<25} {n_anc:>10} {md:>10}")
    lines.append("-" * 50)
    # Aggregate "nodes" follows the paper convention: distinct
    # subject+object names across all edges (≈ lattice groups + a few
    # off-lattice nodes that appear only as edge endpoints).
    n_nodes    = len({r["subject"] for r in relations} | {r["object"] for r in relations})
    n_edges    = len(relations)
    n_anchors  = sum(len(r.get("anchor_list") or []) for r in relations)
    lines.append(f"Aggregate graph: {n_nodes} nodes, {n_edges} edges, {n_anchors} anchors")
    return "\n".join(lines)


# ─── Table 5 source-type classification ───────────────────────────────

_RX_PDF      = re.compile(r"\.pdf(\b|$)|arxiv\.org/(abs|pdf)/", re.I)
_RX_HF_BLOG  = re.compile(r"https?://huggingface\.co/blog/", re.I)
_RX_BLOG     = re.compile(r"://[^/]+\.(blog|substack)\.|/blog/|/announce", re.I)
# HF card: only huggingface.co URLs (model / dataset / collection
# pages). Cached card markdown saved under /batch/<topic>.md is treated
# as code (it sits in the same workflow tree as scripts and configs).
_RX_HF_CARD  = re.compile(r"https?://huggingface\.co/(?!blog/)", re.I)
# Code: training/eval scripts, configs, and downloaded code-repo files.
# Includes .md files because the deeper-nested ones are READMEs/docs
# from cloned code repos (not HF cards).
_RX_CODE_EXT  = re.compile(r"\.(py|yaml|yml|sh|json|jsonl|tsv|csv|toml|cfg|ini|cu|cpp|c|h|md|txt)(\b|$)", re.I)
_RX_CODE_PATH = re.compile(r"/configs?/|/scripts?/|/recipes?/|/training/|/pretraining/|"
                           r"/midtraining/|/finetuning/|/post[-_]?training/|/data_prep/|"
                           r"/resources_servers/|github\.com/[^/]+/[^/]+/(blob|raw|tree)/", re.I)


def classify_source(src: str) -> str:
    """Return one of: hf_card, code, pdf, blog, other.

    Order matters: PDF / HF blog / generic blog are checked first because
    they are the most specific. HF cards are only ``huggingface.co``
    URLs. Anything else with a code extension or a code-repo path is
    classified as *code*. Anything left is *other docs* (which mostly
    captures stray markdown / readme references that don't sit inside
    a recognized code repo).
    """
    s = src.lower()
    if _RX_PDF.search(s):     return "pdf"
    if _RX_HF_BLOG.search(s): return "blog"
    if _RX_BLOG.search(s):    return "blog"
    if _RX_HF_CARD.search(s): return "hf_card"
    # Anything left that has a code/script extension or sits inside a
    # cloned code repo is "training code / scripts".
    if _RX_CODE_PATH.search(s) or _RX_CODE_EXT.search(s):
        return "code"
    return "other"


def relation_source_set(rel: dict) -> set[str]:
    out = set()
    for a in (rel.get("anchor_list") or []):
        if not isinstance(a, dict):
            continue
        src = a.get("source") or a.get("source_id") or ""
        if isinstance(src, str) and src:
            out.add(classify_source(src))
    return out


_TABLE5_LABELS = [
    ("hf_card", "Only Hugging Face cards"),
    ("code",    "Only training code / scripts"),
    ("pdf",     "Only PDFs / technical reports"),
    ("blog",    "Only release blogs / docs"),
    ("other",   "Only other docs"),
    ("multi",   "Multiple source types"),
]


def table5(relations: list[dict]) -> str:
    """Return a printable Table 5."""
    counts: Counter = Counter()
    for r in relations:
        types = relation_source_set(r)
        if not types:
            continue
        if len(types) == 1:
            counts[next(iter(types))] += 1
        else:
            counts["multi"] += 1
    total = sum(counts.values())

    lines = [
        f"Table 5 — Source-type distribution of recovered operations  (over {total} operations with anchors)",
        "",
        f"{'Source support':<35} {'Count':>6} {'Share':>7}",
        "-" * 55,
    ]
    for key, label in _TABLE5_LABELS:
        n = counts.get(key, 0)
        share = n / total if total else 0.0
        lines.append(f"{label:<35} {n:>6} {share*100:>6.1f}%")
    lines.append("-" * 55)
    lines.append(f"{'Total operations':<35} {total:>6} {'100.0%':>7}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--merge-artifact", required=True, type=Path,
                   help="Path to ModSleuth's merged graph JSON.")
    p.add_argument("--tables", default="2,4,5",
                   help="Comma-separated subset of {2,4,5} to print.")
    args = p.parse_args()

    if not args.merge_artifact.exists():
        sys.exit(f"merge artifact not found: {args.merge_artifact}")
    G = json.loads(args.merge_artifact.read_text())
    relations = G.get("relations") or []
    groups    = G.get("lattice", {}).get("groups", []) or []
    if not relations:
        sys.exit("merge artifact has no `relations` field")

    wanted = {t.strip() for t in args.tables.split(",") if t.strip()}
    if "2" in wanted:
        print(table2(relations));  print()
    if "4" in wanted:
        print(table4(relations, groups));  print()
    if "5" in wanted:
        print(table5(relations));  print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
