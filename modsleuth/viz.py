#!/usr/bin/env python3
"""ModSleuth dependency-graph visualizer (tuned for 20k-edge graphs).

Loads a merged or cleaned graph JSON (output of ``modsleuth run merge`` or
``modsleuth dedup``) and serves an interactive vis-network frontend on
``http://<host>:<port>/``. Default initial view caps at the top 200 nodes
by degree (min-degree ≥ 10) with physics off; toggle force layout, ego
modes (1-hop / 2-hop), or raise the slider to explore further.

Run via the CLI:

    modsleuth viz --source path/to/graph.json --port 8102

or directly as a module:

    python -m modsleuth.viz --source path/to/graph.json --port 8102

Open: http://127.0.0.1:8102/
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def load_data(path: Path) -> dict:
    G = json.loads(path.read_text())
    groups = G.get("lattice", {}).get("groups", [])
    relations = G.get("relations", [])

    nodes: dict[str, dict] = {}
    family_counts: dict[str, int] = defaultdict(int)
    for grp in groups:
        family = grp.get("family") or grp.get("id") or ""
        for item in grp.get("items") or []:
            formal = item.get("formal_name")
            if not formal:
                continue
            family_counts[family] += 1
            links = item.get("links") or []
            primary_url = ""
            primary_kind = ""
            for ln in links:
                if isinstance(ln, dict) and ln.get("url"):
                    primary_url = ln["url"]; primary_kind = ln.get("kind") or ""; break
            nodes[formal] = {
                "id": formal,
                "kind": item.get("kind") or "model",
                "family": family,
                "identity": item.get("identity") or {},
                "aliases": item.get("aliases") or [],
                "links": links,
                "n_links": len(links),
                "primary_url": primary_url,
                "primary_link_kind": primary_kind,
                "description": item.get("description") or "",
                "in_degree": 0, "out_degree": 0,
            }

    edges: list[dict] = []
    for e in relations:
        s = e.get("subject"); o = e.get("object")
        if not s or not o: continue
        for name in (s, o):
            if name not in nodes:
                nodes[name] = {
                    "id": name, "kind": "off_lattice",
                    "family": "(off-lattice)", "identity": {}, "aliases": [],
                    "links": [], "n_links": 0,
                    "primary_url": "", "primary_link_kind": "",
                    "description": (e.get("description") or "")[:200],
                    "in_degree": 0, "out_degree": 0,
                }
        edges.append({
            "subject": s, "object": o, "object_id": o,
            "object_in_lattice": True,
            "relation": e.get("relation") or "",
            "dependency_kind": e.get("dependency_kind") or "direct",
            "description": e.get("description") or "",
            "anchor_list": e.get("anchor_list") or [],
            "operation_id": None, "batch": "v4",
        })

    for e in edges:
        nodes[e["subject"]]["out_degree"] += 1
        nodes[e["object_id"]]["in_degree"] += 1

    rel_counts: dict[str, int] = defaultdict(int)
    dep_kind_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        rel_counts[e["relation"]] += 1
        dep_kind_counts[e.get("dependency_kind") or "unknown"] += 1

    return {
        "lattice_path": str(path),
        "nodes": list(nodes.values()),
        "edges": edges,
        "operations": [],
        "stats": {
            "node_count": len(nodes),
            "lattice_node_count": sum(1 for n in nodes.values() if n.get("kind") in ("model", "dataset")),
            "off_lattice_node_count": sum(1 for n in nodes.values() if n.get("kind") == "off_lattice"),
            "edge_count": len(edges),
            "operation_count": 0,
            "family_count": len(family_counts),
            "families": sorted(family_counts.items(), key=lambda kv: -kv[1])[:50],
            "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
            "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
        },
    }


# Edge-relevance weights for seeded expansion. Higher = more lineage-bearing.
# Negative scores discount evaluation/citation clutter that would otherwise
# drown out the actual training graph during BFS.
_EDGE_RELEVANCE: dict[str, int] = {
    "trained_from": 8, "trained_on": 7, "generated_by": 6,
    "transformed_by": 5, "filtered_by": 5, "merged_from": 6,
    "composed_from": 5, "tokenized_by": 3, "deduplicated_by": 3,
    "decontaminated_by": 3, "released_with": 2, "inspired_by": 0,
    "used_for_ablation": -2, "used_for_evaluation": -4,
    "cited_as_baseline": -4,
}


def _score_edge(edge: dict) -> int:
    score = _EDGE_RELEVANCE.get(edge.get("relation") or "", 1)
    if (edge.get("dependency_kind") or "") == "direct":
        score += 2
    score += min(3, len(edge.get("anchor_list") or []))
    return score


def _resolve_seed(payload: dict, pattern: str) -> str | None:
    """Match `pattern` (case-insensitive substring) against each node's id +
    aliases; return the highest-degree match's id, or None."""
    needle = pattern.lower()
    candidates: list[tuple[int, str]] = []
    for n in payload["nodes"]:
        hay = (n["id"] + " " + " ".join(n.get("aliases") or [])).lower()
        if needle in hay:
            deg = n.get("in_degree", 0) + n.get("out_degree", 0)
            candidates.append((deg, n["id"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _seeded_expand(payload: dict, seed_id: str, *,
                   depth: int, target_size: int) -> dict:
    """BFS from `seed_id` up to `depth` hops, greedily admitting the
    highest-relevance-scored neighbors at each frontier until we hit
    `target_size` nodes. Edges between admitted nodes are kept."""
    nodes_by_id = {n["id"]: n for n in payload["nodes"]}
    if seed_id not in nodes_by_id:
        return payload

    incident: dict[str, list[dict]] = defaultdict(list)
    for e in payload["edges"]:
        incident[e["subject"]].append(e)
        incident[e["object_id"]].append(e)

    keep: set[str] = {seed_id}
    frontier: set[str] = {seed_id}
    for _ in range(max(1, depth)):
        if len(keep) >= target_size:
            break
        candidates: dict[str, int] = {}
        for src in frontier:
            for e in incident.get(src, []):
                other = e["object_id"] if e["subject"] == src else e["subject"]
                if not other or other in keep:
                    continue
                s = _score_edge(e)
                if s > candidates.get(other, -10**9):
                    candidates[other] = s
        if not candidates:
            break
        ranked = sorted(
            candidates.items(),
            key=lambda kv: (-kv[1],
                            -(nodes_by_id[kv[0]]["in_degree"]
                              + nodes_by_id[kv[0]]["out_degree"])),
        )
        next_frontier: set[str] = set()
        for other_id, _score in ranked:
            if len(keep) >= target_size:
                break
            keep.add(other_id)
            next_frontier.add(other_id)
        frontier = next_frontier

    pruned_nodes = [n for n in payload["nodes"] if n["id"] in keep]
    pruned_edges = [e for e in payload["edges"]
                    if e["subject"] in keep and e["object_id"] in keep]

    for n in pruned_nodes:
        n["in_degree"] = 0
        n["out_degree"] = 0
    by_id = {n["id"]: n for n in pruned_nodes}
    for e in pruned_edges:
        by_id[e["subject"]]["out_degree"] += 1
        by_id[e["object_id"]]["in_degree"] += 1

    rel_counts: dict[str, int] = defaultdict(int)
    dep_kind_counts: dict[str, int] = defaultdict(int)
    for e in pruned_edges:
        rel_counts[e["relation"]] += 1
        dep_kind_counts[e.get("dependency_kind") or "unknown"] += 1
    family_counts: dict[str, int] = defaultdict(int)
    for n in pruned_nodes:
        family_counts[n.get("family") or ""] += 1

    new_stats = dict(payload["stats"])
    new_stats.update({
        "node_count": len(pruned_nodes),
        "lattice_node_count": sum(1 for n in pruned_nodes
                                   if n.get("kind") in ("model", "dataset")),
        "off_lattice_node_count": sum(1 for n in pruned_nodes
                                       if n.get("kind") == "off_lattice"),
        "edge_count": len(pruned_edges),
        "family_count": len(family_counts),
        "families": sorted(family_counts.items(), key=lambda kv: -kv[1])[:50],
        "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
        "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
        "seed": seed_id,
    })
    return {**payload, "nodes": pruned_nodes, "edges": pruned_edges,
            "stats": new_stats}


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>graph v4 viewer</title>
<style>
  :root { color-scheme: light; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fafafa; color: #222; }
  #app { display: grid; grid-template-columns: 380px 1fr; height: 100vh; }
  #side { padding: 12px; overflow-y: auto; border-right: 1px solid #ddd; background: #fff; }
  #graph { position: relative; height: 100vh; }
  #graph-canvas { width: 100%; height: 100%; }
  h1 { font-size: 14px; margin: 0 0 8px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #777; margin: 14px 0 6px; }
  input[type=search], input[type=text], select, input[type=range], input[type=number] { width: 100%; box-sizing: border-box; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
  .row { margin: 6px 0; font-size: 13px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; background: #eef; color: #336; margin-right: 4px; }
  .kind-model { background: #d8eafd; color: #0a66c2; }
  .kind-dataset { background: #d3f0d3; color: #1a7f37; }
  .kind-off_lattice { background: #f3e3c0; color: #7a4f00; }
  .stat { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
  .stat .label { color: #777; }
  .filter-chip { display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0; border: 1px solid #ccc; border-radius: 12px; font-size: 11px; cursor: pointer; user-select: none; background: #fff; }
  .filter-chip.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  #detail { background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 12.5px; }
  #detail .name { font-weight: 600; font-size: 13.5px; word-break: break-all; }
  #detail .desc { color: #444; margin-top: 4px; line-height: 1.4; }
  #detail .alias { display: inline-block; background: #eef; padding: 1px 6px; border-radius: 3px; margin: 1px; font-size: 11px; }
  #detail a { color: #0969da; text-decoration: none; word-break: break-all; }
  .nav-results { margin-top: 6px; max-height: 280px; overflow-y: auto; border: 1px solid #eee; border-radius: 4px; }
  .nav-result { padding: 4px 8px; cursor: pointer; font-size: 12px; border-bottom: 1px solid #f4f4f4; }
  .nav-result:hover { background: #eaf3ff; }
  .nav-result .formal { font-weight: 500; }
  .nav-result .meta { color: #888; font-size: 11px; }
  details { margin: 6px 0; }
  details summary { cursor: pointer; font-size: 12px; padding: 3px 0; }
  #status-bar { position: absolute; top: 8px; right: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 4px 10px; font-size: 11px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; }
  .toolbar { position: absolute; top: 8px; left: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; display: flex; gap: 6px; flex-wrap: wrap; }
  .toolbar button { font-size: 11px; padding: 4px 8px; border: 1px solid #ccc; background: #fafafa; border-radius: 3px; cursor: pointer; }
  .toolbar button:hover { background: #fff; }
  .toolbar button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  .slider-row { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .slider-row .num { font-family: ui-monospace, monospace; min-width: 30px; text-align: right; }
  #empty-msg { text-align: center; color: #888; padding: 30px; font-size: 13px; }
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>graph v4 viewer</h1>
    <div id="stats"></div>

    <h2>Search (recommended start)</h2>
    <input type="search" id="search" placeholder="e.g., Olmo-3-Think, MMLU, Tulu-3..." />
    <div class="nav-results" id="search-results"></div>

    <h2>Filter</h2>
    <div class="row slider-row">
      <span>Min degree:</span>
      <input type="range" id="min-degree" min="0" max="50" value="10" />
      <span class="num" id="min-degree-val">10</span>
    </div>
    <div class="row slider-row">
      <span>Max nodes:</span>
      <input type="number" id="max-nodes" value="200" min="20" max="2000" step="10" style="width: 80px" />
    </div>
    <div class="row">
      <strong style="font-size:11px;">Kind</strong><br/>
      <span class="filter-chip kind-model" data-filter="kind:model">model</span>
      <span class="filter-chip kind-dataset" data-filter="kind:dataset">dataset</span>
      <span class="filter-chip kind-off_lattice" data-filter="kind:off_lattice">off-lattice</span>
    </div>
    <div class="row">
      <strong style="font-size:11px;">Relation</strong>
      <div id="rel-chips" style="margin-top:4px;"></div>
    </div>

    <h2>Selected</h2>
    <div id="detail">Click a node to see edges, anchors, identity. Or use search above to focus on one model.</div>
  </aside>

  <main id="graph">
    <div class="toolbar">
      <button id="layout-force">Force layout</button>
      <button id="layout-hier">Hierarchical</button>
      <button id="ego-mode">Ego (1-hop)</button>
      <button id="ego-2hop">Ego (2-hop)</button>
      <button id="reset-view">Reset</button>
      <button id="fit-view">Fit</button>
    </div>
    <div id="status-bar"></div>
    <div id="graph-canvas"></div>
  </main>
</div>

<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script>
const KIND_COLORS = { model: "#1f6feb", dataset: "#1a7f37", off_lattice: "#b48a4a" };
const DIR_COLORS = { direct: "#b14a00", indirect: "#6e3aac" };

const state = {
  raw: null,
  filters: { kinds: new Set(), rels: new Set(), minDegree: 10, maxNodes: 200 },
  network: null, nodesDS: null, edgesDS: null,
  selectedNodeId: null, egoMode: 0,  // 0=off, 1=1-hop, 2=2-hop
  edgeIndex: [],  // visible edge index → raw edge
};

function escapeHTML(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }

async function fetchData() {
  document.getElementById("status-bar").textContent = "loading…";
  const r = await fetch("/api/graph");
  state.raw = await r.json();
  populateChips();
  populateStats();
  rebuild();
}

function populateStats() {
  const s = state.raw.stats;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><span class="label">Total nodes</span><span><strong>${s.node_count.toLocaleString()}</strong></span></div>
    <div class="stat"><span class="label">Lattice nodes</span><span>${s.lattice_node_count.toLocaleString()}</span></div>
    <div class="stat"><span class="label">Off-lattice</span><span>${s.off_lattice_node_count.toLocaleString()}</span></div>
    <div class="stat"><span class="label">Edges</span><span><strong>${s.edge_count.toLocaleString()}</strong></span></div>
  `;
}

function populateChips() {
  const cont = document.getElementById("rel-chips");
  for (const [rel, n] of state.raw.stats.relations) {
    const span = document.createElement("span");
    span.className = "filter-chip";
    span.dataset.filter = `rel:${rel}`;
    span.textContent = `${rel} (${n})`;
    cont.appendChild(span);
  }
}

function getVisible() {
  const f = state.filters;
  // Compute degree map
  let candidateNodes = state.raw.nodes.filter(n => {
    if (f.kinds.size && !f.kinds.has(n.kind)) return false;
    return (n.in_degree + n.out_degree) >= f.minDegree;
  });

  // If ego mode, override candidate set
  if (state.egoMode > 0 && state.selectedNodeId) {
    const ego = state.selectedNodeId;
    const tier1 = new Set([ego]);
    for (const e of state.raw.edges) {
      if (e.subject === ego) tier1.add(e.object_id);
      if (e.object_id === ego) tier1.add(e.subject);
    }
    let tier = tier1;
    if (state.egoMode >= 2) {
      const tier2 = new Set(tier1);
      for (const e of state.raw.edges) {
        if (tier1.has(e.subject)) tier2.add(e.object_id);
        if (tier1.has(e.object_id)) tier2.add(e.subject);
      }
      tier = tier2;
    }
    candidateNodes = state.raw.nodes.filter(n => tier.has(n.id));
  } else {
    // Cap to max-nodes by degree (keep highest-degree first)
    candidateNodes.sort((a, b) => (b.in_degree + b.out_degree) - (a.in_degree + a.out_degree));
    if (candidateNodes.length > f.maxNodes) candidateNodes = candidateNodes.slice(0, f.maxNodes);
  }

  const visibleIds = new Set(candidateNodes.map(n => n.id));
  const visibleEdges = state.raw.edges.filter(e => {
    if (!visibleIds.has(e.subject) || !visibleIds.has(e.object_id)) return false;
    if (f.rels.size && !f.rels.has(e.relation)) return false;
    return true;
  });
  return { nodes: candidateNodes, edges: visibleEdges };
}

function rebuild() {
  const { nodes, edges } = getVisible();
  state.edgeIndex = edges;

  const visNodes = nodes.map(n => ({
    id: n.id,
    label: n.id.length > 38 ? n.id.slice(0, 35) + "…" : n.id,
    title: n.id,
    shape: n.kind === "dataset" ? "box" : (n.kind === "off_lattice" ? "diamond" : "dot"),
    color: { background: KIND_COLORS[n.kind] || "#888", border: "#333" },
    font: { size: 11, color: "#222" },
    size: Math.min(28, 6 + Math.sqrt((n.in_degree + n.out_degree) || 1) * 2.5),
  }));

  const visEdges = edges.map((e, i) => ({
    id: i,
    from: e.subject, to: e.object_id,
    arrows: "to",
    color: { color: DIR_COLORS[e.dependency_kind] || "#888", opacity: 0.55 },
    dashes: e.dependency_kind === "indirect" ? [4, 4] : false,
    font: { size: 9, color: "#666", strokeWidth: 2, strokeColor: "#fff" },
    label: edges.length < 300 ? e.relation : undefined,
    width: 1, smooth: false,
    title: `${e.subject} —[${e.relation}]→ ${e.object || ""}`,
  }));

  const sb = document.getElementById("status-bar");
  sb.textContent = `${visNodes.length} nodes · ${visEdges.length} edges`;

  if (visNodes.length === 0) {
    if (state.network) { state.nodesDS.clear(); state.edgesDS.clear(); }
    return;
  }

  if (state.network) {
    state.nodesDS.clear(); state.edgesDS.clear();
    state.nodesDS.add(visNodes); state.edgesDS.add(visEdges);
    state.network.fit();
    return;
  }
  state.nodesDS = new vis.DataSet(visNodes);
  state.edgesDS = new vis.DataSet(visEdges);
  state.network = new vis.Network(
    document.getElementById("graph-canvas"),
    { nodes: state.nodesDS, edges: state.edgesDS },
    {
      layout: { improvedLayout: false },
      physics: { enabled: false },  // OFF by default — toggle via toolbar
      interaction: { hover: true, multiselect: true, navigationButtons: true, keyboard: true },
      edges: { smooth: false },
      nodes: { borderWidth: 1.5 },
    }
  );
  state.network.on("selectNode", e => {
    state.selectedNodeId = e.nodes[0];
    showNodeDetail(state.selectedNodeId);
    if (state.egoMode > 0) rebuild();
  });
  state.network.on("selectEdge", e => {
    if (e.nodes.length > 0) return;
    const idx = e.edges[0];
    const visEdge = state.edgesDS.get(idx);
    if (visEdge != null) showEdgeDetail(state.edgeIndex[idx]);
  });
}

function showNodeDetail(id) {
  const n = state.raw.nodes.find(x => x.id === id);
  if (!n) return;
  const out = state.raw.edges.filter(e => e.subject === id);
  const incoming = state.raw.edges.filter(e => e.object_id === id);
  const linksHTML = (n.links || []).filter(l => l && l.url).map(l => `<a href="${escapeHTML(l.url)}" target="_blank" style="display:block;font-size:11px;">[${escapeHTML(l.kind||'')}] ${escapeHTML(l.url)}</a>`).join("");
  const aliasesHTML = (n.aliases || []).slice(0, 30).map(a => `<span class="alias">${escapeHTML(a)}</span>`).join("");
  const rowOf = (e, dir) => {
    const other = dir === "out" ? (e.object || "") : e.subject;
    return `<div style="padding:2px 0;border-bottom:1px dotted #eee;font-size:11.5px;">
      <span class="pill" style="background:${e.dependency_kind==='direct'?'#ffe1d6':'#f0e1ff'};color:${e.dependency_kind==='direct'?'#b14a00':'#6e3aac'}">${escapeHTML(e.dependency_kind || "")}</span>
      <span style="font-family:ui-monospace,monospace;">${escapeHTML(e.relation)}</span>
      <span style="color:#888;">→</span> ${escapeHTML(other)}
    </div>`;
  };
  document.getElementById("detail").innerHTML = `
    <div class="name">${escapeHTML(n.id)}</div>
    <div style="margin-top:4px;">
      <span class="pill kind-${n.kind}">${escapeHTML(n.kind)}</span>
      <span style="font-size:11px;color:#777;">${n.in_degree} in / ${n.out_degree} out</span>
    </div>
    ${n.description ? `<div class="desc">${escapeHTML(n.description.slice(0, 400))}</div>` : ""}
    ${aliasesHTML ? `<details><summary>aliases (${n.aliases.length})</summary>${aliasesHTML}</details>` : ""}
    ${linksHTML ? `<details open><summary>links (${n.links.length})</summary>${linksHTML}</details>` : ""}
    ${out.length ? `<details open><summary>outgoing (${out.length})</summary>${out.slice(0, 30).map(e => rowOf(e, "out")).join("")}${out.length > 30 ? `<div style='font-size:11px;color:#888'>… ${out.length-30} more</div>` : ""}</details>` : ""}
    ${incoming.length ? `<details open><summary>incoming (${incoming.length})</summary>${incoming.slice(0, 30).map(e => rowOf(e, "in")).join("")}${incoming.length > 30 ? `<div style='font-size:11px;color:#888'>… ${incoming.length-30} more</div>` : ""}</details>` : ""}
  `;
}

function showEdgeDetail(e) {
  if (!e) return;
  const anchorsHTML = (e.anchor_list || []).slice(0, 8).map(a => {
    const src = a.source || a.path || a.url || "";
    return `<div style="margin:4px 0;padding:4px;background:#fafafa;border-left:2px solid #ccc;font-size:11px;">
      ${src ? `<div><a href="${escapeHTML(src)}" target="_blank">${escapeHTML(src.slice(0,140))}</a></div>` : ""}
      ${a.position ? `<div><strong>pos:</strong> ${escapeHTML(a.position)}</div>` : ""}
      <div>${escapeHTML((a.explanation||"").slice(0, 300))}</div>
    </div>`;
  }).join("");
  document.getElementById("detail").innerHTML = `
    <div class="name">${escapeHTML(e.subject)}<br/>—[<span style="font-family:ui-monospace,monospace">${escapeHTML(e.relation)}</span>]→<br/>${escapeHTML(e.object || "")}</div>
    <div style="margin-top:4px;">
      <span class="pill" style="background:${e.dependency_kind==='direct'?'#ffe1d6':'#f0e1ff'};color:${e.dependency_kind==='direct'?'#b14a00':'#6e3aac'}">${escapeHTML(e.dependency_kind || "")}</span>
      <span style="font-size:11px;color:#888;">${(e.anchor_list||[]).length} anchor${(e.anchor_list||[]).length===1?"":"s"}</span>
    </div>
    ${e.description ? `<div class="desc">${escapeHTML(e.description)}</div>` : ""}
    ${anchorsHTML ? `<details open><summary>anchors</summary>${anchorsHTML}</details>` : ""}
  `;
}

document.getElementById("search").addEventListener("input", (ev) => {
  const q = ev.target.value.trim().toLowerCase();
  const out = document.getElementById("search-results");
  if (!q) { out.innerHTML = ""; return; }
  const matches = state.raw.nodes.filter(n => {
    if (n.id.toLowerCase().includes(q)) return true;
    return (n.aliases || []).some(a => (a || "").toLowerCase().includes(q));
  }).sort((a,b) => (b.in_degree+b.out_degree) - (a.in_degree+a.out_degree)).slice(0, 50);
  out.innerHTML = matches.map(n => `
    <div class="nav-result" data-node-id="${escapeHTML(n.id)}">
      <div class="formal">${escapeHTML(n.id)}</div>
      <div class="meta"><span class="pill kind-${n.kind}">${escapeHTML(n.kind)}</span> ${n.in_degree}↘ ${n.out_degree}↗</div>
    </div>`).join("");
});

document.getElementById("search-results").addEventListener("click", (ev) => {
  const r = ev.target.closest(".nav-result"); if (!r) return;
  const id = r.dataset.nodeId;
  state.selectedNodeId = id;
  state.egoMode = state.egoMode || 1;
  document.getElementById("ego-mode").classList.toggle("active", state.egoMode === 1);
  document.getElementById("ego-2hop").classList.toggle("active", state.egoMode === 2);
  rebuild();
  setTimeout(() => {
    if (state.nodesDS && state.nodesDS.get(id)) {
      state.network.selectNodes([id]);
      state.network.focus(id, { scale: 1.0, animation: false });
    }
    showNodeDetail(id);
  }, 100);
});

document.querySelectorAll(".filter-chip").forEach(el => {
  el.addEventListener("click", () => {
    el.classList.toggle("active");
    const [k, v] = el.dataset.filter.split(":");
    const target = ({ kind: state.filters.kinds, rel: state.filters.rels })[k];
    if (target.has(v)) target.delete(v); else target.add(v);
    rebuild();
  });
});

document.getElementById("min-degree").addEventListener("input", (ev) => {
  state.filters.minDegree = parseInt(ev.target.value, 10);
  document.getElementById("min-degree-val").textContent = ev.target.value;
});
document.getElementById("min-degree").addEventListener("change", () => rebuild());
document.getElementById("max-nodes").addEventListener("change", (ev) => {
  state.filters.maxNodes = parseInt(ev.target.value, 10);
  rebuild();
});

document.getElementById("layout-force").addEventListener("click", (ev) => {
  if (!state.network) return;
  const isOn = ev.target.classList.toggle("active");
  state.network.setOptions({
    layout: { hierarchical: false },
    physics: {
      enabled: isOn,
      stabilization: { iterations: 100, fit: true },
      barnesHut: { gravitationalConstant: -8000, springLength: 100, avoidOverlap: 0.3 },
    },
  });
  document.getElementById("layout-hier").classList.remove("active");
});
document.getElementById("layout-hier").addEventListener("click", (ev) => {
  if (!state.network) return;
  ev.target.classList.add("active");
  document.getElementById("layout-force").classList.remove("active");
  state.network.setOptions({
    physics: { enabled: false },
    layout: { hierarchical: { direction: "UD", sortMethod: "directed", levelSeparation: 100, nodeSpacing: 120 } },
  });
});

document.getElementById("ego-mode").addEventListener("click", (ev) => {
  state.egoMode = state.egoMode === 1 ? 0 : 1;
  ev.target.classList.toggle("active", state.egoMode === 1);
  document.getElementById("ego-2hop").classList.remove("active");
  rebuild();
});
document.getElementById("ego-2hop").addEventListener("click", (ev) => {
  state.egoMode = state.egoMode === 2 ? 0 : 2;
  ev.target.classList.toggle("active", state.egoMode === 2);
  document.getElementById("ego-mode").classList.remove("active");
  rebuild();
});
document.getElementById("reset-view").addEventListener("click", () => {
  state.egoMode = 0;
  state.selectedNodeId = null;
  document.getElementById("ego-mode").classList.remove("active");
  document.getElementById("ego-2hop").classList.remove("active");
  rebuild();
});
document.getElementById("fit-view").addEventListener("click", () => {
  state.network && state.network.fit({ animation: false });
});

fetchData();
</script>
</body>
</html>
"""


def make_handler(graph_payload: dict):
    body_bytes = PAGE_HTML.encode("utf-8")
    graph_json = json.dumps(graph_payload, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(body)
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", body_bytes)
            elif path == "/api/graph":
                self._send(200, "application/json; charset=utf-8", graph_json)
            elif path == "/healthz":
                self._send(200, "text/plain", b"ok")
            else:
                self._send(404, "text/plain", b"not found")
        def log_message(self, fmt: str, *args: Any) -> None:
            return
    return Handler


def serve(source: Path, host: str = "127.0.0.1", port: int = 8102,
          seed: str | None = None, depth: int = 2,
          target_size: int = 80) -> None:
    """Serve the interactive viewer for the merged graph at *source*.

    When *seed* is given, the payload is pre-pruned to a focused subgraph
    centered on the matching node (BFS up to *depth* hops, capped at
    *target_size* nodes, neighbors ranked by lineage-bearing edge score).
    Otherwise the full graph is served and the client's degree-slider
    handles size."""
    src_path = Path(source)
    if not src_path.exists():
        print(f"ERROR: source not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {src_path}")
    payload = load_data(src_path)
    s = payload["stats"]
    print(f"  → {s['node_count']:,} nodes  ({s['lattice_node_count']:,} lattice + {s['off_lattice_node_count']:,} off-lattice)")
    print(f"  → {s['edge_count']:,} edges, {len(s['relations'])} distinct relations")

    if seed:
        seed_id = _resolve_seed(payload, seed)
        if not seed_id:
            print(f"ERROR: seed pattern {seed!r} matched no node.", file=sys.stderr)
            sys.exit(1)
        before = (s["node_count"], s["edge_count"])
        payload = _seeded_expand(payload, seed_id,
                                  depth=depth, target_size=target_size)
        s = payload["stats"]
        print(f"  → seeded on {seed_id!r}: {before[0]:,}→{s['node_count']:,} nodes, "
              f"{before[1]:,}→{s['edge_count']:,} edges "
              f"(depth={depth}, target_size={target_size})")

    print()
    if seed:
        print(f"Centered on {payload['stats'].get('seed')!r}. "
              f"Use the search box or ego mode to refocus.")
    else:
        print("Default initial view: top 200 nodes by degree (≥10), physics OFF.")
        print("Use the search box to focus on a specific model — it'll pivot to ego mode automatically.")
    print()

    handler = make_handler(payload)
    server = HTTPServer((host, port), handler)
    print(f"Serving on http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a merged dependency-graph JSON.")
    p.add_argument("--source", required=True, type=Path,
                   help="Path to the merged graph JSON (e.g. merge_artifact.json or a deduped output).")
    p.add_argument("--seed", default=None,
                   help="Pattern to match (case-insensitive substring on formal_name "
                        "and aliases) for a seeded ego-expansion. Highest-degree match wins.")
    p.add_argument("--depth", type=int, default=2,
                   help="Hops to expand from --seed (default: 2).")
    p.add_argument("--target-size", type=int, default=80,
                   help="Approximate target node count for --seed expansion (default: 80).")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    serve(source=args.source, host=args.host, port=args.port,
          seed=args.seed, depth=args.depth, target_size=args.target_size)


if __name__ == "__main__":
    main()
