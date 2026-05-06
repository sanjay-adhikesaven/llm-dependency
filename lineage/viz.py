"""Interactive lattice + relation graph viewer.

Reads the latest lattice artifact (audit / organize) and every
completed relate artifact, materializes a graph (nodes = lattice items,
edges = relations grouped by operation), and serves it on localhost.

Usage: `python -m lineage.cli viz --port 8102`
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .pipeline import _latest_lattice_artifact_path
from .store import all_rows, read_json


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_graph_data() -> dict:
    lattice_path = _latest_lattice_artifact_path()
    lattice = read_json(str(lattice_path))

    # Build nodes by formal_name
    nodes: dict[str, dict] = {}
    family_counts: dict[str, int] = {}
    for grp in lattice.get("groups") or []:
        family = grp.get("family") or ""
        family_counts[family] = family_counts.get(family, 0) + len(grp.get("items") or [])
        for item in grp.get("items") or []:
            formal = item.get("formal_name")
            if not formal:
                continue
            links = item.get("links") or []
            primary_url = ""
            primary_kind = ""
            for ln in links:
                if isinstance(ln, dict) and ln.get("url"):
                    primary_url = ln["url"]
                    primary_kind = ln.get("kind") or ""
                    break
            nodes[formal] = {
                "id": formal,
                "kind": item.get("kind"),
                "family": family,
                "identity": item.get("identity") or {},
                "aliases": item.get("aliases") or [],
                "links": links,
                "n_links": len(links),
                "primary_url": primary_url,
                "primary_link_kind": primary_kind,
                "description": item.get("description") or "",
                "in_degree": 0,
                "out_degree": 0,
            }

    # Load every completed relate artifact and merge events + edges
    artifact_rows = all_rows(
        "SELECT batch_id, artifact_path, attrs FROM batch_artifacts "
        "WHERE stage='relate' AND status='complete'"
    )
    operations: list[dict] = []  # event records (one per JSONL line)
    edges: list[dict] = []
    batch_label_for_id: dict[str, str] = {}
    for row in all_rows("SELECT id, label FROM batches"):
        batch_label_for_id[row["id"]] = row["label"] or row["id"][:8]

    for r in artifact_rows:
        path = Path(r["artifact_path"])
        if not path.exists():
            continue
        artifact = read_json(str(path))
        bid = r["batch_id"]
        b_label = batch_label_for_id.get(bid, bid[:8])
        for event_idx, op in enumerate(artifact.get("operations") or []):
            event_id = f"{b_label}/event-{event_idx}"
            operations.append({
                "id": event_id,
                "batch": b_label,
                "description": op.get("description") or "",
                "anchor_list": op.get("anchor_list") or [],
                "edge_count": len(op.get("edges") or []),
            })
            for edge in op.get("edges") or []:
                obj = edge.get("object") or ""
                # Object resolves to a lattice node if its string is a
                # known formal_name; otherwise it's free-text.
                in_lattice = obj in nodes
                object_id = obj if in_lattice else f"text::{obj}"
                edges.append({
                    "subject": edge.get("subject"),
                    "object": obj,
                    "object_id": object_id,
                    "object_in_lattice": in_lattice,
                    "relation": edge.get("relation"),
                    "dependency_kind": edge.get("dependency_kind"),
                    "description": edge.get("description") or "",
                    "anchor_list": edge.get("anchor_list") or [],
                    "operation_id": event_id,
                    "batch": b_label,
                })

    # Off-lattice synthetic nodes for free-text object endpoints
    off_lattice_nodes: dict[str, dict] = {}
    for e in edges:
        if e["object_in_lattice"]:
            continue
        if e["object_id"] in off_lattice_nodes:
            continue
        if e["object"]:
            off_lattice_nodes[e["object_id"]] = {
                "id": e["object_id"],
                "kind": "off_lattice",
                "family": "(off-lattice)",
                "identity": {},
                "aliases": [],
                "links": [],
                "n_links": 0,
                "primary_url": "",
                "primary_link_kind": "",
                "description": e["object"],
                "in_degree": 0,
                "out_degree": 0,
            }
    nodes.update(off_lattice_nodes)

    # Compute degrees
    for e in edges:
        s = e["subject"]
        o = e["object_id"]
        if s in nodes:
            nodes[s]["out_degree"] += 1
        if o and o in nodes:
            nodes[o]["in_degree"] += 1

    # Relation label histogram + dependency-kind breakdown
    rel_counts: dict[str, int] = {}
    dep_kind_counts: dict[str, int] = {}
    for e in edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
        dk = e.get("dependency_kind") or "unknown"
        dep_kind_counts[dk] = dep_kind_counts.get(dk, 0) + 1

    return {
        "lattice_path": str(lattice_path),
        "nodes": list(nodes.values()),
        "edges": edges,
        "operations": operations,
        "stats": {
            "node_count": len(nodes),
            "lattice_node_count": sum(1 for n in nodes.values() if n.get("kind") in ("model", "dataset")),
            "off_lattice_node_count": sum(1 for n in nodes.values() if n.get("kind") == "off_lattice"),
            "edge_count": len(edges),
            "operation_count": len(operations),
            "family_count": len(family_counts),
            "families": sorted(family_counts.items(), key=lambda kv: -kv[1]),
            "relations": sorted(rel_counts.items(), key=lambda kv: -kv[1]),
            "dependency_kinds": sorted(dep_kind_counts.items(), key=lambda kv: -kv[1]),
        },
    }


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>graph/ lattice viewer</title>
<style>
  :root { color-scheme: light; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fafafa; color: #222; }
  #app { display: grid; grid-template-columns: 360px 1fr; height: 100vh; }
  #side { padding: 12px; overflow-y: auto; border-right: 1px solid #ddd; background: #fff; box-shadow: 1px 0 0 #eee; }
  #graph { position: relative; height: 100vh; }
  #graph-canvas { width: 100%; height: 100%; }
  h1 { font-size: 14px; margin: 0 0 8px; letter-spacing: 0.02em; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #777; margin: 16px 0 6px; }
  input[type=search], input[type=text], select { width: 100%; box-sizing: border-box; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
  .row { margin: 6px 0; font-size: 13px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; background: #eef; color: #336; margin-right: 4px; }
  .kind-model { background: #d8eafd; color: #0a66c2; }
  .kind-dataset { background: #d3f0d3; color: #1a7f37; }
  .kind-off_lattice { background: #f3e3c0; color: #7a4f00; }
  .dir-direct { background: #ffe1d6; color: #b14a00; }
  .dir-indirect { background: #f0e1ff; color: #6e3aac; }
  .stat { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px; }
  .stat .label { color: #777; }
  .badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; background: #eee; color: #555; margin-left: 4px; }
  .filter-chip { display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0; border: 1px solid #ccc; border-radius: 12px; font-size: 11px; cursor: pointer; user-select: none; background: #fff; }
  .filter-chip.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  .filter-chip.dir-direct.active { background: #b14a00; border-color: #b14a00; }
  .filter-chip.dir-indirect.active { background: #6e3aac; border-color: #6e3aac; }
  #detail { background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 12.5px; }
  #detail .name { font-weight: 600; font-size: 13.5px; word-break: break-all; }
  #detail .desc { color: #444; margin-top: 4px; line-height: 1.4; }
  #detail .links a { display: block; word-break: break-all; color: #0969da; text-decoration: none; font-size: 11.5px; margin: 2px 0; }
  #detail .links a:hover { text-decoration: underline; }
  #detail .alias { display: inline-block; background: #eef; padding: 1px 6px; border-radius: 3px; margin: 1px; font-size: 11px; }
  #detail .evidence { background: #fff; border-left: 3px solid #1f6feb; padding: 6px 10px; margin: 6px 0; font-size: 11.5px; font-family: ui-monospace, monospace; white-space: pre-wrap; }
  .nav-results { margin-top: 6px; max-height: 240px; overflow-y: auto; border: 1px solid #eee; border-radius: 4px; }
  .nav-result { padding: 4px 8px; cursor: pointer; font-size: 12px; border-bottom: 1px solid #f4f4f4; }
  .nav-result:hover { background: #eaf3ff; }
  .nav-result .formal { font-weight: 500; }
  .nav-result .meta { color: #888; font-size: 11px; }
  details { margin: 6px 0; }
  details summary { cursor: pointer; font-size: 12px; padding: 3px 0; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-swatch { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #999; }
  #status-bar { position: absolute; top: 8px; right: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 4px 10px; font-size: 11px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; }
  .toolbar { position: absolute; top: 8px; left: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 10; display: flex; gap: 6px; }
  .toolbar button { font-size: 11px; padding: 4px 8px; border: 1px solid #ccc; background: #fafafa; border-radius: 3px; cursor: pointer; }
  .toolbar button:hover { background: #fff; }
  .toolbar button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>graph/ lattice viewer</h1>
    <div id="stats"></div>

    <h2>Search</h2>
    <input type="search" id="search" placeholder="Substring of formal_name or alias…" />
    <div class="nav-results" id="search-results"></div>

    <h2>Filter</h2>
    <div class="row">
      <strong style="font-size:11px;">Kind</strong><br/>
      <span class="filter-chip kind-model" data-filter="kind:model">model</span>
      <span class="filter-chip kind-dataset" data-filter="kind:dataset">dataset</span>
      <span class="filter-chip kind-off_lattice" data-filter="kind:off_lattice">off-lattice</span>
    </div>
    <div class="row">
      <strong style="font-size:11px;">Direction</strong><br/>
      <span class="filter-chip dir-direct" data-filter="dir:direct">direct</span>
      <span class="filter-chip dir-indirect" data-filter="dir:indirect">indirect</span>
    </div>
    <div class="row">
      <strong style="font-size:11px;">Relation type</strong>
      <div id="rel-chips" style="margin-top:4px;"></div>
    </div>
    <div class="row">
      <strong style="font-size:11px;">Family</strong><br/>
      <select id="family-select"><option value="">— any family —</option></select>
    </div>
    <div class="row">
      <label style="font-size:12px;"><input type="checkbox" id="hide-unlinked" /> Hide unlinked items</label>
    </div>
    <div class="row">
      <label style="font-size:12px;"><input type="checkbox" id="hide-isolated" checked /> Hide isolated nodes (degree 0)</label>
    </div>

    <h2>Selected</h2>
    <div id="detail">Click a node or edge to see details.</div>
  </aside>

  <main id="graph">
    <div class="toolbar">
      <button data-layout="hierarchical">Hierarchical</button>
      <button data-layout="force" class="active">Force</button>
      <button id="reset-view">Reset</button>
      <button id="ego-mode">Ego</button>
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
  filters: { kinds: new Set(), dirs: new Set(), rels: new Set(), family: "", hideUnlinked: false, hideIsolated: true, query: "" },
  network: null,
  nodesDS: null,
  edgesDS: null,
  selectedNodeId: null,
  egoMode: false,
};

function escapeHTML(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c])); }

async function fetchData() {
  const r = await fetch("/api/graph");
  state.raw = await r.json();
  populateChips();
  populateStats();
  rebuild();
}

function populateStats() {
  const s = state.raw.stats;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><span class="label">Lattice items</span><span><strong>${s.lattice_node_count}</strong></span></div>
    <div class="stat"><span class="label">Off-lattice nodes</span><span>${s.off_lattice_node_count}</span></div>
    <div class="stat"><span class="label">Edges</span><span><strong>${s.edge_count}</strong></span></div>
    <div class="stat"><span class="label">Operations</span><span>${s.operation_count}</span></div>
    <div class="stat"><span class="label">Families</span><span>${s.family_count}</span></div>
  `;
  const sel = document.getElementById("family-select");
  for (const [fam, n] of s.families) {
    const opt = document.createElement("option");
    opt.value = fam;
    opt.textContent = `${fam} (${n})`;
    sel.appendChild(opt);
  }
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

function applyFilters() {
  const f = state.filters;
  const visibleNodeIds = new Set();
  const allowed = (n) => {
    if (f.kinds.size && !f.kinds.has(n.kind)) return false;
    if (f.family && n.family !== f.family) return false;
    if (f.hideUnlinked && n.kind !== "off_lattice" && n.n_links === 0) return false;
    if (f.hideIsolated && (n.in_degree + n.out_degree) === 0) return false;
    return true;
  };
  for (const n of state.raw.nodes) if (allowed(n)) visibleNodeIds.add(n.id);

  const visibleEdges = state.raw.edges.filter(e => {
    if (!visibleNodeIds.has(e.subject)) return false;
    if (e.object_id && !visibleNodeIds.has(e.object_id)) return false;
    if (f.dirs.size && !f.dirs.has(e.dependency_kind)) return false;
    if (f.rels.size && !f.rels.has(e.relation)) return false;
    return true;
  });

  // Egomode: limit to selected node + 1-hop neighbors
  let nodeIdSet = visibleNodeIds;
  let edgeSet = visibleEdges;
  if (state.egoMode && state.selectedNodeId) {
    const ego = state.selectedNodeId;
    const neighbors = new Set([ego]);
    edgeSet = visibleEdges.filter(e => e.subject === ego || e.object_id === ego);
    for (const e of edgeSet) {
      neighbors.add(e.subject);
      if (e.object_id) neighbors.add(e.object_id);
    }
    nodeIdSet = neighbors;
  }
  return {
    nodes: state.raw.nodes.filter(n => nodeIdSet.has(n.id)),
    edges: edgeSet,
  };
}

function rebuild() {
  const { nodes, edges } = applyFilters();

  const visNodes = nodes.map(n => {
    const fadeUnlinked = n.kind !== "off_lattice" && n.n_links === 0;
    return {
      id: n.id,
      label: n.id.length > 40 ? n.id.slice(0, 37) + "…" : n.id,
      title: n.id,
      shape: n.kind === "dataset" ? "box" : (n.kind === "off_lattice" ? "diamond" : "dot"),
      color: { background: KIND_COLORS[n.kind] || "#888", border: fadeUnlinked ? "#bbb" : "#333" },
      opacity: fadeUnlinked ? 0.45 : 1.0,
      font: { size: 11, color: "#222" },
      size: Math.min(20, 8 + Math.sqrt((n.in_degree + n.out_degree) || 0) * 2),
    };
  });

  // Edge id is index — we let vis collapse multi-edges visually.
  const visEdges = edges.map((e, i) => {
    const isLiteral = !e.object_id;
    if (isLiteral) return null;
    return {
      id: i,
      from: e.subject,
      to: e.object_id,
      arrows: "to",
      color: { color: DIR_COLORS[e.dependency_kind] || "#888", opacity: 0.65 },
      dashes: e.dependency_kind === "indirect" ? [4, 4] : false,
      font: { size: 9, color: "#666", strokeWidth: 0 },
      label: e.relation,
      width: 1,
      title: `${e.subject} -[${e.relation}]-> ${e.object || "(literal)"}`,
    };
  }).filter(Boolean);

  document.getElementById("status-bar").textContent = `${visNodes.length} nodes · ${visEdges.length} edges shown`;

  if (state.network) {
    state.nodesDS.clear();
    state.edgesDS.clear();
    state.nodesDS.add(visNodes);
    state.edgesDS.add(visEdges);
    return;
  }
  state.nodesDS = new vis.DataSet(visNodes);
  state.edgesDS = new vis.DataSet(visEdges);
  const opts = {
    layout: { improvedLayout: false },
    physics: { stabilization: { iterations: 200 }, barnesHut: { gravitationalConstant: -8000, springLength: 110, avoidOverlap: 0.2 } },
    interaction: { hover: true, multiselect: true, navigationButtons: true, keyboard: true },
    edges: { smooth: { enabled: true, type: "dynamic" } },
    nodes: { borderWidth: 1.5, shadow: false },
  };
  state.network = new vis.Network(document.getElementById("graph-canvas"), { nodes: state.nodesDS, edges: state.edgesDS }, opts);
  state.network.on("selectNode", e => {
    state.selectedNodeId = e.nodes[0];
    showNodeDetail(state.selectedNodeId);
    if (state.egoMode) rebuild();
  });
  state.network.on("selectEdge", e => {
    if (e.nodes.length > 0) return;
    const edgeIdx = e.edges[0];
    const visEdge = state.edgesDS.get(edgeIdx);
    if (visEdge) showEdgeDetail(edges[edgeIdx]);
  });
  state.network.on("deselectNode", () => { /* keep last */ });
}

function showNodeDetail(id) {
  const n = state.raw.nodes.find(x => x.id === id);
  if (!n) return;
  const myEdges = state.raw.edges.filter(e => e.subject === id || e.object_id === id);
  const linksHTML = (n.links || []).map(l => `<a href="${escapeHTML(l.url)}" target="_blank">[${escapeHTML(l.kind)}] ${escapeHTML(l.url)}</a>`).join("");
  const aliasesHTML = (n.aliases || []).map(a => `<span class="alias">${escapeHTML(a)}</span>`).join("");
  const identityHTML = Object.entries(n.identity || {}).map(([k,v]) => `<span class="alias">${escapeHTML(k)}=${escapeHTML(v)}</span>`).join("");
  const out = myEdges.filter(e => e.subject === id);
  const incoming = myEdges.filter(e => e.object_id === id);
  const renderEdgeRow = (e, dir) => {
    const other = dir === "out" ? (e.object || "") : e.subject;
    return `<div class="row" data-edge-row="1" style="cursor:pointer; padding:3px 0; border-bottom:1px dotted #eee;">
      <span class="pill dir-${e.dependency_kind}">${escapeHTML(e.dependency_kind || "")}</span>
      <span style="font-family:ui-monospace,monospace;font-size:11px;">${escapeHTML(e.relation)}</span>
      <span style="font-size:11px;color:#888;">→</span>
      <span style="font-size:11px;">${escapeHTML(other)}</span>
    </div>`;
  };
  const desc = n.description ? `<div class="desc">${escapeHTML(n.description)}</div>` : "";
  document.getElementById("detail").innerHTML = `
    <div class="name">${escapeHTML(n.id)}</div>
    <div style="margin-top:4px;">
      <span class="pill kind-${n.kind}">${escapeHTML(n.kind)}</span>
      <span class="pill">${escapeHTML(n.family)}</span>
      <span class="badge">${n.in_degree} in / ${n.out_degree} out</span>
    </div>
    ${desc}
    ${identityHTML ? `<details open><summary>identity</summary>${identityHTML}</details>` : ""}
    ${aliasesHTML ? `<details><summary>aliases (${n.aliases.length})</summary>${aliasesHTML}</details>` : ""}
    ${linksHTML ? `<details open><summary>links (${n.links.length})</summary><div class="links">${linksHTML}</div></details>` : ""}
    ${out.length ? `<details open><summary>outgoing (${out.length})</summary>${out.slice(0, 30).map(e => renderEdgeRow(e, "out")).join("")}${out.length > 30 ? `<div style='font-size:11px;color:#888'>… ${out.length - 30} more</div>` : ""}</details>` : ""}
    ${incoming.length ? `<details open><summary>incoming (${incoming.length})</summary>${incoming.slice(0, 30).map(e => renderEdgeRow(e, "in")).join("")}${incoming.length > 30 ? `<div style='font-size:11px;color:#888'>… ${incoming.length - 30} more</div>` : ""}</details>` : ""}
  `;
}

function showEdgeDetail(e) {
  if (!e) return;
  const op = e.operation_id ? state.raw.operations.find(o => o.id === e.operation_id) : null;
  const renderAnchor = (a) => `
    <div class="anchor" style="margin-top:4px;padding:4px;background:#fafafa;border-left:2px solid #ccc;font-size:11px;">
      <div><strong>source:</strong> <a href="${escapeHTML(a.source||"")}" target="_blank">${escapeHTML(a.source||"")}</a></div>
      ${a.position ? `<div><strong>position:</strong> ${escapeHTML(a.position)}</div>` : ""}
      <div>${escapeHTML(a.explanation||"")}</div>
    </div>`;
  const edgeAnchors = (e.anchor_list || []).map(renderAnchor).join("");
  const eventAnchors = op ? ((op.anchor_list || []).map(renderAnchor).join("")) : "";
  document.getElementById("detail").innerHTML = `
    <div class="name">edge: ${escapeHTML(e.subject)}<br/>—[${escapeHTML(e.relation)}]→<br/>${escapeHTML(e.object || "")}</div>
    <div style="margin-top:4px;">
      <span class="pill dir-${e.dependency_kind}">${escapeHTML(e.dependency_kind || "")}</span>
      ${e.object_in_lattice ? "" : `<span class="pill kind-off_lattice">off-lattice</span>`}
      <span class="pill">${escapeHTML(e.batch || "")}</span>
    </div>
    <div class="desc"><strong>edge desc:</strong> ${escapeHTML(e.description)}</div>
    ${edgeAnchors ? `<details open><summary>edge anchors (${(e.anchor_list||[]).length})</summary>${edgeAnchors}</details>` : ""}
    ${op ? `
      <h2>Event: ${escapeHTML(op.id)}</h2>
      <div class="desc">${escapeHTML(op.description)}</div>
      ${eventAnchors ? `<details><summary>event anchors (${(op.anchor_list||[]).length})</summary>${eventAnchors}</details>` : ""}
    ` : `<div style="font-size:11px;color:#888;margin-top:6px;">No event grouping</div>`}
  `;
}

document.getElementById("search").addEventListener("input", (ev) => {
  const q = ev.target.value.trim().toLowerCase();
  const out = document.getElementById("search-results");
  if (!q) { out.innerHTML = ""; return; }
  const matches = state.raw.nodes.filter(n => {
    if (n.id.toLowerCase().includes(q)) return true;
    return (n.aliases || []).some(a => (a || "").toLowerCase().includes(q));
  }).slice(0, 50);
  out.innerHTML = matches.map(n => `
    <div class="nav-result" data-node-id="${escapeHTML(n.id)}">
      <div class="formal">${escapeHTML(n.id)}</div>
      <div class="meta"><span class="pill kind-${n.kind}">${escapeHTML(n.kind)}</span> ${escapeHTML(n.family)} · ${n.n_links} link${n.n_links===1?"":"s"}</div>
    </div>`).join("");
});

document.getElementById("search-results").addEventListener("click", (ev) => {
  const r = ev.target.closest(".nav-result");
  if (!r) return;
  const id = r.dataset.nodeId;
  state.selectedNodeId = id;
  if (state.network && state.nodesDS.get(id)) {
    state.network.selectNodes([id]);
    state.network.focus(id, { scale: 1.2, animation: true });
  } else {
    state.egoMode = true;
    document.getElementById("ego-mode").classList.add("active");
    rebuild();
    setTimeout(() => {
      if (state.nodesDS.get(id)) {
        state.network.selectNodes([id]);
        state.network.focus(id, { scale: 1.2, animation: true });
      }
    }, 150);
  }
  showNodeDetail(id);
});

document.querySelectorAll(".filter-chip").forEach(el => {
  el.addEventListener("click", () => {
    el.classList.toggle("active");
    const [k, v] = el.dataset.filter.split(":");
    const target = ({ kind: state.filters.kinds, dir: state.filters.dirs, rel: state.filters.rels })[k];
    if (target.has(v)) target.delete(v); else target.add(v);
    rebuild();
  });
});
document.getElementById("family-select").addEventListener("change", (ev) => {
  state.filters.family = ev.target.value;
  rebuild();
});
document.getElementById("hide-unlinked").addEventListener("change", (ev) => {
  state.filters.hideUnlinked = ev.target.checked;
  rebuild();
});
document.getElementById("hide-isolated").addEventListener("change", (ev) => {
  state.filters.hideIsolated = ev.target.checked;
  rebuild();
});

document.getElementById("reset-view").addEventListener("click", () => {
  state.egoMode = false;
  document.getElementById("ego-mode").classList.remove("active");
  state.network && state.network.fit();
  rebuild();
});
document.getElementById("ego-mode").addEventListener("click", () => {
  state.egoMode = !state.egoMode;
  document.getElementById("ego-mode").classList.toggle("active", state.egoMode);
  rebuild();
});

document.querySelectorAll(".toolbar button[data-layout]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".toolbar button[data-layout]").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    const layout = btn.dataset.layout;
    if (!state.network) return;
    if (layout === "hierarchical") {
      state.network.setOptions({ layout: { hierarchical: { direction: "UD", sortMethod: "directed", levelSeparation: 110 } }, physics: { enabled: false } });
    } else {
      state.network.setOptions({ layout: { hierarchical: false }, physics: { enabled: true } });
    }
  });
});

fetchData();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def make_handler(graph_payload: dict):
    body_bytes = PAGE_HTML.encode("utf-8")
    graph_json = json.dumps(graph_payload, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", body_bytes)
            elif path == "/api/graph":
                self._send(200, "application/json; charset=utf-8", graph_json)
            elif path == "/healthz":
                self._send(200, "text/plain", b"ok")
            else:
                self._send(404, "text/plain", b"not found")

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # quieter log
            return

    return Handler


def serve(port: int = 8102, host: str = "127.0.0.1") -> None:
    payload = load_graph_data()
    print(f"loaded: {payload['stats']['node_count']} nodes, "
          f"{payload['stats']['edge_count']} edges, "
          f"{payload['stats']['operation_count']} operations from "
          f"{payload['lattice_path']}")
    handler = make_handler(payload)
    server = HTTPServer((host, port), handler)
    print(f"serving on http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()
