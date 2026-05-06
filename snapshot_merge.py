#!/usr/bin/env python3
"""Snapshot merge — runs `gdb run merge` over current artifacts, into an
ISOLATED GDB_STORAGE so it doesn't interfere with the in-progress
beam_v5/auto_merge_combined pipeline.

Outputs:
  /Users/sanjayadhikesaven/Downloads/graph/storage/snapshots/snapshot_<ts>/
    └── runs/<run_id>/merge_artifact.json   (the snapshot graph)
  run-logs/SNAPSHOT_MERGE.log
"""
from __future__ import annotations
import os, json, subprocess, time
from pathlib import Path

REPO = Path("/Users/sanjayadhikesaven/Downloads/graph")
LOG_DIR = REPO / "run-logs"
TS = time.strftime("%Y%m%d_%H%M")
SNAPSHOT_STORAGE = REPO / "storage" / "snapshots" / f"snapshot_{TS}"
SUMMARY_LOG = LOG_DIR / "SNAPSHOT_MERGE.log"
MERGE_LOG = LOG_DIR / f"snapshot_merge_{TS}.log"

SNAPSHOT_STORAGE.mkdir(parents=True, exist_ok=True)

def _ts(): return time.strftime("%Y-%m-%d %H:%M:%S")
def _log(msg):
    line = f"[{_ts()}] {msg}\n"
    with open(SUMMARY_LOG, "a") as f: f.write(line)
    print(line, end="")

_log(f"SNAPSHOT MERGE START — storage={SNAPSHOT_STORAGE}")

# Gather every artifact path on disk (same logic as auto_merge_combined)
audit_paths, organize_paths, relate_paths = [], [], []
for seed_dir in (REPO/"storage"/"seeds").glob("*"):
    if seed_dir.is_dir():
        audit_paths.extend(seed_dir.glob("runs/*/audit_artifact.json"))
        organize_paths.extend(seed_dir.glob("runs/*/organize_artifact.json"))
        relate_paths.extend(seed_dir.glob("runs/*/relate_artifact.json"))
for ws_dir in (REPO/"storage"/"workers").glob("*"):
    if ws_dir.is_dir():
        organize_paths.extend(ws_dir.glob("runs/*/organize_artifact.json"))
        relate_paths.extend(ws_dir.glob("runs/*/relate_artifact.json"))

_log(f"Found: {len(audit_paths)} audit / {len(organize_paths)} organize / "
     f"{len(relate_paths)} relate artifacts")

# Build lattice list: prefer audit_artifact, fall back to organize_artifact
lattices = []
seen = set()
for p in audit_paths:
    if p.parent.name in seen: continue
    seen.add(p.parent.name); lattices.append(p)
for p in organize_paths:
    if p.parent.name in seen: continue
    if (p.parent / "audit_artifact.json").exists(): continue
    seen.add(p.parent.name); lattices.append(p)

_log(f"Selected {len(lattices)} unique lattices for merge")

if not lattices:
    _log("ERROR: no lattices found; aborting"); raise SystemExit(1)

cmd = ["gdb", "run", "merge"]
for p in lattices: cmd += ["--source", str(p)]
for p in relate_paths: cmd += ["--relations", str(p)]
cmdline_len = sum(len(s) for s in cmd)
_log(f"Launching gdb run merge (sources={len(lattices)}, "
     f"relations={len(relate_paths)}, cmdline_len={cmdline_len} chars)")

env = os.environ.copy()
env["GDB_STORAGE"] = str(SNAPSHOT_STORAGE)  # ISOLATE from main pipeline

with open(MERGE_LOG, "ab") as f:
    f.write(f"\n=== {_ts()} :: snapshot merge ===\n".encode()); f.flush()
    result = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=f, stderr=f)

if result.returncode != 0:
    _log(f"MERGE FAILED rc={result.returncode}")
    _log(f"  see {MERGE_LOG} for stderr"); raise SystemExit(1)

_log("MERGE COMPLETE rc=0")

# Find the merge artifact
merge_artifacts = list(SNAPSHOT_STORAGE.glob("runs/*/merge_artifact.json"))
if not merge_artifacts:
    _log("WARNING: no merge_artifact.json found in snapshot storage")
    raise SystemExit(1)

ma = max(merge_artifacts, key=lambda p: p.stat().st_mtime)
_log(f"merge_artifact: {ma}")
_log(f"  size: {ma.stat().st_size:,} bytes")

try:
    d = json.loads(ma.read_text())
    groups = d.get("groups", [])
    items = sum(len(g.get("items", [])) for g in groups)
    operations = d.get("operations", [])
    # Each op has edges
    edges = sum(len(op.get("edges", [])) for op in operations)
    anchors = sum(len(e.get("anchor_list", []) or []) for op in operations for e in op.get("edges", []))
    _log(f"  groups: {len(groups):,}")
    _log(f"  items: {items:,}")
    _log(f"  operations: {len(operations):,}")
    _log(f"  edges: {edges:,}")
    _log(f"  anchors: {anchors:,}")
    # Per-kind breakdown
    from collections import Counter
    by_kind = Counter()
    for op in operations:
        for e in op.get("edges", []):
            by_kind[e.get("relation","?")] += 1
    _log(f"  Top relation types:")
    for k, n in by_kind.most_common(15):
        _log(f"    {n:,}  {k}")
except Exception as e:
    _log(f"  could not parse: {e!r}")

_log("SNAPSHOT MERGE END")
