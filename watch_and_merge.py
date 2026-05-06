#!/usr/bin/env python3
"""Watch retry_v3.py for exit, then run final merge with dedup.

Polls every 30s. When retry_v3.py is no longer running (BFS drained
or process died), waits 60s for final disk writes, gathers every
artifact path on disk, and runs `gdb run merge`. The merge command
performs:
  - lattice union with item-level dedup across runs
  - relation corroboration (stacking anchors when independent
    sources describe the same dependency)
  - conflict detection (sibling-endpoint disagreements flagged)

Output: run-logs/WATCH_AND_MERGE.log (this watcher's progress) +
run-logs/merge.log (gdb run merge stdout/stderr).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
LOG_DIR = REPO_ROOT / "run-logs"
SUMMARY_LOG = LOG_DIR / "WATCH_AND_MERGE.log"
MERGE_LOG = LOG_DIR / "merge.log"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}\n"
    with open(SUMMARY_LOG, "a") as f:
        f.write(line)


def main() -> None:
    SUMMARY_LOG.write_text("")
    _log("WATCH-AND-MERGE START")
    _log("Polling pgrep retry_v3.py every 30s; will run merge when it exits.")

    # Wait for retry_v3.py to exit
    while True:
        result = subprocess.run(
            ["pgrep", "-f", "retry_v3.py"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            _log("retry_v3.py is no longer running. Proceeding to merge.")
            break
        time.sleep(30)

    # Buffer for final disk writes from any tail-end workers
    _log("Waiting 60s for tail-end disk writes...")
    time.sleep(60)

    # Gather all artifact paths
    audit_paths: list[Path] = []
    organize_paths: list[Path] = []
    relate_paths: list[Path] = []

    seed_root = REPO_ROOT / "storage" / "seeds"
    if seed_root.exists():
        for seed_dir in seed_root.iterdir():
            if seed_dir.is_dir():
                audit_paths.extend(seed_dir.glob("runs/*/audit_artifact.json"))
                organize_paths.extend(seed_dir.glob("runs/*/organize_artifact.json"))
                relate_paths.extend(seed_dir.glob("runs/*/relate_artifact.json"))

    worker_root = REPO_ROOT / "storage" / "workers"
    if worker_root.exists():
        for ws_dir in worker_root.iterdir():
            if ws_dir.is_dir():
                organize_paths.extend(ws_dir.glob("runs/*/organize_artifact.json"))
                relate_paths.extend(ws_dir.glob("runs/*/relate_artifact.json"))

    _log(f"Found: {len(audit_paths)} audit / {len(organize_paths)} organize / "
         f"{len(relate_paths)} relate artifacts")

    # Build lattice list: prefer audit_artifact (revised), fall back to
    # organize_artifact for runs that never went through audit (recursive
    # workers ran with --skip audit).
    lattices: list[Path] = []
    seen_runs: set[str] = set()
    for p in audit_paths:
        if p.parent.name in seen_runs:
            continue
        seen_runs.add(p.parent.name)
        lattices.append(p)
    for p in organize_paths:
        if p.parent.name in seen_runs:
            continue
        if (p.parent / "audit_artifact.json").exists():
            continue
        seen_runs.add(p.parent.name)
        lattices.append(p)

    _log(f"Selected {len(lattices)} unique lattices for merge "
         f"({len(audit_paths)} audited + "
         f"{len(lattices) - len([p for p in audit_paths if p in lattices])} "
         f"organize-only)")

    if not lattices:
        _log("ERROR: no lattices found; aborting merge")
        return

    cmd = ["gdb", "run", "merge"]
    for p in lattices:
        cmd += ["--source", str(p)]
    for p in relate_paths:
        cmd += ["--relations", str(p)]

    _log(f"Launching gdb run merge with {len(lattices)} sources, "
         f"{len(relate_paths)} relations (cmdline length: {sum(len(s) for s in cmd)} chars)")

    with open(MERGE_LOG, "ab") as f:
        f.write(f"\n=== {_ts()} :: gdb run merge (started by watch_and_merge.py) ===\n".encode())
        f.flush()
        result = subprocess.run(
            cmd, env=os.environ.copy(), cwd=str(REPO_ROOT),
            stdout=f, stderr=f,
        )

    if result.returncode == 0:
        _log("MERGE COMPLETE rc=0")
        # Locate and inspect the merge artifact
        runs_root = REPO_ROOT / "storage" / "runs"
        merge_artifacts = list(runs_root.glob("*/merge_artifact.json")) if runs_root.exists() else []
        if merge_artifacts:
            ma = max(merge_artifacts, key=lambda p: p.stat().st_mtime)
            _log(f"merge_artifact: {ma} ({ma.stat().st_size} bytes)")
            try:
                import json
                d = json.loads(ma.read_text())
                _log(f"  groups: {len(d.get('groups', []))}")
                _log(f"  items: {sum(len(g.get('items', [])) for g in d.get('groups', []))}")
                _log(f"  operations: {len(d.get('operations', []))}")
            except Exception as e:
                _log(f"  (could not parse: {e!r})")
        else:
            _log("WARNING: no merge_artifact.json found in storage/runs/")
    else:
        _log(f"MERGE FAILED rc={result.returncode}")
        _log(f"  see {MERGE_LOG} for full output")
        # Show last 30 lines of merge log
        try:
            tail = MERGE_LOG.read_text(errors="replace").splitlines()[-30:]
            for line in tail:
                _log(f"  | {line}")
        except Exception:
            pass

    _log("WATCH-AND-MERGE END")


if __name__ == "__main__":
    main()
