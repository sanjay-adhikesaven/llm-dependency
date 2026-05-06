#!/usr/bin/env python3
"""Auto-merge after BOTH recurse_v4 AND beam_v5 finish.

Polls every 30s for both to be absent, then runs `gdb run merge` over
all artifacts on disk and writes RUN_DONE.txt with the final stats.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
LOG_DIR = REPO_ROOT / "run-logs"
SUMMARY_LOG = LOG_DIR / "AUTO_MERGE_COMBINED.log"
MERGE_LOG = LOG_DIR / "merge.log"
RUN_DONE = LOG_DIR / "RUN_DONE.txt"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    with open(SUMMARY_LOG, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")


def _alive(name: str) -> bool:
    r = subprocess.run(["pgrep", "-f", name],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def main() -> None:
    SUMMARY_LOG.write_text("")
    _log("AUTO-MERGE-COMBINED START")
    _log("Polling for both recurse_v4.py AND beam_v5.py to exit (every 30s)...")

    while True:
        rec = _alive("recurse_v4.py")
        beam = _alive("beam_v5.py")
        if not rec and not beam:
            _log("Both recurse_v4 and beam_v5 have exited. Proceeding to merge.")
            break
        time.sleep(30)

    _log("Waiting 60s for tail-end disk writes...")
    time.sleep(60)

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

    _log(f"Selected {len(lattices)} unique lattices for merge")

    if not lattices:
        _log("ERROR: no lattices found; aborting merge")
        RUN_DONE.write_text(f"{_ts()} :: FAILED — no lattices found\n")
        return

    cmd = ["gdb", "run", "merge"]
    for p in lattices:
        cmd += ["--source", str(p)]
    for p in relate_paths:
        cmd += ["--relations", str(p)]

    _log(f"Launching gdb run merge with {len(lattices)} sources, "
         f"{len(relate_paths)} relations (cmdline {sum(len(s) for s in cmd)} chars)")

    with open(MERGE_LOG, "ab") as f:
        f.write(f"\n=== {_ts()} :: gdb run merge (auto_merge_combined) ===\n".encode())
        f.flush()
        result = subprocess.run(
            cmd, env=os.environ.copy(), cwd=str(REPO_ROOT),
            stdout=f, stderr=f,
        )

    if result.returncode == 0:
        _log("MERGE COMPLETE rc=0")
        runs_root = REPO_ROOT / "storage" / "runs"
        merge_artifacts = list(runs_root.glob("*/merge_artifact.json")) if runs_root.exists() else []
        if merge_artifacts:
            ma = max(merge_artifacts, key=lambda p: p.stat().st_mtime)
            _log(f"merge_artifact: {ma} ({ma.stat().st_size} bytes)")
            try:
                d = json.loads(ma.read_text())
                groups = len(d.get('groups', []))
                items = sum(len(g.get('items', [])) for g in d.get('groups', []))
                operations = len(d.get('operations', []))
                _log(f"  groups: {groups}")
                _log(f"  items: {items}")
                _log(f"  operations: {operations}")
                RUN_DONE.write_text(
                    f"{_ts()} :: SUCCESS (Path D + beam_v5)\n"
                    f"merge_artifact: {ma}\n"
                    f"groups={groups}\n"
                    f"items={items}\n"
                    f"operations={operations}\n"
                    f"sources={len(lattices)}\n"
                    f"relations={len(relate_paths)}\n"
                )
            except Exception as e:
                _log(f"  (could not parse merge artifact: {e!r})")
                RUN_DONE.write_text(
                    f"{_ts()} :: SUCCESS but artifact parse failed\n"
                    f"merge_artifact: {ma}\n"
                    f"parse_error: {e!r}\n"
                )
        else:
            _log("WARNING: no merge_artifact.json found in storage/runs/")
            RUN_DONE.write_text(
                f"{_ts()} :: PARTIAL (merge succeeded but no artifact on disk)\n"
            )
    else:
        _log(f"MERGE FAILED rc={result.returncode}")
        try:
            tail = MERGE_LOG.read_text(errors="replace").splitlines()[-30:]
            for line in tail:
                _log(f"  | {line}")
        except Exception:
            pass
        RUN_DONE.write_text(
            f"{_ts()} :: FAILED — gdb run merge exited rc={result.returncode}\n"
            f"see {MERGE_LOG} for details\n"
        )

    _log("AUTO-MERGE-COMBINED END")


if __name__ == "__main__":
    main()
