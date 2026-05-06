#!/usr/bin/env python3
"""Recurse-v4: focused expansion on the top-10 most-suggested ghost nodes.

Picks up where the BFS left off — these are nodes that triage_artifacts
recommended expanding (auto_expand) but never got worker capacity.

NO chain recursion: each node is expanded independently to depth+0; we
do NOT enqueue their auto_expand children. This keeps the run bounded
(~2h for 10 nodes with WORKER_CAP=4).

Each worker runs: gdb init --fresh → gdb run expand → gdb run triage,
with per-worker GDB_STORAGE so artifacts survive for the final merge.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
WORKER_ROOT = REPO_ROOT / "storage" / "workers"
LOG_DIR = REPO_ROOT / "run-logs"

WORKER_CAP = 4
PLANNER = "opus"
SUBAGENT = "opus"
GDB_MAX_PARALLEL = "16"

WORKER_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOP_GHOSTS = [
    "allenai/WildChat-1M",
    "BytedTsinghua-SIA/DAPO-Math-17k",
    "allenai/tulu-3-sft-mixture",
    "allenai/tulu-3-sft-personas-math-grade",
    "nvidia/OpenMathInstruct-2",
    "OpenAssistant/oasst1",
    "allenai/tulu-3-sft-personas-code",
    "AceCoder",
    "allenai/coconot",
    "allenai/wildguardmix",
]

SUMMARY_LOG = LOG_DIR / "RECURSE_V4_SUMMARY.log"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}\n"
    with open(SUMMARY_LOG, "a") as f:
        f.write(line)


def _run_cmd(cmd: list[str], log_path: Path,
             env_overrides: dict | None = None) -> int:
    env = os.environ.copy()
    env["GDB_MAX_PARALLEL_BATCHES"] = GDB_MAX_PARALLEL
    if env_overrides:
        env.update(env_overrides)
    with open(log_path, "ab") as logf:
        logf.write(f"\n=== {_ts()} :: {' '.join(cmd)} ===\n".encode())
        logf.flush()
        result = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=logf, stderr=logf,
        )
    return result.returncode


def _find_artifact(storage: Path, filename: str) -> Path | None:
    runs = storage / "runs"
    if not runs.exists():
        return None
    candidates = list(runs.glob(f"*/{filename}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _cleanup_worker_disk(worker_storage: Path) -> None:
    for ws in worker_storage.glob("runs/*/workspace"):
        if ws.is_dir():
            shutil.rmtree(ws, ignore_errors=True)
    for sj in worker_storage.glob("runs/*/stream.jsonl"):
        try:
            sj.unlink()
        except Exception:
            pass


def expand_node(node: str) -> tuple[str, str]:
    """Run init+expand+triage on a node. Returns (node, status)."""
    worker_id = str(uuid.uuid4())[:8]
    worker_storage = WORKER_ROOT / worker_id
    worker_log = LOG_DIR / f"worker_{worker_id}.log"

    _log(f"  >> [v4-recurse] expand worker={worker_id} node={node!r}")
    env = {"GDB_STORAGE": str(worker_storage)}

    rc = _run_cmd(["gdb", "init", "--fresh", "--yes", "--I-mean-it"],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v4-recurse] ERROR init failed; node={node}")
        return (node, "init-failed")

    rc = _run_cmd(["gdb", "run", "expand", "--node", node, "--skip", "audit",
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v4-recurse] ERROR expand failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "expand-failed")

    organize_artifact = _find_artifact(worker_storage, "organize_artifact.json")
    if not organize_artifact:
        _log(f"     [v4-recurse] ERROR no organize artifact; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "no-organize")

    rc = _run_cmd(["gdb", "run", "triage",
                   "--lattice", str(organize_artifact),
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v4-recurse] WARN triage failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "triage-failed")

    triage_artifact = _find_artifact(worker_storage, "triage_artifact.json")
    if not triage_artifact:
        _cleanup_worker_disk(worker_storage)
        return (node, "no-triage")

    _log(f"     [v4-recurse] << complete; node={node} worker={worker_id}")
    _cleanup_worker_disk(worker_storage)
    return (node, "complete")


def main() -> None:
    SUMMARY_LOG.write_text("")
    _log("RECURSE-V4 START")
    _log(f"worker_cap={WORKER_CAP} planner={PLANNER} subagent={SUBAGENT}")
    _log(f"targets ({len(TOP_GHOSTS)}):")
    for n in TOP_GHOSTS:
        _log(f"  - {n}")

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=WORKER_CAP) as executor:
        futures: list[Future] = [executor.submit(expand_node, n)
                                 for n in TOP_GHOSTS]
        for fut in futures:
            try:
                node, status = fut.result()
                results[node] = status
            except Exception as e:
                _log(f"worker error: {e!r}")

    _log("=== RECURSE-V4 COMPLETE ===")
    by_status: dict[str, int] = {}
    for n, s in results.items():
        _log(f"  {n} :: {s}")
        by_status[s] = by_status.get(s, 0) + 1
    for s, c in by_status.items():
        _log(f"  status {s}: {c}")


if __name__ == "__main__":
    main()
