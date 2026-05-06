#!/usr/bin/env python3
"""Retry-pass v3: re-fire every node that didn't complete.

Walks all 4 seed_logs and pulls up two categories of unfinished nodes:
  1. Nodes with `ERROR ... expand failed; node=X` markers (~34)
  2. Nodes with `>> expand ... node='X'` but no matching completion or
     failure (in-flight when the orchestrator was killed; ~20)

Each is re-fired in a fresh BFS pool with WORKER_CAP=6 (no other
orchestrator running). Children of completed retries get chain-queued
so depth recursion continues.

Disk-usage hardening: each worker has its own GDB_STORAGE, but
workspace/ dirs accumulate to ~150 MB per worker. With 50+ retries
plus chain children we'd hit the same disk-full issue. To prevent:
this script periodically cleans up workspace/ and stream.jsonl from
completed worker storages (preserving artifacts).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
WORKER_ROOT = REPO_ROOT / "storage" / "workers"
LOG_DIR = REPO_ROOT / "run-logs"

SEED_LOGS = [
    "seed_OLMo_3.log",
    "seed_rl-research_DR-Tulu-8B.log",
    "seed_nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4.log",
    "seed_HuggingFaceTB_SmolLM3-3B.log",
]

DEPTH_CAP = 5
WORKER_CAP = 6
PLANNER = "opus"
SUBAGENT = "opus"
GDB_MAX_PARALLEL = "16"

WORKER_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RX_STARTED = re.compile(
    r">>\s*(?:\[(?:retry|resume|v2-retry|v3-retry)\]\s*)?expand depth=(\d+) worker=(\S+) node=(['\"])(.+?)\3"
)
RX_FAILED = re.compile(
    r"ERROR(?:\s*\[(?:retry|resume|v2-retry|v3-retry)\])?\s*expand failed;\s*node=(.+?)\s*$"
)
RX_COMPLETE = re.compile(
    r"<<\s*(?:\[(?:retry|resume|v2-retry|v3-retry)\]\s*)?complete; expand=\d+"
)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(path: Path, msg: str) -> None:
    with open(path, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")


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
    """After a worker completes, remove workspace/ and stream.jsonl to
    keep disk usage under control. Preserves all *_artifact.json."""
    for ws in worker_storage.glob("runs/*/workspace"):
        if ws.is_dir():
            shutil.rmtree(ws, ignore_errors=True)
    for sj in worker_storage.glob("runs/*/stream.jsonl"):
        try:
            sj.unlink()
        except Exception:
            pass


def expand_node(seed_log: Path, node: str, depth: int) -> list[tuple[str, int]]:
    worker_id = str(uuid.uuid4())[:8]
    worker_storage = WORKER_ROOT / worker_id
    worker_log = LOG_DIR / f"worker_{worker_id}.log"

    _log(seed_log, f"  >> [v3-retry] expand depth={depth} worker={worker_id} node={node!r}")
    env = {"GDB_STORAGE": str(worker_storage)}

    rc = _run_cmd(["gdb", "init", "--fresh", "--yes", "--I-mean-it"],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [v3-retry] ERROR init failed; node={node}")
        return []

    rc = _run_cmd(["gdb", "run", "expand", "--node", node, "--skip", "audit",
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [v3-retry] ERROR expand failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return []

    organize_artifact = _find_artifact(worker_storage, "organize_artifact.json")
    if not organize_artifact:
        _log(seed_log, f"     [v3-retry] ERROR no organize artifact; node={node}")
        _cleanup_worker_disk(worker_storage)
        return []

    rc = _run_cmd(["gdb", "run", "triage",
                   "--lattice", str(organize_artifact),
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [v3-retry] WARN triage failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return []

    triage_artifact = _find_artifact(worker_storage, "triage_artifact.json")
    if not triage_artifact:
        _cleanup_worker_disk(worker_storage)
        return []
    triage_data = json.loads(triage_artifact.read_text())
    children: list[tuple[str, int]] = []
    for entry in triage_data.get("auto_expand", []):
        child_node = entry["formal_name"]
        rationale = entry.get("rationale", "")
        _log(seed_log,
             f"     [v3-retry] ++ auto_expand depth={depth+1}: {child_node!r} :: {rationale}")
        children.append((child_node, depth + 1))

    declined = len(triage_data.get("decline", []))
    manual = len(triage_data.get("manual", []))
    _log(seed_log,
         f"     [v3-retry] << complete; expand={len(children)} decline={declined} manual={manual}")

    # Free disk: remove workspace + stream after artifacts captured
    _cleanup_worker_disk(worker_storage)
    return children


def gather_unfinished_nodes() -> list[tuple[str, int, Path, str]]:
    """Returns [(node, depth, seed_log, reason)] for every node that
    didn't reach <<complete>> in its seed_log."""
    out: list[tuple[str, int, Path, str]] = []

    for log_name in SEED_LOGS:
        seed_log = LOG_DIR / log_name
        if not seed_log.exists():
            continue

        # Build worker_id -> (depth, node) from start lines
        worker_starts: dict[str, tuple[int, str]] = {}
        for line in seed_log.read_text().splitlines():
            m = RX_STARTED.search(line)
            if m:
                depth, worker_id, _q, node = m.groups()
                worker_starts[worker_id] = (int(depth), node)

        # For each worker, check if it has triage_artifact (= completed)
        completed_nodes: set[str] = set()
        for worker_id, (depth, node) in worker_starts.items():
            ws = WORKER_ROOT / worker_id
            if (_find_artifact(ws, "triage_artifact.json") is not None):
                completed_nodes.add(node)

        # Failed nodes from log
        failed_nodes: set[str] = set()
        for line in seed_log.read_text().splitlines():
            m = RX_FAILED.search(line)
            if m:
                failed_nodes.add(m.group(1).strip("'\""))

        # Unfinished = started but not completed
        # Note: a node may appear in BOTH failed AND completed via different retries
        # In that case treat as completed.
        seen_in_seed: set[str] = set()
        for worker_id, (depth, node) in worker_starts.items():
            if node in completed_nodes:
                continue
            if node in seen_in_seed:
                continue
            seen_in_seed.add(node)
            reason = "failed" if node in failed_nodes else "in-flight-killed"
            out.append((node, depth, seed_log, reason))

    return out


def main() -> None:
    summary_log = LOG_DIR / "RETRY_V3_SUMMARY.log"
    summary_log.write_text("")
    _log(summary_log, "RETRY-V3 ORCHESTRATOR START")
    _log(summary_log,
         f"depth_cap={DEPTH_CAP} worker_cap={WORKER_CAP} "
         f"gdb_max_parallel={GDB_MAX_PARALLEL} planner={PLANNER}")

    initial = gather_unfinished_nodes()
    _log(summary_log, f"loaded {len(initial)} unfinished nodes to retry")
    by_reason: dict[str, int] = {}
    for node, depth, seed_log, reason in initial:
        by_reason[reason] = by_reason.get(reason, 0) + 1
        _log(summary_log,
             f"  - {node!r} depth={depth} reason={reason} parent={seed_log.name}")
    for r, n in by_reason.items():
        _log(summary_log, f"  by reason: {r} = {n}")

    # BFS pool
    seen: set[str] = set()
    seen_lock = threading.Lock()

    def _wrap(seed_log: Path, node: str, depth: int):
        return (seed_log, expand_node(seed_log, node, depth))

    def maybe_enqueue(executor, futures, node, depth, seed_log):
        if depth > DEPTH_CAP:
            return
        with seen_lock:
            if node in seen:
                return
            seen.add(node)
        futures.append(executor.submit(_wrap, seed_log, node, depth))

    with ThreadPoolExecutor(max_workers=WORKER_CAP) as executor:
        futures: list[Future] = []
        for node, depth, seed_log, _ in initial:
            with seen_lock:
                if node in seen:
                    continue
                seen.add(node)
            futures.append(executor.submit(_wrap, seed_log, node, depth))

        idx = 0
        while idx < len(futures):
            fut = futures[idx]
            idx += 1
            try:
                seed_log, children = fut.result()
            except Exception as e:
                _log(summary_log, f"worker error: {e!r}")
                continue
            for cn, cd in children:
                maybe_enqueue(executor, futures, cn, cd, seed_log)

    _log(summary_log, f"=== RETRY-V3 COMPLETE :: total processed = {len(seen)} ===")


if __name__ == "__main__":
    main()
