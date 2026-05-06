#!/usr/bin/env python3
"""Resume BFS expansion across all 4 seeds with reduced concurrency.

Replaces the 3 dead orchestrators (run_seeds.py, resume_smollm3.py,
retry_failures.py). Per-seed lattices and 12+ completed expansions on
disk are preserved. State is recovered from existing artifacts and
seed_logs.

Concurrency: WORKER_CAP=6 per seed × 4 seeds = 24 max concurrent
expand subprocesses (vs ~64 in the original config) to keep the
machine stable.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
SEED_ROOT = REPO_ROOT / "storage" / "seeds"
WORKER_ROOT = REPO_ROOT / "storage" / "workers"
LOG_DIR = REPO_ROOT / "run-logs"

# (seed_target, seed_log_basename, seed_storage_dirname)
SEEDS = [
    ("OLMo 3", "OLMo_3"),
    ("rl-research/DR-Tulu-8B", "rl-research_DR-Tulu-8B"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
     "nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
    ("HuggingFaceTB/SmolLM3-3B", "HuggingFaceTB_SmolLM3-3B"),
]
DEPTH_CAP = 5
WORKER_CAP = 6  # per seed; 4 seeds × 6 = 24 total
PLANNER = "opus"
SUBAGENT = "opus"

# Throttle within-batch parallelism too — extract/relate fan out to
# this many concurrent claude processes per worker. Lowered from 64.
GDB_MAX_PARALLEL = "16"

WORKER_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

worker_storages: list[Path] = []
worker_storages_lock = threading.Lock()

# Regex for parsing seed_log
RX_EXPAND_START = re.compile(
    r">>\s*(?:\[retry\]\s*)?expand depth=(\d+) worker=(\S+) node=(['\"])(.+?)\3"
)
RX_EXPAND_FAILED = re.compile(
    r"ERROR\s*(?:\[retry\]\s*)?expand failed; node=(\S+)"
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


def expand_node(seed_log: Path, node: str, depth: int) -> list[tuple[str, int]]:
    """Run isolated expand+triage for one node. Returns child (node, depth+1) pairs."""
    worker_id = str(uuid.uuid4())[:8]
    worker_storage = WORKER_ROOT / worker_id
    worker_log = LOG_DIR / f"worker_{worker_id}.log"

    _log(seed_log, f"  >> [resume] expand depth={depth} worker={worker_id} node={node!r}")
    env = {"GDB_STORAGE": str(worker_storage)}

    rc = _run_cmd(["gdb", "init", "--fresh", "--yes", "--I-mean-it"],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [resume] ERROR init failed; node={node}")
        return []

    rc = _run_cmd(["gdb", "run", "expand", "--node", node, "--skip", "audit",
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [resume] ERROR expand failed; node={node}")
        return []

    organize_artifact = _find_artifact(worker_storage, "organize_artifact.json")
    if not organize_artifact:
        _log(seed_log, f"     [resume] ERROR no organize artifact; node={node}")
        return []

    rc = _run_cmd(["gdb", "run", "triage",
                   "--lattice", str(organize_artifact),
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(seed_log, f"     [resume] WARN triage failed; node={node}")
        return []

    triage_artifact = _find_artifact(worker_storage, "triage_artifact.json")
    if not triage_artifact:
        return []
    triage_data = json.loads(triage_artifact.read_text())
    children: list[tuple[str, int]] = []
    for entry in triage_data.get("auto_expand", []):
        child_node = entry["formal_name"]
        rationale = entry.get("rationale", "")
        _log(seed_log,
             f"     [resume] ++ auto_expand depth={depth+1}: {child_node!r} :: {rationale}")
        children.append((child_node, depth + 1))

    declined = len(triage_data.get("decline", []))
    manual = len(triage_data.get("manual", []))
    _log(seed_log,
         f"     [resume] << complete; expand={len(children)} decline={declined} manual={manual}")

    with worker_storages_lock:
        worker_storages.append(worker_storage)
    return children


def gather_state_for_seed(seed_target: str, seed_slug: str, seed_log: Path
                          ) -> tuple[set, list[tuple[str, int]]]:
    """Recover seed state from disk. Returns (seen_set, queue).

    seen = nodes we should NOT redo (completed or permanently failed).
    queue = nodes to re-spawn (initial depth-1 + chained children of
            completed workers + in-flight nodes whose work was lost).
    """
    seen: set[str] = set()
    queue_candidates: list[tuple[str, int]] = []

    seed_storage = SEED_ROOT / seed_slug

    # 1. Initial depth-1 queue from seed's own triage_artifact.json
    seed_triage = _find_artifact(seed_storage, "triage_artifact.json")
    if seed_triage and seed_triage.exists():
        try:
            triage_data = json.loads(seed_triage.read_text())
            for entry in triage_data.get("auto_expand", []):
                queue_candidates.append((entry["formal_name"], 1))
        except Exception as e:
            _log(seed_log, f"  [resume] WARN: could not parse seed triage: {e!r}")

    # 2. Walk seed_log to find all worker starts and permanent failures.
    workers_started: dict[str, tuple[int, str]] = {}  # worker_id -> (depth, node)
    if seed_log.exists():
        for line in seed_log.read_text().splitlines():
            m = RX_EXPAND_START.search(line)
            if m:
                depth, worker_id, _, node = m.groups()
                workers_started[worker_id] = (int(depth), node)
            m = RX_EXPAND_FAILED.search(line)
            if m:
                seen.add(m.group(1).strip("'\""))

    # 3. For each worker that started: check if it completed (has triage
    #    artifact). If so, mark its node seen and chain its children.
    for worker_id, (depth, node) in workers_started.items():
        worker_storage = WORKER_ROOT / worker_id
        worker_triage = _find_artifact(worker_storage, "triage_artifact.json")
        if worker_triage and worker_triage.exists():
            seen.add(node)
            try:
                wd = json.loads(worker_triage.read_text())
                for entry in wd.get("auto_expand", []):
                    queue_candidates.append(
                        (entry["formal_name"], depth + 1)
                    )
            except Exception:
                pass
        # else: worker did not complete → its node will be re-tried
        # via the initial-queue / chained-children path (unless it's
        # not in either, in which case it was a depth-2+ in-flight
        # whose parent did complete; we may miss redoing those, which
        # is acceptable).

    # 4. Filter queue: dedupe, drop seen, drop > DEPTH_CAP.
    seen_in_queue: set[str] = set()
    filtered: list[tuple[str, int]] = []
    for n, d in queue_candidates:
        if n in seen or n in seen_in_queue:
            continue
        if d > DEPTH_CAP:
            continue
        seen_in_queue.add(n)
        filtered.append((n, d))

    return seen, filtered


def process_seed(seed_target: str, seed_slug: str) -> int:
    """Run BFS pool for one seed. Returns count of newly-processed nodes."""
    seed_log = LOG_DIR / f"seed_{seed_slug}.log"
    _log(seed_log, f"=== RESUME: {seed_target} (WORKER_CAP={WORKER_CAP}) ===")

    seen, initial_queue = gather_state_for_seed(seed_target, seed_slug, seed_log)
    _log(seed_log,
         f"  state recovered: |seen|={len(seen)}  |queue|={len(initial_queue)}")
    if not initial_queue:
        _log(seed_log, "  no work to do; skipping")
        return 0

    seen_lock = threading.Lock()

    def maybe_enqueue(executor: ThreadPoolExecutor, futures: list[Future],
                      node: str, depth: int) -> None:
        if depth > DEPTH_CAP:
            return
        with seen_lock:
            if node in seen:
                return
            seen.add(node)
        fut = executor.submit(expand_node, seed_log, node, depth)
        futures.append(fut)

    new_count = 0
    with ThreadPoolExecutor(max_workers=WORKER_CAP) as executor:
        futures: list[Future] = []
        for n, d in initial_queue:
            with seen_lock:
                if n in seen:
                    continue
                seen.add(n)
            fut = executor.submit(expand_node, seed_log, n, d)
            futures.append(fut)
            new_count += 1

        idx = 0
        while idx < len(futures):
            fut = futures[idx]
            idx += 1
            try:
                children = fut.result()
            except Exception as e:
                _log(seed_log, f"  [resume] worker error: {e!r}")
                continue
            for cn, cd in children:
                before = len(seen)
                maybe_enqueue(executor, futures, cn, cd)
                if len(seen) > before:
                    new_count += 1

    _log(seed_log, f"=== {seed_target} RESUME COMPLETE :: new_processed = {new_count} ===")
    return new_count


def main() -> None:
    summary_log = LOG_DIR / "RESUME_BFS_SUMMARY.log"
    summary_log.write_text("")
    _log(summary_log, "RESUME-BFS ORCHESTRATOR START")
    _log(summary_log,
         f"depth_cap={DEPTH_CAP} per_seed_worker_cap={WORKER_CAP} "
         f"gdb_max_parallel={GDB_MAX_PARALLEL} "
         f"planner={PLANNER} subagent={SUBAGENT}")
    _log(summary_log, f"seeds: {[s for s, _ in SEEDS]}")

    # Run all 4 seeds in parallel.
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = {executor.submit(process_seed, t, s): t for t, s in SEEDS}
        for fut in futures:
            seed = futures[fut]
            try:
                fut.result()
                _log(summary_log, f"--- finished seed: {seed}")
            except Exception as e:
                _log(summary_log, f"--- seed crashed: {seed} :: {e!r}")

    # Final merge across all storages.
    _log(summary_log, "FINAL MERGE")
    audit_paths: list[Path] = []
    organize_paths: list[Path] = []
    relate_paths: list[Path] = []
    for _, slug in SEEDS:
        ss = SEED_ROOT / slug
        if ss.exists():
            audit_paths.extend(ss.glob("runs/*/audit_artifact.json"))
            organize_paths.extend(ss.glob("runs/*/organize_artifact.json"))
            relate_paths.extend(ss.glob("runs/*/relate_artifact.json"))
    if WORKER_ROOT.exists():
        for ws in WORKER_ROOT.iterdir():
            if ws.is_dir():
                organize_paths.extend(ws.glob("runs/*/organize_artifact.json"))
                relate_paths.extend(ws.glob("runs/*/relate_artifact.json"))

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

    cmd = ["gdb", "run", "merge"]
    for p in lattices:
        cmd += ["--source", str(p)]
    for p in relate_paths:
        cmd += ["--relations", str(p)]

    merge_log = LOG_DIR / "merge.log"
    _log(summary_log,
         f"merge: {len(lattices)} lattices, {len(relate_paths)} relations")
    rc = _run_cmd(cmd, merge_log)
    _log(summary_log, "merge complete" if rc == 0 else f"merge FAILED rc={rc}")
    _log(summary_log, "RESUME-BFS ORCHESTRATOR END")


if __name__ == "__main__":
    main()
