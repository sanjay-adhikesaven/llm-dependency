#!/usr/bin/env python3
"""Final orchestrator: chain-retries until convergence, then merge.

Hardened version. Wake-up-to-done entrypoint.

Robustness:
- Merge: retried up to 3 times on failure.
- Disk pre-flight: if free disk < 3 GB before merge, aggressive cleanup.
- Final state file: `run-logs/RUN_DONE.txt` written at completion.
- Exception handler: logs trace before re-raising.
- Idempotent: if RUN_DONE.txt already exists, exits early.
- Lock: refuses to start a second merge if one is already running.

Outer wrapper: `keep_alive_orchestrator.sh` re-runs this script if it
dies prematurely (before RUN_DONE.txt is written).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
LOG_DIR = REPO_ROOT / "run-logs"
SUMMARY_LOG = LOG_DIR / "FINAL_ORCHESTRATOR.log"
MERGE_LOG = LOG_DIR / "merge.log"
RUN_DONE_FILE = LOG_DIR / "RUN_DONE.txt"
LOCKFILE = LOG_DIR / ".final_orchestrator.lock"

MAX_CHAIN_RETRIES = 2  # v3 + 2 chain passes = 3 retry layers
MAX_MERGE_ATTEMPTS = 3
DISK_PREFLIGHT_GB = 3

SEED_LOGS = [
    "seed_OLMo_3.log",
    "seed_rl-research_DR-Tulu-8B.log",
    "seed_nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4.log",
    "seed_HuggingFaceTB_SmolLM3-3B.log",
]

RX_STARTED = re.compile(
    r">>\s*(?:\[(?:retry|resume|v2-retry|v3-retry)\]\s*)?expand depth=(\d+) worker=(\S+) node=(['\"])(.+?)\3"
)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}\n"
    with open(SUMMARY_LOG, "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def free_disk_gb() -> float:
    stat = shutil.disk_usage(str(REPO_ROOT))
    return stat.free / (1024**3)


def aggressive_cleanup() -> None:
    """Free disk by removing all worker workspaces and stream files,
    even from in-flight workers (they shouldn't exist by merge time)."""
    n_ws = 0
    n_streams = 0
    for ws in (REPO_ROOT / "storage" / "workers").glob("*/runs/*/workspace"):
        if ws.is_dir():
            shutil.rmtree(ws, ignore_errors=True)
            n_ws += 1
    for sj in (REPO_ROOT / "storage" / "workers").glob("*/runs/*/stream.jsonl"):
        try:
            sj.unlink()
            n_streams += 1
        except Exception:
            pass
    _log(f"  aggressive_cleanup: removed {n_ws} workspaces, {n_streams} stream files")


def wait_for_exit(process_substr: str, poll_seconds: int = 30) -> None:
    while True:
        result = subprocess.run(
            ["pgrep", "-f", process_substr],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return
        time.sleep(poll_seconds)


def count_unfinished_nodes() -> tuple[int, dict[str, int]]:
    worker_root = REPO_ROOT / "storage" / "workers"
    unfinished_per_seed: dict[str, set[str]] = {}
    for log_name in SEED_LOGS:
        seed_log = LOG_DIR / log_name
        if not seed_log.exists():
            unfinished_per_seed[log_name] = set()
            continue
        worker_starts: dict[str, str] = {}
        for line in seed_log.read_text().splitlines():
            m = RX_STARTED.search(line)
            if m:
                _, worker_id, _, node = m.groups()
                worker_starts[worker_id] = node
        nodes_unfinished: set[str] = set()
        for worker_id, node in worker_starts.items():
            ws = worker_root / worker_id
            if not ws.exists():
                nodes_unfinished.add(node)
                continue
            if not list(ws.glob("runs/*/triage_artifact.json")):
                nodes_unfinished.add(node)
        unfinished_per_seed[log_name] = nodes_unfinished
    total = sum(len(s) for s in unfinished_per_seed.values())
    return total, {k: len(v) for k, v in unfinished_per_seed.items()}


def run_retry_pass(label: str) -> int:
    _log(f"Launching {label} (running retry_v3.py)")
    cmd = ["/opt/anaconda3/bin/python", str(REPO_ROOT / "retry_v3.py")]
    log_path = LOG_DIR / f"{label.lower().replace(' ', '_')}.log"
    with open(log_path, "ab") as f:
        f.write(f"\n=== {_ts()} :: {label} (final_orchestrator) ===\n".encode())
        f.flush()
        proc = subprocess.Popen(
            cmd, env=os.environ.copy(), cwd=str(REPO_ROOT),
            stdout=f, stderr=f,
        )
        while proc.poll() is None:
            time.sleep(30)
    rc = proc.returncode
    _log(f"{label} exited rc={rc}")
    return rc


def gather_artifact_paths() -> tuple[list[Path], list[Path]]:
    audit_paths: list[Path] = []
    organize_paths: list[Path] = []
    relate_paths: list[Path] = []
    seed_root = REPO_ROOT / "storage" / "seeds"
    if seed_root.exists():
        for sd in seed_root.iterdir():
            if sd.is_dir():
                audit_paths.extend(sd.glob("runs/*/audit_artifact.json"))
                organize_paths.extend(sd.glob("runs/*/organize_artifact.json"))
                relate_paths.extend(sd.glob("runs/*/relate_artifact.json"))
    worker_root = REPO_ROOT / "storage" / "workers"
    if worker_root.exists():
        for ws in worker_root.iterdir():
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
    return lattices, relate_paths


def run_merge_once() -> tuple[int, dict | None]:
    """Run gdb run merge once. Returns (rc, stats_or_none)."""
    free_gb = free_disk_gb()
    _log(f"  pre-merge disk: {free_gb:.2f} GB free")
    if free_gb < DISK_PREFLIGHT_GB:
        _log(f"  WARNING free disk ({free_gb:.2f} GB) < {DISK_PREFLIGHT_GB} GB; running cleanup")
        aggressive_cleanup()
        free_gb = free_disk_gb()
        _log(f"  post-cleanup: {free_gb:.2f} GB free")

    lattices, relate_paths = gather_artifact_paths()
    _log(f"  inputs: {len(lattices)} lattices, {len(relate_paths)} relations")
    if not lattices:
        _log("  ERROR: no lattices found")
        return -1, None

    cmd = ["gdb", "run", "merge"]
    for p in lattices:
        cmd += ["--source", str(p)]
    for p in relate_paths:
        cmd += ["--relations", str(p)]

    with open(MERGE_LOG, "ab") as f:
        f.write(f"\n=== {_ts()} :: gdb run merge ===\n".encode())
        f.flush()
        result = subprocess.run(
            cmd, env=os.environ.copy(), cwd=str(REPO_ROOT),
            stdout=f, stderr=f,
        )

    if result.returncode != 0:
        _log(f"  merge rc={result.returncode}; see {MERGE_LOG}")
        return result.returncode, None

    runs_root = REPO_ROOT / "storage" / "runs"
    merge_artifacts = list(runs_root.glob("*/merge_artifact.json")) if runs_root.exists() else []
    if not merge_artifacts:
        _log("  merge returned 0 but no merge_artifact.json found")
        return -2, None
    ma = max(merge_artifacts, key=lambda p: p.stat().st_mtime)
    try:
        d = json.loads(ma.read_text())
        return 0, {
            "merge_artifact": str(ma),
            "groups": len(d.get("groups", [])),
            "items": sum(len(g.get("items", [])) for g in d.get("groups", [])),
            "operations": len(d.get("operations", [])),
            "size_bytes": ma.stat().st_size,
        }
    except Exception as e:
        _log(f"  parse error: {e!r}")
        return -3, None


def run_merge_with_retry() -> tuple[int, dict | None]:
    """Run merge up to MAX_MERGE_ATTEMPTS times."""
    last_rc = -1
    last_stats = None
    for attempt in range(1, MAX_MERGE_ATTEMPTS + 1):
        _log(f"MERGE attempt {attempt}/{MAX_MERGE_ATTEMPTS}")
        rc, stats = run_merge_once()
        if rc == 0:
            return 0, stats
        last_rc, last_stats = rc, stats
        if attempt < MAX_MERGE_ATTEMPTS:
            _log(f"  sleeping 60s before retry...")
            time.sleep(60)
    return last_rc, last_stats


def write_run_done(stats: dict | None, rc: int, error: str | None = None) -> None:
    lines = [
        "=" * 60,
        f"FINAL RUN OUTCOME ({_ts()})",
        "=" * 60,
    ]
    if stats and rc == 0:
        lines += [
            "STATUS: ✅ SUCCESS",
            "",
            f"merge_artifact: {stats['merge_artifact']}",
            f"size: {stats['size_bytes']:,} bytes",
            f"groups: {stats['groups']:,}",
            f"items: {stats['items']:,}",
            f"operations (edges): {stats['operations']:,}",
        ]
    else:
        lines += [
            f"STATUS: ❌ MERGE FAILED rc={rc}",
            f"see {MERGE_LOG} for details",
        ]
        if error:
            lines += ["", f"error: {error}"]
    lines += ["", "Logs:",
              f"  {SUMMARY_LOG}",
              f"  {MERGE_LOG}",
              "  run-logs/seed_*.log (per-seed expansion timeline)"]
    RUN_DONE_FILE.write_text("\n".join(lines) + "\n")


def acquire_lock() -> bool:
    if LOCKFILE.exists():
        # Check if the PID in the lockfile is still alive
        try:
            old_pid = int(LOCKFILE.read_text().strip())
            r = subprocess.run(["ps", "-p", str(old_pid)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                _log(f"Lock held by PID {old_pid}; another instance is running. Exiting.")
                return False
            else:
                _log(f"Stale lock from PID {old_pid}; reclaiming.")
        except Exception:
            pass
    LOCKFILE.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    try:
        LOCKFILE.unlink()
    except Exception:
        pass


def main() -> int:
    # Idempotent: if RUN_DONE.txt already exists, exit early.
    if RUN_DONE_FILE.exists():
        _log(f"RUN_DONE.txt already exists; this run is complete. Exiting.")
        return 0

    if not acquire_lock():
        return 0  # Another instance is running

    try:
        _log("FINAL ORCHESTRATOR START")
        _log(f"max_chain_retries={MAX_CHAIN_RETRIES} max_merge_attempts={MAX_MERGE_ATTEMPTS}")

        # Wait for retry_v3.py to exit (it might already be exited if we're a restart)
        _log("Waiting for retry_v3.py to exit...")
        wait_for_exit("retry_v3.py")
        _log("retry_v3.py is not running.")
        time.sleep(60)  # cleanup buffer

        # Chain retry passes
        for i in range(MAX_CHAIN_RETRIES):
            total, per_seed = count_unfinished_nodes()
            _log(f"Pre-chain-pass-{i+1} state: {total} unfinished nodes")
            for seed, n in per_seed.items():
                if n > 0:
                    _log(f"  {seed}: {n} unfinished")
            if total == 0:
                _log("All nodes complete — no further retries needed.")
                break
            run_retry_pass(f"CHAIN-RETRY-{i+1}")
            time.sleep(60)
        else:
            total, per_seed = count_unfinished_nodes()
            _log(f"Reached MAX_CHAIN_RETRIES. Final unfinished: {total}")

        # Final merge with retries
        _log("=" * 60)
        _log("FINAL MERGE PHASE")
        rc, stats = run_merge_with_retry()
        if rc == 0 and stats:
            _log("=" * 60)
            _log("✅ MERGE COMPLETE")
            _log(f"  artifact: {stats['merge_artifact']}")
            _log(f"  size: {stats['size_bytes']:,} bytes")
            _log(f"  groups: {stats['groups']:,}")
            _log(f"  items: {stats['items']:,}")
            _log(f"  operations: {stats['operations']:,}")
            write_run_done(stats, 0)
        else:
            _log(f"❌ MERGE FAILED after {MAX_MERGE_ATTEMPTS} attempts")
            write_run_done(stats, rc)

        _log("FINAL ORCHESTRATOR END")
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"UNCAUGHT EXCEPTION: {e!r}")
        _log(tb)
        write_run_done(None, -100, error=f"{e!r}\n{tb}")
        raise
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
