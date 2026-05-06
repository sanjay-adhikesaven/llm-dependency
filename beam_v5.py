#!/usr/bin/env python3
"""Beam-v5: per-seed bounded-beam BFS to DEPTH_TARGET=4.

For each seed in SEEDS:
  - Build the seed-lineage tree from existing artifacts on disk
    (BFS from seed root through every completed triage_artifact's
    auto_expand list).
  - At each depth level d in 2..DEPTH_TARGET:
      - Find candidate set: nodes appearing in auto_expand[] of any
        seed-lineage node at depth d-1, NOT yet attempted globally.
      - Rank by per-seed parent-suggestion count.
      - Pick top K_PER_LEVEL.
  - Expand top-K nodes in parallel (WORKER_CAP).

Runs alongside recurse_v4.py (Path D). Reads attempted_subjects fresh
at each phase so we never duplicate Path D's work.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path("/Users/sanjayadhikesaven/Downloads/graph")
WORKER_ROOT = REPO_ROOT / "storage" / "workers"
SEED_STORAGE_ROOT = REPO_ROOT / "storage" / "seeds"
LOG_DIR = REPO_ROOT / "run-logs"

WORKER_CAP = 8
PLANNER = "opus"
SUBAGENT = "opus"
GDB_MAX_PARALLEL = "16"

K_PER_LEVEL = 5
DEPTH_TARGET = 4

# Hardcoded bridge nodes — picked because each one is expected to unlock
# at least one paper-quality 3+ hop chain (license / inconsistency /
# missed-dep). Expanded as a "phase 0" before the depth-BFS phases
# so their triages inform deeper picks.
BRIDGE_NODES = [
    # Closed-data laundering chains (license risk via Qwen/DeepSeek/Phi)
    "nvidia/OpenScienceReasoning-2",
    "nvidia/Nemotron-CC-v2",
    "nvidia/Nemotron-CC-Math-v1",
    "nvidia/AceReason-Math",
    "nvidia/Nemotron-Cascade-SFT-Stage-1",
    # SwallowMath/CraneMath license example
    "tokyotech-llm/swallow-math",
    "tokyotech-llm/swallow-code",
    # DR-Tulu missed-dep chain (Claude Sonnet via ScholarQA)
    "facebook/natural_reasoning",
    # OLMo Dolci-* family (now expandable after validator patch)
    "allenai/Dolci-Instruct-SFT",
    "allenai/Dolci-Think-SFT-Olmo-Hybrid",
    # SmolLM3 lineage hubs
    "HuggingFaceTB/smoltalk",
    "allenai/dolma",
]

SEEDS = [
    "allenai/OLMo-3-32B-Think",
    "rl-research/DR-Tulu-8B",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "HuggingFaceTB/SmolLM3-3B",
]
# DR-Tulu already at depth=4; skip
START_DEPTH_BY_SEED = {
    "allenai/OLMo-3-32B-Think": 2,
    "rl-research/DR-Tulu-8B": 99,  # skip — already at 4
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 2,
    "HuggingFaceTB/SmolLM3-3B": 3,
}

SUMMARY_LOG = LOG_DIR / "BEAM_V5_SUMMARY.log"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    with open(SUMMARY_LOG, "a") as f:
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
    for ws in worker_storage.glob("runs/*/workspace"):
        if ws.is_dir():
            shutil.rmtree(ws, ignore_errors=True)
    for sj in worker_storage.glob("runs/*/stream.jsonl"):
        try:
            sj.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# State reading: build subject→triage map and attempted-subject set
# ---------------------------------------------------------------------------

# Hard-blocked: Path D's planned 10 ghosts. Some may be in-flight,
# others queued but not yet started. Either way, beam_v5 must skip
# them so we don't duplicate expansion.
PATH_D_GHOSTS = {
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
}


SEED_LOG_TO_SEED = {
    "seed_OLMo_3.log": "allenai/OLMo-3-32B-Think",
    "seed_rl-research_DR-Tulu-8B.log": "rl-research/DR-Tulu-8B",
    "seed_nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4.log":
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "seed_HuggingFaceTB_SmolLM3-3B.log": "HuggingFaceTB/SmolLM3-3B",
}


def build_state() -> tuple[dict[str, Path], set[str]]:
    """Returns (subject→triage_path, attempted_subjects).

    The organize_artifact schema does NOT carry a subject field. We
    derive subject from the seed_log markers (which contain
    `node='X'`) and worker_dir naming convention.
    """
    subject_to_triage: dict[str, Path] = {}
    attempted: set[str] = set()

    # Per-worker mapping comes from seed_log markers.
    rx_started = re.compile(r"expand depth=\d+ worker=(\S+) node=(['\"])(.+?)\2")
    rx_v4 = re.compile(r"\[v4-recurse\]\s*expand worker=(\S+) node=(['\"])(.+?)\2")
    rx_v5 = re.compile(r"\[v5-beam[^\]]*\]\s*worker=(\S+) node=(['\"])(.+?)\2")

    for log_name in SEED_LOG_TO_SEED.keys():
        sl = LOG_DIR / log_name
        if not sl.exists():
            continue
        for line in sl.read_text(errors="replace").splitlines():
            m = rx_started.search(line)
            if not m:
                continue
            wid, _q, node = m.group(1), m.group(2), m.group(3)
            attempted.add(node)
            ws = WORKER_ROOT / wid
            if ws.exists():
                tris = list(ws.glob("runs/*/triage_artifact.json"))
                if tris:
                    subject_to_triage[node] = max(tris, key=lambda p: p.stat().st_mtime)

    # Seeds themselves
    for log_name, seed_name in SEED_LOG_TO_SEED.items():
        attempted.add(seed_name)
        seed_dir_name = log_name.replace("seed_", "").replace(".log", "")
        sd = SEED_STORAGE_ROOT / seed_dir_name
        if sd.exists():
            tris = list(sd.glob("runs/*/triage_artifact.json"))
            if tris:
                subject_to_triage[seed_name] = max(tris, key=lambda p: p.stat().st_mtime)

    # v4-recurse (Path D)
    rec = LOG_DIR / "RECURSE_V4_SUMMARY.log"
    if rec.exists():
        for line in rec.read_text(errors="replace").splitlines():
            m = rx_v4.search(line)
            if not m:
                continue
            wid, _q, node = m.group(1), m.group(2), m.group(3)
            attempted.add(node)
            ws = WORKER_ROOT / wid
            if ws.exists():
                tris = list(ws.glob("runs/*/triage_artifact.json"))
                if tris:
                    subject_to_triage[node] = max(tris, key=lambda p: p.stat().st_mtime)

    # v5-beam (this run)
    if SUMMARY_LOG.exists():
        for line in SUMMARY_LOG.read_text(errors="replace").splitlines():
            m = rx_v5.search(line)
            if not m:
                continue
            wid, _q, node = m.group(1), m.group(2), m.group(3)
            attempted.add(node)
            ws = WORKER_ROOT / wid
            if ws.exists():
                tris = list(ws.glob("runs/*/triage_artifact.json"))
                if tris:
                    subject_to_triage[node] = max(tris, key=lambda p: p.stat().st_mtime)

    # Exclude Path D's planned ghosts so beam_v5 doesn't duplicate them
    attempted |= PATH_D_GHOSTS

    return subject_to_triage, attempted


# ---------------------------------------------------------------------------
# Per-seed lineage / frontier
# ---------------------------------------------------------------------------

def lineage_for_seed(seed: str, max_depth: int,
                     subject_to_triage: dict[str, Path]
                     ) -> dict[int, set[str]]:
    """{depth: nodes-in-lineage}. depth=0 is just {seed}."""
    by_depth: dict[int, set[str]] = {0: {seed}}
    for d in range(1, max_depth + 1):
        children: set[str] = set()
        for parent in by_depth[d - 1]:
            tri_path = subject_to_triage.get(parent)
            if not tri_path:
                continue
            try:
                tri = json.loads(tri_path.read_text())
            except Exception:
                continue
            for entry in tri.get("auto_expand", []):
                fn = entry.get("formal_name")
                if fn:
                    children.add(fn)
        by_depth[d] = children
    return by_depth


def pick_topK_for_seed(seed: str, target_depth: int,
                       subject_to_triage: dict[str, Path],
                       attempted: set[str], k: int) -> list[str]:
    """Pick top-k unexpanded candidates at target_depth for seed."""
    lineage = lineage_for_seed(seed, target_depth - 1, subject_to_triage)
    parents = lineage.get(target_depth - 1, set())
    counts: Counter = Counter()
    for parent in parents:
        tri_path = subject_to_triage.get(parent)
        if not tri_path:
            continue
        try:
            tri = json.loads(tri_path.read_text())
        except Exception:
            continue
        for entry in tri.get("auto_expand", []):
            fn = entry.get("formal_name")
            if fn and fn not in attempted:
                counts[fn] += 1
    return [fn for fn, _ in counts.most_common(k)]


# ---------------------------------------------------------------------------
# Node expansion (init+expand+triage)
# ---------------------------------------------------------------------------

def expand_node(node: str, depth: int, seed: str) -> tuple[str, str]:
    worker_id = str(uuid.uuid4())[:8]
    worker_storage = WORKER_ROOT / worker_id
    worker_log = LOG_DIR / f"worker_{worker_id}.log"

    _log(f"  >> [v5-beam d={depth} seed={seed.split('/')[-1]}] worker={worker_id} node={node!r}")
    env = {"GDB_STORAGE": str(worker_storage)}

    rc = _run_cmd(["gdb", "init", "--fresh", "--yes", "--I-mean-it"],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v5-beam] ERROR init failed; node={node}")
        return (node, "init-failed")

    rc = _run_cmd(["gdb", "run", "expand", "--node", node, "--skip", "audit",
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v5-beam] ERROR expand failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "expand-failed")

    organize_artifact = _find_artifact(worker_storage, "organize_artifact.json")
    if not organize_artifact:
        _log(f"     [v5-beam] ERROR no organize artifact; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "no-organize")

    rc = _run_cmd(["gdb", "run", "triage",
                   "--lattice", str(organize_artifact),
                   "--planner-model", PLANNER, "--subagent-model", SUBAGENT],
                  worker_log, env_overrides=env)
    if rc != 0:
        _log(f"     [v5-beam] WARN triage failed; node={node}")
        _cleanup_worker_disk(worker_storage)
        return (node, "triage-failed")

    triage_artifact = _find_artifact(worker_storage, "triage_artifact.json")
    if not triage_artifact:
        _cleanup_worker_disk(worker_storage)
        return (node, "no-triage")

    _log(f"     [v5-beam] << complete; node={node} worker={worker_id}")
    _cleanup_worker_disk(worker_storage)
    return (node, "complete")


# ---------------------------------------------------------------------------
# Main loop: depth-by-depth
# ---------------------------------------------------------------------------

def main() -> None:
    SUMMARY_LOG.write_text("")
    _log("BEAM-V5 START")
    _log(f"K_PER_LEVEL={K_PER_LEVEL}  DEPTH_TARGET={DEPTH_TARGET}  WORKER_CAP={WORKER_CAP}")
    _log(f"planner={PLANNER}  subagent={SUBAGENT}")

    # ===== PHASE 0: BRIDGE_NODES =====
    # Expand hand-picked nodes that unlock 3+ hop chains regardless of
    # parent-suggestion count. Skip a bridge only if it already has a
    # triage_artifact on disk (i.e. previously COMPLETED — not just
    # attempted-and-failed, since the validator patch may now let
    # previously-failing nodes succeed).
    _log("\n=== PHASE 0: BRIDGE_NODES ===")
    s2t_now, _ = build_state()
    bridges_to_run = [n for n in BRIDGE_NODES if n not in s2t_now]
    skipped = [n for n in BRIDGE_NODES if n in s2t_now]
    for n in skipped:
        _log(f"  skip bridge (has triage already): {n}")
    _log(f"PHASE 0: {len(bridges_to_run)} bridge nodes to expand")

    if bridges_to_run:
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=WORKER_CAP) as executor:
            futures: list[Future] = [
                executor.submit(expand_node, n, 0, "BRIDGE")
                for n in bridges_to_run
            ]
            for fut in futures:
                try:
                    node, status = fut.result()
                    results[node] = status
                except Exception as e:
                    _log(f"worker error: {e!r}")
        ok = sum(1 for s in results.values() if s == "complete")
        _log(f"PHASE 0 DONE: {ok}/{len(bridges_to_run)} bridges complete")
        for n, s in results.items():
            _log(f"    {n} :: {s}")

    for depth in range(2, DEPTH_TARGET + 1):
        _log(f"\n=== PHASE depth={depth} ===")
        # Refresh state at the START of every phase — picks up any nodes
        # that finished in Path D or in beam_v5's previous phases.
        subject_to_triage, attempted = build_state()
        _log(f"State: {len(attempted)} attempted subjects, "
             f"{len(subject_to_triage)} subjects with triage artifacts")

        per_seed_picks: list[tuple[str, str]] = []  # (seed, node)
        for seed in SEEDS:
            if depth < START_DEPTH_BY_SEED[seed]:
                _log(f"  seed {seed.split('/')[-1]} skip d={depth} (start_depth={START_DEPTH_BY_SEED[seed]})")
                continue
            picks = pick_topK_for_seed(seed, depth, subject_to_triage,
                                        attempted, K_PER_LEVEL)
            _log(f"  seed {seed.split('/')[-1]} d={depth} top-{K_PER_LEVEL}: {picks}")
            for n in picks:
                per_seed_picks.append((seed, n))

        if not per_seed_picks:
            _log(f"PHASE depth={depth} — no candidates picked, advancing")
            continue

        _log(f"PHASE depth={depth}: {len(per_seed_picks)} nodes to expand")

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=WORKER_CAP) as executor:
            futures: list[Future] = [
                executor.submit(expand_node, node, depth, seed)
                for seed, node in per_seed_picks
            ]
            for fut in futures:
                try:
                    node, status = fut.result()
                    results[node] = status
                except Exception as e:
                    _log(f"worker error: {e!r}")

        ok = sum(1 for s in results.values() if s == "complete")
        _log(f"PHASE depth={depth} DONE: {ok}/{len(per_seed_picks)} complete")
        for n, s in results.items():
            _log(f"    {n} :: {s}")

    _log("=== BEAM-V5 COMPLETE ===")


if __name__ == "__main__":
    main()
