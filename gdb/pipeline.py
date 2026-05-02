from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click

from . import config
from .store import (
    all_rows,
    compute_batch_fingerprint,
    db,
    dumps,
    json_text,
    loads,
    materialize_batch,
    new_id,
    now,
    read_json,
    scan_and_register,
    set_batch_artifact,
    upsert_batch_by_fingerprint,
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{new_id()}.{path.name}")
    try:
        tmp.write_text(json_text(payload))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def new_run(stage: str, *, seed: str | None = None, label: str | None = None,
            parent_run_id: str | None = None) -> str:
    run_id = new_id()
    with db() as conn:
        conn.execute(
            """INSERT INTO runs (id, stage, seed, parent_run_id, label, attrs, started_at)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, stage, seed, parent_run_id, label, "{}", now()),
        )
        conn.commit()
    return run_id


def close_run(run_id: str, attrs: dict) -> None:
    with db() as conn:
        row = conn.execute("SELECT attrs FROM runs WHERE id=?", (run_id,)).fetchone()
        existing = loads(row["attrs"], default={}) if row else {}
        existing.update(attrs)
        conn.execute("UPDATE runs SET attrs=?, ended_at=? WHERE id=?", (dumps(existing), now(), run_id))
        conn.commit()


def subagent_prompt_for(model: str) -> str:
    if model.startswith("codex-"):
        effort = model.removeprefix("codex-")
        return config.SUBAGENT_PROMPT_CODEX.format(codex_model=config.CODEX_MODEL, effort=effort)
    return config.SUBAGENT_PROMPT_CLAUDE.format(model=model)


def render_prompt(stage: str, variables: dict[str, str]) -> str:
    prompt_path = config.PROMPTS_DIR / f"{stage}.md"
    if not prompt_path.exists():
        raise click.ClickException(f"prompt not found: {prompt_path}")
    text = prompt_path.read_text()
    if "subagent_model" in variables and "subagent_prompt" not in variables:
        variables = {**variables, "subagent_prompt": subagent_prompt_for(variables["subagent_model"])}
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def runtime_env(run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env[config.GDB_STORAGE_ENV] = str(config.STORAGE)
    env[config.GDB_PATH_ENV] = str(config.DB_PATH)
    env[config.GDB_RUN_ID_ENV] = run_id
    return env


def child_pids(pid: int) -> list[int]:
    try:
        result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def kill_descendants(pid: int, sig: signal.Signals) -> None:
    for child in child_pids(pid):
        kill_descendants(child, sig)
        try:
            os.kill(child, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate_pgrp(pid: int) -> None:
    kill_descendants(pid, signal.SIGTERM)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + config.PROCESS_KILL_GRACE_S
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    kill_descendants(pid, signal.SIGKILL)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def parse_stream_json(stream_path: Path) -> dict:
    out: dict[str, Any] = {
        "turns": 0, "cost_usd": 0.0,
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "tool_calls": [], "final_text": None,
    }
    if not stream_path.exists():
        return out
    for line in stream_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("type")
        if kind == "assistant":
            out["turns"] += 1
            for content in (rec.get("message") or {}).get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "tool_use":
                    out["tool_calls"].append(content.get("name") or "tool_use")
                elif content.get("type") == "text":
                    text = content.get("text")
                    if text:
                        out["final_text"] = text
        elif kind == "result":
            cost = rec.get("total_cost_usd")
            if cost is not None:
                out["cost_usd"] = float(cost)
            usage = rec.get("usage") or {}
            for key in ("input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    out[key] = int(value)
    return out


def spawn_claude(run_id: str, prompt: str, *, model: str = config.CLAUDE_MODEL) -> dict:
    if not shutil.which("claude"):
        raise click.ClickException("claude CLI not found; pass --artifact to ingest an existing stage artifact")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / config.RUN_PROMPT_FILE).write_text(prompt)
    stream_path = run_root / config.RUN_STREAM_FILE
    err_path = run_root / config.RUN_STDERR_FILE
    cmd = [
        "claude", "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--disallowedTools", "ScheduleWakeup",
    ]
    started = time.monotonic()
    with stream_path.open("w") as stdout, err_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd, cwd=config.ROOT, env=runtime_env(run_id),
            stdout=stdout, stderr=stderr, text=True, start_new_session=True,
        )
        try:
            rc = proc.wait()
        except (KeyboardInterrupt, SystemExit):
            terminate_pgrp(proc.pid)
            raise
    elapsed = time.monotonic() - started
    stats = parse_stream_json(stream_path)
    tool_calls = stats.pop("tool_calls", [])
    final_text = stats.pop("final_text", None)
    attrs = {
        "runtime": "claude", "model": model, "exit_code": rc, "elapsed_s": elapsed,
        "tool_call_count": len(tool_calls),
        "tool_calls_by_name": {name: tool_calls.count(name) for name in set(tool_calls)},
        **stats,
    }
    close_run(run_id, attrs)
    if final_text:
        (run_root / "final.txt").write_text(final_text)
    return {"run_id": run_id, "exit_code": rc, "elapsed_s": elapsed, "log_dir": str(run_root)}


def spawn_codex(run_id: str, prompt: str, *, effort: str) -> dict:
    if not shutil.which("codex"):
        raise click.ClickException("codex CLI not found; pass --artifact to ingest an existing stage artifact")
    if effort not in config.CODEX_EFFORT_CHOICES:
        raise click.ClickException(f"unknown codex effort {effort!r}")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / config.RUN_PROMPT_FILE).write_text(prompt)
    out_path = run_root / config.RUN_STDOUT_FILE
    err_path = run_root / config.RUN_STDERR_FILE
    cmd = [
        "codex", "exec",
        "-m", config.CODEX_MODEL,
        "-c", f"model_reasoning_effort={effort}",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        prompt,
    ]
    started = time.monotonic()
    with out_path.open("w") as stdout, err_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd, cwd=config.ROOT, env=runtime_env(run_id),
            stdout=stdout, stderr=stderr, text=True, start_new_session=True,
        )
        try:
            rc = proc.wait()
        except (KeyboardInterrupt, SystemExit):
            terminate_pgrp(proc.pid)
            raise
    elapsed = time.monotonic() - started
    close_run(run_id, {"runtime": "codex", "model": f"codex-{effort}",
                       "exit_code": rc, "elapsed_s": elapsed})
    return {"run_id": run_id, "exit_code": rc, "elapsed_s": elapsed, "log_dir": str(run_root)}


def dispatch_spawn(run_id: str, prompt: str, *, model: str) -> dict:
    if model.startswith("codex-"):
        return spawn_codex(run_id, prompt, effort=model.removeprefix("codex-"))
    return spawn_claude(run_id, prompt, model=model)


# ---------------------------------------------------------------------------
# Stage 1 — discover
# ---------------------------------------------------------------------------


def ingest_discovery_artifact(artifact: dict, workspace_dir: Path) -> dict:
    enriched, per_batch_maps = scan_and_register(workspace_dir, artifact)
    maps = {m["batch_idx"]: m["file_map"] for m in per_batch_maps}
    with db() as conn:
        cur = conn.cursor()
        for idx, batch in enumerate(enriched.get("batches") or []):
            source_ids = [s.get("source_id") for s in batch.get("sources") or [] if s.get("source_id")]
            if not source_ids:
                continue
            fingerprint = compute_batch_fingerprint(cur, source_ids)
            batch_id, created = upsert_batch_by_fingerprint(
                cur,
                fingerprint=fingerprint,
                source_ids=source_ids,
                label=batch.get("label"),
                summary=batch.get("summary"),
                file_map=maps.get(idx) or {},
            )
            batch["batch_id"] = batch_id
            batch["created"] = created
        conn.commit()
    return enriched


def run_discover(
    *,
    target: str,
    artifact_path: str | None = None,
    workspace_dir: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
) -> dict:
    run_id = new_run("discover", seed=target, label=f"discover:{target}")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    workspace = Path(workspace_dir).resolve() if workspace_dir else run_root / config.WORKSPACE_SUBDIR
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.DISCOVER_ARTIFACT_FILE
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / config.RUN_INPUT_FILE).write_text(json_text({"target": target, "workspace_dir": str(workspace)}))
    if artifact_path:
        artifact = read_json(artifact_path)
        used_artifact = Path(artifact_path)
    else:
        prompt = render_prompt("discover", {
            "run_id": run_id,
            "target": target,
            "workspace_dir": str(workspace),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0:
            raise click.ClickException(f"discover failed; logs at {spawn['log_dir']}")
        if not artifact_out.exists():
            raise click.ClickException(f"discover wrote no artifact at {artifact_out}")
        artifact = read_json(artifact_out)
        used_artifact = artifact_out
    enriched = ingest_discovery_artifact(artifact, workspace)
    close_run(run_id, {"artifact_path": str(used_artifact), "batch_count": len(enriched.get("batches") or [])})
    return {
        "run_id": run_id,
        "artifact_path": str(used_artifact),
        "batches": [
            {"batch_id": b.get("batch_id"), "created": b.get("created"),
             "source_count": len(b.get("sources") or [])}
            for b in enriched.get("batches") or []
        ],
    }


# ---------------------------------------------------------------------------
# Stage 2 — extract (per batch, name + kind only)
# ---------------------------------------------------------------------------


def commit_names(artifact: dict, *, batch_id: str | None = None,
                 run_id: str | None = None) -> dict:
    """Commit `{type, name}` records from an extract artifact.

    Schema accepted: `{"mentions": [{"type": "model"|"dataset", "name": "..."}, ...]}`.
    Skips entries that are missing either field, have an invalid kind,
    or are exact (kind, name) duplicates of another entry in this artifact.
    No anchors, atoms, identity, links, or descriptions live here.
    """
    if not isinstance(artifact, dict):
        return {"status": "failed", "errors": [{"code": "invalid_artifact"}],
                "names_committed": 0, "names_skipped": 0}
    raw = artifact.get("mentions")
    if not isinstance(raw, list):
        return {"status": "failed", "errors": [{"code": "invalid_artifact"}],
                "names_committed": 0, "names_skipped": 0}

    seen: set[tuple[str, str]] = set()
    skipped: list[dict] = []
    accepted: list[tuple[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            skipped.append({"index": idx, "reason": "not_a_dict"})
            continue
        kind = (item.get("type") or item.get("kind") or "").strip().casefold()
        name = (item.get("name") or "").strip()
        if not name:
            skipped.append({"index": idx, "reason": "empty_name"})
            continue
        if kind not in ("model", "dataset"):
            skipped.append({"index": idx, "reason": "invalid_kind", "name": name, "kind": kind})
            continue
        key = (kind, name)
        if key in seen:
            skipped.append({"index": idx, "reason": "duplicate", "name": name, "kind": kind})
            continue
        seen.add(key)
        accepted.append(key)

    committed = 0
    with db() as conn:
        cur = conn.cursor()
        if batch_id:
            cur.execute("DELETE FROM names WHERE batch_id=?", (batch_id,))
        for kind, name in accepted:
            cur.execute(
                """INSERT INTO names (id, batch_id, run_id, kind, name, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (new_id(), batch_id, run_id, kind, name, now()),
            )
            committed += 1
        if batch_id:
            set_batch_artifact(
                cur,
                batch_id=batch_id,
                stage="extract",
                artifact_path=str(Path(artifact.get("_artifact_path", "")).resolve())
                              if artifact.get("_artifact_path") else "",
                status="complete",
                attrs={"names_committed": committed, "names_skipped": len(skipped)},
            )
        conn.commit()
    return {
        "status": "complete",
        "names_committed": committed,
        "names_skipped": len(skipped),
        "skipped": skipped[:50],
    }


def run_extract(
    *,
    batch_id: str | None = None,
    artifact_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
) -> dict:
    if artifact_path:
        artifact = read_json(artifact_path)
        artifact["_artifact_path"] = str(artifact_path)
        return commit_names(artifact, batch_id=batch_id)
    batch_ids = [batch_id] if batch_id else [
        row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")
    ]
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def extract_one(bid: str) -> dict:
        run_id = new_run("extract", label=f"extract:{bid[:8]}")
        run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
        batch_dir = materialize_batch(bid, run_root / config.BATCH_SUBDIR)
        artifact_out = run_root / config.EXTRACT_ARTIFACT_FILE
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / config.RUN_INPUT_FILE).write_text(
            json_text({"batch_id": bid, "batch_dir": str(batch_dir)})
        )
        prompt = render_prompt("extract", {
            "run_id": run_id,
            "batch_id": bid,
            "batch_dir": str(batch_dir),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        artifact = read_json(artifact_out)
        artifact["_artifact_path"] = str(artifact_out)
        result = commit_names(artifact, batch_id=bid, run_id=run_id)
        result["batch_id"] = bid
        result["run_id"] = run_id
        return result

    results: list[dict] = []
    if workers == 1:
        for bid in batch_ids:
            results.append(extract_one(bid))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: str(r.get("batch_id") or ""))
    failed = [r for r in results if r.get("status") != "complete"]
    return {"results": results, "failed": len(failed), "parallel_workers": workers}


# ---------------------------------------------------------------------------
# Stage 3 — organize (one planner reads names file, emits lattice)
# ---------------------------------------------------------------------------


def names_packet() -> dict:
    """The deduped `{type, name}` list the organize planner reads.

    Counts are intentionally absent — they don't change how the planner
    decides whether two surfaces refer to the same entity.
    """
    rows = all_rows(
        "SELECT DISTINCT kind, name FROM names ORDER BY kind, name"
    )
    return {"names": [{"type": r["kind"], "name": r["name"]} for r in rows]}


def _validate_organize_artifact(artifact: dict) -> tuple[int, int]:
    """Sanity-check the organize artifact's groups+items shape and
    return (group_count, item_count). Raises if shape is wrong."""
    groups = artifact.get("groups") if isinstance(artifact, dict) else None
    if not isinstance(groups, list):
        raise click.ClickException("organize artifact missing groups[]")
    item_count = 0
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise click.ClickException(f"groups[{i}] is not a dict")
        items = group.get("items")
        if not isinstance(items, list):
            raise click.ClickException(f"groups[{i}].items is not a list")
        item_count += len(items)
    return len(groups), item_count


def run_organize(
    *,
    artifact_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """Single planner reads the consolidated names file, groups by
    family, collapses surface variants, picks a canonical formal_name
    and structured identity per item, and writes one record per real
    artifact.

    With `--artifact`, ingest an externally produced organize artifact
    instead of spawning a planner. The artifact lives on disk; we
    record its location in the run row and stop.
    """
    if artifact_path:
        run_id = new_run("organize", label="organize:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        group_count, item_count = _validate_organize_artifact(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count}

    run_id = new_run("organize", label="organize")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    names_path = run_root / config.ORGANIZE_NAMES_FILE
    artifact_out = run_root / config.ORGANIZE_ARTIFACT_FILE
    atomic_write_json(names_path, names_packet())
    prompt = render_prompt("organize", {
        "run_id": run_id,
        "names_path": str(names_path),
        "input_path": str(names_path),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model)
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"organize failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    group_count, item_count = _validate_organize_artifact(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "group_count": group_count,
        "item_count": item_count,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count}
