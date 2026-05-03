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


# ---------------------------------------------------------------------------
# Stage 4 — audit (revise the lattice in place; same shape in, same out)
# ---------------------------------------------------------------------------


def _latest_lattice_artifact_path() -> Path:
    """Return the path of the most recent groups+items artifact.

    Searches both `organize` and `audit` runs since audit emits the
    same shape and is the authoritative successor when present.
    """
    rows = all_rows(
        "SELECT id, stage, attrs FROM runs "
        "WHERE stage IN ('organize','audit') AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1"
    )
    if not rows:
        raise click.ClickException(
            "no organize or audit run found; run `gdb run organize` first"
        )
    attrs = loads(rows[0]["attrs"], default={}) or {}
    path = attrs.get("artifact_path")
    if not path or not Path(path).exists():
        raise click.ClickException(
            f"{rows[0]['stage']} artifact missing on disk for run {rows[0]['id']}"
        )
    return Path(path)


def run_audit(
    *,
    artifact_path: str | None = None,
    source_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """Read the latest lattice artifact, revise it, write the result.

    Audit's output schema matches organize's (`groups[].items[]`). The
    agent makes edits directly — splits, merges, formal_name fixes,
    identity_key adjustments — and emits the whole revised lattice.

    With `--artifact`, ingest an externally produced audit artifact
    instead of spawning a planner. With `--source`, audit a specific
    artifact (organize or prior audit) instead of the most recent one.
    """
    def _short_notes(art: dict) -> str | None:
        n = art.get("notes")
        return n[:500] if isinstance(n, str) else None

    if artifact_path:
        run_id = new_run("audit", label="audit:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        group_count, item_count = _validate_organize_artifact(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "notes": _short_notes(artifact),
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count}

    source_artifact_path = (
        Path(source_path).resolve() if source_path
        else _latest_lattice_artifact_path()
    )

    run_id = new_run("audit", label="audit", seed=str(source_artifact_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.AUDIT_ARTIFACT_FILE
    prompt = render_prompt("audit", {
        "run_id": run_id,
        "organize_path": str(source_artifact_path),
        "input_path": str(source_artifact_path),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model)
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"audit failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    group_count, item_count = _validate_organize_artifact(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "source_artifact_path": str(source_artifact_path),
        "group_count": group_count,
        "item_count": item_count,
        "notes": _short_notes(artifact),
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count}


# ---------------------------------------------------------------------------
# Stage 5 — linker (attach official URLs to every item)
# ---------------------------------------------------------------------------


def _latest_lattice_or_audit_or_linker_path() -> Path:
    """Return the most recent groups+items artifact across organize,
    audit, OR linker — linker is idempotent over its own output, so a
    re-run picks up the previous linker's output by default."""
    rows = all_rows(
        "SELECT id, stage, attrs FROM runs "
        "WHERE stage IN ('organize','audit','linker') AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC LIMIT 1"
    )
    if not rows:
        raise click.ClickException(
            "no organize / audit / linker run found; run `gdb run organize` first"
        )
    attrs = loads(rows[0]["attrs"], default={}) or {}
    path = attrs.get("artifact_path")
    if not path or not Path(path).exists():
        raise click.ClickException(
            f"{rows[0]['stage']} artifact missing on disk for run {rows[0]['id']}"
        )
    return Path(path)


def _count_link_stats(artifact: dict) -> dict:
    """Tally link / description / kind-correction stats across the
    artifact: total items, items with >=1 link, items with a
    description, items where linker re-typed kind, total links,
    link-kind histogram."""
    total_items = 0
    items_with_links = 0
    items_with_description = 0
    n_models = 0
    n_datasets = 0
    total_links = 0
    by_kind: dict[str, int] = {}
    for group in artifact.get("groups") or []:
        for item in group.get("items") or []:
            total_items += 1
            links = item.get("links") or []
            if links:
                items_with_links += 1
            if isinstance(item.get("description"), str) and item.get("description").strip():
                items_with_description += 1
            kind = item.get("kind")
            if kind == "model":
                n_models += 1
            elif kind == "dataset":
                n_datasets += 1
            for link in links:
                if not isinstance(link, dict):
                    continue
                total_links += 1
                k = link.get("kind") or "unknown"
                by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "total_items": total_items,
        "items_with_links": items_with_links,
        "items_without_links": total_items - items_with_links,
        "items_with_description": items_with_description,
        "items_without_description": total_items - items_with_description,
        "n_models": n_models,
        "n_datasets": n_datasets,
        "total_links": total_links,
        "links_by_kind": by_kind,
    }


def run_linker(
    *,
    artifact_path: str | None = None,
    source_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """Attach official URLs to every item in the latest lattice.

    Output schema = input schema + a `links: [{kind, url}]` array per
    item. Reuses `_validate_organize_artifact` since the structural
    shape is unchanged.

    With `--artifact`, ingest an externally produced linker artifact.
    With `--source`, link a specific lattice artifact instead of the
    most recent one.
    """
    if artifact_path:
        run_id = new_run("linker", label="linker:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        group_count, item_count = _validate_organize_artifact(artifact)
        link_stats = _count_link_stats(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            **link_stats,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                **link_stats}

    source_artifact_path = (
        Path(source_path).resolve() if source_path
        else _latest_lattice_or_audit_or_linker_path()
    )

    run_id = new_run("linker", label="linker", seed=str(source_artifact_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.LINKER_ARTIFACT_FILE
    prompt = render_prompt("linker", {
        "run_id": run_id,
        "lattice_path": str(source_artifact_path),
        "input_path": str(source_artifact_path),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model)
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"linker failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    group_count, item_count = _validate_organize_artifact(artifact)
    link_stats = _count_link_stats(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "source_artifact_path": str(source_artifact_path),
        "group_count": group_count,
        "item_count": item_count,
        **link_stats,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count,
            **link_stats}


# ---------------------------------------------------------------------------
# Stage 6 — relate (per batch, lattice-anchored typed edges)
# ---------------------------------------------------------------------------


_DIRECT_RELATIONS = (
    "trained_on", "initialized_from", "distilled_from",
    "transformed_by", "filtered_by",
)
_INDIRECT_RELATIONS = (
    "inspired_by", "used_for_ablation", "used_for_evaluation",
)
_STRUCTURAL_RELATIONS = (
    "subset_of", "supersedes", "released_with", "contains",
    # numeric-fact relations carried as STRUCTURAL edges
    "size", "training_tokens", "context_length", "release_date",
    "parameter_count", "composition_count",
)
# Canonical labels — guidance for the relate prompt and tracking.
# `relation` is OPEN: the planner may coin a new snake_case label when
# none of these canonical values fits the source's described event.
CANONICAL_RELATION_VALUES = (
    *_DIRECT_RELATIONS, *_INDIRECT_RELATIONS, *_STRUCTURAL_RELATIONS,
)
# Legacy alias kept so external tooling still resolves the symbol;
# treated as the canonical set, not a closed enum.
RELATION_VALUES = CANONICAL_RELATION_VALUES
DIRECTION_VALUES = ("DIRECT", "INDIRECT", "STRUCTURAL")
# Canonical provenance-kind labels — also OPEN; planner may coin new
# source classes (e.g., `notebook_cell`, `wandb_log`) when warranted.
CANONICAL_PROVENANCE_KINDS = (
    "paper_prose", "paper_table", "paper_figure",
    "hf_frontmatter", "hf_card_body",
    "script_flag", "code_constant", "code_comment",
    "config_yaml", "markdown_doc",
)


def _is_snake_case_label(value: object) -> bool:
    """Lightweight shape check for coined labels. Avoids accepting empty
    strings, whitespace, or sentence fragments (e.g.
    'training data filter')."""
    if not isinstance(value, str) or not value.strip():
        return False
    s = value.strip()
    if not s.replace("_", "").replace("-", "").isalnum():
        return False
    if len(s) > 64:
        return False
    return True


def _validate_relate_artifact(artifact: dict, *,
                              lattice_formal_names: set[str] | None = None
                              ) -> dict:
    """Sanity-check a relate artifact's shape. Returns a stats dict:
    `{operation_count, relation_count, off_lattice_object_count,
       coined_relations: {label: count}, coined_provenance_kinds: {label: count}}`.
    Raises on structural errors only.

    Open-vocabulary fields:
      - `relation`: must be a non-empty snake_case label. Values
        outside `CANONICAL_RELATION_VALUES` are allowed but counted
        as coined.
      - `provenance_kind`: must be a non-empty snake_case label.
        Values outside `CANONICAL_PROVENANCE_KINDS` are counted as
        coined.

    Closed-vocabulary fields (still enforced):
      - `direction` ∈ {DIRECT, INDIRECT, STRUCTURAL}
      - `subject_in_lattice` must be true
      - `subject` must be a lattice formal_name when the lattice is
        provided.
      - `operation_id` must reference an existing operation when set.
    """
    if not isinstance(artifact, dict):
        raise click.ClickException("relate artifact is not a dict")
    operations = artifact.get("operations")
    relations = artifact.get("relations")
    if not isinstance(operations, list):
        raise click.ClickException("relate artifact missing operations[]")
    if not isinstance(relations, list):
        raise click.ClickException("relate artifact missing relations[]")

    op_ids: set[str] = set()
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise click.ClickException(f"operations[{i}] is not a dict")
        op_id = op.get("id")
        if not isinstance(op_id, str) or not op_id.strip():
            raise click.ClickException(f"operations[{i}].id is missing")
        if op_id in op_ids:
            raise click.ClickException(f"operations[{i}].id {op_id!r} is duplicated")
        op_ids.add(op_id)
        desc = op.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise click.ClickException(
                f"operations[{i}].description is missing or empty"
            )

    off_lattice = 0
    coined_relations: dict[str, int] = {}
    coined_provenance: dict[str, int] = {}
    canonical_relations = set(CANONICAL_RELATION_VALUES)
    canonical_provenance = set(CANONICAL_PROVENANCE_KINDS)

    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            raise click.ClickException(f"relations[{i}] is not a dict")
        subject = rel.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise click.ClickException(f"relations[{i}].subject is missing")
        if rel.get("subject_in_lattice") is not True:
            raise click.ClickException(
                f"relations[{i}].subject_in_lattice must be true"
            )
        if lattice_formal_names is not None and subject not in lattice_formal_names:
            raise click.ClickException(
                f"relations[{i}].subject {subject!r} is not a lattice formal_name"
            )
        relation = rel.get("relation")
        if not _is_snake_case_label(relation):
            raise click.ClickException(
                f"relations[{i}].relation {relation!r} is not a valid label "
                f"(non-empty snake_case string ≤64 chars)"
            )
        if relation not in canonical_relations:
            coined_relations[relation] = coined_relations.get(relation, 0) + 1
        direction = rel.get("direction")
        if direction not in DIRECTION_VALUES:
            raise click.ClickException(
                f"relations[{i}].direction {direction!r} not in {DIRECTION_VALUES}"
            )
        provenance = rel.get("provenance_kind")
        if not _is_snake_case_label(provenance):
            raise click.ClickException(
                f"relations[{i}].provenance_kind {provenance!r} is not a "
                f"valid label (non-empty snake_case string ≤64 chars)"
            )
        if provenance not in canonical_provenance:
            coined_provenance[provenance] = coined_provenance.get(provenance, 0) + 1
        op_id = rel.get("operation_id")
        if op_id is not None:
            if not isinstance(op_id, str):
                raise click.ClickException(
                    f"relations[{i}].operation_id must be a string or null"
                )
            if op_id not in op_ids:
                raise click.ClickException(
                    f"relations[{i}].operation_id {op_id!r} not found in operations[]"
                )
        if not rel.get("object_in_lattice"):
            off_lattice += 1
    return {
        "operation_count": len(operations),
        "relation_count": len(relations),
        "off_lattice_object_count": off_lattice,
        "coined_relations": coined_relations,
        "coined_provenance_kinds": coined_provenance,
    }


def _lattice_formal_names(lattice_artifact: dict) -> set[str]:
    names: set[str] = set()
    for group in lattice_artifact.get("groups") or []:
        for item in group.get("items") or []:
            formal = item.get("formal_name")
            if isinstance(formal, str) and formal:
                names.add(formal)
    return names


def _filter_lattice_to_linked(lattice_artifact: dict) -> dict:
    """Drop items with empty `links` and any group that ends up empty.

    Used as the relate-stage input so the planner only anchors edges
    on items the linker confirmed are publicly resolvable. Items without
    a verified link are not safe subjects for closed-vocabulary edges —
    they may be private / gated / phantom names that the lattice should
    not propagate downstream.
    """
    out_groups: list[dict] = []
    for group in lattice_artifact.get("groups") or []:
        kept_items = [
            item for item in (group.get("items") or [])
            if isinstance(item, dict) and (item.get("links") or [])
        ]
        if not kept_items:
            continue
        out_groups.append({**group, "items": kept_items})
    out: dict = {"groups": out_groups}
    if "notes" in lattice_artifact:
        out["notes"] = lattice_artifact["notes"]
    return out


def commit_relations_artifact(
    artifact: dict, *,
    batch_id: str | None = None,
    run_id: str | None = None,
    artifact_path: Path | None = None,
    lattice_formal_names: set[str] | None = None,
) -> dict:
    """Validate a relate artifact and record it as a per-batch
    artifact. No DB rows for individual operations or relations —
    the JSON file on disk is the data, the run + batch_artifact
    rows index it.

    Returns a dict including coined-vocabulary tallies so operators
    can see what new relation / provenance labels the planner
    introduced this batch.
    """
    stats = _validate_relate_artifact(
        artifact, lattice_formal_names=lattice_formal_names
    )
    if batch_id and artifact_path:
        with db() as conn:
            cur = conn.cursor()
            set_batch_artifact(
                cur,
                batch_id=batch_id,
                stage="relate",
                artifact_path=str(artifact_path.resolve()),
                status="complete",
                run_id=run_id,
                attrs=stats,
            )
            conn.commit()
    return {"status": "complete", **stats}


def run_relate(
    *,
    batch_id: str | None = None,
    artifact_path: str | None = None,
    lattice_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
) -> dict:
    """Per-batch parallel: spawn one Claude planner per batch to
    extract typed lattice-anchored edges. Subjects must be lattice
    `formal_name`s; the closed 8-bucket relation taxonomy is enforced
    on ingest by `_validate_relate_artifact`.

    With `--artifact`, ingest an externally produced relate artifact
    instead of spawning a planner.
    """
    if artifact_path:
        if not batch_id:
            raise click.ClickException("--batch-id is required with --artifact")
        artifact = read_json(artifact_path)
        # When ingesting standalone, we don't have the lattice in hand,
        # so subject formal-name validation is shape-only.
        result = commit_relations_artifact(
            artifact,
            batch_id=batch_id,
            artifact_path=Path(artifact_path),
        )
        return result

    source_lattice_path = (
        Path(lattice_path).resolve() if lattice_path
        else _latest_lattice_or_audit_or_linker_path()
    )
    lattice_artifact = read_json(str(source_lattice_path))
    # Pass the full lattice to the relate planner. Items without a
    # verified link can still be valid edge endpoints (e.g., gated HF
    # repos, API-only OpenAI judges with no canonical vendor_docs page,
    # internal AI2 names referenced in source). Filtering them out
    # systematically removes filtered_by / distilled_from edges to API
    # judges. The relate prompt's off-lattice channel handles entities
    # not in the lattice; the planner should not be deprived of
    # extracted-but-unlinkable entities.
    formal_names = _lattice_formal_names(lattice_artifact)
    n_total = sum(len(g.get("items") or []) for g in lattice_artifact.get("groups") or [])
    n_linked = sum(
        1 for g in lattice_artifact.get("groups") or []
        for it in g.get("items") or [] if (it.get("links") or [])
    )

    batch_ids = [batch_id] if batch_id else [
        row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")
    ]
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def relate_one(bid: str) -> dict:
        run_id = new_run("relate", label=f"relate:{bid[:8]}",
                         seed=str(source_lattice_path))
        run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
        batch_dir = materialize_batch(bid, run_root / config.BATCH_SUBDIR)
        artifact_out = run_root / config.RELATE_ARTIFACT_FILE
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / config.RUN_INPUT_FILE).write_text(
            json_text({"batch_id": bid, "batch_dir": str(batch_dir),
                       "lattice_path": str(source_lattice_path)})
        )
        prompt = render_prompt("relate", {
            "run_id": run_id,
            "batch_id": bid,
            "batch_dir": str(batch_dir),
            "lattice_path": str(source_lattice_path),
            "worker_dir": str(run_root / config.WORKERS_SUBDIR),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        artifact = read_json(str(artifact_out))
        try:
            result = commit_relations_artifact(
                artifact,
                batch_id=bid,
                run_id=run_id,
                artifact_path=artifact_out,
                lattice_formal_names=formal_names,
            )
        except click.ClickException as exc:
            return {"batch_id": bid, "status": "failed",
                    "log_dir": spawn["log_dir"], "error": str(exc)}
        result["batch_id"] = bid
        result["run_id"] = run_id
        result["artifact_path"] = str(artifact_out)
        return result

    results: list[dict] = []
    if workers == 1:
        for bid in batch_ids:
            results.append(relate_one(bid))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(relate_one, bid): bid for bid in batch_ids}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: str(r.get("batch_id") or ""))
    failed = [r for r in results if r.get("status") != "complete"]
    return {"results": results, "failed": len(failed),
            "lattice_path": str(source_lattice_path),
            "lattice_total_items": n_total,
            "lattice_linked_items": n_linked,
            "parallel_workers": workers}


# ---------------------------------------------------------------------------
# Stage 7 — triage (one planner classifies upstream nodes)
# ---------------------------------------------------------------------------


_TRIAGE_BUCKETS = ("auto_expand", "decline", "manual")


def _validate_triage_artifact(artifact: dict) -> dict[str, int]:
    """Validate triage artifact shape; return per-bucket counts."""
    if not isinstance(artifact, dict):
        raise click.ClickException("triage artifact is not a dict")
    counts: dict[str, int] = {}
    for bucket in _TRIAGE_BUCKETS:
        items = artifact.get(bucket)
        if not isinstance(items, list):
            raise click.ClickException(
                f"triage artifact missing {bucket!r} list"
            )
        for i, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise click.ClickException(
                    f"triage.{bucket}[{i}] is not a dict"
                )
            for required in ("formal_name", "rationale"):
                if not entry.get(required):
                    raise click.ClickException(
                        f"triage.{bucket}[{i}] missing {required!r}"
                    )
        counts[bucket] = len(items)
    return counts


def _aggregate_relations_artifact(out_path: Path) -> Path:
    """Concatenate every batch's relate artifact into one file the
    triage planner reads. The merge is straightforward: the union of
    `relations[]` arrays plus a `batch_ids` field tracking origin.
    """
    rows = all_rows(
        "SELECT batch_id, artifact_path FROM batch_artifacts "
        "WHERE stage='relate' AND status='complete'"
    )
    if not rows:
        raise click.ClickException(
            "no relate artifacts found; run `gdb run relate` first"
        )
    merged: list[dict] = []
    batch_ids: list[str] = []
    for row in rows:
        path = Path(row["artifact_path"])
        if not path.exists():
            continue
        artifact = read_json(str(path))
        relations = artifact.get("relations") or []
        for rel in relations:
            if isinstance(rel, dict):
                merged.append({**rel, "_batch_id": row["batch_id"]})
        batch_ids.append(row["batch_id"])
    atomic_write_json(out_path, {
        "batch_ids": batch_ids,
        "relations": merged,
    })
    return out_path


def run_triage(
    *,
    artifact_path: str | None = None,
    lattice_path: str | None = None,
    relations_path: str | None = None,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str | None = None,
) -> dict:
    """One planner reads the merged lattice + relations and classifies
    every upstream entity-leaf as auto_expand / decline / manual.

    With `--artifact`, ingest an externally produced triage artifact
    instead of spawning a planner.
    """
    if artifact_path:
        run_id = new_run("triage", label="triage:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        counts = _validate_triage_artifact(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            **{f"{bucket}_count": counts[bucket] for bucket in _TRIAGE_BUCKETS},
        })
        return {"run_id": run_id, "artifact_path": str(used), **counts}

    source_lattice_path = (
        Path(lattice_path).resolve() if lattice_path
        else _latest_lattice_or_audit_or_linker_path()
    )

    run_id = new_run("triage", label="triage", seed=str(source_lattice_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if relations_path:
        relations_file = Path(relations_path).resolve()
    else:
        relations_file = run_root / config.TRIAGE_RELATIONS_FILE
        _aggregate_relations_artifact(relations_file)

    artifact_out = run_root / config.TRIAGE_ARTIFACT_FILE
    prompt = render_prompt("triage", {
        "run_id": run_id,
        "lattice_path": str(source_lattice_path),
        "relations_path": str(relations_file),
        "input_path": str(relations_file),
        "artifact_path": str(artifact_out),
        "worker_dir": str(run_root / config.WORKERS_SUBDIR),
        "planner_model": planner_model,
        "subagent_model": subagent_model or planner_model,
    })
    spawn = dispatch_spawn(run_id, prompt, model=planner_model)
    if spawn["exit_code"] != 0 or not artifact_out.exists():
        raise click.ClickException(f"triage failed; logs at {spawn['log_dir']}")
    artifact = read_json(str(artifact_out))
    counts = _validate_triage_artifact(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "lattice_path": str(source_lattice_path),
        "relations_path": str(relations_file),
        **{f"{bucket}_count": counts[bucket] for bucket in _TRIAGE_BUCKETS},
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out), **counts}


# ---------------------------------------------------------------------------
# Stage 8 — merge (pure-Python cross-run lattice + relations merge)
# ---------------------------------------------------------------------------


def _merge_lattices(artifacts: list[dict]) -> tuple[dict, list[dict]]:
    """Pure-Python merge of N lattice artifacts. Items unify by
    (formal_name, primary_link_url). Aliases and identity dicts merge;
    conflicts surface in the returned conflicts list.
    """
    by_family: dict[str, dict] = {}
    conflicts: list[dict] = []

    def primary_link(item: dict) -> str | None:
        for link in item.get("links") or []:
            if isinstance(link, dict) and link.get("url"):
                return str(link.get("url"))
        return None

    items_by_key: dict[tuple[str, str | None], dict] = {}
    for art in artifacts:
        for grp in art.get("groups") or []:
            family = grp.get("family") or ""
            family_entry = by_family.setdefault(family, {
                "family": family,
                "identity_keys": list(grp.get("identity_keys") or []),
                "items": [],
            })
            existing_keys = list(family_entry["identity_keys"])
            for key in grp.get("identity_keys") or []:
                if key not in existing_keys:
                    existing_keys.append(key)
            family_entry["identity_keys"] = existing_keys

            for item in grp.get("items") or []:
                formal = item.get("formal_name") or ""
                key = (formal, primary_link(item))
                if key in items_by_key:
                    target = items_by_key[key]
                    aliases = list(target.get("aliases") or [])
                    for alias in item.get("aliases") or []:
                        if alias not in aliases:
                            aliases.append(alias)
                    target["aliases"] = aliases
                    target_links = {
                        (l.get("kind"), l.get("url")): l
                        for l in (target.get("links") or [])
                        if isinstance(l, dict)
                    }
                    for link in item.get("links") or []:
                        if not isinstance(link, dict):
                            continue
                        target_links.setdefault(
                            (link.get("kind"), link.get("url")), link
                        )
                    target["links"] = list(target_links.values())
                    target_identity = dict(target.get("identity") or {})
                    new_identity = item.get("identity") or {}
                    for ikey, ival in new_identity.items():
                        if ikey not in target_identity:
                            target_identity[ikey] = ival
                        elif target_identity[ikey] != ival:
                            conflicts.append({
                                "kind": "identity_value",
                                "formal_name": formal,
                                "identity_key": ikey,
                                "values": sorted({
                                    str(target_identity[ikey]),
                                    str(ival),
                                }),
                            })
                    target["identity"] = target_identity
                else:
                    new_item = {
                        "kind": item.get("kind"),
                        "formal_name": formal,
                        "identity": dict(item.get("identity") or {}),
                        "aliases": list(item.get("aliases") or []),
                        "links": [
                            dict(l) for l in (item.get("links") or [])
                            if isinstance(l, dict)
                        ],
                    }
                    items_by_key[key] = new_item
                    family_entry["items"].append(new_item)

    return ({"groups": list(by_family.values())}, conflicts)


def _merge_relations(artifacts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pure-Python merge of N relations artifacts. Edges unify by
    (subject, relation, object_ref or object_text). Evidence /
    source_path / provenance_kind accumulate. Differing descriptions
    surface in conflicts.
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    conflicts: list[dict] = []
    for art in artifacts:
        for rel in art.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            object_id = rel.get("object_ref") or rel.get("object_text") or ""
            key = (rel.get("subject") or "", rel.get("relation") or "",
                   str(object_id))
            evidence_record = {
                "evidence": rel.get("evidence"),
                "source_path": rel.get("source_path"),
                "source_line": rel.get("source_line"),
                "provenance_kind": rel.get("provenance_kind"),
                "description": rel.get("description"),
            }
            if key in by_key:
                target = by_key[key]
                target.setdefault("provenance", []).append(evidence_record)
                target_desc = target.get("description")
                this_desc = rel.get("description")
                if (this_desc and target_desc and this_desc != target_desc
                        and this_desc not in (target.get("description_variants") or [])):
                    variants = list(target.get("description_variants") or [])
                    if target_desc not in variants:
                        variants.append(target_desc)
                    variants.append(this_desc)
                    target["description_variants"] = variants
                    conflicts.append({
                        "kind": "description_variant",
                        "subject": rel.get("subject"),
                        "relation": rel.get("relation"),
                        "object": object_id,
                        "variants": variants,
                    })
            else:
                merged = {
                    "subject": rel.get("subject"),
                    "subject_in_lattice": rel.get("subject_in_lattice"),
                    "relation": rel.get("relation"),
                    "direction": rel.get("direction"),
                    "object_ref": rel.get("object_ref"),
                    "object_in_lattice": rel.get("object_in_lattice"),
                    "object_text": rel.get("object_text"),
                    "object_value": rel.get("object_value"),
                    "object_unit": rel.get("object_unit"),
                    "description": rel.get("description"),
                    "provenance": [evidence_record],
                }
                by_key[key] = merged
    return (list(by_key.values()), conflicts)


def run_merge(
    *,
    sources: list[str] | None = None,
    relations_sources: list[str] | None = None,
    artifact_path: str | None = None,
) -> dict:
    """Pure-Python cross-run merge. Reads N lattice JSONs and N
    relations JSONs (counts may differ — relations merge is optional)
    and writes one merged artifact.

    With `--artifact`, ingest an externally produced merge artifact
    for shape validation only.
    """
    if artifact_path:
        run_id = new_run("merge", label="merge:ingest")
        used = Path(artifact_path).resolve()
        artifact = read_json(str(used))
        if not isinstance(artifact, dict) or "lattice" not in artifact:
            raise click.ClickException("merge artifact missing 'lattice' field")
        group_count = len(artifact.get("lattice", {}).get("groups") or [])
        item_count = sum(
            len(g.get("items") or [])
            for g in artifact.get("lattice", {}).get("groups") or []
        )
        relation_count = len(artifact.get("relations") or [])
        conflict_count = len(artifact.get("conflicts") or [])
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "relation_count": relation_count,
            "conflict_count": conflict_count,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                "relation_count": relation_count,
                "conflict_count": conflict_count}

    if not sources:
        raise click.ClickException(
            "merge requires --sources (paths to lattice artifacts)"
        )
    lattice_artifacts = [read_json(s) for s in sources]
    merged_lattice, lattice_conflicts = _merge_lattices(lattice_artifacts)

    merged_relations: list[dict] = []
    relation_conflicts: list[dict] = []
    if relations_sources:
        rel_artifacts = [read_json(s) for s in relations_sources]
        merged_relations, relation_conflicts = _merge_relations(rel_artifacts)

    run_id = new_run("merge", label="merge")
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / config.MERGE_ARTIFACT_FILE

    payload = {
        "sources": list(sources),
        "relations_sources": list(relations_sources or []),
        "lattice": merged_lattice,
        "relations": merged_relations,
        "conflicts": lattice_conflicts + relation_conflicts,
    }
    atomic_write_json(artifact_out, payload)

    group_count = len(merged_lattice.get("groups") or [])
    item_count = sum(
        len(g.get("items") or []) for g in merged_lattice.get("groups") or []
    )
    relation_count = len(merged_relations)
    conflict_count = len(payload["conflicts"])
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "sources": list(sources),
        "relations_sources": list(relations_sources or []),
        "group_count": group_count,
        "item_count": item_count,
        "relation_count": relation_count,
        "conflict_count": conflict_count,
    })
    return {
        "run_id": run_id,
        "artifact_path": str(artifact_out),
        "group_count": group_count,
        "item_count": item_count,
        "relation_count": relation_count,
        "conflict_count": conflict_count,
    }


# ---------------------------------------------------------------------------
# expand — operator-driven recursion. CLI wrapper that runs the full
# pipeline against an upstream node (queued by triage).
# ---------------------------------------------------------------------------


def run_expand(
    *,
    node: str,
    planner_model: str = config.CLAUDE_MODEL,
    subagent_model: str = config.CLAUDE_MODEL,
    skip: tuple[str, ...] = (),
) -> dict:
    """Run the full pipeline against `node` as a fresh target. Default
    skips none; pass `skip=("relate",)` to stop earlier. Each stage's
    result is captured in the returned dict.
    """
    out: dict[str, Any] = {"node": node, "stages": {}}
    if "discover" not in skip:
        out["stages"]["discover"] = run_discover(
            target=node,
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "extract" not in skip:
        out["stages"]["extract"] = run_extract(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "organize" not in skip:
        out["stages"]["organize"] = run_organize(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "audit" not in skip:
        out["stages"]["audit"] = run_audit(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "linker" not in skip:
        out["stages"]["linker"] = run_linker(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    if "relate" not in skip:
        out["stages"]["relate"] = run_relate(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    return out
