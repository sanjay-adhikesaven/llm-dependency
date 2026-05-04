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


def _stream_indicates_rate_limit(stream_path: Path) -> bool:
    """Return True iff the stream JSONL contains rate-limit / 429 /
    overloaded error markers. Used by `dispatch_spawn` to decide
    whether a non-zero exit is worth retrying."""
    if not stream_path.exists():
        return False
    needles = (
        "rate_limit", "rate-limit", "overloaded_error",
        "429", "too many requests", "RATE_LIMIT",
    )
    try:
        for line in stream_path.read_text(errors="replace").splitlines():
            if not line:
                continue
            low = line.lower()
            if any(n.lower() in low for n in needles):
                return True
    except OSError:
        return False
    return False


def dispatch_spawn(
    run_id: str,
    prompt: str,
    *,
    model: str,
    max_retries: int = 4,
) -> dict:
    """Dispatch one Claude / Codex spawn. On non-zero exit, retry up to
    `max_retries` times with exponential backoff (10s, 30s, 90s, 270s)
    when the failure looks rate-limit-related; fail immediately
    otherwise. Each retry creates a NEW run row so logs / streams
    don't clobber.

    Rate-limit detection scans the stream JSONL for `rate_limit`,
    `429`, `overloaded_error`, etc. Codex retries are also rate-limit-
    triggered but use a coarser stderr scan since codex doesn't emit
    a JSONL stream.
    """
    backoff_schedule = (10, 30, 90, 270)

    def _spawn_once(rid: str) -> dict:
        if model.startswith("codex-"):
            return spawn_codex(rid, prompt, effort=model.removeprefix("codex-"))
        return spawn_claude(rid, prompt, model=model)

    attempt_run_id = run_id
    last_result: dict = {}
    for attempt in range(max_retries + 1):
        result = _spawn_once(attempt_run_id)
        last_result = result
        rc = result.get("exit_code", 0)
        if rc == 0:
            return result
        if attempt >= max_retries:
            break

        # Decide: is this a rate-limit failure worth retrying?
        run_root = config.STORAGE / config.RUNS_SUBDIR / attempt_run_id
        rate_limited = False
        if not model.startswith("codex-"):
            rate_limited = _stream_indicates_rate_limit(
                run_root / config.RUN_STREAM_FILE
            )
        else:
            err_path = run_root / config.RUN_STDERR_FILE
            if err_path.exists():
                try:
                    err = err_path.read_text(errors="replace").lower()
                    rate_limited = any(
                        n in err for n in
                        ("rate_limit", "rate-limit", "429",
                         "too many requests", "overloaded")
                    )
                except OSError:
                    pass
        if not rate_limited:
            break

        sleep_s = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
        time.sleep(sleep_s)
        # Mint a fresh run id so the next attempt's stream doesn't
        # overwrite the failed one.
        attempt_run_id = new_run(
            "retry", seed=run_id,
            label=f"retry:{run_id[:8]}:attempt{attempt + 2}",
        )
    return last_result


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


_LINK_KINDS = frozenset({
    "hf_model", "hf_dataset", "hf_dataset_config", "hf_collection",
    "github", "paper", "blog", "vendor_docs",
})


def _validate_organize_artifact(artifact: dict) -> tuple[int, int]:
    """Sanity-check the organize / audit artifact's groups+items shape
    and return (group_count, item_count). Each item must carry a
    `links` list (possibly empty — audit may drop empty-links items
    after one more resolution attempt) and a `description` field
    (string or null). When `links` is non-empty, the first entry's
    `kind` must be one of the closed-vocabulary link kinds and `url`
    must be an http(s) URL string."""
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
        for j, item in enumerate(items):
            if not isinstance(item, dict):
                raise click.ClickException(
                    f"groups[{i}].items[{j}] is not a dict"
                )
            if "links" not in item:
                raise click.ClickException(
                    f"groups[{i}].items[{j}] missing required `links` field"
                )
            links = item["links"]
            if not isinstance(links, list):
                raise click.ClickException(
                    f"groups[{i}].items[{j}].links is not a list"
                )
            if links:
                head = links[0]
                if not isinstance(head, dict):
                    raise click.ClickException(
                        f"groups[{i}].items[{j}].links[0] is not a dict"
                    )
                kind = head.get("kind")
                if kind not in _LINK_KINDS:
                    raise click.ClickException(
                        f"groups[{i}].items[{j}].links[0].kind={kind!r} "
                        f"not in {sorted(_LINK_KINDS)}"
                    )
                url = head.get("url")
                if not isinstance(url, str) or not url.startswith(
                    ("http://", "https://")
                ):
                    raise click.ClickException(
                        f"groups[{i}].items[{j}].links[0].url={url!r} "
                        "must be an http(s) URL string"
                    )
            description = item.get("description")
            if description is not None and not isinstance(description, str):
                raise click.ClickException(
                    f"groups[{i}].items[{j}].description must be string or null"
                )
            # subsets[] is only required on datasets, but we accept it on
            # any item shape so audit-time augmentation doesn't trip a
            # spurious model-side error.
            if "subsets" in item:
                subsets = item["subsets"]
                if not isinstance(subsets, list):
                    raise click.ClickException(
                        f"groups[{i}].items[{j}].subsets must be a list"
                    )
                for k, s in enumerate(subsets):
                    if not isinstance(s, str) or not s.strip():
                        raise click.ClickException(
                            f"groups[{i}].items[{j}].subsets[{k}] "
                            "must be a non-empty string"
                        )
            # Every item MUST trace to ≥1 real input surface form via
            # aliases. Empty aliases signals a phantom item invented by
            # the planner enumerating HF releases the input never named.
            # The narrow exception is a family-concept root whose
            # identity carries only broad keys (no size/stage/date/etc.).
            aliases = item.get("aliases") or []
            if not aliases:
                identity = item.get("identity") or {}
                broad_keys = {"org", "collection", "vendor", "family"}
                if not _is_family_concept_root(identity, broad_keys):
                    raise click.ClickException(
                        f"groups[{i}].items[{j}] formal_name="
                        f"{item.get('formal_name')!r} has empty aliases — "
                        "every item must fold ≥1 real input surface form. "
                        "Family-concept roots (identity only carrying "
                        f"{sorted(broad_keys)}) are the only exception."
                    )
        item_count += len(items)
    return len(groups), item_count


def _is_family_concept_root(identity: dict, broad_keys: set[str]) -> bool:
    """Return True iff `identity` carries only broad-vocabulary keys
    (no size / stage / date / quantization / variant / subset / harness),
    qualifying it as a family-concept root that may have empty aliases."""
    if not isinstance(identity, dict):
        return False
    keys = set(identity.keys())
    return bool(keys) and keys.issubset(broad_keys)


def _count_link_stats(artifact: dict) -> dict:
    """Tally link / description / kind stats across the artifact:
    total items, items with >=1 link, items with a description, model
    vs dataset counts, total links, link-kind histogram."""
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
    link_stats = _count_link_stats(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "group_count": group_count,
        "item_count": item_count,
        **link_stats,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count,
            **link_stats}


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
        link_stats = _count_link_stats(artifact)
        close_run(run_id, {
            "artifact_path": str(used),
            "group_count": group_count,
            "item_count": item_count,
            "notes": _short_notes(artifact),
            **link_stats,
        })
        return {"run_id": run_id, "artifact_path": str(used),
                "group_count": group_count, "item_count": item_count,
                **link_stats}

    source_artifact_path = (
        Path(source_path).resolve() if source_path
        else _latest_lattice_artifact_path()
    )

    run_id = new_run("audit", label="audit", seed=str(source_artifact_path))
    run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Python pre-pass — populate subsets[] on every dataset
    # node, then cross-check dropped[] against populated subsets[] and
    # restore matches as child items. The pre-processed lattice is
    # what the LLM auditor sees.
    pre_processed = read_json(str(source_artifact_path))
    try:
        from .subsets import populate_then_flag
        subset_stats = populate_then_flag(pre_processed)
    except Exception as exc:  # network / parse failures shouldn't block audit
        subset_stats = {"populate": {"error": str(exc)},
                        "restore": {"error": str(exc)}}
    pre_processed_path = run_root / "audit_input_with_subsets.json"
    atomic_write_json(pre_processed_path, pre_processed)

    artifact_out = run_root / config.AUDIT_ARTIFACT_FILE
    prompt = render_prompt("audit", {
        "run_id": run_id,
        "organize_path": str(pre_processed_path),
        "input_path": str(pre_processed_path),
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
    link_stats = _count_link_stats(artifact)
    close_run(run_id, {
        "artifact_path": str(artifact_out),
        "source_artifact_path": str(source_artifact_path),
        "pre_processed_path": str(pre_processed_path),
        "subset_stats": subset_stats,
        "group_count": group_count,
        "item_count": item_count,
        "notes": _short_notes(artifact),
        **link_stats,
    })
    return {"run_id": run_id, "artifact_path": str(artifact_out),
            "group_count": group_count, "item_count": item_count,
            "subset_stats": subset_stats,
            **link_stats}


# ---------------------------------------------------------------------------
# Stage 5 — relate (per batch, lattice-anchored typed edges)
# ---------------------------------------------------------------------------


_DIRECT_RELATIONS = (
    "trained_on", "trained_from", "generated_by",
    "transformed_by", "filtered_by",
)
_INDIRECT_RELATIONS = (
    "inspired_by", "used_for_ablation", "used_for_evaluation",
)
# Canonical labels — guidance for the relate prompt and tracking.
# `relation` is OPEN: the planner may coin a new snake_case label when
# none of these canonical values fits the source's described event.
CANONICAL_RELATION_VALUES = (
    *_DIRECT_RELATIONS, *_INDIRECT_RELATIONS,
)
# Map of canonical relation → its `dependency_kind` bucket.
RELATION_DEPENDENCY_KIND = {
    **{r: "direct" for r in _DIRECT_RELATIONS},
    **{r: "indirect" for r in _INDIRECT_RELATIONS},
}
# Closed vocabulary for `dependency_kind`.
DEPENDENCY_KIND_VALUES = ("direct", "indirect")


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


def _validate_anchor_list(anchors: object, where: str) -> None:
    """Validate `anchor_list` shape: non-empty list of dicts with required
    `source: str` and `explanation: str`; optional `position: str`."""
    if not isinstance(anchors, list) or not anchors:
        raise click.ClickException(f"{where}.anchor_list must be a non-empty list")
    for j, anc in enumerate(anchors):
        if not isinstance(anc, dict):
            raise click.ClickException(f"{where}.anchor_list[{j}] is not a dict")
        src = anc.get("source")
        if not isinstance(src, str) or not src.strip():
            raise click.ClickException(
                f"{where}.anchor_list[{j}].source must be a non-empty string"
            )
        expl = anc.get("explanation")
        if not isinstance(expl, str) or not expl.strip():
            raise click.ClickException(
                f"{where}.anchor_list[{j}].explanation must be a non-empty string"
            )
        pos = anc.get("position")
        if pos is not None and not isinstance(pos, str):
            raise click.ClickException(
                f"{where}.anchor_list[{j}].position must be a string when present"
            )


def _validate_relate_artifact(artifact: dict, *,
                              lattice_formal_names: set[str] | None = None
                              ) -> dict:
    """Sanity-check the assembled relate artifact's shape. Returns:

    {
      "operation_count":       int,
      "edge_count":            int,
      "singleton_event_count": int,   # events with exactly 1 edge
      "off_lattice_object_count": int,
      "direct_count":          int,
      "indirect_count":        int,
      "coined_relations":      {label: count},
    }

    Schema (post-fix):

    {
      "batch_id":       "...",
      "batch_label":    "...",
      "operations": [
        {
          "description": "...",
          "anchor_list": [{"source": "...", "position"?: "...", "explanation": "..."}],
          "edges": [
            {
              "subject":         "<lattice formal_name>",
              "relation":        "trained_on" | ... | "<coined>",
              "dependency_kind": "direct" | "indirect",
              "object":          "<formal_name OR free-text>",
              "description":     "...",
              "anchor_list":     [...]
            }
          ]
        }
      ]
    }

    Closed-vocab enforced:
      - `dependency_kind` ∈ {direct, indirect}.
      - `subject` must be a lattice formal_name when lattice is provided.

    Open-vocab tracked:
      - `relation`: snake_case; values outside `CANONICAL_RELATION_VALUES`
        are counted as coined but not rejected.
    """
    if not isinstance(artifact, dict):
        raise click.ClickException("relate artifact is not a dict")
    operations = artifact.get("operations")
    if not isinstance(operations, list):
        raise click.ClickException("relate artifact missing operations[]")

    canonical_relations = set(CANONICAL_RELATION_VALUES)

    edge_total = 0
    singleton_events = 0
    off_lattice = 0
    direct_count = 0
    indirect_count = 0
    coined_relations: dict[str, int] = {}

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise click.ClickException(f"operations[{i}] is not a dict")
        desc = op.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise click.ClickException(
                f"operations[{i}].description is missing or empty"
            )
        _validate_anchor_list(op.get("anchor_list"), f"operations[{i}]")

        edges = op.get("edges")
        if not isinstance(edges, list) or not edges:
            raise click.ClickException(
                f"operations[{i}].edges must be a non-empty list"
            )
        if len(edges) == 1:
            singleton_events += 1
        edge_total += len(edges)

        for j, edge in enumerate(edges):
            where = f"operations[{i}].edges[{j}]"
            if not isinstance(edge, dict):
                raise click.ClickException(f"{where} is not a dict")

            subject = edge.get("subject")
            if not isinstance(subject, str) or not subject.strip():
                raise click.ClickException(f"{where}.subject is missing")
            if (lattice_formal_names is not None
                    and subject not in lattice_formal_names):
                raise click.ClickException(
                    f"{where}.subject {subject!r} is not a lattice formal_name"
                )

            relation = edge.get("relation")
            if not _is_snake_case_label(relation):
                raise click.ClickException(
                    f"{where}.relation {relation!r} is not a valid label "
                    f"(non-empty snake_case string ≤64 chars)"
                )
            if relation not in canonical_relations:
                coined_relations[relation] = coined_relations.get(relation, 0) + 1

            dep_kind = edge.get("dependency_kind")
            if dep_kind not in DEPENDENCY_KIND_VALUES:
                raise click.ClickException(
                    f"{where}.dependency_kind {dep_kind!r} not in "
                    f"{DEPENDENCY_KIND_VALUES}"
                )
            if dep_kind == "direct":
                direct_count += 1
            else:
                indirect_count += 1

            obj = edge.get("object")
            if not isinstance(obj, str) or not obj.strip():
                raise click.ClickException(
                    f"{where}.object must be a non-empty string"
                )
            if (lattice_formal_names is not None
                    and obj not in lattice_formal_names):
                off_lattice += 1

            edge_desc = edge.get("description")
            if not isinstance(edge_desc, str) or not edge_desc.strip():
                raise click.ClickException(
                    f"{where}.description is missing or empty"
                )

            _validate_anchor_list(edge.get("anchor_list"), where)

    return {
        "operation_count": len(operations),
        "edge_count": edge_total,
        "singleton_event_count": singleton_events,
        "off_lattice_object_count": off_lattice,
        "direct_count": direct_count,
        "indirect_count": indirect_count,
        "coined_relations": coined_relations,
    }


def assemble_relate_artifact_from_jsonl(
    events_path: Path, *,
    batch_id: str | None = None,
    batch_label: str | None = None,
) -> dict:
    """Read JSONL events from `events_path`, one event per line, and
    assemble into a single relate artifact dict:

    {batch_id, batch_label, operations: [<event>, ...]}.

    Each event is the parsed JSON object on its line. The pipeline calls
    this after the planner exits — the planner appends events as it
    works, so the JSONL file is the durable record."""
    operations: list[dict] = []
    if not events_path.exists():
        return {
            "batch_id": batch_id,
            "batch_label": batch_label,
            "operations": operations,
        }
    text = events_path.read_text()
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            op = loads(line, default=None)
        except Exception:
            op = None
        if op is None:
            try:
                op = __import__("json").loads(line)
            except Exception as e:
                raise click.ClickException(
                    f"{events_path} line {n}: not valid JSON: {e!r}"
                )
        if not isinstance(op, dict):
            raise click.ClickException(
                f"{events_path} line {n}: top-level must be a JSON object"
            )
        operations.append(op)
    return {
        "batch_id": batch_id,
        "batch_label": batch_label,
        "operations": operations,
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
    on items the organize / audit stages confirmed are publicly
    resolvable. Items without a verified link are not safe subjects
    for closed-vocabulary edges — they may be private / gated /
    phantom names that the lattice should not propagate downstream.
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
        else _latest_lattice_artifact_path()
    )
    lattice_artifact = read_json(str(source_lattice_path))
    # Pass the full lattice to the relate planner. Items without a
    # verified link can still be valid edge endpoints (e.g., gated HF
    # repos, API-only judges, internal AI2 names referenced in source).
    # The relate prompt's free-text `object` field handles off-lattice
    # mentions; the planner should not be deprived of extracted-but-
    # unlinkable entities.
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

    def _batch_label(bid: str) -> str | None:
        rows = all_rows("SELECT label FROM batches WHERE id=?", (bid,))
        if rows:
            return rows[0]["label"]
        return None

    def relate_one(bid: str) -> dict:
        run_id = new_run("relate", label=f"relate:{bid[:8]}",
                         seed=str(source_lattice_path))
        run_root = config.STORAGE / config.RUNS_SUBDIR / run_id
        batch_dir = materialize_batch(bid, run_root / config.BATCH_SUBDIR)
        # The planner appends events as JSONL into events_path during
        # its turn. After it exits we assemble that JSONL into the
        # canonical relate_artifact.json.
        events_path = run_root / config.RELATE_EVENTS_FILE
        artifact_out = run_root / config.RELATE_ARTIFACT_FILE
        run_root.mkdir(parents=True, exist_ok=True)
        events_path.touch()  # ensure the planner can append from line 1
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
            "artifact_path": str(events_path),  # JSONL append target
            "input_path": str(run_root / config.RUN_INPUT_FILE),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = dispatch_spawn(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0:
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        # Assemble JSONL → JSON
        try:
            artifact = assemble_relate_artifact_from_jsonl(
                events_path, batch_id=bid, batch_label=_batch_label(bid),
            )
        except click.ClickException as exc:
            return {"batch_id": bid, "status": "failed",
                    "log_dir": spawn["log_dir"], "error": str(exc)}
        atomic_write_json(artifact_out, artifact)
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
        result["events_path"] = str(events_path)
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
# Stage 6 — triage (one planner classifies upstream nodes)
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
    triage planner reads. Each batch artifact has the
    `operations[].edges[]` shape; we flatten edges across all
    operations and tag each with the batch / event it came from.
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
        operations = artifact.get("operations") or []
        for op_idx, op in enumerate(operations):
            if not isinstance(op, dict):
                continue
            event_desc = op.get("description")
            event_anchors = op.get("anchor_list") or []
            for edge in op.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                merged.append({
                    **edge,
                    "_batch_id": row["batch_id"],
                    "_event_index": op_idx,
                    "_event_description": event_desc,
                    "_event_anchor_list": event_anchors,
                })
        batch_ids.append(row["batch_id"])
    atomic_write_json(out_path, {
        "batch_ids": batch_ids,
        "edges": merged,
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
        else _latest_lattice_artifact_path()
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
# Stage 7 — merge (pure-Python cross-run lattice + relations merge)
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
    """Pure-Python merge of N relate artifacts. Edges unify by
    (subject, relation, object). The accumulated `anchor_list` of
    each merged edge carries every source from every contributing
    artifact. Differing per-edge descriptions surface in conflicts.

    Each artifact is shaped as `{operations: [{description, anchor_list,
    edges: [...]}]}`; edges are flattened across all operations.
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    conflicts: list[dict] = []
    for art in artifacts:
        for op in art.get("operations") or []:
            if not isinstance(op, dict):
                continue
            for edge in op.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                key = (
                    edge.get("subject") or "",
                    edge.get("relation") or "",
                    edge.get("object") or "",
                )
                anchors = list(edge.get("anchor_list") or [])
                if key in by_key:
                    target = by_key[key]
                    target.setdefault("anchor_list", []).extend(anchors)
                    target_desc = target.get("description")
                    this_desc = edge.get("description")
                    if (this_desc and target_desc and this_desc != target_desc
                            and this_desc not in (target.get("description_variants") or [])):
                        variants = list(target.get("description_variants") or [])
                        if target_desc not in variants:
                            variants.append(target_desc)
                        variants.append(this_desc)
                        target["description_variants"] = variants
                        conflicts.append({
                            "kind": "description_variant",
                            "subject": edge.get("subject"),
                            "relation": edge.get("relation"),
                            "object": edge.get("object"),
                            "variants": variants,
                        })
                else:
                    by_key[key] = {
                        "subject": edge.get("subject"),
                        "relation": edge.get("relation"),
                        "dependency_kind": edge.get("dependency_kind"),
                        "object": edge.get("object"),
                        "description": edge.get("description"),
                        "anchor_list": anchors,
                    }
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
    if "relate" not in skip:
        out["stages"]["relate"] = run_relate(
            planner_model=planner_model, subagent_model=subagent_model,
        )
    return out
