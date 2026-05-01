from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click

from . import config
from .artifacts import (
    aggregate_mentions,
    detect_conflicts,
    normalize_link_candidates,
    normalize_mention,
    apply_audit_updates,
    validate_mention_artifact,
)
from .grouping import group_mentions_for_review
from .lattice import build_lattice
from .linker import link_candidates_from_mentions, verify_candidates
from .store import (
    all_rows,
    batch_file_map,
    compute_batch_fingerprint,
    db,
    dumps,
    emit_json,
    json_text,
    loads,
    materialize_batch,
    new_id,
    now,
    read_json,
    scan_and_register,
    set_batch_artifact,
    truncate,
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


def new_run(stage: str, *, seed: str | None = None, label: str | None = None, parent_run_id: str | None = None) -> str:
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


def render_prompt(stage: str, variables: dict[str, str]) -> str:
    prompt_path = config.PROMPTS_DIR / f"{stage}.md"
    if not prompt_path.exists():
        raise click.ClickException(f"prompt not found: {prompt_path}")
    text = prompt_path.read_text()
    shared = config.PROMPTS_DIR / "shared-context.md"
    if stage != "shared-context" and shared.exists():
        text = f"{shared.read_text()}\n---\n\n{text}"
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def runtime_env(run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env[config.GDB_STORAGE_ENV] = str(config.STORAGE)
    env[config.GDB_PATH_ENV] = str(config.DB_PATH)
    env[config.GDB_RUN_ID_ENV] = run_id
    return env


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


def parse_run_usage(stdout_path: Path, stderr_path: Path) -> dict:
    text = ""
    for path in (stdout_path, stderr_path):
        if path.exists():
            text += "\n" + path.read_text(errors="replace")[-20000:]
    attrs: dict[str, Any] = {}
    patterns = {
        "cost_usd": r"(?:cost|total cost)\D+\$?([0-9]+(?:\.[0-9]+)?)",
        "input_tokens": r"input tokens\D+([0-9,]+)",
        "output_tokens": r"output tokens\D+([0-9,]+)",
        "turns": r"turns\D+([0-9,]+)",
        "tool_calls": r"tool calls\D+([0-9,]+)",
    }
    for key_name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        attrs[key_name] = float(raw) if key_name == "cost_usd" else int(raw)
    return attrs


def spawn_claude(run_id: str, prompt: str, *, model: str = config.CLAUDE_MODEL) -> dict:
    if not shutil.which("claude"):
        raise click.ClickException("claude CLI not found; pass --artifact to ingest an existing stage artifact")
    run_root = config.STORAGE / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "prompt.md").write_text(prompt)
    out_path = run_root / "stdout.txt"
    err_path = run_root / "stderr.txt"
    cmd = [
        "claude",
        "-p",
        prompt,
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "text",
        "--model",
        model,
    ]
    started = time.monotonic()
    with out_path.open("w") as stdout, err_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=config.ROOT,
            env=runtime_env(run_id),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            rc = proc.wait()
        except (KeyboardInterrupt, SystemExit):
            terminate_pgrp(proc.pid)
            raise
    elapsed = time.monotonic() - started
    usage = parse_run_usage(out_path, err_path)
    close_run(run_id, {"runtime": "claude", "model": model, "exit_code": rc, "elapsed_s": elapsed, **usage})
    return {"run_id": run_id, "exit_code": rc, "elapsed_s": elapsed, "log_dir": str(run_root)}


def ingest_discovery_artifact(artifact: dict, workspace_dir: Path) -> dict:
    enriched, per_batch_maps = scan_and_register(workspace_dir, artifact)
    maps = {m["batch_idx"]: m["file_map"] for m in per_batch_maps}
    with db() as conn:
        cur = conn.cursor()
        for idx, batch in enumerate(enriched.get("batches") or []):
            source_ids = [source.get("source_id") for source in batch.get("sources") or [] if source.get("source_id")]
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
    run_root = config.STORAGE / "runs" / run_id
    workspace = Path(workspace_dir).resolve() if workspace_dir else run_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_out = run_root / "discover_artifact.json"
    variables = {
        "run_id": run_id,
        "target": target,
        "workspace_dir": str(workspace),
        "worker_dir": str(run_root / "workers"),
        "artifact_path": str(artifact_out),
        "input_path": str(run_root / "input.json"),
        "planner_model": planner_model,
        "subagent_model": subagent_model,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "input.json").write_text(json_text({"target": target, "workspace_dir": str(workspace)}))
    if artifact_path:
        artifact = read_json(artifact_path)
        used_artifact = Path(artifact_path)
    else:
        prompt = render_prompt("discover", variables)
        spawn = spawn_claude(run_id, prompt, model=planner_model)
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
            {"batch_id": batch.get("batch_id"), "created": batch.get("created"), "source_count": len(batch.get("sources") or [])}
            for batch in enriched.get("batches") or []
        ],
    }


def _source_id_for_mention(cur, batch_id: str, mention: dict) -> str | None:
    if mention.get("source_id"):
        return mention["source_id"]
    anchors = mention.get("anchors") or []
    file = ""
    if anchors:
        file = anchors[0].get("file") or ""
    file_map = batch_file_map(cur, batch_id)
    for filename, source_id in file_map.items():
        if filename.casefold() == file.casefold():
            return source_id
    return None


def commit_mentions(artifact: dict, *, batch_id: str | None = None) -> dict:
    errors = validate_mention_artifact(artifact)
    if errors:
        return {"status": "failed", "errors": errors, "mentions_committed": 0}
    mentions = artifact.get("mentions") or []
    committed = 0
    with db() as conn:
        cur = conn.cursor()
        if batch_id:
            cur.execute("DELETE FROM mentions WHERE batch_id=?", (batch_id,))
        for raw in mentions:
            source_id = None
            if batch_id:
                provisional = normalize_mention(raw, batch_id=batch_id)
                source_id = _source_id_for_mention(cur, batch_id, provisional)
            mention = normalize_mention(raw, batch_id=batch_id, source_id=source_id)
            mention_id = mention.get("id") or new_id()
            cur.execute(
                """INSERT OR REPLACE INTO mentions
                   (id, batch_id, source_id, kind, surface, surface_key,
                    identity_json, identity_key, descriptors_json, aliases_json,
                    links_json, subsets_json, context_roles_json, atoms_json,
                    referent_scope, links_json, concept_path_json,
                    aux_json, relationships_json, anchors_json, description,
                    notes, attrs, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mention_id,
                    mention.get("batch_id"),
                    mention.get("source_id"),
                    mention["kind"],
                    mention["surface"],
                    mention["surface_key"],
                    dumps(mention["identity"]),
                    mention["identity_key"],
                    dumps(mention["descriptors"]),
                    dumps(mention["aliases"]),
                    dumps(mention["links"]),
                    dumps(mention["subsets"]),
                    dumps(mention["context_roles"]),
                    dumps(mention["atoms"]),
                    mention["referent_scope"],
                    dumps(mention["links"]),
                    dumps(mention["concept_path"]),
                    dumps(mention["aux"]),
                    dumps(mention["relationships"]),
                    dumps(mention["anchors"]),
                    mention.get("description"),
                    mention.get("notes"),
                    dumps(mention.get("attrs") or {}),
                    "active",
                    now(),
                    now(),
                ),
            )
            committed += 1
        if batch_id:
            set_batch_artifact(
                cur,
                batch_id=batch_id,
                stage="extract_mentions",
                artifact_path=str(Path(artifact.get("_artifact_path", "")).resolve()) if artifact.get("_artifact_path") else "",
                status="complete",
                attrs={"mentions_committed": committed},
            )
        conn.commit()
    return {"status": "complete", "mentions_committed": committed, "errors": []}


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
        return commit_mentions(artifact, batch_id=batch_id)
    batch_ids = [batch_id] if batch_id else [row["id"] for row in all_rows("SELECT id FROM batches ORDER BY created_at")]
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def extract_one(bid: str) -> dict:
        run_id = new_run("extract-mentions", label=f"extract-mentions:{bid[:8]}")
        run_root = config.STORAGE / "runs" / run_id
        batch_dir = materialize_batch(bid, run_root / "batch")
        artifact_out = run_root / "extract-mentions_artifact.json"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "input.json").write_text(json_text({"batch_id": bid, "batch_dir": str(batch_dir)}))
        prompt = render_prompt("extract", {
            "run_id": run_id,
            "batch_id": bid,
            "batch_dir": str(batch_dir),
            "worker_dir": str(run_root / "workers"),
            "artifact_path": str(artifact_out),
            "input_path": str(run_root / "input.json"),
            "planner_model": planner_model,
            "subagent_model": subagent_model,
        })
        spawn = spawn_claude(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            return {"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]}
        artifact = read_json(artifact_out)
        artifact["_artifact_path"] = str(artifact_out)
        result = commit_mentions(artifact, batch_id=bid)
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


def mention_rows() -> list[dict]:
    rows = all_rows("SELECT * FROM mentions WHERE status != 'dropped' ORDER BY created_at, id")
    out: list[dict] = []
    for row in rows:
        out.append({
            "id": row["id"],
            "batch_id": row["batch_id"],
            "source_id": row["source_id"],
            "kind": row["kind"],
            "surface": row["surface"],
            "identity": loads(row["identity_json"], default={}),
            "descriptors": loads(row["descriptors_json"], default={}),
            "aliases": loads(row["aliases_json"], default=[]),
            "links": loads(row["links_json"], default={}),
            "subsets": loads(row["subsets_json"], default=[]),
            "context_roles": loads(row["context_roles_json"], default=[]),
            "atoms": loads(row.get("atoms_json"), default=[]),
            "referent_scope": row.get("referent_scope") or "ambiguous",
            "links": loads(row.get("links_json"), default=[]),
            "concept_path": loads(row.get("concept_path_json"), default=[]),
            "aux": loads(row.get("aux_json"), default={}),
            "relationships": loads(row.get("relationships_json"), default=[]),
            "anchors": loads(row["anchors_json"], default=[]),
            "description": row.get("description"),
            "notes": row["notes"],
            "attrs": loads(row["attrs"], default={}),
        })
    return out


def run_check(*, artifact_path: str | None = None) -> dict:
    if artifact_path:
        artifact = read_json(artifact_path)
        validation_errors = validate_mention_artifact(artifact)
        mentions = artifact.get("mentions") if isinstance(artifact, dict) else []
        conflicts = detect_conflicts(mentions or [])
    else:
        validation_errors = []
        conflicts = detect_conflicts(mention_rows())
    violations = [
        {"code": err["code"], "severity": "error", "subject_key": err.get("path"), "details": err}
        for err in validation_errors
    ] + conflicts
    with db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE mention_violations SET status='resolved' WHERE status='open'")
        for violation in violations:
            cur.execute(
                """INSERT INTO mention_violations
                   (id, code, severity, subject_key, details_json, status, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    violation["code"],
                    violation.get("severity") or "error",
                    violation.get("subject_key"),
                    dumps(violation.get("details") or {}),
                    "open",
                    now(),
                ),
            )
        conn.commit()
    return {"violation_count": len(violations), "violations": violations}


def cluster_packet() -> dict:
    """Build the cluster + violations packet that the audit stage reads.

    The audit prompt expects clusters (deduped via aggregate_mentions)
    plus the open violations. The planner decides how to bucket these
    and dispatches subagents.
    """
    mentions = mention_rows()
    clusters = aggregate_mentions(mentions)
    violations = all_rows("SELECT * FROM mention_violations WHERE status='open' ORDER BY created_at, code")
    members_by_cluster: dict[str, list[str]] = {}
    for mention in mentions:
        normalized = normalize_mention(mention)
        from .artifacts import cluster_key_for_mention
        ck = cluster_key_for_mention(normalized)
        if ck:
            members_by_cluster.setdefault(ck, []).append(normalized.get("id") or "")
    return {
        "clusters": [
            {**cluster, "member_mention_ids": members_by_cluster.get(cluster["cluster_key"], [])}
            for cluster in clusters
        ],
        "violations": [
            {
                "id": violation["id"],
                "code": violation["code"],
                "severity": violation["severity"],
                "subject_key": violation["subject_key"],
                "details": loads(violation["details_json"], default={}),
            }
            for violation in violations
        ],
    }


def run_audit(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL, subagent_model: str | None = None) -> dict:
    """Single-planner cluster audit. The planner reads the cluster packet,
    buckets clusters, dispatches subagents, and writes one update artifact
    that we apply to the mention rows.

    Replaces the old separate repair + link-unresolved stages.
    """
    if not artifact_path:
        run_id = new_run("audit", label="audit")
        run_root = config.STORAGE / "runs" / run_id
        packet_path = run_root / "cluster_packet.json"
        artifact_out = run_root / "audit_artifact.json"
        atomic_write_json(packet_path, cluster_packet())
        prompt = render_prompt("audit", {
            "run_id": run_id,
            "cluster_packet_path": str(packet_path),
            "artifact_path": str(artifact_out),
            "planner_model": planner_model,
            "subagent_model": subagent_model or planner_model,
        })
        spawn = spawn_claude(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            raise click.ClickException(f"audit failed; logs at {spawn['log_dir']}")
        artifact_path = str(artifact_out)
    audit_artifact = read_json(artifact_path)
    updates = audit_artifact.get("updates") or []
    expanded = _expand_cluster_updates(updates)
    audited = apply_audit_updates(mention_rows(), {"updates": expanded})
    with db() as conn:
        cur = conn.cursor()
        for mention in audited:
            status = mention.get("status") or "active"
            cur.execute(
                """UPDATE mentions
                      SET kind=?, surface=?, surface_key=?, identity_json=?, identity_key=?,
                          descriptors_json=?, aliases_json=?, subsets_json=?,
                          context_roles_json=?, atoms_json=?, referent_scope=?,
                          links_json=?, concept_path_json=?, aux_json=?,
                          relationships_json=?, anchors_json=?, description=?,
                          notes=?, status=?, updated_at=?
                    WHERE id=?""",
                (
                    mention["kind"],
                    mention["surface"],
                    mention["surface_key"],
                    dumps(mention["identity"]),
                    mention["identity_key"],
                    dumps(mention["descriptors"]),
                    dumps(mention["aliases"]),
                    dumps(mention["subsets"]),
                    dumps(mention["context_roles"]),
                    dumps(mention["atoms"]),
                    mention["referent_scope"],
                    dumps(mention["links"]),
                    dumps(mention["concept_path"]),
                    dumps(mention["aux"]),
                    dumps(mention["relationships"]),
                    dumps(mention["anchors"]),
                    mention.get("description"),
                    mention.get("notes"),
                    status,
                    now(),
                    mention["id"],
                ),
            )
        conn.commit()
    check = run_check()
    return {"audited_mentions": len(audited), "post_audit": check}


def _expand_cluster_updates(updates: list[dict]) -> list[dict]:
    """Expand cluster-keyed updates to per-mention updates."""
    if not isinstance(updates, list):
        return []
    mentions = [normalize_mention(m) for m in mention_rows()]
    from .artifacts import cluster_key_for_mention
    members_by_cluster: dict[str, list[str]] = {}
    for mention in mentions:
        ck = cluster_key_for_mention(mention)
        if ck:
            members_by_cluster.setdefault(ck, []).append(mention.get("id") or "")
    expanded: list[dict] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        if update.get("mention_id"):
            expanded.append(update)
            continue
        cluster_key = update.get("cluster_key")
        members = members_by_cluster.get(cluster_key) if cluster_key else []
        for member_id in members:
            expanded.append({**update, "mention_id": member_id})
    return expanded


def run_verify_links() -> dict:
    mentions = mention_rows()
    candidates = link_candidates_from_mentions(mentions)
    checks = verify_candidates(candidates)
    with db() as conn:
        cur = conn.cursor()
        for check in checks:
            cur.execute(
                """INSERT INTO link_checks
                   (id, cluster_key, kind, link_kind, link_value, url, ok,
                    status_code, error, checked_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    check["cluster_key"],
                    check["kind"],
                    check["link_kind"],
                    check["link_value"],
                    check["url"],
                    1 if check["ok"] else 0,
                    check.get("status_code"),
                    check.get("error"),
                    now(),
                ),
            )
        conn.commit()
    return {"candidate_count": len(candidates), "verified_count": sum(1 for c in checks if c["ok"]), "checks": checks}


def run_build_lattice() -> dict:
    mentions = mention_rows()
    checks = all_rows("SELECT * FROM link_checks")
    lattice = build_lattice(mentions, checks)
    with db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM lattice_edges")
        cur.execute("DELETE FROM lattice_nodes")
        for node in lattice["nodes"]:
            cur.execute(
                """INSERT INTO lattice_nodes
                   (id, node_key, kind, node_type, identity_json, concept_path_json,
                    display_name, aliases_json, descriptors_json, links_json,
                    verified_links_json, aux_json, description, occurrence_count,
                    projection, flags_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    node["node_key"],
                    node["kind"],
                    node["node_type"],
                    dumps(node["identity"]),
                    dumps(node["concept_path"]),
                    node["display_name"],
                    dumps(node["aliases"]),
                    dumps(node["descriptors"]),
                    dumps(node["links"]),
                    dumps(node["verified_links"]),
                    dumps(node["aux"]),
                    node.get("description"),
                    node["occurrence_count"],
                    1 if node["projection"] else 0,
                    dumps(node["flags"]),
                    now(),
                ),
            )
        for edge in lattice["edges"]:
            cur.execute(
                """INSERT OR IGNORE INTO lattice_edges
                   (parent_node_key, child_node_key, rationale)
                   VALUES (?,?,?)""",
                (edge["parent_node_key"], edge["child_node_key"], edge["rationale"]),
            )
        conn.commit()
    return {
        "node_count": len(lattice["nodes"]),
        "edge_count": len(lattice["edges"]),
        "flagged_nodes": [node for node in lattice["nodes"] if node["flags"]],
        "forests": lattice.get("forests") or [],
        "audit": lattice.get("audit") or {},
    }




def run_describe(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL, subagent_model: str | None = None) -> dict:
    """Single-planner per-entity-leaf description with inline HF fetch.

    The planner reads the lattice, filters concept nodes out, buckets
    entity leaves, dispatches subagents that fetch HF metadata via
    enrich_hf_link and write descriptions. Concepts get no descriptions.
    """
    if not artifact_path:
        run_id = new_run("describe", label="describe")
        run_root = config.STORAGE / "runs" / run_id
        lattice_path = run_root / "lattice.json"
        artifact_out = run_root / "describe_artifact.json"
        mentions = mention_rows()
        checks = all_rows("SELECT * FROM link_checks")
        lattice = build_lattice(mentions, checks)
        atomic_write_json(lattice_path, lattice)
        prompt = render_prompt("describe", {
            "run_id": run_id,
            "lattice_path": str(lattice_path),
            "artifact_path": str(artifact_out),
            "planner_model": planner_model,
            "subagent_model": subagent_model or planner_model,
        })
        spawn = spawn_claude(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            raise click.ClickException(f"describe failed; logs at {spawn['log_dir']}")
        artifact_path = str(artifact_out)
    describe_artifact = read_json(artifact_path)
    descriptions = describe_artifact.get("descriptions") or []
    with db() as conn:
        cur = conn.cursor()
        for item in descriptions:
            entity_key = item.get("entity_key")
            if not entity_key:
                continue
            kind = item.get("kind") or "model"
            existing = cur.execute(
                "SELECT entity_key FROM entity_descriptions WHERE entity_key=?",
                (entity_key,),
            ).fetchone()
            if existing:
                cur.execute(
                    """UPDATE entity_descriptions
                          SET kind=?, display_name=?, links_json=?, description=?,
                              metadata_json=?, source_json=?, updated_at=?
                        WHERE entity_key=?""",
                    (
                        kind,
                        item.get("display_name") or "",
                        dumps(item.get("links") or []),
                        item.get("description") or "",
                        dumps(item.get("metadata") or {}),
                        dumps(item.get("source") or {}),
                        now(),
                        entity_key,
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO entity_descriptions
                       (entity_key, kind, display_name, links_json, description,
                        metadata_json, source_json, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        entity_key,
                        kind,
                        item.get("display_name") or "",
                        dumps(item.get("links") or []),
                        item.get("description") or "",
                        dumps(item.get("metadata") or {}),
                        dumps(item.get("source") or {}),
                        now(),
                        now(),
                    ),
                )
        conn.commit()
    return {"description_count": len(descriptions), "descriptions": descriptions}
