from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import click

from . import config
from .artifacts import (
    aggregate_mentions,
    detect_conflicts,
    normalize_mention,
    repair_mentions,
    validate_mention_artifact,
)
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
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


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
    close_run(run_id, {"runtime": "claude", "model": model, "exit_code": rc, "elapsed_s": elapsed})
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


def run_discover_target(
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
    evidence = mention.get("evidence") or []
    file = ""
    if evidence:
        file = evidence[0].get("file") or ""
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
                    links_json, subsets_json, context_roles_json, evidence_json,
                    notes, attrs, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    dumps(mention["evidence"]),
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


def run_extract_mentions(
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
    results: list[dict] = []
    for bid in batch_ids:
        run_id = new_run("extract-mentions", label=f"extract-mentions:{bid[:8]}")
        run_root = config.STORAGE / "runs" / run_id
        batch_dir = materialize_batch(bid, run_root / "batch")
        artifact_out = run_root / "extract-mentions_artifact.json"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "input.json").write_text(json_text({"batch_id": bid, "batch_dir": str(batch_dir)}))
        prompt = render_prompt("extract-mentions", {
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
            results.append({"batch_id": bid, "status": "failed", "log_dir": spawn["log_dir"]})
            continue
        artifact = read_json(artifact_out)
        artifact["_artifact_path"] = str(artifact_out)
        result = commit_mentions(artifact, batch_id=bid)
        result["batch_id"] = bid
        result["run_id"] = run_id
        results.append(result)
    failed = [r for r in results if r.get("status") != "complete"]
    return {"results": results, "failed": len(failed)}


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
            "evidence": loads(row["evidence_json"], default=[]),
            "notes": row["notes"],
            "attrs": loads(row["attrs"], default={}),
        })
    return out


def run_check_mentions(*, artifact_path: str | None = None) -> dict:
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


def compact_repair_packet() -> dict:
    violations = all_rows("SELECT * FROM mention_violations WHERE status='open' ORDER BY created_at, code")
    mentions = mention_rows()
    by_surface = {m["surface"].casefold(): m for m in mentions}
    examples = []
    for violation in violations[:100]:
        details = loads(violation["details_json"], default={})
        surface = details.get("surface") or details.get("surface_key") or violation["subject_key"]
        candidate = by_surface.get(str(surface).casefold())
        examples.append({
            "violation": {
                "id": violation["id"],
                "code": violation["code"],
                "subject_key": violation["subject_key"],
                "details": details,
            },
            "mention": candidate,
        })
    return {"violations": examples, "repair_schema": {"updates": [{"mention_id": "...", "identity": {}, "aliases": []}]}}


def run_repair_mentions(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL) -> dict:
    if not artifact_path:
        run_id = new_run("repair-mentions", label="repair-mentions")
        run_root = config.STORAGE / "runs" / run_id
        packet_path = run_root / "repair_packet.json"
        artifact_out = run_root / "repair-mentions_artifact.json"
        packet = compact_repair_packet()
        atomic_write_json(packet_path, packet)
        prompt = render_prompt("repair-mentions", {
            "run_id": run_id,
            "repair_packet_path": str(packet_path),
            "artifact_path": str(artifact_out),
            "planner_model": planner_model,
        })
        spawn = spawn_claude(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            raise click.ClickException(f"repair-mentions failed; logs at {spawn['log_dir']}")
        artifact_path = str(artifact_out)
    repair_artifact = read_json(artifact_path)
    repaired = repair_mentions(mention_rows(), repair_artifact)
    with db() as conn:
        cur = conn.cursor()
        for mention in repaired:
            status = mention.get("status") or "active"
            cur.execute(
                """UPDATE mentions
                      SET kind=?, surface=?, surface_key=?, identity_json=?, identity_key=?,
                          descriptors_json=?, aliases_json=?, links_json=?, subsets_json=?,
                          context_roles_json=?, evidence_json=?, notes=?, status=?, updated_at=?
                    WHERE id=?""",
                (
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
                    dumps(mention["evidence"]),
                    mention.get("notes"),
                    status,
                    now(),
                    mention["id"],
                ),
            )
        conn.commit()
    check = run_check_mentions()
    return {"repaired_mentions": len(repaired), "post_repair": check}


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


def unresolved_clusters() -> list[dict]:
    clusters = aggregate_mentions(mention_rows())
    checks = all_rows("SELECT * FROM link_checks WHERE ok=1")
    verified = {(row["cluster_key"], row["link_kind"], row["link_value"]) for row in checks}
    out = []
    for cluster in clusters:
        has_verified = any(
            (cluster["cluster_key"], field, value) in verified
            for field in ("hf_ids", "github_repos", "official_urls", "papers")
            for value in cluster.get("links", {}).get(field) or []
        )
        if not has_verified:
            out.append(cluster)
    return out


def run_link_unresolved(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL) -> dict:
    if not artifact_path:
        run_id = new_run("link-unresolved", label="link-unresolved")
        run_root = config.STORAGE / "runs" / run_id
        packet_path = run_root / "unresolved_clusters.json"
        artifact_out = run_root / "link-unresolved_artifact.json"
        atomic_write_json(packet_path, {"clusters": unresolved_clusters()})
        prompt = render_prompt("link-unresolved", {
            "run_id": run_id,
            "unresolved_clusters_path": str(packet_path),
            "artifact_path": str(artifact_out),
            "planner_model": planner_model,
        })
        spawn = spawn_claude(run_id, prompt, model=planner_model)
        if spawn["exit_code"] != 0 or not artifact_out.exists():
            raise click.ClickException(f"link-unresolved failed; logs at {spawn['log_dir']}")
        artifact_path = str(artifact_out)
    artifact = read_json(artifact_path)
    links = artifact.get("links") if isinstance(artifact, dict) else []
    if not isinstance(links, list):
        links = []
    mentions = mention_rows()
    by_cluster = {cluster["cluster_key"]: cluster for cluster in aggregate_mentions(mentions)}
    updated_mentions = 0
    with db() as conn:
        cur = conn.cursor()
        for link_update in links:
            cluster_key = link_update.get("cluster_key")
            if cluster_key not in by_cluster:
                continue
            cluster_mentions = set(by_cluster[cluster_key].get("mention_ids") or [])
            normalized_links = (link_update.get("links") or {})
            for mention in mentions:
                if mention["id"] not in cluster_mentions:
                    continue
                merged = mention.get("links") or {}
                for field, values in normalized_links.items():
                    if field not in merged:
                        merged[field] = []
                    values = values if isinstance(values, list) else [values]
                    for value in values:
                        if value not in merged[field]:
                            merged[field].append(value)
                cur.execute("UPDATE mentions SET links_json=?, updated_at=? WHERE id=?", (dumps(merged), now(), mention["id"]))
                updated_mentions += 1
        conn.commit()
    verify = run_verify_links()
    return {"updated_mentions": updated_mentions, "verify": verify}


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
                   (id, node_key, kind, identity_json, display_name, aliases_json,
                    descriptors_json, links_json, verified_links_json, occurrence_count,
                    projection, flags_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    node["node_key"],
                    node["kind"],
                    dumps(node["identity"]),
                    node["display_name"],
                    dumps(node["aliases"]),
                    dumps(node["descriptors"]),
                    dumps(node["links"]),
                    dumps(node["verified_links"]),
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
    }

