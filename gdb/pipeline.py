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
    normalize_anchor_candidates,
    normalize_mention,
    repair_mentions,
    validate_mention_artifact,
)
from .enrich import enrich_hf_anchor
from .grouping import group_mentions_for_review, policy_from_review_updates
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
                    referent_scope, anchor_candidates_json, concept_path_json,
                    aux_json, relationships_json, evidence_json, description,
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
                    dumps(mention["anchor_candidates"]),
                    dumps(mention["concept_path"]),
                    dumps(mention["aux"]),
                    dumps(mention["relationships"]),
                    dumps(mention["evidence"]),
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
    workers = max(1, min(config.MAX_PARALLEL_BATCHES, len(batch_ids) or 1))

    def extract_one(bid: str) -> dict:
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
            "anchor_candidates": loads(row.get("anchor_candidates_json"), default=[]),
            "concept_path": loads(row.get("concept_path_json"), default=[]),
            "aux": loads(row.get("aux_json"), default={}),
            "relationships": loads(row.get("relationships_json"), default=[]),
            "evidence": loads(row["evidence_json"], default=[]),
            "description": row.get("description"),
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
                          context_roles_json=?, atoms_json=?, referent_scope=?,
                          anchor_candidates_json=?, concept_path_json=?, aux_json=?,
                          relationships_json=?, evidence_json=?, description=?,
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
                    dumps(mention["links"]),
                    dumps(mention["subsets"]),
                    dumps(mention["context_roles"]),
                    dumps(mention["atoms"]),
                    mention["referent_scope"],
                    dumps(mention["anchor_candidates"]),
                    dumps(mention["concept_path"]),
                    dumps(mention["aux"]),
                    dumps(mention["relationships"]),
                    dumps(mention["evidence"]),
                    mention.get("description"),
                    mention.get("notes"),
                    status,
                    now(),
                    mention["id"],
                ),
            )
        conn.commit()
    check = run_check_mentions()
    return {"repaired_mentions": len(repaired), "post_repair": check}


def entity_review_packet() -> dict:
    policies = all_rows("SELECT kind, root, policy_json, evidence_json FROM family_policies ORDER BY kind, root")
    metadata = {
        row["anchor_type"] + ":" + row["anchor_value"]: {
            "configs": loads(row["configs_json"], default=[]),
            "collections": loads(row["collections_json"], default=[]),
            "relationships": loads(row["relationships_json"], default=[]),
            "description": row["description"],
        }
        for row in all_rows("SELECT * FROM hf_metadata ORDER BY anchor_type, anchor_value")
    }
    groups = []
    for group in group_mentions_for_review(mention_rows(), max_group_size=40):
        packed_mentions = []
        for normalized in group["mentions"]:
            anchor_metadata = {}
            for anchor in normalized.get("anchor_candidates") or []:
                key_name = anchor["type"] + ":" + anchor["value"]
                if key_name in metadata:
                    anchor_metadata[key_name] = metadata[key_name]
            packed_mentions.append({
                "id": normalized["id"],
                "kind": normalized["kind"],
                "surface": normalized["surface"],
                "atoms": normalized["atoms"],
                "referent_scope": normalized["referent_scope"],
                "concept_path": normalized["concept_path"],
                "anchors": normalized["anchor_candidates"],
                "anchor_metadata": anchor_metadata,
                "aux": normalized["aux"],
                "relationships": normalized["relationships"],
                "context_roles": normalized["context_roles"],
                "evidence": normalized["evidence"][:2],
            })
        groups.append({
            "group_key": group["group_key"],
            "root_candidates": group["root_candidates"],
            "prefix_candidates": group["prefix_candidates"],
            "mentions": packed_mentions,
        })
    return {
        "family_policies": [
            {
                "kind": row["kind"],
                "root": row["root"],
                "policy": loads(row["policy_json"], default={}),
                "evidence": loads(row["evidence_json"], default=[]),
            }
            for row in policies
        ],
        "groups": groups,
        "update_schema": {
            "updates": [{
                "mention_id": "...",
                "referent_scope": "entity|concept|ambiguous",
                "concept_path": ["Family", "subfamily-or-umbrella"],
                "anchors": [{"type": "hf_model", "value": "org/repo", "exact": True}],
                "aux": {"release_size": "100B"},
                "description": "...",
            }]
        },
    }


def upsert_family_policies_from_review(review_artifact: dict) -> int:
    updates = review_artifact.get("updates") if isinstance(review_artifact, dict) else []
    if not isinstance(updates, list):
        return 0
    policies = policy_from_review_updates(updates)
    with db() as conn:
        cur = conn.cursor()
        count = 0
        for policy in policies:
            existing = cur.execute(
                "SELECT id, policy_json, evidence_json FROM family_policies WHERE kind=? AND lower(root)=?",
                (policy["kind"], policy["root"].casefold()),
            ).fetchone()
            if existing:
                current_policy = loads(existing["policy_json"], default={})
                current_paths = current_policy.get("known_paths") if isinstance(current_policy, dict) else []
                for path in policy["policy"].get("known_paths") or []:
                    if path not in current_paths:
                        current_paths.append(path)
                current_policy["known_paths"] = current_paths
                evidence = loads(existing["evidence_json"], default=[])
                evidence.extend(policy.get("evidence") or [])
                cur.execute(
                    "UPDATE family_policies SET policy_json=?, evidence_json=?, updated_at=? WHERE id=?",
                    (dumps(current_policy), dumps(evidence), now(), existing["id"]),
                )
            else:
                cur.execute(
                    """INSERT INTO family_policies
                       (id, kind, root, policy_json, evidence_json, source, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (new_id(), policy["kind"], policy["root"], dumps(policy["policy"]), dumps(policy.get("evidence") or []), "review", now(), now()),
                )
            count += 1
        conn.commit()
    return count


def run_review_entities(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL) -> dict:
    if not artifact_path:
        run_id = new_run("review-entities", label="review-entities")
        run_root = config.STORAGE / "runs" / run_id
        packet_path = run_root / "entity_review_packet.json"
        artifact_out = run_root / "review-entities_artifact.json"
        packet = entity_review_packet()
        atomic_write_json(packet_path, packet)
        groups = packet.get("groups") or []
        workers = max(1, min(config.MAX_PARALLEL_BATCHES, (len(groups) + config.MAX_REVIEW_GROUPS_PER_WORKER - 1) // config.MAX_REVIEW_GROUPS_PER_WORKER or 1))
        if workers == 1:
            prompt = render_prompt("review-entities", {
                "run_id": run_id,
                "review_packet_path": str(packet_path),
                "artifact_path": str(artifact_out),
                "planner_model": planner_model,
                "subagent_model": planner_model,
            })
            spawn = spawn_claude(run_id, prompt, model=planner_model)
            if spawn["exit_code"] != 0 or not artifact_out.exists():
                raise click.ClickException(f"review-entities failed; logs at {spawn['log_dir']}")
        else:
            chunk_paths = []
            for idx in range(workers):
                chunk = groups[idx * config.MAX_REVIEW_GROUPS_PER_WORKER:(idx + 1) * config.MAX_REVIEW_GROUPS_PER_WORKER]
                if not chunk:
                    continue
                chunk_path = run_root / f"entity_review_packet_{idx + 1}.json"
                chunk_artifact = run_root / f"review-entities_artifact_{idx + 1}.json"
                atomic_write_json(chunk_path, {**packet, "groups": chunk})
                chunk_paths.append((idx + 1, chunk_path, chunk_artifact))

            def review_chunk(item: tuple[int, Path, Path]) -> dict:
                idx, chunk_path, chunk_artifact = item
                child_run_id = new_run("review-entities", label=f"review-entities:{idx}", parent_run_id=run_id)
                prompt = render_prompt("review-entities", {
                    "run_id": child_run_id,
                    "review_packet_path": str(chunk_path),
                    "artifact_path": str(chunk_artifact),
                    "planner_model": planner_model,
                    "subagent_model": planner_model,
                })
                spawn = spawn_claude(child_run_id, prompt, model=planner_model)
                if spawn["exit_code"] != 0 or not chunk_artifact.exists():
                    return {"status": "failed", "run_id": child_run_id, "log_dir": spawn["log_dir"]}
                return {"status": "complete", "run_id": child_run_id, "artifact": read_json(chunk_artifact)}

            results = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(review_chunk, item) for item in chunk_paths]
                for future in as_completed(futures):
                    results.append(future.result())
            failed = [r for r in results if r.get("status") != "complete"]
            if failed:
                raise click.ClickException(f"review-entities failed in {len(failed)} worker(s); first logs at {failed[0].get('log_dir')}")
            merged = {"updates": []}
            for result in results:
                artifact = result.get("artifact") or {}
                updates = artifact.get("updates") if isinstance(artifact, dict) else []
                if isinstance(updates, list):
                    merged["updates"].extend(updates)
            atomic_write_json(artifact_out, merged)
            close_run(run_id, {"runtime": "claude-parallel", "workers": workers, "artifact_path": str(artifact_out)})
        artifact_path = str(artifact_out)
    review_artifact = read_json(artifact_path)
    policy_count = upsert_family_policies_from_review(review_artifact)
    reviewed = repair_mentions(mention_rows(), review_artifact)
    with db() as conn:
        cur = conn.cursor()
        for mention in reviewed:
            cur.execute(
                """UPDATE mentions
                      SET identity_json=?, identity_key=?, atoms_json=?, referent_scope=?,
                          anchor_candidates_json=?, concept_path_json=?, aux_json=?,
                          relationships_json=?, description=?, links_json=?, status=?,
                          updated_at=?
                    WHERE id=?""",
                (
                    dumps(mention["identity"]),
                    mention["identity_key"],
                    dumps(mention["atoms"]),
                    mention["referent_scope"],
                    dumps(mention["anchor_candidates"]),
                    dumps(mention["concept_path"]),
                    dumps(mention["aux"]),
                    dumps(mention["relationships"]),
                    mention.get("description"),
                    dumps(mention["links"]),
                    mention.get("status") or "repaired",
                    now(),
                    mention["id"],
                ),
            )
        conn.commit()
    check = run_check_mentions()
    return {"reviewed_mentions": len(reviewed), "family_policies_upserted": policy_count, "post_review": check}


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


def all_hf_anchors_from_mentions() -> list[dict]:
    anchors: dict[tuple[str, str], dict] = {}
    for mention in mention_rows():
        for anchor in normalize_mention(mention).get("anchor_candidates") or []:
            if anchor.get("type") in {"hf_model", "hf_dataset", "hf_dataset_config"}:
                anchors[(anchor["type"], anchor["value"])] = anchor
    return [anchors[key] for key in sorted(anchors)]


def run_investigate_hf() -> dict:
    anchors = all_hf_anchors_from_mentions()
    results = [enrich_hf_anchor(anchor) for anchor in anchors]
    with db() as conn:
        cur = conn.cursor()
        for result in results:
            anchor = result["anchor"]
            anchor_key = f"{anchor['type']}:{anchor['value']}"
            kind = "dataset" if anchor["type"] in {"hf_dataset", "hf_dataset_config"} else "model"
            cur.execute(
                """INSERT OR REPLACE INTO hf_metadata
                   (anchor_key, anchor_type, anchor_value, kind, ok, repo_url,
                    readme_url, api_url, metadata_json, card_data_json,
                    configs_json, collections_json, relationships_json,
                    description, error, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    anchor_key,
                    anchor["type"],
                    anchor["value"],
                    kind,
                    1 if result.get("ok") else 0,
                    result.get("repo_url"),
                    result.get("readme_url"),
                    result.get("api_url"),
                    dumps(result.get("metadata") or {}),
                    dumps(result.get("card_data") or {}),
                    dumps(result.get("configs") or []),
                    dumps(result.get("collections") or []),
                    dumps(result.get("relationships") or []),
                    result.get("description") or "",
                    result.get("error") or None,
                    now(),
                ),
            )
        conn.commit()
    applied = apply_hf_metadata_to_mentions()
    return {
        "anchor_count": len(anchors),
        "metadata_count": len(results),
        "ok_count": sum(1 for r in results if r.get("ok")),
        "applied_updates": applied,
    }


def apply_hf_metadata_to_mentions() -> int:
    rows = all_rows("SELECT * FROM hf_metadata")
    metadata_by_anchor = {row["anchor_key"]: row for row in rows}
    updated = 0
    with db() as conn:
        cur = conn.cursor()
        for mention in mention_rows():
            normalized = normalize_mention(mention)
            aux = dict(normalized.get("aux") or {})
            relationships = list(normalized.get("relationships") or [])
            description = normalized.get("description")
            changed = False
            for anchor in normalized.get("anchor_candidates") or []:
                row = metadata_by_anchor.get(f"{anchor['type']}:{anchor['value']}")
                if not row:
                    continue
                meta = loads(row["metadata_json"], default={})
                front = meta.get("front_matter") if isinstance(meta.get("front_matter"), dict) else {}
                config_valid = meta.get("config_valid")
                if config_valid is not None:
                    aux["hf_config_valid"] = bool(config_valid)
                    changed = True
                if front.get("base_model"):
                    aux["base_model"] = front["base_model"]
                    changed = True
                for rel in loads(row["relationships_json"], default=[]):
                    if rel not in relationships:
                        relationships.append(rel)
                        changed = True
                if not description and row["description"]:
                    description = row["description"]
                    changed = True
            if changed:
                cur.execute(
                    """UPDATE mentions
                          SET aux_json=?, relationships_json=?, description=?, updated_at=?
                        WHERE id=?""",
                    (dumps(aux), dumps(relationships), description, now(), normalized["id"]),
                )
                updated += 1
        conn.commit()
    return updated


def unresolved_clusters() -> list[dict]:
    mentions = [normalize_mention(m) for m in mention_rows()]
    checks = all_rows("SELECT * FROM link_checks WHERE ok=1")
    verified = {(row["link_kind"], row["link_value"]) for row in checks}
    out = []
    for group in group_mentions_for_review(mentions, max_group_size=25):
        unresolved_mentions = []
        for mention in group["mentions"]:
            anchors = [a for a in mention.get("anchor_candidates") or [] if a.get("exact")]
            if anchors and any((a["type"], a["value"]) in verified or a["type"] == "api_model_id" for a in anchors):
                continue
            unresolved_mentions.append({
                "id": mention["id"],
                "surface": mention["surface"],
                "kind": mention["kind"],
                "atoms": mention["atoms"],
                "concept_path": mention["concept_path"],
                "anchors": mention["anchor_candidates"],
                "evidence": mention["evidence"][:2],
            })
        if unresolved_mentions:
            out.append({
                "group_key": group["group_key"],
                "root_candidates": group["root_candidates"],
                "prefix_candidates": group["prefix_candidates"],
                "mentions": unresolved_mentions,
            })
    return out


def run_link_unresolved(*, artifact_path: str | None = None, planner_model: str = config.CLAUDE_MODEL) -> dict:
    if not artifact_path:
        run_id = new_run("link-unresolved", label="link-unresolved")
        run_root = config.STORAGE / "runs" / run_id
        packet_path = run_root / "unresolved_clusters.json"
        artifact_out = run_root / "link-unresolved_artifact.json"
        atomic_write_json(packet_path, {"unresolved_groups": unresolved_clusters()})
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
    updates = artifact.get("updates") or artifact.get("links") if isinstance(artifact, dict) else []
    if not isinstance(updates, list):
        updates = []
    mentions = mention_rows()
    by_id = {mention["id"]: mention for mention in mentions}
    updated_mentions = 0
    with db() as conn:
        cur = conn.cursor()
        for update in updates:
            target_id = update.get("mention_id") or update.get("id")
            if not target_id or target_id not in by_id:
                continue
            mention = normalize_mention(by_id[target_id])
            anchors = normalize_anchor_candidates(update.get("anchors") or update.get("anchor_candidates") or [], kind=mention["kind"])
            if not anchors and update.get("links"):
                patched = normalize_mention({**mention, "links": update.get("links")})
                anchors = patched["anchor_candidates"]
            if not anchors:
                continue
            merged_anchors = normalize_anchor_candidates([*mention.get("anchor_candidates", []), *anchors], kind=mention["kind"])
            cur.execute(
                "UPDATE mentions SET anchor_candidates_json=?, updated_at=? WHERE id=?",
                (dumps(merged_anchors), now(), mention["id"]),
            )
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
                   (id, node_key, kind, node_type, identity_json, concept_path_json,
                    display_name, aliases_json, descriptors_json, links_json,
                    anchors_json, verified_links_json, verified_anchors_json,
                    aux_json, description, occurrence_count, projection,
                    flags_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    dumps(node["anchors"]),
                    dumps(node["verified_links"]),
                    dumps(node["verified_anchors"]),
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


def run_build_relationships() -> dict:
    checks = all_rows("SELECT * FROM link_checks")
    lattice = build_lattice(mention_rows(), checks)
    entity_by_anchor = {}
    for node in lattice["nodes"]:
        if node.get("node_type") != "entity":
            continue
        for anchor in node.get("anchors") or []:
            entity_by_anchor[f"{anchor['type']}:{anchor['value']}"] = node["node_key"]
    rows: list[dict] = []
    for mention in mention_rows():
        normalized = normalize_mention(mention)
        source_anchor = None
        for anchor in normalized.get("anchor_candidates") or []:
            key_name = f"{anchor['type']}:{anchor['value']}"
            if key_name in entity_by_anchor:
                source_anchor = anchor
                break
        source_key = entity_by_anchor.get(f"{source_anchor['type']}:{source_anchor['value']}") if source_anchor else None
        for rel in normalized.get("relationships") or []:
            relation = rel.get("relation") or rel.get("type")
            if not relation:
                continue
            target_anchor = rel.get("target_anchor") if isinstance(rel.get("target_anchor"), dict) else {}
            target_name = rel.get("target") or rel.get("target_name")
            rows.append({
                "source_entity_key": source_key,
                "source_anchor": source_anchor or {},
                "relation": relation,
                "target_anchor": target_anchor,
                "target_name": target_name,
                "evidence": rel.get("evidence") or normalized.get("evidence") or [],
                "metadata": {"mention_id": normalized.get("id")},
            })
    for row in all_rows("SELECT * FROM hf_metadata"):
        source_anchor = {"type": row["anchor_type"], "value": row["anchor_value"], "exact": True}
        source_key = entity_by_anchor.get(f"{row['anchor_type']}:{row['anchor_value']}")
        for rel in loads(row["relationships_json"], default=[]):
            rows.append({
                "source_entity_key": source_key,
                "source_anchor": source_anchor,
                "relation": rel.get("relation") or "related_to",
                "target_anchor": rel.get("target_anchor") or {},
                "target_name": rel.get("target"),
                "evidence": rel.get("evidence") or [],
                "metadata": {"source": "hf_metadata"},
            })
    with db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM entity_relationships")
        for row in rows:
            cur.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_key, source_anchor_json, relation,
                    target_anchor_json, target_name, evidence_json,
                    metadata_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    new_id(),
                    row["source_entity_key"],
                    dumps(row["source_anchor"]),
                    row["relation"],
                    dumps(row["target_anchor"]),
                    row["target_name"],
                    dumps(row["evidence"]),
                    dumps(row["metadata"]),
                    now(),
                ),
            )
        conn.commit()
    return {"relationship_count": len(rows), "relationships": rows}


def _entity_description_for_node(node: dict, *, fetch_hf: bool = True) -> dict:
    metadata: dict = {}
    source: dict = {}
    description = node.get("description") or ""
    anchors = node.get("anchors") or []
    hf_rows = {
        row["anchor_key"]: row
        for row in all_rows("SELECT * FROM hf_metadata")
    }
    if fetch_hf:
        for anchor in anchors:
            if anchor.get("type") not in {"hf_model", "hf_dataset", "hf_dataset_config"}:
                continue
            row = hf_rows.get(f"{anchor['type']}:{anchor['value']}")
            if row:
                metadata = loads(row["metadata_json"], default={})
                source = {"repo_url": row["repo_url"], "readme_url": row["readme_url"], "api_url": row["api_url"]}
                if not description:
                    description = row["description"] or ""
            else:
                enriched = enrich_hf_anchor(anchor)
                metadata = enriched.get("metadata") or {}
                source = enriched.get("source") or {}
                if not description:
                    description = enriched.get("description") or ""
            break
    aux_parts = [f"{k}={v}" for k, v in sorted((node.get("aux") or {}).items()) if v not in (None, "", [], {})]
    if aux_parts:
        description = "; ".join([p for p in [description, "aux: " + ", ".join(aux_parts)] if p])
    if not description and anchors:
        first = anchors[0]
        description = f"{first.get('type')}:{first.get('value')}"
    return {
        "entity_key": node["node_key"],
        "kind": node["kind"],
        "display_name": node["display_name"],
        "anchors": anchors,
        "description": description,
        "metadata": metadata,
        "source": source,
    }


def run_describe_entities(*, fetch_hf: bool = True) -> dict:
    mentions = mention_rows()
    checks = all_rows("SELECT * FROM link_checks")
    lattice = build_lattice(mentions, checks)
    descriptions = [
        _entity_description_for_node(node, fetch_hf=fetch_hf)
        for node in lattice["nodes"]
        if node.get("node_type") == "entity"
    ]
    with db() as conn:
        cur = conn.cursor()
        for item in descriptions:
            existing = cur.execute("SELECT entity_key FROM entity_descriptions WHERE entity_key=?", (item["entity_key"],)).fetchone()
            if existing:
                cur.execute(
                    """UPDATE entity_descriptions
                          SET kind=?, display_name=?, anchors_json=?, description=?,
                              metadata_json=?, source_json=?, updated_at=?
                        WHERE entity_key=?""",
                    (
                        item["kind"],
                        item["display_name"],
                        dumps(item["anchors"]),
                        item["description"],
                        dumps(item["metadata"]),
                        dumps(item["source"]),
                        now(),
                        item["entity_key"],
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO entity_descriptions
                       (entity_key, kind, display_name, anchors_json, description,
                        metadata_json, source_json, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        item["entity_key"],
                        item["kind"],
                        item["display_name"],
                        dumps(item["anchors"]),
                        item["description"],
                        dumps(item["metadata"]),
                        dumps(item["source"]),
                        now(),
                        now(),
                    ),
                )
        conn.commit()
    return {"entity_count": len(descriptions), "descriptions": descriptions}
