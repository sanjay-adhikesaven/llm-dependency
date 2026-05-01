from __future__ import annotations

from pathlib import Path

import click

from . import config
from .pipeline import (
    run_audit,
    run_build_lattice,
    run_check,
    run_describe,
    run_discover,
    run_extract,
    run_verify_links,
)
from .store import all_rows, db, emit_json, loads


@click.group()
def main():
    """gdb: discover/extract/link lattice prototype."""


@main.command()
@click.option("--fresh", is_flag=True, help="Delete the SQLite DB first.")
@click.option("--yes", is_flag=True, help="Required with --fresh.")
@click.option("--I-mean-it", "i_mean_it", is_flag=True, help="Required with --fresh.")
def init(fresh: bool, yes: bool, i_mean_it: bool):
    """Create local storage and initialize SQLite."""
    if fresh:
        if not (yes and i_mean_it):
            raise click.ClickException("--fresh requires --yes and --I-mean-it")
        for path in (config.DB_PATH, Path(str(config.DB_PATH) + "-wal"), Path(str(config.DB_PATH) + "-shm")):
            path.unlink(missing_ok=True)
    config.STORAGE.mkdir(parents=True, exist_ok=True)
    with db():
        pass
    click.echo(f"storage: {config.STORAGE}")
    click.echo(f"db:      {config.DB_PATH}")


@main.command()
def summary():
    """Show table counts and open violation count."""
    tables = (
        "runs",
        "sources",
        "batches",
        "batch_artifacts",
        "mentions",
        "mention_violations",
        "link_checks",
        "entity_descriptions",
        "hf_metadata",
        "family_policies",
        "entity_relationships",
        "lattice_nodes",
        "lattice_edges",
    )
    counts = {table: all_rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] for table in tables}
    violations = all_rows("SELECT code, COUNT(*) AS n FROM mention_violations WHERE status='open' GROUP BY code")
    emit_json({"counts": counts, "open_violations": violations})


@main.group()
def run():
    """Run pipeline stages."""


@run.command("discover")
@click.option("--target", required=True)
@click.option("--artifact", "artifact_path", help="Ingest an existing discover artifact instead of launching an agent.")
@click.option("--workspace", "workspace_dir", help="Workspace holding paths referenced by --artifact.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True)
def discover_cmd(target: str, artifact_path: str | None, workspace_dir: str | None, planner_model: str, subagent_model: str):
    emit_json(run_discover(
        target=target,
        artifact_path=artifact_path,
        workspace_dir=workspace_dir,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("extract")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path", help="Ingest an existing extract artifact instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True)
@click.option("--max-workers", type=int, help="Override GDB_MAX_PARALLEL_BATCHES for this process.")
def extract_cmd(batch_id: str | None, artifact_path: str | None, planner_model: str, subagent_model: str, max_workers: int | None):
    if artifact_path and not batch_id:
        raise click.ClickException("--batch-id is required with --artifact")
    if max_workers:
        config.MAX_PARALLEL_BATCHES = max(1, max_workers)
    emit_json(run_extract(
        batch_id=batch_id,
        artifact_path=artifact_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("check")
@click.option("--artifact", "artifact_path", help="Check a JSON artifact directly instead of DB mentions.")
def check_cmd(artifact_path: str | None):
    emit_json(run_check(artifact_path=artifact_path))


@run.command("audit")
@click.option("--artifact", "artifact_path", help="Apply an audit artifact directly instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True)
def audit_cmd(artifact_path: str | None, planner_model: str, subagent_model: str):
    emit_json(run_audit(artifact_path=artifact_path, planner_model=planner_model, subagent_model=subagent_model))


@run.command("verify-links")
def verify_links_cmd():
    emit_json(run_verify_links())


@run.command("build-lattice")
def build_lattice_cmd():
    emit_json(run_build_lattice())


@run.command("describe")
@click.option("--artifact", "artifact_path", help="Apply a describe artifact directly instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, show_default=True)
def describe_cmd(artifact_path: str | None, planner_model: str, subagent_model: str):
    emit_json(run_describe(artifact_path=artifact_path, planner_model=planner_model, subagent_model=subagent_model))


@main.group()
def debug():
    """Read-only inspection helpers."""


@debug.command("mentions")
@click.option("--limit", type=int)
def debug_mentions(limit: int | None):
    sql = "SELECT * FROM mentions ORDER BY created_at, id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = all_rows(sql)
    for row in rows:
        list_fields = {
            "aliases_json",
            "subsets_json",
            "context_roles_json",
            "atoms_json",
            "links_json",
            "concept_path_json",
            "relationships_json",
            "anchors_json",
        }
        for field in (
            "identity_json",
            "descriptors_json",
            "aliases_json",
            "subsets_json",
            "context_roles_json",
            "atoms_json",
            "links_json",
            "concept_path_json",
            "aux_json",
            "relationships_json",
            "anchors_json",
            "attrs",
        ):
            row[field.removesuffix("_json")] = loads(row.pop(field), default=[] if field in list_fields else {})
    emit_json({"mentions": rows})


@debug.command("lattice")
def debug_lattice():
    emit_json({
        "nodes": all_rows("SELECT * FROM lattice_nodes ORDER BY kind, display_name"),
        "edges": all_rows("SELECT * FROM lattice_edges ORDER BY parent_node_key, child_node_key"),
    })


if __name__ == "__main__":
    main()
