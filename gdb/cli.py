from __future__ import annotations

from pathlib import Path

import click

from . import config
from .pipeline import (
    run_build_lattice,
    run_check_mentions,
    run_discover_target,
    run_extract_mentions,
    run_link_unresolved,
    run_repair_mentions,
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
@click.option("--planner-model", default=config.CLAUDE_MODEL, type=click.Choice(config.PLANNER_CHOICES), show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, type=click.Choice(config.SUBAGENT_CHOICES), show_default=True)
def discover_cmd(target: str, artifact_path: str | None, workspace_dir: str | None, planner_model: str, subagent_model: str):
    emit_json(run_discover_target(
        target=target,
        artifact_path=artifact_path,
        workspace_dir=workspace_dir,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("extract-mentions")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path", help="Ingest an existing extract artifact instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, type=click.Choice(config.PLANNER_CHOICES), show_default=True)
@click.option("--subagent-model", default=config.CLAUDE_MODEL, type=click.Choice(config.SUBAGENT_CHOICES), show_default=True)
def extract_mentions_cmd(batch_id: str | None, artifact_path: str | None, planner_model: str, subagent_model: str):
    if artifact_path and not batch_id:
        raise click.ClickException("--batch-id is required with --artifact")
    emit_json(run_extract_mentions(
        batch_id=batch_id,
        artifact_path=artifact_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("check-mentions")
@click.option("--artifact", "artifact_path", help="Check a JSON artifact directly instead of DB mentions.")
def check_mentions_cmd(artifact_path: str | None):
    emit_json(run_check_mentions(artifact_path=artifact_path))


@run.command("repair-mentions")
@click.option("--artifact", "artifact_path", help="Apply a repair artifact directly instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, type=click.Choice(config.PLANNER_CHOICES), show_default=True)
def repair_mentions_cmd(artifact_path: str | None, planner_model: str):
    emit_json(run_repair_mentions(artifact_path=artifact_path, planner_model=planner_model))


@run.command("verify-links")
def verify_links_cmd():
    emit_json(run_verify_links())


@run.command("link-unresolved")
@click.option("--artifact", "artifact_path", help="Apply a link artifact directly instead of launching an agent.")
@click.option("--planner-model", default=config.CLAUDE_MODEL, type=click.Choice(config.PLANNER_CHOICES), show_default=True)
def link_unresolved_cmd(artifact_path: str | None, planner_model: str):
    emit_json(run_link_unresolved(artifact_path=artifact_path, planner_model=planner_model))


@run.command("build-lattice")
def build_lattice_cmd():
    emit_json(run_build_lattice())


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
        for field in ("identity_json", "descriptors_json", "aliases_json", "links_json", "subsets_json", "context_roles_json", "evidence_json", "attrs"):
            row[field.removesuffix("_json")] = loads(row.pop(field), default=[] if field.endswith("aliases_json") else {})
    emit_json({"mentions": rows})


@debug.command("lattice")
def debug_lattice():
    emit_json({
        "nodes": all_rows("SELECT * FROM lattice_nodes ORDER BY kind, display_name"),
        "edges": all_rows("SELECT * FROM lattice_edges ORDER BY parent_node_key, child_node_key"),
    })


if __name__ == "__main__":
    main()

