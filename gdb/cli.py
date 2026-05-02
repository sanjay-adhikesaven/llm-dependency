from __future__ import annotations

from pathlib import Path

import click

from . import config
from .pipeline import names_packet, run_audit, run_discover, run_extract, run_organize
from .store import all_rows, db, emit_json, loads, read_json


@click.group()
def main():
    """gdb: discover → extract names → organize lattice."""


@main.command()
@click.option("--fresh", is_flag=True, help="Delete the SQLite DB first.")
@click.option("--yes", is_flag=True, help="Required with --fresh.")
@click.option("--I-mean-it", "i_mean_it", is_flag=True, help="Required with --fresh.")
def init(fresh: bool, yes: bool, i_mean_it: bool):
    """Create local storage and initialize SQLite."""
    if fresh:
        if not (yes and i_mean_it):
            raise click.ClickException("--fresh requires --yes and --I-mean-it")
        for path in (config.DB_PATH,
                     Path(str(config.DB_PATH) + "-wal"),
                     Path(str(config.DB_PATH) + "-shm")):
            path.unlink(missing_ok=True)
    config.STORAGE.mkdir(parents=True, exist_ok=True)
    with db():
        pass
    click.echo(f"storage: {config.STORAGE}")
    click.echo(f"db:      {config.DB_PATH}")


@main.command()
def summary():
    """Show table counts."""
    tables = ("runs", "sources", "batches", "batch_artifacts", "names")
    counts = {table: all_rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] for table in tables}
    emit_json({"counts": counts})


@main.group()
def run():
    """Run pipeline stages."""


@run.command("discover")
@click.option("--target", required=True)
@click.option("--artifact", "artifact_path",
              help="Ingest an existing discover artifact instead of launching an agent.")
@click.option("--workspace", "workspace_dir",
              help="Workspace holding paths referenced by --artifact.")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
def discover_cmd(target: str, artifact_path: str | None, workspace_dir: str | None,
                 planner_model: str, subagent_model: str):
    emit_json(run_discover(
        target=target,
        artifact_path=artifact_path,
        workspace_dir=workspace_dir,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("extract")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing extract artifact instead of launching an agent.")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--max-workers", type=int,
              help="Override GDB_MAX_PARALLEL_BATCHES for this process.")
def extract_cmd(batch_id: str | None, artifact_path: str | None,
                planner_model: str, subagent_model: str, max_workers: int | None):
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


@run.command("organize")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing organize artifact instead of launching an agent.")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
def organize_cmd(artifact_path: str | None, planner_model: str, subagent_model: str):
    emit_json(run_organize(
        artifact_path=artifact_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("audit")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing audit artifact instead of launching an agent.")
@click.option("--source", "source_path",
              help="Audit a specific lattice artifact (default: most recent organize or audit).")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
def audit_cmd(artifact_path: str | None, source_path: str | None,
              planner_model: str, subagent_model: str):
    """Read the latest lattice artifact, revise it, write the result."""
    emit_json(run_audit(
        artifact_path=artifact_path,
        source_path=source_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@main.group()
def debug():
    """Read-only inspection helpers."""


@debug.command("names")
@click.option("--limit", type=int)
@click.option("--kind", type=click.Choice(["model", "dataset"]))
def debug_names(limit: int | None, kind: str | None):
    """List collected names from extract."""
    sql = "SELECT * FROM names"
    params: tuple = ()
    if kind:
        sql += " WHERE kind = ?"
        params = (kind,)
    sql += " ORDER BY kind, name"
    if limit:
        sql += f" LIMIT {int(limit)}"
    emit_json({"names": all_rows(sql, params)})


@debug.command("names-packet")
def debug_names_packet():
    """Show the deduped (type, name) packet that organize will read.
    Useful for sanity-checking how many distinct names exist before
    spending an organize call."""
    emit_json(names_packet())


@debug.command("organize")
@click.option("--latest/--all", default=True,
              help="Show only the most recent organize run (default) or all of them.")
def debug_organize(latest: bool):
    """Show the groups+items artifact(s) the organize stage produced.

    The artifact lives on disk; the run row's `attrs.artifact_path`
    points at it. We read the file at display time so consumers get
    the current contents, not a stale DB snapshot.
    """
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='organize' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "artifact_path": path,
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"organize_runs": out})


@debug.command("audit")
@click.option("--latest/--all", default=True,
              help="Show only the most recent audit run (default) or all of them.")
def debug_audit(latest: bool):
    """Show the revised lattice produced by audit (same shape as organize)."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='audit' AND ended_at IS NOT NULL "
        "ORDER BY started_at DESC"
    )
    if latest:
        rows = rows[:1]
    out = []
    for row in rows:
        attrs = loads(row["attrs"], default={})
        path = attrs.get("artifact_path")
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "run_id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "source_artifact_path": attrs.get("source_artifact_path"),
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "notes": attrs.get("notes"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"audit_runs": out})


if __name__ == "__main__":
    main()
