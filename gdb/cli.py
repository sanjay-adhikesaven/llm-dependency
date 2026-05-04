from __future__ import annotations

from pathlib import Path

import click

from . import config
from .pipeline import (
    names_packet,
    run_audit,
    run_discover,
    run_expand,
    run_extract,
    run_merge,
    run_organize,
    run_relate,
    run_triage,
)
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


@run.command("relate")
@click.option("--batch-id", help="Limit to one batch. Required when --artifact is used.")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing relate artifact instead of launching an agent.")
@click.option("--lattice", "lattice_path",
              help="Lattice path (default: most recent organize / audit).")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--max-workers", type=int,
              help="Override GDB_MAX_PARALLEL_BATCHES for this process.")
def relate_cmd(batch_id: str | None, artifact_path: str | None,
               lattice_path: str | None, planner_model: str,
               subagent_model: str, max_workers: int | None):
    """Per-batch lattice-anchored relation extraction."""
    if artifact_path and not batch_id:
        raise click.ClickException("--batch-id is required with --artifact")
    if max_workers:
        config.MAX_PARALLEL_BATCHES = max(1, max_workers)
    emit_json(run_relate(
        batch_id=batch_id,
        artifact_path=artifact_path,
        lattice_path=lattice_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("triage")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing triage artifact instead of launching an agent.")
@click.option("--lattice", "lattice_path",
              help="Lattice path (default: most recent organize / audit).")
@click.option("--relations", "relations_path",
              help="Pre-aggregated relations file (default: aggregate completed relate artifacts).")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
def triage_cmd(artifact_path: str | None, lattice_path: str | None,
               relations_path: str | None,
               planner_model: str, subagent_model: str):
    """Classify upstream entity-leaves as auto_expand / decline / manual."""
    emit_json(run_triage(
        artifact_path=artifact_path,
        lattice_path=lattice_path,
        relations_path=relations_path,
        planner_model=planner_model,
        subagent_model=subagent_model,
    ))


@run.command("merge")
@click.option("--artifact", "artifact_path",
              help="Ingest an existing merge artifact for shape validation.")
@click.option("--source", "sources", multiple=True,
              help="Lattice artifact path. Pass multiple times for multiple runs.")
@click.option("--relations", "relations_sources", multiple=True,
              help="Relations artifact path. Pass multiple times.")
def merge_cmd(artifact_path: str | None, sources: tuple[str, ...],
              relations_sources: tuple[str, ...]):
    """Pure-Python cross-run merge of lattices and relations."""
    emit_json(run_merge(
        artifact_path=artifact_path,
        sources=list(sources) if sources else None,
        relations_sources=list(relations_sources) if relations_sources else None,
    ))


@run.command("expand")
@click.option("--node", required=True,
              help="Lattice formal_name to expand into a fresh discover-through-relate run.")
@click.option("--planner-model", type=click.Choice(config.PLANNER_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--subagent-model", type=click.Choice(config.SUBAGENT_CHOICES),
              default=config.CLAUDE_MODEL, show_default=True)
@click.option("--skip", multiple=True,
              type=click.Choice(["discover", "extract", "organize", "audit",
                                 "relate"]),
              help="Skip one or more stages. Pass multiple times to skip several.")
def expand_cmd(node: str, planner_model: str, subagent_model: str,
               skip: tuple[str, ...]):
    """Run the full pipeline against an upstream node as a fresh target."""
    emit_json(run_expand(
        node=node,
        planner_model=planner_model,
        subagent_model=subagent_model,
        skip=tuple(skip),
    ))


@main.command("viz")
@click.option("--port", type=int, default=8102, show_default=True,
              help="HTTP port to serve on.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address.")
def viz_cmd(port: int, host: str):
    """Serve an interactive graph viewer of the current run's lattice + relations."""
    from .viz import serve
    serve(port=port, host=host)


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


@debug.command("lattice")
@click.option("--query", "-q", help="Substring to match against formal_name or aliases (case-insensitive).")
@click.option("--kind", type=click.Choice(["model", "dataset"]),
              help="Filter by kind.")
@click.option("--family", help="Substring match against family name.")
@click.option("--include-unlinked", is_flag=True, default=False,
              help="Also surface items with no resolved link (hidden by default).")
@click.option("--unlinked-only", is_flag=True, default=False,
              help="Show ONLY items with no resolved link.")
@click.option("--source", "source_path",
              help="Search a specific lattice artifact (default: most recent organize / audit).")
@click.option("--limit", type=int, default=50, show_default=True,
              help="Max results.")
@click.option("--full", is_flag=True,
              help="Dump full item JSON instead of compact one-line summary.")
def debug_lattice(query: str | None, kind: str | None, family: str | None,
                  include_unlinked: bool, unlinked_only: bool,
                  source_path: str | None,
                  limit: int, full: bool):
    """Search the latest lattice (organize / audit) by name, kind, family.

    By default, only items with at least one verified link are shown.
    Pass --include-unlinked to also surface unresolved items, or
    --unlinked-only to see ONLY the unresolved pile.

    The compact output shows: kind, formal_name, link count, family.
    Use --full for the complete item record.
    """
    from .pipeline import _latest_lattice_artifact_path

    if source_path:
        path = Path(source_path).resolve()
    else:
        path = _latest_lattice_artifact_path()
    artifact = read_json(str(path))

    if unlinked_only and include_unlinked:
        raise click.ClickException("--include-unlinked and --unlinked-only are mutually exclusive")

    needle = (query or "").casefold()
    fam_needle = (family or "").casefold()
    matches: list[dict] = []
    for grp in artifact.get("groups") or []:
        fam_name = grp.get("family") or ""
        if fam_needle and fam_needle not in fam_name.casefold():
            continue
        for item in grp.get("items") or []:
            if kind and item.get("kind") != kind:
                continue
            has_link = bool(item.get("links") or [])
            if unlinked_only and has_link:
                continue
            if not unlinked_only and not include_unlinked and not has_link:
                continue
            if needle:
                hay = [(item.get("formal_name") or "")] + list(item.get("aliases") or [])
                if not any(needle in (s or "").casefold() for s in hay):
                    continue
            matches.append({**item, "_family": fam_name})
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    if full:
        emit_json({"lattice_path": str(path), "match_count": len(matches),
                   "matches": matches})
        return
    # Compact one-line per match.
    out_rows: list[dict] = []
    for it in matches:
        first_link = ""
        links = it.get("links") or []
        if links and isinstance(links[0], dict):
            first_link = links[0].get("url") or ""
        out_rows.append({
            "kind": it.get("kind"),
            "formal_name": it.get("formal_name"),
            "family": it.get("_family"),
            "n_links": len(links),
            "first_link": first_link,
            "description": (it.get("description") or "")[:120],
        })
    emit_json({"lattice_path": str(path), "match_count": len(matches),
               "matches": out_rows})


@debug.command("relate")
@click.option("--batch-id", help="Limit to one batch's relate artifact.")
def debug_relate(batch_id: str | None):
    """Show the per-batch relate artifacts (typed lattice-anchored edges)."""
    sql = ("SELECT batch_id, artifact_path, status, attrs, updated_at "
           "FROM batch_artifacts WHERE stage='relate'")
    params: tuple = ()
    if batch_id:
        sql += " AND batch_id=?"
        params = (batch_id,)
    sql += " ORDER BY updated_at DESC"
    out = []
    for row in all_rows(sql, params):
        attrs = loads(row["attrs"], default={})
        path = row["artifact_path"]
        artifact = None
        missing = False
        if path and Path(path).exists():
            artifact = read_json(path)
        elif path:
            missing = True
        out.append({
            "batch_id": row["batch_id"],
            "status": row["status"],
            "updated_at": row["updated_at"],
            "relation_count": attrs.get("relation_count"),
            "off_lattice_object_count": attrs.get("off_lattice_object_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"relate_artifacts": out})


@debug.command("triage")
@click.option("--latest/--all", default=True,
              help="Show only the most recent triage run (default) or all.")
def debug_triage(latest: bool):
    """Show the upstream-node classification artifact."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='triage' AND ended_at IS NOT NULL "
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
            "auto_expand_count": attrs.get("auto_expand_count"),
            "decline_count": attrs.get("decline_count"),
            "manual_count": attrs.get("manual_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"triage_runs": out})


@debug.command("merge")
@click.option("--latest/--all", default=True,
              help="Show only the most recent merge run (default) or all.")
def debug_merge(latest: bool):
    """Show the cross-run merged lattice + relations."""
    rows = all_rows(
        "SELECT id, attrs, started_at, ended_at FROM runs "
        "WHERE stage='merge' AND ended_at IS NOT NULL "
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
            "sources": attrs.get("sources"),
            "relations_sources": attrs.get("relations_sources"),
            "group_count": attrs.get("group_count"),
            "item_count": attrs.get("item_count"),
            "relation_count": attrs.get("relation_count"),
            "conflict_count": attrs.get("conflict_count"),
            "artifact_path": path,
            "artifact_missing_on_disk": missing,
            "artifact": artifact,
        })
    emit_json({"merge_runs": out})


if __name__ == "__main__":
    main()
