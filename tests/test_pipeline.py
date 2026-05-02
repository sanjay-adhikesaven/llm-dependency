"""End-to-end tests for the discover → extract → organize pipeline.

The actual planner spawns are exercised via `--artifact` ingestion
paths so the tests don't need the claude CLI.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_render_prompts_have_no_unfilled_placeholders(fresh_runtime):
    from gdb.pipeline import render_prompt

    common = {
        "run_id": "r", "target": "T",
        "workspace_dir": "/w", "worker_dir": "/w/workers",
        "artifact_path": "/a.json", "input_path": "/i.json",
        "names_path": "/n.json", "batch_id": "b",
        "batch_dir": "/b", "organize_path": "/o.json",
        "planner_model": "opus", "subagent_model": "sonnet",
    }
    for stage in ("discover", "extract", "organize", "audit"):
        text = render_prompt(stage, common)
        # Confirm no `{{name}}` placeholders survived.
        assert "{{" not in text, f"unfilled placeholder in {stage}: {text}"


def test_commit_names_accepts_well_formed_artifact(fresh_runtime):
    from gdb.pipeline import commit_names, new_run
    from gdb.store import all_rows

    run_id = new_run("extract", label="t")
    artifact = {
        "mentions": [
            {"type": "model", "name": "Qwen3-7B-Instruct"},
            {"type": "model", "name": "Qwen3-7B-Instruct"},      # exact dup, dropped
            {"type": "model", "name": "qwen3-7b-instruct"},      # different name → kept
            {"type": "dataset", "name": "MMLU-Pro"},
            {"type": "dataset", "name": "Qwen3-7B-Instruct"},    # different kind → kept
            {"type": "license", "name": "Apache-2.0"},           # invalid kind → dropped
            {"type": "model", "name": ""},                       # empty name → dropped
            {"name": "no_kind"},                                  # missing kind → dropped
            "not_a_dict",                                         # wrong shape → dropped
        ]
    }
    result = commit_names(artifact, run_id=run_id)
    assert result["status"] == "complete"
    assert result["names_committed"] == 4
    assert result["names_skipped"] == 5

    rows = all_rows("SELECT kind, name FROM names ORDER BY kind, name")
    pairs = {(r["kind"], r["name"]) for r in rows}
    assert pairs == {
        ("dataset", "MMLU-Pro"),
        ("dataset", "Qwen3-7B-Instruct"),
        ("model", "Qwen3-7B-Instruct"),
        ("model", "qwen3-7b-instruct"),
    }


def test_commit_names_fails_cleanly_on_invalid_artifact(fresh_runtime):
    from gdb.pipeline import commit_names

    assert commit_names({})["status"] == "failed"
    assert commit_names({"mentions": "not-a-list"})["status"] == "failed"
    assert commit_names("not-a-dict")["status"] == "failed"


def test_names_packet_dedups_kind_and_name_only(fresh_runtime):
    """The packet that organize reads is a deduped (type, name) list.
    Counts are intentionally absent — they don't change organize's
    decision about which surfaces collapse to the same entity."""
    from gdb.pipeline import commit_names, names_packet, new_run

    run_a = new_run("extract", label="a")
    run_b = new_run("extract", label="b")
    commit_names({"mentions": [
        {"type": "model", "name": "Qwen3-7B-Instruct"},
        {"type": "model", "name": "Qwen3 7B Instruct"},
        {"type": "model", "name": "OLMo-3-7B"},
    ]}, run_id=run_a)
    commit_names({"mentions": [
        # second batch, same Qwen3-7B-Instruct again — must collapse
        # to one packet entry, not two.
        {"type": "model", "name": "Qwen3-7B-Instruct"},
        {"type": "dataset", "name": "MMLU-Pro"},
    ]}, run_id=run_b)

    packet = names_packet()
    pairs = {(n["type"], n["name"]) for n in packet["names"]}
    assert pairs == {
        ("model", "Qwen3-7B-Instruct"),
        ("model", "Qwen3 7B Instruct"),
        ("model", "OLMo-3-7B"),
        ("dataset", "MMLU-Pro"),
    }
    # The packet entries carry exactly two fields, no occurrence count.
    for entry in packet["names"]:
        assert set(entry.keys()) == {"type", "name"}


def test_run_organize_ingests_artifact_path(fresh_runtime, tmp_path):
    """An organize run records only the artifact path + counts in the
    run row's attrs. The artifact itself stays on disk, not duplicated
    in the DB."""
    from gdb.pipeline import run_organize
    from gdb.store import all_rows, loads

    artifact = {
        "groups": [
            {
                "family": "Qwen3",
                "identity_keys": ["org", "collection", "size", "stage"],
                "items": [
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3-4B-Base",
                        "identity": {"org": "Qwen", "collection": "Qwen3",
                                     "size": "4B", "stage": "Base"},
                        "aliases": ["Qwen3-4B-Base", "qwen3-4b-base"],
                    },
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3-7B-Instruct",
                        "identity": {"org": "Qwen", "collection": "Qwen3",
                                     "size": "7B", "stage": "Instruct"},
                        "aliases": ["Qwen3-7B-Instruct", "Qwen 3 7B Instruct",
                                    "qwen3-7b-instruct"],
                    },
                ],
            },
            {
                "family": "MMLU",
                "identity_keys": ["family", "subset"],
                "items": [
                    {
                        "kind": "dataset",
                        "formal_name": "cais/mmlu",
                        "identity": {"family": "MMLU"},
                        "aliases": ["MMLU"],
                    },
                ],
            },
        ],
    }
    artifact_path = tmp_path / "organize.json"
    artifact_path.write_text(json.dumps(artifact))

    result = run_organize(artifact_path=str(artifact_path))
    assert result["group_count"] == 2
    assert result["item_count"] == 3
    assert Path(result["artifact_path"]).read_text() == artifact_path.read_text()

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='organize'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["group_count"] == 2
    assert attrs["item_count"] == 3
    assert Path(attrs["artifact_path"]).exists()


def test_run_organize_rejects_artifact_missing_required_lists(fresh_runtime, tmp_path):
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad_groups = tmp_path / "no_groups.json"
    bad_groups.write_text(json.dumps({"groups": "not-a-list"}))
    with pytest.raises(click.ClickException):
        run_organize(artifact_path=str(bad_groups))

    bad_items = tmp_path / "no_items.json"
    bad_items.write_text(json.dumps({"groups": [{"family": "X"}]}))
    with pytest.raises(click.ClickException):
        run_organize(artifact_path=str(bad_items))


def test_cli_run_help_lists_only_active_stages(fresh_runtime):
    from click.testing import CliRunner

    from gdb.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    out = result.output
    for stage in ("discover", "extract", "organize", "audit"):
        assert stage in out
    for dropped in ("describe", "verify-links", "build-lattice", "check", "fuzz"):
        assert dropped not in out


def test_run_audit_ingests_revised_lattice(fresh_runtime, tmp_path):
    """Audit emits a revised groups+items artifact (same shape as
    organize) plus optional notes. The run attrs record group_count,
    item_count, and the notes summary."""
    from gdb.pipeline import run_audit
    from gdb.store import all_rows, loads

    revised = {
        "groups": [
            {
                "family": "Qwen3",
                "identity_keys": ["org", "collection", "size", "variant"],
                "items": [
                    {"kind": "model", "formal_name": "Qwen/Qwen3-8B",
                     "identity": {"org": "Qwen", "collection": "Qwen3",
                                  "size": "8B", "variant": "no-thinking"},
                     "aliases": ["Qwen3-8B"]},
                    {"kind": "model", "formal_name": "Qwen/Qwen3-8B-Thinking",
                     "identity": {"org": "Qwen", "collection": "Qwen3",
                                  "size": "8B", "variant": "thinking"},
                     "aliases": ["qwen3-reasoning-8b"]},
                ],
            },
        ],
        "notes": "Split Qwen3-8B by reasoning variant.",
    }
    artifact_path = tmp_path / "audit.json"
    artifact_path.write_text(json.dumps(revised))

    result = run_audit(artifact_path=str(artifact_path))
    assert result["group_count"] == 1
    assert result["item_count"] == 2

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='audit'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["group_count"] == 1
    assert attrs["item_count"] == 2
    assert "Split Qwen3-8B" in (attrs.get("notes") or "")
    assert Path(attrs["artifact_path"]).exists()


def test_run_audit_rejects_non_groups_artifact(fresh_runtime, tmp_path):
    """Audit's output must validate as a groups+items artifact."""
    import click
    import pytest

    from gdb.pipeline import run_audit

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"issues": "not-a-list"}))  # old shape
    with pytest.raises(click.ClickException):
        run_audit(artifact_path=str(bad))


def test_run_audit_without_lattice_run_raises(fresh_runtime):
    """Audit needs a prior organize (or audit) run to read from."""
    import click
    import pytest

    from gdb.pipeline import run_audit

    with pytest.raises(click.ClickException):
        run_audit()


def test_cli_summary_after_init(fresh_runtime):
    from click.testing import CliRunner

    from gdb.cli import main

    runner = CliRunner()
    init = runner.invoke(main, ["init"])
    assert init.exit_code == 0

    summary = runner.invoke(main, ["summary"])
    assert summary.exit_code == 0
    counts = json.loads(summary.output)["counts"]
    # Tables exist with zero rows after a fresh init.
    assert counts == {"runs": 0, "sources": 0, "batches": 0,
                      "batch_artifacts": 0, "names": 0}
