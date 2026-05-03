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
        "lattice_path": "/l.json", "relations_path": "/r.json",
        "planner_model": "opus", "subagent_model": "sonnet",
    }
    for stage in ("discover", "extract", "organize", "audit", "linker",
                  "relate", "triage"):
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
    for stage in ("discover", "extract", "organize", "audit", "linker",
                  "relate", "triage", "merge", "expand"):
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


def test_run_linker_ingests_artifact_with_links(fresh_runtime, tmp_path):
    """Linker output is groups+items+links. Run attrs record link
    coverage stats."""
    from gdb.pipeline import run_linker
    from gdb.store import all_rows, loads

    linked = {
        "groups": [
            {
                "family": "Qwen3",
                "identity_keys": ["org", "collection", "size"],
                "items": [
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3-4B",
                        "identity": {"org": "Qwen", "collection": "Qwen3", "size": "4B"},
                        "aliases": ["Qwen3-4B"],
                        "links": [
                            {"kind": "hf_model", "url": "https://huggingface.co/Qwen/Qwen3-4B"},
                            {"kind": "github", "url": "https://github.com/QwenLM/Qwen3"},
                            {"kind": "paper", "url": "https://arxiv.org/abs/2509.18888"},
                        ],
                    },
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3",
                        "identity": {"org": "Qwen", "collection": "Qwen3"},
                        "aliases": ["Qwen3"],
                        "links": [
                            {"kind": "hf_collection",
                             "url": "https://huggingface.co/collections/Qwen/qwen3"},
                            {"kind": "paper", "url": "https://arxiv.org/abs/2509.18888"},
                        ],
                    },
                    {
                        "kind": "model",
                        "formal_name": "obscure/no-link-found",
                        "identity": {"org": "obscure", "collection": "no-link-found"},
                        "aliases": ["obscure-thing"],
                        "links": [],
                    },
                ],
            },
        ],
    }
    artifact_path = tmp_path / "linker.json"
    artifact_path.write_text(json.dumps(linked))

    result = run_linker(artifact_path=str(artifact_path))
    assert result["group_count"] == 1
    assert result["item_count"] == 3
    assert result["items_with_links"] == 2
    assert result["items_without_links"] == 1
    assert result["total_links"] == 5
    assert result["links_by_kind"]["hf_model"] == 1
    assert result["links_by_kind"]["hf_collection"] == 1
    assert result["links_by_kind"]["github"] == 1
    assert result["links_by_kind"]["paper"] == 2

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='linker'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["total_links"] == 5
    assert Path(attrs["artifact_path"]).exists()


def test_run_linker_without_lattice_run_raises(fresh_runtime):
    """Linker needs a prior organize / audit / linker run."""
    import click
    import pytest

    from gdb.pipeline import run_linker

    with pytest.raises(click.ClickException):
        run_linker()


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


# ---------------------------------------------------------------------------
# relate / triage / merge / expand tests (ingestion paths only — no LLM spawn)
# ---------------------------------------------------------------------------


def _well_formed_relate_artifact() -> dict:
    return {
        "batch_id": "b1",
        "batch_label": "olmo-3-base",
        "operations": [
            {
                "id": "op-001",
                "description": "OLMo-3 7B Base pretraining (stage 1) on dolma3-mix; PDF text via olmOCR.",
                "evidence": "Pretrained on dolma3 mix; PDFs OCR'd by olmOCR.",
                "source_path": "olmo-3-7b-base.md",
                "source_line": 5,
                "provenance_kind": "hf_card_body",
            },
        ],
        "relations": [
            {
                "operation_id": "op-001",
                "subject": "allenai/Olmo-3-7B-Base",
                "subject_in_lattice": True,
                "relation": "trained_on",
                "direction": "DIRECT",
                "object_ref": "allenai/dolma3-mix",
                "object_in_lattice": True,
                "object_text": None,
                "object_value": None,
                "object_unit": None,
                "description": "stage 1 pretraining mixture",
                "evidence": "trained on allenai/dolma3-mix",
                "source_path": "olmo-3-7b-base.md",
                "source_line": 12,
                "provenance_kind": "hf_frontmatter",
            },
            {
                "operation_id": "op-001",
                "subject": "allenai/Olmo-3-7B-Base",
                "subject_in_lattice": True,
                "relation": "transformed_by",
                "direction": "DIRECT",
                "object_ref": "allenai/olmOCR-7B-0225",
                "object_in_lattice": True,
                "object_text": None,
                "object_value": None,
                "object_unit": None,
                "description": "PDF pages OCR'd before tokenization",
                "evidence": "We use olmOCR (Poznanski et al., 2025a,b)",
                "source_path": "paper.pdf",
                "source_line": None,
                "provenance_kind": "paper_prose",
            },
        ],
    }


def test_run_relate_ingests_edges(fresh_runtime, tmp_path):
    """Standalone-ingest path validates shape and registers a per-batch
    artifact row. No LLM spawn."""
    from gdb.pipeline import run_relate
    from gdb.store import all_rows, db, dumps, now

    # Set up a batch row so the per-batch artifact registration succeeds.
    with db() as conn:
        conn.execute(
            "INSERT INTO batches (id, label, content_fingerprint, attrs, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("b1", "olmo-3-base", "fp1", dumps({}), now(), now()),
        )
        conn.commit()

    artifact_path = tmp_path / "relate.json"
    artifact_path.write_text(json.dumps(_well_formed_relate_artifact()))

    result = run_relate(batch_id="b1", artifact_path=str(artifact_path))
    assert result["status"] == "complete"
    assert result["operation_count"] == 1
    assert result["relation_count"] == 2
    assert result["off_lattice_object_count"] == 0

    rows = all_rows(
        "SELECT batch_id, stage, status, attrs FROM batch_artifacts "
        "WHERE batch_id='b1' AND stage='relate'"
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"


def test_relate_subject_must_be_lattice_formal_name(fresh_runtime):
    """When validation is given the lattice formal-name set, subjects
    that don't match must raise."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    artifact = _well_formed_relate_artifact()
    # Lattice has only one formal_name; the second relation's subject
    # is the same name, so both pass. Now restrict to a smaller set —
    # the relations should fail.
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(
            artifact,
            lattice_formal_names={"some/other-model"},
        )

    # And `subject_in_lattice` must be true.
    bad = json.loads(json.dumps(artifact))
    bad["relations"][0]["subject_in_lattice"] = False
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(bad)


def test_relate_allows_coined_relations_and_tracks_them(fresh_runtime):
    """`relation` is open vocabulary: snake_case labels outside the
    canonical set are allowed and counted as coined."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["relations"][0]["relation"] = "merged_from"
    stats = _validate_relate_artifact(artifact)
    assert stats["coined_relations"] == {"merged_from": 1}
    assert stats["relation_count"] == 2


def test_relate_rejects_malformed_relation_label(fresh_runtime):
    """Empty / whitespace / overly long labels still raise."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["relations"][0]["relation"] = ""
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(artifact)

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["relations"][0]["relation"] = "training data filter"  # spaces
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(artifact)


def test_relate_allows_coined_provenance_kinds(fresh_runtime):
    """`provenance_kind` is also open vocabulary."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["relations"][0]["provenance_kind"] = "wandb_log"
    stats = _validate_relate_artifact(artifact)
    assert stats["coined_provenance_kinds"] == {"wandb_log": 1}


def test_relate_requires_operations_array(fresh_runtime):
    """The operations[] list is required even when empty edges have no
    operation_id."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    del bad["operations"]
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(bad)


def test_relate_rejects_dangling_operation_id(fresh_runtime):
    """A relation pointing at an unknown operation_id must raise."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["relations"][0]["operation_id"] = "op-does-not-exist"
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(bad)


def test_relate_allows_null_operation_id_for_literals(fresh_runtime):
    """STRUCTURAL literal-value edges and INDIRECT eval edges may have
    operation_id=null."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["relations"].append({
        "operation_id": None,
        "subject": "allenai/Olmo-3-7B-Base",
        "subject_in_lattice": True,
        "relation": "size",
        "direction": "STRUCTURAL",
        "object_ref": None,
        "object_in_lattice": False,
        "object_text": None,
        "object_value": 102014,
        "object_unit": "prompts",
        "description": "total prompt count",
        "evidence": "Total Samples: 102,014",
        "source_path": "x.md",
        "source_line": 1,
        "provenance_kind": "hf_card_body",
    })
    stats = _validate_relate_artifact(artifact)
    assert stats["operation_count"] == 1
    assert stats["relation_count"] == 3
    assert stats["off_lattice_object_count"] == 1
    assert stats["coined_relations"] == {}
    assert stats["coined_provenance_kinds"] == {}


def test_run_triage_ingests_classification(fresh_runtime, tmp_path):
    from gdb.pipeline import run_triage
    from gdb.store import all_rows, loads

    triage_artifact = {
        "auto_expand": [
            {"formal_name": "allenai/dolma3-mix", "kind": "dataset",
             "primary_link": "https://huggingface.co/datasets/allenai/dolma3-mix",
             "rationale": "trained_on at stage 1",
             "motivating_relations": ["trained_on"]},
        ],
        "decline": [
            {"formal_name": "Qwen/Qwen3-7B-Instruct", "kind": "model",
             "primary_link": "https://huggingface.co/Qwen/Qwen3-7B-Instruct",
             "rationale": "evaluated against, not trained from",
             "motivating_relations": ["used_for_evaluation"]},
        ],
        "manual": [],
    }
    artifact_path = tmp_path / "triage.json"
    artifact_path.write_text(json.dumps(triage_artifact))

    result = run_triage(artifact_path=str(artifact_path))
    assert result["auto_expand"] == 1
    assert result["decline"] == 1
    assert result["manual"] == 0

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='triage'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["auto_expand_count"] == 1
    assert attrs["decline_count"] == 1
    assert attrs["manual_count"] == 0


def test_triage_rejects_missing_bucket(fresh_runtime, tmp_path):
    import click
    import pytest

    from gdb.pipeline import run_triage

    bad = tmp_path / "triage.json"
    bad.write_text(json.dumps({"auto_expand": [], "decline": []}))  # no manual
    with pytest.raises(click.ClickException):
        run_triage(artifact_path=str(bad))


def test_run_merge_unifies_items_and_edges(fresh_runtime, tmp_path):
    """Two lattices and two relation files merge by (formal_name,
    primary_link) for items and (subject, relation, object) for
    edges. Aliases and provenance accumulate."""
    from gdb.pipeline import run_merge

    lattice_a = {
        "groups": [{
            "family": "Olmo3",
            "identity_keys": ["org", "collection", "size"],
            "items": [{
                "kind": "model",
                "formal_name": "allenai/Olmo-3-7B-Base",
                "identity": {"org": "allenai", "collection": "Olmo3",
                             "size": "7B"},
                "aliases": ["Olmo-3-7B-Base"],
                "links": [{"kind": "hf_model",
                           "url": "https://huggingface.co/allenai/Olmo-3-7B-Base"}],
            }],
        }],
    }
    lattice_b = {
        "groups": [{
            "family": "Olmo3",
            "identity_keys": ["org", "collection", "stage"],
            "items": [{
                "kind": "model",
                "formal_name": "allenai/Olmo-3-7B-Base",
                "identity": {"org": "allenai", "collection": "Olmo3",
                             "stage": "Base"},
                "aliases": ["allenai/Olmo-3-7B-Base"],
                "links": [{"kind": "hf_model",
                           "url": "https://huggingface.co/allenai/Olmo-3-7B-Base"},
                          {"kind": "github",
                           "url": "https://github.com/allenai/OLMo"}],
            }],
        }],
    }
    relations_a = {
        "relations": [{
            "subject": "allenai/Olmo-3-7B-Base",
            "subject_in_lattice": True,
            "relation": "trained_on",
            "direction": "DIRECT",
            "object_ref": "allenai/dolma3-mix",
            "object_in_lattice": True,
            "object_text": None,
            "description": "stage-1 mix",
            "evidence": "card-1",
            "source_path": "a.md",
            "source_line": 1,
            "provenance_kind": "hf_frontmatter",
        }],
    }
    relations_b = {
        "relations": [{
            "subject": "allenai/Olmo-3-7B-Base",
            "subject_in_lattice": True,
            "relation": "trained_on",
            "direction": "DIRECT",
            "object_ref": "allenai/dolma3-mix",
            "object_in_lattice": True,
            "object_text": None,
            "description": "pretraining corpus, ~6T tokens",
            "evidence": "card-2",
            "source_path": "b.md",
            "source_line": 1,
            "provenance_kind": "hf_card_body",
        }],
    }
    la = tmp_path / "la.json"; la.write_text(json.dumps(lattice_a))
    lb = tmp_path / "lb.json"; lb.write_text(json.dumps(lattice_b))
    ra = tmp_path / "ra.json"; ra.write_text(json.dumps(relations_a))
    rb = tmp_path / "rb.json"; rb.write_text(json.dumps(relations_b))

    result = run_merge(sources=[str(la), str(lb)],
                       relations_sources=[str(ra), str(rb)])
    assert result["group_count"] == 1
    assert result["item_count"] == 1
    assert result["relation_count"] == 1
    # The two descriptions differ → conflict surfaced.
    assert result["conflict_count"] >= 1

    artifact = json.loads(Path(result["artifact_path"]).read_text())
    item = artifact["lattice"]["groups"][0]["items"][0]
    assert set(item["aliases"]) == {"Olmo-3-7B-Base", "allenai/Olmo-3-7B-Base"}
    assert len(item["links"]) == 2
    # identity_keys union: org, collection, size, stage all present.
    keys = artifact["lattice"]["groups"][0]["identity_keys"]
    assert set(keys) == {"org", "collection", "size", "stage"}

    edge = artifact["relations"][0]
    assert len(edge["provenance"]) == 2

    # Description-variant conflict surfaces with both values.
    desc_conflicts = [c for c in artifact["conflicts"]
                      if c.get("kind") == "description_variant"]
    assert desc_conflicts
    assert "stage-1 mix" in desc_conflicts[0]["variants"]


def test_run_merge_surfaces_identity_conflicts(fresh_runtime, tmp_path):
    """Same item from two runs with conflicting identity values for
    the same key → surfaced in conflicts[]."""
    from gdb.pipeline import run_merge

    lattice_a = {
        "groups": [{
            "family": "Olmo3", "identity_keys": ["size"],
            "items": [{"kind": "model",
                       "formal_name": "allenai/Olmo-3-7B-Base",
                       "identity": {"size": "7B"},
                       "aliases": [],
                       "links": [{"kind": "hf_model",
                                  "url": "https://huggingface.co/allenai/Olmo-3-7B-Base"}]}],
        }],
    }
    lattice_b = {
        "groups": [{
            "family": "Olmo3", "identity_keys": ["size"],
            "items": [{"kind": "model",
                       "formal_name": "allenai/Olmo-3-7B-Base",
                       "identity": {"size": "32B"},  # conflicting!
                       "aliases": [],
                       "links": [{"kind": "hf_model",
                                  "url": "https://huggingface.co/allenai/Olmo-3-7B-Base"}]}],
        }],
    }
    la = tmp_path / "la.json"; la.write_text(json.dumps(lattice_a))
    lb = tmp_path / "lb.json"; lb.write_text(json.dumps(lattice_b))

    result = run_merge(sources=[str(la), str(lb)])
    assert result["conflict_count"] >= 1

    artifact = json.loads(Path(result["artifact_path"]).read_text())
    identity_conflicts = [c for c in artifact["conflicts"]
                          if c.get("kind") == "identity_value"]
    assert identity_conflicts
    assert set(identity_conflicts[0]["values"]) == {"7B", "32B"}


def test_run_merge_requires_sources(fresh_runtime):
    """Merge without --sources must raise."""
    import click
    import pytest

    from gdb.pipeline import run_merge

    with pytest.raises(click.ClickException):
        run_merge()
