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
    for stage in ("discover", "extract", "organize", "audit",
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
                        "links": [
                            {"kind": "hf_model",
                             "url": "https://huggingface.co/Qwen/Qwen3-4B-Base"},
                        ],
                        "description": "Base variant of Qwen3-4B.",
                    },
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3-7B-Instruct",
                        "identity": {"org": "Qwen", "collection": "Qwen3",
                                     "size": "7B", "stage": "Instruct"},
                        "aliases": ["Qwen3-7B-Instruct", "Qwen 3 7B Instruct",
                                    "qwen3-7b-instruct"],
                        "links": [
                            {"kind": "hf_model",
                             "url": "https://huggingface.co/Qwen/Qwen3-7B-Instruct"},
                            {"kind": "paper",
                             "url": "https://arxiv.org/abs/2509.18888"},
                        ],
                        "description": "Instruction-tuned 7B Qwen3 model.",
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
                        "links": [
                            {"kind": "hf_dataset",
                             "url": "https://huggingface.co/datasets/cais/mmlu"},
                        ],
                        "description": "Massive Multitask Language Understanding benchmark.",
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
    assert result["items_with_links"] == 3
    assert result["total_links"] == 4
    assert result["links_by_kind"]["hf_model"] == 2
    assert result["links_by_kind"]["hf_dataset"] == 1
    assert result["links_by_kind"]["paper"] == 1
    assert Path(result["artifact_path"]).read_text() == artifact_path.read_text()

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='organize'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["group_count"] == 2
    assert attrs["item_count"] == 3
    assert attrs["items_with_links"] == 3
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


def test_run_organize_rejects_item_missing_links_field(fresh_runtime, tmp_path):
    """Each item must carry a `links` array (possibly empty). Missing
    the field outright is a structural failure."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    no_links = tmp_path / "no_links.json"
    no_links.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["org"],
            "items": [{"kind": "model", "formal_name": "X/Y",
                       "identity": {"org": "X"}, "aliases": []}],
        }],
    }))
    with pytest.raises(click.ClickException, match="links"):
        run_organize(artifact_path=str(no_links))


def test_run_organize_rejects_invalid_link_kind(fresh_runtime, tmp_path):
    """Primary link must use a closed-vocabulary kind."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad_kind = tmp_path / "bad_kind.json"
    bad_kind.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["org"],
            "items": [{
                "kind": "model", "formal_name": "X/Y",
                "identity": {"org": "X"}, "aliases": [],
                "links": [{"kind": "twitter_thread",
                           "url": "https://twitter.com/x/status/1"}],
                "description": None,
            }],
        }],
    }))
    with pytest.raises(click.ClickException, match="kind"):
        run_organize(artifact_path=str(bad_kind))


def test_run_organize_rejects_invalid_link_url(fresh_runtime, tmp_path):
    """Primary link URL must be an http(s) string."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad_url = tmp_path / "bad_url.json"
    bad_url.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["org"],
            "items": [{
                "kind": "model", "formal_name": "X/Y",
                "identity": {"org": "X"}, "aliases": [],
                "links": [{"kind": "hf_model", "url": "ftp://example/foo"}],
                "description": None,
            }],
        }],
    }))
    with pytest.raises(click.ClickException, match="url"):
        run_organize(artifact_path=str(bad_url))


def test_run_organize_allows_empty_links_array(fresh_runtime, tmp_path):
    """Items may carry `links: []` (audit will revisit before dropping)."""
    from gdb.pipeline import run_organize

    empty_links = tmp_path / "empty_links.json"
    empty_links.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["org"],
            "items": [{
                "kind": "model", "formal_name": "X/Y",
                "identity": {"org": "X"}, "aliases": [],
                "links": [],
                "description": None,
            }],
        }],
    }))
    result = run_organize(artifact_path=str(empty_links))
    assert result["group_count"] == 1
    assert result["item_count"] == 1
    assert result["items_with_links"] == 0
    assert result["items_without_links"] == 1


def test_cli_run_help_lists_only_active_stages(fresh_runtime):
    from click.testing import CliRunner

    from gdb.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    out = result.output
    for stage in ("discover", "extract", "organize", "audit",
                  "relate", "triage", "merge", "expand"):
        assert stage in out
    for dropped in ("linker", "describe", "verify-links", "build-lattice",
                    "check", "fuzz"):
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
                     "aliases": ["Qwen3-8B"],
                     "links": [{"kind": "hf_model",
                                "url": "https://huggingface.co/Qwen/Qwen3-8B"}],
                     "description": "Qwen3 8B without reasoning."},
                    {"kind": "model", "formal_name": "Qwen/Qwen3-8B-Thinking",
                     "identity": {"org": "Qwen", "collection": "Qwen3",
                                  "size": "8B", "variant": "thinking"},
                     "aliases": ["qwen3-reasoning-8b"],
                     "links": [{"kind": "hf_model",
                                "url": "https://huggingface.co/Qwen/Qwen3-8B-Thinking"}],
                     "description": "Qwen3 8B reasoning variant."},
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


# ---------------------------------------------------------------------------
# relate / triage / merge / expand tests (ingestion paths only — no LLM spawn)
# ---------------------------------------------------------------------------


def _well_formed_anchor() -> dict:
    return {
        "source": "https://arxiv.org/abs/2512.13961",
        "position": "Section 3.5, paragraph 2",
        "explanation": "Paper documents Olmo-3-7B-Base stage-1 pretraining on the dolma3-mix.",
    }


def _well_formed_relate_artifact() -> dict:
    return {
        "batch_id": "b1",
        "batch_label": "olmo-3-base",
        "operations": [
            {
                "description": "OLMo-3 7B Base stage-1 pretraining event: trained on dolma3-mix; PDF content via olmOCR.",
                "anchor_list": [_well_formed_anchor()],
                "edges": [
                    {
                        "subject": "allenai/Olmo-3-7B-Base",
                        "relation": "trained_on",
                        "dependency_kind": "direct",
                        "object": "allenai/dolma3-mix",
                        "description": "Stage-1 pretraining mixture for Olmo-3-7B-Base.",
                        "anchor_list": [_well_formed_anchor()],
                    },
                    {
                        "subject": "allenai/Olmo-3-7B-Base",
                        "relation": "transformed_by",
                        "dependency_kind": "direct",
                        "object": "allenai/olmOCR-7B-0225",
                        "description": "PDF pages OCR'd by olmOCR before tokenization.",
                        "anchor_list": [_well_formed_anchor()],
                    },
                ],
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
    assert result["edge_count"] == 2
    assert result["singleton_event_count"] == 0
    assert result["direct_count"] == 2
    assert result["indirect_count"] == 0
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
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(
            artifact,
            lattice_formal_names={"some/other-model"},
        )


def test_relate_allows_coined_relations_and_tracks_them(fresh_runtime):
    """`relation` is open vocabulary: snake_case labels outside the
    canonical set are allowed and counted as coined."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"][0]["relation"] = "merged_from"
    stats = _validate_relate_artifact(artifact)
    assert stats["coined_relations"] == {"merged_from": 1}
    assert stats["edge_count"] == 2


def test_relate_rejects_malformed_relation_label(fresh_runtime):
    """Empty / whitespace / overly long labels still raise."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"][0]["relation"] = ""
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(artifact)

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"][0]["relation"] = "training data filter"
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(artifact)


def test_relate_validates_dependency_kind(fresh_runtime):
    """`dependency_kind` is closed: direct | indirect."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"][0]["dependency_kind"] = "structural"
    with pytest.raises(click.ClickException, match="dependency_kind"):
        _validate_relate_artifact(bad)

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"][0]["dependency_kind"] = "DIRECT"  # uppercase
    with pytest.raises(click.ClickException, match="dependency_kind"):
        _validate_relate_artifact(bad)


def test_relate_requires_operations_array(fresh_runtime):
    """The operations[] list is required."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    del bad["operations"]
    with pytest.raises(click.ClickException):
        _validate_relate_artifact(bad)


def test_relate_requires_edges_in_each_event(fresh_runtime):
    """Each event MUST have a non-empty edges[] array (singleton ok)."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"] = []
    with pytest.raises(click.ClickException, match="edges"):
        _validate_relate_artifact(bad)


def test_relate_validates_anchor_list_shape(fresh_runtime):
    """anchor_list[] must be non-empty and each entry needs source +
    explanation."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    # Edge with empty anchor_list
    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"][0]["anchor_list"] = []
    with pytest.raises(click.ClickException, match="anchor_list"):
        _validate_relate_artifact(bad)

    # Anchor missing required source
    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"][0]["anchor_list"][0]["source"] = ""
    with pytest.raises(click.ClickException, match="source"):
        _validate_relate_artifact(bad)

    # Anchor missing required explanation
    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["edges"][0]["anchor_list"][0]["explanation"] = ""
    with pytest.raises(click.ClickException, match="explanation"):
        _validate_relate_artifact(bad)

    # Event-level anchor_list also required
    bad = json.loads(json.dumps(_well_formed_relate_artifact()))
    bad["operations"][0]["anchor_list"] = []
    with pytest.raises(click.ClickException, match="anchor_list"):
        _validate_relate_artifact(bad)


def test_relate_off_lattice_object_counts(fresh_runtime):
    """Edges whose object isn't in the lattice formal-names set are
    counted as off-lattice — they're still valid (free-text descriptor)."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"].append({
        "subject": "allenai/Olmo-3-7B-Base",
        "relation": "filtered_by",
        "dependency_kind": "direct",
        "object": "GPT-4 (used as classifier; no canonical model id)",
        "description": "GPT-4 served as a classifier for inclusion of training samples.",
        "anchor_list": [_well_formed_anchor()],
    })
    stats = _validate_relate_artifact(
        artifact,
        lattice_formal_names={
            "allenai/Olmo-3-7B-Base", "allenai/dolma3-mix",
            "allenai/olmOCR-7B-0225",
        },
    )
    assert stats["edge_count"] == 3
    assert stats["off_lattice_object_count"] == 1
    assert stats["direct_count"] == 3


def test_relate_singleton_event_counts(fresh_runtime):
    """Pair-only facts are valid singleton-edge events."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"].append({
        "description": "Methodology borrowing: OLMo-3 inherits SwallowMath rewriting recipe from prior work.",
        "anchor_list": [_well_formed_anchor()],
        "edges": [{
            "subject": "allenai/Olmo-3-7B-Base",
            "relation": "inspired_by",
            "dependency_kind": "indirect",
            "object": "tokyotech-llm/swallow-math",
            "description": "Olmo-3-7B-Base mid-training reuses SwallowMath's Llama-rewriting recipe with Qwen3-32B substituted as the rewriter.",
            "anchor_list": [_well_formed_anchor()],
        }],
    })
    stats = _validate_relate_artifact(artifact)
    assert stats["operation_count"] == 2
    assert stats["edge_count"] == 3
    assert stats["singleton_event_count"] == 1
    assert stats["direct_count"] == 2
    assert stats["indirect_count"] == 1


def test_assemble_relate_artifact_from_jsonl(fresh_runtime, tmp_path):
    """JSONL append target is assembled to a single relate artifact dict
    by the pipeline after the planner exits."""
    from gdb.pipeline import (assemble_relate_artifact_from_jsonl,
                              _validate_relate_artifact)

    events_path = tmp_path / "events.jsonl"
    line1 = {
        "description": "Event 1",
        "anchor_list": [_well_formed_anchor()],
        "edges": [{
            "subject": "allenai/Olmo-3-7B-Base",
            "relation": "trained_on",
            "dependency_kind": "direct",
            "object": "allenai/dolma3-mix",
            "description": "Stage-1.",
            "anchor_list": [_well_formed_anchor()],
        }],
    }
    line2 = {
        "description": "Event 2",
        "anchor_list": [_well_formed_anchor()],
        "edges": [{
            "subject": "allenai/Olmo-3-7B-Base",
            "relation": "transformed_by",
            "dependency_kind": "direct",
            "object": "allenai/olmOCR-7B-0225",
            "description": "OCR.",
            "anchor_list": [_well_formed_anchor()],
        }],
    }
    events_path.write_text(json.dumps(line1) + "\n" + json.dumps(line2) + "\n")

    artifact = assemble_relate_artifact_from_jsonl(
        events_path, batch_id="b1", batch_label="olmo-3-base",
    )
    assert artifact["batch_id"] == "b1"
    assert artifact["batch_label"] == "olmo-3-base"
    assert len(artifact["operations"]) == 2

    stats = _validate_relate_artifact(artifact)
    assert stats["operation_count"] == 2
    assert stats["edge_count"] == 2


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
        "operations": [{
            "description": "Stage-1 pretraining event",
            "anchor_list": [{"source": "card-1", "explanation": "..."}],
            "edges": [{
                "subject": "allenai/Olmo-3-7B-Base",
                "relation": "trained_on",
                "dependency_kind": "direct",
                "object": "allenai/dolma3-mix",
                "description": "stage-1 mix",
                "anchor_list": [{"source": "a.md", "position": "line 1",
                                 "explanation": "card-1"}],
            }],
        }],
    }
    relations_b = {
        "operations": [{
            "description": "Stage-1 pretraining event (per card)",
            "anchor_list": [{"source": "card-2", "explanation": "..."}],
            "edges": [{
                "subject": "allenai/Olmo-3-7B-Base",
                "relation": "trained_on",
                "dependency_kind": "direct",
                "object": "allenai/dolma3-mix",
                "description": "pretraining corpus, ~6T tokens",
                "anchor_list": [{"source": "b.md", "position": "line 1",
                                 "explanation": "card-2"}],
            }],
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
    # Both anchors stack on the merged edge
    assert len(edge["anchor_list"]) == 2

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


# ---------------------------------------------------------------------------
# subsets module — parsing + cross-check (no network)
# ---------------------------------------------------------------------------


def test_subsets_parse_yaml_configs():
    """HF YAML frontmatter `configs:` field becomes subsets[]."""
    from gdb.subsets import parse_subsets

    readme = """---
license: cc-by-4.0
configs:
  - config_name: finemath-3plus
    data_files: foo.parquet
  - config_name: finemath-4plus
    data_files: bar.parquet
---
# FineMath
"""
    subs = parse_subsets(readme)
    assert "finemath-3plus" in subs
    assert "finemath-4plus" in subs


def test_subsets_parse_components_table():
    """A markdown table under a Components / Composition heading
    contributes its first column as subset slugs."""
    from gdb.subsets import parse_subsets

    readme = """---
license: cc-by-4.0
---
# Dolma 3 Dolmino Mix (100B)

## Components

| Subset    | Tokens |
|-----------|--------|
| CraneMath | 5B     |
| CraneCode | 8B     |
| FineMath4+| 12B    |
"""
    subs = parse_subsets(readme)
    assert "cranemath" in subs
    assert "cranecode" in subs
    # `+` is preserved in subset slugs (HF allows it); cross-check
    # uses `_name_variants` to bridge `finemath4+` ↔ `finemath4plus`.
    assert any(s.startswith("finemath4") for s in subs)


def test_subsets_skip_header_cells():
    """Header rows ('Subset', 'Name', '---') are not treated as data."""
    from gdb.subsets import parse_subsets

    readme = """## Sources

| Source | Tokens |
| ------ | ------ |
| FOO    | 100M   |
| BAR    | 200M   |
"""
    subs = parse_subsets(readme)
    assert "subset" not in subs
    assert "source" not in subs
    assert "foo" in subs
    assert "bar" in subs


# ---------------------------------------------------------------------------
# subsets pre-pass — purely additive (populate + flag)
# ---------------------------------------------------------------------------


def test_organize_validates_subsets_shape(fresh_runtime, tmp_path):
    """subsets[] must be a list of non-empty strings when present."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "groups": [{
            "family": "X", "identity_keys": ["org"],
            "items": [{
                "kind": "dataset", "formal_name": "X/Y",
                "identity": {"org": "X"}, "aliases": [],
                "links": [{"kind": "hf_dataset", "url": "https://huggingface.co/datasets/X/Y"}],
                "description": None,
                "subsets": ["valid", "", "also valid"],   # empty string mid-list
            }],
        }],
    }))
    with pytest.raises(click.ClickException, match="subsets"):
        run_organize(artifact_path=str(bad))


def test_organize_rejects_empty_aliases_when_identity_is_specific(fresh_runtime, tmp_path):
    """Items with full identity (size/stage/date/etc.) MUST have ≥1 alias.
    Empty aliases means the item was invented by HF org enumeration,
    not folded from any input surface form. Hard validator error."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad = tmp_path / "phantom.json"
    bad.write_text(json.dumps({
        "groups": [{
            "family": "Qwen3", "identity_keys": ["org", "collection", "size", "stage", "date"],
            "items": [{
                "kind": "model",
                "formal_name": "Qwen/Qwen3-4B-Instruct-2507",
                "identity": {"org": "Qwen", "collection": "Qwen3",
                             "size": "4B", "stage": "Instruct", "date": "2507"},
                "aliases": [],
                "links": [{"kind": "hf_model",
                           "url": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"}],
                "description": "...",
            }],
        }],
    }))
    with pytest.raises(click.ClickException, match="empty aliases"):
        run_organize(artifact_path=str(bad))


def test_organize_allows_empty_aliases_for_family_concept_root(fresh_runtime, tmp_path):
    """A family-concept root (identity only carrying broad keys: org,
    collection, vendor, family) may have empty aliases."""
    from gdb.pipeline import run_organize

    ok = tmp_path / "concept_root.json"
    ok.write_text(json.dumps({
        "groups": [{
            "family": "Qwen3", "identity_keys": ["org", "collection", "size", "stage"],
            "items": [{
                "kind": "model",
                "formal_name": "Qwen/Qwen3",
                "identity": {"org": "Qwen", "collection": "Qwen3"},
                "aliases": [],
                "links": [{"kind": "hf_collection",
                           "url": "https://huggingface.co/collections/Qwen/qwen3"}],
                "description": "Qwen3 is the latest generation...",
            }],
        }],
    }))
    result = run_organize(artifact_path=str(ok))
    assert result["item_count"] == 1


def test_flag_audit_issues_purely_additive():
    """The flag pass MUST NOT mutate the lattice — no items moved,
    no drops restored, no renames. It only adds an `audit_hints[]`
    array."""
    from gdb.subsets import flag_audit_issues
    import json as _json

    lattice = {
        "groups": [{
            "family": "Dolma3",
            "identity_keys": ["org", "collection"],
            "items": [{
                "kind": "dataset",
                "formal_name": "allenai/dolma3_dolmino_mix-100B-1025",
                "identity": {"org": "allenai", "collection": "Dolma3-Dolmino-Mix"},
                "aliases": [],
                "links": [{"kind": "hf_dataset",
                           "url": "https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025"}],
                "description": "...",
                "subsets": ["cranemath", "cranecode"],
            }],
        }],
        "dropped": [
            {"name": "FineMath4+", "kind": "dataset", "reason": "..."},
        ],
    }
    snapshot = _json.dumps(lattice, sort_keys=True)
    counts = flag_audit_issues(lattice)

    # Lattice unchanged except for added audit_hints[] field
    lattice_minus_hints = {k: v for k, v in lattice.items() if k != "audit_hints"}
    assert _json.dumps(lattice_minus_hints, sort_keys=True) == snapshot

    # audit_hints[] populated — the dolmino_mix is foundational (broad
    # identity), so flagging item_matches_parent_subset for cranemath
    # / cranecode would only fire if those items existed in the
    # lattice already. They don't, so no item-side hints expected.
    # But there's a parent with empty aliases AND broad identity →
    # this is a valid family-concept root, NO phantom hint expected.
    hints = lattice.get("audit_hints") or []
    kinds = {h["kind"] for h in hints}
    assert "phantom_item" not in kinds  # broad identity is OK


def test_flag_item_matches_parent_subset_with_role_classification():
    """When an item's slug appears in a parent's subsets[], emit a
    hint with the item's role (canonical / soft-anchored / concept)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [
            {
                "family": "Dolma3",
                "identity_keys": ["org", "collection"],
                "items": [{
                    "kind": "dataset",
                    "formal_name": "allenai/dolma3_dolmino_pool",
                    "identity": {"org": "allenai", "collection": "Dolma3"},
                    "aliases": [],
                    "links": [{"kind": "hf_dataset",
                               "url": "https://huggingface.co/datasets/allenai/dolma3_dolmino_pool"}],
                    "description": "...",
                    "subsets": ["cranemath", "common-crawl"],
                }],
            },
            {
                "family": "Olmo 3 / Dolma 3",
                "identity_keys": ["family", "stage", "size"],
                "items": [
                    # soft-anchored sub-component — typical reshape target
                    {"kind": "dataset",
                     "formal_name": "olmo3-tr-cranemath",
                     "identity": {"family": "olmo3-tr", "stage": "tech-report-introduced",
                                  "size": "5.62B"},
                     "aliases": ["CraneMath"],
                     "links": [{"kind": "blog", "url": "https://allenai.org/olmo-3-tech-report.pdf"}],
                     "description": "...", "subsets": []},
                ],
            },
            {
                "family": "Common Crawl",
                "identity_keys": ["org", "collection"],
                "items": [
                    # concept root — broad identity AND bare-domain URL —
                    # hint should label item_role='concept', signaling
                    # "keep standalone" rather than reshape
                    {"kind": "dataset",
                     "formal_name": "Common Crawl",
                     "identity": {"org": "Common Crawl Foundation", "collection": "Common Crawl"},
                     "aliases": ["common_crawl"],
                     "links": [{"kind": "blog", "url": "https://commoncrawl.org/"}],
                     "description": "...", "subsets": []},
                ],
            },
        ],
    }

    counts = flag_audit_issues(lattice)
    hints = [h for h in (lattice.get("audit_hints") or [])
             if h["kind"] == "item_matches_parent_subset"]
    by_item = {h["item_formal_name"]: h for h in hints}

    # CraneMath flagged with item_role='soft-anchored'
    assert "olmo3-tr-cranemath" in by_item
    assert by_item["olmo3-tr-cranemath"]["item_role"] == "soft-anchored"
    assert by_item["olmo3-tr-cranemath"]["matched_parent"] == "allenai/dolma3_dolmino_pool"

    # Common Crawl flagged with item_role='concept' (broad identity OR bare-domain)
    assert "Common Crawl" in by_item
    assert by_item["Common Crawl"]["item_role"] == "concept"


def test_flag_dropped_matches_parent_subset():
    """A dropped name whose slug appears in some kept item's subsets[]
    gets flagged for restoration (but NOT auto-restored)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "FineMath",
            "identity_keys": ["org", "collection"],
            "items": [{
                "kind": "dataset",
                "formal_name": "HuggingFaceTB/finemath",
                "identity": {"org": "HuggingFaceTB", "collection": "FineMath"},
                "aliases": ["FineMath"],
                "links": [{"kind": "hf_dataset",
                           "url": "https://huggingface.co/datasets/HuggingFaceTB/finemath"}],
                "description": "...",
                "subsets": ["finemath-3plus", "finemath-4plus"],
            }],
        }],
        "dropped": [
            {"name": "FineMath4+", "kind": "dataset", "reason": "no exact match"},
            {"name": "code_fresh", "kind": "dataset", "reason": "AI2-internal"},
        ],
    }

    counts = flag_audit_issues(lattice)
    hints = [h for h in (lattice.get("audit_hints") or [])
             if h["kind"] == "dropped_matches_parent_subset"]
    assert len(hints) == 1
    assert hints[0]["dropped_name"] == "FineMath4+"
    assert hints[0]["matched_parent"] == "HuggingFaceTB/finemath"

    # dropped[] is NOT mutated — both entries still there
    assert len(lattice["dropped"]) == 2


def test_flag_cross_org_family_and_sibling_collision():
    """Two more flag types: cross-org families and identity collisions."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [
            {
                "family": "HumanEval",
                "identity_keys": ["org"],
                "items": [
                    {"kind": "dataset", "formal_name": "openai/openai_humaneval",
                     "identity": {"org": "openai"}, "aliases": ["HumanEval"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/openai/openai_humaneval"}],
                     "description": "...", "subsets": []},
                    {"kind": "dataset", "formal_name": "evalplus/humanevalplus",
                     "identity": {"org": "evalplus"}, "aliases": ["humaneval+"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/evalplus/humanevalplus"}],
                     "description": "...", "subsets": []},
                ],
            },
            {
                "family": "Reasoning-traces",
                "identity_keys": ["org", "collection"],
                "items": [
                    # Identity collision — same identity for both
                    {"kind": "dataset", "formal_name": "allenai/r1-traces",
                     "identity": {"org": "allenai", "collection": "reasoning-traces"},
                     "aliases": ["r1"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/allenai/r1-traces"}],
                     "description": "...", "subsets": []},
                    {"kind": "dataset", "formal_name": "allenai/qwq-traces",
                     "identity": {"org": "allenai", "collection": "reasoning-traces"},
                     "aliases": ["qwq"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/allenai/qwq-traces"}],
                     "description": "...", "subsets": []},
                ],
            },
        ],
    }

    flag_audit_issues(lattice)
    hint_kinds = [h["kind"] for h in (lattice.get("audit_hints") or [])]
    assert "cross_org_family" in hint_kinds
    assert "sibling_identity_collision" in hint_kinds


def test_flag_canonical_url_mismatch():
    """When formal_name doesn't match the canonical path inside the
    primary HF URL, emit a rename hint."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "MMLU",
            "identity_keys": ["family"],
            "items": [{
                "kind": "dataset", "formal_name": "MMLU",
                "identity": {"family": "MMLU"},
                "aliases": ["MMLU"],
                "links": [{"kind": "hf_dataset",
                           "url": "https://huggingface.co/datasets/cais/mmlu"}],
                "description": "...",
                "subsets": [],
            }],
        }],
    }

    flag_audit_issues(lattice)
    hints = [h for h in (lattice.get("audit_hints") or [])
             if h["kind"] == "formal_name_vs_canonical_url_mismatch"]
    assert len(hints) == 1
    assert hints[0]["item_formal_name"] == "MMLU"
    assert hints[0]["canonical_from_url"] == "cais/mmlu"


def test_flag_branch_variant_in_formal_name():
    """`@branch` in formal_name is HF git-revspec syntax. Flag for
    collapse into the canonical repo (part before @)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "Marin",
            "identity_keys": ["org", "collection"],
            "items": [
                {"kind": "model",
                 "formal_name": "marin-community/marin-8b-base@phoenix",
                 "identity": {"org": "marin-community", "collection": "Marin"},
                 "aliases": ["Marin 8B Phoenix"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/marin-community/marin-8b-base"}],
                 "description": "...", "subsets": []},
                {"kind": "model",
                 "formal_name": "marin-community/marin-8b-base@starling",
                 "identity": {"org": "marin-community", "collection": "Marin"},
                 "aliases": ["Marin 8B Starling"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/marin-community/marin-8b-base"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }

    flag_audit_issues(lattice)
    branch_hints = [h for h in (lattice.get("audit_hints") or [])
                    if h["kind"] == "branch_variant_in_formal_name"]
    assert len(branch_hints) == 2
    bases = {h["canonical_repo"] for h in branch_hints}
    branches = {h["branch"] for h in branch_hints}
    assert bases == {"marin-community/marin-8b-base"}
    assert branches == {"phoenix", "starling"}

    # Lattice itself is unchanged (purely additive)
    items = lattice["groups"][0]["items"]
    assert len(items) == 2
    assert items[0]["formal_name"] == "marin-community/marin-8b-base@phoenix"
