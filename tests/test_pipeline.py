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
        "batch_dir": "/b", "batches_dir": "/bs",
        "organize_path": "/o.json",
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
    in the DB. Each group has a family root + entity leaves."""
    from gdb.pipeline import run_organize
    from gdb.store import all_rows, loads

    artifact = {
        "groups": [
            {
                "family": "Qwen3",
                "identity_keys": ["family", "size", "stage"],
                "items": [
                    {
                        "kind": "model",
                        "formal_name": "Qwen3",
                        "identity": {"family": "Qwen3"},
                        "aliases": ["Qwen 3", "Qwen3"],
                        "links": [
                            {"kind": "paper",
                             "url": "https://arxiv.org/abs/2509.18888"},
                        ],
                        "description": "The Qwen3 family of open-weight language models.",
                    },
                    {
                        "kind": "model",
                        "formal_name": "Qwen/Qwen3-4B-Base",
                        "identity": {"family": "Qwen3", "size": "4B", "stage": "Base"},
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
                        "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
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
                "identity_keys": ["family"],
                "items": [
                    {
                        "kind": "dataset",
                        "formal_name": "cais/mmlu",
                        # MMLU happens to have only one canonical release; the
                        # leaf and the root collapse — but we still need a
                        # distinct family root, so we model it with a paper anchor.
                        "identity": {"family": "MMLU", "org": "cais"},
                        "aliases": ["MMLU"],
                        "links": [
                            {"kind": "hf_dataset",
                             "url": "https://huggingface.co/datasets/cais/mmlu"},
                        ],
                        "description": "Massive Multitask Language Understanding benchmark.",
                    },
                    {
                        "kind": "dataset",
                        "formal_name": "MMLU",
                        "identity": {"family": "MMLU"},
                        "aliases": ["MMLU"],
                        "links": [
                            {"kind": "paper",
                             "url": "https://arxiv.org/abs/2009.03300"},
                        ],
                        "description": "MMLU benchmark family root.",
                    },
                ],
            },
        ],
    }
    artifact_path = tmp_path / "organize.json"
    artifact_path.write_text(json.dumps(artifact))

    result = run_organize(artifact_path=str(artifact_path))
    assert result["group_count"] == 2
    assert result["item_count"] == 5
    assert result["items_with_links"] == 5
    assert result["n_family_roots"] == 2
    assert result["n_entity_leaves"] == 3
    assert result["links_by_kind"]["hf_model"] == 2
    assert result["links_by_kind"]["hf_dataset"] == 1
    assert result["links_by_kind"]["paper"] == 3
    assert Path(result["artifact_path"]).read_text() == artifact_path.read_text()

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='organize'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["group_count"] == 2
    assert attrs["item_count"] == 5
    assert attrs["n_family_roots"] == 2
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


def _root_only_group(family="X"):
    """Helper: a minimal valid group with only a family root."""
    return {
        "family": family,
        "identity_keys": ["family"],
        "items": [{
            "kind": "model",
            "formal_name": family,
            "identity": {"family": family},
            "aliases": [family],
            "links": [],
            "description": None,
        }],
    }


def test_run_organize_allows_missing_links_field(fresh_runtime, tmp_path):
    """The `links` field is optional. When absent, treated as empty.
    Audit fills in tentative URLs and verifies them."""
    from gdb.pipeline import run_organize

    no_links = tmp_path / "no_links.json"
    no_links.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family"],
            "items": [{"kind": "model", "formal_name": "X",
                       "identity": {"family": "X"}, "aliases": ["X"]}],
        }],
    }))
    result = run_organize(artifact_path=str(no_links))
    assert result["item_count"] == 1
    assert result["items_with_links"] == 0


def test_run_organize_rejects_invalid_link_kind(fresh_runtime, tmp_path):
    """Primary link must use a closed-vocabulary kind."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad_kind = tmp_path / "bad_kind.json"
    bad_kind.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family"],
            "items": [{
                "kind": "model", "formal_name": "X",
                "identity": {"family": "X"}, "aliases": ["X"],
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
            "identity_keys": ["family", "size"],
            "items": [
                # Family root
                {"kind": "model", "formal_name": "X", "identity": {"family": "X"},
                 "aliases": ["X"], "links": [], "description": None},
                # Leaf with bad URL
                {"kind": "model", "formal_name": "X/Y",
                 "identity": {"family": "X", "size": "1B"}, "aliases": ["X 1B"],
                 "links": [{"kind": "hf_model", "url": "ftp://example/foo"}],
                 "description": None},
            ],
        }],
    }))
    with pytest.raises(click.ClickException, match="url"):
        run_organize(artifact_path=str(bad_url))


def test_run_organize_allows_root_only_family(fresh_runtime, tmp_path):
    """A family with only a root (no leaves) is valid: foundational data
    resources like Common Crawl, AoPS forums often have no specific HF
    release. Root may have empty links or paper/blog/hf_collection link."""
    from gdb.pipeline import run_organize

    artifact = tmp_path / "root_only.json"
    artifact.write_text(json.dumps({
        "groups": [_root_only_group("Common Crawl")],
    }))
    result = run_organize(artifact_path=str(artifact))
    assert result["group_count"] == 1
    assert result["item_count"] == 1
    assert result["n_family_roots"] == 1
    assert result["n_entity_leaves"] == 0
    assert result["items_with_links"] == 0


def test_run_organize_allows_paper_anchored_leaf(fresh_runtime, tmp_path):
    """An entity leaf (identity has facets beyond `family`) MAY have a
    paper-only link or no link. The production-vs-concept distinction
    is advisory — what makes a node a leaf is its facet set, not its
    link kind. This supports paper-anchored leaves, internal-codename
    leaves referenced in source code, and sub-components without a
    standalone HF release."""
    from gdb.pipeline import run_organize

    ok = tmp_path / "paper_leaf.json"
    ok.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family", "size"],
            "items": [
                {"kind": "model", "formal_name": "X", "identity": {"family": "X"},
                 "aliases": ["X"], "links": [], "description": None},
                {"kind": "model", "formal_name": "X-7B",
                 "identity": {"family": "X", "size": "7B"}, "aliases": ["X 7B"],
                 "links": [{"kind": "paper", "url": "https://arxiv.org/abs/0000.0000"}],
                 "description": None},
                {"kind": "model", "formal_name": "X-13B-internal",
                 "identity": {"family": "X", "size": "13B"},
                 "aliases": ["X 13B internal"],
                 "links": [],  # internal codename, no link
                 "description": None},
            ],
        }],
    }))
    result = run_organize(artifact_path=str(ok))
    assert result["item_count"] == 3
    assert result["n_family_roots"] == 1
    assert result["n_entity_leaves"] == 2


def test_run_organize_allows_single_release_family_root_with_production_link(
        fresh_runtime, tmp_path):
    """A family root MAY carry a production link when the family has
    exactly one canonical release. The root acts as both top and
    bottom of a single-leaf lattice — useful for one-off models /
    datasets like `tomh/toxigen_roberta` or `openai-community/gpt2`."""
    from gdb.pipeline import run_organize

    ok = tmp_path / "single_release.json"
    ok.write_text(json.dumps({
        "groups": [{
            "family": "ToxiGen RoBERTa",
            "identity_keys": ["family"],
            "items": [{
                "kind": "model", "formal_name": "tomh/toxigen_roberta",
                "identity": {"family": "ToxiGen RoBERTa"},
                "aliases": ["ToxiGen RoBERTa", "toxigen-roberta"],
                "links": [{"kind": "hf_model",
                           "url": "https://huggingface.co/tomh/toxigen_roberta"}],
                "description": "ToxiGen RoBERTa hate-speech classifier.",
            }],
        }],
    }))
    result = run_organize(artifact_path=str(ok))
    assert result["item_count"] == 1
    assert result["n_family_roots"] == 1


def test_run_organize_rejects_missing_family_facet(fresh_runtime, tmp_path):
    """Every item's identity MUST include a `family` key."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad = tmp_path / "no_family.json"
    bad.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["org", "size"],
            "items": [{"kind": "model", "formal_name": "X/Y",
                       "identity": {"org": "X", "size": "7B"},
                       "aliases": ["X-Y"],
                       "links": [{"kind": "hf_model",
                                  "url": "https://huggingface.co/X/Y"}],
                       "description": None}],
        }],
    }))
    with pytest.raises(click.ClickException, match="identity.family"):
        run_organize(artifact_path=str(bad))


def test_run_organize_synthesizes_missing_family_root(fresh_runtime, tmp_path):
    """When a group lacks a family root, the Python completion pre-pass
    synthesizes one (formal_name=family, identity={family: X}, aliases=[X],
    no links, null description) before validation. The on-disk artifact
    is the post-completion lattice."""
    from gdb.pipeline import run_organize

    no_root = tmp_path / "no_root.json"
    no_root.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family", "size"],
            "items": [{"kind": "model", "formal_name": "X/Y-7B",
                       "identity": {"family": "X", "size": "7B"},
                       "aliases": ["X-7B"],
                       "links": [{"kind": "hf_model",
                                  "url": "https://huggingface.co/X/Y-7B"}],
                       "description": None}],
        }],
    }))
    result = run_organize(artifact_path=str(no_root))
    # Completion synthesized 1 root + echoed formal_name into aliases
    assert result["item_count"] == 2
    assert result["n_family_roots"] == 1
    assert result["n_entity_leaves"] == 1
    assert result["completion"]["roots_synthesized"] == 1


def test_run_organize_rejects_inconsistent_family_value(fresh_runtime, tmp_path):
    """All items in a group must share the same identity.family value."""
    import click
    import pytest

    from gdb.pipeline import run_organize

    bad = tmp_path / "mixed_family.json"
    bad.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family"],
            "items": [
                {"kind": "model", "formal_name": "X",
                 "identity": {"family": "X"}, "aliases": ["X"],
                 "links": [], "description": None},
                {"kind": "model", "formal_name": "Y/Z",
                 "identity": {"family": "Y", "size": "7B"}, "aliases": ["Y-Z"],
                 "links": [{"kind": "hf_model", "url": "https://huggingface.co/Y/Z"}],
                 "description": None},
            ],
        }],
    }))
    with pytest.raises(click.ClickException, match="differs from sibling family"):
        run_organize(artifact_path=str(bad))


def test_run_organize_completion_fills_empty_aliases(fresh_runtime, tmp_path):
    """The Python completion pre-pass echoes formal_name into aliases[]
    for any item missing it, so empty aliases doesn't fail validation
    when the formal_name is non-empty (the typical planner mistake)."""
    from gdb.pipeline import run_organize

    art = tmp_path / "empty_aliases.json"
    art.write_text(json.dumps({
        "groups": [{
            "family": "X",
            "identity_keys": ["family"],
            "items": [{"kind": "model", "formal_name": "X",
                       "identity": {"family": "X"}, "aliases": [],
                       "links": [], "description": None}],
        }],
    }))
    result = run_organize(artifact_path=str(art))
    assert result["item_count"] == 1
    assert result["completion"]["aliases_added"] == 1


def test_cli_run_help_lists_only_active_stages(fresh_runtime):
    from click.testing import CliRunner

    from gdb.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    out = result.output
    for stage in ("discover", "extract", "organize", "audit",
                  "relate", "reconcile", "triage", "merge", "expand"):
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
                "identity_keys": ["family", "size", "variant"],
                "items": [
                    # Family root — required
                    {"kind": "model", "formal_name": "Qwen3",
                     "identity": {"family": "Qwen3"},
                     "aliases": ["Qwen3", "Qwen 3"],
                     "links": [{"kind": "paper",
                                "url": "https://arxiv.org/abs/2509.18888"}],
                     "description": "The Qwen3 family of language models."},
                    {"kind": "model", "formal_name": "Qwen/Qwen3-8B",
                     "identity": {"family": "Qwen3", "size": "8B",
                                  "variant": "no-thinking"},
                     "aliases": ["Qwen3-8B"],
                     "links": [{"kind": "hf_model",
                                "url": "https://huggingface.co/Qwen/Qwen3-8B"}],
                     "description": "Qwen3 8B without reasoning."},
                    {"kind": "model", "formal_name": "Qwen/Qwen3-8B-Thinking",
                     "identity": {"family": "Qwen3", "size": "8B",
                                  "variant": "thinking"},
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
    # Audit ingest runs expand_concept_lattice — the two leaves with
    # 2 non-family facets each produce 3 interior concept projections
    # ({size:8B}, {variant:thinking}, {variant:no-thinking}). Total
    # items = 1 root + 2 leaves + 3 synthesized concepts = 6.
    assert result["group_count"] == 1
    assert result["item_count"] == 6
    assert result["expansion"]["concepts_synthesized"] == 3

    rows = all_rows("SELECT id, attrs FROM runs WHERE stage='audit'")
    assert len(rows) == 1
    attrs = loads(rows[0]["attrs"])
    assert attrs["group_count"] == 1
    assert attrs["item_count"] == 6
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


def test_parse_virtual_address():
    """Virtual concept address parsing for relate edge endpoints."""
    from gdb.pipeline import parse_virtual_address

    assert parse_virtual_address("OLMo 3 [stage=Base]") == (
        "OLMo 3", {"stage": "Base"})
    assert parse_virtual_address("Qwen3 [size=4B, stage=Base]") == (
        "Qwen3", {"size": "4B", "stage": "Base"})
    assert parse_virtual_address("olmOCR [version=v1]") == (
        "olmOCR", {"version": "v1"})
    # Plain formal_name → not a virtual address
    assert parse_virtual_address("allenai/Olmo-3-1025-7B") is None
    assert parse_virtual_address("Qwen3") is None
    # Malformed
    assert parse_virtual_address("OLMo 3 [stage]") is None  # no =
    assert parse_virtual_address("OLMo 3 []") == ("OLMo 3", {})  # empty facets ok
    assert parse_virtual_address("") is None
    assert parse_virtual_address(None) is None


def test_relate_accepts_virtual_concept_address(fresh_runtime):
    """Subjects/objects can be virtual concept addresses when their
    family pivots to a known lattice family."""
    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"][0]["subject"] = "OLMo 3 [stage=Base]"
    stats = _validate_relate_artifact(
        artifact,
        lattice_formal_names={"allenai/Olmo-3-7B-Base", "allenai/dolma3-mix",
                              "allenai/olmOCR-7B-0225"},
        lattice_family_names={"OLMo 3", "Dolma 3", "olmOCR"},
    )
    assert stats["edge_count"] == 2


def test_relate_rejects_virtual_address_with_unknown_family(fresh_runtime):
    """Virtual address pivoting to an unknown family is rejected."""
    import click
    import pytest

    from gdb.pipeline import _validate_relate_artifact

    artifact = json.loads(json.dumps(_well_formed_relate_artifact()))
    artifact["operations"][0]["edges"][0]["subject"] = "PhantomFamily [size=7B]"
    with pytest.raises(click.ClickException, match="virtual address"):
        _validate_relate_artifact(
            artifact,
            lattice_formal_names={"allenai/Olmo-3-7B-Base"},
            lattice_family_names={"OLMo 3"},
        )


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


def test_reconcile_subsumption_collapses_vague_into_specific():
    """If one source says 'OLMo 3 Base trained_on Dolma 3' and another
    says 'allenai/Olmo-3-1025-7B trained_on allenai/dolma3_mix-6T-1025-7B',
    reconcile marks the vague edge as subsumed by the specific."""
    from gdb.pipeline import _reconcile_edges

    lattice = {
        "groups": [
            {"family": "OLMo 3", "items": [
                {"formal_name": "OLMo 3", "identity": {"family": "OLMo 3"}},
                {"formal_name": "allenai/Olmo-3-1025-7B",
                 "identity": {"family": "OLMo 3", "size": "7B",
                              "stage": "Base", "date": "1025"}},
            ]},
            {"family": "Dolma 3", "items": [
                {"formal_name": "Dolma 3", "identity": {"family": "Dolma 3"}},
                {"formal_name": "allenai/dolma3_mix-6T-1025-7B",
                 "identity": {"family": "Dolma 3", "size": "6T",
                              "date": "1025", "target": "7B"}},
            ]},
        ],
    }
    edges = [
        {"subject": "OLMo 3 [stage=Base]", "relation": "trained_on",
         "dependency_kind": "direct", "object": "Dolma 3",
         "description": "Olmo 3 base was pretrained on Dolma 3.",
         "anchor_list": [{"source": "paper.pdf", "explanation": "..."}],
         "_batch_id": "b1"},
        {"subject": "allenai/Olmo-3-1025-7B", "relation": "trained_on",
         "dependency_kind": "direct", "object": "allenai/dolma3_mix-6T-1025-7B",
         "description": "...", "anchor_list": [
             {"source": "config.yaml", "explanation": "..."}
         ], "_batch_id": "b2"},
    ]
    result = _reconcile_edges(edges, lattice)

    assert result["total_edge_count"] == 2
    canonical = [e for e in result["edges"] if e["subsumed_by"] is None]
    subsumed = [e for e in result["edges"] if e["subsumed_by"] is not None]
    assert len(canonical) == 1
    assert len(subsumed) == 1
    # The specific edge subsumes the vague edge
    assert canonical[0]["subject"] == "allenai/Olmo-3-1025-7B"
    assert subsumed[0]["subject"] == "OLMo 3 [stage=Base]"
    # And inherits its anchors
    sources = {a["source"] for a in canonical[0]["anchor_list"]}
    assert "paper.pdf" in sources
    assert "config.yaml" in sources


def test_reconcile_corroboration_stacks_anchors():
    """Same (subject, relation, object) from two sources stacks evidence."""
    from gdb.pipeline import _reconcile_edges

    lattice = {"groups": [{"family": "OLMo 3", "items": [
        {"formal_name": "OLMo 3", "identity": {"family": "OLMo 3"}},
    ]}]}
    edges = [
        {"subject": "OLMo 3", "relation": "trained_on",
         "dependency_kind": "direct", "object": "OLMo 3",  # toy
         "description": "Same event, source A.",
         "anchor_list": [{"source": "paper.pdf", "explanation": "..."}],
         "_batch_id": "b1"},
        {"subject": "OLMo 3", "relation": "trained_on",
         "dependency_kind": "direct", "object": "OLMo 3",
         "description": "Same event, source B.",
         "anchor_list": [{"source": "blog.html", "explanation": "..."}],
         "_batch_id": "b2"},
    ]
    result = _reconcile_edges(edges, lattice)
    assert result["total_edge_count"] == 1
    e = result["edges"][0]
    assert e["corroboration_count"] == 2
    assert {a["source"] for a in e["anchor_list"]} == {"paper.pdf", "blog.html"}
    # Differing descriptions kept as variants
    assert "Same event, source B." in e["description_variants"]


def test_reconcile_conflict_flags_sibling_endpoints():
    """Same subject + relation, but objects are different siblings in
    the same family — flag as conflict."""
    from gdb.pipeline import _reconcile_edges

    lattice = {
        "groups": [
            {"family": "OLMo 3", "items": [
                {"formal_name": "OLMo 3", "identity": {"family": "OLMo 3"}},
                {"formal_name": "allenai/Olmo-3-1025-7B",
                 "identity": {"family": "OLMo 3", "size": "7B"}},
            ]},
            {"family": "Dolma 3", "items": [
                {"formal_name": "Dolma 3", "identity": {"family": "Dolma 3"}},
                {"formal_name": "allenai/dolma3_mix-A",
                 "identity": {"family": "Dolma 3", "variant": "A"}},
                {"formal_name": "allenai/dolma3_mix-B",
                 "identity": {"family": "Dolma 3", "variant": "B"}},
            ]},
        ],
    }
    edges = [
        {"subject": "allenai/Olmo-3-1025-7B", "relation": "trained_on",
         "dependency_kind": "direct", "object": "allenai/dolma3_mix-A",
         "description": "...", "anchor_list": [{"source": "a", "explanation": "."}],
         "_batch_id": "b1"},
        {"subject": "allenai/Olmo-3-1025-7B", "relation": "trained_on",
         "dependency_kind": "direct", "object": "allenai/dolma3_mix-B",
         "description": "...", "anchor_list": [{"source": "b", "explanation": "."}],
         "_batch_id": "b2"},
    ]
    result = _reconcile_edges(edges, lattice)
    assert result["conflict_count"] == 1
    conflict = result["conflicts"][0]
    assert conflict["subject"] == "allenai/Olmo-3-1025-7B"
    assert {conflict["object_a"], conflict["object_b"]} == {
        "allenai/dolma3_mix-A", "allenai/dolma3_mix-B",
    }


def test_run_reconcile_ingests_artifact(fresh_runtime, tmp_path):
    """`--artifact` ingests a pre-computed reconcile artifact."""
    from gdb.pipeline import run_reconcile

    artifact = tmp_path / "reconcile.json"
    artifact.write_text(json.dumps({
        "edges": [
            {"subject": "a", "relation": "trained_on", "object": "b",
             "subsumed_by": None, "subsumes": [],
             "anchor_list": [], "corroboration_count": 1,
             "source_batch_ids": ["b1"]},
        ],
        "conflicts": [],
        "canonical_edge_count": 1, "subsumed_edge_count": 0,
        "total_edge_count": 1, "corroboration_count": 0, "conflict_count": 0,
    }))
    result = run_reconcile(artifact_path=str(artifact))
    assert result["total_edge_count"] == 1
    assert result["conflict_count"] == 0


def test_parse_virtual_address_via_reconcile_resolution():
    """The address resolver routes formal_names AND virtual addresses
    to identity dicts; off-lattice strings return None."""
    from gdb.pipeline import _identity_for_address

    lattice = {"groups": [{"family": "X", "items": [
        {"formal_name": "X", "identity": {"family": "X"}},
        {"formal_name": "X/leaf", "identity": {"family": "X", "size": "7B"}},
    ]}]}
    assert _identity_for_address("X", lattice) == {"family": "X"}
    assert _identity_for_address("X/leaf", lattice) == {"family": "X", "size": "7B"}
    # Virtual address — reuses parser even for non-existent leaf
    assert _identity_for_address("X [stage=Base]", lattice) == {
        "family": "X", "stage": "Base"}
    # Free-text (no formal_name match, not a virtual address) → None
    assert _identity_for_address("totally unknown thing", lattice) is None


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
            "family": "X", "identity_keys": ["family"],
            "items": [
                {"kind": "dataset", "formal_name": "X",
                 "identity": {"family": "X"}, "aliases": ["X"],
                 "links": [], "description": None},
                {"kind": "dataset", "formal_name": "X/Y",
                 "identity": {"family": "X", "size": "1B"}, "aliases": ["X-Y"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/X/Y"}],
                 "description": None,
                 "subsets": ["valid", "", "also valid"]},
            ],
        }],
    }))
    with pytest.raises(click.ClickException, match="subsets"):
        run_organize(artifact_path=str(bad))


def test_organize_family_root_with_concept_link_is_valid(fresh_runtime, tmp_path):
    """Family roots may carry paper / blog / hf_collection links —
    those describe the concept without pinning a specific release."""
    from gdb.pipeline import run_organize

    ok = tmp_path / "concept_root.json"
    ok.write_text(json.dumps({
        "groups": [{
            "family": "Qwen3", "identity_keys": ["family", "size"],
            "items": [
                {"kind": "model", "formal_name": "Qwen3",
                 "identity": {"family": "Qwen3"}, "aliases": ["Qwen3"],
                 "links": [{"kind": "hf_collection",
                            "url": "https://huggingface.co/collections/Qwen/qwen3"}],
                 "description": "Qwen3 is the latest generation..."},
                {"kind": "model", "formal_name": "Qwen/Qwen3-7B",
                 "identity": {"family": "Qwen3", "size": "7B"},
                 "aliases": ["Qwen3-7B"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-7B"}],
                 "description": "..."},
            ],
        }],
    }))
    result = run_organize(artifact_path=str(ok))
    assert result["item_count"] == 2
    assert result["n_family_roots"] == 1
    assert result["n_entity_leaves"] == 1


def test_flag_audit_issues_purely_additive():
    """The flag pass MUST NOT mutate the lattice — no items moved,
    no drops restored, no renames. It only adds an `audit_hints[]`
    array."""
    from gdb.subsets import flag_audit_issues
    import json as _json

    lattice = {
        "groups": [{
            "family": "Dolma3",
            "identity_keys": ["family", "size", "date"],
            "items": [
                {"kind": "dataset", "formal_name": "Dolma3",
                 "identity": {"family": "Dolma3"}, "aliases": ["Dolma 3"],
                 "links": [], "description": "..."},
                {"kind": "dataset",
                 "formal_name": "allenai/dolma3_dolmino_mix-100B-1025",
                 "identity": {"family": "Dolma3", "size": "100B", "date": "1025"},
                 "aliases": ["dolma3-dolmino-mix-100B-1025"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025"}],
                 "description": "...",
                 "subsets": ["cranemath", "cranecode"]},
            ],
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

    # No phantoms because every item has aliases. No missing root.
    hints = lattice.get("audit_hints") or []
    kinds = {h["kind"] for h in hints}
    assert "phantom_item" not in kinds
    assert "missing_family_root" not in kinds


def test_flag_item_matches_parent_subset_with_role_classification():
    """When an item's slug appears in a parent's subsets[], emit a
    hint with the item's role (canonical / concept / family-root)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [
            {
                "family": "Dolma3",
                "identity_keys": ["family", "size"],
                "items": [
                    {"kind": "dataset", "formal_name": "Dolma3",
                     "identity": {"family": "Dolma3"}, "aliases": ["Dolma 3"],
                     "links": [], "description": "...", "subsets": []},
                    {"kind": "dataset",
                     "formal_name": "allenai/dolma3_dolmino_pool",
                     "identity": {"family": "Dolma3", "size": "pool"},
                     "aliases": ["dolma3-dolmino-pool"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/allenai/dolma3_dolmino_pool"}],
                     "description": "...",
                     "subsets": ["cranemath", "common-crawl"]},
                ],
            },
            {
                "family": "olmo3-tr",
                "identity_keys": ["family", "size"],
                "items": [
                    {"kind": "dataset", "formal_name": "olmo3-tr",
                     "identity": {"family": "olmo3-tr"}, "aliases": ["olmo3-tr"],
                     "links": [], "description": "...", "subsets": []},
                    # soft-anchored sub-component — paper-anchor, audit may reshape
                    {"kind": "dataset",
                     "formal_name": "allenai/olmo3-tr-cranemath",
                     "identity": {"family": "olmo3-tr", "size": "5.62B"},
                     "aliases": ["CraneMath"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/allenai/olmo3-tr-cranemath"}],
                     "description": "...", "subsets": []},
                ],
            },
            {
                "family": "Common Crawl",
                "identity_keys": ["family"],
                "items": [
                    # family root with bare-domain URL — hint should label
                    # item_role='family-root', signaling "keep standalone"
                    {"kind": "dataset",
                     "formal_name": "Common Crawl",
                     "identity": {"family": "Common Crawl"},
                     "aliases": ["Common Crawl", "common_crawl"],
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

    # CraneMath leaf flagged with item_role='canonical' (HF link → leaf)
    assert "allenai/olmo3-tr-cranemath" in by_item
    assert by_item["allenai/olmo3-tr-cranemath"]["item_role"] == "canonical"
    assert by_item["allenai/olmo3-tr-cranemath"]["matched_parent"] == "allenai/dolma3_dolmino_pool"

    # Common Crawl flagged with item_role='family-root'
    assert "Common Crawl" in by_item
    assert by_item["Common Crawl"]["item_role"] == "family-root"


def test_flag_dropped_matches_parent_subset():
    """A dropped name whose slug appears in some kept item's subsets[]
    gets flagged for restoration (but NOT auto-restored)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "FineMath",
            "identity_keys": ["family"],
            "items": [
                {"kind": "dataset", "formal_name": "FineMath",
                 "identity": {"family": "FineMath"}, "aliases": ["FineMath"],
                 "links": [], "description": "...", "subsets": []},
                {"kind": "dataset",
                 "formal_name": "HuggingFaceTB/finemath",
                 "identity": {"family": "FineMath", "org": "HuggingFaceTB"},
                 "aliases": ["HuggingFaceTB/finemath"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/HuggingFaceTB/finemath"}],
                 "description": "...",
                 "subsets": ["finemath-3plus", "finemath-4plus"]},
            ],
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
                "identity_keys": ["family", "org"],
                "items": [
                    {"kind": "dataset", "formal_name": "HumanEval",
                     "identity": {"family": "HumanEval"}, "aliases": ["HumanEval"],
                     "links": [], "description": "...", "subsets": []},
                    {"kind": "dataset", "formal_name": "openai/openai_humaneval",
                     "identity": {"family": "HumanEval", "org": "openai"},
                     "aliases": ["openai_humaneval"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/openai/openai_humaneval"}],
                     "description": "...", "subsets": []},
                    {"kind": "dataset", "formal_name": "evalplus/humanevalplus",
                     "identity": {"family": "HumanEval", "org": "evalplus"},
                     "aliases": ["humaneval+"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/evalplus/humanevalplus"}],
                     "description": "...", "subsets": []},
                ],
            },
            {
                "family": "Reasoning-traces",
                "identity_keys": ["family", "org", "collection"],
                "items": [
                    {"kind": "dataset", "formal_name": "Reasoning-traces",
                     "identity": {"family": "Reasoning-traces"},
                     "aliases": ["Reasoning traces"], "links": [],
                     "description": "...", "subsets": []},
                    # Identity collision — same identity for both leaves
                    {"kind": "dataset", "formal_name": "allenai/r1-traces",
                     "identity": {"family": "Reasoning-traces", "org": "allenai",
                                  "collection": "reasoning-traces"},
                     "aliases": ["r1"],
                     "links": [{"kind": "hf_dataset",
                                "url": "https://huggingface.co/datasets/allenai/r1-traces"}],
                     "description": "...", "subsets": []},
                    {"kind": "dataset", "formal_name": "allenai/qwq-traces",
                     "identity": {"family": "Reasoning-traces", "org": "allenai",
                                  "collection": "reasoning-traces"},
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
            "items": [
                {"kind": "dataset", "formal_name": "MMLU-Family",
                 "identity": {"family": "MMLU"}, "aliases": ["MMLU"],
                 "links": [], "description": "...", "subsets": []},
                {"kind": "dataset", "formal_name": "MMLU",
                 "identity": {"family": "MMLU", "org": "cais"},
                 "aliases": ["cais/mmlu"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/cais/mmlu"}],
                 "description": "...", "subsets": []},
            ],
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
            "identity_keys": ["family", "size", "branch"],
            "items": [
                {"kind": "model", "formal_name": "Marin",
                 "identity": {"family": "Marin"}, "aliases": ["Marin"],
                 "links": [], "description": "...", "subsets": []},
                {"kind": "model",
                 "formal_name": "marin-community/marin-8b-base@phoenix",
                 "identity": {"family": "Marin", "size": "8B", "branch": "phoenix"},
                 "aliases": ["Marin 8B Phoenix"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/marin-community/marin-8b-base"}],
                 "description": "...", "subsets": []},
                {"kind": "model",
                 "formal_name": "marin-community/marin-8b-base@starling",
                 "identity": {"family": "Marin", "size": "8B", "branch": "starling"},
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
    assert len(items) == 3


def test_complete_lattice_structure_echoes_formal_name_into_aliases():
    """Phase 1: every item's formal_name is added to aliases[] if missing."""
    from gdb.subsets import complete_lattice_structure

    lattice = {
        "groups": [{
            "family": "X",
            "items": [
                {"kind": "model", "formal_name": "X",
                 "identity": {"family": "X"}, "aliases": [],
                 "links": [], "description": None},
                {"kind": "model", "formal_name": "X/Y-7B",
                 "identity": {"family": "X", "size": "7B"},
                 "aliases": ["x-y-7b"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/X/Y-7B"}],
                 "description": None},
            ],
        }],
    }
    stats = complete_lattice_structure(lattice)
    assert stats["aliases_added"] == 2  # X for item 0, X/Y-7B for item 1
    items = lattice["groups"][0]["items"]
    assert "X" in items[0]["aliases"]
    assert "X/Y-7B" in items[1]["aliases"]


def test_complete_lattice_structure_synthesizes_virtual_root():
    """Phase 2: synthesize a virtual family root for groups that don't
    have one. Idempotent: calling twice doesn't duplicate."""
    from gdb.subsets import complete_lattice_structure

    lattice = {
        "groups": [{
            "family": "X",
            "items": [
                {"kind": "model", "formal_name": "X/Y-7B",
                 "identity": {"family": "X", "size": "7B"},
                 "aliases": ["x-y-7b"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/X/Y-7B"}],
                 "description": None},
            ],
        }],
    }
    stats = complete_lattice_structure(lattice)
    assert stats["roots_synthesized"] == 1
    items = lattice["groups"][0]["items"]
    assert len(items) == 2
    # Synthesized root is at index 0
    root = items[0]
    assert root["identity"] == {"family": "X"}
    assert root["formal_name"] == "X"
    assert "X" in root["aliases"]
    assert root["links"] == []
    assert root.get("_synthesized") is True

    # Idempotent: a second call doesn't add another root
    stats2 = complete_lattice_structure(lattice)
    assert stats2["roots_synthesized"] == 0
    assert len(lattice["groups"][0]["items"]) == 2


def test_complete_lattice_structure_skips_when_root_exists():
    """If a root already exists, don't synthesize another."""
    from gdb.subsets import complete_lattice_structure

    lattice = {
        "groups": [{
            "family": "X",
            "items": [
                {"kind": "model", "formal_name": "X-display",
                 "identity": {"family": "X"}, "aliases": ["X"],
                 "links": [{"kind": "paper", "url": "https://arxiv.org/abs/0"}],
                 "description": "..."},
                {"kind": "model", "formal_name": "X/Y-7B",
                 "identity": {"family": "X", "size": "7B"},
                 "aliases": ["x-y-7b"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/X/Y-7B"}],
                 "description": None},
            ],
        }],
    }
    stats = complete_lattice_structure(lattice)
    assert stats["roots_synthesized"] == 0


def test_flag_missing_family_root():
    """When a group has 2+ items but no family root (identity == {family: X}
    only), emit a `missing_family_root` hint so audit synthesizes one."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "Qwen3",
            "identity_keys": ["family", "size", "stage"],
            "items": [
                # Two leaves, no root
                {"kind": "model", "formal_name": "Qwen/Qwen3-7B",
                 "identity": {"family": "Qwen3", "size": "7B", "stage": "chat"},
                 "aliases": ["Qwen3-7B"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-7B"}],
                 "description": "...", "subsets": []},
                {"kind": "model", "formal_name": "Qwen/Qwen3-32B",
                 "identity": {"family": "Qwen3", "size": "32B", "stage": "chat"},
                 "aliases": ["Qwen3-32B"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-32B"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    missing = [h for h in (lattice.get("audit_hints") or [])
               if h["kind"] == "missing_family_root"]
    assert len(missing) == 1
    assert missing[0]["family"] == "Qwen3"
    assert missing[0]["item_count"] == 2


def test_flag_over_specified_alias():
    """When a leaf carries a bare family-name alias (e.g., 'olmOCR') AND
    its formal_name pins specific facets, emit an `over_specified` hint
    so audit can split — moving the bare alias to the family root."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "olmOCR",
            "identity_keys": ["family", "size", "version"],
            "items": [
                {"kind": "model", "formal_name": "olmOCR",
                 "identity": {"family": "olmOCR"}, "aliases": ["olmOCR-family"],
                 "links": [{"kind": "paper",
                            "url": "https://arxiv.org/abs/2502.18443"}],
                 "description": "olmOCR family", "subsets": []},
                {"kind": "model",
                 "formal_name": "allenai/olmOCR-7B-0225-preview",
                 "identity": {"family": "olmOCR", "size": "7B", "version": "0225-preview"},
                 # Bare 'olmOCR' alias on a fully-specific leaf — flag
                 "aliases": ["olmOCR", "olmOCR-7B-0225-preview"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/allenai/olmOCR-7B-0225-preview"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    over = [h for h in (lattice.get("audit_hints") or [])
            if h["kind"] == "over_specified"]
    assert len(over) == 1
    assert over[0]["item_formal_name"] == "allenai/olmOCR-7B-0225-preview"
    assert over[0]["bare_alias"] == "olmOCR"
    assert over[0]["item_family"] == "olmOCR"


def test_flag_concept_subsumed_candidate():
    """A.facets ⊂ B.facets within a family, A has no unique anchor →
    `concept_subsumed_candidate` hint. A is likely a concept."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "Qwen3",
            "identity_keys": ["family", "size", "stage"],
            "items": [
                # A: identity={family, stage:Base}, no links
                {"kind": "model", "formal_name": "Qwen3-Base",
                 "identity": {"family": "Qwen3", "stage": "Base"},
                 "aliases": ["Qwen3-Base"], "links": [],
                 "description": None, "subsets": []},
                # B: full identity, has hf_model
                {"kind": "model", "formal_name": "Qwen/Qwen3-4B-Base",
                 "identity": {"family": "Qwen3", "size": "4B", "stage": "Base"},
                 "aliases": ["Qwen/Qwen3-4B-Base"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-4B-Base"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    hints = [h for h in (lattice.get("audit_hints") or [])
             if h["kind"] == "concept_subsumed_candidate"]
    assert len(hints) == 1
    assert hints[0]["subsumed_formal_name"] == "Qwen3-Base"
    assert hints[0]["subsumes_formal_name"] == "Qwen/Qwen3-4B-Base"


def test_flag_subset_with_anchor():
    """A.facets ⊂ B.facets (both multi-facet), BOTH have unique anchors →
    `subset_with_anchor` hint (dataset-config / subset-of relationship).
    Single-key family-root cases are filtered out as design noise."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "FineMath",
            "identity_keys": ["family", "version", "filter"],
            "items": [
                # Family root concept — filtered out of subsumption check
                {"kind": "dataset", "formal_name": "FineMath",
                 "identity": {"family": "FineMath"},
                 "aliases": ["FineMath"],
                 "links": [], "description": "...", "subsets": []},
                # Parent dataset — multi-facet identity, has its own URL
                {"kind": "dataset", "formal_name": "HuggingFaceTB/finemath",
                 "identity": {"family": "FineMath", "version": "v1"},
                 "aliases": ["HuggingFaceTB/finemath"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/HuggingFaceTB/finemath"}],
                 "description": "...", "subsets": []},
                # Child — filter facet adds a key
                {"kind": "dataset", "formal_name": "infimm-webmath/infiwebmath-3+",
                 "identity": {"family": "FineMath", "version": "v1", "filter": "score>=3"},
                 "aliases": ["infiwebmath-3+"],
                 "links": [{"kind": "hf_dataset",
                            "url": "https://huggingface.co/datasets/infimm-webmath/infiwebmath-3plus"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    hints = [h for h in (lattice.get("audit_hints") or [])
             if h["kind"] == "subset_with_anchor"]
    assert len(hints) == 1
    assert hints[0]["child_formal_name"] == "HuggingFaceTB/finemath"
    assert hints[0]["parent_formal_name"] == "infimm-webmath/infiwebmath-3+"


def test_flag_same_url_duplicate():
    """Two items in the same family share the same primary URL → dup hint.
    The thinking/no-thinking case."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "Qwen3",
            "identity_keys": ["family", "size", "stage", "variant"],
            "items": [
                {"kind": "model", "formal_name": "Qwen/Qwen3-4B",
                 "identity": {"family": "Qwen3", "size": "4B",
                              "stage": "chat", "variant": "thinking"},
                 "aliases": ["Qwen3-4B (thinking on)"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-4B"}],
                 "description": "...", "subsets": []},
                {"kind": "model", "formal_name": "Qwen/Qwen3-4B",
                 "identity": {"family": "Qwen3", "size": "4B",
                              "stage": "chat", "variant": "no-thinking"},
                 "aliases": ["Qwen3-4B (thinking off)"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-4B"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    dups = [h for h in (lattice.get("audit_hints") or [])
            if h["kind"] == "same_url_duplicate"]
    assert len(dups) == 1
    assert dups[0]["shared_url"] == "https://huggingface.co/Qwen/Qwen3-4B"
    assert "Qwen/Qwen3-4B" in dups[0]["items"]


def test_expand_concept_lattice_synthesizes_interior_concepts():
    """expand_concept_lattice projects leaves onto subsets of
    identity_keys; concepts that don't already exist get materialized."""
    from gdb.subsets import expand_concept_lattice

    lattice = {
        "groups": [{
            "family": "OLMo 3",
            "identity_keys": ["family", "size", "stage"],
            "items": [
                # Family root (concept, exists already)
                {"kind": "model", "formal_name": "OLMo 3",
                 "identity": {"family": "OLMo 3"},
                 "aliases": ["OLMo 3"], "links": [], "description": None},
                # Two leaves
                {"kind": "model", "formal_name": "allenai/Olmo-3-7B-Base",
                 "identity": {"family": "OLMo 3", "size": "7B", "stage": "Base"},
                 "aliases": ["allenai/Olmo-3-7B-Base"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/allenai/Olmo-3-7B-Base"}],
                 "description": "..."},
                {"kind": "model", "formal_name": "allenai/Olmo-3-32B-Base",
                 "identity": {"family": "OLMo 3", "size": "32B", "stage": "Base"},
                 "aliases": ["allenai/Olmo-3-32B-Base"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/allenai/Olmo-3-32B-Base"}],
                 "description": "..."},
            ],
        }],
    }
    stats = expand_concept_lattice(lattice)
    items = lattice["groups"][0]["items"]

    # Each leaf has 2 non-family keys → 2 projections (size-only,
    # stage-only). Across two leaves: {size:7B}, {size:32B},
    # {stage:Base} (deduped — same {stage:Base} from both leaves).
    # Total new: 3.
    assert stats["concepts_synthesized"] == 3
    sigs = {tuple(sorted(it["identity"].items())) for it in items}
    assert (("family", "OLMo 3"), ("size", "7B")) in sigs
    assert (("family", "OLMo 3"), ("size", "32B")) in sigs
    assert (("family", "OLMo 3"), ("stage", "Base")) in sigs
    # Synthesized concepts have _generated flag
    gen = [it for it in items if it.get("_generated")]
    assert len(gen) == 3
    for it in gen:
        assert it["links"] == []
        assert it["aliases"]  # non-empty


def test_expand_concept_lattice_idempotent():
    """Running expand_concept_lattice twice adds nothing the second time."""
    from gdb.subsets import expand_concept_lattice

    lattice = {
        "groups": [{
            "family": "X",
            "identity_keys": ["family", "size", "stage"],
            "items": [
                {"kind": "model", "formal_name": "X",
                 "identity": {"family": "X"},
                 "aliases": ["X"], "links": [], "description": None},
                {"kind": "model", "formal_name": "X/Y-7B-Base",
                 "identity": {"family": "X", "size": "7B", "stage": "Base"},
                 "aliases": ["X/Y-7B-Base"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/X/Y-7B-Base"}],
                 "description": "..."},
            ],
        }],
    }
    s1 = expand_concept_lattice(lattice)
    n1 = len(lattice["groups"][0]["items"])
    s2 = expand_concept_lattice(lattice)
    n2 = len(lattice["groups"][0]["items"])
    assert s1["concepts_synthesized"] == 2  # {size:7B}, {stage:Base}
    assert s2["concepts_synthesized"] == 0
    assert n1 == n2


def test_flag_same_url_cross_family():
    """Same primary URL across DIFFERENT families → cross-family hint."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [
            {"family": "Dolmino", "identity_keys": ["family"],
             "items": [
                 {"kind": "dataset", "formal_name": "allenai/dolmino-mix-1124",
                  "identity": {"family": "Dolmino"},
                  "aliases": ["dolmino-mix-1124"],
                  "links": [{"kind": "hf_dataset",
                             "url": "https://huggingface.co/datasets/allenai/dolmino-mix-1124"}],
                  "description": "...", "subsets": []},
             ]},
            {"family": "OLMo 2", "identity_keys": ["family"],
             "items": [
                 {"kind": "dataset", "formal_name": "allenai/dolmino-mix-1124",
                  "identity": {"family": "OLMo 2"},
                  "aliases": ["allenai/dolmino-mix-1124"],
                  "links": [{"kind": "hf_dataset",
                             "url": "https://huggingface.co/datasets/allenai/dolmino-mix-1124"}],
                  "description": "...", "subsets": []},
             ]},
        ],
    }
    flag_audit_issues(lattice)
    h = [x for x in (lattice.get("audit_hints") or [])
         if x["kind"] == "same_url_cross_family"]
    assert len(h) == 1
    assert set(h[0]["families"]) == {"Dolmino", "OLMo 2"}
    assert "dolmino-mix-1124" in h[0]["shared_url"]


def test_flag_concept_with_no_entity():
    """Family with 2+ concepts but no entity → concept_with_no_entity."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "olmOCR",
            "identity_keys": ["family", "version"],
            "items": [
                {"kind": "model", "formal_name": "olmOCR",
                 "identity": {"family": "olmOCR"},
                 "aliases": ["olmOCR"],
                 "links": [{"kind": "hf_collection",
                            "url": "https://huggingface.co/collections/allenai/olmocr-x"}],
                 "description": "...", "subsets": []},
                {"kind": "model", "formal_name": "olmOCR 2",
                 "identity": {"family": "olmOCR", "version": "2"},
                 "aliases": ["olmOCR 2"],
                 "links": [{"kind": "blog",
                            "url": "https://allenai.org/blog/olmocr-2"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    flag_audit_issues(lattice)
    h = [x for x in (lattice.get("audit_hints") or [])
         if x["kind"] == "concept_with_no_entity"]
    assert len(h) == 1
    assert h[0]["family"] == "olmOCR"


def test_flag_family_root_invented_alias():
    """Family root with no alias from input pile → invented_alias hint
    (only fires when input_names_set is provided)."""
    from gdb.subsets import flag_audit_issues

    lattice = {
        "groups": [{
            "family": "Phi (Microsoft)",
            "identity_keys": ["family", "version", "size", "stage"],
            "items": [
                {"kind": "model", "formal_name": "Phi (Microsoft)",
                 "identity": {"family": "Phi (Microsoft)"},
                 "aliases": ["Phi (Microsoft)"],  # synthesized — not in input
                 "links": [], "description": None, "subsets": []},
                {"kind": "model", "formal_name": "microsoft/phi-4",
                 "identity": {"family": "Phi (Microsoft)", "version": "4"},
                 "aliases": ["Phi-4", "Phi 4"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/microsoft/phi-4"}],
                 "description": "...", "subsets": []},
            ],
        }],
    }
    input_names = {"Phi-4", "Phi 4", "Phi4-Mini-Instruct", "phi4-mini-instruct"}
    flag_audit_issues(lattice, input_names_set=input_names)
    h = [x for x in (lattice.get("audit_hints") or [])
         if x["kind"] == "family_root_invented_alias"]
    assert len(h) == 1
    assert h[0]["family"] == "Phi (Microsoft)"

    # Without input_names_set, the check is skipped
    lattice2 = {"groups": [lattice["groups"][0]]}
    lattice2["audit_hints"] = []
    flag_audit_issues(lattice2)  # no input_names_set
    h2 = [x for x in (lattice2.get("audit_hints") or [])
          if x["kind"] == "family_root_invented_alias"]
    assert len(h2) == 0


def test_expand_concept_lattice_skips_existing_projection():
    """If a projection's identity already matches an existing item
    (even one organize emitted as a source-mentioned concept),
    expansion skips it — no duplicate."""
    from gdb.subsets import expand_concept_lattice

    lattice = {
        "groups": [{
            "family": "Qwen3",
            "identity_keys": ["family", "size", "stage"],
            "items": [
                {"kind": "model", "formal_name": "Qwen3",
                 "identity": {"family": "Qwen3"},
                 "aliases": ["Qwen3"], "links": [], "description": None},
                # Source-mentioned concept (already materialized)
                {"kind": "model", "formal_name": "Qwen3-Base",
                 "identity": {"family": "Qwen3", "stage": "Base"},
                 "aliases": ["Qwen3-Base"], "links": [], "description": None},
                # Leaf
                {"kind": "model", "formal_name": "Qwen/Qwen3-4B-Base",
                 "identity": {"family": "Qwen3", "size": "4B", "stage": "Base"},
                 "aliases": ["Qwen/Qwen3-4B-Base"],
                 "links": [{"kind": "hf_model",
                            "url": "https://huggingface.co/Qwen/Qwen3-4B-Base"}],
                 "description": "..."},
            ],
        }],
    }
    stats = expand_concept_lattice(lattice)
    # Projections from the leaf: {size:4B} (new) + {stage:Base} (exists).
    # Only {size:4B} is synthesized.
    assert stats["concepts_synthesized"] == 1
    items = lattice["groups"][0]["items"]
    sigs = {tuple(sorted(it["identity"].items())) for it in items}
    assert (("family", "Qwen3"), ("size", "4B")) in sigs
    # Existing Qwen3-Base concept is preserved (not duplicated)
    base_concepts = [it for it in items
                     if it["identity"] == {"family": "Qwen3", "stage": "Base"}]
    assert len(base_concepts) == 1
    assert base_concepts[0]["formal_name"] == "Qwen3-Base"
    assert not base_concepts[0].get("_generated")
