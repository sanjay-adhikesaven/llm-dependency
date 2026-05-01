from __future__ import annotations


def _init_batch():
    from gdb.store import db, dumps, now

    with db() as conn:
        conn.execute(
            """INSERT INTO batches (id, label, summary, content_fingerprint, attrs, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            ("batch-1", "b", "", "fp", dumps({}), now(), now()),
        )
        conn.commit()


def test_investigate_hf_persists_metadata_and_applies_base_model(monkeypatch, fresh_runtime):
    import gdb.pipeline as pipeline
    from gdb.pipeline import commit_mentions, mention_rows, run_investigate_hf
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m1",
        "surface": "Qwen/Qwen3-4B",
        "kind": "model",
        "concept_path": ["Qwen3", "4B"],
        "anchor_candidates": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
        "evidence": [{"excerpt": "model_name = \"Qwen/Qwen3-4B\""}],
    }]})

    def fake_enrich(anchor):
        return {
            "anchor": anchor,
            "ok": True,
            "repo_url": "https://huggingface.co/Qwen/Qwen3-4B",
            "readme_url": "https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md",
            "api_url": "https://huggingface.co/api/models/Qwen%2FQwen3-4B",
            "metadata": {"front_matter": {"base_model": ["Qwen/Qwen3-4B-Base"]}},
            "card_data": {},
            "configs": [],
            "collections": [{"title": "Qwen3", "repos": ["Qwen/Qwen3-4B"]}],
            "relationships": [{"relation": "base_model", "target": "Qwen/Qwen3-4B-Base"}],
            "description": "Qwen3-4B; base_model=Qwen/Qwen3-4B-Base",
            "error": "",
        }

    monkeypatch.setattr(pipeline, "enrich_hf_anchor", fake_enrich)
    result = run_investigate_hf()

    assert result["ok_count"] == 1
    assert all_rows("SELECT COUNT(*) AS n FROM hf_metadata")[0]["n"] == 1
    mention = mention_rows()[0]
    assert mention["aux"]["base_model"] == ["Qwen/Qwen3-4B-Base"]
    assert mention["relationships"][0]["relation"] == "base_model"


def test_review_artifact_upserts_family_policy(fresh_runtime):
    from gdb.pipeline import commit_mentions, run_review_entities
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m1",
        "surface": "dolma3_longmino_mix-100B-1125",
        "kind": "dataset",
        "atoms": ["dolma3", "longmino", "mix", "100B", "1125"],
        "concept_path": ["Dolma3"],
        "evidence": [{"excerpt": "dolma3_longmino_mix-100B-1125"}],
    }]})
    artifact = {"updates": [{
        "mention_id": "m1",
        "kind": "dataset",
        "concept_path": ["Dolma3", "longmino"],
        "aux": {"mix_size": "100B", "date": "1125"},
    }]}
    path = fresh_runtime / "review.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_review_entities(artifact_path=str(path))

    assert result["family_policies_upserted"] == 1
    policy = all_rows("SELECT * FROM family_policies")[0]
    assert "longmino" in policy["policy_json"]


def test_build_relationships_materializes_hf_base_model_hint(monkeypatch, fresh_runtime):
    import gdb.pipeline as pipeline
    from gdb.pipeline import commit_mentions, run_build_relationships, run_investigate_hf
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m1",
        "surface": "Qwen/Qwen3-4B",
        "kind": "model",
        "concept_path": ["Qwen3", "4B"],
        "anchor_candidates": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
        "evidence": [{"excerpt": "Qwen/Qwen3-4B"}],
    }]})

    def fake_enrich(anchor):
        return {
            "anchor": anchor, "ok": True, "repo_url": "", "readme_url": "", "api_url": "",
            "metadata": {"front_matter": {"base_model": ["Qwen/Qwen3-4B-Base"]}},
            "card_data": {}, "configs": [], "collections": [],
            "relationships": [{"relation": "base_model", "target": "Qwen/Qwen3-4B-Base"}],
            "description": "Qwen3-4B", "error": "",
        }

    monkeypatch.setattr(pipeline, "enrich_hf_anchor", fake_enrich)
    run_investigate_hf()
    result = run_build_relationships()

    assert result["relationship_count"] >= 1
    assert all_rows("SELECT relation, target_name FROM entity_relationships")[0]["relation"] == "base_model"


def test_link_unresolved_applies_anchor_updates(fresh_runtime):
    from gdb.pipeline import commit_mentions, run_link_unresolved
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m1",
        "surface": "infiwebmath-3plus",
        "kind": "dataset",
        "concept_path": ["InfiWebMath", "3plus"],
        "evidence": [{"excerpt": "infiwebmath-3plus"}],
    }]})
    artifact = {"updates": [{
        "mention_id": "m1",
        "anchors": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::infiwebmath-3plus"}],
    }]}
    path = fresh_runtime / "links.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_link_unresolved(artifact_path=str(path))

    assert result["updated_mentions"] == 1
    row = all_rows("SELECT anchor_candidates_json FROM mentions WHERE id='m1'")[0]
    assert "infiwebmath-3plus" in row["anchor_candidates_json"]


def test_paper_anchor_requires_exact_release_evidence():
    from gdb.artifacts import validate_mention_artifact

    errors = validate_mention_artifact({"mentions": [{
        "surface": "Qwen3 technical report",
        "kind": "model",
        "concept_path": ["Qwen3"],
        "anchor_candidates": [{"type": "paper_release", "value": "https://arxiv.org/abs/2505.09388"}],
        "evidence": [{"excerpt": "Qwen3 technical report describes the model family."}],
    }]})

    assert any(error["code"] == "paper_anchor_not_exact_release" for error in errors)

