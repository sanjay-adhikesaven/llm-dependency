from __future__ import annotations


def test_audit_applies_link_updates(fresh_runtime):
    from gdb.pipeline import commit_mentions, run_audit
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m1",
        "surface": "infiwebmath-3plus",
        "kind": "dataset",
        "concept_path": ["InfiWebMath", "3plus"],
        "anchors": [{"excerpt": "infiwebmath-3plus"}],
    }]})
    artifact = {"updates": [{
        "mention_id": "m1",
        "links": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::infiwebmath-3plus"}],
    }]}
    path = fresh_runtime / "audit.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_audit(artifact_path=str(path))

    assert result["audited_mentions"] == 1
    row = all_rows("SELECT links_json FROM mentions WHERE id='m1'")[0]
    assert "infiwebmath-3plus" in row["links_json"]



