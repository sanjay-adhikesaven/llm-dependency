from __future__ import annotations


def test_commit_check_and_build_lattice_roundtrip(fresh_runtime):
    from gdb.pipeline import commit_mentions, run_build_lattice, run_check_mentions
    from gdb.store import all_rows, db

    with db():
        pass
    result = commit_mentions({
        "mentions": [
            {
                "id": "m1",
                "surface": "Qwen3-7B-Base",
                "kind": "model",
                "identity": {"family": "Qwen3", "size": "7B", "stage": "Base"},
                "links": {"hf_ids": ["Qwen/Qwen3-7B-Base"]},
                "evidence": [{"file": "card.md", "excerpt": "Qwen3-7B-Base"}],
            }
        ]
    })

    assert result["status"] == "complete"
    assert run_check_mentions()["violation_count"] == 0
    lattice = run_build_lattice()
    assert lattice["node_count"] == 4
    assert lattice["edge_count"] == 3
    assert all_rows("SELECT COUNT(*) AS n FROM lattice_nodes")[0]["n"] == 4
    entity = all_rows("SELECT * FROM lattice_nodes WHERE node_type='entity'")[0]
    assert "Qwen/Qwen3-7B-Base" in entity["identity_json"]


def test_commit_mentions_wipes_batch_before_recommit(fresh_runtime):
    from gdb.pipeline import commit_mentions
    from gdb.store import all_rows, db, dumps, now

    with db() as conn:
        conn.execute(
            """INSERT INTO batches (id, label, summary, content_fingerprint, attrs, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            ("batch-1", "b", "", "fp", dumps({}), now(), now()),
        )
        conn.commit()
    first = commit_mentions({
        "mentions": [
            {"id": "old", "surface": "Qwen3", "kind": "model", "concept_path": ["Qwen3"], "evidence": [{"excerpt": "Qwen3"}]},
        ]
    }, batch_id="batch-1")
    second = commit_mentions({
        "mentions": [
            {"id": "new", "surface": "Qwen3-4B", "kind": "model", "concept_path": ["Qwen3", "4B"], "evidence": [{"excerpt": "Qwen3-4B"}]},
        ]
    }, batch_id="batch-1")

    assert first["status"] == "complete"
    assert second["status"] == "complete"
    rows = all_rows("SELECT id, surface FROM mentions WHERE batch_id='batch-1'")
    assert rows == [{"id": "new", "surface": "Qwen3-4B"}]
