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
    assert lattice["edge_count"] == 4
    assert all_rows("SELECT COUNT(*) AS n FROM lattice_nodes")[0]["n"] == 4

