from __future__ import annotations


def test_commit_check_and_build_lattice_roundtrip(fresh_runtime):
    from gdb.pipeline import commit_mentions, run_build_lattice, run_check
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
                "links": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Base", "exact": True}],
                "anchors": [{"file": "card.md", "excerpt": "Qwen3-7B-Base"}],
            }
        ]
    })

    assert result["status"] == "complete"
    assert run_check()["violation_count"] == 0
    lattice = run_build_lattice()
    # Mechanical tokenizer splits Qwen3 → [Qwen, 3], so the concept
    # spine for {family=Qwen3, size=7B, stage=Base} is
    # [Qwen, 3, 7B, Base] = 4 concept nodes + 1 entity leaf.
    assert lattice["node_count"] == 5
    assert lattice["edge_count"] == 4
    assert all_rows("SELECT COUNT(*) AS n FROM lattice_nodes")[0]["n"] == 5
    entity = all_rows("SELECT * FROM lattice_nodes WHERE node_type='entity'")[0]
    assert "Qwen/Qwen3-7B-Base" in entity["identity_json"]


def test_aggregate_collapses_surface_drift_clusters(fresh_runtime):
    """Mentions of the same artifact with surface drift (case, hyphens,
    spaces) should collapse to a single cluster after aggregation —
    NOT one cluster per surface variant."""
    from gdb.artifacts import aggregate_mentions

    mentions = [
        # Three surface forms of the same model — all drift, no links
        {"surface": "OLMo 3 7B", "kind": "model",
         "identity": {"family": "OLMo 3", "size": "7B"}, "atoms": ["OLMo 3 7B"],
         "anchors": [{"file": "a.md", "excerpt": "OLMo 3 7B"}]},
        {"surface": "Olmo 3 7B", "kind": "model",
         "identity": {"family": "Olmo 3", "size": "7B"}, "atoms": ["Olmo 3 7B"],
         "anchors": [{"file": "b.md", "excerpt": "Olmo 3 7B"}]},
        {"surface": "Olmo-3-7B", "kind": "model",
         "identity": {"family": "Olmo-3", "size": "7B"}, "atoms": ["Olmo-3-7B"],
         "anchors": [{"file": "c.md", "excerpt": "Olmo-3-7B"}]},
        # A genuinely different version that should NOT merge with the above
        {"surface": "OLMo 2 7B", "kind": "model",
         "identity": {"family": "OLMo 2", "size": "7B"}, "atoms": ["OLMo 2 7B"],
         "anchors": [{"file": "d.md", "excerpt": "OLMo 2 7B"}]},
        # Same kind/family-tier name as above (no size) — also distinct from sized variants
        {"surface": "OLMo 3", "kind": "model",
         "identity": {"family": "OLMo 3"}, "atoms": ["OLMo 3"],
         "anchors": [{"file": "e.md", "excerpt": "OLMo 3"}]},
    ]
    clusters = aggregate_mentions(mentions)
    # 3 distinct clusters expected: OLMo 3 7B (collapsed from 3), OLMo 2 7B, OLMo 3 (family-tier)
    assert len(clusters) == 3, f"expected 3 clusters, got {len(clusters)}: {[c['display_name'] for c in clusters]}"
    by_size = {c.get("identity", {}).get("size", "(none)"): c for c in clusters}
    assert "7B" in by_size  # at least one 7B cluster (the OLMo 3 collapse OR OLMo 2)
    # The OLMo 3 7B cluster (whichever family form won) should now have 3 anchors
    olmo3_7b = [c for c in clusters
                if c.get("identity", {}).get("size") == "7B"
                and ("3" in (c.get("identity", {}).get("family") or ""))]
    assert len(olmo3_7b) == 1
    assert olmo3_7b[0]["occurrence_count"] == 3
    assert len(olmo3_7b[0]["anchors"]) == 3


def test_aggregate_does_not_collapse_when_links_disagree(fresh_runtime):
    """If two drift-keyed clusters point at DIFFERENT canonical links,
    they're genuinely different entities — keep them distinct."""
    from gdb.artifacts import aggregate_mentions

    mentions = [
        {"surface": "Cosmopedia", "kind": "dataset",
         "identity": {"family": "Cosmopedia"},
         "links": [{"type": "hf_dataset", "value": "HuggingFaceTB/cosmopedia", "exact": True}],
         "anchors": [{"file": "a.md", "excerpt": "Cosmopedia"}]},
        {"surface": "cosmopedia", "kind": "dataset",
         "identity": {"family": "cosmopedia"},
         "links": [{"type": "hf_dataset", "value": "HuggingFaceTB/cosmopedia-v2", "exact": True}],
         "anchors": [{"file": "b.md", "excerpt": "cosmopedia"}]},
    ]
    clusters = aggregate_mentions(mentions)
    # Different canonical links → keep as 2 separate clusters
    assert len(clusters) == 2


def test_commit_mentions_routes_anchor_failures_to_rejected_table(fresh_runtime):
    """Hard validator failures (no anchors) route the offending mention
    to `rejected_mentions`; sibling mentions in the same artifact still
    commit successfully."""
    from gdb.pipeline import commit_mentions
    from gdb.store import all_rows, db, dumps, now

    with db() as conn:
        conn.execute(
            """INSERT INTO batches (id, label, summary, content_fingerprint, attrs, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            ("rj-batch", "rj", "", "fp-rj", dumps({}), now(), now()),
        )
        conn.commit()
    result = commit_mentions({
        "mentions": [
            # valid: full anchor
            {
                "surface": "Foo",
                "kind": "dataset",
                "identity": {"family": "Foo"},
                "links": [{"type": "hf_dataset", "value": "owner/foo", "exact": True}],
                "anchors": [{"file": "x.md", "source_id": "deadbeef", "location": "L1", "excerpt": "Foo"}],
            },
            # hard fail: no anchors
            {
                "surface": "Bar",
                "kind": "dataset",
                "identity": {"family": "Bar"},
                "anchors": [],
            },
            # soft fail: bad link shape — mention should still commit, link stripped
            {
                "surface": "C4",
                "kind": "dataset",
                "identity": {"family": "C4"},
                "links": [{"type": "hf_dataset", "value": "c4", "exact": True}],  # bare value
                "anchors": [{"file": "y.md", "source_id": "feedface", "location": "L2", "excerpt": "C4"}],
            },
        ]
    }, batch_id="rj-batch")

    assert result["status"] == "complete"
    assert result["mentions_committed"] == 2  # Foo + C4
    assert result["mentions_rejected"] == 1   # Bar (no anchors)
    assert result["rejected"][0]["surface"] == "Bar"
    assert "empty_anchors" in result["rejected"][0]["codes"]

    # The two valid mentions are in DB
    rows = all_rows("SELECT surface FROM mentions WHERE batch_id='rj-batch' ORDER BY surface")
    assert [r["surface"] for r in rows] == ["C4", "Foo"]
    # C4's bad link was stripped (soft repair)
    c4 = all_rows("SELECT links_json FROM mentions WHERE surface='C4'")[0]
    import json
    assert json.loads(c4["links_json"]) == []
    # Bar landed in rejected_mentions
    rejected = all_rows("SELECT * FROM rejected_mentions WHERE batch_id='rj-batch'")
    assert len(rejected) == 1
    assert rejected[0]["surface"] == "Bar"
    assert rejected[0]["status"] == "pending"


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
            {"id": "old", "surface": "Qwen3", "kind": "model", "concept_path": ["Qwen3"], "anchors": [{"excerpt": "Qwen3"}]},
        ]
    }, batch_id="batch-1")
    second = commit_mentions({
        "mentions": [
            {"id": "new", "surface": "Qwen3-4B", "kind": "model", "concept_path": ["Qwen3", "4B"], "anchors": [{"excerpt": "Qwen3-4B"}]},
        ]
    }, batch_id="batch-1")

    assert first["status"] == "complete"
    assert second["status"] == "complete"
    rows = all_rows("SELECT id, surface FROM mentions WHERE batch_id='batch-1'")
    assert rows == [{"id": "new", "surface": "Qwen3-4B"}]
