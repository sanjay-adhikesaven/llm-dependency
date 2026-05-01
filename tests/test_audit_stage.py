from __future__ import annotations


def test_audit_artifact_applies_link_update_to_cluster_member(fresh_runtime):
    """Audit stage with mention-keyed update writes new link on the member mention."""
    from gdb.pipeline import commit_mentions, run_audit
    from gdb.store import all_rows, db

    with db():
        pass
    commit_mentions({"mentions": [{
        "id": "m-prose",
        "surface": "Qwen3-4B",
        "kind": "model",
        "concept_path": ["Qwen3", "4B"],
        "anchors": [{"file": "paper.md", "excerpt": "Qwen3-4B"}],
    }]})
    artifact = {"updates": [{
        "mention_id": "m-prose",
        "links": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
    }]}
    path = fresh_runtime / "audit.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_audit(artifact_path=str(path))

    assert result["audited_mentions"] == 1
    row = all_rows("SELECT links_json FROM mentions WHERE id='m-prose'")[0]
    assert "Qwen/Qwen3-4B" in row["links_json"]


def test_audit_cluster_keyed_update_expands_to_all_members(fresh_runtime):
    """A cluster_key update fans out to every mention belonging to that cluster."""
    from gdb.artifacts import cluster_key_for_mention, normalize_mention
    from gdb.pipeline import commit_mentions, run_audit
    from gdb.store import all_rows, db

    with db():
        pass
    member_mentions = [
        {
            "id": "m1",
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True}],
            "anchors": [{"file": "a.md", "excerpt": "Qwen3-7B-Instruct"}],
        },
        {
            "id": "m2",
            "surface": "Qwen/Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True}],
            "anchors": [{"file": "b.py", "excerpt": "from_pretrained(\"Qwen/Qwen3-7B-Instruct\")"}],
        },
    ]
    commit_mentions({"mentions": member_mentions})
    cluster_key = cluster_key_for_mention(normalize_mention(member_mentions[0]))
    artifact = {"updates": [{
        "cluster_key": cluster_key,
        "aux": {"context_length": "32768"},
    }]}
    path = fresh_runtime / "audit.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_audit(artifact_path=str(path))

    assert result["audited_mentions"] == 2
    rows = all_rows("SELECT id, aux_json FROM mentions ORDER BY id")
    for row in rows:
        assert "32768" in row["aux_json"]


def test_cluster_packet_key_matches_cluster_key_for_mention(fresh_runtime):
    """cluster_packet builds member_mention_ids using cluster_key_for_mention.
    The keys it indexes by must match aggregate_mentions' cluster_key, or the
    member list is silently empty for exact-link clusters."""
    from gdb.pipeline import cluster_packet, commit_mentions
    from gdb.store import db

    with db():
        pass
    commit_mentions({"mentions": [
        {
            "id": "m1",
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True}],
            "anchors": [{"file": "a.md", "excerpt": "Qwen3-7B-Instruct"}],
        },
        {
            "id": "m2",
            "surface": "Qwen/Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True}],
            "anchors": [{"file": "b.py", "excerpt": "from_pretrained(\"Qwen/Qwen3-7B-Instruct\")"}],
        },
    ]})

    packet = cluster_packet()
    assert len(packet["clusters"]) == 1
    cluster = packet["clusters"][0]
    assert cluster["cluster_key"].startswith("model:link:")
    assert set(cluster["member_mention_ids"]) == {"m1", "m2"}
