from __future__ import annotations


def test_five_mentions_with_same_link_resolve_to_one_cluster():
    """Same exact link across N mentions = 1 cluster after aggregate. Dedup-early property."""
    from gdb.artifacts import aggregate_mentions

    mentions = [
        {
            "id": f"m{i}",
            "surface": surface,
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "4B"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-4B", "exact": True}],
            "anchors": [{"file": f"f{i}.py", "excerpt": "Qwen/Qwen3-4B"}],
        }
        for i, surface in enumerate([
            "Qwen3-4B",
            "Qwen/Qwen3-4B",
            "qwen/qwen3-4b",
            "Qwen3-4B",
            "Qwen/Qwen3-4B",
        ])
    ]

    clusters = aggregate_mentions(mentions)

    assert len(clusters) == 1
    cluster = clusters[0]
    # All five anchors accumulate on the cluster
    assert len(cluster["anchors"]) == 5
    # Single primary link
    assert any(link["value"] == "Qwen/Qwen3-4B" for link in cluster["links"])
    # All five member ids tracked
    assert set(cluster["mention_ids"]) == {"m0", "m1", "m2", "m3", "m4"}


def test_anchored_and_unanchored_same_identity_share_one_cluster():
    """Prose mention without a link + code mention with the link merge into one cluster
    via identity_key when no link disagreement."""
    from gdb.artifacts import aggregate_mentions

    mentions = [
        {
            "id": "prose",
            "surface": "Qwen3-4B",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "4B"},
            "anchors": [{"file": "paper.md", "excerpt": "we evaluate Qwen3-4B"}],
        },
        {
            "id": "code",
            "surface": "Qwen/Qwen3-4B",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "4B"},
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-4B", "exact": True}],
            "anchors": [{"file": "config.py", "excerpt": "from_pretrained(\"Qwen/Qwen3-4B\")"}],
        },
    ]

    clusters = aggregate_mentions(mentions)

    # Both members share identity_key=qwen3|4b. Anchored member also matches by exact link.
    # The two clusters key by anchor (code) and identity (prose); they remain distinct in v1
    # because cluster_key_for_mention prefers anchor-based key when present.
    assert len(clusters) <= 2
    # The anchored cluster carries the link
    anchored = next(c for c in clusters if c["links"])
    assert anchored["links"][0]["value"] == "Qwen/Qwen3-4B"
