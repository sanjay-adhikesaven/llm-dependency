from __future__ import annotations


def _node_by_identity(lattice, family, **facets):
    expected = {"family": family, **facets}
    for node in lattice["nodes"]:
        if node["identity"] == expected:
            return node
    raise AssertionError(f"node not found: {expected}")


def test_qwen3_leaf_has_size_and_stage_cover_parents():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "Qwen3-7B-Base",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Base"},
            "evidence": [{"file": "card.md", "excerpt": "Qwen3-7B-Base"}],
        }
    ])

    leaf = _node_by_identity(lattice, "Qwen3", size="7B", stage="Base")
    parent_size = _node_by_identity(lattice, "Qwen3", size="7B")
    parent_stage = _node_by_identity(lattice, "Qwen3", stage="Base")
    parent_edges = {
        edge["parent_node_key"]
        for edge in lattice["edges"]
        if edge["child_node_key"] == leaf["node_key"]
    }
    assert parent_edges == {parent_size["node_key"], parent_stage["node_key"]}


def test_olmo3_size_date_stage_variants_do_not_collapse():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "OLMo-3-7B-Base-2025-07", "kind": "model", "identity": {"family": "OLMo-3", "size": "7B", "stage": "Base", "date": "2025-07"}, "evidence": [{"file": "r.md", "excerpt": "OLMo-3 7B Base"}]},
        {"surface": "OLMo-3-32B-Base-2025-07", "kind": "model", "identity": {"family": "OLMo-3", "size": "32B", "stage": "Base", "date": "2025-07"}, "evidence": [{"file": "r.md", "excerpt": "OLMo-3 32B Base"}]},
    ])

    leaves = [
        node for node in lattice["nodes"]
        if not node["projection"] and node["identity"].get("family") == "OLMo-3"
    ]
    assert len(leaves) == 2
    assert {node["identity"]["size"] for node in leaves} == {"7B", "32B"}


def test_dolma3_mix_siblings_are_incomparable_under_family():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "dolmino_mix-100B-1025", "kind": "dataset", "identity": {"family": "Dolma3", "mix_variant": "dolmino_mix", "size": "100B", "date": "1025"}, "evidence": [{"file": "d.md", "excerpt": "dolmino_mix"}]},
        {"surface": "longmino_mix-50B-1025", "kind": "dataset", "identity": {"family": "Dolma3", "mix_variant": "longmino_mix", "size": "50B", "date": "1025"}, "evidence": [{"file": "d.md", "excerpt": "longmino_mix 50B"}]},
        {"surface": "longmino_mix-100B-1125", "kind": "dataset", "identity": {"family": "Dolma3", "mix_variant": "longmino_mix", "size": "100B", "date": "1125"}, "evidence": [{"file": "d.md", "excerpt": "longmino_mix 100B"}]},
    ])

    leaves = [node for node in lattice["nodes"] if not node["projection"]]
    leaf_keys = {node["node_key"] for node in leaves}
    assert not any(edge["parent_node_key"] in leaf_keys and edge["child_node_key"] in leaf_keys for edge in lattice["edges"])


def test_finemath_subset_and_quality_cut_variants_are_distinct():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "FineMath-3+", "kind": "dataset", "identity": {"family": "FineMath", "subset": "3plus"}, "evidence": [{"file": "f.md", "excerpt": "FineMath-3+"}]},
        {"surface": "FineMath quality 4+", "kind": "dataset", "identity": {"family": "FineMath", "quality_cut": "4plus"}, "evidence": [{"file": "f.md", "excerpt": "FineMath quality 4+"}]},
    ])

    leaves = [node for node in lattice["nodes"] if not node["projection"]]
    assert len(leaves) == 2
    assert {tuple(sorted(node["identity"].items())) for node in leaves} == {
        (("family", "FineMath"), ("subset", "3plus")),
        (("family", "FineMath"), ("quality_cut", "4plus")),
    }

