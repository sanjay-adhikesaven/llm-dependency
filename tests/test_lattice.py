from __future__ import annotations


def _node_by_type_and_name(lattice, node_type, display_name):
    for node in lattice["nodes"]:
        if node["node_type"] == node_type and node["display_name"] == display_name:
            return node
    raise AssertionError(f"node not found: {node_type}:{display_name}")


def test_qwen3_entity_leaf_attaches_to_reviewed_path_not_powerset():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "Qwen/Qwen3-7B-Base",
            "kind": "model",
            "concept_path": ["Qwen3", "7B", "Base"],
            "anchor_candidates": [{"type": "hf_model", "value": "Qwen/Qwen3-7B-Base"}],
            "aliases": [{"surface": "Qwen3-7B-Base"}],
            "evidence": [{"file": "card.md", "excerpt": "Qwen/Qwen3-7B-Base"}],
        }
    ])

    leaf = _node_by_type_and_name(lattice, "entity", "Qwen3-7B-Base")
    parent = _node_by_type_and_name(lattice, "concept", "Qwen3-7B-Base")
    assert leaf["identity"] == {"anchor_type": "hf_model", "anchor": "Qwen/Qwen3-7B-Base"}
    assert {edge["parent_node_key"] for edge in lattice["edges"] if edge["child_node_key"] == leaf["node_key"]} == {parent["node_key"]}
    assert len([node for node in lattice["nodes"] if node["node_type"] == "concept"]) == 3


def test_olmo3_size_date_stage_variants_do_not_collapse():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "allenai/Olmo-3-1025-7B", "kind": "model", "concept_path": ["Olmo-3", "7B", "Base"], "anchor_candidates": [{"type": "hf_model", "value": "allenai/Olmo-3-1025-7B"}], "aux": {"date": "1025"}, "evidence": [{"file": "r.md", "excerpt": "Olmo-3 7B Base"}]},
        {"surface": "allenai/Olmo-3-1125-32B", "kind": "model", "concept_path": ["Olmo-3", "32B", "Base"], "anchor_candidates": [{"type": "hf_model", "value": "allenai/Olmo-3-1125-32B"}], "aux": {"date": "1125"}, "evidence": [{"file": "r.md", "excerpt": "Olmo-3 32B Base"}]},
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert len(leaves) == 2
    assert {node["identity"]["anchor"] for node in leaves} == {"allenai/Olmo-3-1025-7B", "allenai/Olmo-3-1125-32B"}


def test_dolma3_mix_siblings_are_incomparable_entities_under_paths():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "allenai/dolma3_dolmino_mix-100B-1025", "kind": "dataset", "concept_path": ["Dolma3", "dolmino"], "anchor_candidates": [{"type": "hf_dataset", "value": "allenai/dolma3_dolmino_mix-100B-1025"}], "aux": {"mix_size": "100B", "date": "1025"}, "evidence": [{"file": "d.md", "excerpt": "dolmino_mix"}]},
        {"surface": "allenai/dolma3_longmino_mix-50B-1025", "kind": "dataset", "concept_path": ["Dolma3", "longmino"], "anchor_candidates": [{"type": "hf_dataset", "value": "allenai/dolma3_longmino_mix-50B-1025"}], "aux": {"mix_size": "50B", "date": "1025"}, "evidence": [{"file": "d.md", "excerpt": "longmino_mix 50B"}]},
        {"surface": "allenai/dolma3_longmino_mix-100B-1125", "kind": "dataset", "concept_path": ["Dolma3", "longmino"], "anchor_candidates": [{"type": "hf_dataset", "value": "allenai/dolma3_longmino_mix-100B-1125"}], "aux": {"mix_size": "100B", "date": "1125"}, "evidence": [{"file": "d.md", "excerpt": "longmino_mix 100B"}]},
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    leaf_keys = {node["node_key"] for node in leaves}
    assert len(leaves) == 3
    assert not any(edge["parent_node_key"] in leaf_keys and edge["child_node_key"] in leaf_keys for edge in lattice["edges"])


def test_finemath_config_anchors_are_distinct_from_parent_repo():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "HuggingFaceTB/finemath", "kind": "dataset", "concept_path": ["FineMath"], "anchor_candidates": [{"type": "hf_dataset", "value": "HuggingFaceTB/finemath"}], "evidence": [{"file": "f.md", "excerpt": "HuggingFaceTB/finemath"}]},
        {"surface": "finemath-3plus", "kind": "dataset", "concept_path": ["FineMath", "3plus"], "anchor_candidates": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}], "evidence": [{"file": "f.md", "excerpt": "finemath-3plus"}]},
        {"surface": "finemath-4plus", "kind": "dataset", "concept_path": ["FineMath", "4plus"], "anchor_candidates": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-4plus"}], "evidence": [{"file": "f.md", "excerpt": "finemath-4plus"}]},
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert {node["identity"]["anchor"] for node in leaves} == {
        "HuggingFaceTB/finemath",
        "HuggingFaceTB/finemath::finemath-3plus",
        "HuggingFaceTB/finemath::finemath-4plus",
    }


def test_dataset_config_primary_anchor_prevents_repo_level_collapse():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "HuggingFaceTB/finemath::finemath-3plus",
            "kind": "dataset",
            "concept_path": ["FineMath", "3plus"],
            "anchor_candidates": [
                {"type": "hf_dataset", "value": "HuggingFaceTB/finemath"},
                {"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"},
            ],
            "evidence": [{"file": "f.md", "excerpt": "load_dataset('HuggingFaceTB/finemath', 'finemath-3plus')"}],
        },
        {
            "surface": "HuggingFaceTB/finemath::finemath-4plus",
            "kind": "dataset",
            "concept_path": ["FineMath", "4plus"],
            "anchor_candidates": [
                {"type": "hf_dataset", "value": "HuggingFaceTB/finemath"},
                {"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-4plus"},
            ],
            "evidence": [{"file": "f.md", "excerpt": "load_dataset('HuggingFaceTB/finemath', 'finemath-4plus')"}],
        },
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert {node["identity"]["anchor"] for node in leaves} == {
        "HuggingFaceTB/finemath::finemath-3plus",
        "HuggingFaceTB/finemath::finemath-4plus",
    }
