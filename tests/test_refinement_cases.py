from __future__ import annotations


def test_qwen3_4b_can_be_concept_and_exact_hf_entity_with_same_name():
    from gdb.artifacts import detect_conflicts
    from gdb.lattice import build_lattice

    mentions = [
        {
            "surface": "Qwen3-4B",
            "kind": "model",
            "atoms": ["Qwen3", "4B"],
            "referent_scope": "concept",
            "concept_path": ["Qwen3", "4B"],
            "anchors": [{"file": "paper.md", "excerpt": "Qwen3-4B models"}],
        },
        {
            "surface": "Qwen3-4B",
            "kind": "model",
            "atoms": ["Qwen3", "4B"],
            "referent_scope": "entity",
            "concept_path": ["Qwen3", "4B"],
            "links": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
            "anchors": [{"file": "serve.py", "excerpt": "model_name = \"Qwen/Qwen3-4B\""}],
        },
    ]

    assert not detect_conflicts(mentions)
    lattice = build_lattice(mentions)
    same_name = [node for node in lattice["nodes"] if node["display_name"] == "Qwen3-4B"]
    assert {node["node_type"] for node in same_name} == {"concept", "entity"}
    entity = next(node for node in same_name if node["node_type"] == "entity")
    assert entity["identity"] == {"link_type": "hf_model", "link_value": "Qwen/Qwen3-4B"}


def test_qwen_collection_boundaries_are_reviewed_paths_not_hyphen_rules():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "Qwen3-VL", "kind": "model", "atoms": ["Qwen3", "VL"], "concept_path": ["Qwen3", "VL"], "anchors": [{"excerpt": "Qwen3-VL collection"}]},
        {"surface": "Qwen3Guard", "kind": "model", "atoms": ["Qwen3", "Guard"], "concept_path": ["Qwen3", "Guard"], "anchors": [{"excerpt": "Qwen3Guard collection"}]},
        {"surface": "Qwen3.5", "kind": "model", "atoms": ["Qwen3.5"], "concept_path": ["Qwen3.5"], "anchors": [{"excerpt": "Qwen3.5 models"}]},
    ])

    concept_paths = {tuple(node["concept_path"]) for node in lattice["nodes"] if node["node_type"] == "concept"}
    assert ("Qwen3", "VL") in concept_paths
    assert ("Qwen3", "Guard") in concept_paths
    assert ("Qwen3.5",) in concept_paths


def test_anchor_linker_builds_hf_dataset_config_candidate():
    from gdb.linker import link_candidates_from_mentions

    candidates = link_candidates_from_mentions([
        {
            "surface": "finemath-3plus",
            "kind": "dataset",
            "concept_path": ["FineMath", "3plus"],
            "links": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}],
            "anchors": [{"excerpt": "finemath-3plus"}],
        }
    ])

    assert any(c.link_kind == "hf_dataset_config" and c.url == "https://huggingface.co/datasets/HuggingFaceTB/finemath" for c in candidates)
