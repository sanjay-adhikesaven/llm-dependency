from __future__ import annotations


def test_qwen3_4b_can_be_concept_and_exact_hf_entity_with_same_name():
    """A concept-tier mention and an entity-tier mention with the same
    display name (`Qwen3-4B`) coexist as distinct lattice nodes. After
    mechanical tokenization the concept_path is [Qwen, 3, 4B], and the
    deepest concept node carries display_name `Qwen-3-4B` while the
    entity leaf retains the source-form display `Qwen3-4B`."""
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
    # Both concept and entity nodes exist for this artifact.
    types = {n["node_type"] for n in lattice["nodes"]}
    assert "concept" in types and "entity" in types
    entity = next(n for n in lattice["nodes"] if n["node_type"] == "entity")
    assert entity["identity"] == {"link_type": "hf_model", "link_value": "Qwen/Qwen3-4B"}
    # The deepest concept node sits at path [Qwen, 3, 4B]
    deepest = max((n for n in lattice["nodes"] if n["node_type"] == "concept"),
                  key=lambda n: len(n.get("concept_path") or []))
    assert deepest["concept_path"] == ["Qwen", "3", "4B"]


def test_qwen_collection_boundaries_are_reviewed_paths_not_hyphen_rules():
    """With mechanical tokenization, sibling collection branches under
    Qwen3 become children of the shared `Qwen / 3` prefix."""
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "Qwen3-VL", "kind": "model", "atoms": ["Qwen3", "VL"], "concept_path": ["Qwen3", "VL"], "anchors": [{"excerpt": "Qwen3-VL collection"}]},
        {"surface": "Qwen3Guard", "kind": "model", "atoms": ["Qwen3", "Guard"], "concept_path": ["Qwen3", "Guard"], "anchors": [{"excerpt": "Qwen3Guard collection"}]},
        {"surface": "Qwen3.5", "kind": "model", "atoms": ["Qwen3.5"], "concept_path": ["Qwen3.5"], "anchors": [{"excerpt": "Qwen3.5 models"}]},
    ])

    concept_paths = {tuple(node["concept_path"]) for node in lattice["nodes"] if node["node_type"] == "concept"}
    # After tokenization, Qwen3 → [Qwen, 3], so the VL/Guard branches
    # share the [Qwen, 3] prefix:
    assert ("Qwen", "3", "VL") in concept_paths
    assert ("Qwen", "3", "Guard") in concept_paths
    # Qwen3.5 keeps its dotted version as one atom: [Qwen, 3.5]
    assert ("Qwen", "3.5") in concept_paths


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
