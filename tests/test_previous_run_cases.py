from __future__ import annotations


def test_olmo3_broad_base_alias_conflict_is_flagged():
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "id": "base-7b",
            "surface": "Olmo-3-1025-7B",
            "kind": "model",
            "identity": {"family": "Olmo-3", "date": "1025", "size": "7B", "stage": "Base"},
            "aliases": [{"surface": "Olmo-3-Base", "descriptors": {}}],
            "anchors": [{"file": "paper.md", "excerpt": "Olmo 3 Base: Olmo-3-1025-7B Olmo-3-1125-32B"}],
        },
        {
            "id": "base-32b",
            "surface": "Olmo-3-1125-32B",
            "kind": "model",
            "identity": {"family": "Olmo-3", "date": "1125", "size": "32B", "stage": "Base"},
            "aliases": [{"surface": "Olmo-3-Base", "descriptors": {}}],
            "anchors": [{"file": "paper.md", "excerpt": "Olmo 3 Base: Olmo-3-1025-7B Olmo-3-1125-32B"}],
        },
    ])

    conflict = [v for v in violations if v["code"] == "surface_identity_conflict"]
    assert conflict
    assert conflict[0]["subject_key"] == "olmo-3-base"


def test_olmo3_stage_date_size_and_version_variants_remain_distinct():
    from gdb.lattice import build_lattice

    mentions = [
        {"surface": "allenai/Olmo-3-1025-7B", "kind": "model", "concept_path": ["Olmo-3", "7B", "Base"], "links": [{"type": "hf_model", "value": "allenai/Olmo-3-1025-7B"}], "aux": {"date": "1025"}, "anchors": [{"file": "m.md", "excerpt": "Olmo-3-1025-7B"}]},
        {"surface": "allenai/Olmo-3-1125-32B", "kind": "model", "concept_path": ["Olmo-3", "32B", "Base"], "links": [{"type": "hf_model", "value": "allenai/Olmo-3-1125-32B"}], "aux": {"date": "1125"}, "anchors": [{"file": "m.md", "excerpt": "Olmo-3-1125-32B"}]},
        {"surface": "allenai/Olmo-3-7B-Think-SFT", "kind": "model", "concept_path": ["Olmo-3", "7B", "Think-SFT"], "links": [{"type": "hf_model", "value": "allenai/Olmo-3-7B-Think-SFT"}], "anchors": [{"file": "m.md", "excerpt": "Olmo-3-7B-Think-SFT"}]},
        {"surface": "allenai/Olmo-3-7B-Think-DPO", "kind": "model", "concept_path": ["Olmo-3", "7B", "Think-DPO"], "links": [{"type": "hf_model", "value": "allenai/Olmo-3-7B-Think-DPO"}], "anchors": [{"file": "m.md", "excerpt": "Olmo-3-7B-Think-DPO"}]},
        {"surface": "allenai/Olmo-3.1-32B-Think", "kind": "model", "concept_path": ["Olmo-3.1", "32B", "Think"], "links": [{"type": "hf_model", "value": "allenai/Olmo-3.1-32B-Think"}], "anchors": [{"file": "m.md", "excerpt": "Olmo-3.1-32B-Think"}]},
    ]

    lattice = build_lattice(mentions)
    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    anchors = {node["identity"]["link_value"] for node in leaves}

    assert len(leaves) == 5
    assert "allenai/Olmo-3-1025-7B" in anchors
    assert "allenai/Olmo-3-1125-32B" in anchors
    assert "allenai/Olmo-3.1-32B-Think" in anchors


def test_smollm2_quantized_and_format_artifacts_share_concept_but_stay_entities():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "SmolLM2-360M-Instruct",
            "kind": "model",
            "concept_path": ["SmolLM2", "360M", "Instruct"],
            "links": [{"type": "hf_model", "value": "HuggingFaceTB/SmolLM2-360M-Instruct"}],
            "aliases": [{"surface": "SmolLM2-360M-Instruct", "descriptors": {}}],
            "anchors": [{"file": "card.md", "excerpt": "SmolLM2-360M-Instruct"}],
        },
        {
            "surface": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF",
            "kind": "model",
            "concept_path": ["SmolLM2", "360M", "Instruct"],
            "links": [{"type": "hf_model", "value": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF"}],
            "aliases": [
                {
                    "surface": "SmolLM2-360M-Instruct-Q8_0-GGUF",
                    "descriptors": {"quantization": "Q8_0", "format": "GGUF", "namespace": "ngxson"},
                }
            ],
            "links": [{"type": "hf_model", "value": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF", "exact": True}],
            "anchors": [{"file": "gguf.md", "excerpt": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF"}],
        },
        {
            "surface": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx",
            "kind": "model",
            "concept_path": ["SmolLM2", "360M", "Instruct"],
            "links": [{"type": "hf_model", "value": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx"}],
            "aliases": [
                {
                    "surface": "SmolLM2-360M-Instruct-Q8-mlx",
                    "descriptors": {"quantization": "Q8", "format": "MLX", "namespace": "reach-vb"},
                }
            ],
            "links": [{"type": "hf_model", "value": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx", "exact": True}],
            "anchors": [{"file": "mlx.md", "excerpt": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx"}],
        },
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert len(leaves) == 3
    assert {tuple(node["concept_path"]) for node in leaves} == {("SmolLM2", "360M", "Instruct")}
    aliases = {
        alias["surface"]: alias["descriptors"]
        for node in leaves
        for alias in node["aliases"]
    }
    assert aliases["SmolLM2-360M-Instruct-Q8_0-GGUF"]["format"] == "GGUF"
    assert aliases["SmolLM2-360M-Instruct-Q8-mlx"]["format"] == "MLX"


def test_smollm2_context_length_variant_is_identity_not_descriptor():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "kind": "model", "concept_path": ["SmolLM2", "1.7B", "Instruct"], "links": [{"type": "hf_model", "value": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}], "aux": {"context_length": "8k"}, "anchors": [{"file": "base.md", "excerpt": "SmolLM2-1.7B-Instruct"}]},
        {"surface": "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k", "kind": "model", "concept_path": ["SmolLM2", "1.7B", "Instruct"], "links": [{"type": "hf_model", "value": "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k"}], "aux": {"context_length": "16k"}, "anchors": [{"file": "16k.md", "excerpt": "SmolLM2-1.7B-Instruct-16k"}]},
    ])

    leaves = [node for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert len(leaves) == 2
    assert {node["aux"]["context_length"] for node in leaves} == {"8k", "16k"}


def test_finemath_parent_with_dataset_subsets_materializes_subset_nodes():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "HuggingFaceTB/finemath",
            "kind": "dataset",
            "concept_path": ["FineMath"],
            "links": [{"type": "hf_dataset", "value": "HuggingFaceTB/finemath"}],
            "subsets": [
                {
                    "name": "finemath-3plus",
                    "identity": {"family": "FineMath", "subset": "finemath-3plus", "quality_cut": "3+"},
                    "links": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}],
                    "anchors": [{"file": "card.md", "excerpt": "configs include finemath-3plus"}],
                },
                {
                    "name": "finemath-4plus",
                    "identity": {"family": "FineMath", "subset": "finemath-4plus", "quality_cut": "4+"},
                    "links": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-4plus"}],
                    "anchors": [{"file": "card.md", "excerpt": "configs include finemath-4plus"}],
                },
            ],
            "anchors": [{"file": "card.md", "excerpt": "HuggingFaceTB/finemath exposes FineMath configs."}],
        }
    ])

    anchors = [node["identity"]["link_value"] for node in lattice["nodes"] if node["node_type"] == "entity"]
    assert "HuggingFaceTB/finemath" in anchors
    assert "HuggingFaceTB/finemath::finemath-3plus" in anchors
    assert "HuggingFaceTB/finemath::finemath-4plus" in anchors


def test_finemath_same_hf_repo_parent_vs_subset_identity_is_review_conflict():
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "HuggingFaceTB/finemath",
            "kind": "dataset",
            "identity": {"family": "FineMath"},
            "links": [{"type": "hf_dataset", "value": "HuggingFaceTB/finemath", "exact": True}],
            "anchors": [{"file": "a.md", "excerpt": "HuggingFaceTB/finemath"}],
        },
        {
            "surface": "FineMath-3+",
            "kind": "dataset",
            "identity": {"family": "FineMath", "subset": "finemath-3plus"},
            "links": [{"type": "hf_dataset", "value": "HuggingFaceTB/finemath", "exact": True}],
            "anchors": [{"file": "b.md", "excerpt": "FineMath-3+ is in HuggingFaceTB/finemath"}],
        },
    ])

    assert any(v["code"] == "link_identity_conflict" for v in violations)


def test_qwen25_base_and_instruct_are_distinct_but_license_blob_link_is_invalid():
    from gdb.artifacts import detect_conflicts, validate_mention_artifact
    from gdb.lattice import build_lattice

    mentions = [
        {"surface": "Qwen/Qwen2.5-1.5B", "kind": "model", "identity": {"family": "Qwen2.5", "size": "1.5B"}, "links": [{"type": "hf_model", "value": "Qwen/Qwen2.5-1.5B", "exact": True}], "context_roles": ["base_model"], "anchors": [{"file": "card.md", "excerpt": "base_model: Qwen/Qwen2.5-1.5B"}]},
        {"surface": "Qwen/Qwen2.5-1.5B-Instruct", "kind": "model", "identity": {"family": "Qwen2.5", "size": "1.5B", "stage": "Instruct"}, "links": [{"type": "hf_model", "value": "Qwen/Qwen2.5-1.5B-Instruct", "exact": True}], "context_roles": ["teacher_model"], "anchors": [{"file": "recipe.py", "excerpt": "model_name = \"Qwen/Qwen2.5-1.5B-Instruct\""}]},
    ]
    lattice = build_lattice(mentions)
    assert len([node for node in lattice["nodes"] if not node["projection"]]) == 2
    assert not detect_conflicts(mentions)

    errors = validate_mention_artifact({
        "mentions": [
            {
                "surface": "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE",
                "kind": "model",
                "identity": {"family": "Qwen2.5", "size": "1.5B"},
                "links": [{"type": "hf_model", "value": "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE", "exact": True}],
                "anchors": [{"file": "card.md", "excerpt": "license_link: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE"}],
            }
        ]
    })
    assert any(error["code"] == "invalid_link_shape" for error in errors)
