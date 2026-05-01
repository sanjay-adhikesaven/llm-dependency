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
            "evidence": [{"file": "paper.md", "excerpt": "Olmo 3 Base: Olmo-3-1025-7B Olmo-3-1125-32B"}],
        },
        {
            "id": "base-32b",
            "surface": "Olmo-3-1125-32B",
            "kind": "model",
            "identity": {"family": "Olmo-3", "date": "1125", "size": "32B", "stage": "Base"},
            "aliases": [{"surface": "Olmo-3-Base", "descriptors": {}}],
            "evidence": [{"file": "paper.md", "excerpt": "Olmo 3 Base: Olmo-3-1025-7B Olmo-3-1125-32B"}],
        },
    ])

    conflict = [v for v in violations if v["code"] == "surface_identity_conflict"]
    assert conflict
    assert conflict[0]["subject_key"] == "olmo-3-base"


def test_olmo3_stage_date_size_and_version_variants_remain_distinct():
    from gdb.lattice import build_lattice

    mentions = [
        {"surface": "Olmo-3-1025-7B", "kind": "model", "identity": {"family": "Olmo-3", "date": "1025", "size": "7B", "stage": "Base"}, "evidence": [{"file": "m.md", "excerpt": "Olmo-3-1025-7B"}]},
        {"surface": "Olmo-3-1125-32B", "kind": "model", "identity": {"family": "Olmo-3", "date": "1125", "size": "32B", "stage": "Base"}, "evidence": [{"file": "m.md", "excerpt": "Olmo-3-1125-32B"}]},
        {"surface": "Olmo-3-7B-Think-SFT", "kind": "model", "identity": {"family": "Olmo-3", "size": "7B", "stage": "Think-SFT"}, "evidence": [{"file": "m.md", "excerpt": "Olmo-3-7B-Think-SFT"}]},
        {"surface": "Olmo-3-7B-Think-DPO", "kind": "model", "identity": {"family": "Olmo-3", "size": "7B", "stage": "Think-DPO"}, "evidence": [{"file": "m.md", "excerpt": "Olmo-3-7B-Think-DPO"}]},
        {"surface": "Olmo-3.1-32B-Think", "kind": "model", "identity": {"family": "Olmo-3.1", "size": "32B", "stage": "Think"}, "evidence": [{"file": "m.md", "excerpt": "Olmo-3.1-32B-Think"}]},
    ]

    lattice = build_lattice(mentions)
    leaves = [node for node in lattice["nodes"] if not node["projection"]]
    identities = {tuple(sorted(node["identity"].items())) for node in leaves}

    assert len(leaves) == 5
    assert (("date", "1025"), ("family", "Olmo-3"), ("size", "7B"), ("stage", "Base")) in identities
    assert (("date", "1125"), ("family", "Olmo-3"), ("size", "32B"), ("stage", "Base")) in identities
    assert (("family", "Olmo-3.1"), ("size", "32B"), ("stage", "Think")) in identities


def test_smollm2_quantized_and_format_aliases_merge_but_preserve_descriptors():
    from gdb.artifacts import aggregate_mentions

    clusters = aggregate_mentions([
        {
            "surface": "SmolLM2-360M-Instruct",
            "kind": "model",
            "identity": {"family": "SmolLM2", "size": "360M", "stage": "Instruct"},
            "aliases": [{"surface": "SmolLM2-360M-Instruct", "descriptors": {}}],
            "evidence": [{"file": "card.md", "excerpt": "SmolLM2-360M-Instruct"}],
        },
        {
            "surface": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF",
            "kind": "model",
            "identity": {"family": "SmolLM2", "size": "360M", "stage": "Instruct"},
            "aliases": [
                {
                    "surface": "SmolLM2-360M-Instruct-Q8_0-GGUF",
                    "descriptors": {"quantization": "Q8_0", "format": "GGUF", "namespace": "ngxson"},
                }
            ],
            "links": {"hf_ids": ["ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF"]},
            "evidence": [{"file": "gguf.md", "excerpt": "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF"}],
        },
        {
            "surface": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx",
            "kind": "model",
            "identity": {"family": "SmolLM2", "size": "360M", "stage": "Instruct"},
            "aliases": [
                {
                    "surface": "SmolLM2-360M-Instruct-Q8-mlx",
                    "descriptors": {"quantization": "Q8", "format": "MLX", "namespace": "reach-vb"},
                }
            ],
            "links": {"hf_ids": ["reach-vb/SmolLM2-360M-Instruct-Q8-mlx"]},
            "evidence": [{"file": "mlx.md", "excerpt": "reach-vb/SmolLM2-360M-Instruct-Q8-mlx"}],
        },
    ])

    assert len(clusters) == 1
    aliases = {alias["surface"]: alias["descriptors"] for alias in clusters[0]["aliases"]}
    assert aliases["SmolLM2-360M-Instruct-Q8_0-GGUF"]["format"] == "GGUF"
    assert aliases["SmolLM2-360M-Instruct-Q8-mlx"]["format"] == "MLX"


def test_smollm2_context_length_variant_is_identity_not_descriptor():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "SmolLM2-1.7B-Instruct", "kind": "model", "identity": {"family": "SmolLM2", "size": "1.7B", "stage": "Instruct", "context_length": "8k"}, "evidence": [{"file": "base.md", "excerpt": "SmolLM2-1.7B-Instruct"}]},
        {"surface": "SmolLM2-1.7B-Instruct-16k", "kind": "model", "identity": {"family": "SmolLM2", "size": "1.7B", "stage": "Instruct", "context_length": "16k"}, "evidence": [{"file": "16k.md", "excerpt": "SmolLM2-1.7B-Instruct-16k"}]},
    ])

    leaves = [node for node in lattice["nodes"] if not node["projection"]]
    assert len(leaves) == 2
    assert {node["identity"]["context_length"] for node in leaves} == {"8k", "16k"}


def test_finemath_parent_with_dataset_subsets_materializes_subset_nodes():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "HuggingFaceTB/finemath",
            "kind": "dataset",
            "identity": {"family": "FineMath"},
            "links": {"hf_ids": ["HuggingFaceTB/finemath"]},
            "subsets": [
                {
                    "name": "finemath-3plus",
                    "identity": {"subset": "finemath-3plus", "quality_cut": "3+"},
                    "evidence": [{"file": "card.md", "excerpt": "configs include finemath-3plus"}],
                },
                {
                    "name": "finemath-4plus",
                    "identity": {"subset": "finemath-4plus", "quality_cut": "4+"},
                    "evidence": [{"file": "card.md", "excerpt": "configs include finemath-4plus"}],
                },
            ],
            "evidence": [{"file": "card.md", "excerpt": "HuggingFaceTB/finemath exposes FineMath configs."}],
        }
    ])

    identities = [node["identity"] for node in lattice["nodes"] if not node["projection"]]
    assert {"family": "FineMath"} in identities
    assert {"family": "FineMath", "subset": "finemath-3plus", "quality_cut": "3+"} in identities
    assert {"family": "FineMath", "subset": "finemath-4plus", "quality_cut": "4+"} in identities


def test_finemath_same_hf_repo_parent_vs_subset_identity_is_review_conflict():
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "HuggingFaceTB/finemath",
            "kind": "dataset",
            "identity": {"family": "FineMath"},
            "links": {"hf_ids": ["HuggingFaceTB/finemath"]},
            "evidence": [{"file": "a.md", "excerpt": "HuggingFaceTB/finemath"}],
        },
        {
            "surface": "FineMath-3+",
            "kind": "dataset",
            "identity": {"family": "FineMath", "subset": "finemath-3plus"},
            "links": {"hf_ids": ["HuggingFaceTB/finemath"]},
            "evidence": [{"file": "b.md", "excerpt": "FineMath-3+ is in HuggingFaceTB/finemath"}],
        },
    ])

    assert any(v["code"] == "link_identity_conflict" for v in violations)


def test_qwen25_base_and_instruct_are_distinct_but_license_blob_link_is_invalid():
    from gdb.artifacts import detect_conflicts, validate_mention_artifact
    from gdb.lattice import build_lattice

    mentions = [
        {"surface": "Qwen/Qwen2.5-1.5B", "kind": "model", "identity": {"family": "Qwen2.5", "size": "1.5B"}, "links": {"hf_ids": ["Qwen/Qwen2.5-1.5B"]}, "context_roles": ["base_model"], "evidence": [{"file": "card.md", "excerpt": "base_model: Qwen/Qwen2.5-1.5B"}]},
        {"surface": "Qwen/Qwen2.5-1.5B-Instruct", "kind": "model", "identity": {"family": "Qwen2.5", "size": "1.5B", "stage": "Instruct"}, "links": {"hf_ids": ["Qwen/Qwen2.5-1.5B-Instruct"]}, "context_roles": ["teacher_model"], "evidence": [{"file": "recipe.py", "excerpt": "model_name = \"Qwen/Qwen2.5-1.5B-Instruct\""}]},
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
                "links": {"hf_ids": ["https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE"]},
                "evidence": [{"file": "card.md", "excerpt": "license_link: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE"}],
            }
        ]
    })
    assert any(error["code"] == "invalid_link_shape" for error in errors)

