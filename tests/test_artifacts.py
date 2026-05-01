from __future__ import annotations


def test_identity_canonicalization_is_open_form_and_sorts_extra():
    from gdb.artifacts import canonical_identity, identity_signature

    ident = canonical_identity({
        "family": "  Qwen3  ",
        "size": "7B",
        "stage": "",
        "organization": "Qwen",
        "extra": {"z": " last ", "a": "", "m": "mid"},
    })

    assert ident == {"family": "Qwen3", "size": "7B", "organization": "Qwen", "extra": {"m": "mid", "z": "last"}}
    assert identity_signature("model", ident) == identity_signature("model", {"size": "7B", "family": "qwen3", "organization": "qwen", "extra": {"z": "last", "m": "mid"}})


def test_artifact_validation_blocks_non_model_dataset_and_empty_evidence():
    from gdb.artifacts import validate_mention_artifact

    errors = validate_mention_artifact({
        "mentions": [
            {"surface": "Apache-2.0", "kind": "license", "identity": {"family": "Apache-2.0"}, "evidence": [{"excerpt": "License Apache-2.0"}]},
            {"surface": "Qwen3", "kind": "model", "identity": {"family": "Qwen3"}, "evidence": []},
        ]
    })

    codes = {error["code"] for error in errors}
    assert "invalid_kind" in codes
    assert "empty_evidence" in codes


def test_alias_aggregation_preserves_alias_local_descriptors():
    from gdb.artifacts import aggregate_mentions

    clusters = aggregate_mentions([
        {
            "surface": "Qwen3-7B-Instruct-FP8",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "aliases": [{"surface": "Qwen3-7B-Instruct-FP8", "descriptors": {"precision": "FP8"}}],
            "evidence": [{"file": "card.md", "excerpt": "Qwen3-7B-Instruct-FP8 is available."}],
        },
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "aliases": [{"surface": "Qwen3-7B-Instruct", "descriptors": {}}],
            "evidence": [{"file": "card.md", "excerpt": "Qwen3-7B-Instruct is the base name."}],
        },
    ])

    assert len(clusters) == 1
    aliases = {alias["surface"]: alias["descriptors"] for alias in clusters[0]["aliases"]}
    assert aliases["Qwen3-7B-Instruct-FP8"] == {"precision": "FP8"}
    assert clusters[0]["display_name"] == "Qwen3-7B-Instruct"


def test_conflict_detection_same_surface_and_same_link():
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "FineMath-3+",
            "kind": "dataset",
            "identity": {"family": "FineMath", "subset": "3plus"},
            "links": {"hf_ids": ["HuggingFaceTB/finemath"]},
            "evidence": [{"file": "a.md", "excerpt": "FineMath-3+ was used."}],
        },
        {
            "surface": "FineMath-3+",
            "kind": "dataset",
            "identity": {"family": "FineMath", "quality_cut": "3plus"},
            "links": {"hf_ids": ["HuggingFaceTB/finemath"]},
            "evidence": [{"file": "b.md", "excerpt": "FineMath-3+ was used."}],
        },
    ])

    codes = {violation["code"] for violation in violations}
    assert "link_identity_conflict" in codes


def test_open_context_roles_and_exact_dataset_config_anchor():
    from gdb.artifacts import normalize_mention

    mention = normalize_mention({
        "surface": "finemath-3plus",
        "kind": "dataset",
        "concept_path": ["FineMath", "3plus"],
        "context_roles": ["preference_data_seed"],
        "anchor_candidates": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}],
        "evidence": [{"file": "cfg.yaml", "excerpt": "finemath-3plus"}],
    })

    assert mention["context_roles"] == ["preference_data_seed"]
    assert mention["referent_scope"] == "entity"
    assert mention["anchor_candidates"][0]["type"] == "hf_dataset_config"
