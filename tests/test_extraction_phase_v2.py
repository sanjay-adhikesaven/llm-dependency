from __future__ import annotations


def test_quantization_suffix_collapses_to_alias_of_canonical():
    """LLM emits canonical + FP8 as alias under same identity; aggregate yields one cluster."""
    from gdb.artifacts import aggregate_mentions

    clusters = aggregate_mentions([
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True},
            ],
            "aliases": [
                {"surface": "Qwen3-7B-Instruct", "descriptors": {}},
                {
                    "surface": "Qwen3-7B-Instruct-FP8",
                    "descriptors": {"quantization": "FP8"},
                    "anchors": [
                        {"type": "hf_model", "value": "Org/Qwen3-7B-Instruct-FP8", "exact": True},
                    ],
                },
            ],
            "evidence": [{"file": "card.md", "excerpt": "Qwen3-7B-Instruct (and FP8 variant)"}],
        },
    ])

    assert len(clusters) == 1
    aliases_by_surface = {a["surface"]: a for a in clusters[0]["aliases"]}
    assert "Qwen3-7B-Instruct-FP8" in aliases_by_surface
    fp8 = aliases_by_surface["Qwen3-7B-Instruct-FP8"]
    assert fp8["descriptors"] == {"quantization": "FP8"}
    assert any(anchor.get("value") == "Org/Qwen3-7B-Instruct-FP8" for anchor in fp8.get("anchors") or [])


def test_alias_can_carry_its_own_hf_anchor():
    """normalize_aliases preserves per-alias anchor_candidates."""
    from gdb.artifacts import normalize_aliases

    aliases = normalize_aliases(
        [
            {
                "surface": "Qwen3-7B-Instruct-FP8",
                "descriptors": {"precision": "FP8"},
                "anchors": [{"type": "hf_model", "value": "Org/Qwen3-7B-Instruct-FP8"}],
            },
        ],
        kind="model",
    )

    assert len(aliases) == 1
    assert aliases[0]["surface"] == "Qwen3-7B-Instruct-FP8"
    assert aliases[0]["descriptors"] == {"precision": "FP8"}
    assert aliases[0]["anchors"][0]["type"] == "hf_model"
    assert aliases[0]["anchors"][0]["value"] == "Org/Qwen3-7B-Instruct-FP8"
    assert aliases[0]["anchors"][0]["url"] == "https://huggingface.co/Org/Qwen3-7B-Instruct-FP8"


def test_aux_conflict_flagged_not_silently_merged():
    """Same cluster (same primary anchor), conflicting aux values, emits aux_conflict."""
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True},
            ],
            "aux": {"context_length": "8192"},
            "evidence": [{"file": "a.md", "excerpt": "Qwen3-7B-Instruct supports 8192 tokens"}],
        },
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True},
            ],
            "aux": {"context_length": "16384"},
            "evidence": [{"file": "b.md", "excerpt": "Qwen3-7B-Instruct supports 16384 tokens"}],
        },
    ])

    aux_conflicts = [v for v in violations if v["code"] == "aux_conflict"]
    assert aux_conflicts, "aux_conflict violation not emitted"
    detail = aux_conflicts[0]["details"]
    assert detail["key"] == "context_length"
    assert {str(v) for v in detail["values"]} == {"8192", "16384"}


def test_olmo3_dates_distinguish_separate_entities():
    """Distinct identity.extra.date keeps OLMo-3-1025 and OLMo-3-1125 as separate clusters."""
    from gdb.artifacts import aggregate_mentions

    clusters = aggregate_mentions([
        {
            "surface": "Olmo-3-1025-7B-Base",
            "kind": "model",
            "identity": {"family": "Olmo-3", "size": "7B", "stage": "Base", "extra": {"date": "1025"}},
            "anchor_candidates": [
                {"type": "hf_model", "value": "allenai/Olmo-3-1025-7B", "exact": True},
            ],
            "evidence": [{"file": "a.md", "excerpt": "Olmo-3 1025 release"}],
        },
        {
            "surface": "Olmo-3-1125-7B-Base",
            "kind": "model",
            "identity": {"family": "Olmo-3", "size": "7B", "stage": "Base", "extra": {"date": "1125"}},
            "anchor_candidates": [
                {"type": "hf_model", "value": "allenai/Olmo-3-1125-7B", "exact": True},
            ],
            "evidence": [{"file": "b.md", "excerpt": "Olmo-3 1125 release"}],
        },
    ])

    assert len(clusters) == 2
    dates = {(c["identity"].get("extra") or {}).get("date") for c in clusters}
    assert dates == {"1025", "1125"}


def test_forest_manifest_partitions_multi_family_input_by_root():
    """Two unrelated families produce two forest entries, each with its own root."""
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "concept_path": ["Qwen3", "7B", "Instruct"],
            "anchor_candidates": [
                {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True},
            ],
            "evidence": [{"file": "q.md", "excerpt": "Qwen3-7B-Instruct"}],
        },
        {
            "surface": "Llama-3-8B-Instruct",
            "kind": "model",
            "identity": {"family": "Llama-3", "size": "8B", "stage": "Instruct"},
            "concept_path": ["Llama-3", "8B", "Instruct"],
            "anchor_candidates": [
                {"type": "hf_model", "value": "meta-llama/Llama-3-8B-Instruct", "exact": True},
            ],
            "evidence": [{"file": "l.md", "excerpt": "Llama-3-8B-Instruct"}],
        },
    ])

    forests = lattice.get("forests") or []
    assert len(forests) == 2
    root_names = {f["root_display_name"] for f in forests}
    assert root_names == {"Qwen3", "Llama-3"}
    qwen_forest = next(f for f in forests if f["root_display_name"] == "Qwen3")
    llama_forest = next(f for f in forests if f["root_display_name"] == "Llama-3")
    qwen_keys = {n["node_key"] for n in qwen_forest["nodes"]}
    llama_keys = {n["node_key"] for n in llama_forest["nodes"]}
    assert qwen_keys and llama_keys
    assert not qwen_keys & llama_keys


def test_lattice_audit_lists_bare_leaf_concepts():
    """Concept node with no entity child and no anchor evidence is flagged as bare leaf."""
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {
            "surface": "MysteryModel",
            "kind": "model",
            "identity": {"family": "MysteryModel"},
            "concept_path": ["MysteryModel"],
            "evidence": [{"file": "x.md", "excerpt": "We used MysteryModel."}],
        },
    ])

    audit = lattice.get("audit") or {}
    bare = audit.get("bare_leaf_concepts") or []
    assert any(item["display_name"] == "MysteryModel" for item in bare)


def test_lattice_audit_paper_only_concept_classified_advisory_not_error():
    """Entity with only verified paper_release anchor lands in entities_with_only_paper_anchors."""
    from gdb.lattice import build_lattice

    lattice = build_lattice(
        [
            {
                "surface": "AIME-2024",
                "kind": "dataset",
                "identity": {"family": "AIME-2024"},
                "concept_path": ["AIME-2024"],
                "anchor_candidates": [
                    {
                        "type": "paper_release",
                        "value": "https://arxiv.org/abs/2404.12345",
                        "exact": True,
                    },
                ],
                "evidence": [
                    {"file": "a.md", "excerpt": "AIME 2024 benchmark released in arxiv 2404.12345"},
                ],
            },
        ],
        link_checks=[
            {"link_kind": "paper_release", "link_value": "https://arxiv.org/abs/2404.12345", "ok": 1},
        ],
    )

    audit = lattice.get("audit") or {}
    only_paper = audit.get("entities_with_only_paper_anchors") or []
    assert any(item["display_name"] == "AIME-2024" for item in only_paper)


def test_should_be_alias_advisory_when_canonical_sibling_present():
    """Variant-suffix surface alongside canonical sibling emits should_be_alias warning."""
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "Qwen3-7B-Instruct",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Qwen/Qwen3-7B-Instruct", "exact": True},
            ],
            "evidence": [{"file": "a.md", "excerpt": "Qwen3-7B-Instruct"}],
        },
        {
            "surface": "Qwen3-7B-Instruct-FP8",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Org/Qwen3-7B-Instruct-FP8", "exact": True},
            ],
            "evidence": [{"file": "b.md", "excerpt": "Qwen3-7B-Instruct-FP8 quantized variant"}],
        },
    ])

    advisories = [v for v in violations if v["code"] == "should_be_alias"]
    assert advisories, "should_be_alias advisory not emitted"
    detail = advisories[0]["details"]
    assert detail["variant_surface"] == "Qwen3-7B-Instruct-FP8"
    assert detail["canonical_surface"] == "Qwen3-7B-Instruct"
    assert "FP8" in detail["suffixes"]
    assert advisories[0]["severity"] == "warning"


def test_should_be_alias_skipped_when_only_variant_present():
    """Variant surface alone (no canonical sibling) does NOT emit should_be_alias."""
    from gdb.artifacts import detect_conflicts

    violations = detect_conflicts([
        {
            "surface": "Qwen3-7B-Instruct-FP8",
            "kind": "model",
            "identity": {"family": "Qwen3", "size": "7B", "stage": "Instruct"},
            "anchor_candidates": [
                {"type": "hf_model", "value": "Org/Qwen3-7B-Instruct-FP8", "exact": True},
            ],
            "evidence": [{"file": "b.md", "excerpt": "Qwen3-7B-Instruct-FP8"}],
        },
    ])

    advisories = [v for v in violations if v["code"] == "should_be_alias"]
    assert not advisories


def test_dataset_github_canonical_hint_round_trips():
    """github_repo + hf_dataset (with mirror metadata) survive normalization."""
    from gdb.artifacts import normalize_anchor_candidates

    anchors = normalize_anchor_candidates(
        [
            {"type": "github_repo", "value": "microsoft/MASS", "exact": True},
            {
                "type": "hf_dataset",
                "value": "OtherOrg/MASS-mirror",
                "exact": True,
                "metadata": {"mirror": True},
            },
        ],
        kind="dataset",
    )

    types = {a["type"] for a in anchors}
    assert types == {"github_repo", "hf_dataset"}
    hf = next(a for a in anchors if a["type"] == "hf_dataset")
    assert hf.get("metadata", {}).get("mirror") is True
