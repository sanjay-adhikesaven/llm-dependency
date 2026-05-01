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
            "evidence": [{"file": "paper.md", "excerpt": "Qwen3-4B models"}],
        },
        {
            "surface": "Qwen3-4B",
            "kind": "model",
            "atoms": ["Qwen3", "4B"],
            "referent_scope": "entity",
            "concept_path": ["Qwen3", "4B"],
            "anchor_candidates": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
            "evidence": [{"file": "serve.py", "excerpt": "model_name = \"Qwen/Qwen3-4B\""}],
        },
    ]

    assert not detect_conflicts(mentions)
    lattice = build_lattice(mentions)
    same_name = [node for node in lattice["nodes"] if node["display_name"] == "Qwen3-4B"]
    assert {node["node_type"] for node in same_name} == {"concept", "entity"}
    entity = next(node for node in same_name if node["node_type"] == "entity")
    assert entity["identity"] == {"anchor_type": "hf_model", "anchor": "Qwen/Qwen3-4B"}


def test_qwen_collection_boundaries_are_reviewed_paths_not_hyphen_rules():
    from gdb.lattice import build_lattice

    lattice = build_lattice([
        {"surface": "Qwen3-VL", "kind": "model", "atoms": ["Qwen3", "VL"], "concept_path": ["Qwen3", "VL"], "evidence": [{"excerpt": "Qwen3-VL collection"}]},
        {"surface": "Qwen3Guard", "kind": "model", "atoms": ["Qwen3", "Guard"], "concept_path": ["Qwen3", "Guard"], "evidence": [{"excerpt": "Qwen3Guard collection"}]},
        {"surface": "Qwen3.5", "kind": "model", "atoms": ["Qwen3.5"], "concept_path": ["Qwen3.5"], "evidence": [{"excerpt": "Qwen3.5 models"}]},
    ])

    concept_paths = {tuple(node["concept_path"]) for node in lattice["nodes"] if node["node_type"] == "concept"}
    assert ("Qwen3", "VL") in concept_paths
    assert ("Qwen3", "Guard") in concept_paths
    assert ("Qwen3.5",) in concept_paths


def test_code_reference_helper_extracts_model_and_dataset_config_refs():
    from gdb.code_refs import extract_code_references

    refs = extract_code_references(
        'model = AutoModel.from_pretrained("Qwen/Qwen3-4B")\n'
        'data = load_dataset("HuggingFaceTB/finemath", "finemath-3plus")\n',
        file="recipe.py",
    )

    surfaces = {ref["surface"] for ref in refs}
    assert "Qwen/Qwen3-4B" in surfaces
    assert "HuggingFaceTB/finemath::finemath-3plus" in surfaces
    assert not any(ref["surface"] == "HuggingFaceTB/finemath" and ref["kind"] == "model" for ref in refs)
    dataset = next(ref for ref in refs if ref["surface"].startswith("HuggingFaceTB/finemath::"))
    assert dataset["anchor_candidates"][0]["type"] == "hf_dataset"
    assert dataset["anchor_candidates"][1]["type"] == "hf_dataset_config"


def test_code_reference_helper_avoids_common_regex_false_positives():
    from gdb.code_refs import extract_code_references

    refs = extract_code_references(
        'dataset = next(ref for ref in refs)\n'
        'url = "https://huggingface.co/datasets/HuggingFaceTB/finemath"\n',
        file="recipe.py",
    )

    assert not any(ref["surface"] == "next" for ref in refs)
    assert not any(ref["surface"] == "huggingface.co/datasets" for ref in refs)
    assert any(ref["surface"] == "HuggingFaceTB/finemath" and ref["kind"] == "dataset" for ref in refs)


def test_hf_front_matter_enrichment_parses_base_model():
    from gdb.enrich import enrich_hf_anchor

    readme = """---
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-4B-Base
license: apache-2.0
---
# Qwen3-4B
"""

    def fetch(_url):
        return 200, readme, None

    def fetch_json(_url):
        return 200, {"cardData": {}}, None

    enriched = enrich_hf_anchor({"type": "hf_model", "value": "Qwen/Qwen3-4B"}, fetch_text=fetch, fetch_json=fetch_json)

    assert enriched["metadata"]["front_matter"]["base_model"] == ["Qwen/Qwen3-4B-Base"]
    assert "base_model=Qwen/Qwen3-4B-Base" in enriched["description"]


def test_hf_collection_candidates_are_inferred_from_name_and_tag_overlap():
    from gdb.enrich import infer_collection_candidates, normalize_collection_payloads

    candidates = infer_collection_candidates(
        {"type": "hf_model", "value": "ExampleOrg/NewFamily-vision-12B"},
        {},
        {"tags": ["NewFamily", "text-generation"]},
    )
    slugs = {candidate["slug"] for candidate in candidates if candidate.get("slug")}

    assert "newfamily" in slugs
    assert "newfamily-vision" in slugs
    assert "text-generation" not in slugs
    assert any(candidate["query_type"] == "item" and "item=models%2FExampleOrg%2FNewFamily-vision-12B" in candidate["url"] for candidate in candidates)

    collections = normalize_collection_payloads([{
        "slug": "ExampleOrg/newfamily-vision-abc123",
        "title": "NewFamily Vision",
        "items": [{"item": {"id": "ExampleOrg/NewFamily-vision-12B", "type": "model"}}],
    }])
    assert collections[0]["url"] == "https://huggingface.co/collections/ExampleOrg/newfamily-vision-abc123"
    assert collections[0]["repos"] == ["ExampleOrg/NewFamily-vision-12B"]


def test_anchor_linker_builds_hf_dataset_config_candidate():
    from gdb.linker import link_candidates_from_mentions

    candidates = link_candidates_from_mentions([
        {
            "surface": "finemath-3plus",
            "kind": "dataset",
            "concept_path": ["FineMath", "3plus"],
            "anchor_candidates": [{"type": "hf_dataset_config", "value": "HuggingFaceTB/finemath::finemath-3plus"}],
            "evidence": [{"excerpt": "finemath-3plus"}],
        }
    ])

    assert any(c.link_kind == "hf_dataset_config" and c.url == "https://huggingface.co/datasets/HuggingFaceTB/finemath" for c in candidates)
