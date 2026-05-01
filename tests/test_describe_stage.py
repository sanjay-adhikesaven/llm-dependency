from __future__ import annotations


def test_describe_artifact_persists_entity_description(fresh_runtime):
    """Apply a describe artifact via run_describe; entity_descriptions row appears."""
    from gdb.pipeline import run_describe
    from gdb.store import all_rows, db

    with db():
        pass
    artifact = {
        "descriptions": [
            {
                "entity_key": "entity:model:hf_model:abcdef",
                "kind": "model",
                "display_name": "Qwen/Qwen3-4B",
                "links": [{"type": "hf_model", "value": "Qwen/Qwen3-4B"}],
                "description": "Qwen3-4B chat model.",
                "metadata": {"front_matter": {"pipeline_tag": "text-generation"}},
                "source": {"repo_url": "https://huggingface.co/Qwen/Qwen3-4B"},
            }
        ]
    }
    path = fresh_runtime / "describe.json"
    path.write_text(__import__("json").dumps(artifact))

    result = run_describe(artifact_path=str(path))

    assert result["description_count"] == 1
    rows = all_rows("SELECT entity_key, description, links_json FROM entity_descriptions")
    assert len(rows) == 1
    assert rows[0]["description"] == "Qwen3-4B chat model."
    assert "Qwen/Qwen3-4B" in rows[0]["links_json"]


def test_enrich_hf_link_parses_front_matter(monkeypatch):
    """enrich_hf_link extracts base_model and pipeline_tag from card YAML."""
    from gdb.enrich import enrich_hf_link

    readme = """---
pipeline_tag: text-generation
base_model: Qwen/Qwen3-4B-Base
license: apache-2.0
---
# Qwen3-4B
"""

    def fake_text(_url):
        return 200, readme, None

    def fake_json(_url):
        return 200, {"cardData": {}}, None

    enriched = enrich_hf_link(
        {"type": "hf_model", "value": "Qwen/Qwen3-4B"},
        fetch_text=fake_text,
        fetch_json=fake_json,
    )

    assert enriched["ok"] is True
    assert enriched["metadata"]["front_matter"]["base_model"] == "Qwen/Qwen3-4B-Base"
    assert enriched["metadata"]["front_matter"]["pipeline_tag"] == "text-generation"
    assert "base_model=Qwen/Qwen3-4B-Base" in enriched["description"]
