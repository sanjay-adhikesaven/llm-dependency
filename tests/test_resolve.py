"""Tests for lineage.resolve — the lattice resolution / search function."""

from lineage.resolve import resolve


def _lattice(*items_per_group):
    """Helper: build a tiny lattice from item dicts grouped by family."""
    groups = []
    for items in items_per_group:
        fam = items[0]["identity"]["family"]
        groups.append({"family": fam, "identity_keys": ["family"], "items": items})
    return {"groups": groups}


def test_exact_alias_match_returns_top1_leaf():
    """Mention exactly matches a leaf alias → leaf wins top-1."""
    lat = _lattice([
        {"kind": "model", "formal_name": "Qwen3",
         "identity": {"family": "Qwen3"}, "aliases": ["Qwen3"], "links": []},
        {"kind": "model", "formal_name": "Qwen/Qwen3-4B",
         "identity": {"family": "Qwen3", "size": "4B", "stage": "chat"},
         "aliases": ["Qwen/Qwen3-4B", "Qwen3-4B"],
         "links": [{"kind": "hf_model", "url": "https://huggingface.co/Qwen/Qwen3-4B"}]},
    ])
    cands = resolve("Qwen/Qwen3-4B", lat, k=3)
    assert cands
    assert cands[0]["formal_name"] == "Qwen/Qwen3-4B"
    assert cands[0]["address_form"] == "leaf"
    assert cands[0]["score"] >= 100


def test_concept_preferred_over_leaf_for_bare_mention():
    """Bare mention with no specifics → concept (partial spec) wins
    over the leaf when both have the alias. Framework-correct."""
    lat = _lattice([
        {"kind": "model", "formal_name": "Apertus",
         "identity": {"family": "Apertus"}, "aliases": ["Apertus"], "links": []},
        # Synthesized concept has the bare-form alias
        {"kind": "model", "formal_name": "Apertus 8B",
         "identity": {"family": "Apertus", "size": "8B"},
         "aliases": ["Apertus 8B"], "_generated": True, "links": []},
        # Leaf with full identity also has the bare alias (planner placement)
        {"kind": "model", "formal_name": "swiss-ai/Apertus-8B-2509",
         "identity": {"family": "Apertus", "size": "8B", "date": "2509"},
         "aliases": ["Apertus 8B", "swiss-ai/Apertus-8B-2509"],
         "links": [{"kind": "hf_model",
                    "url": "https://huggingface.co/swiss-ai/Apertus-8B-2509"}]},
    ])
    cands = resolve("Apertus 8B", lat, k=3)
    # Both concept and leaf have exact-alphanum match; concept wins
    # because leaf has more facets (lower coverage_i). Concept's
    # _generated penalty is small (-0.5).
    assert cands[0]["formal_name"] == "Apertus 8B"
    assert cands[0]["address_form"] == "concept"
    # Leaf is in top-3
    assert any(c["formal_name"] == "swiss-ai/Apertus-8B-2509" for c in cands)


def test_specific_mention_picks_leaf():
    """When mention has more specific tokens, leaf wins (via better
    item-coverage)."""
    lat = _lattice([
        {"kind": "model", "formal_name": "Qwen3",
         "identity": {"family": "Qwen3"}, "aliases": ["Qwen3"], "links": []},
        {"kind": "model", "formal_name": "Qwen/Qwen3-4B",
         "identity": {"family": "Qwen3", "size": "4B", "stage": "chat"},
         "aliases": ["Qwen/Qwen3-4B"],
         "links": [{"kind": "hf_model", "url": "https://huggingface.co/Qwen/Qwen3-4B"}]},
    ])
    # Mention has org prefix → leaf's exact match dominates
    cands = resolve("Qwen/Qwen3-4B", lat, k=2)
    assert cands[0]["formal_name"] == "Qwen/Qwen3-4B"


def test_subset_slug_match_surfaces_parent():
    """Mention exactly matches a subset slug in some parent's subsets[]
    → parent surfaced with subset_of payload, score 70 (below direct
    exact match of 100)."""
    lat = _lattice([
        {"kind": "dataset", "formal_name": "LLM360/MegaMath",
         "identity": {"family": "MegaMath"},
         "aliases": ["LLM360/MegaMath"],
         "links": [{"kind": "hf_dataset",
                    "url": "https://huggingface.co/datasets/LLM360/MegaMath"}],
         "subsets": ["web", "web-pro", "code", "qa"]},
    ])
    # Mention isn't in any item's surfaces, but IS a subset slug
    cands = resolve("web-pro", lat, k=3)
    assert cands
    assert cands[0]["address_form"] == "subset"
    assert cands[0]["formal_name"] == "LLM360/MegaMath"
    assert cands[0]["subset_of"]["slug"] == "web-pro"


def test_direct_match_beats_subset_match():
    """Common Crawl is its own family root AND appears as a subset
    slug in some parent's subsets[]. Direct family root wins."""
    lat = _lattice(
        [{"kind": "dataset", "formal_name": "Common Crawl",
          "identity": {"family": "Common Crawl"},
          "aliases": ["Common Crawl"], "links": [], "subsets": []}],
        [{"kind": "dataset", "formal_name": "allenai/dolma3_pool",
          "identity": {"family": "Dolma"}, "aliases": ["allenai/dolma3_pool"],
          "links": [{"kind": "hf_dataset",
                     "url": "https://huggingface.co/datasets/allenai/dolma3_pool"}],
          "subsets": ["Common Crawl", "olmOCR Science PDFs"]}],
    )
    cands = resolve("Common Crawl", lat, k=3)
    # Direct family root (score 113) wins over subset hint (score 70)
    assert cands[0]["formal_name"] == "Common Crawl"
    assert cands[0]["address_form"] == "root"
    # Subset candidate is also surfaced
    assert any(c["address_form"] == "subset" for c in cands)


def test_subset_via_suffix_pattern():
    """Mention with subset suffix (`-pool`, `-RefinedWeb`) where
    prefix matches an existing item → subset surfaces parent."""
    lat = _lattice([
        {"kind": "dataset", "formal_name": "DCLM",
         "identity": {"family": "DCLM"},
         "aliases": ["DCLM", "DCLM-pool", "DCLM-RefinedWeb"],
         "links": [{"kind": "github", "url": "https://github.com/mlfoundations/dclm"}],
         "subsets": []},
    ])
    cands = resolve("DCLM-pool", lat, k=3)
    # Direct alias match wins (score 100+); subset suffix detection
    # also fires as a secondary candidate
    assert cands[0]["formal_name"] == "DCLM"
    sub = next((c for c in cands if c["address_form"] == "subset"), None)
    assert sub is not None
    assert sub["subset_of"]["slug"] == "pool"


def test_no_match_returns_low_score():
    """Mention with no overlap returns low scores; relate falls back
    to free-text per the < 50 rule."""
    lat = _lattice([
        {"kind": "model", "formal_name": "Qwen3",
         "identity": {"family": "Qwen3"}, "aliases": ["Qwen3"], "links": []},
    ])
    cands = resolve("totally-unrelated-codename-xyz123", lat, k=3)
    # Either no candidates or all below score threshold
    if cands:
        assert all(c["score"] < 50 for c in cands)


def test_empty_mention_returns_empty():
    """Empty / whitespace mention → no candidates."""
    lat = _lattice([
        {"kind": "model", "formal_name": "X",
         "identity": {"family": "X"}, "aliases": ["X"], "links": []},
    ])
    assert resolve("", lat, k=3) == []
    assert resolve("   ", lat, k=3) == []


def test_match_reasons_populated():
    """Every returned candidate has non-empty match_reasons."""
    lat = _lattice([
        {"kind": "model", "formal_name": "Qwen3",
         "identity": {"family": "Qwen3"}, "aliases": ["Qwen3"], "links": []},
    ])
    cands = resolve("Qwen3", lat, k=1)
    assert cands
    assert isinstance(cands[0]["match_reasons"], list)
    assert len(cands[0]["match_reasons"]) > 0
