from __future__ import annotations


def test_candidate_urls_for_hf_model_dataset_github_arxiv_and_official():
    from gdb.linker import candidate_url

    assert candidate_url("model", "hf_ids", "Qwen/Qwen3-7B") == "https://huggingface.co/Qwen/Qwen3-7B"
    assert candidate_url("dataset", "hf_ids", "HuggingFaceTB/finemath") == "https://huggingface.co/datasets/HuggingFaceTB/finemath"
    assert candidate_url("model", "github_repos", "allenai/OLMo") == "https://github.com/allenai/OLMo"
    assert candidate_url("model", "papers", "https://arxiv.org/abs/2501.12345") == "https://arxiv.org/abs/2501.12345"
    assert candidate_url("dataset", "official_urls", "https://example.com/x") == "https://example.com/x"


def test_verify_candidates_uses_mock_fetcher():
    from gdb.linker import LinkCandidate, verify_candidates

    seen = []

    def fetch(url):
        seen.append(url)
        return (url.endswith("/ok"), 200 if url.endswith("/ok") else 404, None)

    checks = verify_candidates([
        LinkCandidate("c1", "model", "official_urls", "https://example.com/ok", "https://example.com/ok"),
        LinkCandidate("c2", "model", "official_urls", "https://example.com/missing", "https://example.com/missing"),
    ], fetch=fetch)

    assert seen == ["https://example.com/ok", "https://example.com/missing"]
    assert [check["ok"] for check in checks] == [True, False]

