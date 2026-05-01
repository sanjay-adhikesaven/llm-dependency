from __future__ import annotations


def test_prompt_render_includes_shared_context_and_variables(fresh_runtime):
    from gdb.pipeline import render_prompt

    prompt = render_prompt("extract-mentions", {
        "batch_dir": "/tmp/batch",
        "artifact_path": "/tmp/artifact.json",
        "input_path": "/tmp/input.json",
        "planner_model": "opus",
        "subagent_model": "sonnet",
    })

    assert "model and dataset mentions" in prompt
    assert "/tmp/batch" in prompt
    assert "/tmp/artifact.json" in prompt


def test_all_stage_prompts_render(fresh_runtime):
    from gdb.pipeline import render_prompt

    variables = {
        "target": "T",
        "workspace_dir": "/w",
        "artifact_path": "/a.json",
        "input_path": "/i.json",
        "planner_model": "opus",
        "subagent_model": "sonnet",
        "repair_packet_path": "/repair.json",
        "unresolved_clusters_path": "/unresolved.json",
        "run_id": "r",
        "worker_dir": "/workers",
        "batch_id": "b",
        "batch_dir": "/batch",
    }
    for stage in ["discover", "extract-mentions", "repair-mentions", "link-unresolved", "audit"]:
        assert render_prompt(stage, variables).strip()

