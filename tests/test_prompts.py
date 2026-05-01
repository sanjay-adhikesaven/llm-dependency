from __future__ import annotations


def test_prompt_render_includes_shared_context_and_variables(fresh_runtime):
    from gdb.pipeline import render_prompt

    prompt = render_prompt("extract", {
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
    from gdb import config
    from gdb.pipeline import render_prompt

    variables = {
        "target": "T",
        "workspace_dir": "/w",
        "artifact_path": "/a.json",
        "input_path": "/i.json",
        "planner_model": "opus",
        "subagent_model": "sonnet",
        "cluster_packet_path": "/cluster.json",
        "lattice_path": "/lattice.json",
        "run_id": "r",
        "worker_dir": "/workers",
        "batch_id": "b",
        "batch_dir": "/batch",
    }
    llm_stages = [
        stage for stage in config.STAGE_NAMES
        if (config.PROMPTS_DIR / f"{stage}.md").exists()
    ]
    assert llm_stages, "no stage prompts found"
    for stage in llm_stages:
        assert render_prompt(stage, variables).strip()
