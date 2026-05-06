# LLM Dependency — baselines and evaluation harness

This repo contains the artifacts needed to reproduce the baseline runs and
evaluation reported in the paper, and to evaluate any new submission against
the same pool.

```
prompts/                       # investigator prompt (master + per-subject copies)
baselines/
  launch_baselines.py          # spawns 16 baseline runs (4 systems × 4 subjects)
  outputs/                     # the 16 baseline JSON graphs we produced
eval/
  pooled_eval.py               # the pooled LLM-as-judge verifier
  outputs/                     # verifications.jsonl, score.json, score_per_target.json
```

## Targets and baselines

Four target models (each with substantial public artifacts):

- **OLMo 3** (32B base) — `allenai/olmo-3-1125-32b`
- **Nemotron 3 Super** — `nvidia/nvidia-nemotron-3-super-120b-a12b-bf16`
- **DR-Tulu** — `rl-research/dr-tulu-8b`
- **SmolLM3** — `huggingfacetb/smollm3-3b`

Four baseline systems, each given the same per-subject investigator prompt:

| Slug | System | Configuration |
|---|---|---|
| `gpt55pro` | GPT-5.5-Pro | Responses API, web_search_preview tool, background mode |
| `gpt54pro` | GPT-5.4-Pro | Responses API, web_search_preview tool, background mode |
| `o3dr` | OpenAI Deep Research (`o3-deep-research`) | Responses API, web_search_preview tool, background mode |
| `cc` | Single-prompt Claude Code | `claude -p` headless, default tools (Opus 4.7 1M, default effort) |

Each combination produces one JSON file at `baselines/outputs/<slug>_<subject>.json`.

## Quick start

```bash
pip install -r requirements.txt

# 1. Reproduce the baseline runs (skips outputs that already exist)
cd baselines
OPENAI_API_KEY=sk-... python3 launch_baselines.py

# 2. Run the pooled eval
cd ../eval
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py
```

See `baselines/README.md` and `eval/README.md` for details.

## Evaluating a new submission

If your system follows the same per-subject convention (one
`{nodes, edges}` JSON file per subject), drop the files into
`baselines/outputs/` (or any other directory; pass via `--graphs-dir`)
named `<slug>_<subject>.json`, then re-run `pooled_eval.py` with your slug
appended:

```bash
cd eval
ANTHROPIC_API_KEY=sk-ant-... python3 pooled_eval.py \
    --systems gpt55pro,gpt54pro,cc,o3dr,mysystem
```

Verifications are appended incrementally to `eval/outputs/verifications.jsonl`,
so only the clusters your system contributes that aren't already covered get
fresh verifier calls. Existing verdicts persist; a single verifier verdict
attributes back to every system that proposed an edge in that cluster.

## Reported numbers

The verifications and aggregate scores in `eval/outputs/` are the exact
numbers used to produce the evaluation table in the paper. See
`eval/outputs/score.json` for totals and `eval/outputs/score_per_target.json`
for per-subject breakdown.

The verifier in all reported runs is `claude-sonnet-4-6` with the
`web_search_20250305` server tool (max 6 search uses per call).
