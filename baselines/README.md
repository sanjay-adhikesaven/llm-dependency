# Baselines

`launch_baselines.py` fires all 16 runs (4 systems × 4 subjects) in parallel:

- 12 OpenAI runs go through the Responses API in **background mode** with the
  `web_search_preview` tool, polled every 15 s up to a 90 min cap.
- 4 Claude Code runs spawn a `claude -p --dangerously-skip-permissions`
  subprocess each, in a fresh `tempfile.TemporaryDirectory()` so any local
  `CLAUDE.md` / project state doesn't leak in.

Each (system, subject) combination produces one file at
`outputs/<slug>_<subject>.json`. Re-running skips outputs that already
exist, so a partial failure is recoverable by deleting just the failing
`<slug>_<subject>.json` files and re-launching.

## Run

```bash
pip install -r ../requirements.txt
OPENAI_API_KEY=sk-... python3 launch_baselines.py 2>&1 | tee /tmp/baselines.log
```

Wall time: roughly 30–90 min total. The slowest path is `o3-deep-research`,
which can take 15–30 min per subject. Watch `/tmp/baselines.log` to track
progress, or `ls -la outputs/` to see files appearing as they complete.

## Notes

- **Claude Code** uses whatever model and effort level your `~/.claude/settings.json`
  defaults to. The reported numbers use Opus 4.7 (1M context) at default effort.
  To force a specific model, set `"model"` in your settings file before launching.
- **OpenAI model IDs** — `gpt-5.5-pro` and `gpt-5.4-pro` are the API-side
  identifiers exposed to our account. If your account exposes them under
  different IDs, edit `OPENAI_SYSTEMS` in `launch_baselines.py`.
- A few Claude Code runs may preface the JSON with one line of model preamble
  (e.g., "I have all the information I need. Now I'll compose..."). Strip
  before parsing if your downstream tool can't tolerate it. The pooled
  evaluator handles either form.
