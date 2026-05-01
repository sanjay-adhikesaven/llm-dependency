from __future__ import annotations

import os
import sysconfig
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def _resolve_runtime_path(name: str) -> Path:
    primary = ROOT / name
    if primary.exists():
        return primary
    data = Path(sysconfig.get_path("data")) / "share" / "gdb-lattice" / name
    return data


SCHEMA_PATH = _resolve_runtime_path("schema.sql")
PROMPTS_DIR = ROOT / "gdb" / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = Path(sysconfig.get_path("data")) / "share" / "gdb-lattice" / "prompts"

load_dotenv(ROOT / ".env")

# Environment variable names
GDB_STORAGE_ENV = "GDB_STORAGE"
GDB_PATH_ENV = "GDB_PATH"
GDB_RUN_ID_ENV = "GDB_RUN_ID"

STORAGE = Path(os.environ.get(GDB_STORAGE_ENV) or ROOT / "storage").resolve()
DB_PATH = Path(os.environ.get(GDB_PATH_ENV) or STORAGE / "graph.db").resolve()

# Storage layout — directory names under STORAGE and run_root
RUNS_SUBDIR = "runs"
SOURCES_SUBDIR = "sources"
WORKSPACE_SUBDIR = "workspace"
WORKERS_SUBDIR = "workers"
BATCH_SUBDIR = "batch"

# Per-run files (under STORAGE/runs/<run_id>/)
RUN_PROMPT_FILE = "prompt.md"
RUN_STDOUT_FILE = "stdout.txt"      # codex (plain text)
RUN_STREAM_FILE = "stream.jsonl"    # claude (--output-format stream-json)
RUN_STDERR_FILE = "stderr.txt"
RUN_INPUT_FILE = "input.json"
BATCH_MANIFEST_FILE = "MANIFEST.txt"

# Pipeline stages (execution order)
STAGE_NAMES = (
    "discover",
    "extract",
    "check",
    "audit",
    "verify-links",
    "build-lattice",
    "describe",
)

# Per-stage artifact filenames written under each run_root
DISCOVER_ARTIFACT_FILE = "discover_artifact.json"
EXTRACT_ARTIFACT_FILE = "extract_artifact.json"
AUDIT_ARTIFACT_FILE = "audit_artifact.json"
DESCRIBE_ARTIFACT_FILE = "describe_artifact.json"
CLUSTER_PACKET_FILE = "cluster_packet.json"
LATTICE_FILE = "lattice.json"

# Directory walk filter (skipped during scan, fingerprint, copytree)
SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", ".git"}

# Timeouts and limits
SQLITE_BUSY_TIMEOUT_S = 30.0
LINK_TIMEOUT_S = 10.0
PROCESS_KILL_GRACE_S = 5.0
MAX_PARALLEL_BATCHES = int(os.environ.get("GDB_MAX_PARALLEL_BATCHES", "32"))
RUN_LOG_TAIL_CHARS = 20000   # tail of stdout/stderr scanned for usage stats
HASH_CHUNK_BYTES = 1 << 20   # streaming chunk size for sha256_file

# Domain enums (paired with schema CHECK constraints)
VALID_KINDS = ("model", "dataset")
REFERENT_SCOPES = ("entity", "concept", "ambiguous")

# Link policy
LINK_TYPES = (
    "hf_model",
    "hf_dataset",
    "hf_dataset_config",
    "github_repo",
    "github_ref",
    "api_model_id",
    "official_release_url",
    "paper_release",
)
URL_LINK_TYPES = (
    "hf_model",
    "hf_dataset",
    "hf_dataset_config",
    "github_repo",
    "github_ref",
    "official_release_url",
    "paper_release",
)
# Primary-link selection order. A slot may be a single type or a tuple of
# types tied at the same priority.
PRIMARY_LINK_ORDER = (
    "hf_dataset_config",
    ("hf_model", "hf_dataset"),
    "github_ref",
    "github_repo",
    "api_model_id",
    "official_release_url",
    "paper_release",
)

# Alias surface filter (is_invalid_alias_surface)
MAX_ALIAS_SURFACE_LEN = 180
MAX_ALIAS_SURFACE_WORDS = 4

# Violation reporting
VIOLATION_EXAMPLE_CAP = 3

# External services
HF_API_BASE = os.environ.get("GDB_HF_API_BASE", "https://huggingface.co/api")
HF_BASE = os.environ.get("GDB_HF_BASE", "https://huggingface.co")

# Models
CLAUDE_MODEL = os.environ.get("GDB_CLAUDE_MODEL", "opus")
CODEX_MODEL = os.environ.get("GDB_CODEX_MODEL", "gpt-5.5")
CODEX_EFFORT_CHOICES = ("low", "medium", "high", "xhigh")

# CLI-restricted model choices. The planner is always Claude (only
# Claude can drive the Task tool / artifact-writing flow); the
# subagent may be Claude or Codex.
PLANNER_CHOICES = ("opus", "sonnet")
SUBAGENT_CHOICES = (
    "opus", "sonnet",
    "codex-low", "codex-medium", "codex-high", "codex-xhigh",
)

# `{{subagent_prompt}}` templates rendered by
# `pipeline.subagent_prompt_for(model)`. The planner reads one of
# these to learn how to dispatch sub-work this run. Claude Code knows
# how to launch subagents natively, so the Claude template only needs
# the model. Codex has no native multi-agent primitive, so its
# template gives the dispatch command directly.
#
# These match prov/config.py verbatim — same wording, same anti-pattern
# block. The polling-across-bash-calls anti-pattern has bitten us
# before (see trace/'s SESSION_LOG.md hang post-mortem), and the
# explicit enumeration here is the actual fix.

SUBAGENT_PROMPT_CLAUDE = (
    "## Subagent dispatch (Task tool)\n"
    "\n"
    "The Task tool is available — subagents run as `{model}`. "
    "Use them aggressively when the work has parallel structure: a "
    "directory of sources, a list of clusters, an enumerable set of "
    "entity leaves, anything where one unit can be analyzed without "
    "reading the others. Each Task call's reading + reasoning happens "
    "in the subagent's own context, not yours, so dispatching keeps "
    "your main context free for synthesis.\n"
    "\n"
    "Why this matters: if you instead read every source / cluster / "
    "entity yourself in your main loop, the accumulating tool results "
    "fill your context window. On long stages (dozens of sources, "
    "hundreds of clusters, many entity leaves) that pressure degrades "
    "your reasoning before you reach final synthesis. Dispatching is "
    "how you stay sharp at the aggregation step.\n"
    "\n"
    "**You decide whether to dispatch — it's not mandatory.** Run "
    "inline when the work is small (a single file, one cluster, one "
    "trivial fetch) where dispatch overhead exceeds the context "
    "savings. Dispatch when there's real fan-out.\n"
    "\n"
    "Right-size each subagent: a topically coherent unit a careful "
    "reader would treat as one pass. Too narrow (one-per-file, "
    "one-per-record) duplicates context-loading. Too wide (the whole "
    "batch in one call) just shifts the same context pressure into "
    "the subagent.\n"
    "\n"
    "When you dispatch, brief the subagent like a stranger — it has "
    "none of your context. Transcribe the relevant rules from this "
    "prompt verbatim; rule erosion at dispatch is the main cause of "
    "subagent output drifting from the rules you were given."
)

SUBAGENT_PROMPT_CODEX = (
    "You are running as the **orchestrator**. Plan and synthesize "
    "the final artifact yourself, but delegate every unit of "
    "reading-and-analysis to a codex subagent. **Do NOT call the "
    "`Agent` or `Task` tool** — they are off-policy for this run. "
    "Subagents are dispatched ONLY via the codex CLI.\n"
    "\n"
    "Codex was selected by the operator because gpt-5.5's "
    "reasoning depth is needed for per-unit work. Skipping codex "
    "produces thinner output and defeats the operator's choice. "
    "When in doubt, dispatch.\n"
    "\n"
    "Each codex subagent runs in a **fresh process with no shared "
    "context** — it cannot see your prompt or your running state. "
    "Brief it like a stranger: include filesystem scope, schema, "
    "rules, and the completion contract in its prompt. How you "
    "structure I/O (argv prompt, stdin pipe, scratch files) and "
    "where you put any temp artifacts is your choice; pick what "
    "fits the work-unit.\n"
    "\n"
    "## CLI invocation\n"
    "\n"
    "```bash\n"
    "codex exec -m {codex_model} \\\n"
    "  -c model_reasoning_effort={effort} \\\n"
    "  --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \\\n"
    "  \"<self-contained prompt: scope + schema + rules + completion>\"\n"
    "```\n"
    "\n"
    "## Waiting on concurrent codex calls\n"
    "\n"
    "Dispatch and wait for codex subagents in **one** Bash "
    "invocation, not across multiple. The shape:\n"
    "\n"
    "```bash\n"
    "codex exec ... \"$P1\" > out-1.log 2>&1 &\n"
    "codex exec ... \"$P2\" > out-2.log 2>&1 &\n"
    "wait                  # blocks until ALL background jobs exit\n"
    "ls out-*.log          # then check outputs\n"
    "```\n"
    "\n"
    "Pass `timeout: 600000` (10 minutes) on this Bash call. The "
    "Bash tool's default 2-minute timeout will kill a long `wait` "
    "and leave codex grandchildren orphaned in the process tree. "
    "Codex runs at high reasoning effort routinely exceed the "
    "default Bash timeout.\n"
    "\n"
    "Anti-patterns (will silently drop work):\n"
    "- `&` without `wait` — the Bash call returns immediately and "
    "  your next turn sees codex outputs as not-yet-written.\n"
    "- Polling across separate Bash calls (`until [ -z $(ps ...) ]; "
    "  do sleep 30; done` in its own invocation) — hits the 2-min "
    "  Bash timeout, leaves codex running, and there is no graceful "
    "  recovery path.\n"
    "- `ScheduleWakeup` — does not fire in non-interactive `claude "
    "  -p` runs. The harness blocks the tool entirely; if you call "
    "  it your turn ends and the artifact is never written.\n"
    "\n"
    "## Sizing each subagent's work\n"
    "\n"
    "Each codex call sees a unit of work you choose. Two failure "
    "modes:\n"
    "- **Too narrow** (one-per-file, one-per-page, one-per-record): "
    "  duplicates context-loading, fragments cross-reference signal, "
    "  and turns the planner into the worker.\n"
    "- **Too wide** (the entire batch in one call): dilutes attention, "
    "  produces shallower reasoning, costs more per call than the "
    "  reasoning depth justifies.\n"
    "\n"
    "Aim for the unit that a single careful reader would naturally "
    "treat as one pass: a topically coherent section group, a "
    "directory of related recipes, a set of clusters that share a "
    "decision pattern. Split by topical coherence, not by uniform "
    "size or one-per-input-record.\n"
    "\n"
    "## When to run inline (no codex)\n"
    "\n"
    "Trivial one-shots: `curl`, `git rev-parse`, file existence "
    "checks. Final assembly of subagent outputs (your job). "
    "Anything else that involves reading-and-analysis goes through "
    "codex."
)
