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
RUN_STDOUT_FILE = "stdout.txt"
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
MAX_PARALLEL_BATCHES = int(os.environ.get("GDB_MAX_PARALLEL_BATCHES", "4"))
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

# Subagent dispatch instructions injected into stage prompts as
# `{{subagent_prompt}}`. Pipeline.subagent_prompt_for(model) renders
# whichever block matches the chosen subagent runtime.
SUBAGENT_PROMPT_CLAUDE = (
    "Launch subagents in parallel for independent units of work via "
    "the Task tool. Subagents run as `{model}`."
)

SUBAGENT_PROMPT_CODEX = (
    "## Subagent dispatch (codex)\n"
    "\n"
    "You are running as the **orchestrator**. Plan and synthesize the "
    "final artifact yourself, but delegate every unit of "
    "reading-and-analysis to a codex subagent. **Do NOT call the "
    "`Agent` or `Task` tool** — they are off-policy for this run. "
    "Subagents are dispatched ONLY via the codex CLI.\n"
    "\n"
    "Each codex subagent runs in a **fresh process with no shared "
    "context** — it cannot see your prompt or your running state. "
    "Brief it like a stranger: include filesystem scope, schema, "
    "rules, and the completion contract in its prompt.\n"
    "\n"
    "### CLI invocation\n"
    "\n"
    "```bash\n"
    "codex exec -m {codex_model} \\\n"
    "  -c model_reasoning_effort={effort} \\\n"
    "  --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \\\n"
    "  \"<self-contained prompt: scope + schema + rules + completion>\"\n"
    "```\n"
    "\n"
    "### Waiting on concurrent codex calls\n"
    "\n"
    "Dispatch and wait for codex subagents in **one** Bash invocation, "
    "not across multiple. The shape:\n"
    "\n"
    "```bash\n"
    "codex exec ... \"$P1\" > out-1.log 2>&1 &\n"
    "codex exec ... \"$P2\" > out-2.log 2>&1 &\n"
    "wait                  # blocks until ALL background jobs exit\n"
    "ls out-*.log          # then check outputs\n"
    "```\n"
    "\n"
    "Codex at `{effort}` effort can take 30–60+ minutes on dense "
    "long-context tasks. Pass `timeout: 7200000` (120 minutes) on "
    "this Bash call so the wait doesn't get cut short."
)
