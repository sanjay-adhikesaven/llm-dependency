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

GDB_STORAGE_ENV = "GDB_STORAGE"
GDB_PATH_ENV = "GDB_PATH"
GDB_RUN_ID_ENV = "GDB_RUN_ID"

STORAGE = Path(os.environ.get(GDB_STORAGE_ENV) or ROOT / "storage").resolve()
DB_PATH = Path(os.environ.get(GDB_PATH_ENV) or STORAGE / "graph.db").resolve()

SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", ".git"}
SQLITE_BUSY_TIMEOUT_S = 30.0
LINK_TIMEOUT_S = 10.0

CLAUDE_MODEL = os.environ.get("GDB_CLAUDE_MODEL", "opus")
PLANNER_CHOICES = ("opus", "sonnet")
SUBAGENT_CHOICES = ("opus", "sonnet")

PROCESS_KILL_GRACE_S = 5.0
LOG_ERROR_DETAIL_MAX_CHARS = 240
MAX_PARALLEL_BATCHES = int(os.environ.get("GDB_MAX_PARALLEL_BATCHES", "4"))
MAX_REVIEW_GROUPS_PER_WORKER = int(os.environ.get("GDB_MAX_REVIEW_GROUPS_PER_WORKER", "12"))
HF_API_BASE = os.environ.get("GDB_HF_API_BASE", "https://huggingface.co/api")
HF_BASE = os.environ.get("GDB_HF_BASE", "https://huggingface.co")

REFERENT_SCOPES = ("entity", "concept", "ambiguous")

ANCHOR_TYPES = (
    "hf_model",
    "hf_dataset",
    "hf_dataset_config",
    "github_repo",
    "github_ref",
    "api_model_id",
    "official_release_url",
    "paper_release",
)

URL_ANCHOR_TYPES = (
    "hf_model",
    "hf_dataset",
    "hf_dataset_config",
    "github_repo",
    "github_ref",
    "official_release_url",
    "paper_release",
)
