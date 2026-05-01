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

IDENTITY_FIELDS = (
    "family",
    "size",
    "stage",
    "version",
    "date",
    "subset",
    "quality_cut",
    "mix_variant",
    "modality",
    "domain",
    "context_length",
    "checkpoint",
)

DESCRIPTOR_FIELDS = (
    "organization",
    "namespace",
    "context_roles",
    "quantization",
    "precision",
    "format",
    "adapter",
    "language",
    "token_count",
    "sample_count",
    "notes",
)

CONTEXT_ROLES = (
    "training_data",
    "pretraining_data",
    "sft_data",
    "preference_data",
    "base_model",
    "teacher_model",
    "judge_model",
    "generator_model",
    "filter_or_classifier",
    "evaluation_benchmark",
    "comparison_baseline",
    "released_artifact",
    "unknown",
)

