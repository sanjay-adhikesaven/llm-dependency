from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

GRAPH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GRAPH_ROOT))


RUNTIME_MODULES = [
    name for name in list(sys.modules)
    if name == "gdb" or name.startswith("gdb.")
]


def clear_gdb_modules() -> None:
    for name in list(sys.modules):
        if name == "gdb" or name.startswith("gdb."):
            del sys.modules[name]


@pytest.fixture
def fresh_runtime(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="gdb-test-"))
    monkeypatch.setenv("GDB_STORAGE", str(tmp / "storage"))
    monkeypatch.setenv("GDB_PATH", str(tmp / "storage" / "graph.db"))
    clear_gdb_modules()
    yield tmp
    clear_gdb_modules()
    shutil.rmtree(tmp, ignore_errors=True)

