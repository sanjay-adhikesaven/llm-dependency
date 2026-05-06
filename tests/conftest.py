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
    if name == "lineage" or name.startswith("lineage.")
]


def clear_lineage_modules() -> None:
    for name in list(sys.modules):
        if name == "lineage" or name.startswith("lineage."):
            del sys.modules[name]


@pytest.fixture
def fresh_runtime(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="lineage-test-"))
    monkeypatch.setenv("LINEAGE_STORAGE", str(tmp / "storage"))
    monkeypatch.setenv("LINEAGE_PATH", str(tmp / "storage" / "graph.db"))
    clear_lineage_modules()
    yield tmp
    clear_lineage_modules()
    shutil.rmtree(tmp, ignore_errors=True)

