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
    if name == "modsleuth" or name.startswith("modsleuth.")
]


def clear_modsleuth_modules() -> None:
    for name in list(sys.modules):
        if name == "modsleuth" or name.startswith("modsleuth."):
            del sys.modules[name]


@pytest.fixture
def fresh_runtime(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="modsleuth-test-"))
    monkeypatch.setenv("MODSLEUTH_STORAGE", str(tmp / "storage"))
    monkeypatch.setenv("MODSLEUTH_PATH", str(tmp / "storage" / "graph.db"))
    clear_modsleuth_modules()
    yield tmp
    clear_modsleuth_modules()
    shutil.rmtree(tmp, ignore_errors=True)
