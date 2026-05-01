from __future__ import annotations

import json
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_smollm2_fixture_flags_software_and_license_noise():
    from gdb.artifacts import validate_mention_artifact

    artifact = json.loads((FIXTURES / "smollm2_noise.json").read_text())
    errors = validate_mention_artifact(artifact)

    invalid = [error for error in errors if error["code"] == "invalid_kind"]
    assert len(invalid) == 2
    assert {error["value"] for error in invalid} == {"software", "license"}


def test_olmo3_fixture_flags_surface_conflict():
    from gdb.artifacts import detect_conflicts

    artifact = json.loads((FIXTURES / "olmo3_conflicts.json").read_text())
    violations = detect_conflicts(artifact["mentions"])

    assert any(violation["code"] == "surface_identity_conflict" for violation in violations)

