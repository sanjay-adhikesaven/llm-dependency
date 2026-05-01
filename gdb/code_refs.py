from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import normalize_anchor_candidate
from .store import json_text, normalize_space


CALL_RE = re.compile(
    r"(?P<fn>from_pretrained|load_dataset)\(\s*['\"](?P<name>[^'\"]+)['\"]"
    r"(?:\s*,\s*['\"](?P<config>[^'\"]+)['\"])?",
    re.MULTILINE,
)
KEY_RE = re.compile(
    r"(?P<key>model_name_or_path|model_name|model_id|tokenizer_name|dataset_name|dataset|repo_id)\s*[:=]\s*['\"]?(?P<name>[A-Za-z0-9._/-]+)['\"]?",
    re.MULTILINE,
)
HF_RE = re.compile(r"(?<![A-Za-z0-9._-])(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)(?![A-Za-z0-9._-])")
PATH_RE = re.compile(r"(?P<name>[A-Za-z0-9._-]+(?:-[0-9]+plus|-[34]plus|_[0-9]+plus))")
MODEL_KEYS = {"model_name_or_path", "model_name", "model_id", "tokenizer_name", "repo_id", "base_model"}
DATASET_KEYS = {"dataset_name", "dataset", "datasets", "data_name"}
DOMAIN_PREFIXES = ("huggingface.co/", "github.com/", "arxiv.org/")


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _record(kind: str, name: str, *, file: str | None, line: int, source: str, config: str | None = None) -> dict[str, Any]:
    anchor_candidates = []
    if "/" in name:
        anchor_kind = "hf_dataset" if kind == "dataset" else "hf_model"
        anchor = normalize_anchor_candidate({"type": anchor_kind, "value": name}, kind=kind)
        if anchor:
            anchor_candidates.append(anchor)
    if config and kind == "dataset":
        anchor = normalize_anchor_candidate({"type": "hf_dataset_config", "repo": name, "config": config}, kind=kind)
        if anchor:
            anchor_candidates.append(anchor)
    atoms = [part for part in re.split(r"[-_/:\s]+", name) if part]
    return {
        "surface": name if not config else f"{name}::{config}",
        "kind": kind,
        "atoms": atoms,
        "anchor_candidates": anchor_candidates,
        "context_roles": ["unknown"],
        "referent_scope": "entity" if anchor_candidates else "ambiguous",
        "evidence": [{"file": file or "", "location": f"L{line}", "excerpt": source.strip()}],
    }


def _looks_like_reference(name: str) -> bool:
    cleaned = normalize_space(name)
    if not cleaned or cleaned.casefold().startswith(DOMAIN_PREFIXES):
        return False
    if "/" in cleaned:
        return True
    if re.search(r"\d", cleaned) and re.search(r"[-_.]", cleaned):
        return True
    return bool(re.search(r"(model|dataset|checkpoint|corpus|benchmark|pretrain|sft|dpo)", cleaned, re.IGNORECASE))


def _line_context(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].casefold()


def _kind_from_context(context: str) -> str:
    if any(term in context for term in ("load_dataset", "dataset", "datasets", "/datasets/")):
        return "dataset"
    return "model"


def extract_code_references(text: str, *, file: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None, int]] = set()

    def add(ref: dict[str, Any]) -> None:
        key = (ref["kind"], ref["surface"], None, int(str(ref["evidence"][0].get("location", "L0")).removeprefix("L") or 0))
        if key not in seen:
            out.append(ref)
            seen.add(key)

    for ref in extract_python_ast_references(text, file=file):
        add(ref)
    for ref in extract_structured_references(text, file=file):
        add(ref)

    for match in CALL_RE.finditer(text):
        fn = match.group("fn")
        name = normalize_space(match.group("name"))
        config = normalize_space(match.group("config") or "") or None
        kind = "dataset" if fn == "load_dataset" else "model"
        line = _line_for_offset(text, match.start())
        key = (kind, name, config, line)
        if key not in seen:
            add(_record(kind, name, file=file, line=line, source=match.group(0), config=config))
            seen.add(key)

    for match in KEY_RE.finditer(text):
        key_name = match.group("key")
        name = normalize_space(match.group("name"))
        if not _looks_like_reference(name):
            continue
        kind = "dataset" if "dataset" in key_name else "model"
        line = _line_for_offset(text, match.start())
        key = (kind, name, None, line)
        if key not in seen:
            add(_record(kind, name, file=file, line=line, source=match.group(0)))
            seen.add(key)

    for match in HF_RE.finditer(text):
        name = normalize_space(match.group("name"))
        if not _looks_like_reference(name):
            continue
        line = _line_for_offset(text, match.start())
        kind = _kind_from_context(_line_context(text, match.start(), match.end()))
        key = (kind, name, None, line)
        if key not in seen:
            add(_record(kind, name, file=file, line=line, source=name))
            seen.add(key)

    for match in PATH_RE.finditer(text):
        name = normalize_space(match.group("name"))
        if len(name) < 4:
            continue
        line = _line_for_offset(text, match.start())
        key = ("dataset", name, None, line)
        if key not in seen:
            add(_record("dataset", name, file=file, line=line, source=name))
            seen.add(key)
    return out


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return normalize_space(node.value)
    return None


def extract_python_ast_references(text: str, *, file: str | None = None) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = ""
            if isinstance(node.func, ast.Attribute):
                fn = node.func.attr
            elif isinstance(node.func, ast.Name):
                fn = node.func.id
            if fn in {"from_pretrained", "load_dataset"} and node.args:
                name = _literal_string(node.args[0])
                if not name:
                    continue
                config = _literal_string(node.args[1]) if len(node.args) > 1 else None
                kind = "dataset" if fn == "load_dataset" else "model"
                out.append(_record(kind, name, file=file, line=getattr(node, "lineno", 1), source=ast.get_source_segment(text, node) or fn, config=config))
        elif isinstance(node, ast.Assign):
            values = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.append(target.id)
                elif isinstance(target, ast.Attribute):
                    values.append(target.attr)
            name = _literal_string(node.value)
            if not name:
                continue
            for key_name in values:
                if key_name in MODEL_KEYS:
                    out.append(_record("model", name, file=file, line=getattr(node, "lineno", 1), source=ast.get_source_segment(text, node) or key_name))
                if key_name in DATASET_KEYS:
                    out.append(_record("dataset", name, file=file, line=getattr(node, "lineno", 1), source=ast.get_source_segment(text, node) or key_name))
    return out


def _walk_structured(value: Any, *, file: str | None, path: str = "", line: int = 1) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = normalize_space(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if isinstance(child, str):
                if key_text in MODEL_KEYS:
                    out.append(_record("model", normalize_space(child), file=file, line=line, source=f"{child_path}: {child}"))
                elif key_text in DATASET_KEYS:
                    out.append(_record("dataset", normalize_space(child), file=file, line=line, source=f"{child_path}: {child}"))
            else:
                out.extend(_walk_structured(child, file=file, path=child_path, line=line))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            out.extend(_walk_structured(child, file=file, path=f"{path}[{idx}]", line=line))
    return out


def extract_structured_references(text: str, *, file: str | None = None) -> list[dict[str, Any]]:
    stripped = text.lstrip()
    parsed = None
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            parsed = None
    if parsed is None:
        return []
    return _walk_structured(parsed, file=file)


def extract_file_references(path: Path) -> list[dict[str, Any]]:
    return extract_code_references(path.read_text(errors="replace"), file=str(path))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        text = sys.stdin.read()
        file = None
    else:
        path = Path(argv[0])
        text = path.read_text(errors="replace")
        file = str(path)
    print(json_text({"mentions": extract_code_references(text, file=file)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
