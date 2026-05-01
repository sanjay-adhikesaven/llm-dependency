from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .artifacts import normalize_mention, normalize_surface, primary_anchor


def root_from_atoms(atoms: list[str], surface: str) -> str:
    if atoms:
        return atoms[0]
    if "/" in surface:
        return surface.rsplit("/", 1)[-1].split("-", 1)[0]
    return surface.split("-", 1)[0].split(" ", 1)[0]


def prefix_key(atoms: list[str], depth: int = 2) -> str:
    return " / ".join(atom.casefold() for atom in atoms[:depth] if atom)


def namespace_key(anchor: dict | None) -> str | None:
    if not anchor:
        return None
    value = anchor.get("value") or ""
    if "/" in value:
        return value.split("/", 1)[0].casefold()
    if anchor.get("type") == "api_model_id" and "/" in value:
        return value.split("/", 1)[0].casefold()
    return None


def group_mentions_for_review(mentions: Iterable[dict], *, max_group_size: int = 40) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for raw in mentions:
        mention = normalize_mention(raw)
        anchor = primary_anchor(mention.get("anchor_candidates") or [])
        ns = namespace_key(anchor)
        root = mention["concept_path"][0] if mention.get("concept_path") else root_from_atoms(mention.get("atoms") or [], mention.get("surface") or "")
        prefix = prefix_key(mention.get("atoms") or [], 2)
        key = ns or normalize_surface(root) or prefix or mention["surface_key"][:24]
        buckets[key].append(mention)

    groups: list[dict] = []
    for group_key, items in sorted(buckets.items()):
        sorted_items = sorted(items, key=lambda m: (m.get("surface_key") or "", m.get("id") or ""))
        for idx in range(0, len(sorted_items), max_group_size):
            chunk = sorted_items[idx:idx + max_group_size]
            roots = sorted({root_from_atoms(m.get("atoms") or [], m.get("surface") or "") for m in chunk if m.get("surface")})
            prefixes = sorted({prefix_key(m.get("atoms") or [], 3) for m in chunk if m.get("atoms")})
            groups.append({
                "group_key": group_key if idx == 0 else f"{group_key}:{idx // max_group_size + 1}",
                "root_candidates": roots[:20],
                "prefix_candidates": [p for p in prefixes if p][:30],
                "mentions": chunk,
            })
    return groups


def policy_from_review_updates(updates: list[dict]) -> list[dict]:
    policies: dict[tuple[str, str], dict] = {}
    for update in updates:
        if not isinstance(update, dict):
            continue
        kind = update.get("kind") or "model"
        path = update.get("concept_path") or update.get("lattice_path") or []
        if not isinstance(path, list) or not path:
            continue
        root = str(path[0])
        policy = policies.setdefault((kind, root.casefold()), {
            "kind": kind,
            "root": root,
            "policy": {"known_paths": []},
            "evidence": [],
        })
        if path not in policy["policy"]["known_paths"]:
            policy["policy"]["known_paths"].append(path)
        evidence = update.get("evidence") or []
        if evidence:
            policy["evidence"].extend(evidence if isinstance(evidence, list) else [evidence])
    return list(policies.values())

