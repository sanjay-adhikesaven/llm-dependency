from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterable

from . import config
from .artifacts import (
    LINK_FIELDS,
    aggregate_mentions,
    anchors_from_links,
    choose_display_name,
    concept_display_name,
    concept_path_from_identity,
    links_from_anchors,
    merge_alias_lists,
    merge_descriptor_values,
    merge_links,
    normalize_anchor_candidates,
    normalize_mention,
    primary_anchor,
)
from .store import dumps, hash_text, normalize_space


def _signature(payload: Any) -> str:
    return hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def concept_node_key(kind: str, concept_path: list[str]) -> str:
    normalized = [normalize_space(part).casefold() for part in concept_path]
    return f"concept:{kind}:{_signature(normalized)}"


def entity_node_key(kind: str, anchor: dict) -> str:
    return f"entity:{kind}:{anchor['type']}:{_signature(anchor['value'])}"


def anchor_identity(anchor: dict) -> dict:
    return {"anchor_type": anchor["type"], "anchor": anchor["value"]}


def concept_identity(concept_path: list[str]) -> dict:
    return {"concept_path": list(concept_path)}


def verified_links_for_values(links: dict, link_checks: Iterable[dict]) -> dict:
    checks = {
        (check.get("link_kind"), check.get("link_value"))
        for check in link_checks
        if check.get("ok")
    }
    out = {field: [] for field in LINK_FIELDS}
    for field in LINK_FIELDS:
        for value in links.get(field) or []:
            if (field, value) in checks and value not in out[field]:
                out[field].append(value)
    return out


def verified_anchors_for_values(anchors: list[dict], link_checks: Iterable[dict]) -> list[dict]:
    checks = {
        (check.get("link_kind"), check.get("link_value"))
        for check in link_checks
        if check.get("ok")
    }
    out: list[dict] = []
    for anchor in anchors:
        anchor_type = anchor.get("type")
        value = anchor.get("value")
        if anchor_type == "api_model_id" or (anchor_type, value) in checks:
            out.append(anchor)
    return out


def _first_description(*values: str | None) -> str | None:
    for value in values:
        cleaned = normalize_space(value)
        if cleaned:
            return cleaned
    return None


def _ensure_concept(
    nodes: dict[str, dict],
    *,
    kind: str,
    concept_path: list[str],
) -> dict:
    key = concept_node_key(kind, concept_path)
    if key not in nodes:
        nodes[key] = {
            "node_key": key,
            "kind": kind,
            "node_type": "concept",
            "identity": concept_identity(concept_path),
            "concept_path": list(concept_path),
            "display_name": concept_display_name(concept_path),
            "aliases": [],
            "descriptors": {},
            "links": {field: [] for field in LINK_FIELDS},
            "anchors": [],
            "verified_links": {field: [] for field in LINK_FIELDS},
            "verified_anchors": [],
            "aux": {},
            "description": None,
            "occurrence_count": 0,
            "projection": True,
            "flags": [],
        }
    return nodes[key]


def _add_concept_path(nodes: dict[str, dict], edges: dict[tuple[str, str], dict], kind: str, concept_path: list[str]) -> str | None:
    if not concept_path:
        return None
    parent_key: str | None = None
    for depth in range(1, len(concept_path) + 1):
        node = _ensure_concept(nodes, kind=kind, concept_path=concept_path[:depth])
        if parent_key and parent_key != node["node_key"]:
            edges[(parent_key, node["node_key"])] = {
                "parent_node_key": parent_key,
                "child_node_key": node["node_key"],
                "rationale": "reviewed concept path prefix",
            }
        parent_key = node["node_key"]
    return parent_key


def _fallback_concept_path(cluster: dict) -> list[str]:
    if cluster.get("concept_path"):
        return list(cluster["concept_path"])
    identity_path = concept_path_from_identity(cluster.get("identity") or {})
    if identity_path:
        return identity_path
    aliases = cluster.get("aliases") or []
    if aliases:
        return [choose_display_name(aliases, cluster.get("identity") or {})]
    return []


def _exact_anchors(cluster: dict) -> list[dict]:
    anchors = [anchor for anchor in cluster.get("anchor_candidates") or [] if anchor.get("exact")]
    if anchors:
        return normalize_anchor_candidates(anchors, kind=cluster.get("kind"))
    link_anchors = anchors_from_links(cluster.get("links") or {}, kind=cluster.get("kind"))
    return normalize_anchor_candidates([anchor for anchor in link_anchors if anchor.get("exact")], kind=cluster.get("kind"))


def _primary_and_secondary_anchors(cluster: dict) -> tuple[dict | None, list[dict]]:
    anchors = _exact_anchors(cluster)
    primary = primary_anchor(anchors)
    if not primary:
        return None, []
    return primary, normalize_anchor_candidates(anchors, kind=cluster.get("kind"))


def _entity_display_name(cluster: dict, anchor: dict) -> str:
    aliases = cluster.get("aliases") or []
    if aliases:
        return choose_display_name(aliases, cluster.get("identity") or {})
    value = anchor.get("value") or ""
    if anchor.get("type") in {"hf_model", "hf_dataset", "github_repo", "github_ref"} and "/" in value:
        return value.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    if anchor.get("type") == "hf_dataset_config":
        return value.split("::", 1)[1]
    return value or "entity"


def build_lattice(mentions: Iterable[dict], link_checks: Iterable[dict] = ()) -> dict:
    clusters = aggregate_mentions(mentions)
    checks = list(link_checks)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    leaf_keys: set[str] = set()
    concept_has_entity: set[str] = set()

    for cluster in clusters:
        kind = cluster["kind"]
        concept_path = _fallback_concept_path(cluster)
        parent_key = _add_concept_path(nodes, edges, kind, concept_path)
        if parent_key:
            concept = nodes[parent_key]
            concept["occurrence_count"] += cluster.get("occurrence_count") or 0
            concept["aliases"] = merge_alias_lists(concept["aliases"], cluster.get("aliases") or [])
            concept["descriptors"] = merge_descriptor_values(concept["descriptors"], cluster.get("descriptors") or {})
            concept["aux"] = merge_descriptor_values(concept["aux"], cluster.get("aux") or {})
            concept["description"] = _first_description(concept.get("description"), cluster.get("description"))

        primary, exact_anchors = _primary_and_secondary_anchors(cluster)
        if not primary:
            if parent_key:
                nodes[parent_key]["flags"].append("observed_without_exact_anchor")
            continue

        for anchor in [primary]:
            node_key = entity_node_key(kind, anchor)
            if node_key not in nodes:
                links = merge_links({field: [] for field in LINK_FIELDS}, links_from_anchors([anchor]))
                nodes[node_key] = {
                    "node_key": node_key,
                    "kind": kind,
                    "node_type": "entity",
                    "identity": anchor_identity(anchor),
                    "concept_path": list(concept_path),
                    "display_name": _entity_display_name(cluster, anchor),
                    "aliases": [],
                    "descriptors": {},
                    "links": links,
                    "anchors": deepcopy(exact_anchors),
                    "verified_links": {field: [] for field in LINK_FIELDS},
                    "verified_anchors": [],
                    "aux": {},
                    "description": None,
                    "occurrence_count": 0,
                    "projection": False,
                    "flags": [],
                }
            entity = nodes[node_key]
            entity["aliases"] = merge_alias_lists(entity["aliases"], cluster.get("aliases") or [])
            entity["descriptors"] = merge_descriptor_values(entity["descriptors"], cluster.get("descriptors") or {})
            entity["links"] = merge_links(entity["links"], cluster.get("links") or {})
            entity["links"] = merge_links(entity["links"], links_from_anchors([anchor]))
            entity["anchors"] = normalize_anchor_candidates([*entity["anchors"], *exact_anchors], kind=kind)
            entity["verified_links"] = verified_links_for_values(entity["links"], checks)
            entity["verified_anchors"] = verified_anchors_for_values(entity["anchors"], checks)
            entity["aux"] = merge_descriptor_values(entity["aux"], cluster.get("aux") or {})
            entity["description"] = _first_description(entity.get("description"), cluster.get("description"))
            entity["occurrence_count"] += cluster.get("occurrence_count") or 0
            if anchor.get("type") in config.URL_ANCHOR_TYPES and not entity["verified_anchors"]:
                entity["flags"].append("unverified_exact_anchor")
            if not entity["anchors"]:
                entity["flags"].append("entity_without_exact_anchor")
            leaf_keys.add(node_key)
            if parent_key:
                edges[(parent_key, node_key)] = {
                    "parent_node_key": parent_key,
                    "child_node_key": node_key,
                    "rationale": "exact entity anchor under reviewed concept path",
                }
                concept_has_entity.add(parent_key)

    for node in nodes.values():
        if node["node_type"] == "concept" and node["node_key"] not in concept_has_entity:
            has_child_concept = any(edge["parent_node_key"] == node["node_key"] for edge in edges.values())
            if not has_child_concept and "observed_without_exact_anchor" in node["flags"]:
                node["flags"].append("concept_without_entity_leaf")
        node["flags"] = sorted(set(node["flags"]))

    return {
        "nodes": sorted(nodes.values(), key=lambda n: (n["kind"], n["node_type"], dumps(n["identity"]))),
        "edges": sorted(edges.values(), key=lambda e: (e["parent_node_key"], e["child_node_key"])),
        "clusters": clusters,
        "leaf_node_keys": sorted(leaf_keys),
    }
