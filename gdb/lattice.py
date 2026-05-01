from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Iterable

from .artifacts import LINK_FIELDS, aggregate_mentions, choose_display_name, identity_key, merge_descriptor_values, merge_links
from .facets import Facets, cover_parents, flatten_identity
from .store import dumps


def projection_identities(identity: dict) -> list[dict]:
    flat = flatten_identity(identity)
    if not flat or "family" not in flat:
        return []
    keys = sorted(flat)
    required = {"family"}
    optional = [k for k in keys if k not in required]
    projections: list[dict] = []
    for size in range(0, len(optional) + 1):
        for combo in itertools.combinations(optional, size):
            selected = sorted(required | set(combo))
            projections.append(_unflatten_projection({k: flat[k] for k in selected}))
    return projections


def _unflatten_projection(flat: dict[str, str]) -> dict:
    out: dict[str, Any] = {}
    extra: dict[str, str] = {}
    for key, value in flat.items():
        if key.startswith("extra."):
            extra[key.removeprefix("extra.")] = value
        else:
            out[key] = value
    if extra:
        out["extra"] = dict(sorted(extra.items()))
    return out


def display_from_identity(identity: dict) -> str:
    parts = [
        identity.get("family"),
        identity.get("size"),
        identity.get("stage"),
        identity.get("version"),
        identity.get("date"),
        identity.get("subset"),
        identity.get("quality_cut"),
        identity.get("mix_variant"),
    ]
    label = "-".join(str(p) for p in parts if p)
    if label:
        return label
    if identity.get("extra"):
        extra = ",".join(f"{k}={v}" for k, v in sorted(identity["extra"].items()))
        return f"{identity.get('family', 'projection')}[{extra}]"
    return "projection"


def verified_links_for_cluster(cluster_key: str, link_checks: Iterable[dict]) -> dict:
    out = {field: [] for field in LINK_FIELDS}
    for check in link_checks:
        if check.get("cluster_key") != cluster_key or not check.get("ok"):
            continue
        link_kind = check.get("link_kind")
        link_value = check.get("link_value")
        if link_kind in out and link_value and link_value not in out[link_kind]:
            out[link_kind].append(link_value)
    return out


def has_occurrence_marker(cluster: dict) -> bool:
    for evidence in cluster.get("evidence") or []:
        if evidence.get("source_id") or (evidence.get("file") and evidence.get("excerpt")):
            return True
    return False


def build_lattice(mentions: Iterable[dict], link_checks: Iterable[dict] = ()) -> dict:
    clusters = aggregate_mentions(mentions)
    checks = list(link_checks)
    nodes: dict[str, dict] = {}
    leaf_keys: set[str] = set()

    for cluster in clusters:
        for identity in projection_identities(cluster["identity"]):
            node_key = identity_key(cluster["kind"], identity)
            projection = identity != cluster["identity"]
            if node_key not in nodes:
                nodes[node_key] = {
                    "node_key": node_key,
                    "kind": cluster["kind"],
                    "identity": deepcopy(identity),
                    "display_name": display_from_identity(identity),
                    "aliases": [],
                    "descriptors": {},
                    "links": {field: [] for field in LINK_FIELDS},
                    "verified_links": {field: [] for field in LINK_FIELDS},
                    "occurrence_count": 0,
                    "projection": projection,
                    "flags": [],
                }
        leaf_key = identity_key(cluster["kind"], cluster["identity"])
        leaf_keys.add(leaf_key)
        leaf = nodes[leaf_key]
        leaf["projection"] = False
        leaf["display_name"] = choose_display_name(cluster["aliases"], cluster["identity"])
        leaf["aliases"] = cluster["aliases"]
        leaf["descriptors"] = merge_descriptor_values(leaf["descriptors"], cluster["descriptors"])
        leaf["links"] = merge_links(leaf["links"], cluster["links"])
        leaf["verified_links"] = verified_links_for_cluster(cluster["cluster_key"], checks)
        leaf["occurrence_count"] += cluster["occurrence_count"]
        if not any(leaf["verified_links"].values()) and not has_occurrence_marker(cluster):
            leaf["flags"].append("leaf_without_verified_link_or_occurrence")

    facets = {
        node_key: Facets.from_identity(node["kind"], node["identity"])
        for node_key, node in nodes.items()
    }
    edges: list[dict] = []
    candidates = list(facets.items())
    for child_key, child_facets in candidates:
        parents = cover_parents(child_facets, [facets_value for _key, facets_value in candidates])
        for parent in parents:
            parent_key = next(key for key, value in candidates if value == parent)
            edges.append({
                "parent_node_key": parent_key,
                "child_node_key": child_key,
                "rationale": "identity facet cover edge",
            })

    return {
        "nodes": sorted(nodes.values(), key=lambda n: (n["kind"], dumps(n["identity"]))),
        "edges": sorted(edges, key=lambda e: (e["parent_node_key"], e["child_node_key"])),
        "clusters": clusters,
        "leaf_node_keys": sorted(leaf_keys),
    }

