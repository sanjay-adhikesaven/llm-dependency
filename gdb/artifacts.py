from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable
from urllib.parse import urlsplit

from . import config
from .store import dumps, hash_text, key, normalize_space


VALID_KINDS = ("model", "dataset")
LINK_FIELDS = ("hf_ids", "github_repos", "official_urls", "papers")
GENERIC_ALIASES = {
    "default",
    "main",
    "latest",
    "this model",
    "this dataset",
    "the model",
    "the dataset",
    "our model",
    "our dataset",
    "base model",
    "dataset",
    "model",
}

_HF_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?:datasets/)?"
    r"(?P<repo>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)/?$"
)
_GITHUB_REPO_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/"
    r"(?P<repo>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)/?$"
)
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,6}(?:v\d+)?$")
_ARXIV_URL_RE = re.compile(r"^https?://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,6}(?:v\d+)?)/?$")
_HTTP_URL_RE = re.compile(r"^https?://[^\s]+$")


def normalize_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).casefold()


def is_generic_alias(surface: Any) -> bool:
    text = normalize_space(surface)
    lower = text.casefold()
    if not text:
        return True
    if lower in GENERIC_ALIASES:
        return True
    if lower.startswith(("http://", "https://", "huggingface.co/", "github.com/")):
        return True
    if len(text) > 180:
        return True
    if "/" in text and " " in text and len(text.split()) > 4:
        return True
    return False


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = normalize_space(value)
        return cleaned or None
    if isinstance(value, (int, float, bool)):
        return value
    return value


def _canonical_extra(extra: Any) -> dict:
    if not isinstance(extra, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in extra.items():
        k = normalize_space(raw_key)
        if not k:
            continue
        v = _clean_scalar(raw_value)
        if v in (None, "", [], {}):
            continue
        out[k] = v
    return {k: out[k] for k in sorted(out)}


def canonical_identity(identity: Any) -> dict:
    if not isinstance(identity, dict):
        identity = {}
    out: dict[str, Any] = {}
    for field in config.IDENTITY_FIELDS:
        value = _clean_scalar(identity.get(field))
        if value in (None, "", [], {}):
            continue
        out[field] = value
    extra = _canonical_extra(identity.get("extra"))
    if extra:
        out["extra"] = extra
    return out


def identity_signature(kind: str, identity: dict) -> str:
    canonical = canonical_identity(identity)

    def normalize_for_key(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_space(value).casefold()
        if isinstance(value, dict):
            return {k.casefold(): normalize_for_key(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [normalize_for_key(v) for v in value]
        return value

    payload = [kind, normalize_for_key(canonical)]
    return hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def identity_key(kind: str, identity: dict) -> str:
    return f"{kind}:{identity_signature(kind, identity)}"


def normalize_descriptors(descriptors: Any) -> dict:
    if not isinstance(descriptors, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in descriptors.items():
        k = normalize_space(raw_key)
        if not k:
            continue
        value = _clean_scalar(raw_value)
        if value in (None, "", [], {}):
            continue
        if k == "context_roles":
            roles = normalize_roles(value)
            if roles:
                out[k] = roles
        else:
            out[k] = value
    return {k: out[k] for k in sorted(out)}


def normalize_roles(value: Any) -> list[str]:
    raw = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        role = normalize_space(item).casefold()
        if not role:
            continue
        if role not in config.CONTEXT_ROLES:
            role = "unknown"
        if role not in seen:
            out.append(role)
            seen.add(role)
    return out or ["unknown"]


def normalize_aliases(aliases: Any, surface: str | None = None, descriptors: dict | None = None) -> list[dict]:
    raw_items: list[Any] = []
    if surface:
        raw_items.append({"surface": surface, "descriptors": descriptors or {}})
    if isinstance(aliases, list):
        raw_items.extend(aliases)
    elif aliases:
        raw_items.append(aliases)
    out: list[dict] = []
    by_key: dict[str, dict] = {}
    for item in raw_items:
        if isinstance(item, str):
            alias_surface = normalize_space(item)
            alias_descriptors = {}
        elif isinstance(item, dict):
            alias_surface = normalize_space(item.get("surface") or item.get("name") or "")
            alias_descriptors = normalize_descriptors(item.get("descriptors") or {})
        else:
            continue
        if not alias_surface:
            continue
        alias_key = normalize_surface(alias_surface)
        if alias_key in by_key:
            by_key[alias_key]["descriptors"] = merge_descriptor_values(
                by_key[alias_key].get("descriptors") or {},
                alias_descriptors,
            )
            continue
        rec = {"surface": alias_surface, "descriptors": alias_descriptors}
        out.append(rec)
        by_key[alias_key] = rec
    return out


def _dedup(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def normalize_hf_id(value: Any) -> str | None:
    text = normalize_space(value)
    if not text:
        return None
    match = _HF_URL_RE.match(text)
    if match:
        text = match.group("repo")
    if _HF_ID_RE.match(text):
        return text
    return None


def normalize_github_repo(value: Any) -> str | None:
    text = normalize_space(value)
    if not text:
        return None
    match = _GITHUB_URL_RE.match(text)
    if match:
        text = match.group("repo")
    if _GITHUB_REPO_RE.match(text):
        return text
    return None


def normalize_http_url(value: Any) -> str | None:
    text = normalize_space(value)
    if not _HTTP_URL_RE.match(text):
        return None
    parts = urlsplit(text)
    host = parts.netloc.lower()
    scheme = parts.scheme.lower()
    path = parts.path.rstrip("/")
    if scheme == "http" and host in {"huggingface.co", "github.com", "arxiv.org"}:
        scheme = "https"
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{host}{path}{query}"


def normalize_paper(value: Any) -> str | None:
    text = normalize_space(value)
    if not text:
        return None
    match = _ARXIV_URL_RE.match(text)
    if match:
        return f"https://arxiv.org/abs/{match.group('id')}"
    if text.casefold().startswith("arxiv:"):
        text = normalize_space(text.split(":", 1)[1])
    if _ARXIV_ID_RE.match(text):
        return f"https://arxiv.org/abs/{text}"
    return normalize_http_url(text)


def normalize_links(links: Any) -> dict:
    if not isinstance(links, dict):
        links = {}
    hf_values = links.get("hf_ids") or links.get("huggingface_ids") or []
    github_values = links.get("github_repos") or links.get("github") or []
    official_values = links.get("official_urls") or links.get("urls") or []
    paper_values = links.get("papers") or links.get("paper_urls") or []

    if isinstance(hf_values, str):
        hf_values = [hf_values]
    if isinstance(github_values, str):
        github_values = [github_values]
    if isinstance(official_values, str):
        official_values = [official_values]
    if isinstance(paper_values, str):
        paper_values = [paper_values]

    return {
        "hf_ids": _dedup(x for x in (normalize_hf_id(v) for v in hf_values) if x),
        "github_repos": _dedup(x for x in (normalize_github_repo(v) for v in github_values) if x),
        "official_urls": _dedup(x for x in (normalize_http_url(v) for v in official_values) if x),
        "papers": _dedup(x for x in (normalize_paper(v) for v in paper_values) if x),
    }


def normalize_evidence(evidence: Any) -> list[dict]:
    raw = evidence if isinstance(evidence, list) else ([evidence] if evidence else [])
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source_id = normalize_space(item.get("source_id") or "")
        file = normalize_space(item.get("file") or "")
        location = normalize_space(item.get("location") or "")
        excerpt = normalize_space(item.get("excerpt") or "")
        url = normalize_http_url(item.get("url")) if item.get("url") else None
        rec = {
            "source_id": source_id,
            "file": file,
            "location": location,
            "excerpt": excerpt,
        }
        if url:
            rec["url"] = url
        out.append({k: v for k, v in rec.items() if v})
    return out


def normalize_subsets(subsets: Any) -> list[dict]:
    raw = subsets if isinstance(subsets, list) else ([subsets] if subsets else [])
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = normalize_space(item.get("name") or item.get("subset") or "")
        identity = canonical_identity(item.get("identity") or {"subset": name})
        evidence = normalize_evidence(item.get("evidence") or [])
        if name or identity:
            out.append({"name": name, "identity": identity, "evidence": evidence})
    return out


def normalize_mention(item: dict, *, batch_id: str | None = None, source_id: str | None = None) -> dict:
    surface = normalize_space(item.get("surface") or item.get("raw_text") or "")
    kind = normalize_space(item.get("kind") or item.get("parsed_kind") or "").casefold()
    identity = canonical_identity(item.get("identity") or item.get("parsed_identity") or item.get("parsed_facets") or {})
    descriptors = normalize_descriptors(item.get("descriptors") or {})
    aliases = normalize_aliases(item.get("aliases") or [], surface=surface, descriptors=descriptors)
    links = normalize_links(item.get("links") or {})
    roles = normalize_roles(item.get("context_roles") or descriptors.get("context_roles") or ["unknown"])
    evidence = normalize_evidence(item.get("evidence") or {
        "source_id": item.get("source_id") or source_id or "",
        "file": item.get("file") or "",
        "location": item.get("location") or "",
        "excerpt": item.get("excerpt") or "",
    })
    subsets = normalize_subsets(item.get("subsets") or [])
    return {
        "id": item.get("id"),
        "batch_id": item.get("batch_id") or batch_id,
        "source_id": item.get("source_id") or source_id,
        "kind": kind,
        "surface": surface,
        "surface_key": normalize_surface(surface),
        "identity": identity,
        "identity_key": identity_key(kind, identity) if kind in VALID_KINDS else "",
        "descriptors": descriptors,
        "aliases": aliases,
        "links": links,
        "subsets": subsets,
        "context_roles": roles,
        "evidence": evidence,
        "notes": normalize_space(item.get("notes") or item.get("rationale") or "") or None,
        "attrs": item.get("attrs") if isinstance(item.get("attrs"), dict) else {},
    }


def artifact_mentions(artifact: Any) -> list[dict]:
    if not isinstance(artifact, dict):
        return []
    mentions = artifact.get("mentions")
    return mentions if isinstance(mentions, list) else []


def validate_mention_artifact(artifact: Any) -> list[dict]:
    errors: list[dict] = []
    if not isinstance(artifact, dict):
        return [{"code": "invalid_artifact", "message": "artifact must be a JSON object"}]
    mentions = artifact.get("mentions")
    if not isinstance(mentions, list):
        return [{"code": "invalid_artifact", "message": "artifact.mentions must be a list"}]
    for idx, raw in enumerate(mentions):
        if not isinstance(raw, dict):
            errors.append({"code": "invalid_mention", "path": f"mentions[{idx}]", "message": "mention must be an object"})
            continue
        mention = normalize_mention(raw)
        if mention["kind"] not in VALID_KINDS:
            errors.append({"code": "invalid_kind", "path": f"mentions[{idx}].kind", "value": raw.get("kind") or raw.get("parsed_kind")})
        if not mention["surface"]:
            errors.append({"code": "missing_surface", "path": f"mentions[{idx}].surface"})
        if not mention["identity"].get("family"):
            errors.append({"code": "missing_identity_family", "path": f"mentions[{idx}].identity.family", "surface": mention["surface"]})
        if not mention["evidence"] or not any(e.get("excerpt") for e in mention["evidence"]):
            errors.append({"code": "empty_evidence", "path": f"mentions[{idx}].evidence", "surface": mention["surface"]})
        if mention["kind"] != "dataset" and mention["subsets"]:
            errors.append({"code": "dataset_only_subsets", "path": f"mentions[{idx}].subsets", "surface": mention["surface"]})
        for alias_idx, alias in enumerate(mention["aliases"]):
            if is_generic_alias(alias.get("surface")):
                errors.append({
                    "code": "generic_alias",
                    "path": f"mentions[{idx}].aliases[{alias_idx}]",
                    "surface": alias.get("surface"),
                })
        raw_links = raw.get("links") or {}
        if isinstance(raw_links, dict):
            for field in LINK_FIELDS:
                values = raw_links.get(field) or []
                values = [values] if isinstance(values, str) else values
                if not isinstance(values, list):
                    errors.append({"code": "invalid_link_shape", "path": f"mentions[{idx}].links.{field}", "value": values})
                    continue
                for value in values:
                    if field == "hf_ids" and not normalize_hf_id(value):
                        errors.append({"code": "invalid_link_shape", "path": f"mentions[{idx}].links.hf_ids", "value": value})
                    elif field == "github_repos" and not normalize_github_repo(value):
                        errors.append({"code": "invalid_link_shape", "path": f"mentions[{idx}].links.github_repos", "value": value})
                    elif field == "official_urls" and not normalize_http_url(value):
                        errors.append({"code": "invalid_link_shape", "path": f"mentions[{idx}].links.official_urls", "value": value})
                    elif field == "papers" and not normalize_paper(value):
                        errors.append({"code": "invalid_link_shape", "path": f"mentions[{idx}].links.papers", "value": value})
    return errors


def merge_descriptor_values(current: dict, incoming: dict) -> dict:
    out = deepcopy(current)
    for field, value in incoming.items():
        if field not in out:
            out[field] = value
            continue
        if out[field] == value:
            continue
        existing_values = out[field] if isinstance(out[field], list) else [out[field]]
        new_values = value if isinstance(value, list) else [value]
        out[field] = _dedup([str(v) for v in itertools.chain(existing_values, new_values) if v not in (None, "")])
    return out


def merge_links(current: dict, incoming: dict) -> dict:
    out = {field: list(current.get(field) or []) for field in LINK_FIELDS}
    for field in LINK_FIELDS:
        out[field] = _dedup([*out[field], *list(incoming.get(field) or [])])
    return out


def choose_display_name(aliases: list[dict], identity: dict) -> str:
    candidates = [
        alias["surface"] for alias in aliases
        if isinstance(alias, dict) and not is_generic_alias(alias.get("surface"))
    ]
    if candidates:
        return sorted(candidates, key=lambda s: (len(s), s.casefold()))[0]
    if identity.get("family"):
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
        return "-".join(str(p) for p in parts if p)
    return "unnamed"


def aggregate_mentions(mentions: Iterable[dict]) -> list[dict]:
    clusters: dict[str, dict] = {}

    def add_to_cluster(
        *,
        kind: str,
        identity: dict,
        aliases: list[dict],
        descriptors: dict,
        links: dict,
        subsets: list[dict],
        context_roles: list[str],
        evidence: list[dict],
        mention_id: str | None,
    ) -> None:
        if kind not in VALID_KINDS or not identity.get("family"):
            return
        cluster_key = identity_key(kind, identity)
        if cluster_key not in clusters:
            clusters[cluster_key] = {
                "cluster_key": cluster_key,
                "kind": kind,
                "identity": identity,
                "aliases": [],
                "descriptors": {},
                "links": {field: [] for field in LINK_FIELDS},
                "subsets": [],
                "context_roles": [],
                "evidence": [],
                "mention_ids": [],
                "occurrence_count": 0,
            }
        cluster = clusters[cluster_key]
        cluster["aliases"] = merge_alias_lists(cluster["aliases"], aliases)
        cluster["descriptors"] = merge_descriptor_values(cluster["descriptors"], descriptors)
        cluster["links"] = merge_links(cluster["links"], links)
        cluster["subsets"].extend(subsets)
        cluster["context_roles"] = _dedup([*cluster["context_roles"], *context_roles])
        cluster["evidence"].extend(evidence)
        if mention_id:
            cluster["mention_ids"].append(mention_id)
        cluster["occurrence_count"] += 1

    for raw in mentions:
        mention = normalize_mention(raw)
        add_to_cluster(
            kind=mention["kind"],
            identity=mention["identity"],
            aliases=mention["aliases"],
            descriptors=mention["descriptors"],
            links=mention["links"],
            subsets=mention["subsets"],
            context_roles=mention["context_roles"],
            evidence=mention["evidence"],
            mention_id=mention.get("id"),
        )
        if mention["kind"] == "dataset":
            for subset in mention["subsets"]:
                subset_identity = canonical_identity({
                    **mention["identity"],
                    **(subset.get("identity") or {}),
                })
                subset_name = normalize_space(subset.get("name") or "") or choose_display_name([], subset_identity)
                add_to_cluster(
                    kind="dataset",
                    identity=subset_identity,
                    aliases=normalize_aliases([], surface=subset_name),
                    descriptors=mention["descriptors"],
                    links={field: [] for field in LINK_FIELDS},
                    subsets=[],
                    context_roles=mention["context_roles"],
                    evidence=subset.get("evidence") or mention["evidence"],
                    mention_id=mention.get("id"),
                )
    for cluster in clusters.values():
        cluster["display_name"] = choose_display_name(cluster["aliases"], cluster["identity"])
    return sorted(clusters.values(), key=lambda c: (c["kind"], dumps(c["identity"])))


def merge_alias_lists(current: list[dict], incoming: list[dict]) -> list[dict]:
    out = list(current)
    by_key = {normalize_surface(a.get("surface")): a for a in out if isinstance(a, dict)}
    for alias in incoming:
        if not isinstance(alias, dict):
            continue
        alias_key = normalize_surface(alias.get("surface"))
        if not alias_key:
            continue
        if alias_key in by_key:
            by_key[alias_key]["descriptors"] = merge_descriptor_values(
                by_key[alias_key].get("descriptors") or {},
                alias.get("descriptors") or {},
            )
        else:
            rec = {"surface": alias.get("surface"), "descriptors": alias.get("descriptors") or {}}
            out.append(rec)
            by_key[alias_key] = rec
    return out


def detect_conflicts(mentions: Iterable[dict]) -> list[dict]:
    normalized = [normalize_mention(m) for m in mentions]
    violations: list[dict] = []
    for idx, mention in enumerate(normalized):
        if mention["kind"] not in VALID_KINDS:
            violations.append({
                "code": "invalid_kind",
                "severity": "error",
                "subject_key": f"mention:{idx}",
                "details": {"surface": mention["surface"], "kind": mention["kind"]},
            })
        if not mention["identity"].get("family"):
            violations.append({
                "code": "missing_identity_family",
                "severity": "error",
                "subject_key": mention["surface_key"],
                "details": {"surface": mention["surface"], "kind": mention["kind"]},
            })
        if not mention["evidence"] or not any(e.get("excerpt") for e in mention["evidence"]):
            violations.append({
                "code": "empty_evidence",
                "severity": "error",
                "subject_key": mention["surface_key"],
                "details": {"surface": mention["surface"]},
            })
        for alias in mention["aliases"]:
            if is_generic_alias(alias.get("surface")):
                violations.append({
                    "code": "generic_alias",
                    "severity": "warning",
                    "subject_key": normalize_surface(alias.get("surface")),
                    "details": {"surface": mention["surface"], "alias": alias.get("surface")},
                })
    by_surface: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for mention in normalized:
        if mention["kind"] in VALID_KINDS and mention["surface_key"]:
            by_surface[mention["surface_key"]][mention["identity_key"]].append(mention)
        if mention["kind"] in VALID_KINDS:
            for alias in mention["aliases"]:
                alias_surface_key = normalize_surface(alias.get("surface"))
                if alias_surface_key:
                    by_surface[alias_surface_key][mention["identity_key"]].append(mention)
    for surface_key, identities in by_surface.items():
        if len(identities) > 1:
            violations.append({
                "code": "surface_identity_conflict",
                "severity": "error",
                "subject_key": surface_key,
                "details": {
                    "surface_key": surface_key,
                    "identities": [
                        {
                            "identity_key": identity_key_value,
                            "examples": [m["surface"] for m in mentions_for_identity[:3]],
                            "identity": mentions_for_identity[0]["identity"],
                        }
                        for identity_key_value, mentions_for_identity in identities.items()
                    ],
                },
            })
    by_link: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for mention in normalized:
        if mention["kind"] not in VALID_KINDS:
            continue
        for field in LINK_FIELDS:
            for value in mention["links"].get(field) or []:
                by_link[f"{field}:{value}"][mention["identity_key"]].append(mention)
    for link_key, identities in by_link.items():
        if len(identities) > 1:
            violations.append({
                "code": "link_identity_conflict",
                "severity": "error",
                "subject_key": link_key,
                "details": {
                    "link_key": link_key,
                    "identities": [
                        {
                            "identity_key": identity_key_value,
                            "identity": mentions_for_identity[0]["identity"],
                            "surfaces": [m["surface"] for m in mentions_for_identity[:3]],
                        }
                        for identity_key_value, mentions_for_identity in identities.items()
                    ],
                },
            })
    return violations


def repair_mentions(mentions: list[dict], repair_artifact: dict) -> list[dict]:
    """Apply a compact repair artifact to normalized mention objects.

    Shape:
      {"updates": [{"mention_id": "...", "surface_key": "...", "drop": true,
                    "kind": "model", "identity": {...}, "aliases": [...]}]}
    """
    updates = repair_artifact.get("updates") if isinstance(repair_artifact, dict) else []
    if not isinstance(updates, list):
        updates = []
    out = [normalize_mention(m) for m in mentions]
    for update in updates:
        if not isinstance(update, dict):
            continue
        target_id = update.get("mention_id")
        target_surface_key = update.get("surface_key")
        for mention in out:
            if target_id and mention.get("id") != target_id:
                continue
            if target_surface_key and mention.get("surface_key") != target_surface_key:
                continue
            if update.get("drop"):
                mention["status"] = "dropped"
                continue
            if "kind" in update:
                mention["kind"] = normalize_space(update["kind"]).casefold()
            if "identity" in update:
                mention["identity"] = canonical_identity(update["identity"])
            if "descriptors" in update:
                mention["descriptors"] = normalize_descriptors(update["descriptors"])
            if "aliases" in update:
                mention["aliases"] = normalize_aliases(update["aliases"], surface=mention["surface"], descriptors=mention["descriptors"])
            if "links" in update:
                mention["links"] = normalize_links(update["links"])
            mention["identity_key"] = identity_key(mention["kind"], mention["identity"]) if mention["kind"] in VALID_KINDS else ""
            mention["status"] = "repaired"
    return out
