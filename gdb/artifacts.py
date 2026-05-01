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
LINK_TYPES = config.LINK_TYPES
REFERENT_SCOPES = config.REFERENT_SCOPES
PRIMARY_LINK_ORDER: tuple = (
    "hf_dataset_config",
    ("hf_model", "hf_dataset"),
    "github_ref",
    "github_repo",
    "api_model_id",
    "official_release_url",
    "paper_release",
)


def link_priority(link: dict) -> tuple[int, str, str]:
    link_type = link.get("type")
    priority = 99
    for idx, slot in enumerate(PRIMARY_LINK_ORDER):
        if isinstance(slot, tuple):
            if link_type in slot:
                priority = idx
                break
        elif link_type == slot:
            priority = idx
            break
    return (priority, link_type or "", str(link.get("value") or ""))

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
_HF_DATASET_CONFIG_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)::(?P<config>[^:\s]+)$"
)
_GITHUB_REF_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)(?:@(?P<ref>[^:\s]+))?(?::(?P<path>.+))?$"
)


def normalize_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).casefold()


def normalize_atoms(value: Any, *, surface: str | None = None) -> list[str]:
    raw = value if isinstance(value, list) else []
    if not raw and surface:
        raw = re.split(r"[-_/:\s]+", surface)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        atom = normalize_space(item)
        if not atom:
            continue
        atom = atom.strip("-_/")
        if not atom or atom.casefold() in seen:
            continue
        out.append(atom)
        seen.add(atom.casefold())
    return out


def normalize_concept_path(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [part for part in re.split(r"\s*[>/|]\s*", value) if part]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    out: list[str] = []
    for item in raw:
        part = normalize_space(item)
        if part:
            out.append(part)
    return out


def concept_path_from_identity(identity: dict) -> list[str]:
    if not identity.get("family"):
        return []
    path = [str(identity["family"])]
    for field, value in identity.items():
        if field in {"family", "extra"}:
            continue
        if value not in (None, "", [], {}):
            path.append(str(value))
    extra = identity.get("extra")
    if isinstance(extra, dict):
        concept_path = extra.get("concept_path")
        if isinstance(concept_path, str):
            path.extend(normalize_concept_path(concept_path))
        elif isinstance(concept_path, list):
            path.extend(normalize_concept_path(concept_path))
    return path


def identity_from_concept_path(path: list[str]) -> dict:
    if not path:
        return {}
    identity: dict[str, Any] = {"family": path[0]}
    if len(path) > 1:
        identity["extra"] = {"concept_path": " / ".join(path[1:])}
    return canonical_identity(identity)


def concept_display_name(path: list[str]) -> str:
    return "-".join(path) if path else "concept"


def normalize_scope(value: Any, *, anchors: list[dict] | None = None, concept_path: list[str] | None = None) -> str:
    raw = normalize_space(value).casefold()
    if raw in REFERENT_SCOPES:
        return raw
    if anchors:
        return "entity"
    if concept_path:
        return "concept"
    return "ambiguous"


def is_invalid_alias_surface(surface: Any) -> bool:
    """Structural noise filter for alias surfaces — empty, URL-prefixed,
    over-long, or multi-word with a slash. Used by choose_display_name to
    skip surfaces that are clearly not display-name material. Semantic
    judgment ("this model" / "default" / etc.) is the LLM's job per
    shared-context's alias rules."""
    text = normalize_space(surface)
    if not text:
        return True
    lower = text.casefold()
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
    for raw_field, raw_value in identity.items():
        field = normalize_space(raw_field)
        if not field or field == "extra":
            continue
        value = _clean_scalar(raw_value)
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
        if role not in seen:
            out.append(role)
            seen.add(role)
    return out or ["unknown"]


def normalize_aux(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        k = normalize_space(raw_key)
        if not k:
            continue
        if isinstance(raw_value, list):
            cleaned = [_clean_scalar(v) for v in raw_value]
            cleaned = [v for v in cleaned if v not in (None, "", [], {})]
            if cleaned:
                out[k] = cleaned
        elif isinstance(raw_value, dict):
            nested = normalize_aux(raw_value)
            if nested:
                out[k] = nested
        else:
            cleaned = _clean_scalar(raw_value)
            if cleaned not in (None, "", [], {}):
                out[k] = cleaned
    return {k: out[k] for k in sorted(out)}


def normalize_relationship_hints(value: Any) -> list[dict]:
    raw = value if isinstance(value, list) else ([value] if value else [])
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            relation = normalize_space(item)
            if relation:
                out.append({"relation": relation})
        elif isinstance(item, dict):
            relation = normalize_space(item.get("relation") or item.get("type") or "")
            target = normalize_space(item.get("target") or item.get("object") or "")
            rec = {k: v for k, v in {"relation": relation, "target": target}.items() if v}
            if rec:
                out.append(rec)
    return out


def normalize_aliases(
    aliases: Any,
    surface: str | None = None,
    descriptors: dict | None = None,
    *,
    kind: str | None = None,
) -> list[dict]:
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
            alias_descriptors: dict = {}
            alias_anchors: list[dict] = []
        elif isinstance(item, dict):
            alias_surface = normalize_space(item.get("surface") or item.get("name") or "")
            alias_descriptors = normalize_descriptors(item.get("descriptors") or {})
            alias_anchors = normalize_link_candidates(
                item.get("anchors") or item.get("links") or [],
                kind=kind,
            )
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
            if alias_anchors:
                by_key[alias_key]["anchors"] = normalize_link_candidates(
                    [*(by_key[alias_key].get("anchors") or []), *alias_anchors],
                    kind=kind,
                )
            continue
        rec: dict = {"surface": alias_surface, "descriptors": alias_descriptors}
        if alias_anchors:
            rec["anchors"] = alias_anchors
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


def normalize_hf_dataset_config(value: Any, *, repo: Any = None, config_name: Any = None) -> str | None:
    repo_text = normalize_hf_id(repo) if repo else None
    config_text = normalize_space(config_name) if config_name else None
    text = normalize_space(value)
    if text:
        match = _HF_DATASET_CONFIG_RE.match(text)
        if match:
            repo_text = normalize_hf_id(match.group("repo"))
            config_text = normalize_space(match.group("config"))
        elif "::" in text:
            raw_repo, raw_config = text.split("::", 1)
            repo_text = normalize_hf_id(raw_repo)
            config_text = normalize_space(raw_config)
    if repo_text and config_text:
        return f"{repo_text}::{config_text}"
    return None


def normalize_github_ref(value: Any, *, repo: Any = None, ref: Any = None, path: Any = None) -> str | None:
    repo_text = normalize_github_repo(repo) if repo else None
    ref_text = normalize_space(ref) if ref else ""
    path_text = normalize_space(path) if path else ""
    text = normalize_space(value)
    if text:
        match = _GITHUB_REF_RE.match(text)
        if match:
            repo_text = normalize_github_repo(match.group("repo"))
            ref_text = normalize_space(match.group("ref") or ref_text)
            path_text = normalize_space(match.group("path") or path_text)
    if not repo_text:
        return None
    suffix = f"@{ref_text}" if ref_text else ""
    if path_text:
        suffix += f":{path_text}"
    return f"{repo_text}{suffix}"


def normalize_link_candidate(item: Any, *, kind: str | None = None) -> dict | None:
    if isinstance(item, str):
        raw_type = ""
        raw_value = item
        exact = True
        source = ""
        metadata = {}
    elif isinstance(item, dict):
        raw_type = normalize_space(item.get("type") or item.get("anchor_type") or item.get("link_type") or "").casefold()
        raw_value = item.get("value") or item.get("id") or item.get("repo") or item.get("url") or item.get("name") or ""
        exact = bool(item.get("exact", True))
        source = normalize_space(item.get("source") or "")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    else:
        return None

    anchor_type = raw_type
    if anchor_type in {"hf", "hf_id", "huggingface", "huggingface_id"}:
        anchor_type = "hf_dataset" if kind == "dataset" else "hf_model"
    elif anchor_type in {"github", "github_repository"}:
        anchor_type = "github_repo"
    elif anchor_type in {"api", "model_id"}:
        anchor_type = "api_model_id"
    elif anchor_type in {"official", "url"}:
        anchor_type = "official_release_url"
    elif anchor_type in {"paper", "arxiv"}:
        anchor_type = "paper_release"

    if not anchor_type:
        if normalize_hf_id(raw_value):
            anchor_type = "hf_dataset" if kind == "dataset" else "hf_model"
        elif normalize_github_repo(raw_value):
            anchor_type = "github_repo"
        elif normalize_paper(raw_value) and "arxiv" in normalize_space(raw_value).casefold():
            anchor_type = "paper_release"
        elif normalize_http_url(raw_value):
            anchor_type = "official_release_url"
        else:
            anchor_type = "api_model_id"

    value: str | None
    url: str | None = None
    if anchor_type == "hf_model":
        value = normalize_hf_id(raw_value)
        url = f"https://huggingface.co/{value}" if value else None
    elif anchor_type == "hf_dataset":
        value = normalize_hf_id(raw_value)
        url = f"https://huggingface.co/datasets/{value}" if value else None
    elif anchor_type == "hf_dataset_config":
        if isinstance(item, dict):
            value = normalize_hf_dataset_config(raw_value, repo=item.get("repo"), config_name=item.get("config") or item.get("config_name"))
        else:
            value = normalize_hf_dataset_config(raw_value)
        repo_value = value.split("::", 1)[0] if value else None
        url = f"https://huggingface.co/datasets/{repo_value}" if repo_value else None
    elif anchor_type == "github_repo":
        value = normalize_github_repo(raw_value)
        url = f"https://github.com/{value}" if value else None
    elif anchor_type == "github_ref":
        if isinstance(item, dict):
            value = normalize_github_ref(raw_value, repo=item.get("repo"), ref=item.get("ref"), path=item.get("path"))
        else:
            value = normalize_github_ref(raw_value)
        repo_value = value.split("@", 1)[0].split(":", 1)[0] if value else None
        url = f"https://github.com/{repo_value}" if repo_value else None
    elif anchor_type == "api_model_id":
        value = normalize_space(raw_value)
    elif anchor_type == "official_release_url":
        value = normalize_http_url(raw_value)
        url = value
    elif anchor_type == "paper_release":
        value = normalize_paper(raw_value)
        url = value
    else:
        return None

    if not value:
        return None
    rec = {"type": anchor_type, "value": value, "exact": exact}
    if url:
        rec["url"] = url
    if source:
        rec["source"] = source
    if metadata:
        rec["metadata"] = normalize_aux(metadata)
    return rec


def normalize_link_candidates(value: Any, *, kind: str | None = None) -> list[dict]:
    raw = value if isinstance(value, list) else ([value] if value else [])
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        anchor = normalize_link_candidate(item, kind=kind)
        if not anchor:
            continue
        key_tuple = (anchor["type"], anchor["value"])
        if key_tuple in seen:
            continue
        out.append(anchor)
        seen.add(key_tuple)
    return out


def sort_links(links: list[dict]) -> list[dict]:
    return sorted(links, key=link_priority)


def primary_link(links: list[dict]) -> dict | None:
    exact = [link for link in links if link.get("exact")]
    if not exact:
        return None
    return sort_links(exact)[0]


def normalize_anchors(anchors: Any) -> list[dict]:
    raw = anchors if isinstance(anchors, list) else ([anchors] if anchors else [])
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
        links = normalize_link_candidates(item.get("links") or [], kind="dataset")
        source_anchors = normalize_anchors(item.get("source_anchors") or [])
        if name or identity:
            rec = {"name": name, "identity": identity}
            if links:
                rec["links"] = links
            if source_anchors:
                rec["source_anchors"] = source_anchors
            out.append(rec)
    return out


def normalize_mention(item: dict, *, batch_id: str | None = None, source_id: str | None = None) -> dict:
    surface = normalize_space(item.get("surface") or item.get("raw_text") or "")
    kind = normalize_space(item.get("kind") or item.get("parsed_kind") or "").casefold()
    raw_identity = item.get("identity") or item.get("parsed_identity") or item.get("parsed_facets") or {}
    identity = canonical_identity(raw_identity)
    atoms = normalize_atoms(item.get("atoms") or item.get("name_atoms") or [], surface=surface)
    concept_path = normalize_concept_path(
        item.get("concept_path")
        or item.get("lattice_path")
        or item.get("family_path")
        or []
    )
    if not concept_path and identity:
        concept_path = concept_path_from_identity(identity)
    if not identity and concept_path:
        identity = identity_from_concept_path(concept_path)
    descriptors = normalize_descriptors(item.get("descriptors") or {})
    aliases = normalize_aliases(item.get("aliases") or [], surface=surface, descriptors=descriptors, kind=kind)
    links = normalize_link_candidates(
        item.get("links") or item.get("anchor_candidates") or item.get("connections") or [],
        kind=kind,
    )
    roles = normalize_roles(item.get("context_roles") or descriptors.get("context_roles") or ["unknown"])
    raw_anchors = item.get("anchors")
    if raw_anchors is None:
        raw_anchors = item.get("evidence")
    if not raw_anchors and (item.get("file") or item.get("excerpt") or item.get("location")):
        raw_anchors = [{
            "source_id": item.get("source_id") or source_id or "",
            "file": item.get("file") or "",
            "location": item.get("location") or "",
            "excerpt": item.get("excerpt") or "",
        }]
    anchors = normalize_anchors(raw_anchors or [])
    subsets = normalize_subsets(item.get("subsets") or [])
    attrs = item.get("attrs") if isinstance(item.get("attrs"), dict) else {}
    aux = normalize_aux(item.get("aux") or item.get("aux_info") or attrs.get("aux") or {})
    relationships = normalize_relationship_hints(
        item.get("relationships")
        or attrs.get("relationships")
        or []
    )
    description = normalize_space(item.get("description") or attrs.get("description") or "")
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
        "subsets": subsets,
        "context_roles": roles,
        "atoms": atoms,
        "referent_scope": normalize_scope(item.get("referent_scope") or item.get("scope"), anchors=links, concept_path=concept_path),
        "links": links,
        "concept_path": concept_path,
        "aux": aux,
        "relationships": relationships,
        "anchors": anchors,
        "description": description or None,
        "notes": normalize_space(item.get("notes") or item.get("rationale") or "") or None,
        "attrs": attrs,
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
        if not mention["identity"].get("family") and not mention["concept_path"]:
            errors.append({"code": "missing_identity_family", "path": f"mentions[{idx}].identity.family", "surface": mention["surface"]})
        if not mention["anchors"] or not any(e.get("excerpt") for e in mention["anchors"]):
            errors.append({"code": "empty_anchors", "path": f"mentions[{idx}].anchors", "surface": mention["surface"]})
        if mention["kind"] != "dataset" and mention["subsets"]:
            errors.append({"code": "dataset_only_subsets", "path": f"mentions[{idx}].subsets", "surface": mention["surface"]})
        raw_links = raw.get("links") or raw.get("anchor_candidates") or []
        if raw_links:
            link_items = raw_links if isinstance(raw_links, list) else [raw_links]
            for anchor_idx, anchor_item in enumerate(link_items):
                if not normalize_link_candidate(anchor_item, kind=mention["kind"]):
                    errors.append({
                        "code": "invalid_link_shape",
                        "path": f"mentions[{idx}].links[{anchor_idx}]",
                        "value": anchor_item,
                    })
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


def choose_display_name(aliases: list[dict], identity: dict) -> str:
    candidates = [
        alias["surface"] for alias in aliases
        if isinstance(alias, dict) and not is_invalid_alias_surface(alias.get("surface"))
    ]
    if candidates:
        return sorted(candidates, key=lambda s: (len(s), s.casefold()))[0]
    path = concept_path_from_identity(identity)
    if path:
        return concept_display_name(path)
    return "unnamed"


def aggregate_mentions(mentions: Iterable[dict]) -> list[dict]:
    clusters: dict[str, dict] = {}

    def add_to_cluster(
        *,
        kind: str,
        identity: dict,
        aliases: list[dict],
        descriptors: dict,
        subsets: list[dict],
        context_roles: list[str],
        atoms: list[str],
        referent_scope: str,
        links: list[dict],
        concept_path: list[str],
        aux: dict,
        relationships: list[dict],
        anchors: list[dict],
        description: str | None,
        mention_id: str | None,
    ) -> None:
        if kind not in VALID_KINDS or not identity.get("family"):
            return
        exact_links = [link for link in links if link.get("exact")]
        if exact_links:
            primary = primary_link(exact_links)
            cluster_key = f"{kind}:link:{primary['type']}:{hash_text(primary['value'])}"
        else:
            cluster_key = identity_key(kind, identity)
        if cluster_key not in clusters:
            clusters[cluster_key] = {
                "cluster_key": cluster_key,
                "kind": kind,
                "identity": identity,
                "aliases": [],
                "descriptors": {},
                "subsets": [],
                "context_roles": [],
                "atoms": [],
                "referent_scopes": [],
                "links": [],
                "concept_path": concept_path,
                "aux": {},
                "relationships": [],
                "anchors": [],
                "descriptions": [],
                "mention_ids": [],
                "occurrence_count": 0,
            }
        cluster = clusters[cluster_key]
        cluster["aliases"] = merge_alias_lists(cluster["aliases"], aliases, kind=kind)
        cluster["descriptors"] = merge_descriptor_values(cluster["descriptors"], descriptors)
        cluster["subsets"].extend(subsets)
        cluster["context_roles"] = _dedup([*cluster["context_roles"], *context_roles])
        cluster["atoms"] = _dedup([*cluster["atoms"], *atoms])
        cluster["referent_scopes"] = _dedup([*cluster["referent_scopes"], referent_scope])
        cluster["links"] = normalize_link_candidates([*cluster["links"], *links], kind=kind)
        cluster["aux"] = merge_descriptor_values(cluster["aux"], aux)
        cluster["relationships"].extend(relationships)
        cluster["anchors"].extend(anchors)
        if description:
            cluster["descriptions"].append(description)
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
            subsets=mention["subsets"],
            context_roles=mention["context_roles"],
            atoms=mention["atoms"],
            referent_scope=mention["referent_scope"],
            links=mention["links"],
            concept_path=mention["concept_path"],
            aux=mention["aux"],
            relationships=mention["relationships"],
            anchors=mention["anchors"],
            description=mention.get("description"),
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
                    aliases=normalize_aliases([], surface=subset_name, kind="dataset"),
                    descriptors=mention["descriptors"],
                    subsets=[],
                    context_roles=mention["context_roles"],
                    atoms=mention["atoms"],
                    referent_scope=mention["referent_scope"],
                    links=normalize_link_candidates(subset.get("links") or subset.get("anchors") or [], kind="dataset"),
                    concept_path=concept_path_from_identity(subset_identity),
                    aux=mention["aux"],
                    relationships=mention["relationships"],
                    anchors=subset.get("source_anchors") or mention["anchors"],
                    description=mention.get("description"),
                    mention_id=mention.get("id"),
                )
    for cluster in clusters.values():
        cluster["display_name"] = choose_display_name(cluster["aliases"], cluster["identity"])
        cluster["description"] = cluster["descriptions"][0] if cluster["descriptions"] else None
    return sorted(clusters.values(), key=lambda c: (c["kind"], dumps(c["identity"])))


def merge_alias_lists(current: list[dict], incoming: list[dict], *, kind: str | None = None) -> list[dict]:
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
            if alias.get("anchors"):
                by_key[alias_key]["anchors"] = normalize_link_candidates(
                    [*(by_key[alias_key].get("anchors") or []), *(alias.get("anchors") or [])],
                    kind=kind,
                )
        else:
            rec: dict = {"surface": alias.get("surface"), "descriptors": alias.get("descriptors") or {}}
            if alias.get("anchors"):
                rec["anchors"] = normalize_link_candidates(alias.get("anchors") or [], kind=kind)
            out.append(rec)
            by_key[alias_key] = rec
    return out


def primary_referent_signature(mention: dict) -> tuple[str, str]:
    anchors = [a for a in mention.get("links") or [] if a.get("exact")]
    if anchors:
        anchor = sorted(anchors, key=lambda a: (a.get("type", ""), a.get("value", "")))[0]
        return ("entity", f"{anchor.get('type')}:{anchor.get('value')}")
    if mention.get("concept_path"):
        return ("concept", " / ".join(mention["concept_path"]).casefold())
    if mention.get("identity_key"):
        return ("concept", mention["identity_key"])
    return ("ambiguous", mention.get("surface_key") or "")


def cluster_key_for_mention(mention: dict) -> str:
    """Return the cluster key aggregate_mentions would assign to this mention.

    Matches the clustering rule in aggregate_mentions.add_to_cluster: an
    exact primary anchor (if present) takes priority over identity-based
    keying. Returns "" for mentions outside VALID_KINDS or missing
    family.
    """
    kind = mention.get("kind")
    if kind not in VALID_KINDS:
        return ""
    identity = mention.get("identity") or {}
    if not identity.get("family"):
        return ""
    exact_anchors = [a for a in mention.get("links") or [] if a.get("exact")]
    if exact_anchors:
        primary = primary_link(exact_anchors)
        if primary:
            return f"{kind}:anchor:{primary['type']}:{hash_text(primary['value'])}"
    return identity_key(kind, identity)


def allow_concept_entity_surface_duplicate(referents: dict[tuple[str, str], list[dict]]) -> bool:
    if len(referents) != 2:
        return False
    scopes = {scope for scope, _sig in referents}
    return scopes == {"concept", "entity"}


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
        if not mention["identity"].get("family") and not mention["concept_path"]:
            violations.append({
                "code": "missing_identity_family",
                "severity": "error",
                "subject_key": mention["surface_key"],
                "details": {"surface": mention["surface"], "kind": mention["kind"]},
            })
        if not mention["anchors"] or not any(e.get("excerpt") for e in mention["anchors"]):
            violations.append({
                "code": "empty_anchors",
                "severity": "error",
                "subject_key": mention["surface_key"],
                "details": {"surface": mention["surface"]},
            })
    by_surface: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for mention in normalized:
        if mention["kind"] in VALID_KINDS and mention["surface_key"]:
            by_surface[mention["surface_key"]][primary_referent_signature(mention)].append(mention)
        if mention["kind"] in VALID_KINDS:
            for alias in mention["aliases"]:
                alias_surface_key = normalize_surface(alias.get("surface"))
                if alias_surface_key:
                    by_surface[alias_surface_key][primary_referent_signature(mention)].append(mention)
    for surface_key, referents in by_surface.items():
        if len(referents) > 1 and not allow_concept_entity_surface_duplicate(referents):
            violations.append({
                "code": "surface_identity_conflict",
                "severity": "error",
                "subject_key": surface_key,
                "details": {
                    "surface_key": surface_key,
                    "identities": [
                        {
                            "identity_key": signature_value,
                            "referent_scope": scope,
                            "examples": [m["surface"] for m in mentions_for_referent[:3]],
                            "identity": mentions_for_referent[0]["identity"],
                            "concept_path": mentions_for_referent[0]["concept_path"],
                            "anchors": mentions_for_referent[0]["links"],
                        }
                        for (scope, signature_value), mentions_for_referent in referents.items()
                    ],
                },
            })
    by_link: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for mention in normalized:
        if mention["kind"] not in VALID_KINDS:
            continue
        for link in mention.get("links") or []:
            key = f"{link.get('type')}:{link.get('value')}"
            by_link[key][mention["identity_key"]].append(mention)
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
    by_anchor: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for mention in normalized:
        if mention["kind"] not in VALID_KINDS:
            continue
        concept_sig = " / ".join(mention.get("concept_path") or []).casefold() or mention["identity_key"]
        for anchor in mention.get("links") or []:
            if not anchor.get("exact"):
                continue
            by_anchor[f"{anchor['type']}:{anchor['value']}"][concept_sig].append(mention)
    for anchor_key, concept_sigs in by_anchor.items():
        if len(concept_sigs) > 1:
            violations.append({
                "code": "link_concept_conflict",
                "severity": "error",
                "subject_key": anchor_key,
                "details": {
                    "anchor_key": anchor_key,
                    "concepts": [
                        {
                            "concept_signature": concept_sig,
                            "surfaces": [m["surface"] for m in mentions_for_concept[:3]],
                            "concept_path": mentions_for_concept[0]["concept_path"],
                            "identity": mentions_for_concept[0]["identity"],
                        }
                        for concept_sig, mentions_for_concept in concept_sigs.items()
                    ],
                },
            })
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for mention in normalized:
        ckey = cluster_key_for_mention(mention)
        if ckey:
            by_cluster[ckey].append(mention)
    for cluster_key, group in by_cluster.items():
        if len(group) < 2:
            continue
        aux_values_by_key: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for mention in group:
            for aux_key, aux_value in (mention.get("aux") or {}).items():
                if aux_value in (None, "", [], {}):
                    continue
                serialized = json.dumps(aux_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                aux_values_by_key[aux_key].append((serialized, mention))
        for aux_key, value_mentions in aux_values_by_key.items():
            distinct_serialized = {serialized for serialized, _ in value_mentions}
            if len(distinct_serialized) <= 1:
                continue
            seen_values: list[Any] = []
            seen_keys: set[str] = set()
            for serialized, mention in value_mentions:
                if serialized in seen_keys:
                    continue
                seen_keys.add(serialized)
                seen_values.append((mention.get("aux") or {}).get(aux_key))
            violations.append({
                "code": "aux_conflict",
                "severity": "error",
                "subject_key": cluster_key,
                "details": {
                    "key": aux_key,
                    "values": seen_values,
                    "mention_ids": [m.get("id") for _, m in value_mentions if m.get("id")],
                    "surfaces": _dedup([m.get("surface") for _, m in value_mentions if m.get("surface")]),
                },
            })
    return violations


def apply_audit_updates(mentions: list[dict], repair_artifact: dict) -> list[dict]:
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
                mention["aliases"] = normalize_aliases(update["aliases"], surface=mention["surface"], descriptors=mention["descriptors"], kind=mention.get("kind"))
            if "atoms" in update:
                mention["atoms"] = normalize_atoms(update["atoms"], surface=mention["surface"])
            if "concept_path" in update or "lattice_path" in update:
                mention["concept_path"] = normalize_concept_path(update.get("concept_path") or update.get("lattice_path") or [])
                if not mention["identity"].get("family") and mention["concept_path"]:
                    mention["identity"] = identity_from_concept_path(mention["concept_path"])
            if "links" in update or "anchor_candidates" in update:
                mention["links"] = normalize_link_candidates(update.get("links") or update.get("anchor_candidates") or [], kind=mention["kind"])
            if "anchors" in update:
                mention["anchors"] = normalize_anchors(update.get("anchors") or [])
            if "referent_scope" in update or "scope" in update:
                mention["referent_scope"] = normalize_scope(update.get("referent_scope") or update.get("scope"), anchors=mention["links"], concept_path=mention["concept_path"])
            if "aux" in update or "aux_info" in update:
                mention["aux"] = normalize_aux(update.get("aux") or update.get("aux_info") or {})
            if "relationships" in update or "relationship_hints" in update:
                mention["relationships"] = normalize_relationship_hints(update.get("relationships") or update.get("relationship_hints") or [])
            if "description" in update:
                mention["description"] = normalize_space(update.get("description") or "") or None
            mention["identity_key"] = identity_key(mention["kind"], mention["identity"]) if mention["kind"] in VALID_KINDS else ""
            mention["status"] = "repaired"
    return out
