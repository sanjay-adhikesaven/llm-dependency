from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.parse import quote, urlencode

import httpx
import yaml

from . import config
from .store import normalize_space


def hf_repo_id(anchor: dict) -> str | None:
    anchor_type = anchor.get("type")
    value = normalize_space(anchor.get("value") or "")
    if anchor_type in {"hf_model", "hf_dataset"}:
        return value
    if anchor_type == "hf_dataset_config":
        return value.split("::", 1)[0]
    return None


def hf_kind(anchor: dict) -> str | None:
    if anchor.get("type") == "hf_model":
        return "model"
    if anchor.get("type") in {"hf_dataset", "hf_dataset_config"}:
        return "dataset"
    return None


def hf_repo_url(anchor: dict) -> str | None:
    repo_id = hf_repo_id(anchor)
    if not repo_id:
        return None
    if hf_kind(anchor) == "dataset":
        return f"{config.HF_BASE}/datasets/{repo_id}"
    return f"{config.HF_BASE}/{repo_id}"


def hf_readme_url(anchor: dict) -> str | None:
    repo_url = hf_repo_url(anchor)
    return f"{repo_url}/raw/main/README.md" if repo_url else None


def hf_api_url(anchor: dict) -> str | None:
    repo_id = hf_repo_id(anchor)
    kind = hf_kind(anchor)
    if not repo_id or not kind:
        return None
    encoded = quote(repo_id, safe="")
    endpoint = "datasets" if kind == "dataset" else "models"
    return f"{config.HF_API_BASE}/{endpoint}/{encoded}"


def parse_yaml(raw: str) -> Any:
    loaded = yaml.safe_load(raw)
    return loaded if loaded is not None else {}


def parse_hf_readme_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    yaml_text = text[3:end]
    body = text[end + len("\n---"):].lstrip()
    try:
        parsed = parse_yaml(yaml_text)
    except yaml.YAMLError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}, body


def first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return normalize_space(stripped.lstrip("#"))
    return None


def first_body_paragraphs(text: str, *, max_paragraphs: int = 2, max_chars: int = 700) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith(("#", "<", "!", "[", "|", "-", "*")):
            continue
        if not stripped:
            if current:
                paragraphs.append(normalize_space(" ".join(current)))
                current = []
            if len(paragraphs) >= max_paragraphs:
                break
            continue
        current.append(stripped)
    if current and len(paragraphs) < max_paragraphs:
        paragraphs.append(normalize_space(" ".join(current)))
    text_out = " ".join(p for p in paragraphs if p)
    return text_out[:max_chars].rstrip()


def _walk_configs(value: Any) -> list[str]:
    configs: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"config_name", "name"} and isinstance(child, str):
                configs.append(normalize_space(child))
            configs.extend(_walk_configs(child))
    elif isinstance(value, list):
        for child in value:
            configs.extend(_walk_configs(child))
    return [c for c in dict.fromkeys(configs) if c]


def extract_dataset_configs(front_matter: dict, api_payload: dict) -> list[str]:
    configs = []
    configs.extend(_walk_configs(front_matter.get("configs")))
    card_data = api_payload.get("cardData") if isinstance(api_payload, dict) else {}
    if isinstance(card_data, dict):
        configs.extend(_walk_configs(card_data.get("configs")))
    siblings = api_payload.get("siblings") if isinstance(api_payload, dict) else []
    if isinstance(siblings, list):
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            rfilename = sibling.get("rfilename")
            if isinstance(rfilename, str) and rfilename.startswith("data/"):
                parts = rfilename.split("/")
                if len(parts) > 2:
                    configs.append(parts[1])
    return sorted(set(c for c in configs if c and c != "default"))


def collection_api_url(namespace: str, slug: str) -> str:
    return f"{config.HF_API_BASE}/collections/{quote(namespace, safe='')}/{quote(slug, safe='')}"


def collection_search_url(namespace: str, *, item: str | None = None, q: str | None = None, limit: int = 20) -> str:
    params = {"owner": namespace, "limit": str(limit)}
    if item:
        params["item"] = item
    if q:
        params["q"] = q
    return f"{config.HF_API_BASE}/collections?{urlencode(params)}"


def collection_item_id(anchor: dict) -> str | None:
    repo_id = hf_repo_id(anchor)
    kind = hf_kind(anchor)
    if not repo_id or not kind:
        return None
    prefix = "datasets" if kind == "dataset" else "models"
    return f"{prefix}/{repo_id}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", value.lower()) if token]


def _is_release_qualifier(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?[bkmt]?", token) or re.fullmatch(r"\d{3,8}", token))


def infer_collection_candidates(anchor: dict, front_matter: dict, api_payload: dict) -> list[dict]:
    repo_id = hf_repo_id(anchor) or ""
    repo_name = repo_id.rsplit("/", 1)[-1]
    namespace = repo_id.split("/", 1)[0] if "/" in repo_id else ""
    candidates: list[dict] = []
    title_candidates = [repo_name, repo_name.lower()]
    repo_tokens = _name_tokens(repo_name)
    stable_tokens = [token for token in repo_tokens if not _is_release_qualifier(token)]
    for depth in range(1, min(4, len(stable_tokens)) + 1):
        title_candidates.append("-".join(stable_tokens[:depth]))
    for tag in api_payload.get("tags") or []:
        if not isinstance(tag, str):
            continue
        tag_tokens = set(_name_tokens(tag))
        if tag_tokens and tag_tokens.intersection(repo_tokens):
            title_candidates.append(tag)
    item_id = collection_item_id(anchor)
    if namespace and item_id:
        candidates.append({
            "namespace": namespace,
            "query_type": "item",
            "query": item_id,
            "url": collection_search_url(namespace, item=item_id),
        })
    for title in title_candidates:
        slug = _slugify(title)
        if namespace and slug:
            candidates.append({
                "namespace": namespace,
                "query_type": "q",
                "query": slug,
                "slug": slug,
                "url": collection_search_url(namespace, q=slug),
            })
    return [dict(t) for t in {tuple(sorted(c.items())) for c in candidates}]


def normalize_collection_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {}
    items = payload.get("items") or payload.get("repos") or []
    repos = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            repo = item.get("repo") or item.get("item") or {}
            if isinstance(repo, dict):
                repo_id = repo.get("id") or repo.get("name")
            else:
                repo_id = item.get("id") or item.get("name")
            if repo_id:
                repos.append(str(repo_id))
    slug = payload.get("slug") or payload.get("id")
    url = payload.get("url")
    if not url and isinstance(slug, str):
        url = f"{config.HF_BASE}/collections/{slug}"
    title = payload.get("title") or payload.get("name")
    if not slug and not title and not repos:
        return {}
    return {
        "slug": slug,
        "title": title,
        "description": payload.get("description") or "",
        "url": url,
        "repos": sorted(set(repos)),
    }


def normalize_collection_payloads(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in (normalize_collection_payload(p) for p in payload) if item]
    if isinstance(payload, dict):
        collections = payload.get("collections") or payload.get("items")
        if isinstance(collections, list) and not payload.get("slug"):
            return [item for item in (normalize_collection_payload(p) for p in collections) if item]
        normalized = normalize_collection_payload(payload)
        return [normalized] if normalized else []
    return []


def describe_from_hf_metadata(anchor: dict, front_matter: dict[str, Any], api_payload: dict[str, Any], body: str = "") -> str:
    pieces = []
    heading = first_markdown_heading(body)
    if heading:
        pieces.append(heading)
    card_data = api_payload.get("cardData") if isinstance(api_payload.get("cardData"), dict) else {}
    for key in ("pipeline_tag", "library_name", "license", "base_model", "datasets", "tags"):
        value = front_matter.get(key, card_data.get(key))
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value[:8])
        pieces.append(f"{key}={value}")
    body_summary = first_body_paragraphs(body)
    if body_summary:
        pieces.append(body_summary)
    if not pieces:
        pieces.append(f"{anchor.get('type')}:{anchor.get('value')}")
    return "; ".join(pieces)


def default_fetch_text(url: str) -> tuple[int | None, str | None, str | None]:
    try:
        with httpx.Client(timeout=config.LINK_TIMEOUT_S, follow_redirects=True) as client:
            response = client.get(url)
        if 200 <= response.status_code < 400:
            return response.status_code, response.text, None
        return response.status_code, None, response.text[:500] if response.text else None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


def default_fetch_json(url: str) -> tuple[int | None, Any | None, str | None]:
    status, text, error = default_fetch_text(url)
    if not text:
        return status, None, error
    try:
        loaded = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return status, None, f"{type(exc).__name__}: {exc}"
    return status, loaded, None


def relationship_hints_from_metadata(anchor: dict, front_matter: dict, api_payload: dict) -> list[dict]:
    hints: list[dict] = []
    card_data = api_payload.get("cardData") if isinstance(api_payload.get("cardData"), dict) else {}
    for key, relation in (("base_model", "base_model"), ("datasets", "trained_on"), ("dataset", "trained_on")):
        value = front_matter.get(key, card_data.get(key))
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item in (None, "", [], {}):
                continue
            hints.append({
                "source_anchor": anchor,
                "relation": relation,
                "target": normalize_space(item),
                "evidence": [{"excerpt": f"HF metadata {key}: {item}"}],
            })
    return hints


def enrich_hf_anchor(
    anchor: dict,
    *,
    fetch_text: Callable[[str], tuple[int | None, str | None, str | None]] = default_fetch_text,
    fetch_json: Callable[[str], tuple[int | None, dict | None, str | None]] = default_fetch_json,
) -> dict:
    readme_url = hf_readme_url(anchor)
    api_url = hf_api_url(anchor)
    repo_url = hf_repo_url(anchor)
    front_matter: dict[str, Any] = {}
    body = ""
    api_payload: dict[str, Any] = {}
    errors: list[str] = []

    readme_status: int | None = None
    api_status: int | None = None
    if readme_url:
        readme_status, text, error = fetch_text(readme_url)
        if text:
            front_matter, body = parse_hf_readme_front_matter(text)
        elif error:
            errors.append(error)
    if api_url:
        api_status, payload, error = fetch_json(api_url)
        if payload:
            api_payload = payload
        elif error:
            errors.append(error)

    configs = extract_dataset_configs(front_matter, api_payload) if hf_kind(anchor) == "dataset" else []
    config_valid = None
    if anchor.get("type") == "hf_dataset_config":
        config_name = normalize_space((anchor.get("value") or "").split("::", 1)[1] if "::" in (anchor.get("value") or "") else "")
        config_valid = config_name in configs or not configs

    collections = []
    for candidate in infer_collection_candidates(anchor, front_matter, api_payload)[:8]:
        status, payload, error = fetch_json(candidate["url"])
        if payload:
            for normalized in normalize_collection_payloads(payload):
                normalized["query_url"] = candidate["url"]
                normalized["query_type"] = candidate.get("query_type")
                normalized["status_code"] = status
                collections.append(normalized)
        elif error:
            continue
    deduped_collections = []
    seen_collections = set()
    for item in collections:
        key = item.get("slug") or item.get("url") or item.get("title")
        if key in seen_collections:
            continue
        deduped_collections.append(item)
        seen_collections.add(key)

    description = describe_from_hf_metadata(anchor, front_matter, api_payload, body)
    metadata = {
        "front_matter": front_matter,
        "api": api_payload,
        "readme_heading": first_markdown_heading(body),
        "body_summary": first_body_paragraphs(body),
        "config_valid": config_valid,
    }
    return {
        "anchor": anchor,
        "ok": bool(front_matter or api_payload or readme_status in range(200, 400) or api_status in range(200, 400)),
        "repo_url": repo_url,
        "readme_url": readme_url,
        "api_url": api_url,
        "metadata": metadata,
        "card_data": api_payload.get("cardData") if isinstance(api_payload.get("cardData"), dict) else {},
        "configs": configs,
        "collections": deduped_collections,
        "relationships": relationship_hints_from_metadata(anchor, front_matter, api_payload),
        "description": description,
        "error": "; ".join(errors),
        "source": {"readme_status": readme_status, "api_status": api_status},
    }
