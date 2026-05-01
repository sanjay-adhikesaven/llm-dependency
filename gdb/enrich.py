"""HF README front-matter + API fetch for entity descriptions.

Public surface: `enrich_hf_link(link)` returns a dict with the parsed
front-matter, card data, fetched URLs, and a short description seed.
Used inline by `run_describe` for entity-leaf nodes that have an HF
link. The categorized-collection inference and relationship-hint
extraction from the prior version are gone — those concerns belong
to dedicated stages (or v1.5).
"""

from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import quote

import httpx
import yaml

from . import config
from .store import normalize_space


_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def hf_repo_id(link: dict) -> str | None:
    if not isinstance(link, dict):
        return None
    value = link.get("value") or ""
    if "::" in value:
        return value.split("::", 1)[0]
    return value or None


def hf_kind(link: dict) -> str:
    return "dataset" if link.get("type") in {"hf_dataset", "hf_dataset_config"} else "model"


def hf_repo_url(link: dict) -> str | None:
    repo = hf_repo_id(link)
    if not repo:
        return None
    if hf_kind(link) == "dataset":
        return f"{config.HF_BASE}/datasets/{repo}"
    return f"{config.HF_BASE}/{repo}"


def hf_readme_url(link: dict) -> str | None:
    repo = hf_repo_id(link)
    if not repo:
        return None
    base = "datasets/" if hf_kind(link) == "dataset" else ""
    return f"{config.HF_BASE}/{base}{repo}/raw/main/README.md"


def hf_api_url(link: dict) -> str | None:
    repo = hf_repo_id(link)
    if not repo:
        return None
    encoded = quote(repo, safe="")
    if hf_kind(link) == "dataset":
        return f"{config.HF_API_BASE}/datasets/{encoded}"
    return f"{config.HF_API_BASE}/models/{encoded}"


def parse_front_matter(readme: str) -> dict:
    if not isinstance(readme, str):
        return {}
    match = _FRONT_MATTER_RE.match(readme)
    if not match:
        return {}
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def description_from_front_matter(repo: str, front: dict) -> str:
    parts: list[str] = [repo]
    pipeline = front.get("pipeline_tag")
    if pipeline:
        parts.append(f"pipeline_tag={pipeline}")
    base_model = front.get("base_model")
    if isinstance(base_model, str):
        parts.append(f"base_model={base_model}")
    elif isinstance(base_model, list) and base_model:
        parts.append("base_model=" + ", ".join(str(b) for b in base_model[:3]))
    datasets = front.get("datasets")
    if isinstance(datasets, list) and datasets:
        parts.append("datasets=" + ", ".join(str(d) for d in datasets[:3]))
    return "; ".join(parts)


def _default_fetch_text(url: str) -> tuple[int, str | None, str | None]:
    try:
        with httpx.Client(timeout=config.LINK_TIMEOUT_S, follow_redirects=True) as client:
            response = client.get(url)
            return response.status_code, response.text, None
    except Exception as exc:  # noqa: BLE001
        return 0, None, f"{type(exc).__name__}: {exc}"


def _default_fetch_json(url: str) -> tuple[int, Any, str | None]:
    try:
        with httpx.Client(timeout=config.LINK_TIMEOUT_S, follow_redirects=True) as client:
            response = client.get(url)
            try:
                return response.status_code, response.json(), None
            except Exception:  # noqa: BLE001
                return response.status_code, None, None
    except Exception as exc:  # noqa: BLE001
        return 0, None, f"{type(exc).__name__}: {exc}"


def enrich_hf_link(
    link: dict,
    *,
    fetch_text: Callable[[str], tuple[int, str | None, str | None]] = _default_fetch_text,
    fetch_json: Callable[[str], tuple[int, Any, str | None]] = _default_fetch_json,
) -> dict:
    """Fetch README front-matter + API metadata for a single HF link.

    Returns:
      {
        "link": link,
        "ok": bool,
        "repo_url": str,
        "readme_url": str,
        "api_url": str,
        "metadata": {"front_matter": {...}},
        "card_data": {...},
        "description": "<repo>; <facets>",
        "error": str,
      }
    """
    repo = hf_repo_id(link)
    if not repo:
        return {"link": link, "ok": False, "error": "missing hf repo id"}
    repo_url = hf_repo_url(link)
    readme_url = hf_readme_url(link)
    api_url = hf_api_url(link)

    readme_status, readme, readme_err = fetch_text(readme_url)
    front = parse_front_matter(readme or "") if readme else {}

    api_status, api, api_err = fetch_json(api_url)
    card_data = api.get("cardData") if isinstance(api, dict) else {}
    if not isinstance(card_data, dict):
        card_data = {}

    description = normalize_space(description_from_front_matter(repo, front))

    ok = bool(readme is not None and readme_status and 200 <= readme_status < 400)
    return {
        "link": link,
        "ok": ok,
        "repo_url": repo_url,
        "readme_url": readme_url,
        "api_url": api_url,
        "metadata": {"front_matter": front},
        "card_data": card_data,
        "description": description,
        "error": readme_err or api_err or "",
    }
