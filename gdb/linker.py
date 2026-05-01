from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import httpx

from . import config
from .artifacts import LINK_FIELDS, aggregate_mentions


@dataclass(frozen=True)
class LinkCandidate:
    cluster_key: str
    kind: str
    link_kind: str
    link_value: str
    url: str


def candidate_url(kind: str, link_kind: str, value: str) -> str:
    if link_kind == "hf_ids":
        if kind == "dataset":
            return f"https://huggingface.co/datasets/{value}"
        return f"https://huggingface.co/{value}"
    if link_kind == "github_repos":
        return f"https://github.com/{value}"
    if link_kind == "papers":
        return value
    if link_kind == "official_urls":
        return value
    raise ValueError(f"unknown link kind: {link_kind}")


def link_candidates_from_clusters(clusters: Iterable[dict]) -> list[LinkCandidate]:
    out: list[LinkCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for cluster in clusters:
        kind = cluster["kind"]
        cluster_key = cluster["cluster_key"]
        for link_kind in LINK_FIELDS:
            for value in cluster.get("links", {}).get(link_kind) or []:
                key = (cluster_key, link_kind, value)
                if key in seen:
                    continue
                seen.add(key)
                out.append(LinkCandidate(cluster_key, kind, link_kind, value, candidate_url(kind, link_kind, value)))
    return out


def link_candidates_from_mentions(mentions: Iterable[dict]) -> list[LinkCandidate]:
    return link_candidates_from_clusters(aggregate_mentions(mentions))


def default_fetch(url: str) -> tuple[bool, int | None, str | None]:
    try:
        with httpx.Client(timeout=config.LINK_TIMEOUT_S, follow_redirects=True) as client:
            response = client.head(url)
            if response.status_code in {405, 403} or response.status_code >= 500:
                response = client.get(url)
            return 200 <= response.status_code < 400, response.status_code, None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def verify_candidates(
    candidates: Iterable[LinkCandidate],
    *,
    fetch: Callable[[str], tuple[bool, int | None, str | None]] = default_fetch,
) -> list[dict]:
    checks: list[dict] = []
    for candidate in candidates:
        ok, status_code, error = fetch(candidate.url)
        checks.append({
            "cluster_key": candidate.cluster_key,
            "kind": candidate.kind,
            "link_kind": candidate.link_kind,
            "link_value": candidate.link_value,
            "url": candidate.url,
            "ok": bool(ok),
            "status_code": status_code,
            "error": error,
        })
    return checks

