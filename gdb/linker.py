from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import httpx

from . import config
from .artifacts import aggregate_mentions, normalize_link_candidate, normalize_mention


@dataclass(frozen=True)
class LinkCandidate:
    cluster_key: str
    kind: str
    link_kind: str
    link_value: str
    url: str


def candidate_url(kind: str, link_kind: str, value: str) -> str:
    if link_kind == "hf_model":
        return f"https://huggingface.co/{value}"
    if link_kind == "hf_dataset":
        return f"https://huggingface.co/datasets/{value}"
    if link_kind == "hf_dataset_config":
        repo = value.split("::", 1)[0]
        return f"https://huggingface.co/datasets/{repo}"
    if link_kind == "github_ref":
        repo = value.split("@", 1)[0].split(":", 1)[0]
        ref_value = ""
        path_value = ""
        if "@" in value:
            ref_and_path = value.split("@", 1)[1]
            if ":" in ref_and_path:
                ref_value, path_value = ref_and_path.split(":", 1)
            else:
                ref_value = ref_and_path
        elif ":" in value.split("/", 1)[-1]:
            path_value = value.split(":", 1)[1]
        if ref_value and path_value:
            return f"https://github.com/{repo}/blob/{ref_value}/{path_value}"
        if ref_value:
            return f"https://github.com/{repo}/tree/{ref_value}"
        if path_value:
            return f"https://github.com/{repo}/blob/HEAD/{path_value}"
        return f"https://github.com/{repo}"
    if link_kind == "github_repo":
        return f"https://github.com/{value}"
    if link_kind == "paper_release":
        return value
    if link_kind == "official_release_url":
        return value
    if link_kind == "api_model_id":
        return ""
    raise ValueError(f"unknown link kind: {link_kind}")


def _iter_cluster_links(cluster: dict) -> Iterable[dict]:
    """Yield typed links from the cluster body AND from each alias.

    Alias-local links are real public release identifiers (e.g.,
    `Org/Qwen3-7B-Instruct-FP8` for a quantized variant). They need
    HEAD-checking just like the canonical cluster link.
    """
    for link in cluster.get("links") or []:
        yield link
    for alias in cluster.get("aliases") or []:
        if not isinstance(alias, dict):
            continue
        for link in alias.get("links") or []:
            yield link


def link_candidates_from_clusters(clusters: Iterable[dict]) -> list[LinkCandidate]:
    """Emit one verification candidate per (link_kind, link_value).

    Dedup is global across clusters: a github_repo URL that appears in
    two different clusters is HEADed once. The cluster_key on the
    emitted candidate is the FIRST cluster's, which is fine — link
    verification is a function of the URL, not the cluster_key.
    """
    out: list[LinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for cluster in clusters:
        kind = cluster["kind"]
        cluster_key = cluster["cluster_key"]
        for raw_link in _iter_cluster_links(cluster):
            link = normalize_link_candidate(raw_link, kind=kind)
            if not link:
                continue
            link_kind = link["type"]
            value = link["value"]
            if link_kind == "api_model_id" or not link.get("exact"):
                continue
            if (link_kind, value) in seen:
                continue
            seen.add((link_kind, value))
            url = link.get("url") or candidate_url(kind, link_kind, value)
            if not url:
                continue
            out.append(LinkCandidate(cluster_key, kind, link_kind, value, url))
    return out


def link_candidates_from_mentions(mentions: Iterable[dict]) -> list[LinkCandidate]:
    """Build verification candidates from mentions.

    Dedup is keyed on `(link_kind, link_value)` only — verification is
    a function of the URL alone. Cluster-aggregate candidates are
    emitted first (they include alias-local links via
    _iter_cluster_links) so the cluster_key field reflects the deduped
    cluster context.
    """
    candidates = link_candidates_from_clusters(aggregate_mentions(mentions))
    seen: set[tuple[str, str]] = {(c.link_kind, c.link_value) for c in candidates}
    for raw in mentions:
        mention = normalize_mention(raw)
        cluster_key = mention["identity_key"] or mention["surface_key"]
        for link in mention.get("links") or []:
            if not link.get("exact") or link.get("type") == "api_model_id":
                continue
            link_kind = link["type"]
            value = link["value"]
            if (link_kind, value) in seen:
                continue
            url = link.get("url") or candidate_url(mention["kind"], link_kind, value)
            if not url:
                continue
            candidates.append(LinkCandidate(cluster_key, mention["kind"], link_kind, value, url))
            seen.add((link_kind, value))
    return candidates


@contextmanager
def _shared_client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=config.LINK_TIMEOUT_S, follow_redirects=True) as client:
        yield client


def _fetch_with(client: httpx.Client, url: str) -> tuple[bool, int | None, str | None]:
    try:
        response = client.head(url)
        if response.status_code in {405, 403} or response.status_code >= 500:
            response = client.get(url)
        return 200 <= response.status_code < 400, response.status_code, None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def default_fetch(url: str) -> tuple[bool, int | None, str | None]:
    with _shared_client() as client:
        return _fetch_with(client, url)


def verify_candidates(
    candidates: Iterable[LinkCandidate],
    *,
    fetch: Callable[[str], tuple[bool, int | None, str | None]] | None = None,
) -> list[dict]:
    """Verify each candidate URL once.

    When `fetch` is None we open ONE shared `httpx.Client` for the
    whole pass so connection pooling kicks in. Tests inject a callable
    instead.
    """
    candidate_list = list(candidates)
    if fetch is None:
        with _shared_client() as client:
            return _verify(candidate_list, lambda url: _fetch_with(client, url))
    return _verify(candidate_list, fetch)


def _verify(
    candidates: list[LinkCandidate],
    fetch: Callable[[str], tuple[bool, int | None, str | None]],
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
