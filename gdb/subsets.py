"""Pre-audit Python pass — purely additive, never mutates the lattice.

Two phases:

1. **populate_subsets** — fetch each dataset node's HF README and fill
   in the `subsets[]` field with parsed config / component slugs.

2. **flag_audit_issues** — scan the augmented lattice for cases that
   likely warrant audit attention. Surfaces them as a top-level
   `audit_hints[]` array. Doesn't move items, doesn't restore drops,
   doesn't rename anything. The LLM auditor reads hints and decides
   what to do.

Design principle: Python finds patterns deterministically; the LLM
auditor exercises judgment. Python never deletes or rewrites what
organize produced.
"""
from __future__ import annotations

import re
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Subset parsing
# ---------------------------------------------------------------------------

_TABLE_ROW = re.compile(r"^\|\s*(?P<col1>[A-Za-z][\w\-+. /]+?)\s*\|")
_SUBSET_HEADING_KEYWORDS = (
    "component", "subset", "composition", "source", "mix",
    "ingredient", "data source", "constituent", "sub-corpus",
    "sub-corpora", "data mix", "contents",
)
_HEADER_CELLS = frozenset({
    "subset", "subsets", "name", "source", "component", "components",
    "ingredient", "mix", "tokens", "size", "count", "split", "split name",
    "config", "config name", "data source", "data sources", "dataset",
    "datasets", "language", "format", "stage", "domain", "topic", "type",
    "category", "n", "rows", "examples", "samples", "filter", "url",
    "license", "---", ":---:", ":---", "---:", "feature", "features",
    "field", "key", "constituent", "constituents", "section",
})

# Identity-key vocabulary that, when an item uses ONLY these keys,
# marks the item as a family-concept root (foundational concept that
# other items derive from).
_FAMILY_CONCEPT_KEYS = frozenset({
    "org", "collection", "vendor", "family", "language",
})


def fetch_card(formal_name: str, *, timeout: float = 10.0) -> str | None:
    url = f"https://huggingface.co/datasets/{formal_name}/raw/main/README.md"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "graph/0.1"})
    except requests.RequestException:
        return None
    return r.text if r.status_code == 200 else None


def _split_frontmatter(readme: str) -> tuple[dict | None, str]:
    m = re.match(r"---\n(.*?)\n---\n", readme, re.DOTALL)
    if not m:
        return None, readme
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = None
    return (fm if isinstance(fm, dict) else None), readme[m.end():]


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-+.]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def parse_subsets(readme: str) -> list[str]:
    raw_names: list[str] = []
    fm, body = _split_frontmatter(readme)

    if fm:
        configs = fm.get("configs")
        if isinstance(configs, list):
            for c in configs:
                if isinstance(c, dict) and isinstance(c.get("config_name"), str):
                    raw_names.append(c["config_name"])
                elif isinstance(c, str):
                    raw_names.append(c)

    sections = re.split(r"\n(#{1,6}\s+.+?)\n", "\n" + body)
    for i in range(1, len(sections), 2):
        heading_lower = sections[i].lower()
        if not any(kw in heading_lower for kw in _SUBSET_HEADING_KEYWORDS):
            continue
        section_body = sections[i + 1] if (i + 1) < len(sections) else ""
        for line in section_body.splitlines():
            m2 = _TABLE_ROW.match(line)
            if not m2:
                continue
            cell = m2.group("col1").strip()
            if cell.lower() in _HEADER_CELLS:
                continue
            if re.fullmatch(r"-+|:?-+:?", cell):
                continue
            raw_names.append(cell)

    seen, out = set(), []
    for raw in raw_names:
        slug = _slugify(raw)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def populate_subsets(lattice: dict, *, dry_run: bool = False) -> dict:
    """Walk the lattice; for each dataset node with empty `subsets`,
    fetch its HF README and populate `subsets[]`. Idempotent.

    Returns: {populated, failed, skipped, total_subsets}."""
    populated = failed = skipped = 0
    total_subsets = 0
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            if it.get("kind") != "dataset":
                continue
            it.setdefault("subsets", [])
            if it["subsets"]:
                skipped += 1
                continue
            card = fetch_card(it.get("formal_name") or "")
            if card is None:
                failed += 1
                continue
            subs = parse_subsets(card)
            if not dry_run:
                it["subsets"] = subs
            populated += 1
            total_subsets += len(subs)
    return {"populated": populated, "failed": failed, "skipped": skipped,
            "total_subsets": total_subsets}


# ---------------------------------------------------------------------------
# Audit-hint generation — purely additive, surfaces flags only
# ---------------------------------------------------------------------------


def _alnum_form(s: object) -> str:
    """Lowercase, strip everything that isn't [a-z0-9]. Used as a
    canonical fuzzy-match form to bridge `finemath-4plus` ↔
    `finemath4+` ↔ `FineMath4Plus`."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace("+", "plus")
    return re.sub(r"[^a-z0-9]", "", s)


def _name_variants(name: str) -> set[str]:
    """Return slug variants of a name for fuzzy matching against
    populated `subsets[]`. Generates the lowercase-kebab base form,
    common substitutions, and an alphanumeric-only canonical form."""
    s = (name or "").strip()
    slug = _slugify(s)
    out = {
        slug,
        slug.replace("-", ""),
        slug.replace("-", "_"),
        slug.replace("+", "plus"),
        slug.replace("+", "-plus"),
        _alnum_form(s),
    }
    out.add(slug.replace("-", "").replace("+", "plus"))
    out.add(slug.replace("-", "_").replace("+", "plus"))
    out.discard("")
    return out


def _is_bare_domain_url(url: object) -> bool:
    from urllib.parse import urlparse
    if not isinstance(url, str):
        return False
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if not p.scheme or not p.netloc:
        return False
    return not p.path.strip("/")


def _build_subset_index(lattice: dict) -> dict[str, list[tuple[dict, dict]]]:
    """Index: subset_slug → [(group, parent_item)] across all kept items.
    Both the literal slug and its alphanumeric-only canonical form are
    indexed, so a lookup using either form finds the parent."""
    index: dict[str, list[tuple[dict, dict]]] = {}
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            for sub in it.get("subsets") or []:
                if not isinstance(sub, str):
                    continue
                index.setdefault(sub, []).append((grp, it))
                alnum = _alnum_form(sub)
                if alnum and alnum != sub:
                    index.setdefault(alnum, []).append((grp, it))
    return index


def _classify_item_role(it: dict) -> str:
    """Return a one-word descriptor of the item's role for hint context.
    'concept' = family-concept root (broad-only identity);
    'leaf' = specific artifact with narrow identity;
    'soft-anchored' = primary link is paper / blog / vendor_docs;
    'canonical' = primary link is HF / GitHub repo;
    'unanchored' = no links."""
    links = it.get("links") or []
    primary_kind = (links[0].get("kind") if links and isinstance(links[0], dict) else None)
    identity = it.get("identity") or {}
    ident_keys = set(identity.keys()) if isinstance(identity, dict) else set()
    primary_url = links[0].get("url") if links and isinstance(links[0], dict) else None

    if not primary_kind:
        return "unanchored"
    if ident_keys and ident_keys.issubset(_FAMILY_CONCEPT_KEYS):
        return "concept"
    if _is_bare_domain_url(primary_url):
        return "concept"
    if primary_kind in ("hf_model", "hf_dataset", "hf_collection", "github"):
        return "canonical"
    if primary_kind in ("paper", "blog", "vendor_docs"):
        return "soft-anchored"
    return "other"


def flag_audit_issues(lattice: dict) -> dict:
    """Scan the lattice and surface cases that likely warrant audit
    attention. Adds entries to a top-level `audit_hints[]` array.
    Mutates nothing else — items, families, dropped[], subsets[] all
    pass through untouched.

    Hint kinds emitted:

    - `item_matches_parent_subset`: an existing item's name slug
      appears in some other item's `subsets[]`. Audit decides
      whether to reshape under parent (typical for soft-anchored
      sub-components) or keep standalone (typical for foundational
      concepts that are also referenced as subsets).

    - `dropped_matches_parent_subset`: a dropped name's slug
      appears in some kept item's `subsets[]`. Audit decides
      whether to restore as `<parent>/<slug>` child.

    - `sibling_identity_collision`: two items in the same family
      carry identical identity dicts. Audit must add a discriminating
      facet or merge.

    - `cross_org_family`: a family contains items spanning multiple
      `identity.org` / `identity.vendor` values. Audit decides whether
      this is a substring false positive (split) or a legitimate
      product-line grouping (keep, optionally add a facet).

    - `formal_name_vs_canonical_url_mismatch`: the formal_name
      doesn't match the canonical path inside the primary HF URL
      (e.g., formal_name='MMLU' but URL is huggingface.co/datasets/cais/mmlu).
      Audit decides whether to rename to canonical form.

    - `phantom_item`: an item with `aliases: []` whose identity
      isn't a family-concept root (already a validator error, but
      worth surfacing for context).

    Returns counters by hint kind.
    """
    hints: list[dict] = []
    counts: dict[str, int] = {}

    def add(kind: str, **payload: Any) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        hints.append({"kind": kind, **payload})

    subset_index = _build_subset_index(lattice)

    # Hint 1 + 5 + 6: per-item scans
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            fn = it.get("formal_name") or ""
            family = grp.get("family")

            # Phantom items
            aliases = it.get("aliases") or []
            identity = it.get("identity") or {}
            ident_keys = set(identity.keys()) if isinstance(identity, dict) else set()
            is_concept_root = bool(ident_keys) and ident_keys.issubset(_FAMILY_CONCEPT_KEYS)
            if not aliases and not is_concept_root:
                add("phantom_item",
                    item_formal_name=fn, item_family=family,
                    rationale="No aliases trace to input pile and identity isn't a family-concept root.")

            # formal_name vs canonical-URL mismatch
            links = it.get("links") or []
            if links and isinstance(links[0], dict):
                primary_kind = links[0].get("kind")
                primary_url = links[0].get("url") or ""
                if primary_kind in ("hf_model", "hf_dataset"):
                    m = re.search(r"huggingface\.co/(?:datasets/)?([^/]+/[^/?#]+)", primary_url)
                    if m:
                        canonical = m.group(1)
                        if canonical and canonical.lower() != fn.lower():
                            add("formal_name_vs_canonical_url_mismatch",
                                item_formal_name=fn, item_family=family,
                                canonical_from_url=canonical, primary_url=primary_url,
                                rationale="formal_name doesn't match the canonical HF path in primary URL.")

            # item slug matches some OTHER item's subsets[]
            slugs: set[str] = set()
            for n in [fn] + (aliases or []):
                slugs.update(_name_variants(n or ""))
            slugs.discard("")
            seen_parents: set[str] = set()
            for slug in slugs:
                for parent_grp, parent in subset_index.get(slug, []):
                    if parent is it:
                        continue
                    parent_fn = parent.get("formal_name") or ""
                    if parent_fn in seen_parents:
                        continue
                    seen_parents.add(parent_fn)
                    add("item_matches_parent_subset",
                        item_formal_name=fn,
                        item_family=family,
                        item_role=_classify_item_role(it),
                        matched_parent=parent_fn,
                        matched_subset_slug=slug,
                        rationale=(
                            "Item's name appears in parent's subsets[]. "
                            "If item is soft-anchored (paper/blog) and lacks "
                            "its own canonical release, consider reshaping under "
                            "parent as `<parent>/<slug>` child item. If item is "
                            "a foundational concept other things derive from "
                            "(role='concept'), keep standalone — it's referenced "
                            "as a subset incidentally."
                        ))

    # Hint 2: dropped name slug matches some kept item's subsets[]
    for d in lattice.get("dropped") or []:
        if not isinstance(d, dict):
            continue
        name = d.get("name") or ""
        if not name:
            continue
        slugs = _name_variants(name)
        seen_parents = set()
        for slug in slugs:
            for parent_grp, parent in subset_index.get(slug, []):
                parent_fn = parent.get("formal_name") or ""
                if parent_fn in seen_parents:
                    continue
                seen_parents.add(parent_fn)
                add("dropped_matches_parent_subset",
                    dropped_name=name,
                    dropped_kind=d.get("kind"),
                    matched_parent=parent_fn,
                    matched_subset_slug=slug,
                    rationale=(
                        "Dropped name appears in parent's subsets[]. "
                        "Consider restoring as `<parent>/<slug>` child item "
                        "with identity inheriting parent + `subset: <slug>`."
                    ))

    # Hint 3: sibling identity collisions
    from collections import defaultdict
    for grp in lattice.get("groups") or []:
        by_id: dict[str, list[str]] = defaultdict(list)
        for it in grp.get("items") or []:
            key = repr(sorted((it.get("identity") or {}).items()))
            by_id[key].append(it.get("formal_name") or "")
        for key, fns in by_id.items():
            if len(fns) > 1:
                add("sibling_identity_collision",
                    family=grp.get("family"),
                    identity_key=key,
                    items=fns,
                    rationale=(
                        "Siblings carry identical identity dicts; lattice "
                        "can't tell them apart. Add a discriminating facet "
                        "or merge the items."
                    ))

    # Hint 4: cross-org families
    for grp in lattice.get("groups") or []:
        orgs: set[str] = set()
        items = grp.get("items") or []
        if len(items) < 2:
            continue
        for it in items:
            ident = it.get("identity") or {}
            org = ident.get("org") or ident.get("vendor")
            if org:
                orgs.add(org)
        if len(orgs) > 1:
            add("cross_org_family",
                family=grp.get("family"),
                orgs=sorted(orgs),
                item_count=len(items),
                rationale=(
                    "Family contains items from multiple orgs / vendors. "
                    "Audit's call: split into per-org families if substring "
                    "false positive, or keep merged if items share a real "
                    "product line (e.g., a base + its strict-superset extension)."
                ))

    # Hint 5: items with `@branch` in formal_name (HF git-revspec syntax)
    # These name a specific revision within the same canonical repo —
    # different branches are different checkpoints in the training history,
    # but the canonical repo identifier is the part before `@`. Audit
    # should collapse all `@branch` variants into one item under the
    # canonical formal_name, with branch names carried in aliases.
    for grp in lattice.get("groups") or []:
        for it in grp.get("items") or []:
            fn = it.get("formal_name") or ""
            if "@" in fn:
                base, _, branch = fn.partition("@")
                add("branch_variant_in_formal_name",
                    item_formal_name=fn,
                    item_family=grp.get("family"),
                    canonical_repo=base,
                    branch=branch,
                    rationale=(
                        "`@branch` in formal_name is HF git-revspec syntax "
                        "for a revision within the canonical repo. Different "
                        "branches are different checkpoints, but the artifact "
                        "identifier is the repo (the part before `@`). Collapse "
                        "all `@branch` siblings of this base into one item with "
                        f"formal_name='{base}'; carry branch names in aliases."
                    ))

    # Attach hints to lattice (additive only)
    if hints:
        lattice.setdefault("audit_hints", []).extend(hints)
    return counts


def populate_then_flag(lattice: dict, *, dry_run: bool = False) -> dict:
    """Two-phase Python pre-pass for audit:

    1. populate_subsets — fill in subsets[] on every dataset node.
    2. flag_audit_issues — surface suspicious cases as audit_hints[].

    BOTH PHASES ARE PURELY ADDITIVE. Items, families, dropped[]
    pass through untouched. The auditor reads the augmented lattice
    plus the hints and decides what to do.
    """
    pop = populate_subsets(lattice, dry_run=dry_run)
    if dry_run:
        return {"populate": pop, "flag": {}}
    flag = flag_audit_issues(lattice)
    return {"populate": pop, "flag": flag}
