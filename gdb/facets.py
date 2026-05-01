from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .artifacts import canonical_identity


VALID_KINDS = ("model", "dataset")


def flatten_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    canonical = canonical_identity(identity)
    out: dict[str, str] = {}
    for key, value in canonical.items():
        if key == "extra" and isinstance(value, dict):
            for extra_key, extra_value in value.items():
                out[f"extra.{extra_key}"] = str(extra_value)
        elif value is not None:
            out[key] = str(value)
    return out


def unflatten_identity(items: Mapping[str, str]) -> dict:
    out: dict[str, Any] = {}
    extra: dict[str, str] = {}
    for key, value in items.items():
        if key.startswith("extra."):
            extra[key.removeprefix("extra.")] = value
        else:
            out[key] = value
    if extra:
        out["extra"] = dict(sorted(extra.items()))
    return out


@dataclass(frozen=True)
class Facets:
    kind: str
    items: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}: {self.kind!r}")
        seen: set[str] = set()
        for key, _value in self.items:
            if key in seen:
                raise ValueError(f"duplicate facet key: {key!r}")
            seen.add(key)

    @classmethod
    def from_identity(cls, kind: str, identity: Mapping[str, Any]) -> "Facets":
        flat = flatten_identity(identity)
        return cls(kind=kind, items=frozenset((k, v) for k, v in flat.items() if v not in ("", None)))

    @classmethod
    def from_dict(cls, kind: str, values: Mapping[str, Any]) -> "Facets":
        return cls.from_identity(kind, values)

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(key for key, _value in self.items)

    def as_dict(self) -> dict[str, str]:
        return dict(self.items)

    def as_identity(self) -> dict:
        return unflatten_identity(self.as_dict())

    def signature(self) -> str:
        payload = [self.kind, sorted(self.items)]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def subsumes(parent: Facets, child: Facets) -> bool:
    return parent.kind == child.kind and parent.items <= child.items


def strictly_subsumes(parent: Facets, child: Facets) -> bool:
    return parent.kind == child.kind and parent.items < child.items


def cover_parents(child: Facets, candidates: Iterable[Facets]) -> list[Facets]:
    strict = [candidate for candidate in candidates if strictly_subsumes(candidate, child)]
    out: list[Facets] = []
    for parent in strict:
        if not any(
            other != parent and strictly_subsumes(parent, other) and strictly_subsumes(other, child)
            for other in strict
        ):
            out.append(parent)
    return out

