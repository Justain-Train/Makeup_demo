"""Pure vocabulary resolution and registry validation."""

import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .registry import CATEGORIES, KINDS, REGISTRY, ProductType

_MAX_TEXT_LEN = 64
_SEP_RE = re.compile(r"[-_\s]+")


class UnknownTermError(ValueError):
    pass


def normalize(text: str) -> str:
    if not isinstance(text, str) or len(text) > _MAX_TEXT_LEN:
        raise ValueError("Invalid term text")
    folded = unicodedata.normalize("NFKC", text).casefold().replace("'", "").replace("’", "")
    if len(folded) > _MAX_TEXT_LEN:
        raise ValueError("Invalid term text")
    return _SEP_RE.sub("_", folded.strip())


def _build_index(data: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, str]]:
    index: dict[str, Mapping[str, str]] = {}
    for kind, terms in data.items():
        flat: dict[str, str] = {}
        for key, term in terms.items():
            flat[normalize(key)] = key
            for alias in term.aliases:
                flat[normalize(alias)] = key
        index[kind] = MappingProxyType(flat)
    return MappingProxyType(index)


_INDEX = _build_index(REGISTRY)


def resolve_term(kind: str, text: str) -> str | None:
    if not isinstance(kind, str) or kind not in _INDEX:
        raise ValueError("Unknown vocabulary kind")
    return _INDEX[kind].get(normalize(text))


def require_term(kind: str, text: str) -> str:
    resolved = resolve_term(kind, text)
    if resolved is None:
        raise UnknownTermError(f"Unknown {kind} term")
    return resolved


def product_types_for_category(category: str) -> tuple[str, ...]:
    return tuple(k for k, t in REGISTRY["product_types"].items() if t.category == category)


def stage_layer_for(product_type: str) -> tuple[str | None, str | None]:
    term = REGISTRY["product_types"][require_term("product_types", product_type)]
    assert isinstance(term, ProductType)
    return term.stage_key, term.layer_key


def validate_registry(data: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    data = REGISTRY if data is None else data
    for kind in KINDS:
        if kind not in data:
            raise ValueError(f"Missing kind {kind!r}")
        seen: dict[str, str] = {}
        for key, term in data[kind].items():
            for name in (key, *term.aliases):
                norm = normalize(name)
                if norm in seen:
                    raise ValueError(f"Duplicate term {norm!r} in {kind}")
                seen[norm] = key
    for key, pt in data["product_types"].items():
        if pt.category not in CATEGORIES:
            raise ValueError(f"Unknown category for product type {key!r}")
        if (pt.stage_key is None) != (pt.layer_key is None):
            raise ValueError(f"Inconsistent stage/layer for {key!r}")
        if (pt.stage_key is None) != (pt.category == "special_effects"):
            raise ValueError(f"Only special_effects may lack a stage/layer: {key!r}")
    for kind in ("placements", "techniques"):
        for key, term in data[kind].items():
            for cat in term.categories:
                if cat not in CATEGORIES:
                    raise ValueError(f"Unknown category {cat!r} for {key!r}")


validate_registry()
