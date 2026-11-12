"""MakeupComponent schema version compatibility. Same MAJOR with minor >= minimum is accepted."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .versions import parse_vocabulary_version

SUPPORTED_COMPONENT_VERSIONS: Mapping[int, int] = MappingProxyType({1: 0})  # MAJOR -> minimum MINOR


class UnsupportedComponentVersion(Exception):
    pass


def is_supported(version: str) -> bool:
    major, minor = parse_vocabulary_version(version)
    return major in SUPPORTED_COMPONENT_VERSIONS and minor >= SUPPORTED_COMPONENT_VERSIONS[major]


def check(component_dict: Mapping[str, Any]) -> None:
    """Raise UnsupportedComponentVersion for an unsupported or invalid version; unknown fields are ignored."""
    if not isinstance(component_dict, Mapping):
        raise UnsupportedComponentVersion("Component must be a mapping")
    version = component_dict.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedComponentVersion("Missing or invalid schema_version")
    try:
        supported = is_supported(version)
    except ValueError as exc:
        raise UnsupportedComponentVersion("Malformed schema_version") from exc
    if not supported:
        raise UnsupportedComponentVersion("Unsupported component schema version")
