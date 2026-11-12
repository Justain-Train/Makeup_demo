"""MakeupBlueprint schema version compatibility. Same MAJOR with minor >= minimum is accepted."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from app.services.makeup_vocabulary.versions import parse_vocabulary_version

SUPPORTED_BLUEPRINT_VERSIONS: Mapping[int, int] = MappingProxyType({1: 0})  # MAJOR -> minimum MINOR


class UnsupportedBlueprintVersion(Exception):
    pass


def is_supported(version: str) -> bool:
    major, minor = parse_vocabulary_version(version)
    return major in SUPPORTED_BLUEPRINT_VERSIONS and minor >= SUPPORTED_BLUEPRINT_VERSIONS[major]


def check(blueprint_dict: Mapping[str, Any]) -> None:
    """Raise UnsupportedBlueprintVersion for an unsupported or invalid version; unknown fields are ignored."""
    if not isinstance(blueprint_dict, Mapping):
        raise UnsupportedBlueprintVersion("Blueprint must be a mapping")
    version = blueprint_dict.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedBlueprintVersion("Missing or invalid schema_version")
    try:
        supported = is_supported(version)
    except ValueError as exc:
        raise UnsupportedBlueprintVersion("Malformed schema_version") from exc
    if not supported:
        raise UnsupportedBlueprintVersion("Unsupported blueprint schema version")
