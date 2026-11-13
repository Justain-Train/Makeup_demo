from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from app.services.makeup_vocabulary.versions import parse_vocabulary_version

PLAN_SCHEMA_VERSION = "1.0"

PLAN_VERSION_PARSED: tuple[int, int] = parse_vocabulary_version(PLAN_SCHEMA_VERSION)

SUPPORTED_PLAN_VERSIONS: Mapping[int, int] = MappingProxyType({1: 0})  # MAJOR -> minimum MINOR


class UnsupportedPlanVersion(Exception):
    pass


def is_supported(version: str) -> bool:
    major, minor = parse_vocabulary_version(version)
    return major in SUPPORTED_PLAN_VERSIONS and minor >= SUPPORTED_PLAN_VERSIONS[major]


def check(plan_dict: Mapping[str, Any]) -> None:
    """Raise UnsupportedPlanVersion for an unsupported or invalid version; unknown fields are ignored."""
    if not isinstance(plan_dict, Mapping):
        raise UnsupportedPlanVersion("Plan must be a mapping")
    version = plan_dict.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedPlanVersion("Missing or invalid schema_version")
    try:
        supported = is_supported(version)
    except ValueError as exc:
        raise UnsupportedPlanVersion("Malformed schema_version") from exc
    if not supported:
        raise UnsupportedPlanVersion("Unsupported plan schema version")
