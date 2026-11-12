"""FacialProfile schema version compatibility. Same MAJOR with minor >= minimum is accepted."""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

SUPPORTED: Mapping[int, int] = MappingProxyType({1: 0})  # MAJOR -> minimum supported MINOR

_VERSION_RE = re.compile(r"^([0-9]{1,4})\.([0-9]{1,4})$")
_MAX_VERSION_LEN = 16


class UnsupportedProfileVersion(Exception):
    pass


def parse_version(s: str) -> tuple[int, int]:
    if not isinstance(s, str) or len(s) > _MAX_VERSION_LEN:
        raise ValueError("Invalid schema version; expected 'MAJOR.MINOR'")
    match = _VERSION_RE.fullmatch(s)
    if match is None:
        raise ValueError(f"Invalid schema version {s!r}; expected 'MAJOR.MINOR'")
    return int(match.group(1)), int(match.group(2))


def is_supported(version: str) -> bool:
    major, minor = parse_version(version)
    return major in SUPPORTED and minor >= SUPPORTED[major]


def check(profile_dict: Mapping[str, Any]) -> None:
    """Raise UnsupportedProfileVersion for an unsupported or invalid version; unknown fields are ignored."""
    if not isinstance(profile_dict, Mapping):
        raise UnsupportedProfileVersion("Profile must be a mapping")
    version = profile_dict.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedProfileVersion("Missing or invalid schema_version")
    try:
        supported = is_supported(version)
    except ValueError as exc:
        raise UnsupportedProfileVersion("Malformed schema_version") from exc
    if not supported:
        raise UnsupportedProfileVersion(f"Unsupported profile schema version {version!r}")
