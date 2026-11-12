"""Makeup vocabulary version. Same format as the FacialProfile schema version."""

import re

VOCABULARY_VERSION = "1.1"
COMPONENT_SCHEMA_VERSION = "1.0"

_VERSION_RE = re.compile(r"^([0-9]{1,4})\.([0-9]{1,4})$")
_MAX_VERSION_LEN = 16


class InvalidVocabularyVersion(ValueError):
    pass


def parse_vocabulary_version(s: str) -> tuple[int, int]:
    if not isinstance(s, str) or len(s) > _MAX_VERSION_LEN or not s.isascii():
        raise InvalidVocabularyVersion("Invalid vocabulary version; expected 'MAJOR.MINOR'")
    match = _VERSION_RE.fullmatch(s)
    if match is None:
        raise InvalidVocabularyVersion("Invalid vocabulary version; expected 'MAJOR.MINOR'")
    return int(match.group(1)), int(match.group(2))
