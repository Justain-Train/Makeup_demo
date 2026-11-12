from app.services.makeup_vocabulary.versions import parse_vocabulary_version

BLUEPRINT_SCHEMA_VERSION = "1.0"

BLUEPRINT_VERSION_PARSED: tuple[int, int] = parse_vocabulary_version(BLUEPRINT_SCHEMA_VERSION)
