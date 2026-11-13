from collections.abc import Mapping
from types import MappingProxyType

PROFILE_SCHEMA_VERSION = "1.1"

COMPONENT_VERSIONS: Mapping[str, int] = MappingProxyType(
    {"landmarks": 1, "frame": 1, "parsing": 1, "zones": 1, "features": 1}
)

ZONES_ALGORITHM_VERSION = "phase4-70zones-1"
FRAME_ALGORITHM_VERSION = "iris-frame-1"
FEATURES_ALGORITHM_VERSION = "features-1"
