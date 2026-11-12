"""Pure placement-to-zones mapping. No logging; error messages never echo input."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones import catalog
from app.services.makeup_vocabulary import require_term, stage_layer_for

from .placement_zones import PLACEMENT_ZONES

_MAX_ZONES = 128
_MAX_ZONE_ID_LEN = 64

REASON_LAYER_ZONES = "layer_zones"
REASON_PLACEMENT_MATCH = "placement_match"
REASON_EXPLICIT_ZONES = "explicit_zones"
REASON_PLACEMENT_NOT_IN_LAYER = "placement_not_in_layer"
REASON_LAYER_HAS_NO_ZONES = "layer_has_no_zones"
REASON_SPECIAL_EFFECT_NOTE_ONLY = "special_effect_note_only"


class ZoneOutsideLayerError(ValueError):
    pass


@dataclass(frozen=True)
class ZoneMapping:
    zone_ids: tuple[str, ...]
    stage_key: str | None
    layer_key: str | None
    source: Literal["layer", "placement", "explicit", "none"]
    reason: str


class CatalogDriftError(RuntimeError):
    pass


def _validate_explicit(target_zones: Sequence[str]) -> frozenset[str]:
    if not isinstance(target_zones, (list, tuple)) or len(target_zones) > _MAX_ZONES:
        raise ValueError("Invalid target zones")
    for zone in target_zones:
        if not isinstance(zone, str) or len(zone) > _MAX_ZONE_ID_LEN:
            raise ValueError("Invalid target zones")
    return frozenset(target_zones)


def map_zones(
    product_type: str,
    placement: str | None = None,
    target_zones: Sequence[str] = (),
) -> ZoneMapping:
    stage_key, layer_key = stage_layer_for(product_type)
    placement_key = None if placement is None else require_term("placements", placement)
    explicit = _validate_explicit(target_zones)

    if stage_key is None or layer_key is None:
        if not explicit:
            return ZoneMapping((), None, None, "none", REASON_SPECIAL_EFFECT_NOTE_ONLY)
        if not explicit <= catalog.ZONE_SPEC_BY_ID.keys():
            raise ZoneOutsideLayerError("Zone is not a known application zone")
        ordered = tuple(z for z in catalog.ZONE_SPEC_BY_ID if z in explicit)
        return ZoneMapping(ordered, None, None, "explicit", REASON_EXPLICIT_ZONES)

    stage_spec = next((s for s in catalog.STAGES if s.key == stage_key), None)
    if stage_spec is None or layer_key not in stage_spec.layers:
        raise CatalogDriftError("Product layer is missing from the zone catalog")
    layer = stage_spec.layers[layer_key]

    if explicit:
        if not explicit <= catalog.ZONE_SPEC_BY_ID.keys() or not explicit <= set(layer):
            raise ZoneOutsideLayerError("Zone is not in the product layer")
        ordered = tuple(dict.fromkeys(z for z in layer if z in explicit))
        return ZoneMapping(ordered, stage_key, layer_key, "explicit", REASON_EXPLICIT_ZONES)

    if not layer:
        return ZoneMapping((), stage_key, layer_key, "none", REASON_LAYER_HAS_NO_ZONES)

    if placement_key is None:
        return ZoneMapping(
            tuple(dict.fromkeys(layer)), stage_key, layer_key, "layer", REASON_LAYER_ZONES
        )

    candidates = set(PLACEMENT_ZONES[placement_key])
    matched = tuple(dict.fromkeys(z for z in layer if z in candidates))
    if matched:
        return ZoneMapping(matched, stage_key, layer_key, "placement", REASON_PLACEMENT_MATCH)
    return ZoneMapping((), stage_key, layer_key, "none", REASON_PLACEMENT_NOT_IN_LAYER)


def map_component(c: MakeupComponent) -> ZoneMapping:
    return map_zones(c.product_type, c.placement, c.target_zones)
