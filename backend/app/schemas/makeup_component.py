"""MakeupComponent: one versioned makeup step described in the controlled vocabulary.

Phase 7 sees components only partially, so everything beyond category and product_type is
optional. color_lab is derived from color_hex and never taken from the caller.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.application_zones import catalog
from app.services.makeup_vocabulary import (
    CATEGORIES,
    COMPONENT_SCHEMA_VERSION,
    REGISTRY,
    hex_to_lab,
    normalize,
    normalize_hex,
    require_term,
)
from app.services.makeup_vocabulary.compat import is_supported

_MAX_ZONES = 128
_MAX_ZONE_ID_LEN = 64


# Compatibility policy: models keep extra="ignore", so unknown fields from newer
# minor versions are ignored on load.
class MakeupComponent(BaseModel):
    """A single makeup step.

    placement is the semantic intent (e.g. "lifted_cheek"); target_zones is the explicit or
    resolved list of application zone ids. Batch 3 fills target_zones from placement.
    Batch 2 performs no cross-check between them.

    When logging or returning validation errors, use `str(exc)` or `exc.errors(include_input=False)`;
    plain `errors()` includes raw input.
    """

    # Phase 7 feeds untrusted VLM text in and callers may log validation errors, so inputs
    # must not be echoed in error output. This only cleans str(exc); exc.errors() still
    # carries the raw input, so callers must use exc.errors(include_input=False).
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    schema_version: str = COMPONENT_SCHEMA_VERSION
    category: str
    product_type: str
    color_hex: str | None = None
    color_lab: tuple[float, float, float] | None = None
    undertone: Literal["warm", "cool", "neutral", "olive"] | None = None
    finish: str | None = None
    placement: str | None = None
    technique: str | None = None
    direction: str | None = None
    intensity: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    shade_name: str | None = Field(default=None, max_length=64)
    placement_note: str | None = Field(default=None, max_length=256)
    target_zones: list[str] = Field(default_factory=list, max_length=_MAX_ZONES)

    @model_validator(mode="before")
    @classmethod
    def _drop_color_lab(cls, data: Any) -> Any:
        if isinstance(data, dict) and "color_lab" in data:
            return {k: v for k, v in data.items() if k != "color_lab"}
        return data

    @field_validator("schema_version")
    @classmethod
    def _version_supported(cls, v: str) -> str:
        if not is_supported(v):
            raise ValueError("Unsupported component schema version")
        return v

    @field_validator("category")
    @classmethod
    def _category(cls, v: str) -> str:
        norm = normalize(v)
        if norm not in CATEGORIES:
            raise ValueError("Unknown category")
        return norm

    @field_validator("product_type")
    @classmethod
    def _product_type(cls, v: str) -> str:
        return require_term("product_types", v)

    @field_validator("color_hex")
    @classmethod
    def _color_hex(cls, v: str | None) -> str | None:
        return None if v is None else normalize_hex(v)

    @field_validator("finish")
    @classmethod
    def _finish(cls, v: str | None) -> str | None:
        return None if v is None else require_term("finishes", v)

    @field_validator("placement")
    @classmethod
    def _placement(cls, v: str | None) -> str | None:
        return None if v is None else require_term("placements", v)

    @field_validator("technique")
    @classmethod
    def _technique(cls, v: str | None) -> str | None:
        return None if v is None else require_term("techniques", v)

    @field_validator("direction")
    @classmethod
    def _direction(cls, v: str | None) -> str | None:
        return None if v is None else require_term("directions", v)

    @field_validator("intensity", "confidence")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("Value must be finite")
        return v

    @field_validator("target_zones")
    @classmethod
    def _zones(cls, v: list[str]) -> list[str]:
        known = catalog.ZONE_SPEC_BY_ID
        seen: set[str] = set()
        for zone in v:
            if len(zone) > _MAX_ZONE_ID_LEN or zone not in known:
                raise ValueError("Unknown zone id")
            if zone in seen:
                raise ValueError("Duplicate zone id")
            seen.add(zone)
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "MakeupComponent":
        if REGISTRY["product_types"][self.product_type].category != self.category:
            raise ValueError("product_type does not belong to category")
        for kind, value in (("techniques", self.technique), ("placements", self.placement)):
            if value is not None:
                cats = REGISTRY[kind][value].categories  # type: ignore[attr-defined]
                if cats and self.category not in cats:
                    raise ValueError(f"{kind[:-1]} is not valid for category")
        self.color_lab = None if self.color_hex is None else hex_to_lab(self.color_hex)
        return self
