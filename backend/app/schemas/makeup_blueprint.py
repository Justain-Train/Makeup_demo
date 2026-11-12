"""MakeupBlueprint: versioned, self-describing description of a reference makeup look.

Phase 7 output consumed downstream; sources record whether each field came from CV or VLM.
"""

import math
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.facial_profile import ImageSize
from app.schemas.makeup_component import MakeupComponent
from app.services.makeup_blueprint.compat import is_supported
from app.services.makeup_blueprint.versions import BLUEPRINT_SCHEMA_VERSION

_TEXT = 512
_MAX_SOURCES = 16

SourceKind = Literal["cv", "vlm"]

ATTRIBUTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "category",
        "product_type",
        "color_hex",
        "undertone",
        "intensity",
        "target_zones",
        "finish",
        "placement",
        "technique",
        "direction",
    }
)
_CV_ONLY_FIELDS: frozenset[str] = frozenset({"color_hex", "undertone", "intensity", "target_zones"})

STYLE_LABELS: frozenset[str] = frozenset(
    {
        "natural",
        "glam",
        "editorial",
        "bold",
        "soft",
        "dramatic",
        "minimal",
        "glass_skin",
        "smoky",
        "vintage",
    }
)


TimingKey = Annotated[str, Field(max_length=64, pattern=r"^[a-z0-9_]+$")]
TimingValue = Annotated[float, Field(ge=0, le=1e7, allow_inf_nan=False)]


class ReferenceProviderInfo(BaseModel):
    """Server-authored only; never populate from VLM output."""

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    id: str = Field(pattern=r"^[a-z0-9_.-]{1,64}$")
    kind: Literal["cv", "vlm"]
    algorithm_version: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    enabled: bool
    license_warning: str | None = Field(default=None, max_length=_TEXT)


class BlueprintComponent(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    component: MakeupComponent
    sources: dict[str, SourceKind] = Field(default_factory=dict, max_length=_MAX_SOURCES)

    @field_validator("sources")
    @classmethod
    def _source_keys(cls, v: dict[str, SourceKind]) -> dict[str, SourceKind]:
        for key in v:
            if key not in ATTRIBUTABLE_FIELDS:
                raise ValueError("Unknown source key")
        return v

    @model_validator(mode="after")
    def _sources_match_fields(self) -> "BlueprintComponent":
        populated: set[str] = set()
        for name in ATTRIBUTABLE_FIELDS:
            value = getattr(self.component, name)
            if name == "target_zones":
                if value:
                    populated.add(name)
            elif value is not None:
                populated.add(name)
        for name in populated:
            if name not in self.sources:
                raise ValueError("Populated field is missing a source")
        for key in self.sources:
            if key not in populated:
                raise ValueError("Source refers to an unpopulated field")
        for key in _CV_ONLY_FIELDS:
            if key in self.sources and self.sources[key] != "cv":
                raise ValueError("Field must have source 'cv'")
        return self


class BlueprintStyle(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    labels: list[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=8)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    source: Literal["vlm"] = "vlm"

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for raw in v:
            norm = unicodedata.normalize("NFKC", raw).casefold().strip()
            norm = norm.replace(" ", "_").replace("-", "_")
            if len(norm) > 32:
                raise ValueError("Style label too long")
            if norm not in STYLE_LABELS:
                raise ValueError("Unknown style label")
            if norm in out:
                raise ValueError("Duplicate style label")
            out.append(norm)
        return out

    @field_validator("confidence")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class BlueprintWarning(BaseModel):
    """message is server-authored only; never populate from VLM output."""

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    code: Literal[
        "vlm_disabled",
        "vlm_failed",
        "vlm_refused",
        "no_makeup_detected",
        "low_confidence",
        "zone_outside_layer",
        "glasses_present",
    ]
    # message: server-authored only; never populate from VLM output.
    message: str | None = Field(default=None, max_length=256)


class BlueprintQuality(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    warnings: list[BlueprintWarning] = Field(default_factory=list, max_length=32)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("overall_confidence")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class MakeupBlueprint(BaseModel):
    """Versioned makeup blueprint.

    When logging or returning validation errors, use `str(exc)` or
    `exc.errors(include_input=False)`; plain `errors()` includes raw input. Never build
    with model_construct or model_copy on untrusted data: that bypasses validation.

    message: server-authored only; never populate from VLM output.
    """

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    schema_version: str = Field(default=BLUEPRINT_SCHEMA_VERSION, description="MAJOR.MINOR")
    status: Literal["ok", "no_face_detected", "multiple_faces_detected"]
    message: str | None = Field(default=None, max_length=_TEXT)
    image: ImageSize
    components: list[BlueprintComponent] = Field(default_factory=list, max_length=32)
    style: BlueprintStyle | None = None
    quality: BlueprintQuality = Field(default_factory=BlueprintQuality)
    providers: list[ReferenceProviderInfo] = Field(default_factory=list, max_length=8)
    timings_ms: dict[TimingKey, TimingValue] =Field(default_factory=dict, max_length=32)

    @field_validator("schema_version")
    @classmethod
    def _version_supported(cls, v: str) -> str:
        if not is_supported(v):
            raise ValueError("Unsupported blueprint schema version")
        return v

    @model_validator(mode="after")
    def _match_status(self) -> "MakeupBlueprint":
        kinds = [p.kind for p in self.providers]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Provider kinds must be unique")
        if self.status != "ok" and (self.components or self.style is not None or self.providers):
            raise ValueError("components, style and providers must be empty unless status is 'ok'")
        return self
