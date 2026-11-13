"""FacialProfile: one versioned, self-describing bundle of the facial analysis outputs.

Assembly plus geometric feature classification (features). All coordinates are original-image pixels
(after EXIF orientation), origin top-left. Region and zone names are canonical only;
no model-specific class ids appear here.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.face_measurements import FaceMeasurements
from app.schemas.face_segmentation import (
    AlignmentCheck,
    AnatomicalLayer,
    ParseWindow,
    RegionInfo,
    SegmentationWarning,
    Stage,
    Zone,
)
from app.schemas.facial_landmarks import DetectionThresholds, ImageSize, Landmark
from app.services.facial_profile.versions import PROFILE_SCHEMA_VERSION

T = TypeVar("T")

ComponentStatus = Literal["ok", "degraded", "unavailable"]


# Compatibility policy: models keep pydantic's default extra="ignore", so unknown
# fields from newer minor versions are ignored on load.
_TEXT = 512


class ProviderInfo(BaseModel):
    id: str = Field(max_length=_TEXT)
    kind: Literal["landmarks", "parsing", "zones", "features"]
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    algorithm_version: str = Field(max_length=_TEXT)
    license_warning: str | None = Field(default=None, max_length=_TEXT)


class ComponentMeta(BaseModel):
    component_version: int = Field(description="Integer version of this component's payload shape")
    status: ComponentStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Uncalibrated")
    provider: ProviderInfo
    reason: str | None = Field(default=None, max_length=_TEXT)


class Component(BaseModel, Generic[T]):
    meta: ComponentMeta
    data: T | None = Field(default=None, description="None if and only if meta.status is 'unavailable'")

    @model_validator(mode="after")
    def _data_matches_status(self) -> "Component[T]":
        unavailable = self.meta.status == "unavailable"
        if unavailable and self.data is not None:
            raise ValueError("data must be None when status is 'unavailable'")
        if not unavailable and self.data is None:
            raise ValueError("data is required unless status is 'unavailable'")
        return self


class LandmarksData(BaseModel):
    thresholds: DetectionThresholds
    landmarks: list[Landmark] = Field(
        max_length=478, description="478 points: 0-467 face mesh, 468-477 irises"
    )
    measurements: FaceMeasurements | None = None


class FaceFrameData(BaseModel):
    ipd_px: float = Field(description="Interpupillary distance in image pixels")
    roll_deg: float
    u: tuple[float, float] = Field(description="Unit vector, subject's right eye -> left eye (image coordinates)")
    v: tuple[float, float] = Field(description="Unit vector perpendicular to u, down the face (image coordinates)")
    midline: list[tuple[float, float]] = Field(max_length=64, description="Facial midline points in image pixels")


class ParsingData(BaseModel):
    parse_window: ParseWindow | None = None
    includes_masks: bool
    regions: list[RegionInfo] = Field(max_length=64, description="Canonical region names only")
    label_map: AnatomicalLayer | None = Field(
        default=None, description="None when masks are excluded. Canonical region names only."
    )


class ZonesData(BaseModel):
    zones: list[Zone] = Field(
        max_length=128, description="Canonical zone ids only; masks are None when excluded"
    )
    stages: list[Stage] = Field(max_length=32)


L = TypeVar("L", bound=str)

_MEASUREMENT_KEY = r"^[a-z0-9_]{1,64}$"


class Classified(BaseModel, Generic[L]):
    """One geometric classification. Heuristic and uncalibrated; reason is server-authored only."""

    model_config = ConfigDict(hide_input_in_errors=True)

    label: L | Literal["unknown"]
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    measurements: dict[str, float] = Field(default_factory=dict, max_length=16)
    reason: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _check_measurements(self) -> "Classified[L]":
        import math
        import re

        for key, value in self.measurements.items():
            if not re.match(_MEASUREMENT_KEY, key):
                raise ValueError("invalid measurement key")
            if not math.isfinite(value):
                raise ValueError("measurement values must be finite")
        return self


class EyelidType(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    value: Literal["monolid", "inner_fold", "double", "hooded", "unknown"] = "unknown"
    source: Literal["user", "unknown"] = "unknown"


class SkinToneData(BaseModel):
    """Skin depth and undertone with lighting diagnostics. Heuristic and uncalibrated."""

    model_config = ConfigDict(hide_input_in_errors=True)

    depth: Classified[Literal["very_light", "light", "intermediate", "tan", "brown", "dark"]]
    undertone: Classified[Literal["warm", "neutral", "cool", "olive"]]
    lighting_quality: Literal["good", "caution", "poor"]
    lighting_reasons: list[
        Literal["too_dark", "overexposed", "color_cast", "glasses", "insufficient_skin"]
    ] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _unique_reasons(self) -> "SkinToneData":
        if len(set(self.lighting_reasons)) != len(self.lighting_reasons):
            raise ValueError("lighting_reasons must be unique")
        return self


class FeaturesData(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    eye_spacing: Classified[Literal["close", "average", "wide"]]
    eye_tilt: Classified[Literal["upturned", "neutral", "downturned"]]
    eye_shape: Classified[Literal["almond", "round"]]
    eyelid_type: EyelidType = Field(default_factory=EyelidType)
    lip_fullness: Classified[Literal["thin", "medium", "full"]]
    lip_balance: Classified[Literal["top_heavy", "balanced", "bottom_heavy"]]
    lip_width: Classified[Literal["narrow", "average", "wide"]]
    brow_arch: Classified[Literal["straight", "soft_arched", "high_arched"]]
    brow_thickness: Classified[Literal["thin", "medium", "thick"]]
    face_shape: Classified[Literal["oval", "round", "square", "heart", "oblong", "diamond"]] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    skin_tone: SkinToneData | None = Field(default=None, exclude_if=lambda v: v is None)


class ComponentSet(BaseModel):
    landmarks: Component[LandmarksData]
    frame: Component[FaceFrameData]
    parsing: Component[ParsingData]
    zones: Component[ZonesData]
    # Additive in 1.1: omitted from output when absent, so 1.0 readers that iterate components never see null.
    features: Component[FeaturesData] | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _provider_kinds_match_slots(self) -> "ComponentSet":
        expected = {
            "landmarks": "landmarks",
            "frame": "landmarks",
            "parsing": "parsing",
            "zones": "zones",
        }
        for slot, kind in expected.items():
            if getattr(self, slot).meta.provider.kind != kind:
                raise ValueError(f"{slot} component provider kind must be '{kind}'")
        if self.features is not None and self.features.meta.provider.kind != "features":
            raise ValueError("features component provider kind must be 'features'")
        return self


class ProfileQuality(BaseModel):
    alignment: AlignmentCheck | None = None
    warnings: list[SegmentationWarning] = Field(default_factory=list, max_length=32)
    retake_suggested: list[Literal["glasses"]] = Field(
        default_factory=list,
        description="Soft hint only: the photo is never rejected. Empty unless status is 'ok'.",
    )


class FacialProfile(BaseModel):
    schema_version: str = Field(default=PROFILE_SCHEMA_VERSION, description="MAJOR.MINOR")
    status: Literal["ok", "no_face_detected", "multiple_faces_detected"]
    message: str | None = Field(default=None, max_length=_TEXT)
    image: ImageSize
    components: ComponentSet | None = Field(
        default=None, description="Present if and only if status is 'ok'. Canonical region names only."
    )
    quality: ProfileQuality = Field(default_factory=ProfileQuality)
    timings_ms: dict[str, float] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def _components_match_status(self) -> "FacialProfile":
        if self.status == "ok":
            if self.components is None:
                raise ValueError("components are required when status is 'ok'")
        else:
            if self.components is not None:
                raise ValueError("components must be None when status is not 'ok'")
            if self.quality.retake_suggested:
                raise ValueError("retake_suggested must be empty unless status is 'ok'")
        return self
