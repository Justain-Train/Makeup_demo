"""FacialProfile: one versioned, self-describing bundle of the facial analysis outputs.

Assembly only: no feature classification. All coordinates are original-image pixels
(after EXIF orientation), origin top-left. Region and zone names are canonical only;
no model-specific class ids appear here.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

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
    kind: Literal["landmarks", "parsing", "zones"]
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


class ComponentSet(BaseModel):
    landmarks: Component[LandmarksData]
    frame: Component[FaceFrameData]
    parsing: Component[ParsingData]
    zones: Component[ZonesData]

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
