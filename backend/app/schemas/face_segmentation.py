"""Response schemas for POST /api/facial-analysis/segmentation.

All coordinates are original-image pixels (after EXIF orientation), origin
top-left. Region and zone names are canonical (see services/segmentation/regions.py);
no model-specific class ids appear here.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.face_measurements import FacialLandmarkResponse
from app.schemas.facial_landmarks import ImageSize

LICENSE_WARNING = (
    "Trained on CelebAMask-HQ (non-commercial research data). Not cleared for commercial use."
)

ZoneStatus = Literal["ok", "low_confidence", "unavailable"]


class ModelInfo(BaseModel):
    id: str
    architecture: str
    weights_sha256: str
    input_size: int
    license_warning: str = LICENSE_WARNING


class ParseWindow(BaseModel):
    """Square crop given to the parser. It may extend past the image edges."""

    x: int
    y: int
    size: int


class Point(BaseModel):
    x: int
    y: int


class RegionInfo(BaseModel):
    id: int = Field(description="Canonical region id = pixel value in the label map")
    name: str
    side: Literal["left", "right"] | None = Field(
        default=None, description=(
            "Landmark convention: 'right' = the side containing landmarks 33/133/468 (image-left in an "
            "unmirrored photo). Mirrored photos cannot be detected, so their sides follow the image geometry."
        )
    )
    source: Literal["segmentation"] = "segmentation"
    pixel_count: int
    area_fraction: float = Field(description="pixel_count / label map area")
    mean_probability: float | None = Field(
        default=None, description="Mean softmax probability of the winning class. Uncalibrated."
    )
    present: bool


class AnatomicalLayer(BaseModel):
    encoding: Literal["png_label_map_v1"] = "png_label_map_v1"
    origin: Point = Field(description="Image pixel of the label map's top-left corner")
    width: int
    height: int
    not_parsed_value: int = 255
    label_map_png_base64: str = Field(
        description="8-bit grayscale PNG; pixel value = canonical region id"
    )
    regions: list[RegionInfo]


class ZoneBBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ZoneMask(BaseModel):
    encoding: Literal["png_bbox_v1"] = "png_bbox_v1"
    png_base64: str = Field(description="8-bit grayscale PNG of the bbox; 255 = in zone, 0 = out")


class Zone(BaseModel):
    id: str
    name: str
    group: str
    side: Literal["left", "right"] | None = Field(
        description="Same landmark convention as RegionInfo.side (subject-right = side containing landmarks 33/133/468)"
    )
    sources: list[Literal["segmentation", "landmarks"]]
    status: ZoneStatus
    reason: str | None = None
    confidence: float | None = Field(
        default=None,
        description="Mean parser probability, or landmark/parser IoU for validated zones. Uncalibrated.",
    )
    coverage: float | None = Field(
        default=None, description="Clipped area / in-image geometric area (clipped zones only)"
    )
    clipped_to: list[str]
    bbox: ZoneBBox | None
    pixel_count: int
    mask: ZoneMask | None
    polygon: list[tuple[float, float]] | None = Field(
        default=None, description="Landmark-derived outline (image pixels); landmark zones only"
    )


class Stage(BaseModel):
    number: int
    key: str
    name: str
    layers: dict[str, list[str]] = Field(
        description="Product layer key -> zone ids. An empty list means its zones are not implemented yet."
    )


class AlignmentCheck(BaseModel):
    passed: bool
    fraction: float = Field(description="Fraction of landmark/parser consistency checks passed")


WarningCode = Literal[
    "face_partially_outside_image",
    "low_resolution_face",
    "head_tilt",
    "glasses_detected",
    "parser_landmark_mismatch",
    "low_overlap_zone",
    "hair_occludes_forehead",
    "background_face_ignored",
]


class SegmentationWarning(BaseModel):
    code: WarningCode
    message: str


class FaceSegmentationResponse(BaseModel):
    status: Literal["ok", "no_face_detected", "multiple_faces_detected"]
    message: str | None = None
    image: ImageSize
    model: ModelInfo
    landmarks: FacialLandmarkResponse
    parse_window: ParseWindow | None = None
    anatomical: AnatomicalLayer | None = None
    zones: list[Zone] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)
    alignment_check: AlignmentCheck | None = None
    warnings: list[SegmentationWarning] = Field(default_factory=list)
    retake_suggested: list[Literal["glasses"]] = Field(
        default_factory=list,
        description="Soft hint only: the photo is never rejected. Empty unless status is 'ok'.",
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)
