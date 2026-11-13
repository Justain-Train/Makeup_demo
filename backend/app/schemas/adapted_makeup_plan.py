"""AdaptedMakeupPlan: a MakeupBlueprint adapted to one user's FacialProfile.

Embeds the blueprint's validated MakeupComponent unchanged (colour, finish, intensity, ...) and adds
stage, zone selection and the user's zone mask. Use `exc.errors(include_input=False)` when logging.
Never build with model_construct or model_copy.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.face_segmentation import ZoneBBox, ZoneMask
from app.schemas.facial_profile import ImageSize
from app.schemas.makeup_blueprint import SourceKind, TimingKey, TimingValue
from app.schemas.makeup_component import MakeupComponent
from app.services.makeup_adaptation.versions import PLAN_SCHEMA_VERSION, is_supported

_TEXT = 512

PlanStatus = Literal["ok", "no_face_detected", "multiple_faces_detected", "reference_unavailable", "no_makeup_detected"]

ComponentNote = Literal[
    "reference_zones_outside_layer",
    "no_zones_for_placement",
    "zone_unavailable_on_user",
    "zone_mask_invalid",
    "no_stage",
    "direction_angles_unavailable",
]


class PlanProviderSummary(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    id: str = Field(max_length=_TEXT)
    kind: str = Field(max_length=32)
    algorithm_version: str = Field(max_length=_TEXT)
    model: str | None = Field(default=None, max_length=64)


class ReferenceSummary(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    image: ImageSize
    blueprint_schema_version: str = Field(max_length=16)
    style_labels: list[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=8)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    component_count: int = Field(ge=0, le=32)


class PlanWarning(BaseModel):
    """message is server-authored only."""

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    code: Literal[
        "reference_zones_outside_layer",
        "no_zones_for_placement",
        "zone_unavailable_on_user",
        "zone_mask_invalid",
        "no_stage",
        "direction_angles_unavailable",
        "zones_unavailable",
        "features_unavailable",
    ]
    message: str = Field(max_length=256)


RuleReason = Literal[
    "lift_downturned_eyes",
    "balance_upturned_eyes",
    "emphasize_outer_corner_close_set",
    "emphasize_inner_corner_wide_set",
    "elongate_round_eyes",
    "contour_round_face",
    "horizontal_blush_oblong_face",
    "soften_square_jaw",
    "balance_heart_face",
    "add_lip_outer_boundary_thin_lips",
    "emphasize_cupid_bow_bottom_heavy",
    "keep_straight_brows_horizontal",
]

MapperReason = Literal[
    "layer_zones",
    "placement_match",
    "explicit_zones",
    "placement_not_in_layer",
    "layer_has_no_zones",
    "special_effect_note_only",
]

# Coarse direction label; direction_angles on the Adjustment carry the face-frame and image-frame angles.
Direction = Literal["upward_outward", "horizontal", "diagonal_up", "outward"]

RuleFeature = Literal[
    "eye_tilt", "eye_spacing", "eye_shape", "face_shape", "lip_fullness", "lip_balance", "brow_arch"
]

RuleOutcome = Literal[
    "applied",
    "skipped_uncertain",
    "skipped_not_triggered",
    "skipped_zone_unavailable",
    "skipped_not_applicable",
    "skipped_conflict",
]


class DirectionAngle(BaseModel):
    """Angles in degrees, both in (-180, 180]. side is the subject's side."""

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    side: Literal["left", "right"]
    face_deg: float = Field(gt=-180.0, le=180.0, allow_inf_nan=False)
    image_deg: float = Field(gt=-180.0, le=180.0, allow_inf_nan=False)


class Adjustment(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    rule_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    reason: RuleReason
    effect: Literal["zones", "direction", "intensity", "emphasis"]
    direction: Direction | None = None
    direction_angles: list[DirectionAngle] = Field(default_factory=list, max_length=2)
    intensity_multiplier: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    emphasis_zones: list[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=8)


# Labels a rule can record: the classifier label sets of the features rules read, plus "unknown".
RuleLabel = Literal[
    "unknown",
    "close", "average", "wide",            # eye_spacing
    "upturned", "neutral", "downturned",   # eye_tilt
    "almond", "round",                     # eye_shape (round also a face shape)
    "oval", "square", "heart", "oblong", "diamond",  # face_shape
    "thin", "medium", "full",              # lip_fullness
    "top_heavy", "balanced", "bottom_heavy",  # lip_balance
    "straight", "soft_arched", "high_arched",  # brow_arch
]


class RuleEvaluation(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    rule_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    feature: RuleFeature
    label: RuleLabel
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    outcome: RuleOutcome


class ZoneSelection(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    source: Literal["reference", "placement", "layer_default", "none"]
    reason: MapperReason | RuleReason | None = None


class AdaptedComponent(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    index: int = Field(ge=0, le=31)
    component: MakeupComponent
    stage_key: str | None = Field(default=None, max_length=64)
    stage_number: int | None = None
    layer_key: str | None = Field(default=None, max_length=64)
    zones: list[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=70)
    zone_selection: ZoneSelection
    mask: ZoneMask | None = None
    bbox: ZoneBBox | None = None
    centroid: tuple[float, float] | None = None
    pixel_count: int = Field(default=0, ge=0)
    sources: dict[str, SourceKind] = Field(default_factory=dict, max_length=16)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    adjustments: list[Adjustment] = Field(default_factory=list, max_length=16)
    rule_evaluations: list[RuleEvaluation] = Field(default_factory=list, max_length=32)
    notes: list[ComponentNote] = Field(default_factory=list, max_length=8)

    @field_validator("notes")
    @classmethod
    def _unique_notes(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("Notes must be unique")
        return v

    @model_validator(mode="after")
    def _mask_consistency(self) -> "AdaptedComponent":
        if (self.bbox is None) != (self.centroid is None):
            raise ValueError("bbox and centroid must be set together")
        if self.mask is not None and self.bbox is None:
            raise ValueError("mask requires bbox and centroid")
        if self.bbox is None and self.pixel_count != 0:
            raise ValueError("pixel_count must be 0 without a bbox")
        return self


class AdaptedMakeupPlan(BaseModel):
    """message is server-authored only."""

    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    schema_version: str = Field(default=PLAN_SCHEMA_VERSION, description="MAJOR.MINOR")
    status: PlanStatus
    message: str | None = Field(default=None, max_length=_TEXT)
    image: ImageSize
    reference: ReferenceSummary | None = None
    components: list[AdaptedComponent] = Field(default_factory=list, max_length=32)
    warnings: list[PlanWarning] = Field(default_factory=list, max_length=32)
    providers: list[PlanProviderSummary] = Field(default_factory=list, max_length=8)
    timings_ms: dict[TimingKey, TimingValue] = Field(default_factory=dict, max_length=32)

    @field_validator("schema_version")
    @classmethod
    def _version_supported(cls, v: str) -> str:
        if not is_supported(v):
            raise ValueError("Unsupported plan schema version")
        return v

    @model_validator(mode="after")
    def _match_status(self) -> "AdaptedMakeupPlan":
        if self.status != "ok" and self.components:
            raise ValueError("components must be empty unless status is 'ok'")
        if self.reference is not None and self.status not in ("ok", "no_makeup_detected"):
            raise ValueError("reference is only allowed for status 'ok' or 'no_makeup_detected'")
        return self
