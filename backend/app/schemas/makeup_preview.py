"""Response schema for the generative makeup preview (renderer output and per-component report)."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SkipReason = Literal["no_mask", "mask_invalid", "no_colour", "zero_opacity", "empty_mask"]


class ComponentRenderReport(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    index: int = Field(ge=0, le=31)
    status: Literal["rendered", "skipped"]
    reason: SkipReason | None = None
    opacity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    finish_applied: str = Field(max_length=32)
    directional_stretch: Literal["deferred", "applied", "not_applicable"]
    pixels_touched: int = Field(ge=0)


class PreviewResponse(BaseModel):
    model_config = {"extra": "ignore", "hide_input_in_errors": True}

    schema_version: Literal["1.0"] = "1.0"
    image_png_base64: str = Field(max_length=4 * math.ceil(60_000_000 / 3))
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    plan_status: str = Field(max_length=32)
    reports: list[ComponentRenderReport] = Field(default_factory=list, max_length=32)
    warnings: list[Annotated[str, Field(max_length=256)]] = Field(default_factory=list, max_length=32)
