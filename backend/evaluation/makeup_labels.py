"""Pixel-residual labels for FFHQ-Makeup pairs and loss-stage classification. Offline; no logging."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.schemas.makeup_blueprint import MakeupBlueprint
from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_preview.renderer import RenderResult
from evaluation.makeup_signals import Lab, lab_of_hex

LABEL_MARGIN_FRACTION = 0.5
SENSITIVITY_FRACTIONS = (0.35, 0.75)
STAGES = ("not_detected", "dropped_in_adapt", "mask_missing_or_empty", "rendered")


def _region_lab(image: np.ndarray, mask: np.ndarray) -> Lab | None:
    color_hex = cv.trimmed_median_hex(image[mask])
    return None if color_hex is None else lab_of_hex(color_hex)


def label_zone(
    bare: np.ndarray,
    makeup: np.ndarray,
    zone_mask: np.ndarray,
    skin_ref_mask: np.ndarray,
    row: cv.TargetRow,
    margin_fraction: float = LABEL_MARGIN_FRACTION,
) -> tuple[bool, float]:
    """(positive, residual). Residual = (zone change) - (skin-reference change), so a global shift cancels."""
    zb, zm = _region_lab(bare, zone_mask), _region_lab(makeup, zone_mask)
    sb, sm = _region_lab(bare, skin_ref_mask), _region_lab(makeup, skin_ref_mask)
    if zb is None or zm is None or sb is None or sm is None:
        return False, 0.0
    delta = tuple((zm[i] - zb[i]) - (sm[i] - sb[i]) for i in range(3))
    value = cv._metric(replace(row, present_threshold=float("-inf")), delta, (0.0, 0.0, 0.0))  # type: ignore[arg-type]
    residual = 0.0 if value is None else float(value)
    return residual >= margin_fraction * row.present_threshold, residual


def classify_stage(
    label_pos: bool,
    blueprint: MakeupBlueprint | None,
    plan: AdaptedMakeupPlan | None,
    render: RenderResult | None,
    category: str,
) -> tuple[str, str | None] | None:
    """Where a labelled-positive product was lost; None when the label is not positive."""
    if not label_pos:
        return None
    if blueprint is None or not any(c.component.category == category for c in blueprint.components):
        return "not_detected", None
    comps = [c for c in (plan.components if plan is not None else []) if c.component.category == category]
    if not comps:
        return "dropped_in_adapt", None
    comp = comps[0]
    report = None
    if render is not None:
        report = next((r for r in render.reports if r.index == comp.index), None)
    if comp.mask is None or report is None:
        return "mask_missing_or_empty", (comp.notes[0] if comp.notes else "no_mask")
    if report.status != "rendered":
        return "mask_missing_or_empty", report.reason or "skipped"
    if report.pixels_touched <= 0:
        return "mask_missing_or_empty", "empty_mask"
    return "rendered", None
