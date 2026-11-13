"""Raw Phase 7 detection signals: a replica of CvReferenceAnalyzer.analyze (l.229-317) that keeps sub-threshold values.

Uses the production helpers (`bare_skin_reference`, `trimmed_median_hex`, `_metric`, TARGETS). The threshold is
neutralised on a `dataclasses.replace` copy of the row, never by patching module globals. Offline only; no logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from app.schemas.facial_profile import FacialProfile
from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_vocabulary.color import hex_to_lab

Lab = tuple[float, float, float]


@dataclass
class FaceContext:
    skin_ref: np.ndarray
    fallback: bool
    skin_lab: Lab
    glasses: bool
    zone_masks: dict[str, np.ndarray | None] = field(default_factory=dict)
    product_masks: dict[str, np.ndarray | None] = field(default_factory=dict)


def lab_of_hex(color_hex: str) -> Lab:
    lab = hex_to_lab(color_hex)
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def prepare(image: np.ndarray, profile: FacialProfile) -> FaceContext | None:
    """Skin reference, usable zone masks and the glasses flag, mirroring analyze(); None where analyze() returns empty."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    comps = profile.components if profile.status == "ok" else None
    if comps is None:
        return None
    zones_c, parsing_c = comps.zones, comps.parsing
    if zones_c.meta.status == "unavailable" or parsing_c.meta.status == "unavailable" or zones_c.data is None:
        return None

    zone_by_id = {z.id: z for z in zones_c.data.zones}
    crops: dict[str, tuple[np.ndarray, int, int] | None] = {}

    def crop_of(zone_id: str) -> tuple[np.ndarray, int, int] | None:
        if zone_id not in crops:
            zone = zone_by_id.get(zone_id)
            crops[zone_id] = None if zone is None else cv.decode_zone_mask(image.shape, zone)
        return crops[zone_id]

    skin_zone = zone_by_id.get(cv.SKIN_ZONE)
    skin_crop = crop_of(cv.SKIN_ZONE)
    if skin_zone is None or skin_crop is None or skin_zone.status != "ok":
        return None
    ref = cv.bare_skin_reference(image, crop_of)
    if ref is None:
        return None
    ref_mask, fallback = ref
    skin_hex = cv.trimmed_median_hex(image[ref_mask])
    if skin_hex is None:
        return None

    def usable_mask(zone_id: str) -> np.ndarray | None:
        zone = zone_by_id.get(zone_id)
        c = crop_of(zone_id)
        if zone is None or c is None or zone.status != "ok":
            return None
        if int(c[0].sum()) < cv.MIN_ZONE_PIXELS:
            return None
        return cv.full_zone_mask(image, c)

    glasses = any(w.code == "glasses_detected" for w in profile.quality.warnings)
    ctx = FaceContext(ref_mask, fallback, lab_of_hex(skin_hex), glasses)
    for row in cv.TARGETS:
        union: np.ndarray | None = None
        for zid in row.zones:
            m = usable_mask(zid)
            ctx.zone_masks[zid] = m
            if m is not None:
                union = m.copy() if union is None else (union | m)
        ctx.product_masks[row.product_type] = union
    return ctx


def _empty(zones: list[str] | None = None) -> dict[str, Any]:
    return {"value": None, "dl": None, "da": None, "db": None, "gate_ok": False, "usable": False,
            "detected": False, "zones": zones or []}


def _product_signal(image: np.ndarray, ctx: FaceContext, row: cv.TargetRow) -> dict[str, Any]:
    if row.product_type == "eyeshadow" and ctx.glasses:
        return _empty()
    used: list[str] = []
    parts: list[np.ndarray] = []
    for zid in row.zones:
        m = ctx.zone_masks.get(zid)
        if m is None:
            continue
        used.append(zid)
        parts.append(image[m])
    if not parts:
        return _empty()
    color_hex = cv.trimmed_median_hex(np.concatenate(parts, axis=0))
    if color_hex is None:
        return _empty(used)
    color_lab = lab_of_hex(color_hex)
    dl, da, db = (c - s for c, s in zip(color_lab, ctx.skin_lab))
    raw_row = replace(
        row,
        present_threshold=float("-inf"),
        metric="delta_e" if row.metric == "delta_e_a" else row.metric,
    )
    value = cv._metric(raw_row, color_lab, ctx.skin_lab)
    if value is None or not math.isfinite(value):
        return _empty(used)
    gate_ok = da >= cv.BLUSH_MIN_A_DELTA if row.metric == "delta_e_a" else True
    return {
        "value": float(value), "dl": float(dl), "da": float(da), "db": float(db),
        "gate_ok": bool(gate_ok), "usable": True,
        "detected": bool(gate_ok and value >= row.present_threshold), "zones": used,
    }


def signals_from_context(image: np.ndarray, ctx: FaceContext) -> dict[str, Any]:
    return {
        "skin_lab": list(ctx.skin_lab),
        "fallback": ctx.fallback,
        "glasses": ctx.glasses,
        "products": {row.product_type: _product_signal(image, ctx, row) for row in cv.TARGETS},
    }


def raw_signals(image_rgb: np.ndarray, profile: FacialProfile) -> dict[str, Any] | None:
    ctx = prepare(image_rgb, profile)
    if ctx is None:
        return None
    return signals_from_context(image_rgb, ctx)
