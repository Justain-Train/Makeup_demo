"""Skin depth (ITA) and undertone (Lab hue) from the bare-skin reference pixels. No logging by design.

Caveats: depth is photo-measured ITA (exposure- and white-balance-dependent, unlike colorimeter ITA).
Undertone edges are quantile-fitted on one dataset. The cast flag uses background chroma as a proxy
(coloured walls confound it). Downstream code must treat both as soft, confirmable hints and never
gate users on them.
"""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from app.schemas.face_segmentation import Zone
from app.schemas.facial_profile import Classified, SkinToneData
from app.services.facial_features import thresholds as t
from app.services.facial_features._common import band, unknown
from app.services.makeup_blueprint.cv_analysis import (
    MIN_REF_PIXELS,
    SKIN_ZONE,
    bare_skin_reference,
    decode_zone_mask,
    trimmed_median_hex,
)
from app.services.makeup_vocabulary.color import hex_to_lab

_DEPTH_LABELS = t.SKIN_ITA_LABELS


def _unknown_result(reason: str, lighting_reasons: list[str]) -> SkinToneData:
    return SkinToneData(
        depth=unknown(reason),
        undertone=unknown(reason),
        lighting_quality="poor",
        lighting_reasons=lighting_reasons,  # type: ignore[arg-type]
    )


def _cast_chroma(image: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean a*/b* chroma of pixels outside the padded skin bbox, on a strided subgrid."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    h, w = mask.shape
    pad = t.SKIN_BG_DILATE_PX
    y0, y1 = int(rows[0]) - pad, int(rows[-1]) + pad
    x0, x1 = int(cols[0]) - pad, int(cols[-1]) + pad
    step = max(1, max(h, w) // t.SKIN_BG_MAX_SIDE)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    inside = ((ys >= y0) & (ys <= y1))[:, None] & ((xs >= x0) & (xs <= x1))[None, :]
    outside = ~inside
    if int(outside.sum()) < t.SKIN_BG_MIN_PX:
        return None
    sub = np.ascontiguousarray(image[::step, ::step])
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)[outside].astype(np.float64)
    chroma = np.hypot(lab[:, 1] - 128.0, lab[:, 2] - 128.0)
    return float(chroma.mean())


def _undertone(hue: float, chroma: float, measurements: dict[str, float], factor: float) -> Classified:
    if chroma < t.SKIN_MIN_CHROMA:
        return unknown("chroma_too_low", measurements)
    h = hue - 360.0 if hue > 180.0 else hue
    if chroma < t.SKIN_OLIVE_LOW_CHROMA and h > t.SKIN_OLIVE_LOW_CHROMA_HUE:
        edges: tuple[float, ...] = (t.SKIN_HUE_COOL_BELOW, t.SKIN_HUE_WARM_ABOVE)
        labels: tuple[str, ...] = ("cool", "neutral", "olive")
    else:
        edges = (t.SKIN_HUE_COOL_BELOW, t.SKIN_HUE_WARM_ABOVE, t.SKIN_HUE_OLIVE_ABOVE)
        labels = ("cool", "neutral", "warm", "olive")
    return band(
        h, edges, labels, t.SKIN_UNDERTONE_MARGIN, t.SKIN_UNDERTONE_SCALE, measurements,
        t.SKIN_UNDERTONE_CONF_CAP * factor,
    )


def classify_skin_tone(image: np.ndarray, zones: Sequence[Zone], glasses: bool = False) -> SkinToneData:
    try:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return _unknown_result("degenerate_input", [])
        by_id = {z.id: z for z in zones}
        skin_zone = by_id.get(SKIN_ZONE)
        if skin_zone is None or skin_zone.status != "ok":
            return _unknown_result("insufficient_skin", ["insufficient_skin"])

        crops: dict[str, tuple[np.ndarray, int, int] | None] = {}

        def crop_of(zone_id: str) -> tuple[np.ndarray, int, int] | None:
            if zone_id not in crops:
                zone = by_id.get(zone_id)
                crops[zone_id] = None if zone is None else decode_zone_mask(image.shape, zone)
            return crops[zone_id]

        ref = bare_skin_reference(image, crop_of)
        if ref is None or int(ref[0].sum()) < MIN_REF_PIXELS:
            return _unknown_result("insufficient_skin", ["insufficient_skin"])
        mask = ref[0]
        skin_hex = trimmed_median_hex(image[mask])
        if skin_hex is None:
            return _unknown_result("insufficient_skin", ["insufficient_skin"])
        l_star, a_star, b_star = hex_to_lab(skin_hex)
        ita = math.degrees(math.atan2(l_star - 50.0, b_star))
        hue = math.degrees(math.atan2(b_star, a_star)) % 360.0
        chroma = math.hypot(a_star, b_star)
        measurements = {
            "l_star": l_star, "a_star": a_star, "b_star": b_star, "ita_deg": ita,
            "hue_deg": hue, "chroma": chroma, "skin_px": float(mask.sum()),
        }

        px = image[mask]
        crushed_frac = float((px.max(axis=1) <= t.SKIN_CRUSHED_MAX).mean())
        blown_frac = float((px.min(axis=1) >= t.SKIN_BLOWN_MIN).mean())
        measurements["crushed_frac"] = crushed_frac
        measurements["blown_frac"] = blown_frac

        reasons: list[str] = []
        extreme = False
        clip_caution = False
        if crushed_frac > t.SKIN_CLIP_CAUTION:
            reasons.append("too_dark")
            clip_caution = True
            extreme = crushed_frac > t.SKIN_CLIP_POOR
        elif blown_frac > t.SKIN_CLIP_CAUTION:
            reasons.append("overexposed")
            clip_caution = True
            extreme = blown_frac > t.SKIN_CLIP_POOR

        cast = _cast_chroma(image, mask)
        cast_level = "good"
        if cast is not None:
            measurements["cast_chroma"] = cast
            if cast >= t.SKIN_CAST_CHROMA_POOR:
                cast_level = "poor"
            elif cast >= t.SKIN_CAST_CHROMA_CAUTION:
                cast_level = "caution"
        if cast_level != "good":
            reasons.append("color_cast")
        if glasses:
            reasons.append("glasses")

        factor = 1.0
        if clip_caution:
            factor *= t.SKIN_CLIP_CONF_FACTOR
        if cast_level == "caution":
            factor *= t.SKIN_CAST_CONF_FACTOR
        if glasses:
            factor *= t.SKIN_GLASSES_CONF_FACTOR

        if extreme:
            depth = unknown(reasons[0], measurements)
        else:
            depth = band(
                ita, t.SKIN_ITA_EDGES, _DEPTH_LABELS, t.SKIN_ITA_MARGIN, t.SKIN_ITA_SCALE,
                measurements, t.SKIN_DEPTH_CONF_CAP * factor,
            )
        if extreme:
            undertone = unknown(reasons[0], measurements)
        elif cast_level == "poor":
            undertone = unknown("color_cast", measurements)
        else:
            undertone = _undertone(hue, chroma, measurements, factor)

        if extreme or cast_level == "poor":
            quality = "poor"
        elif cast_level == "caution" or clip_caution:
            quality = "caution"
        else:
            quality = "good"
        return SkinToneData(
            depth=depth, undertone=undertone, lighting_quality=quality,  # type: ignore[arg-type]
            lighting_reasons=reasons,  # type: ignore[arg-type]
        )
    except Exception:
        return _unknown_result("degenerate_input", [])

