"""CV-only reference analyzer: (RGB image, FacialProfile with masks) -> blueprint components.

Pure function, no logging, no module state. Colors come from zone pixels using the existing
Lab code. Known limitation: under heavy foundation the bare-skin reference is skin plus
foundation, so blush/highlight/contour deltas are relative to that.
Brows, eyeliner and mascara are not analyzed (natural hair and lashes are indistinguishable).
"""

import base64
import binascii
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from pydantic import ValidationError

from app.schemas.facial_profile import FacialProfile
from app.schemas.face_segmentation import Zone
from app.schemas.makeup_blueprint import (
    BlueprintComponent,
    BlueprintWarning,
    ReferenceProviderInfo,
)
from app.schemas.makeup_component import MakeupComponent
from app.services.makeup_mapping import map_zones
from app.services.makeup_vocabulary.color import hex_to_lab, srgb_to_linear
from app.services.segmentation import masks as mk

PROVIDER_ID = "cv_zone_color"
ALGORITHM_VERSION = "1.0"

MIN_ZONE_PIXELS = 200
MIN_REF_PIXELS = 2000
TRIM_LOW_PCT = 10.0
TRIM_HIGH_PCT = 90.0
FALLBACK_CONFIDENCE_FACTOR = 0.5
DEFAULT_ZONE_CONFIDENCE = 0.5
BLUSH_MIN_A_DELTA = 3.0

UNDERTONE_MIN_CHROMA = 8.0
UNDERTONE_COOL_BELOW = 25.0
UNDERTONE_NEUTRAL_BELOW = 55.0
UNDERTONE_WARM_BELOW = 85.0
# Lab hue (deg): pink/red ~0-25 cool, orange ~55, yellow ~90, green ~136, cyan ~196, blue ~306.
# Yellow-green and green read as warm-leaning (olive); cool starts at blue (user, 2026-11-12).
UNDERTONE_OLIVE_BELOW = 200.0

SKIN_ZONE = "face_skin"
REFERENCE_EXCLUDED_ZONES: tuple[str, ...] = (
    "under_eye_inner_r",
    "under_eye_inner_l",
    "under_eye_outer_r",
    "under_eye_outer_l",
    "forehead_perimeter",
)

Metric = Literal["delta_e", "delta_e_a", "dl_pos", "dl_neg"]

_Y_ROW = (0.2126729, 0.7151522, 0.0721750)
_LINEAR_LUT = np.array([srgb_to_linear(i / 255.0) for i in range(256)], dtype=np.float64)

MSG_NO_MAKEUP = "No makeup detected in the reference."
MSG_LOW_CONFIDENCE = "Reference analysis has reduced confidence."
MSG_GLASSES = "Glasses present; eye makeup was not analyzed."


@dataclass(frozen=True)
class TargetRow:
    product_type: str
    category: str
    zones: tuple[str, ...]
    present_threshold: float
    full_threshold: float
    metric: Metric


TARGETS: tuple[TargetRow, ...] = (
    TargetRow("lipstick", "lip", ("lip_full",), 25.0, 55.0, "delta_e"),
    TargetRow("powder_blush", "blush", ("cheek_apple_l", "cheek_apple_r"), 7.0, 25.0, "delta_e_a"),
    TargetRow("eyeshadow", "eye", ("eyelid_upper_l", "eyelid_upper_r"), 12.0, 40.0, "delta_e"),
    TargetRow("powder_highlighter", "highlighter", ("cheekbone_l", "cheekbone_r"), 6.0, 20.0, "dl_pos"),
    TargetRow("contour_powder", "bronzer_contour", ("contour_hollow_l", "contour_hollow_r"), 6.0, 20.0, "dl_neg"),
)


class CvAnalysisError(Exception):
    pass


@dataclass(frozen=True)
class CvAnalysis:
    components: list[BlueprintComponent]
    warnings: list[BlueprintWarning]
    confidence: float | None


class ReferenceAnalyzer(Protocol):
    def info(self) -> ReferenceProviderInfo: ...

    def analyze(self, image: np.ndarray, profile: FacialProfile) -> CvAnalysis: ...


def undertone_from_lab(a: float, b: float) -> Literal["warm", "cool", "neutral", "olive"]:
    chroma = math.hypot(a, b)
    if chroma < UNDERTONE_MIN_CHROMA:
        return "neutral"
    hue = math.degrees(math.atan2(b, a))
    if hue < 0:
        hue += 360.0
    if hue < UNDERTONE_COOL_BELOW or hue >= UNDERTONE_OLIVE_BELOW:
        return "cool"
    if hue < UNDERTONE_NEUTRAL_BELOW:
        return "neutral"
    if hue < UNDERTONE_WARM_BELOW:
        return "warm"
    return "olive"


def _crop(image_shape: tuple[int, ...], zone: Zone) -> tuple[np.ndarray, int, int] | None:
    """Decoded bbox mask (bool) with its origin, or None if the zone is unusable."""
    bbox, mask = zone.bbox, zone.mask
    if bbox is None or mask is None or mask.encoding != "png_bbox_v1":
        return None
    height, width = image_shape[0], image_shape[1]
    if bbox.width <= 0 or bbox.height <= 0 or bbox.x < 0 or bbox.y < 0:
        return None
    if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
        return None
    if len(mask.png_base64) > 4 * math.ceil((bbox.width * bbox.height + 1024) / 3):
        return None
    try:
        raw = base64.b64decode(mask.png_base64, validate=True)
    except (ValueError, binascii.Error):
        return None
    if (
        len(raw) < 24
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[12:16] != b"IHDR"
        or struct.unpack(">II", raw[16:24]) != (bbox.width, bbox.height)
    ):
        return None
    try:
        arr = mk.decode_png(raw)
    except ValueError:
        return None
    if arr.ndim != 2 or arr.shape != (bbox.height, bbox.width):
        return None
    return arr > 0, bbox.x, bbox.y


def _trimmed_hex(pixels: np.ndarray) -> str | None:
    if pixels.shape[0] == 0:
        return None
    lin = _LINEAR_LUT[pixels]
    y = lin[:, 0] * _Y_ROW[0] + lin[:, 1] * _Y_ROW[1] + lin[:, 2] * _Y_ROW[2]
    lo, hi = np.percentile(y, [TRIM_LOW_PCT, TRIM_HIGH_PCT])
    kept = pixels[(y >= lo) & (y <= hi)]
    if kept.shape[0] == 0:
        return None
    r, g, b = (int(round(float(np.median(kept[:, i])))) for i in range(3))
    return f"#{r:02X}{g:02X}{b:02X}"


def _make(
    category: str,
    product_type: str,
    color_hex: str,
    undertone: str,
    intensity: float,
    confidence: float,
    target_zones: tuple[str, ...],
) -> BlueprintComponent:
    try:
        component = MakeupComponent(
            category=category,
            product_type=product_type,
            color_hex=color_hex,
            undertone=undertone,  # type: ignore[arg-type]
            intensity=intensity,
            confidence=confidence,
            target_zones=list(target_zones),
        )
        return BlueprintComponent(
            component=component,
            sources={
                "category": "cv",
                "product_type": "cv",
                "color_hex": "cv",
                "undertone": "cv",
                "intensity": "cv",
                "target_zones": "cv",
            },
        )
    except ValidationError as exc:
        detail = "; ".join(
            f"{e['type']}@{'.'.join(str(p) for p in e['loc'])}" for e in exc.errors(include_input=False)
        )
        raise CvAnalysisError(f"Component validation failed: {detail}") from None


def _metric(row: TargetRow, color_lab: tuple[float, float, float], skin_lab: tuple[float, float, float]) -> float | None:
    dl, da, db = (c - s for c, s in zip(color_lab, skin_lab))
    if row.metric == "dl_pos":
        value = dl
    elif row.metric == "dl_neg":
        value = -dl
    else:
        value = math.sqrt(dl * dl + da * da + db * db)
        if row.metric == "delta_e_a" and da < BLUSH_MIN_A_DELTA:
            return None
    return value if value >= row.present_threshold else None


class CvReferenceAnalyzer:
    def info(self) -> ReferenceProviderInfo:
        return ReferenceProviderInfo(
            id=PROVIDER_ID, kind="cv", algorithm_version=ALGORITHM_VERSION, model=None, enabled=True
        )

    def analyze(self, image: np.ndarray, profile: FacialProfile) -> CvAnalysis:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise CvAnalysisError("Image must be an RGB uint8 array")

        low = BlueprintWarning(code="low_confidence", message=MSG_LOW_CONFIDENCE)
        empty = CvAnalysis([], [low], None)
        comps = profile.components if profile.status == "ok" else None
        if comps is None:
            return empty
        zones_c, parsing_c = comps.zones, comps.parsing
        if (
            zones_c.meta.status == "unavailable"
            or parsing_c.meta.status == "unavailable"
            or zones_c.data is None
        ):
            return empty

        degraded = zones_c.meta.status == "degraded" or parsing_c.meta.status == "degraded"
        zone_by_id = {z.id: z for z in zones_c.data.zones}
        crops: dict[str, tuple[np.ndarray, int, int] | None] = {}

        def crop_of(zone_id: str) -> tuple[np.ndarray, int, int] | None:
            if zone_id not in crops:
                zone = zone_by_id.get(zone_id)
                crops[zone_id] = None if zone is None else _crop(image.shape, zone)
            return crops[zone_id]

        def usable(zone_id: str) -> np.ndarray | None:
            zone = zone_by_id.get(zone_id)
            c = crop_of(zone_id)
            if zone is None or c is None or zone.status != "ok":
                return None
            mask, x, y = c
            if int(mask.sum()) < MIN_ZONE_PIXELS:
                return None
            return image[y : y + mask.shape[0], x : x + mask.shape[1]][mask]

        glasses = any(w.code == "glasses_detected" for w in profile.quality.warnings)

        skin_crop = crop_of(SKIN_ZONE)
        skin_zone = zone_by_id.get(SKIN_ZONE)
        if (
            skin_zone is None
            or skin_crop is None
            or skin_zone.status != "ok"
            or int(skin_crop[0].sum()) < MIN_ZONE_PIXELS
        ):
            return empty

        ref_mask = np.zeros(image.shape[:2], dtype=bool)
        sm, sx, sy = skin_crop
        ref_mask[sy : sy + sm.shape[0], sx : sx + sm.shape[1]] = sm
        excluded = {z for row in TARGETS for z in row.zones} | set(REFERENCE_EXCLUDED_ZONES)
        for zid in sorted(excluded):
            c = crop_of(zid)
            if c is None:
                continue
            m, x, y = c
            ref_mask[y : y + m.shape[0], x : x + m.shape[1]] &= ~m

        fallback = False
        if int(ref_mask.sum()) >= MIN_REF_PIXELS:
            skin_hex = _trimmed_hex(image[ref_mask])
        else:
            fallback = True
            skin_hex = _trimmed_hex(image[_full(image, skin_crop)])
        if skin_hex is None:
            return empty
        skin_lab = hex_to_lab(skin_hex)

        low_confidence = degraded or fallback
        components: list[BlueprintComponent] = []
        warnings: list[BlueprintWarning] = []
        if glasses:
            warnings.append(BlueprintWarning(code="glasses_present", message=MSG_GLASSES))

        for row in TARGETS:
            if row.product_type == "eyeshadow" and glasses:
                continue
            used: list[str] = []
            parts: list[np.ndarray] = []
            for zid in row.zones:
                px = usable(zid)
                if px is None:
                    low_confidence = True
                    continue
                used.append(zid)
                parts.append(px)
            if not parts:
                continue
            color_hex = _trimmed_hex(np.concatenate(parts, axis=0))
            if color_hex is None:
                continue
            color_lab = hex_to_lab(color_hex)
            m = _metric(row, color_lab, skin_lab)
            if m is None:
                continue
            span = row.full_threshold - row.present_threshold
            intensity = round(
                min(1.0, max(0.0, 0.2 + 0.8 * (m - row.present_threshold) / span)), 3
            )
            zone_conf = float(
                np.mean(
                    [
                        DEFAULT_ZONE_CONFIDENCE if zone_by_id[z].confidence is None else zone_by_id[z].confidence
                        for z in used
                    ]
                )
            )
            confidence = zone_conf * min(1.0, m / (2 * row.present_threshold))
            if fallback:
                confidence *= FALLBACK_CONFIDENCE_FACTOR
            confidence = round(min(1.0, max(0.0, confidence)), 3)
            ordered = map_zones(row.product_type, None, tuple(used)).zone_ids
            components.append(
                _make(
                    row.category,
                    row.product_type,
                    color_hex,
                    undertone_from_lab(color_lab[1], color_lab[2]),
                    intensity,
                    confidence,
                    ordered,
                )
            )

        if not components:
            warnings.insert(0, BlueprintWarning(code="no_makeup_detected", message=MSG_NO_MAKEUP))
        if low_confidence:
            warnings.append(low)
        overall = (
            round(sum(c.component.confidence or 0.0 for c in components) / len(components), 3)
            if components
            else None
        )
        return CvAnalysis(components, warnings, overall)


def _full(image: np.ndarray, skin_crop: tuple[np.ndarray, int, int]) -> np.ndarray:
    mask, x, y = skin_crop
    full = np.zeros(image.shape[:2], dtype=bool)
    full[y : y + mask.shape[0], x : x + mask.shape[1]] = mask
    return full


decode_zone_mask = _crop
trimmed_median_hex = _trimmed_hex
full_zone_mask = _full


def bare_skin_reference(
    image: np.ndarray,
    crop_of: Callable[[str], tuple[np.ndarray, int, int] | None],
) -> tuple[np.ndarray, bool] | None:
    """Bare-skin reference mask (face_skin minus target and excluded zones) and the fallback flag.

    Mirrors the reference logic in CvReferenceAnalyzer.analyze (analyze itself is untouched).
    """
    skin_crop = crop_of(SKIN_ZONE)
    if skin_crop is None or int(skin_crop[0].sum()) < MIN_ZONE_PIXELS:
        return None
    ref_mask = _full(image, skin_crop)
    excluded = {z for row in TARGETS for z in row.zones} | set(REFERENCE_EXCLUDED_ZONES)
    for zid in sorted(excluded):
        c = crop_of(zid)
        if c is None:
            continue
        m, x, y = c
        ref_mask[y : y + m.shape[0], x : x + m.shape[1]] &= ~m
    if int(ref_mask.sum()) >= MIN_REF_PIXELS:
        return ref_mask, False
    return _full(image, skin_crop), True
