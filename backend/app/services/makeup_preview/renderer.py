"""Render an AdaptedMakeupPlan onto an RGB image in Lab space. Pure; never logs pixel data."""

import math
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedComponent, AdaptedMakeupPlan
from app.schemas.makeup_preview import ComponentRenderReport, SkipReason
from app.services.makeup_preview.color import component_target_lab, lab_to_srgb, srgb_to_lab
from app.services.makeup_preview.finishes import finish_delta_l, resolve_finish
from app.services.makeup_preview.masks import decode_mask_full

_DEFAULT_OPACITY = 0.5
_OPACITY_FLOOR = 0.4
_DEFAULT_L_SHIFT = 0.3
_L_SHIFT_BY_STAGE = MappingProxyType({"lips": 0.7, "eyebrow_eye": 0.5})
_FEATHER_FRACTION = 0.04
_MIN_SIGMA = 0.8
_MAX_SIGMA = 64.0
_ALPHA_FLOOR = 1.0 / 255.0


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray  # uint8 HxWx3
    reports: tuple[ComponentRenderReport, ...]
    alpha_union: np.ndarray  # float32 HxW


def _opacity(component: AdaptedComponent) -> float:
    base = component.component.intensity
    base = _DEFAULT_OPACITY if base is None else min(max(base, 0.0), 1.0)
    # Plan intensity describes how strong the look is in the reference, but any detected
    # product must show clearly, so non-zero intensities are lifted to a visible floor.
    value = 0.0 if base <= 0.0 else _OPACITY_FLOOR + (1.0 - _OPACITY_FLOOR) * base
    for adj in component.adjustments:
        if adj.intensity_multiplier is not None:
            value *= adj.intensity_multiplier
    return min(max(value, 0.0), 1.0)


def _sort_key(c: AdaptedComponent) -> tuple[bool, int, int]:
    return (c.stage_number is None, c.stage_number or 0, c.index)


def render_layers(image_rgb: np.ndarray, plan: AdaptedMakeupPlan, *, seed: int = 0) -> RenderResult:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must be a uint8 HxWx3 array")
    height, width = image_rgb.shape[:2]
    if (width, height) != (plan.image.width, plan.image.height):
        raise ValueError("image shape does not match the plan image size")

    out = image_rgb.copy()
    alpha_union = np.zeros((height, width), dtype=np.float32)
    reports: list[ComponentRenderReport] = []

    for comp in sorted(plan.components, key=_sort_key):
        finish = resolve_finish(comp.component.finish)
        stretch = "deferred" if any(a.direction_angles for a in comp.adjustments) else "not_applicable"
        opacity = _opacity(comp)

        def skip(reason: SkipReason, op: float = opacity) -> None:
            reports.append(
                ComponentRenderReport(
                    index=comp.index, status="skipped", reason=reason, opacity=op,
                    finish_applied=finish, directional_stretch=stretch, pixels_touched=0,
                )
            )

        mask, reason = decode_mask_full(comp, (height, width))
        if mask is None:
            skip(reason or "mask_invalid")
            continue
        target = component_target_lab(comp.component)
        if target is None:
            skip("no_colour")
            continue
        if opacity <= 0.0:
            skip("zero_opacity", 0.0)
            continue

        bbox = comp.bbox
        assert bbox is not None
        cw = min(bbox.x + bbox.width, width) - max(bbox.x, 0)
        ch = min(bbox.y + bbox.height, height) - max(bbox.y, 0)
        if cw <= 0 or ch <= 0:
            skip("empty_mask")
            continue
        sigma = min(_MAX_SIGMA, max(_MIN_SIGMA, _FEATHER_FRACTION * max(cw, ch)))
        margin = int(math.ceil(3.0 * sigma))
        x0, y0 = max(bbox.x - margin, 0), max(bbox.y - margin, 0)
        x1, y1 = min(bbox.x + bbox.width + margin, width), min(bbox.y + bbox.height + margin, height)
        if x1 <= x0 or y1 <= y0:
            skip("empty_mask")
            continue

        m_roi = mask[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
        alpha = cv2.GaussianBlur(m_roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha[alpha < _ALPHA_FLOOR] = 0.0
        alpha = alpha.astype(np.float32)
        active = alpha > 0.0
        if not active.any():
            skip("empty_mask")
            continue

        roi = out[y0:y1, x0:x1]
        lab = srgb_to_lab(roi)
        weight = (alpha * np.float32(opacity)).astype(np.float32)
        new = lab.copy()
        new[..., 0] = lab[..., 0] + np.float32(_L_SHIFT_BY_STAGE.get(comp.stage_key or "", _DEFAULT_L_SHIFT)) * (np.float32(target[0]) - lab[..., 0]) * weight
        new[..., 1] = lab[..., 1] + (np.float32(target[1]) - lab[..., 1]) * weight
        new[..., 2] = lab[..., 2] + (np.float32(target[2]) - lab[..., 2]) * weight

        if comp.centroid is not None:
            centroid_local = (comp.centroid[0] - x0, comp.centroid[1] - y0)
        else:
            centroid_local = ((x1 - x0) / 2.0, (y1 - y0) / 2.0)
        rng = np.random.default_rng((int(seed) + comp.index) & 0xFFFFFFFF)
        new[..., 0] += finish_delta_l(finish, alpha, weight, rng, centroid_local)
        new[..., 0] = np.clip(new[..., 0], 0.0, 100.0)

        rgb8 = np.rint(lab_to_srgb(new) * np.float32(255.0)).astype(np.uint8)
        result = np.where(active[..., None], rgb8, roi)
        touched = int(np.count_nonzero(np.any(result != roi, axis=2)))
        out[y0:y1, x0:x1] = result
        np.maximum(alpha_union[y0:y1, x0:x1], alpha, out=alpha_union[y0:y1, x0:x1])

        reports.append(
            ComponentRenderReport(
                index=comp.index, status="rendered", reason=None, opacity=opacity,
                finish_applied=finish, directional_stretch=stretch, pixels_touched=touched,
            )
        )

    return RenderResult(image=out, reports=tuple(reports), alpha_union=alpha_union)
