"""Lip fullness, balance and width from landmarks in the roll-removed face frame."""

import math

import numpy as np

from app.schemas.facial_profile import Classified
from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import thresholds as t
from app.services.facial_features._common import band, guarded, project

# MediaPipe canonical: 0 upper outer, 13 upper inner, 14 lower inner, 17 lower outer,
# 61 / 291 mouth corners (subject's right / left).
UPPER_OUTER, UPPER_INNER, LOWER_INNER, LOWER_OUTER = 0, 13, 14, 17
MOUTH_RIGHT, MOUTH_LEFT = 61, 291


def _thicknesses(pts: np.ndarray, frame: FaceFrame) -> tuple[float, float]:
    v = {i: float(project(pts, frame, i)[1]) for i in (UPPER_OUTER, UPPER_INNER, LOWER_INNER, LOWER_OUTER)}
    return v[UPPER_INNER] - v[UPPER_OUTER], v[LOWER_OUTER] - v[LOWER_INNER]


@guarded
def classify_lip_fullness(pts: np.ndarray, frame: FaceFrame) -> Classified:
    # Upper + lower thickness equals the 0->17 height when closed and ignores mouth opening.
    upper, lower = _thicknesses(pts, frame)
    if upper < t.MIN_LENGTH_PX or lower < t.MIN_LENGTH_PX:
        raise ValueError("degenerate lips")
    ratio = (upper + lower) / frame.ipd
    return band(
        ratio, t.LIP_FULLNESS_EDGES, ("thin", "medium", "full"), t.LIP_FULLNESS_MARGIN, t.LIP_FULLNESS_SCALE,
        {"fullness_ipd": ratio, "upper_px": upper, "lower_px": lower},
    )


@guarded
def classify_lip_balance(pts: np.ndarray, frame: FaceFrame) -> Classified:
    upper, lower = _thicknesses(pts, frame)
    if upper < t.MIN_LENGTH_PX or lower < t.MIN_LENGTH_PX:
        raise ValueError("degenerate lips")
    ratio = upper / lower
    return band(
        ratio, t.LIP_BALANCE_EDGES, ("bottom_heavy", "balanced", "top_heavy"), t.LIP_BALANCE_MARGIN,
        t.LIP_BALANCE_SCALE, {"upper_to_lower": ratio, "upper_px": upper, "lower_px": lower},
    )


@guarded
def classify_lip_width(pts: np.ndarray, frame: FaceFrame) -> Classified:
    d = project(pts, frame, MOUTH_LEFT) - project(pts, frame, MOUTH_RIGHT)
    width = float(math.hypot(d[0], d[1]))
    if not math.isfinite(width) or width < t.MIN_LENGTH_PX:
        raise ValueError("degenerate mouth")
    ratio = width / frame.ipd
    return band(
        ratio, t.LIP_WIDTH_EDGES, ("narrow", "average", "wide"), t.LIP_WIDTH_MARGIN, t.LIP_WIDTH_SCALE,
        {"mouth_width_ipd": ratio, "mouth_width_px": width},
    )
