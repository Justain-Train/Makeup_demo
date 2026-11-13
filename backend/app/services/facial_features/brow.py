"""Brow arch and thickness from landmarks in the roll-removed face frame."""

import math

import numpy as np

from app.schemas.facial_profile import Classified
from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import thresholds as t
from app.services.facial_features._common import agreement, band, guarded, project, unknown
from app.services.facial_features.eye import eye_geometry

# Subject's right brow, tail (outer) to head (inner), as listed in application_zones.frame
# MIRROR_PAIRS. The left brow is derived through FaceFrame.indices.
BROW_UPPER = [70, 63, 105, 66, 107]
BROW_LOWER = [46, 53, 52, 65, 55]


def _arch(pts: np.ndarray, frame: FaceFrame, side: str) -> float:
    chain = np.array([project(pts, frame, i) for i in frame.indices(BROW_UPPER, side)])
    head, tail = chain[-1], chain[0]
    d = tail - head
    length = float(math.hypot(d[0], d[1]))
    if not math.isfinite(length) or length < t.MIN_LENGTH_PX:
        raise ValueError("degenerate brow")
    normal = np.array([-d[1], d[0]]) / length
    if normal[1] > 0:  # make the normal point up the face (smaller v)
        normal = -normal
    rise = max(0.0, float(np.max((chain - head) @ normal)))
    return rise / eye_geometry(pts, frame, side).width


def _thickness(pts: np.ndarray, frame: FaceFrame, side: str) -> float:
    upper = np.array([project(pts, frame, i) for i in frame.indices(BROW_UPPER, side)])
    lower = np.array([project(pts, frame, i) for i in frame.indices(BROW_LOWER, side)])
    mean_gap = float(np.mean(np.hypot(*(upper - lower).T)))
    return mean_gap / eye_geometry(pts, frame, side).width


@guarded
def classify_brow_arch(pts: np.ndarray, frame: FaceFrame) -> Classified:
    r, l = _arch(pts, frame, "right"), _arch(pts, frame, "left")
    mean = (r + l) / 2.0
    return band(
        mean, t.BROW_ARCH_EDGES, ("straight", "soft_arched", "high_arched"), t.BROW_ARCH_MARGIN,
        t.BROW_ARCH_SCALE, {"arch_ratio": mean, "right_arch_ratio": r, "left_arch_ratio": l},
        agreement(r, l, t.BROW_DISAGREE_SCALE),
    )


@guarded
def classify_brow_thickness(pts: np.ndarray, frame: FaceFrame) -> Classified:
    r, l = _thickness(pts, frame, "right"), _thickness(pts, frame, "left")
    mean = (r + l) / 2.0
    return unknown(
        "not_measurable_from_landmarks",
        {"thickness_ratio": mean, "right_thickness_ratio": r, "left_thickness_ratio": l},
    )
