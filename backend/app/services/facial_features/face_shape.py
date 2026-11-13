"""Face shape as the nearest prototype over four width/length ratios in the roll-removed face frame."""

import math

import numpy as np

from app.schemas.facial_profile import Classified
from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import thresholds as t
from app.services.facial_features._common import clamp01, guarded, project, unknown

LABEL_ORDER = tuple(sorted(t.FACE_SHAPE_PROTOTYPES))

# MediaPipe canonical (subject's right, left).
CHEEK = (234, 454)
FOREHEAD = (54, 284)
JAW = (172, 397)
CHIN_SIDE = (176, 400)
CHIN, FOREHEAD_TOP, NOSE_TIP = 152, 10, 1


def _width(pts: np.ndarray, frame: FaceFrame, pair: tuple[int, int]) -> float:
    return abs(float(project(pts, frame, pair[1])[0] - project(pts, frame, pair[0])[0]))


def _asym(pts: np.ndarray, frame: FaceFrame, pair: tuple[int, int], nose_u: float) -> float:
    d_right = abs(float(project(pts, frame, pair[0])[0]) - nose_u)
    d_left = abs(float(project(pts, frame, pair[1])[0]) - nose_u)
    total = d_right + d_left
    if total < t.MIN_LENGTH_PX:
        raise ValueError("degenerate yaw reference")
    return abs(d_right - d_left) / total


def _jaw_angle_deg(pts: np.ndarray, frame: FaceFrame) -> float:
    angles = []
    for k in (0, 1):
        apex = project(pts, frame, JAW[k])
        a = project(pts, frame, CHEEK[k]) - apex
        b = project(pts, frame, CHIN) - apex
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < t.MIN_LENGTH_PX or nb < t.MIN_LENGTH_PX:
            raise ValueError("degenerate jaw")
        cos = float(np.clip(a @ b / (na * nb), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos)))
    return sum(angles) / 2.0


@guarded
def classify_face_shape(pts: np.ndarray, frame: FaceFrame) -> Classified:
    cheek = _width(pts, frame, CHEEK)
    forehead = _width(pts, frame, FOREHEAD)
    jaw = _width(pts, frame, JAW)
    chin_side = _width(pts, frame, CHIN_SIDE)
    length = float(project(pts, frame, CHIN)[1] - project(pts, frame, FOREHEAD_TOP)[1])
    if min(cheek, jaw, length) < t.MIN_LENGTH_PX or not all(
        math.isfinite(x) for x in (cheek, forehead, jaw, chin_side, length)
    ):
        raise ValueError("degenerate face")

    nose_u = float(project(pts, frame, NOSE_TIP)[0])
    yaw = max(_asym(pts, frame, p, nose_u) for p in (CHEEK, JAW))
    vec = (length / cheek, forehead / cheek, jaw / cheek, chin_side / jaw)

    dists = {
        name: math.sqrt(
            sum(((x - p) / s) ** 2 for x, p, s in zip(vec, t.FACE_SHAPE_PROTOTYPES[name], t.FACE_SHAPE_DIM_SCALE))
        )
        for name in LABEL_ORDER
    }
    ranked = sorted(LABEL_ORDER, key=lambda n: dists[n])
    best, runner = ranked[0], ranked[1]
    d1, d2 = dists[best], dists[runner]
    gap = d2 - d1

    measurements = {
        "length_to_width": vec[0],
        "forehead_to_cheek": vec[1],
        "jaw_to_cheek": vec[2],
        "chin_taper": vec[3],
        "jaw_angle_deg": _jaw_angle_deg(pts, frame),
        "yaw_asymmetry": yaw,
        "best_distance": d1,
        "runner_up_distance": d2,
        "runner_up_index": float(LABEL_ORDER.index(runner)),
        "cheek_width_px": cheek,
    }
    if not all(math.isfinite(v) for v in measurements.values()):
        raise ValueError("non-finite measurement")

    if yaw > t.FACE_SHAPE_YAW_UNKNOWN:
        return unknown("head_turned", measurements)
    yaw_factor = clamp01((t.FACE_SHAPE_YAW_UNKNOWN - yaw) / (t.FACE_SHAPE_YAW_UNKNOWN - t.FACE_SHAPE_YAW_SOFT))
    confidence = min(
        t.FACE_SHAPE_CONF_CAP,
        clamp01(gap / t.FACE_SHAPE_GAP_SCALE) * clamp01(1.0 - d1 / t.FACE_SHAPE_DIST_SCALE) * yaw_factor,
    )
    if gap < t.FACE_SHAPE_MARGIN:
        return Classified(
            label="unknown", confidence=confidence, measurements=measurements, reason="within_ambiguity_margin"
        )
    return Classified(label=best, confidence=confidence, measurements=measurements)
