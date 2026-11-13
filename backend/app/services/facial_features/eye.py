"""Eye spacing, tilt and shape from landmarks in the roll-removed face frame."""

import math
from dataclasses import dataclass

import numpy as np

from app.schemas.facial_profile import Classified
from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import thresholds as t
from app.services.facial_features._common import agreement, band, guarded, project

# Subject's right eye (MediaPipe canonical): inner canthus 133, outer canthus 33,
# eyelid top 159, bottom 145. The left eye is derived through FaceFrame.indices (362/263/386/374).
EYE_INNER, EYE_OUTER, EYE_TOP, EYE_BOTTOM = 133, 33, 159, 145


@dataclass(frozen=True)
class EyeGeometry:
    inner: np.ndarray  # (u, v) of the inner canthus
    width: float
    tilt_deg: float  # outer corner higher than inner is positive, for both eyes
    aperture_ratio: float


def eye_geometry(pts: np.ndarray, frame: FaceFrame, side: str) -> EyeGeometry:
    inner_i, outer_i, top_i, bottom_i = frame.indices([EYE_INNER, EYE_OUTER, EYE_TOP, EYE_BOTTOM], side)
    inner, outer = project(pts, frame, inner_i), project(pts, frame, outer_i)
    top, bottom = project(pts, frame, top_i), project(pts, frame, bottom_i)
    d = outer - inner
    width = float(math.hypot(d[0], d[1]))
    if not math.isfinite(width) or width < t.MIN_LENGTH_PX:
        raise ValueError("degenerate eye")
    # u runs right eye -> left eye, so outward is -u for the right eye and +u for the left.
    outward = -1.0 if side == "right" else 1.0
    run = float(d[0]) * outward
    rise = float(inner[1] - outer[1])  # v grows downward, so smaller v is higher
    tilt = math.degrees(math.atan2(rise, run))
    aperture = abs(float(bottom[1] - top[1]))
    return EyeGeometry(inner=inner, width=width, tilt_deg=tilt, aperture_ratio=aperture / width)


@guarded
def classify_eye_spacing(pts: np.ndarray, frame: FaceFrame) -> Classified:
    r, l = eye_geometry(pts, frame, "right"), eye_geometry(pts, frame, "left")
    d = l.inner - r.inner
    inner_distance = float(math.hypot(d[0], d[1]))
    mean_width = (r.width + l.width) / 2.0
    ratio = inner_distance / mean_width
    return band(
        ratio, t.EYE_SPACING_EDGES, ("close", "average", "wide"), t.EYE_SPACING_MARGIN, t.EYE_SPACING_SCALE,
        {"spacing_ratio": ratio, "inner_distance_px": inner_distance, "mean_eye_width_px": mean_width},
        agreement(r.width, l.width, t.EYE_WIDTH_ASYMMETRY_SCALE * mean_width),
    )


@guarded
def classify_eye_tilt(pts: np.ndarray, frame: FaceFrame) -> Classified:
    r, l = eye_geometry(pts, frame, "right"), eye_geometry(pts, frame, "left")
    mean_tilt = (r.tilt_deg + l.tilt_deg) / 2.0
    return band(
        mean_tilt, t.EYE_TILT_EDGES_DEG, ("downturned", "neutral", "upturned"), t.EYE_TILT_MARGIN_DEG,
        t.EYE_TILT_SCALE_DEG,
        {"tilt_deg": mean_tilt, "right_tilt_deg": r.tilt_deg, "left_tilt_deg": l.tilt_deg},
        agreement(r.tilt_deg, l.tilt_deg, t.EYE_TILT_DISAGREE_SCALE_DEG),
    )


@guarded
def classify_eye_shape(pts: np.ndarray, frame: FaceFrame) -> Classified:
    r, l = eye_geometry(pts, frame, "right"), eye_geometry(pts, frame, "left")
    mean_ratio = (r.aperture_ratio + l.aperture_ratio) / 2.0
    return band(
        mean_ratio, t.EYE_SHAPE_EDGES, ("almond", "round"), t.EYE_SHAPE_MARGIN, t.EYE_SHAPE_SCALE,
        {"aperture_ratio": mean_ratio, "right_aperture_ratio": r.aperture_ratio, "left_aperture_ratio": l.aperture_ratio},
        agreement(r.aperture_ratio, l.aperture_ratio, t.EYE_SHAPE_DISAGREE_SCALE),
    )
