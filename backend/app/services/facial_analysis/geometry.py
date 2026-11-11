"""Pure geometry helpers over landmark data. No MediaPipe, no classification.

Input is a FacialLandmarkResult; output is plain pixel measurements. Later
feature-estimation code will build on this module.

Index reference (subject's left/right; MediaPipe face mesh topology):
  33  / 133  right eye outer / inner corner
  263 / 362  left eye outer / inner corner
  468 / 473  right / left iris center (only in the 478-point model)
  61  / 291  mouth corners
  234 / 454  right / left cheek edge (face width)
  10  / 152  forehead top / chin bottom (face height)
"""

import math

from app.schemas.face_measurements import BoundingBox, FaceMeasurements
from app.schemas.facial_landmarks import FacialLandmarkResult

RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_OUTER, LEFT_EYE_INNER = 263, 362
RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER = 468, 473
MOUTH_RIGHT, MOUTH_LEFT = 61, 291
FACE_RIGHT_EDGE, FACE_LEFT_EDGE = 234, 454
FOREHEAD_TOP, CHIN_BOTTOM = 10, 152


def landmark_pixels(result: FacialLandmarkResult) -> dict[int, tuple[float, float]]:
    """Landmark index -> (x, y) in image pixels (post-EXIF space)."""
    # Convert normalized coords to pixels before measuring. Distances in
    # normalized space would be distorted on non-square images.
    w, h = result.image.width, result.image.height
    return {lm.index: (lm.x * w, lm.y * h) for lm in result.landmarks}


_to_pixels = landmark_pixels  # backwards-compatible private alias


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def compute_basic_measurements(result: FacialLandmarkResult) -> FaceMeasurements | None:
    """Return basic pixel measurements, or None if no face was detected."""
    if not result.face_detected or not result.landmarks:
        return None
    p = _to_pixels(result)

    # Use iris centers if available; otherwise use the midpoint between the
    # eye corners as the pupil position.
    if RIGHT_IRIS_CENTER in p and LEFT_IRIS_CENTER in p:
        right_pupil, left_pupil = p[RIGHT_IRIS_CENTER], p[LEFT_IRIS_CENTER]
    else:
        right_pupil = _midpoint(p[RIGHT_EYE_OUTER], p[RIGHT_EYE_INNER])
        left_pupil = _midpoint(p[LEFT_EYE_OUTER], p[LEFT_EYE_INNER])

    xs = [pt[0] for pt in p.values()]
    ys = [pt[1] for pt in p.values()]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)

    return FaceMeasurements(
        face_bounding_box=BoundingBox(
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            width=x_max - x_min, height=y_max - y_min,
        ),
        interpupillary_distance_px=_dist(left_pupil, right_pupil),
        inner_eye_corner_distance_px=_dist(p[LEFT_EYE_INNER], p[RIGHT_EYE_INNER]),
        left_eye_width_px=_dist(p[LEFT_EYE_OUTER], p[LEFT_EYE_INNER]),
        right_eye_width_px=_dist(p[RIGHT_EYE_OUTER], p[RIGHT_EYE_INNER]),
        mouth_width_px=_dist(p[MOUTH_LEFT], p[MOUTH_RIGHT]),
        face_width_px=_dist(p[FACE_LEFT_EDGE], p[FACE_RIGHT_EDGE]),
        face_height_px=_dist(p[FOREHEAD_TOP], p[CHIN_BOTTOM]),
    )
