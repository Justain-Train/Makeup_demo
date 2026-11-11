"""Basic geometry derived from landmarks. Measurements only, no classification.

"left"/"right" are the *subject's* left/right (MediaPipe convention). In an
unmirrored selfie, the subject's left eye appears on the RIGHT side of the image.
"""

from pydantic import BaseModel

from app.schemas.facial_landmarks import FacialLandmarkResult


class BoundingBox(BaseModel):
    """Pixel coordinates, origin top-left."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class FaceMeasurements(BaseModel):
    """All distances are in pixels of the uploaded image (2D, ignores z)."""

    face_bounding_box: BoundingBox
    interpupillary_distance_px: float
    inner_eye_corner_distance_px: float
    left_eye_width_px: float
    right_eye_width_px: float
    mouth_width_px: float
    face_width_px: float
    face_height_px: float


class FacialLandmarkResponse(FacialLandmarkResult):
    """API response: raw landmark result + separately computed measurements."""

    measurements: FaceMeasurements | None = None
