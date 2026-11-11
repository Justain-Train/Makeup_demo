"""Pydantic schemas for raw facial landmark output.

These describe *what MediaPipe found*, with no interpretation. Derived values
(distances, boxes, ...) live in `face_measurements.py`.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Landmark(BaseModel):
    """One MediaPipe face mesh point.

    x, y are normalized to [0, 1] by image width/height (origin = top-left).
    They can fall slightly outside [0, 1] if the face is cut off by the frame.
    z is relative depth with the head center roughly at 0; smaller = closer to
    the camera. It uses roughly the same scale as x (image width).
    """

    index: int = Field(description="MediaPipe face mesh index (0-477)")
    x: float
    y: float
    z: float


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionThresholds(BaseModel):
    """Thresholds MediaPipe used. Face Landmarker gives no per-face confidence
    score: a face is either above these thresholds (returned) or not."""

    min_face_detection_confidence: float
    min_face_presence_confidence: float


class FacialLandmarkResult(BaseModel):
    """Output of FaceLandmarkService.detect_landmarks()."""

    face_detected: bool
    status: Literal["ok", "no_face_detected", "multiple_faces_detected"]
    message: str | None = None
    face_count: int = Field(
        default=0,
        description="Faces found by MediaPipe (0-2; the landmarker looks for at most 2).",
    )
    image: ImageSize
    thresholds: DetectionThresholds
    landmark_count: int
    landmarks: list[Landmark] = Field(
        default_factory=list,
        description="478 points: 0-467 face mesh, 468-477 irises. Empty unless status is ok.",
    )


class FaceMeshTopology(BaseModel):
    """Index pairs to connect for drawing. Taken from MediaPipe itself."""

    tesselation: list[tuple[int, int]]
    contours: dict[str, list[tuple[int, int]]]
