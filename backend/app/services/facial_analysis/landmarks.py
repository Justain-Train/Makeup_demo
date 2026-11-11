"""MediaPipe Face Landmarker wrapper.

This is the only module that imports MediaPipe for landmarking. Everything
else works with the Pydantic `FacialLandmarkResult`, so the backend model can
be swapped later without touching routes or downstream features.

Uses the MediaPipe *Tasks* API (`mediapipe.tasks.python.vision.FaceLandmarker`).
The legacy `mp.solutions.face_mesh` API is deprecated and was removed in
mediapipe 1.x.
"""

import logging
import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from app.schemas.facial_landmarks import (
    DetectionThresholds,
    FaceMeshTopology,
    FacialLandmarkResult,
    ImageSize,
    Landmark,
)


logger = logging.getLogger(__name__)

# 468 mesh points + 10 iris points. Downstream code indexes 468 and 473 directly.
EXPECTED_LANDMARKS = 478


class ModelNotFoundError(FileNotFoundError):
    pass


MULTIPLE_FACES_MESSAGE = "Please upload a photo with only one face."
INCOMPLETE_LANDMARKS_MESSAGE = (
    "Face landmarks were incomplete; please retake the photo with your whole face visible."
)


def select_primary_face(areas: list[float], secondary_ratio: float) -> tuple[int, bool]:
    """Pick the main face from per-face bounding-box areas (any units).

    Returns (index of the largest face, has_significant_second_face). A second
    face is significant when its area >= secondary_ratio * the largest area.
    Never assumes MediaPipe returns the main face first.
    """
    primary = max(range(len(areas)), key=lambda i: areas[i])
    significant = any(
        i != primary and areas[i] >= secondary_ratio * areas[primary] for i in range(len(areas))
    )
    return primary, significant


class FaceLandmarkService:
    """Detects facial landmarks on a still image; analyzes the largest face."""

    def __init__(
        self,
        model_path: Path,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        secondary_face_area_ratio: float = 0.20,
    ) -> None:
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Face Landmarker model not found at {model_path}. "
                "Run `python scripts/download_model.py` from backend/."
            )
        self._secondary_ratio = secondary_face_area_ratio
        self._thresholds = DetectionThresholds(
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            # IMAGE mode: each call is independent (no tracking between frames).
            # Live AR later would use VIDEO / LIVE_STREAM mode instead.
            running_mode=vision.RunningMode.IMAGE,
            # 2 is enough to tell "one face" from "more than one": the largest
            # face is analyzed, and a second face of comparable size is rejected.
            num_faces=2,
            # Confidence for the first-stage face *detector* (where the face is).
            min_face_detection_confidence=min_face_detection_confidence,
            # Confidence that the landmark model's output is really a face.
            min_face_presence_confidence=min_face_presence_confidence,
            # Not needed yet. Blendshapes are expression scores; the matrix is
            # head pose. Both can be switched on later without schema breakage.
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        # The landmarker holds a native graph. It is costly to create, so we
        # create it once at startup and reuse it.
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        # A landmarker instance is not safe to call from several threads at
        # once, and FastAPI runs sync routes in a thread pool.
        self._lock = threading.Lock()

    def detect_landmarks(self, image: np.ndarray) -> FacialLandmarkResult:
        """Run the landmarker on an (H, W, 3) uint8 RGB array."""
        height, width = image.shape[:2]
        # mp.Image wraps the pixel buffer. SRGB = 3-channel RGB, uint8.
        # The array must be C-contiguous.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
        )
        with self._lock:
            result = self._landmarker.detect(mp_image)

        base = dict(image=ImageSize(width=width, height=height), thresholds=self._thresholds)

        # result.face_landmarks: one list per detected face, each holding 478
        # NormalizedLandmark objects (468 mesh points + 10 iris points). The
        # list position is the canonical mesh index; the topology relies on it.
        if not result.face_landmarks:
            return FacialLandmarkResult(
                face_detected=False,
                status="no_face_detected",
                message="No face found. Try a well-lit, front-facing photo.",
                face_count=0,
                landmark_count=0,
                **base,
            )

        faces = result.face_landmarks
        # Pixel-space bbox areas, so non-square images are compared fairly.
        areas = []
        for f in faces:
            if not f:  # a degenerate empty face: smallest possible area, handled below
                areas.append(0.0)
                continue
            xs = [lm.x * width for lm in f]
            ys = [lm.y * height for lm in f]
            areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
        primary, multiple = select_primary_face(areas, self._secondary_ratio)

        if multiple:
            return FacialLandmarkResult(
                face_detected=True,
                status="multiple_faces_detected",
                message=MULTIPLE_FACES_MESSAGE,
                face_count=len(faces),
                landmark_count=0,
                **base,
            )

        face = faces[primary]
        if len(face) < EXPECTED_LANDMARKS:
            # Downstream code indexes the iris points (468, 473). Report a status instead of
            # letting an IndexError become a generic 500. Count only: no coordinates are logged.
            logger.warning("Landmarker returned an incomplete face (%d points)", len(face))
            return FacialLandmarkResult(
                face_detected=False,
                status="no_face_detected",
                message=INCOMPLETE_LANDMARKS_MESSAGE,
                face_count=0,
                landmark_count=0,
                **base,
            )
        landmarks = [
            Landmark(index=i, x=float(lm.x), y=float(lm.y), z=float(lm.z))
            for i, lm in enumerate(face)
        ]
        return FacialLandmarkResult(
            face_detected=True,
            status="ok",
            face_count=len(faces),
            landmark_count=len(landmarks),
            landmarks=landmarks,
            **base,
        )

    def close(self) -> None:
        self._landmarker.close()


def get_face_mesh_topology() -> FaceMeshTopology:
    """MediaPipe's canonical connections, for drawing the mesh/contours.

    'left'/'right' are the subject's own left/right.
    """
    C = vision.FaceLandmarksConnections

    def pairs(conns) -> list[tuple[int, int]]:
        return [(c.start, c.end) for c in conns]

    return FaceMeshTopology(
        tesselation=pairs(C.FACE_LANDMARKS_TESSELATION),
        contours={
            "face_oval": pairs(C.FACE_LANDMARKS_FACE_OVAL),
            "lips": pairs(C.FACE_LANDMARKS_LIPS),
            "left_eye": pairs(C.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": pairs(C.FACE_LANDMARKS_RIGHT_EYE),
            "left_eyebrow": pairs(C.FACE_LANDMARKS_LEFT_EYEBROW),
            "right_eyebrow": pairs(C.FACE_LANDMARKS_RIGHT_EYEBROW),
            "left_iris": pairs(C.FACE_LANDMARKS_LEFT_IRIS),
            "right_iris": pairs(C.FACE_LANDMARKS_RIGHT_IRIS),
            "nose": pairs(C.FACE_LANDMARKS_NOSE),
        },
    )
