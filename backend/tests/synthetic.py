"""Test helpers: landmarks from the real fixture, a synthetic parser, fake services."""

import json
from pathlib import Path

import numpy as np

from app.schemas.facial_landmarks import (
    DetectionThresholds,
    FacialLandmarkResult,
    ImageSize,
    Landmark,
)
from app.services.application_zones import derive as dz
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import (
    ParseResult,
    ParserInfo,
    ParseWindow,
    compute_parse_window,
    covered_region,
)
from app.services.segmentation.regions import Region

FIXTURES = Path(__file__).resolve().parent / "fixtures"
THRESHOLDS = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)


def fixture_landmarks() -> tuple[np.ndarray, int, int]:
    """(478, 2) landmark pixels of portrait.jpg, plus its width and height."""
    d = json.loads((FIXTURES / "portrait_landmarks.json").read_text())
    w, h = d["image"]["width"], d["image"]["height"]
    return np.array(d["landmarks"], dtype=np.float64) * [w, h], w, h


def landmark_result(status: str = "ok", width: int = 820, height: int = 1024, face_count: int = 1) -> FacialLandmarkResult:
    if status == "ok":
        d = json.loads((FIXTURES / "portrait_landmarks.json").read_text())
        lms = [Landmark(index=i, x=x, y=y, z=0.0) for i, (x, y) in enumerate(d["landmarks"])]
    else:
        lms = []
    return FacialLandmarkResult(
        face_detected=status != "no_face_detected",
        status=status,  # type: ignore[arg-type]
        message={"ok": None, "multiple_faces_detected": "Please upload a photo with only one face."}.get(
            status, "No face found."
        ),
        face_count=face_count if status != "no_face_detected" else 0,
        image=ImageSize(width=width, height=height),
        thresholds=THRESHOLDS,
        landmark_count=len(lms),
        landmarks=lms,
    )


class FakeLandmarkService:
    def __init__(self, status: str = "ok", face_count: int = 1):
        self.status, self.face_count = status, face_count

    def detect_landmarks(self, image: np.ndarray) -> FacialLandmarkResult:
        h, w = image.shape[:2]
        return landmark_result(self.status, w, h, self.face_count)


def synthetic_labels(pts: np.ndarray, shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    """A plausible label map built from landmarks: skin everywhere, features on top."""
    labels = np.full(shape, int(Region.SKIN), dtype=np.uint8)
    o = np.array(origin, dtype=np.float64)

    def paint(poly: np.ndarray, region: Region) -> None:
        labels[mk.rasterize_polygon(poly - o, shape)] = int(region)

    paint(pts[dz.NOSE_HULL], Region.NOSE)
    lm = FaceFrame.from_landmarks(pts)
    for side, eye, brow in (("right", Region.EYE_R, Region.BROW_R), ("left", Region.EYE_L, Region.BROW_L)):
        paint(lm.pts[lm.indices(dz.EYE_UPPER + dz.EYE_LOWER[::-1][1:-1], side)], eye)
        paint(lm.pts[lm.indices(dz.BROW_UPPER + dz.BROW_LOWER[::-1], side)], brow)
    paint(np.vstack([pts[dz.LIP_OUTER_UPPER], pts[dz.LIP_INNER_UPPER[::-1]]]), Region.LIP_UPPER)
    paint(np.vstack([pts[dz.LIP_OUTER_LOWER], pts[dz.LIP_INNER_LOWER[::-1]]]), Region.LIP_LOWER)
    return labels


class FakeParser:
    """Parser stand-in that derives its label map from the landmarks (API tests)."""

    info = ParserInfo(id="fake", architecture="fake", weights_sha256="0" * 64, input_size=512)

    def __init__(self, error: Exception | None = None):
        self.error = error

    def parse(self, image: np.ndarray, face_bbox: tuple[float, float, float, float]) -> ParseResult:
        if self.error:
            raise self.error
        pts, _, _ = fixture_landmarks()
        h, w = image.shape[:2]
        pts = pts / [820, 1024] * [w, h]
        window = compute_parse_window(face_bbox, 2.0)
        x0, y0, x1, y1 = covered_region(window, w, h)
        labels = synthetic_labels(pts, (y1 - y0, x1 - x0), (x0, y0))
        return ParseResult(
            labels=labels,
            confidence=np.full(labels.shape, 0.9, dtype=np.float32),
            window=ParseWindow(window.x, window.y, window.size),
            origin=(x0, y0),
            timings_ms={"preprocess": 1.0, "inference": 1.0, "postprocess": 1.0},
        )
