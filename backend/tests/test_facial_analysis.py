"""Run from backend/:  pytest

Unit tests use a fake service (no model needed). The integration test runs
real MediaPipe and is skipped if the model or sample image is missing.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.facial_analysis import get_landmark_service
from app.config import get_settings
from app.main import create_app
from app.schemas.facial_landmarks import (
    DetectionThresholds,
    FacialLandmarkResult,
    ImageSize,
    Landmark,
)
from app.services.facial_analysis.geometry import compute_basic_measurements

BACKEND = Path(__file__).resolve().parent.parent
SAMPLE = BACKEND / "tests" / "fixtures" / "portrait.jpg"
THRESHOLDS = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)


def _png_bytes(w: int = 100, h: int = 50) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


class FakeService:
    def __init__(self, detected: bool):
        self.detected = detected

    def detect_landmarks(self, image: np.ndarray) -> FacialLandmarkResult:
        h, w = image.shape[:2]
        lms = [Landmark(index=i, x=0.5, y=0.5, z=0.0) for i in range(478)] if self.detected else []
        return FacialLandmarkResult(
            face_detected=self.detected,
            status="ok" if self.detected else "no_face_detected",
            image=ImageSize(width=w, height=h),
            thresholds=THRESHOLDS,
            landmark_count=len(lms),
            landmarks=lms,
        )


def _client(service) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: service
    return TestClient(app)


def test_rejects_non_image_content_type():
    r = _client(FakeService(True)).post(
        "/api/facial-analysis/landmarks", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert r.status_code == 415


def test_rejects_corrupt_image():
    r = _client(FakeService(True)).post(
        "/api/facial-analysis/landmarks", files={"file": ("a.png", b"not a png", "image/png")}
    )
    assert r.status_code == 400


def test_rejects_oversized_upload():
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: FakeService(True)
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(
        update={"max_upload_mb": 0.00001}
    )
    r = TestClient(app).post(
        "/api/facial-analysis/landmarks", files={"file": ("a.png", _png_bytes(), "image/png")}
    )
    assert r.status_code == 413


def test_no_face_returns_200_with_flag():
    r = _client(FakeService(False)).post(
        "/api/facial-analysis/landmarks", files={"file": ("a.png", _png_bytes(), "image/png")}
    )
    body = r.json()
    assert r.status_code == 200
    assert body["face_detected"] is False
    assert body["status"] == "no_face_detected"
    assert body["landmarks"] == [] and body["measurements"] is None


def test_face_response_shape():
    r = _client(FakeService(True)).post(
        "/api/facial-analysis/landmarks", files={"file": ("a.png", _png_bytes(), "image/png")}
    )
    body = r.json()
    assert body["image"] == {"width": 100, "height": 50}
    assert body["landmark_count"] == 478
    assert body["landmarks"][5] == {"index": 5, "x": 0.5, "y": 0.5, "z": 0.0}
    assert body["measurements"] is not None


def test_geometry_uses_pixel_space():
    # Non-square image: normalized distances would be wrong, pixel ones are not.
    lms = {i: (0.5, 0.5) for i in range(478)}
    lms.update({61: (0.25, 0.5), 291: (0.75, 0.5)})  # mouth corners
    lms.update({468: (0.4, 0.4), 473: (0.6, 0.4)})  # iris centers
    result = FacialLandmarkResult(
        face_detected=True, status="ok",
        image=ImageSize(width=400, height=200), thresholds=THRESHOLDS,
        landmark_count=478,
        landmarks=[Landmark(index=i, x=x, y=y, z=0) for i, (x, y) in lms.items()],
    )
    m = compute_basic_measurements(result)
    assert m.mouth_width_px == pytest.approx(200)
    assert m.interpupillary_distance_px == pytest.approx(80)
    assert m.face_bounding_box.x_min == pytest.approx(100)
    assert m.face_bounding_box.y_min == pytest.approx(80)


def test_exif_orientation_is_applied():
    # Phone-style JPEG: pixels stored 200x100 (landscape) with EXIF
    # Orientation=6 ("rotate 90° CW to display"). Browsers show it 100x200,
    # so the backend must report the same size or landmarks won't line up.
    from app.services.image_loading import load_rgb_image

    img = Image.new("RGB", (200, 100))
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert load_rgb_image(buf.getvalue()).shape[:2] == (200, 100)  # (H, W)


def test_topology_endpoint():
    r = _client(FakeService(True)).get("/api/facial-analysis/topology")
    body = r.json()
    assert len(body["tesselation"]) > 2000
    assert "left_eye" in body["contours"]


@pytest.mark.skipif(
    not (get_settings().resolved_model_path.is_file() and SAMPLE.is_file()),
    reason="model or sample image not downloaded",
)
def test_real_mediapipe_on_sample_portrait():
    with TestClient(create_app()) as client:  # runs lifespan -> loads real model
        r = client.post(
            "/api/facial-analysis/landmarks",
            files={"file": ("portrait.jpg", SAMPLE.read_bytes(), "image/jpeg")},
        )
    body = r.json()
    assert body["face_detected"] is True
    assert body["landmark_count"] == 478
    # Sanity: the subject's left eye (263) is on the image's right side.
    lm = {p["index"]: p for p in body["landmarks"]}
    assert lm[263]["x"] > lm[33]["x"]
