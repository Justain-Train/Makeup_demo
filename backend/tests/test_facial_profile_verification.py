"""Independent verification gaps for POST /api/facial-analysis/profile (Phase 5 Batch 2)."""

import io
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.main import create_app
from app.schemas.facial_landmarks import Landmark
from app.services.application_zones.frame import MIDLINE, FaceFrame
from app.services.facial_profile.builder import _CORE_REGIONS  # noqa: F401  (import check)
from tests.synthetic import FIXTURES, FakeLandmarkService, FakeParser, landmark_result
from tests.test_phase4_batch3b_verification import needs_models

PROFILE = "/api/facial-analysis/profile"
SEG = "/api/facial-analysis/segmentation"
OUTSIDE = {"face_partially_outside_image"}
SMALL = {"low_resolution_face"}
TILT = {"head_tilt"}


def _png(w: int = 820, h: int = 1024) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def _client(lm=None, parser=None) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: lm or FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser is None else parser
    return TestClient(app)


def _post(client, url=PROFILE, data=None, ctype="image/png"):
    return client.post(url, files={"file": ("a.png", _png() if data is None else data, ctype)})


def _base_px(w: int = 820, h: int = 1024) -> np.ndarray:
    """Fake-landmark pixel points for a w x h image (same mapping the fake uses)."""
    lr = landmark_result("ok", w, h)
    return np.array([(lm.x * w, lm.y * h) for lm in lr.landmarks], dtype=np.float64)


class _Transformed:
    """Landmark service whose 478 points are a pixel-space transform of the fixture face."""

    def __init__(self, fn):
        self.fn = fn

    def detect_landmarks(self, image: np.ndarray):
        h, w = image.shape[:2]
        base = landmark_result("ok", w, h)
        new = self.fn(_base_px(w, h), w, h)
        lms = [Landmark(index=i, x=float(x) / w, y=float(y) / h, z=0.0) for i, (x, y) in enumerate(new)]
        return base.model_copy(update={"landmarks": lms})


def _shift_left(target_min_x: float):
    def fn(p, w, h):
        return p + [target_min_x - p[:, 0].min(), 0.0]

    return fn


def _shrink(scale: float):
    def fn(p, w, h):
        c = p.mean(axis=0)
        return c + (p - c) * scale

    return fn


def _rotate(deg: float):
    def fn(p, w, h):
        t = np.radians(deg)
        r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        c = p.mean(axis=0)
        return c + (p - c) @ r.T

    return fn


def _coincident_irises(p, w, h):
    q = p.copy()
    q[473] = q[468]
    return q


def _flags(body):
    codes = {w["code"] for w in body["quality"]["warnings"]}
    c = body["components"]
    return codes, c["landmarks"]["meta"]["status"] == "degraded", c["frame"]["meta"]["status"] == "degraded"


# ---- Criterion 1: real fixture parity -------------------------------------------------

@needs_models
def test_real_fixture_zones_and_regions_match_segmentation():
    with TestClient(create_app()) as client:
        data = (FIXTURES / "portrait.jpg").read_bytes()
        prof = client.post(PROFILE, files={"file": ("p.jpg", data, "image/jpeg")}).json()
        seg = client.post(SEG, files={"file": ("p.jpg", data, "image/jpeg")}).json()
    assert prof["status"] == "ok" and seg["status"] == "ok"
    d = prof["components"]["zones"]["data"]
    assert d["zones"] == seg["zones"] and len(d["zones"]) > 0
    assert d["stages"] == seg["stages"]
    assert prof["components"]["parsing"]["data"]["regions"] == seg["anatomical"]["regions"]
    assert prof["components"]["parsing"]["data"]["label_map"] == seg["anatomical"]
    assert prof["quality"]["alignment"] == seg["alignment_check"]
    assert prof["quality"]["warnings"] == seg["warnings"]


# ---- Criterion 3: message present ------------------------------------------------------

@pytest.mark.parametrize("status,count", [("no_face_detected", 1), ("multiple_faces_detected", 2)])
def test_no_face_and_multiple_faces_carry_message_matching_segmentation(status, count):
    client = _client(FakeLandmarkService(status, face_count=count))
    p, s = _post(client).json(), _post(client, SEG).json()
    assert p["message"] and isinstance(p["message"], str)
    assert p["message"] == s["message"]
    assert p["components"] is None and p["quality"]["retake_suggested"] == []


# ---- Criterion 4: error codes ----------------------------------------------------------

@pytest.mark.parametrize("data,ctype", [(b"", "image/png"), (_png()[:40], "image/png"), (b"\x89PNG garbage", "image/jpeg")])
@pytest.mark.parametrize("parser", [None, "present"])
def test_corrupt_or_empty_upload_is_400_regardless_of_parser(data, ctype, parser):
    client = _client(parser=None if parser is None else FakeParser())
    r = _post(client, data=data, ctype=ctype)
    assert r.status_code == 400, r.text


def test_503_detail_text_equals_segmentation_text_exactly():
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: None  # real None: parser missing
    client = TestClient(app)
    p, s = _post(client), _post(client, SEG)
    assert p.status_code == s.status_code == 503
    assert p.json()["detail"] == s.json()["detail"] and p.json()["detail"]


# ---- Criterion 6/8: failures and logging -----------------------------------------------

class _BrokenLandmarks(FakeLandmarkService):
    def detect_landmarks(self, image):
        raise RuntimeError("SECRET-LANDMARK-DETAIL 123.45")


def test_forced_500_log_contains_only_exception_type(caplog):
    with caplog.at_level(logging.DEBUG):
        r = _post(_client(_BrokenLandmarks()))
    assert r.status_code == 500 and r.json()["detail"] == "Facial profile failed."
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "app.api.facial_profile"]
    assert msgs == ["Facial profile failed: RuntimeError"]
    assert all(rec.exc_info is None for rec in caplog.records)
    assert "SECRET" not in caplog.text and "123.45" not in caplog.text


# ---- Criterion 7: midline in pixels -----------------------------------------------------

def test_midline_is_in_image_pixels():
    body = _post(_client()).json()
    mid = body["components"]["frame"]["data"]["midline"]
    expected = _base_px()[MIDLINE]
    assert len(mid) == len(MIDLINE)
    assert np.allclose(np.array(mid), np.round(expected, 1), atol=1e-6)
    assert max(p[0] for p in mid) <= 820 and max(p[1] for p in mid) <= 1024
    assert max(p[1] for p in mid) > 1.0  # pixel scale, not normalized


def test_frame_ipd_and_roll_match_facefame_on_scaled_face():
    pts = _base_px()
    f = FaceFrame.from_landmarks(pts)
    d = _post(_client()).json()["components"]["frame"]["data"]
    assert d["ipd_px"] == pytest.approx(f.ipd, rel=1e-6)
    assert d["roll_deg"] == pytest.approx(f.roll_degrees, abs=1e-6)


# ---- Additional: degraded flags agree with _warnings -----------------------------------

@pytest.mark.parametrize(
    "fn,expect_lm,expect_frame",
    [
        (lambda p, w, h: p, set(), set()),
        (_shift_left(-30.0), OUTSIDE, set()),
        (_shrink(0.25), SMALL, SMALL),
        (_rotate(35.0), set(), TILT),
    ],
    ids=["ok", "outside", "small", "tilted"],
)
def test_degraded_flags_agree_with_warnings(fn, expect_lm, expect_frame):
    body = _post(_client(_Transformed(fn))).json()
    codes, lm_deg, fr_deg = _flags(body)
    assert body["status"] == "ok"
    # Sanity: the scenario actually triggers what it is meant to trigger.
    assert codes & (expect_lm | expect_frame) == (expect_lm | expect_frame)
    assert lm_deg == bool(codes & (OUTSIDE | SMALL))
    assert fr_deg == bool(codes & (SMALL | TILT))


@pytest.mark.parametrize("min_x,degraded", [(-1.9, False), (-2.1, True)])
def test_left_edge_tolerance_boundary(min_x, degraded):
    body = _post(_client(_Transformed(_shift_left(min_x)))).json()
    codes, lm_deg, _ = _flags(body)
    assert lm_deg is degraded
    assert bool(codes & OUTSIDE) is degraded


# ---- Additional: frame-unavailable path -------------------------------------------------

def test_frame_unavailable_makes_frame_and_zones_unavailable():
    r = _post(_client(_Transformed(_coincident_irises)))
    assert r.status_code == 200
    c = r.json()["components"]
    assert c["frame"]["meta"]["status"] == "unavailable" and c["frame"]["data"] is None
    assert c["frame"]["meta"]["reason"] == "ValueError"
    assert c["zones"]["meta"]["status"] == "unavailable" and c["zones"]["data"] is None
    assert c["zones"]["meta"]["reason"] == "depends on frame"
    assert c["landmarks"]["meta"]["status"] == "ok"
    assert c["parsing"]["meta"]["status"] in ("ok", "degraded")


def test_frame_unavailable_with_outside_landmarks_warning_consistency():
    """Known gap check: landmarks says 'outside' but no warning is emitted when frame is None."""
    def fn(p, w, h):
        return _shift_left(-30.0)(_coincident_irises(p, w, h), w, h)

    body = _post(_client(_Transformed(fn))).json()
    codes, lm_deg, _ = _flags(body)
    assert body["components"]["frame"]["meta"]["status"] == "unavailable"
    assert lm_deg == bool(codes & OUTSIDE), f"landmarks degraded={lm_deg} but warnings={sorted(codes)}"


# ---- Additional: include_masks=false keeps geometry -------------------------------------

def test_include_masks_false_keeps_zone_geometry_and_regions():
    client = _client()
    full = _post(client).json()["components"]
    r = client.post(PROFILE + "?include_masks=false", files={"file": ("a.png", _png(), "image/png")}).json()["components"]
    assert r["parsing"]["data"]["includes_masks"] is False and r["parsing"]["data"]["label_map"] is None
    assert r["zones"]["data"]["stages"] == full["zones"]["data"]["stages"]
    for a, b in zip(r["zones"]["data"]["zones"], full["zones"]["data"]["zones"]):
        assert a["polygon"] == b["polygon"] and a["bbox"] == b["bbox"] and a["mask"] is None
        assert a["pixel_count"] == b["pixel_count"]


# ---- Drift guard: warning text must match face_regions._warnings ----------------------

def test_outside_warning_text_matches_segmentation_on_frame_unavailable_path():
    client =_client(_Transformed(_shift_left(-30.0)))
    seg_json = client.post(SEG, files={"file": ("a.png", _png(), "image/png")}).json()
    seg_msgs = {w["code"]: w["message"] for w in seg_json["warnings"]}
    assert "face_partially_outside_image" in seg_msgs
    prof_client = _client(_Transformed(lambda p, w, h: _shift_left(-30.0)(_coincident_irises(p, w, h), w, h)))
    prof = _post(prof_client).json()
    prof_msgs = {w["code"]: w["message"] for w in prof["quality"]["warnings"]}
    assert prof["components"]["frame"]["meta"]["status"] == "unavailable"
    assert prof_msgs.get("face_partially_outside_image") == seg_msgs["face_partially_outside_image"]
