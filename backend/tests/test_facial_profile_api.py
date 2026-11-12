"""POST /api/facial-analysis/profile: providers, builder, route."""

import io
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.api.facial_profile import get_landmark_provider, get_parsing_provider, get_zone_provider
from app.main import create_app
from app.schemas.face_segmentation import LICENSE_WARNING
from app.services.application_zones.frame import FaceFrame
from app.services.facial_profile.providers import DefaultParsingProvider, DefaultZoneProvider, MediaPipeLandmarkProvider
from app.services.segmentation.face_parser import FaceParserBusyError
from tests.synthetic import FIXTURES, FakeLandmarkService, FakeParser, fixture_landmarks
from tests.test_phase4_batch3b_verification import needs_models

PROFILE = "/api/facial-analysis/profile"
SEG = "/api/facial-analysis/segmentation"


def _png(w: int = 820, h: int = 1024) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def _client(landmarks=None, parser="default", parsing_provider="default", zone_provider=None) -> TestClient:
    app = create_app(use_lifespan=False)
    lm = landmarks or FakeLandmarkService()
    app.dependency_overrides[get_landmark_service] = lambda: lm
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser == "default" else parser
    if parsing_provider != "default":
        app.dependency_overrides[get_parsing_provider] = lambda: parsing_provider
    if zone_provider is not None:
        app.dependency_overrides[get_zone_provider] = lambda: zone_provider
    return TestClient(app)


def _post(client: TestClient, url: str = PROFILE, data: bytes | None = None, ctype: str = "image/png"):
    return client.post(url, files={"file": ("a.png", _png() if data is None else data, ctype)})


def test_parity_with_segmentation():
    client = _client()
    prof = _post(client).json()
    seg = _post(client, SEG).json()
    assert prof["status"] == "ok"
    assert prof["components"]["zones"]["data"]["zones"] == seg["zones"]
    assert prof["components"]["zones"]["data"]["stages"] == seg["stages"]
    assert prof["components"]["parsing"]["data"]["regions"] == seg["anatomical"]["regions"]
    assert prof["components"]["parsing"]["data"]["label_map"] == seg["anatomical"]
    assert prof["quality"]["alignment"] == seg["alignment_check"]
    assert prof["quality"]["warnings"] == seg["warnings"]
    comps = prof["components"]
    assert [comps[k]["meta"]["status"] for k in ("landmarks", "frame", "parsing", "zones")] == ["ok"] * 4
    assert comps["parsing"]["meta"]["provider"]["license_warning"] == LICENSE_WARNING
    assert comps["zones"]["meta"]["provider"]["license_warning"] == LICENSE_WARNING


def test_include_masks_false_strips_masks_and_label_map_only():
    client = _client()
    full = _post(client).json()["components"]
    r = client.post(PROFILE + "?include_masks=false", files={"file": ("a.png", _png(), "image/png")})
    comps = r.json()["components"]
    assert r.status_code == 200
    assert comps["parsing"]["data"]["label_map"] is None and comps["parsing"]["data"]["includes_masks"] is False
    assert comps["parsing"]["data"]["regions"] == full["parsing"]["data"]["regions"]
    zones = comps["zones"]["data"]["zones"]
    assert all(z["mask"] is None for z in zones)
    for a, b in zip(zones, full["zones"]["data"]["zones"]):
        assert a["bbox"] == b["bbox"] and a["polygon"] == b["polygon"]


@pytest.mark.parametrize("status,count", [("no_face_detected", 1), ("multiple_faces_detected", 2)])
def test_no_face_and_multiple_faces_are_200_without_components(status, count):
    r = _post(_client(FakeLandmarkService(status, face_count=count)))
    body = r.json()
    assert r.status_code == 200 and body["status"] == status and body["components"] is None


def test_parser_missing_is_503_same_text_as_segmentation():
    client = _client(parser=None)
    p, s = _post(client), _post(client, SEG)
    assert p.status_code == 503 and p.json()["detail"] == s.json()["detail"]


def test_invalid_upload_is_400_before_parser_check():
    assert _post(_client(parser=None), data=b"nope").status_code == 400


def test_busy_is_503_with_retry_after():
    r = _post(_client(parser=FakeParser(error=FaceParserBusyError("busy"))))
    assert r.status_code == 503 and r.headers["retry-after"] == "2"


class _StubParsing(DefaultParsingProvider):
    def info(self):
        return super().info().model_copy(update={"id": "stub-parser"})


def test_swapping_parsing_provider_changes_only_its_provider_meta():
    base = _post(_client()).json()
    stub = _post(_client(parsing_provider=_StubParsing(FakeParser()))).json()
    assert stub["components"]["parsing"]["meta"]["provider"]["id"] == "stub-parser"
    for d in (base, stub):
        d["components"]["parsing"]["meta"].pop("provider")
        d.pop("timings_ms")
    assert base == stub


class _RaisingParsing(DefaultParsingProvider):
    def parse(self, image, bbox):
        raise RuntimeError("SECRET-DETAIL")


def test_raising_parsing_provider_makes_parsing_and_zones_unavailable():
    r = _post(_client(parsing_provider=_RaisingParsing(FakeParser())))
    c = r.json()["components"]
    assert r.status_code == 200 and "SECRET-DETAIL" not in r.text
    assert c["parsing"]["meta"]["status"] == c["zones"]["meta"]["status"] == "unavailable"
    assert c["parsing"]["data"] is None and c["parsing"]["meta"]["reason"] == "RuntimeError"
    assert c["zones"]["meta"]["reason"] == "depends on parsing"
    assert c["landmarks"]["meta"]["status"] == c["frame"]["meta"]["status"] == "ok"
    assert r.json()["quality"]["alignment"] is None


class _RaisingZones(DefaultZoneProvider):
    def derive(self, *args):
        raise RuntimeError("SECRET-DETAIL")


def test_raising_zones_provider_makes_only_zones_unavailable():
    r = _post(_client(zone_provider=_RaisingZones()))
    c = r.json()["components"]
    assert r.status_code == 200 and "SECRET-DETAIL" not in r.text
    assert c["zones"]["meta"]["status"] == "unavailable" and c["zones"]["meta"]["reason"] == "RuntimeError"
    assert [c[k]["meta"]["status"] for k in ("landmarks", "frame", "parsing")] == ["ok"] * 3


def test_frame_matches_face_frame():
    pts, w, h = fixture_landmarks()
    pts = pts / [w, h] * [820, 1024]
    frame = FaceFrame.from_landmarks(pts)
    data = _post(_client()).json()["components"]["frame"]["data"]
    assert data["ipd_px"] == pytest.approx(frame.ipd, rel=1e-6)
    assert data["roll_deg"] == pytest.approx(frame.roll_degrees, abs=1e-6)
    assert data["u"] == pytest.approx(list(frame.u)) and data["v"] == pytest.approx(list(frame.v))


class _BrokenLandmarks(FakeLandmarkService):
    def detect_landmarks(self, image: np.ndarray):
        raise RuntimeError("SECRET-LANDMARK-DETAIL")


def test_forced_500_logs_only_exception_type(caplog):
    with caplog.at_level(logging.DEBUG):
        r = _post(_client(_BrokenLandmarks()))
    assert r.status_code == 500 and r.json()["detail"] == "Facial profile failed."
    assert "SECRET-LANDMARK-DETAIL" not in r.text and "SECRET-LANDMARK-DETAIL" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_landmark_provider_without_readable_weights_has_no_hash():
    p = MediaPipeLandmarkProvider(FakeLandmarkService(), None)
    assert p.info().weights_sha256 is None and p.info().kind == "landmarks"


@needs_models
def test_real_pipeline_profile_on_fixture():
    with TestClient(create_app()) as client:
        r = _post(client, data=(FIXTURES / "portrait.jpg").read_bytes(), ctype="image/jpeg")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    c = body["components"]
    assert c["landmarks"]["meta"]["provider"]["weights_sha256"]
    assert c["parsing"]["meta"]["provider"]["weights_sha256"]
    assert all(c[k]["meta"]["status"] in ("ok", "degraded") for k in c)
    assert body["quality"]["alignment"]["passed"] is True
