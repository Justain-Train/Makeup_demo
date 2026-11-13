"""POST /api/makeup/adapt: route, validation, error mapping, privacy. Never touches the network."""

import base64
import io
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.config import Settings, get_settings
from app.main import create_app
from app.services.makeup_blueprint.service import ReferenceAnalysisError
from app.services.makeup_mapping import CatalogDriftError
from app.services.segmentation.face_parser import FaceParserBusyError
from tests.synthetic import FakeLandmarkService, FakeParser
from tests.test_phase4_batch3b_verification import needs_models

URL = "/api/makeup/adapt"
PROFILE = "/api/facial-analysis/profile"
CANARY = "CANARY-SECRET-xyz123"
FAILED = "Makeup adaptation failed."
PORTRAIT = Path(__file__).parent / "fixtures" / "portrait.jpg"
SELFIE_NAME = "selfie-private-name.png"
REF_NAME = "reference-private-name.png"


def _png(w: int = 820, h: int = 1024) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


class _CountingLandmarks(FakeLandmarkService):
    calls = 0

    def detect_landmarks(self, image):
        type(self).calls += 1
        return super().detect_landmarks(image)


class _CountingParser(FakeParser):
    def __init__(self, error=None):
        super().__init__(error)
        self.calls = 0

    def parse(self, image, face_bbox):
        self.calls += 1
        return super().parse(image, face_bbox)


def _client(landmarks=None, parser="default", settings=None) -> TestClient:
    app = create_app(use_lifespan=False)
    lm = landmarks or FakeLandmarkService()
    app.dependency_overrides[get_landmark_service] = lambda: lm
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser == "default" else parser
    app.dependency_overrides[get_settings] = lambda: settings or Settings(_env_file=None)
    return TestClient(app)


def _post(client, selfie=None, reference=None, params=None, sct="image/png", rct="image/png"):
    files = {
        "selfie": (SELFIE_NAME, _png() if selfie is None else selfie, sct),
        "reference": (REF_NAME, _png() if reference is None else reference, rct),
    }
    return client.post(URL, files=files, params=params)


def test_success_with_masks_by_default_and_no_store():
    r = _post(_client())
    body = r.json()
    # A blank synthetic reference has no makeup, so status is no_makeup_detected here;
    # the real-pipeline integration test asserts status ok with components and masks.
    assert r.status_code == 200 and body["status"] in ("ok", "no_makeup_detected")
    assert body["schema_version"] == "1.0"
    assert r.headers["cache-control"] == "no-store"
    assert "anthropic" not in r.text.lower()


@pytest.mark.parametrize("status,count", [("no_face_detected", 1), ("multiple_faces_detected", 2)])
def test_no_face_and_multiple_faces_are_200(status, count):
    r = _post(_client(FakeLandmarkService(status, face_count=count)))
    assert r.status_code == 200 and r.json()["status"] in (status, "reference_unavailable")


def test_include_masks_false_drops_masks_only(monkeypatch):
    from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
    from app.services.makeup_adaptation import pipeline

    captured = {}
    real = pipeline.adapt

    def spy(profile, blueprint, *a, **k):
        plan = real(profile, blueprint, *a, **k)
        captured["plan"] = plan
        return plan

    monkeypatch.setattr(pipeline, "adapt", spy)
    client = _client()
    with_masks = _post(client).json()
    without = _post(client, params={"include_masks": "false"}).json()
    assert with_masks["status"] == without["status"]
    assert isinstance(captured["plan"], AdaptedMakeupPlan)
    assert len(with_masks["components"]) == len(without["components"])
    for a, b in zip(with_masks["components"], without["components"]):
        assert b["mask"] is None
        assert a["bbox"] == b["bbox"] and a["centroid"] == b["centroid"] and a["pixel_count"] == b["pixel_count"]
        assert a["zones"] == b["zones"] and a["component"] == b["component"]


@pytest.mark.parametrize("field", ["selfie", "reference"])
@pytest.mark.parametrize(
    "kind,status",
    [("text", 400), ("gif", 415)],
)
def test_bad_upload_per_field_matches_profile_and_skips_inference(field, kind, status):
    parser = _CountingParser()
    lm = _CountingLandmarks()
    _CountingLandmarks.calls = 0
    client = _client(landmarks=lm, parser=parser)
    data = b"nope" if kind == "text" else None
    ctype = "image/gif" if kind == "gif" else "image/png"
    kw = {field: data, ("sct" if field == "selfie" else "rct"): ctype}
    if data is None:
        kw.pop(field)
    r = _post(client, **kw)
    p = client.post(PROFILE, files={"file": ("a.png", data if data is not None else _png(), ctype)})
    assert r.status_code == status and r.json()["detail"] == p.json()["detail"]
    assert parser.calls == 0 and _CountingLandmarks.calls == 0


@pytest.mark.parametrize("field", ["selfie", "reference"])
def test_oversize_per_field_is_413_and_skips_inference(field):
    parser = _CountingParser()
    settings = Settings(_env_file=None, max_upload_mb=0.0001)
    big = _png()
    small = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    assert len(big) > settings.max_upload_bytes >= len(buf.getvalue())
    kw = {field: big}
    other = "reference" if field == "selfie" else "selfie"
    kw[other] = buf.getvalue()
    client = _client(parser=parser, settings=settings)
    r = _post(client, **kw)
    p = client.post(PROFILE, files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413 and r.json()["detail"] == p.json()["detail"]
    assert parser.calls == 0


def test_missing_field_is_422():
    r = _client().post(URL, files={"selfie": ("a.png", _png(), "image/png")})
    assert r.status_code == 422


def test_parser_missing_is_503_same_text_as_profile():
    client = _client(parser=None)
    r = _post(client)
    p = client.post(PROFILE, files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503 and r.json()["detail"] == p.json()["detail"]


def test_busy_is_503_with_retry_after():
    r = _post(_client(parser=FakeParser(error=FaceParserBusyError("busy"))))
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    assert r.json()["detail"] == "The server is busy. Please retry shortly."


@pytest.mark.parametrize("exc", [RuntimeError(CANARY), ReferenceAnalysisError("cv"), CatalogDriftError(CANARY)])
def test_internal_errors_are_fixed_500(monkeypatch, caplog, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("app.api.makeup_adapt.adapt_from_images", boom)
    with caplog.at_level(logging.DEBUG):
        r = _post(_client())
    assert r.status_code == 500 and r.json() == {"detail": FAILED}
    assert CANARY not in r.text and CANARY not in caplog.text


def test_no_sensitive_data_in_logs(caplog, monkeypatch):
    png = _png()
    b64 = base64.b64encode(png).decode()
    with caplog.at_level(logging.DEBUG):
        _post(_client())
        _post(_client(parser=FakeParser(error=FaceParserBusyError("busy"))))
        monkeypatch.setattr("app.api.makeup_adapt.adapt_from_images", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(CANARY)))
        _post(_client())
    text = "\n".join(r.getMessage() for r in caplog.records)
    for needle in (SELFIE_NAME, REF_NAME, b64[:40], "landmark", CANARY):
        assert needle not in text
    assert "\\x89PNG" not in text


def test_openapi_lists_route():
    spec = _client().get("/openapi.json").json()
    assert "post" in spec["paths"][URL]
    assert "AdaptedMakeupPlan" in spec["components"]["schemas"]


@needs_models
@pytest.mark.skipif(not PORTRAIT.exists(), reason="portrait fixture missing")
def test_integration_real_pipeline_portrait_as_both_fields():
    from app.services.facial_analysis.landmarks import FaceLandmarkService
    from app.services.segmentation.face_parser import OnnxFaceParser

    s = get_settings()
    lm = FaceLandmarkService(
        model_path=s.resolved_model_path,
        min_face_detection_confidence=s.min_face_detection_confidence,
        min_face_presence_confidence=s.min_face_presence_confidence,
        secondary_face_area_ratio=s.secondary_face_area_ratio,
    )
    try:
        parser = OnnxFaceParser(model_path=s.resolved_face_parser_path, expected_sha256=s.face_parser_sha256)
        client = _client(landmarks=lm, parser=parser, settings=Settings(_env_file=None))
        data = PORTRAIT.read_bytes()
        files = {"selfie": ("p.jpg", data, "image/jpeg"), "reference": ("p.jpg", data, "image/jpeg")}
        r = client.post(URL, files=files)
        r2 = client.post(URL, files=files, params={"include_masks": "false"})
    finally:
        if hasattr(lm, "close"):
            lm.close()
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["components"]
    assert any(c["mask"] is not None for c in body["components"])
    assert r2.status_code == 200 and all(c["mask"] is None for c in r2.json()["components"])
