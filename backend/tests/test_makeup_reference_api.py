"""POST /api/makeup/reference-analysis: route, error mapping, caching. Never touches the network."""

import io
import logging
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.api.makeup_reference import get_vlm_provider
from app.config import Settings, get_settings
from app.main import create_app
from app.services.makeup_blueprint import vlm as vlm_mod
from app.services.makeup_blueprint.claude_vlm import ClaudeVlmProvider
from app.services.makeup_blueprint.service import ReferenceAnalysisError
from app.services.segmentation.face_parser import FaceParserBusyError
from tests.synthetic import FakeLandmarkService, FakeParser
from tests.test_phase4_batch3b_verification import needs_models

URL = "/api/makeup/reference-analysis"
PROFILE = "/api/facial-analysis/profile"
CANARY = "CANARY-SECRET-xyz123"
FAILED = "Reference analysis failed."
PORTRAIT = Path(__file__).parent / "fixtures" / "portrait.jpg"


def _png(w: int = 820, h: int = 1024) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def _client(landmarks=None, parser="default", settings=None, vlm=None) -> TestClient:
    app = create_app(use_lifespan=False)
    lm = landmarks or FakeLandmarkService()
    app.dependency_overrides[get_landmark_service] = lambda: lm
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser == "default" else parser
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    if vlm is not None:
        app.dependency_overrides[get_vlm_provider] = lambda: vlm
    return TestClient(app)


def _post(client: TestClient, url: str = URL, data: bytes | None = None, ctype: str = "image/png"):
    return client.post(url, files={"file": ("a.png", _png() if data is None else data, ctype)})


def test_success_shape_and_no_heavy_fields():
    r = _post(_client())
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert "schema_version" in body
    assert [p["id"] for p in body["providers"]] == ["cv_zone_color", "vlm_stub"]
    for word in ("label_map", "mask", "landmarks", "polygon", "png_base64"):
        assert word not in r.text, word
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("status,count", [("no_face_detected", 1), ("multiple_faces_detected", 2)])
def test_no_face_and_multiple_faces_are_200(status, count):
    r = _post(_client(FakeLandmarkService(status, face_count=count)))
    assert r.status_code == 200 and r.json()["status"] == status


def test_text_bytes_as_png_is_400_same_detail_as_profile():
    client = _client()
    r, p = _post(client, data=b"nope"), _post(client, PROFILE, data=b"nope")
    assert r.status_code == 400 and r.json()["detail"] == p.json()["detail"]


def test_gif_is_415_same_detail_as_profile():
    client = _client()
    r, p = _post(client, ctype="image/gif"), _post(client, PROFILE, ctype="image/gif")
    assert r.status_code == 415 and r.json()["detail"] == p.json()["detail"]


def test_oversize_upload_is_413():
    settings = Settings(_env_file=None, max_upload_mb=0.0001)
    client = _client(settings=settings)
    r, p = _post(client), _post(client, PROFILE)
    assert r.status_code == 413 and r.json()["detail"] == p.json()["detail"]


def test_pixel_cap_is_400():
    settings = Settings(_env_file=None, max_image_pixels=100)
    client = _client(settings=settings)
    r, p = _post(client), _post(client, PROFILE)
    assert r.status_code == 400 and r.json()["detail"] == p.json()["detail"]


def test_parser_missing_is_503_same_text_as_profile():
    client = _client(parser=None)
    r, p = _post(client), _post(client, PROFILE)
    assert r.status_code == 503 and r.json()["detail"] == p.json()["detail"]


def test_busy_is_503_with_retry_after():
    r = _post(_client(parser=FakeParser(error=FaceParserBusyError("busy"))))
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    assert r.json()["detail"] == "The server is busy. Please retry shortly."


@pytest.mark.parametrize("exc", [RuntimeError(CANARY), ReferenceAnalysisError("cv")])
def test_internal_errors_are_fixed_500(monkeypatch, caplog, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("app.api.makeup_reference.analyze_reference", boom)
    with caplog.at_level(logging.DEBUG):
        r = _post(_client())
    assert r.status_code == 500 and r.json() == {"detail": FAILED}
    assert CANARY not in r.text and CANARY not in caplog.text
    assert "stage" not in r.text


def test_default_settings_give_vlm_disabled_warning():
    r = _post(_client(settings=Settings(_env_file=None)))
    assert "vlm_disabled" in [w["code"] for w in r.json()["quality"]["warnings"]]


def test_vlm_provider_is_cached_across_requests(monkeypatch):
    calls = []
    real = vlm_mod.build_vlm_provider

    def counting(settings, client=None):
        calls.append(1)
        return real(settings, client)

    monkeypatch.setattr("app.api.makeup_reference.build_vlm_provider", counting)
    client = _client(settings=Settings(_env_file=None))
    assert _post(client).status_code == 200 and _post(client).status_code == 200
    assert len(calls) == 1


class _Messages:
    def parse(self, *a, **k):
        raise RuntimeError(CANARY)


class _Beta:
    messages = _Messages()


class _FailingClient:
    beta = _Beta()


def test_vlm_failure_degrades_to_200_with_warning(monkeypatch, caplog):
    settings = Settings(_env_file=None, makeup_vlm_enabled=True)
    provider = ClaudeVlmProvider(_FailingClient(), settings.makeup_vlm_model, "medium", 16000)
    monkeypatch.setattr("app.api.makeup_reference.build_vlm_provider", lambda s, client=None: provider)
    with caplog.at_level(logging.DEBUG):
        r = _post(_client(settings=settings))
    assert r.status_code == 200
    assert "vlm_failed" in [w["code"] for w in r.json()["quality"]["warnings"]]
    assert CANARY not in r.text and CANARY not in caplog.text


def test_openapi_lists_route_and_schema():
    spec = _client().get("/openapi.json").json()
    assert "post" in spec["paths"][URL]
    assert "MakeupBlueprint" in spec["components"]["schemas"]


@needs_models
@pytest.mark.skipif(not PORTRAIT.exists(), reason="portrait fixture missing")
def test_integration_real_services_no_network(monkeypatch):
    from app.services.facial_analysis.landmarks import FaceLandmarkService
    from app.services.segmentation.face_parser import OnnxFaceParser

    def deny(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
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
        r = client.post(URL, files={"file": ("p.jpg", PORTRAIT.read_bytes(), "image/jpeg")})
    finally:
        if hasattr(lm, "close"):
            lm.close()
    assert r.status_code == 200 and r.json()["status"] == "ok"
