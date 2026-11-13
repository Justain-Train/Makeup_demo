"""POST /api/makeup/preview: lossless round trip, fallbacks, validation, errors, rate limit, privacy."""

import base64
import io
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.config import Settings, get_settings
from app.main import create_app
from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.schemas.facial_profile import ImageSize
from app.services.makeup_blueprint.service import ReferenceAnalysisError
from app.services.makeup_mapping import CatalogDriftError
from app.services.makeup_preview.renderer import render_layers
from tests.synthetic import FakeLandmarkService, FakeParser
from tests.test_makeup_preview_renderer import _comp, _image, _plan

URL = "/api/makeup/preview"
ADAPT_URL = "/api/makeup/adapt"
CANARY = "CANARY-SECRET-xyz123"
FAILED = "Makeup preview failed."
SELFIE_NAME = "selfie-private-name.png"
REF_NAME = "reference-private-name.png"
EMPTY_PLAN_STATUS = "no_makeup_detected"


def _rgb_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _png(w: int = 96, h: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def _decode_b64_png(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _client(parser="default", settings=None, rate=False) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser == "default" else parser
    app.dependency_overrides[get_settings] = lambda: settings or Settings(_env_file=None, rate_limit_enabled=rate)
    return TestClient(app)


def _post(client, url=URL, selfie=None, reference=None, sct="image/png", rct="image/png"):
    files = {
        "selfie": (SELFIE_NAME, _rgb_png(_image()) if selfie is None else selfie, sct),
        "reference": (REF_NAME, _png() if reference is None else reference, rct),
    }
    return client.post(url, files=files)


def _stub_plan(monkeypatch, plan: AdaptedMakeupPlan) -> None:
    monkeypatch.setattr("app.api.makeup_preview.adapt_from_images", lambda *a, **k: plan)


def _ok_plan(*components) -> AdaptedMakeupPlan:
    return _plan(*components)


# --- 1. Success: lossless round trip, dimensions, reports, no-store ---------------------


def test_success_round_trip_matches_render_layers_exactly(monkeypatch):
    selfie = _image()
    plan = _ok_plan(_comp(0, (20, 20, 30, 20), intensity=0.9), _comp(1, (40, 40, 24, 18), color="#2040e0"))
    _stub_plan(monkeypatch, plan)

    r = _post(_client())
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.0"
    assert r.headers["cache-control"] == "no-store"
    assert (body["width"], body["height"]) == (selfie.shape[1], selfie.shape[0])

    decoded = _decode_b64_png(body["image_png_base64"])
    expected = render_layers(selfie, plan, seed=0)
    assert decoded.shape == selfie.shape
    assert np.array_equal(decoded, expected.image)
    assert not np.array_equal(decoded, selfie)  # components actually changed pixels

    assert len(body["reports"]) == len(expected.reports) == 2
    assert body["reports"] == [r_.model_dump() for r_ in expected.reports]
    assert {rep["status"] for rep in body["reports"]} == {"rendered"}
    assert body["plan_status"] == "ok"
    assert body["warnings"] == []


def test_success_is_deterministic_across_requests(monkeypatch):
    _stub_plan(monkeypatch, _ok_plan(_comp(0, (20, 20, 30, 20))))
    client = _client()
    a = _post(client).json()["image_png_base64"]
    b = _post(client).json()["image_png_base64"]
    assert a == b


# --- 4. Not-ok or empty plan: unmodified selfie + warning -------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        AdaptedMakeupPlan(status=EMPTY_PLAN_STATUS, image=ImageSize(width=96, height=80), components=[]),
        AdaptedMakeupPlan(status="ok", image=ImageSize(width=96, height=80), components=[]),
    ],
    ids=["not_ok_status", "ok_but_empty"],
)
def test_not_ok_or_empty_plan_returns_unmodified_selfie_with_warning(monkeypatch, plan):
    selfie = _image()
    _stub_plan(monkeypatch, plan)
    r = _post(_client())
    assert r.status_code == 200
    body = r.json()
    assert np.array_equal(_decode_b64_png(body["image_png_base64"]), selfie)
    assert (body["width"], body["height"]) == (96, 80)
    assert body["reports"] == []
    assert any(w.startswith("no components rendered") for w in body["warnings"])


def test_every_component_skipped_returns_unchanged_with_warning(monkeypatch):
    selfie = _image()
    no_mask = _comp(0, (20, 20, 30, 20)).model_copy(update={"mask": None})
    _stub_plan(monkeypatch, _ok_plan(no_mask))
    body = _post(_client()).json()
    assert np.array_equal(_decode_b64_png(body["image_png_base64"]), selfie)
    assert [rep["reason"] for rep in body["reports"]] == ["no_mask"]
    assert "every component was skipped; the preview is unchanged" in body["warnings"]


def test_partially_skipped_reports_one_per_component(monkeypatch):
    no_mask = _comp(0, (20, 20, 30, 20)).model_copy(update={"mask": None})
    good = _comp(1, (40, 40, 24, 18))
    _stub_plan(monkeypatch, _ok_plan(no_mask, good))
    body = _post(_client()).json()
    by_index = {rep["index"]: rep for rep in body["reports"]}
    assert set(by_index) == {0, 1}
    assert by_index[0]["status"] == "skipped" and by_index[0]["reason"] == "no_mask"
    assert by_index[1]["status"] == "rendered" and by_index[1]["pixels_touched"] > 0
    assert body["warnings"] == []


# --- 3. Validation and error mapping ----------------------------------------------------


@pytest.mark.parametrize("field", ["selfie", "reference"])
@pytest.mark.parametrize("kind,status", [("text", 400), ("gif", 415)])
def test_bad_upload_matches_adapt_status_and_detail(field, kind, status):
    data = b"nope" if kind == "text" else None
    ctype = "image/gif" if kind == "gif" else "image/png"
    bad = data if data is not None else _png()
    kw = {field: bad, ("sct" if field == "selfie" else "rct"): ctype}
    client = _client()
    r = _post(client, **kw)
    a = _post(client, url=ADAPT_URL, **kw)
    assert r.status_code == status == a.status_code
    assert r.json()["detail"] == a.json()["detail"]


@pytest.mark.parametrize("field", ["selfie", "reference"])
def test_oversize_is_413_and_matches_adapt(field):
    settings = Settings(_env_file=None, max_upload_mb=0.0001, rate_limit_enabled=False)
    big = _png(400, 400)
    small = _png(1, 1)
    assert len(big) > settings.max_upload_bytes >= len(small)
    kw = {field: big}
    client = _client(settings=settings)
    r = _post(client, **kw) if field == "selfie" else _post(client, reference=big)
    a = client.post(ADAPT_URL, files={
        "selfie": (SELFIE_NAME, big if field == "selfie" else small, "image/png"),
        "reference": (REF_NAME, big if field == "reference" else small, "image/png"),
    })
    assert r.status_code == 413 and r.json()["detail"] == a.json()["detail"]


def test_missing_parser_is_503(monkeypatch):
    r = _post(_client(parser=None))
    assert r.status_code == 503
    assert r.json()["detail"].startswith("Face parser is not available")


def test_missing_field_is_422():
    r = _client().post(URL, files={"selfie": ("a.png", _png(), "image/png")})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "exc",
    [RuntimeError(CANARY), ReferenceAnalysisError("cv"), CatalogDriftError(CANARY)],
    ids=["runtime", "reference_analysis", "catalog_drift"],
)
def test_internal_failure_is_generic_500(monkeypatch, caplog, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("app.api.makeup_preview.adapt_from_images", boom)
    with caplog.at_level(logging.DEBUG):
        r = _post(_client())
    assert r.status_code == 500 and r.json() == {"detail": FAILED}
    assert CANARY not in r.text and CANARY not in caplog.text


def test_render_failure_is_generic_500(monkeypatch, caplog):
    _stub_plan(monkeypatch, _ok_plan(_comp(0, (20, 20, 30, 20))))

    def boom(*a, **k):
        raise RuntimeError(CANARY)

    monkeypatch.setattr("app.services.makeup_preview.service.render_layers", boom)
    with caplog.at_level(logging.DEBUG):
        r = _post(_client())
    assert r.status_code == 500 and r.json() == {"detail": FAILED}
    assert CANARY not in r.text and CANARY not in caplog.text


# --- 2. Rate limiting: own bucket, independent of adapt ---------------------------------


def test_preview_bucket_limits_after_burst_and_is_independent_of_adapt(monkeypatch):
    _stub_plan(monkeypatch, _ok_plan())
    monkeypatch.setattr("app.api.makeup_adapt.adapt_from_images", lambda *a, **k: _ok_plan())
    client = _client(rate=True)
    for _ in range(3):
        assert _post(client).status_code == 200
    denied = _post(client)
    assert denied.status_code == 429
    assert denied.json() == {"detail": "Too many requests. Please retry later."}
    assert int(denied.headers["Retry-After"]) >= 1
    # adapt has its own bucket and must still be allowed
    assert _post(client, url=ADAPT_URL).status_code == 200


def test_adapt_burst_does_not_exhaust_preview(monkeypatch):
    _stub_plan(monkeypatch, _ok_plan())
    monkeypatch.setattr("app.api.makeup_adapt.adapt_from_images", lambda *a, **k: _ok_plan())
    client = _client(rate=True)
    for _ in range(3):
        assert _post(client, url=ADAPT_URL).status_code == 200
    assert _post(client, url=ADAPT_URL).status_code == 429
    assert _post(client).status_code == 200


# --- 5. Privacy: no image bytes, no IPs in logs ----------------------------------------


def test_no_image_bytes_or_ip_in_logs(monkeypatch, caplog):
    selfie_png = _rgb_png(_image())
    b64 = base64.b64encode(selfie_png).decode()
    _stub_plan(monkeypatch, _ok_plan(_comp(0, (20, 20, 30, 20))))
    with caplog.at_level(logging.DEBUG):
        ok = _post(_client(), selfie=selfie_png)
        busy_client = _client(parser=None)
        _post(busy_client, selfie=selfie_png)
        monkeypatch.setattr(
            "app.api.makeup_preview.adapt_from_images",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError(CANARY)),
        )
        _post(_client(), selfie=selfie_png)
    assert ok.status_code == 200
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    png_b64_head = b64[:40]
    for needle in (SELFIE_NAME, REF_NAME, png_b64_head, CANARY, "testclient", "127.0.0.1", "\\x89PNG"):
        assert needle not in text
    assert ok.json()["image_png_base64"][:40] not in text
