"""FeatureProvider wiring in the FacialProfile builder and route."""

import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.api.facial_profile import get_feature_provider
from app.main import create_app
from app.schemas.face_segmentation import SegmentationWarning
from app.schemas.facial_profile import FeaturesData
from app.services.facial_profile import builder as builder_module
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultFeatureProvider,
    DefaultParsingProvider,
    DefaultZoneProvider,
    MediaPipeLandmarkProvider,
)
from tests.synthetic import FIXTURES, FakeLandmarkService, FakeParser
from tests.test_phase4_batch3b_verification import needs_models

PROFILE = "/api/facial-analysis/profile"


def _skin_png(w: int = 820, h: int = 1024) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (224, 172, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _client(feature_provider=None) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: FakeParser()
    if feature_provider is not None:
        app.dependency_overrides[get_feature_provider] = lambda: feature_provider
    return TestClient(app)


def _post(client: TestClient, url: str = PROFILE, data: bytes | None = None, ctype: str = "image/png"):
    return client.post(url, files={"file": ("a.png", _skin_png() if data is None else data, ctype)})


def _build(feature_provider, include_masks: bool = True):
    image = np.zeros((1024, 820, 3), dtype=np.uint8)
    image[:] = (140, 172, 224)
    return build_profile(
        image,
        MediaPipeLandmarkProvider(FakeLandmarkService()),
        DefaultParsingProvider(FakeParser()),
        DefaultZoneProvider(),
        include_masks,
        feature_provider=feature_provider,
    )


def test_features_present_in_profile():
    body = _post(_client()).json()
    f = body["components"]["features"]
    assert body["schema_version"] == "1.1"
    assert f["meta"]["provider"]["kind"] == "features"
    assert f["meta"]["status"] in ("ok", "degraded")


def test_include_masks_false_still_has_skin_tone_and_no_mask_data():
    client = _client()
    r = client.post(PROFILE + "?include_masks=false", files={"file": ("a.png", _skin_png(), "image/png")})
    f = r.json()["components"]["features"]
    assert r.status_code == 200
    assert f["data"]["skin_tone"]["depth"]["label"] != "unknown"
    dumped = json.dumps(f)
    assert "mask" not in dumped and "png_base64" not in dumped


class _Raising(DefaultFeatureProvider):
    def classify(self, *args, **kwargs):
        raise RuntimeError("SECRET-DETAIL")


def test_raising_feature_provider_makes_only_features_unavailable():
    r = _post(_client(_Raising()))
    c = r.json()["components"]
    assert r.status_code == 200 and "SECRET-DETAIL" not in r.text
    assert c["features"]["meta"]["status"] == "unavailable"
    assert c["features"]["data"] is None
    assert c["features"]["meta"]["reason"] == "RuntimeError"
    assert [c[k]["meta"]["status"] for k in ("landmarks", "frame", "parsing", "zones")] == ["ok"] * 4


def test_build_without_feature_provider_omits_features():
    dumped = _build(None).model_dump(mode="json", exclude_none=False)
    assert "features" not in dumped["components"]


def test_frame_unavailable_makes_features_unavailable(monkeypatch):
    def _boom(*args, **kwargs):
        raise ValueError("x")

    monkeypatch.setattr(builder_module.FaceFrame, "from_landmarks", _boom)
    f = _build(DefaultFeatureProvider()).components.features
    assert f is not None and f.meta.status == "unavailable"
    assert f.data is None and f.meta.reason == "depends on frame"


def test_glasses_warning_reaches_provider(monkeypatch):
    seen: list[bool] = []

    class _Spy(DefaultFeatureProvider):
        def classify(self, image, pts, frame, zones, glasses):
            seen.append(glasses)
            return super().classify(image, pts, frame, zones, glasses)

    monkeypatch.setattr(
        builder_module,
        "_warnings",
        lambda *a, **k: [SegmentationWarning(code="glasses_detected", message="Glasses detected.")],
    )
    _build(_Spy())
    assert seen == [True]


def test_no_feature_values_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        r = _post(_client())
    assert r.status_code == 200
    for rec in caplog.records:
        assert not rec.name.startswith(("app.services.facial_features", "app.services.facial_profile"))


@needs_models
def test_real_pipeline_features_labels_in_literal_sets():
    with TestClient(create_app()) as client:
        r = _post(client, data=(FIXTURES / "portrait.jpg").read_bytes(), ctype="image/jpeg")
    f = r.json()["components"]["features"]
    assert f["meta"]["status"] in ("ok", "degraded")
    data = FeaturesData.model_validate(f["data"])
    classified = [
        data.eye_spacing, data.eye_tilt, data.eye_shape, data.lip_fullness, data.lip_balance,
        data.lip_width, data.brow_arch, data.brow_thickness, data.face_shape,
    ]
    if data.skin_tone is not None:
        classified += [data.skin_tone.depth, data.skin_tone.undertone]
    for c in classified:
        if c is not None:
            assert 0.0 <= c.confidence <= 1.0 and isinstance(c.label, str)


class _PoorLightingProvider(DefaultFeatureProvider):
    def classify(self, image, pts, frame, zones, glasses):
        dump = super().classify(image, pts, frame, zones, glasses).model_dump()
        if dump.get("skin_tone"):
            dump["skin_tone"]["lighting_quality"] = "poor"
        return FeaturesData.model_validate(dump)


class _NoZonesProvider(DefaultFeatureProvider):
    def classify(self, image, pts, frame, zones, glasses):
        return super().classify(image, pts, frame, None, glasses)


@pytest.mark.parametrize(
    "provider,reason",
    [(_PoorLightingProvider(), "Lighting is poor."), (_NoZonesProvider(), "Skin tone unavailable.")],
)
def test_degraded_paths(provider, reason):
    meta = _post(_client(provider)).json()["components"]["features"]["meta"]
    assert meta["status"] == "degraded"
    assert meta["reason"] == reason
