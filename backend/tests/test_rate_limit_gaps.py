"""Gap coverage for the rate limiter: per-IP HTTP isolation, no inference on denied requests, LRU order."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.config import Settings, get_settings
from app.main import create_app
from app.services.rate_limit import InMemoryRateLimiter
from tests.synthetic import FakeLandmarkService, FakeParser

_URL = "/api/facial-analysis/profile"
_ROUTES = [
    ("app.api.facial_profile", "/api/facial-analysis/profile", ["file"]),
    ("app.api.makeup_reference", "/api/makeup/reference-analysis", ["file"]),
    ("app.api.makeup_adapt", "/api/makeup/adapt", ["selfie", "reference"]),
]


def _client(counters: dict, peer=("1.2.3.4", 5000)) -> TestClient:
    settings = Settings(_env_file=None, rate_limit_enabled=True)

    def _counted(name, factory):
        def dep():
            counters[name] = counters.get(name, 0) + 1
            return factory()
        return dep

    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = _counted("landmark", lambda: FakeLandmarkService())
    app.dependency_overrides[get_face_parser] = _counted("parser", lambda: FakeParser())
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.rate_limiter = InMemoryRateLimiter(10, 3, 100)
    return TestClient(app, client=peer)


def _post(client, url, fields):
    return client.post(url, files={f: (f"{f}.png", b"x", "image/png") for f in fields})


@pytest.mark.parametrize(("module", "url", "fields"), _ROUTES)
def test_denied_request_runs_no_inference_or_upload_read(monkeypatch, module, url, fields):
    reader = Mock(side_effect=HTTPException(400, "bad"))
    monkeypatch.setattr(f"{module}.read_validated_image", reader)
    counters: dict = {}
    client = _client(counters)
    for _ in range(3):
        _post(client, url, fields)
    counters.clear()
    reader.reset_mock()
    resp = _post(client, url, fields)
    assert resp.status_code == 429
    assert reader.call_count == 0
    assert counters == {}


def test_different_peer_ip_is_independent_over_http(monkeypatch):
    monkeypatch.setattr("app.api.facial_profile.read_validated_image", Mock(side_effect=HTTPException(400, "bad")))
    app_client = _client({}, peer=("1.2.3.4", 5000))
    limiter = app_client.app.state.rate_limiter
    for _ in range(3):
        assert _post(app_client, _URL, ["file"]).status_code == 400
    assert _post(app_client, _URL, ["file"]).status_code == 429
    other = TestClient(app_client.app, client=("5.6.7.8", 5000))
    assert _post(other, _URL, ["file"]).status_code == 400
    assert app_client.app.state.rate_limiter is limiter


def test_missing_client_uses_unknown_key_over_http(monkeypatch):
    from app.api import rate_limit as rl

    seen = []
    real = rl.client_key

    def spy(request):
        key = real(request)
        seen.append(key)
        return key

    monkeypatch.setattr(rl, "client_key", spy)
    monkeypatch.setattr("app.api.facial_profile.read_validated_image", Mock(side_effect=HTTPException(400, "bad")))
    client = _client({}, peer=None)
    _post(client, _URL, ["file"])
    assert seen and set(seen) <= {"unknown", "testclient"}


def test_lru_keeps_recently_used_key():
    clock_now = [0.0]
    lim = InMemoryRateLimiter(10, 3, 2, clock=lambda: clock_now[0])
    lim.check("a", "b")
    lim.check("c", "b")
    lim.check("a", "b")  # refresh a
    lim.check("d", "b")  # should evict c, not a
    keys = {k[1] for k in lim._state}
    assert keys == {"a", "d"}
