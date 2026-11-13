"""Rate limiter: token-bucket core and the dependency wired on the three upload routes."""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.api.rate_limit import client_key
from app.config import Settings, get_settings
from app.main import create_app
from app.services.rate_limit import InMemoryRateLimiter
from tests.synthetic import FakeLandmarkService, FakeParser


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limiter(clock=None, rate=10, burst=3, max_keys=100) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(rate, burst, max_keys, clock=clock or Clock())


def test_burst_then_denied_then_refill():
    clock = Clock()
    lim = _limiter(clock)
    assert all(lim.check("a", "b").allowed for _ in range(3))
    denied = lim.check("a", "b")
    assert not denied.allowed
    assert denied.retry_after_s == 6
    assert isinstance(denied.retry_after_s, int)
    clock.now += 6
    assert lim.check("a", "b").allowed
    assert not lim.check("a", "b").allowed


def test_allowed_decision_has_zero_retry():
    assert _limiter().check("a", "b").retry_after_s == 0


def test_isolation_by_key_and_bucket():
    lim = _limiter()
    for _ in range(3):
        lim.check("a", "x")
    assert not lim.check("a", "x").allowed
    assert lim.check("b", "x").allowed
    assert lim.check("a", "y").allowed


@pytest.mark.parametrize(
    "kwargs", [{"rate": 0}, {"rate": -1}, {"burst": 0}, {"max_keys": 0}]
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        _limiter(**kwargs)


def test_eviction_bounds_table():
    lim = _limiter(max_keys=5)
    for i in range(50):
        lim.check(f"k{i}", "b")
    assert len(lim._state) <= 5


def test_thread_smoke_exactly_burst_allowed():
    lim = _limiter(burst=3)
    allowed: list[bool] = []
    guard = threading.Lock()

    def worker() -> None:
        local = [lim.check("one", "b").allowed for _ in range(100)]
        with guard:
            allowed.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 800
    assert sum(allowed) == 3


def test_client_key_missing_client_is_unknown():
    assert client_key(SimpleNamespace(client=None)) == "unknown"
    assert client_key(SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"))) == "1.2.3.4"


def _client(enabled=True) -> TestClient:
    settings = Settings(_env_file=None, rate_limit_enabled=enabled)
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: FakeParser()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


_ROUTES = [
    ("app.api.facial_profile", "/api/facial-analysis/profile", ["file"]),
    ("app.api.makeup_reference", "/api/makeup/reference-analysis", ["file"]),
    ("app.api.makeup_adapt", "/api/makeup/adapt", ["selfie", "reference"]),
]


def _post(client, url, fields, headers=None):
    files = {f: (f"{f}.png", b"x", "image/png") for f in fields}
    return client.post(url, files=files, headers=headers)


@pytest.mark.parametrize(("module", "url", "fields"), _ROUTES)
def test_route_limits_before_reading_upload(monkeypatch, module, url, fields):
    reader = Mock(side_effect=HTTPException(400, "bad"))
    monkeypatch.setattr(f"{module}.read_validated_image", reader)
    client = _client()
    for _ in range(3):
        assert _post(client, url, fields).status_code == 400
    assert reader.call_count == 3
    resp = _post(client, url, fields)
    assert resp.status_code == 429
    assert resp.json() == {"detail": "Too many requests. Please retry later."}
    assert resp.headers["Retry-After"].isdigit()
    assert int(resp.headers["Retry-After"]) >= 1
    assert reader.call_count == 3


def test_disabled_flag_never_limits(monkeypatch):
    reader = Mock(side_effect=HTTPException(400, "bad"))
    monkeypatch.setattr("app.api.facial_profile.read_validated_image", reader)
    client = _client(enabled=False)
    codes = [_post(client, "/api/facial-analysis/profile", ["file"]).status_code for _ in range(6)]
    assert 429 not in codes
    assert reader.call_count == 6


def test_forwarded_for_is_ignored(monkeypatch):
    monkeypatch.setattr("app.api.facial_profile.read_validated_image", Mock(side_effect=HTTPException(400, "bad")))
    client = _client()
    url = "/api/facial-analysis/profile"
    for i in range(3):
        assert _post(client, url, ["file"], {"X-Forwarded-For": f"9.9.9.{i}"}).status_code == 400
    assert _post(client, url, ["file"], {"X-Forwarded-For": "8.8.8.8"}).status_code == 429


def test_logs_bucket_but_never_client_ip(monkeypatch, caplog):
    monkeypatch.setattr("app.api.facial_profile.read_validated_image", Mock(side_effect=HTTPException(400, "bad")))
    client = _client()
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            _post(client, "/api/facial-analysis/profile", ["file"], {"X-Forwarded-For": "7.7.7.7"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "bucket=profile" in text
    assert "testclient" not in text
    assert "7.7.7.7" not in text
