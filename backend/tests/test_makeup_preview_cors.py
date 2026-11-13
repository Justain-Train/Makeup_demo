"""CORS exposes Retry-After to the frontend origin so /preview can show 'try again in N s' (Phase 9a-3)."""

from tests.test_makeup_preview_route import _client

_FRONTEND = "http://localhost:3000"
_BAD_FILES = {
    "selfie": ("a.txt", b"not an image", "text/plain"),
    "reference": ("b.txt", b"not an image", "text/plain"),
}


def _post(client, origin=_FRONTEND):
    return client.post("/api/makeup/preview", headers={"Origin": origin}, files=_BAD_FILES)


def test_allowed_origin_exposes_retry_after_on_429():
    client = _client(rate=True)
    statuses = [_post(client) for _ in range(4)]
    last = statuses[-1]
    assert [r.status_code for r in statuses[:3]] == [415, 415, 415]
    assert last.status_code == 429
    assert last.headers["access-control-allow-origin"] == _FRONTEND
    exposed = [h.strip().lower() for h in last.headers["access-control-expose-headers"].split(",")]
    assert "retry-after" in exposed
    assert last.headers["retry-after"].isdigit()


def test_foreign_origin_gets_no_cors_grant():
    client = _client(rate=False)
    r = _post(client, origin="http://evil.example")
    # Starlette adds expose-headers unconditionally; without allow-origin the browser still blocks the read.
    assert "access-control-allow-origin" not in r.headers
