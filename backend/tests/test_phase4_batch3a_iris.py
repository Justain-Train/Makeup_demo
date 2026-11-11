"""Phase 4 batch 3a: behavioural test of the iris demotion in face_regions.segment_face (real pipeline)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import face_regions
from tests.synthetic import FIXTURES

SEG = "/api/facial-analysis/segmentation"
MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    not ((MODELS / "face_landmarker.task").exists() and (MODELS / "face_parser_resnet18.onnx").exists()),
    reason="model weights not downloaded",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def _post(c):
    data = (FIXTURES / "portrait.jpg").read_bytes()
    r = c.post(SEG, files={"file": ("p.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    return {z["id"]: z for z in r.json()["zones"]}


def _eye_group(zs, side):
    return [z for z in zs.values() if z["group"] == "eye" and z["id"].endswith(f"_{side}")]


def test_unpatched_fixture_keeps_every_eye_group_zone_ok(client):
    zs = _post(client)
    for side in "rl":
        group = _eye_group(zs, side)
        assert len(group) >= 8, side
        assert all(z["status"] == "ok" for z in group), [(z["id"], z["status"]) for z in group]


@pytest.mark.parametrize("side", ["r", "l"])
def test_failed_iris_check_demotes_the_upper_eyelid_that_side_only(client, monkeypatch, side):
    baseline = _post(client)
    other = "l" if side == "r" else "r"
    assert baseline[f"eyelid_upper_{side}"]["status"] == "ok" and baseline[f"eyelid_upper_{other}"]["status"] == "ok"
    checks = (False, True) if side == "r" else (True, False)
    monkeypatch.setattr(face_regions, "iris_checks", lambda *a, **k: checks)
    zs = _post(client)
    assert zs[f"eyelid_upper_{side}"]["status"] != "ok" and zs[f"eyelid_upper_{side}"]["reason"]
    assert zs[f"eyelid_upper_{other}"]["status"] == "ok"


@pytest.mark.parametrize("side", ["r", "l"])
def test_failed_iris_check_demotes_that_side_only(client, monkeypatch, side):
    baseline = _post(client)
    checks = (False, True) if side == "r" else (True, False)  # iris_checks returns (right, left)
    monkeypatch.setattr(face_regions, "iris_checks", lambda *a, **k: checks)
    zs = _post(client)
    mine = _eye_group(baseline, side)
    other_side = "l" if side == "r" else "r"
    assert len(mine) >= 8
    for z in mine:
        got = zs[z["id"]]
        assert got["status"] != "ok" and got["reason"], z["id"]
    for z in _eye_group(baseline, other_side):
        got = zs[z["id"]]
        assert got["status"] == z["status"] == "ok", z["id"]
        assert got["mask"] == z["mask"], z["id"]
