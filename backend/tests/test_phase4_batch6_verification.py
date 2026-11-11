"""Phase 4 batch 6 verification: lower eyelid; the 68 earlier zones stay bit-identical."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.application_zones import derive as dv
from app.services.application_zones.catalog import STAGES, ZONE_SPEC_BY_ID, ZONE_SPECS
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401
from tests.test_phase4_batch3a_verification import _glasses_over, _sig
from tests.test_phase4_batch3b_verification import _labels, _scenario_b, needs_models
from tests.test_phase4_batch4_verification import _symmetrized

GOLDEN = json.loads((Path(__file__).parent / "golden_batch6.json").read_text())
NEW = ["lower_eyelid_r", "lower_eyelid_l"]
SEG = "/api/facial-analysis/segmentation"


def _ok(pts_, labels=None):
    zs = _derive(pts_, _labels(pts_) if labels is None else labels)
    assert all(zs[z].status == "ok" and zs[z].mask is not None and zs[z].mask.any() for z in NEW), \
        [(z, zs[z].status, zs[z].reason) for z in NEW]
    return zs


def test_zone_count_is_seventy():
    assert len(ZONE_SPECS) == 70


@pytest.mark.parametrize("scenario", ["synthetic", "roll_-20", "roll_20", "glasses_over_eyes", "glasses_over_brows"])
def test_existing_zones_are_bit_identical_to_golden(pts, scenario):
    p, labels = _scenario_b(scenario, pts)
    zs = _derive(p, labels)
    expected = GOLDEN["scenarios"][scenario]
    assert len(expected) == 68
    for zid, sig in expected.items():
        assert _sig(zs[zid]) == sig, (scenario, zid)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@needs_models
def test_existing_zones_are_identical_through_the_api(client):
    from tests.synthetic import FIXTURES

    r = client.post(SEG, files={"file": ("p.jpg", (FIXTURES / "portrait.jpg").read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    zs = {z["id"]: z for z in r.json()["zones"]}
    expected = GOLDEN["fixture_api"]
    assert len(expected) == 68
    for zid, sig in expected.items():
        z = zs[zid]
        got = {
            "status": z["status"],
            "reason": z["reason"],
            "confidence": z["confidence"],
            "coverage": z["coverage"],
            "mask_png_sha256": None if z["mask"] is None else hashlib.sha256(
                z["mask"]["png_base64"].encode()).hexdigest(),
            "polygon_sha256": None if z["polygon"] is None else hashlib.sha256(
                json.dumps(z["polygon"]).encode()).hexdigest(),
        }
        assert got == sig, zid


def test_new_zones_are_ok_non_empty_and_inside_skin(pts):
    labels = _labels(pts)
    zs = _ok(pts, labels)
    skin = labels == int(Region.SKIN)
    for zid in NEW:
        assert (zs[zid].mask <= skin).all(), zid


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_no_overlap_with_eye_class_or_dilated_eye_polygon_and_nested_in_under_eye(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    labels = _labels(p)
    zs = _ok(p, labels)
    o = np.array(ORIGIN, dtype=np.float64)
    for side, k in (("right", "r"), ("left", "l")):
        m = zs[f"lower_eyelid_{k}"].mask
        eye = labels == int(Region.EYE_R if side == "right" else Region.EYE_L)
        assert not (m & eye).any(), k
        poly = f.pts[f.indices(dv.EYE_UPPER + dv.EYE_LOWER[::-1][1:-1], side)]
        assert not (m & mk.dilate(mk.rasterize_polygon(poly - o, SHAPE), 1)).any(), k
        assert (m <= zs[f"under_eye_{k}"].mask).all(), k


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_vertical_ordering_on_face_frame_axes(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    for k in "rl":
        up = np.dot(_centroid(zs[f"eyelid_upper_{k}"]), f.v)
        low = np.dot(_centroid(zs[f"lower_eyelid_{k}"]), f.v)
        ue = np.dot(_centroid(zs[f"under_eye_{k}"]), f.v)
        assert up < low < ue, (k, up, low, ue)


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_are_mirrored(pts, degrees):
    p = _symmetrized(pts)
    if degrees:
        p = _rolled(p, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    mid = (p[468] + p[473]) / 2.0
    r, l = zs["lower_eyelid_r"], zs["lower_eyelid_l"]
    ratio = r.pixel_count / l.pixel_count
    assert 0.75 < ratio < 1.33, ratio
    cr, cl = _centroid(r) - mid, _centroid(l) - mid
    assert abs(np.dot(cr, f.u) + np.dot(cl, f.u)) < 0.1 * f.ipd
    assert abs(np.dot(cr, f.v) - np.dot(cl, f.v)) < 0.1 * f.ipd


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_tolerate_fixture_yaw(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    zs = _ok(p)
    ratio = zs["lower_eyelid_r"].pixel_count / zs["lower_eyelid_l"].pixel_count
    assert 0.65 < ratio < 1.5, ratio


def test_glasses_over_the_eyes_demote_lower_eyelids_like_the_upper_eyelid(pts):
    _ok(pts)
    zs = _derive(pts, _glasses_over(_labels(pts), (Region.EYE_L, Region.EYE_R)))
    for k in "rl":
        z = zs[f"lower_eyelid_{k}"]
        assert z.status == "low_confidence", (k, z.status)
        assert z.reason and z.reason == zs[f"eyelid_upper_{k}"].reason, k


def test_stage_membership_is_the_eye_layer_only():
    found = [(st.key, layer) for st in STAGES for layer, ids in st.layers.items()
             for zid in ids if zid in NEW]
    assert found == [("eyebrow_eye", "eye")] * 2
    assert all(zid in ZONE_SPEC_BY_ID for zid in NEW)


def test_no_crease_zone_exists():
    assert not any("crease" in z.id for z in ZONE_SPECS)
