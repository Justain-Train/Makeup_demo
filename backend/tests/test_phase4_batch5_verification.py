"""Phase 4 batch 5 verification: nose zones; the 62 earlier zones stay bit-identical."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.application_zones import derive as dv
from app.services.segmentation import masks as mk
from app.services.application_zones.catalog import STAGES, ZONE_SPEC_BY_ID
from app.services.application_zones.frame import MIDLINE, MIRROR, FaceFrame
from app.services.segmentation.regions import Region
from tests.synthetic import FIXTURES, synthetic_labels
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401
from tests.test_phase4_batch3a_verification import _glasses_over, _sig
from tests.test_phase4_batch3b_verification import _labels, _scenario_b, needs_models
from tests.test_phase4_batch4_verification import _symmetrized

GOLDEN = json.loads((Path(__file__).parent / "golden_batch5.json").read_text())
PAIRS = ["nose_side", "nostril"]
SINGLES = ["nose_bridge", "nose_tip"]
NEW = SINGLES + [f"{b}_{k}" for b in PAIRS for k in "rl"]
SEG = "/api/facial-analysis/segmentation"


def _ok(pts_, labels=None):
    zs = _derive(pts_, _labels(pts_) if labels is None else labels)
    assert all(zs[z].status == "ok" and zs[z].mask is not None and zs[z].mask.any() for z in NEW), \
        [(z, zs[z].status, zs[z].reason) for z in NEW]
    return zs


@pytest.mark.parametrize("scenario", ["synthetic", "roll_-20", "roll_20", "glasses_over_eyes", "glasses_over_brows"])
def test_existing_zones_are_bit_identical_to_golden(pts, scenario):
    p, labels = _scenario_b(scenario, pts)
    zs = _derive(p, labels)
    expected = GOLDEN["scenarios"][scenario]
    assert len(expected) == 62
    for zid, sig in expected.items():
        assert _sig(zs[zid]) == sig, (scenario, zid)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@needs_models
def test_existing_zones_are_identical_through_the_api(client):
    r = client.post(SEG, files={"file": ("p.jpg", (FIXTURES / "portrait.jpg").read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    zs = {z["id"]: z for z in r.json()["zones"]}
    expected = GOLDEN["fixture_api"]
    assert len(expected) == 62
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


def test_nose_hull_is_a_simple_polygon(pts):
    o = np.array(ORIGIN, dtype=np.float64)
    poly = pts[dv.NOSE_HULL]
    hull = cv2.convexHull(poly[:, :2].astype(np.float32)).reshape(-1, 2)
    filled = mk.rasterize_polygon(poly - o, SHAPE).sum()
    assert filled >= 0.95 * mk.rasterize_polygon(hull - o, SHAPE).sum()
    nose = synthetic_labels(pts, SHAPE, ORIGIN) == int(Region.NOSE)
    f = FaceFrame.from_landmarks(pts)
    yy, xx = np.nonzero(nose)
    proj = (xx + ORIGIN[0]) * f.u[0] + (yy + ORIGIN[1]) * f.u[1]
    split = pts[168][0] * f.u[0] + pts[168][1] * f.u[1]
    ratio = (proj < split).sum() / (proj >= split).sum()
    assert 0.75 < ratio < 1.33, ratio


def test_new_zones_are_ok_non_empty_and_inside_the_nose_class(pts):
    labels = _labels(pts)
    zs = _ok(pts, labels)
    nose = labels == int(Region.NOSE)
    for zid in NEW:
        assert (zs[zid].mask <= nose).all(), zid


def test_bridge_tip_and_sides_are_disjoint_and_sides_avoid_the_midline(pts):
    zs = _ok(pts)
    f = FaceFrame.from_landmarks(pts)
    ids = ["nose_bridge", "nose_tip", "nose_side_r", "nose_side_l"]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not (zs[a].mask & zs[b].mask).any(), (a, b)
    a, b = pts[168][:2], pts[5][:2]
    axis = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-axis[1], axis[0]])
    for k in "rl":
        m = zs[f"nose_side_{k}"].mask
        yy, xx = np.nonzero(m)
        xy = np.stack([xx + ORIGIN[0], yy + ORIGIN[1]], axis=1).astype(float)
        dist = np.abs((xy - a) @ normal)
        assert (dist >= 0.03 * f.ipd).all(), k


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_are_mirrored(pts, degrees):
    p = _symmetrized(pts)
    if degrees:
        p = _rolled(p, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    mid = (p[468] + p[473]) / 2.0
    for b in PAIRS:
        r, l = zs[f"{b}_r"], zs[f"{b}_l"]
        ratio = r.pixel_count / l.pixel_count
        assert 0.75 < ratio < 1.33, (b, ratio)
        cr, cl = _centroid(r) - mid, _centroid(l) - mid
        assert abs(np.dot(cr, f.u) + np.dot(cl, f.u)) < 0.1 * f.ipd, b
        assert abs(np.dot(cr, f.v) - np.dot(cl, f.v)) < 0.1 * f.ipd, b
        assert np.dot(cr, f.u) < 0 < np.dot(cl, f.u), b


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_tolerate_fixture_yaw(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    zs = _ok(p)
    for b in PAIRS:
        ratio = zs[f"{b}_r"].pixel_count / zs[f"{b}_l"].pixel_count
        assert 0.65 < ratio < 1.5, (b, ratio)


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_singles_are_centred_on_the_face_midline(pts, degrees):
    p = _symmetrized(pts)
    if degrees:
        p = _rolled(p, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    mid = (p[468] + p[473]) / 2.0
    for zid in SINGLES:
        assert abs(np.dot(_centroid(zs[zid]) - mid, f.u)) < 0.1 * f.ipd, zid


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_zone_ordering_on_roll_aware_axes(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    c = {zid: _centroid(zs[zid]) for zid in NEW}
    assert np.dot(c["nose_bridge"], f.v) < np.dot(c["nose_tip"], f.v)
    assert np.dot(c["nose_side_r"], f.u) < np.dot(c["nose_bridge"], f.u) < np.dot(c["nose_side_l"], f.u)
    for k in "rl":
        assert np.dot(c[f"nostril_{k}"], f.v) > np.dot(c[f"nose_side_{k}"], f.v), k
    assert np.dot(c["nostril_r"], f.u) < np.dot(c["nose_tip"], f.u) < np.dot(c["nostril_l"], f.u)


def test_glasses_over_the_eyes_only_keep_the_bridge_ok(pts):
    _ok(pts)
    zs = _derive(pts, _glasses_over(_labels(pts), (Region.EYE_L, Region.EYE_R)))
    assert zs["nose_bridge"].status == "ok", (zs["nose_bridge"].status, zs["nose_bridge"].reason)


def test_glasses_over_the_bridge_demote_only_the_bridge(pts):
    labels = _labels(pts)
    zs = _ok(pts, labels)
    f = FaceFrame.from_landmarks(pts)
    yy, xx = np.nonzero(zs["nose_bridge"].mask)
    proj = (yy + ORIGIN[1]) * f.v[1] + (xx + ORIGIN[0]) * f.v[0]
    upper = proj <= np.quantile(proj, 0.4)
    painted = labels.copy()
    painted[yy[upper], xx[upper]] = int(Region.GLASSES)
    out = _derive(pts, painted)
    assert out["nose_bridge"].status == "low_confidence"
    assert "Glasses" in out["nose_bridge"].reason
    for zid in ("nose_tip", "nostril_r", "nostril_l"):
        assert out[zid].status == "ok", (zid, out[zid].status, out[zid].reason)


def test_background_only_zones_are_unavailable(pts):
    _ok(pts)
    zs = _derive(pts, np.zeros(SHAPE, dtype=np.uint8))
    for zid in NEW:
        assert zs[zid].status == "unavailable" and zs[zid].mask is None and zs[zid].reason, zid


def test_stage_layers_are_exact_nostrils_unstaged_and_ids_exist():
    layers = {(st.key, layer): ids for st in STAGES for layer, ids in st.layers.items()}
    assert layers[("bronzer_contour", "contour")] == [
        "jawline", "contour_hollow_r", "contour_hollow_l", "nose_side_r", "nose_side_l"]
    assert layers[("highlighter", "highlighter")] == [
        "brow_bone_r", "brow_bone_l", "cheekbone_r", "cheekbone_l", "nose_bridge", "nose_tip"]
    for ids in layers.values():
        assert "nostril_r" not in ids and "nostril_l" not in ids
        assert all(i in ZONE_SPEC_BY_ID for i in ids)


def test_new_spec_landmarks_are_mirrored_or_midline():
    specs = [dv.NOSE_BRIDGE_SPEC, dv.NOSE_TIP_SPEC, dv.NOSE_SIDE_SPEC, [dv.NOSTRIL_CENTRE]]
    for spec in specs:
        for i, *_ in spec:
            assert i in MIRROR or i in MIDLINE, i
            if i in MIRROR:
                assert MIRROR[MIRROR[i]] == i
