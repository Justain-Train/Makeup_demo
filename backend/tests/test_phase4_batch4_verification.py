"""Phase 4 batch 4 verification: cheek zones; the 52 earlier zones stay bit-identical."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.application_zones.catalog import STAGES, ZONE_SPEC_BY_ID
from app.services.application_zones.frame import MIRROR, FaceFrame
from app.services.segmentation.regions import Region
from tests.synthetic import FIXTURES
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401
from tests.test_phase4_batch3a_verification import _glasses_over, _sig
from tests.test_phase4_batch3b_verification import _labels, _scenario_b, needs_models

GOLDEN = json.loads((Path(__file__).parent / "golden_batch4.json").read_text())
BASES = ("cheek_apple", "cheek_upper", "cheek_outer", "cheekbone", "contour_hollow")
NEW = [f"{b}_{k}" for b in BASES for k in "rl"]
SEG = "/api/facial-analysis/segmentation"
BATCH4_LANDMARKS = [33, 133, 234, 93, 61, 129]


def _symmetrized(p: np.ndarray) -> np.ndarray:
    """Average each landmark with the reflection of its mirror partner across the eye-midline axis, removing the fixture's yaw."""
    p = np.asarray(p, float).copy()
    f = FaceFrame.from_landmarks(p)
    mid = (p[468, :2] + p[473, :2]) / 2
    u = np.asarray(f.u[:2], float)
    u /= np.linalg.norm(u)
    xy = p[:, :2] - mid
    refl = xy - 2 * np.outer(xy @ u, u)
    m = ((refl[:, None, :] - xy[None, :, :]) ** 2).sum(-1).argmin(1)  # nearest mirror landmark
    p[:, :2] = mid + (xy + (xy[m] - 2 * np.outer(xy[m] @ u, u))) / 2
    return p


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
    assert len(expected) == 52
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
    assert len(expected) == 52
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


def test_new_zones_are_ok_and_non_empty_on_the_clean_face(pts):
    _ok(pts)


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_are_mirrored(pts, degrees):
    p = _symmetrized(pts)  # the test uses the yaw-free face
    if degrees:
        p = _rolled(p, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    mid = (p[468] + p[473]) / 2.0
    for b in BASES:
        r, l = zs[f"{b}_r"], zs[f"{b}_l"]
        ratio = r.pixel_count / l.pixel_count
        assert 0.75 < ratio < 1.33, (b, ratio)
        cr, cl = _centroid(r) - mid, _centroid(l) - mid
        assert abs(np.dot(cr, f.u) + np.dot(cl, f.u)) < 0.1 * f.ipd, b
        assert abs(np.dot(cr, f.v) - np.dot(cl, f.v)) < 0.1 * f.ipd, b
        assert np.dot(cr, f.u) < 0 < np.dot(cl, f.u), b


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_left_and_right_tolerate_fixture_yaw(pts, degrees):
    # The real fixture is slightly yawed (234/454 are 0.926 asymmetric). The outer cheek and temple
    # strips next to the face edge exaggerate this (1.34 on the fixture vs 0.975 once the yaw is
    # removed); this test bounds how much real yaw may skew them.
    p = _rolled(pts, degrees) if degrees else pts
    zs = _ok(p)
    for b in BASES:
        ratio = zs[f"{b}_r"].pixel_count / zs[f"{b}_l"].pixel_count
        assert 0.65 < ratio < 1.5, (b, ratio)


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_zone_ordering_on_roll_aware_axes(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    zs = _ok(p)
    for k in "rl":
        sign = -1.0 if k == "r" else 1.0  # medial is toward the face midline

        def c(base):
            return _centroid(zs[f"{base}_{k}"])

        def down(base):
            return float(np.dot(c(base), f.v))

        def medial(base):
            return sign * -float(np.dot(c(base), f.u))

        assert down("cheekbone") < down("contour_hollow") < float(np.dot(_centroid(zs["jawline"]), f.v)), k
        assert down("cheek_apple") > down("cheekbone"), k
        assert medial("cheek_apple") > medial("cheekbone"), k
        assert medial("cheek_outer") < medial("cheek_apple"), k
        assert down("cheek_upper") < down("cheek_apple"), k


def test_cheekbone_and_hollow_are_disjoint(pts):
    zs = _ok(pts)
    for k in "rl":
        assert not (zs[f"cheekbone_{k}"].mask & zs[f"contour_hollow_{k}"].mask).any(), k


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_masks_are_skin_and_avoid_eye_brow_nose_lips_under_eye(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    labels = _labels(p)
    zs = _ok(p, labels)
    skin = labels == int(Region.SKIN)
    for zid in NEW:
        m = zs[zid].mask
        assert (m <= skin).all(), zid
        k = zid[-1]
        for other in (f"eye_{k}", f"brow_{k}", "nose", "lip_full", f"under_eye_{k}"):
            assert not (m & zs[other].mask).any(), (zid, other)


def test_glasses_over_eyes_demote_upper_cheek_and_cheekbone(pts):
    _ok(pts)
    zs = _derive(pts, _glasses_over(_labels(pts), (Region.EYE_L, Region.EYE_R)))
    for base in ("cheek_upper", "cheekbone"):
        for k in "rl":
            z = zs[f"{base}_{k}"]
            assert z.status == "low_confidence" and z.reason, (base, k, z.status)


def test_background_only_zones_are_unavailable(pts):
    _ok(pts)
    zs = _derive(pts, np.zeros(SHAPE, dtype=np.uint8))
    for zid in NEW:
        assert zs[zid].status == "unavailable" and zs[zid].mask is None and zs[zid].reason, zid


def test_stage_layers_are_exact_and_ids_exist():
    layers = {(st.key, layer): ids for st in STAGES for layer, ids in st.layers.items()}
    assert layers[("bronzer_contour", "contour")] == [
        "jawline", "contour_hollow_r", "contour_hollow_l", "nose_side_r", "nose_side_l"]
    assert layers[("bronzer_contour", "bronzer")] == [
        "temple_r", "temple_l", "forehead_perimeter", "cheek_outer_r", "cheek_outer_l",
        "jawline_outer_r", "jawline_outer_l"]
    assert layers[("blush", "blush")] == ["cheek_apple_r", "cheek_apple_l", "cheek_upper_r", "cheek_upper_l",
                                          "cheek_outer_r", "cheek_outer_l"]
    assert layers[("highlighter", "highlighter")] == [
        "brow_bone_r", "brow_bone_l", "cheekbone_r", "cheekbone_l", "nose_bridge", "nose_tip",
        "forehead_center", "chin_center", "eye_corner_inner_r", "eye_corner_inner_l", "cupid_bow"]
    for ids in layers.values():
        assert all(i in ZONE_SPEC_BY_ID for i in ids)


def test_mirror_map_covers_the_new_landmarks():
    for i in BATCH4_LANDMARKS:
        assert i in MIRROR or i in (10, 152), i
        if i in MIRROR:
            assert MIRROR[MIRROR[i]] == i
