"""Phase 4 batch 3b verification: brow parts, upper eyelid, brow bone; existing zones stay bit-identical."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.application_zones import derive as dz
from app.services.application_zones.catalog import STAGES
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.synthetic import FIXTURES, synthetic_labels
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401
from tests.test_phase4_batch3a_verification import _glasses_over, _scenario, _sig

GOLDEN = json.loads((Path(__file__).parent / "golden_batch3b.json").read_text())
SIDES = (("right", "r"), ("left", "l"))
PARTS = ("head", "arch", "tail")
BROW_PARTS = [f"brow_{n}_{k}" for n in PARTS for k in "lr"]
SKIN_ZONES = [f"{n}_{k}" for n in ("eyelid_upper", "brow_bone") for k in "lr"]
NEW = BROW_PARTS + SKIN_ZONES
SEG = "/api/facial-analysis/segmentation"
MODELS = Path(__file__).resolve().parents[1] / "models"
needs_models = pytest.mark.skipif(
    not ((MODELS / "face_landmarker.task").exists() and (MODELS / "face_parser_resnet18.onnx").exists()),
    reason="model weights not downloaded",
)


def _labels(p):
    return synthetic_labels(p, SHAPE, ORIGIN)


def _scenario_b(name, pts):
    if name == "glasses_over_brows":
        return pts, _glasses_over(_labels(pts), (Region.BROW_L, Region.BROW_R))
    return _scenario(name, pts)


def _clean_ok(pts, ids):
    zs = _derive(pts, _labels(pts))
    assert all(zs[z].status == "ok" and zs[z].mask is not None and zs[z].mask.any() for z in ids), \
        [(z, zs[z].status, zs[z].reason) for z in ids]
    return zs


@pytest.mark.parametrize("scenario", ["synthetic", "roll_-20", "roll_20", "glasses_over_eyes", "glasses_over_brows"])
def test_existing_zones_are_bit_identical_to_golden(pts, scenario):
    p, labels = _scenario_b(scenario, pts)
    zs = _derive(p, labels)
    expected = GOLDEN["scenarios"][scenario]
    assert len(expected) == 20
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
    assert len(expected) == 20
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


def _proj(c, p, f):
    return float(np.dot(c - p[1], f.u))


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_brow_parts_partition_the_brow_polygon_in_order(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    zs = _derive(p, _labels(p))
    f = FaceFrame.from_landmarks(p)
    for side, k in SIDES:
        parts = [zs[f"brow_{n}_{k}"] for n in PARTS]
        assert all(z.status == "ok" and z.mask.any() for z in parts), (k, [z.reason for z in parts])
        poly = p[f.indices(dz.BROW_UPPER + dz.BROW_LOWER[::-1], side)]
        brow = mk.rasterize_polygon(poly - np.array(ORIGIN, dtype=np.float64), SHAPE)
        union = parts[0].mask | parts[1].mask | parts[2].mask
        assert (union == brow).all()
        assert not (parts[0].mask & parts[1].mask).any()
        assert not (parts[0].mask & parts[2].mask).any()
        assert not (parts[1].mask & parts[2].mask).any()
        i_head, i_tail = f.indices([107, 70], side)
        axis = p[i_tail] - p[i_head]
        s = [float(np.dot(_centroid(z) - p[i_head], axis)) for z in parts]
        assert s[0] < s[1] < s[2], (k, s)


def test_brow_parts_lie_within_tolerance_of_the_parser_brow(pts):
    labels = _labels(pts)
    zs = _clean_ok(pts, BROW_PARTS)
    ipd = FaceFrame.from_landmarks(pts).ipd
    for region, k in ((Region.BROW_R, "r"), (Region.BROW_L, "l")):
        near = mk.dilate(labels == int(region), dz.BROW_PART_TOLERANCE * ipd)
        union = zs[f"brow_head_{k}"].mask | zs[f"brow_arch_{k}"].mask | zs[f"brow_tail_{k}"].mask
        assert (union <= near).all(), k


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_eyelid_and_brow_bone_are_disjoint_from_brow_eye_and_each_other(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    labels = _labels(p)
    zs = _derive(p, labels)
    clip = zs["face_skin"].mask & (labels == int(Region.SKIN))
    for side, k in SIDES:
        lid, bone = zs[f"eyelid_upper_{k}"], zs[f"brow_bone_{k}"]
        assert lid.status == "ok" and bone.status == "ok" and lid.mask.any() and bone.mask.any(), k
        eye = labels == int(Region.EYE_R if side == "right" else Region.EYE_L)
        brow = labels == int(Region.BROW_R if side == "right" else Region.BROW_L)
        poly = p[f.indices(dz.BROW_UPPER + dz.BROW_LOWER[::-1], side)] - np.array(ORIGIN, dtype=np.float64)
        brow_poly = mk.rasterize_polygon(poly, SHAPE)
        others = {"eye": eye, "brow": brow, "brow_polygon": brow_poly}
        for name, other in others.items():
            assert not (lid.mask & other).any(), (k, "eyelid", name)
            assert not (bone.mask & other).any(), (k, "bone", name)
        assert not (lid.mask & bone.mask).any(), k
        assert (lid.mask <= clip).all() and (bone.mask <= clip).all(), k


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_sides_and_head_tail_order_hold_under_roll(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    zs = _derive(p, _labels(p))
    for base in ("brow_head", "brow_arch", "brow_tail", "eyelid_upper", "brow_bone"):
        assert _proj(_centroid(zs[f"{base}_l"]), p, f) > 0 > _proj(_centroid(zs[f"{base}_r"]), p, f), base
    for k in "lr":
        head = abs(_proj(_centroid(zs[f"brow_head_{k}"]), p, f))
        tail = abs(_proj(_centroid(zs[f"brow_tail_{k}"]), p, f))
        assert head < tail, k


def test_glasses_over_eyes_demote_the_eyelid(pts):
    _clean_ok(pts, SKIN_ZONES)
    zs = _derive(pts, _glasses_over(_labels(pts), (Region.EYE_L, Region.EYE_R)))
    for k in "lr":
        z = zs[f"eyelid_upper_{k}"]
        assert z.status in ("low_confidence", "unavailable") and z.reason, k


def test_glasses_over_brows_demote_brow_zones_but_not_the_eyelid(pts):
    clean = _clean_ok(pts, NEW)
    zs = _derive(pts, _glasses_over(_labels(pts), (Region.BROW_L, Region.BROW_R)))
    for zid in BROW_PARTS + ["brow_bone_l", "brow_bone_r"]:
        assert zs[zid].status in ("low_confidence", "unavailable") and zs[zid].reason, zid
    for k in "lr":
        assert zs[f"eyelid_upper_{k}"].status == "ok"
        assert (zs[f"eyelid_upper_{k}"].mask == clean[f"eyelid_upper_{k}"].mask).all()


def test_translated_parser_brow_makes_brow_parts_not_ok(pts):
    labels = _labels(pts)
    _clean_ok(pts, BROW_PARTS)
    ipd = FaceFrame.from_landmarks(pts).ipd
    ys, xs = np.nonzero(np.isin(labels, [int(Region.BROW_L), int(Region.BROW_R)]))
    moved = labels.copy()
    moved[ys, xs] = int(Region.SKIN)
    moved[np.clip(ys + int(round(0.3 * ipd)), 0, SHAPE[0] - 1), xs] = labels[ys, xs]
    zs = _derive(pts, moved)
    for zid in BROW_PARTS:
        assert zs[zid].status in ("low_confidence", "unavailable"), zid
        if zs[zid].status == "low_confidence":
            assert zs[zid].reason and "coverage within a tolerance" in zs[zid].reason, zid


def test_background_only_skin_zones_are_unavailable_without_mask(pts):
    _clean_ok(pts, SKIN_ZONES)
    zs = _derive(pts, np.zeros(SHAPE, dtype=np.uint8))
    for zid in SKIN_ZONES:
        assert zs[zid].status == "unavailable" and zs[zid].mask is None and zs[zid].reason, zid


def test_eye_area_guard_without_parent_still_gates_on_the_eye(pts):
    labels = _labels(pts)
    clean = _clean_ok(pts, ["lash_line_upper_r", "eye_r", "brow_r"])
    d = dz.ZoneDeriver(FaceFrame.from_landmarks(pts), labels, np.full(labels.shape, 0.9, np.float32), ORIGIN)
    d._zones = {"eye_r": dataclasses.replace(clean["eye_r"], status="low_confidence"), "brow_r": clean["brow_r"]}
    base = clean["lash_line_upper_r"]
    demoted = d._eye_area_guard(dataclasses.replace(base), "r", base.mask)
    assert demoted.status == "low_confidence"
    assert demoted.reason == "The eye is not parsed or disagrees with the landmarks."
    kept = d._eye_area_guard(dataclasses.replace(base), "r", base.mask, parent="brow_r")
    assert kept.status == "ok"


def test_new_zones_are_staged_in_the_expected_layers():
    layers = {(st.key, layer): ids for st in STAGES for layer, ids in st.layers.items()}
    assert {"brow_bone_r", "brow_bone_l"} <= set(layers[("highlighter", "highlighter")])
    assert layers[("eyebrow_eye", "eyebrow")][:2] == ["brow_r", "brow_l"]
    assert layers[("eyebrow_eye", "eyebrow")][2:] == [f"brow_{n}_{k}" for k in "rl" for n in PARTS]
    assert layers[("eyebrow_eye", "eye")][-4:] == ["eyelid_upper_r", "eyelid_upper_l", "lower_eyelid_r", "lower_eyelid_l"]
