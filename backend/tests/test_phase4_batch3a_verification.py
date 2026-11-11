"""Phase 4 batch 3a verification: eye-area zones and the bit-identical guarantee for existing zones."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from app.services import face_regions
from app.services.application_zones import derive as dz
from app.services.application_zones.catalog import STAGES, ZONE_SPECS
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.synthetic import synthetic_labels
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401

GOLDEN = json.loads((Path(__file__).parent / "golden_batch3.json").read_text())
SIDES = (("right", "r"), ("left", "l"))
HALVES = [f"under_eye_{n}_{k}" for n in ("inner", "outer") for k in "lr"]
CORNERS = [f"eye_corner_{n}_{k}" for n in ("inner", "outer") for k in "lr"]
LASHES = [f"lash_line_{n}_{k}" for n in ("upper", "lower") for k in "lr"]
NEW = HALVES + CORNERS + LASHES
FORBIDDEN = [int(r) for r in (Region.EYE_L, Region.EYE_R, Region.LIP_UPPER, Region.LIP_LOWER, Region.NOSE)]


def _labels(p):
    return synthetic_labels(p, SHAPE, ORIGIN)


def _sig(z):
    return {
        "status": z.status,
        "reason": z.reason,
        "confidence": None if z.confidence is None else round(float(z.confidence), 6),
        "coverage": None if z.coverage is None else round(float(z.coverage), 6),
        "pixel_count": z.pixel_count if z.mask is not None else None,
        "mask_sha256": None if z.mask is None else hashlib.sha256(z.mask.astype(bool).tobytes()).hexdigest(),
        "polygon_sha256": None if z.polygon is None else hashlib.sha256(
            np.round(z.polygon.astype(np.float64), 4).tobytes()).hexdigest(),
    }


def _scenario(name, pts):
    if name == "synthetic":
        return pts, _labels(pts)
    if name.startswith("roll_"):
        p = _rolled(pts, int(name[5:]))
        return p, _labels(p)
    if name == "glasses_over_eyes":
        labels = _labels(pts)
        labels[np.isin(labels, [int(Region.EYE_L), int(Region.EYE_R)])] = int(Region.GLASSES)
        return pts, labels
    assert name == "background_only"
    return pts, np.zeros(SHAPE, dtype=np.uint8)


@pytest.mark.parametrize("scenario", ["synthetic", "roll_-20", "roll_20", "glasses_over_eyes", "background_only"])
def test_existing_zones_are_bit_identical_to_golden(pts, scenario):
    p, labels = _scenario(scenario, pts)
    zs = _derive(p, labels)
    expected = GOLDEN["scenarios"][scenario]
    assert expected
    for zid, sig in expected.items():
        assert _sig(zs[zid]) == sig, (scenario, zid)


def _proj(c, p, f):
    return float(np.dot(c - p[1], f.u))


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_halves_partition_the_parent_under_eye(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    zs = _derive(p, _labels(p))
    for k in "lr":
        parent = zs[f"under_eye_{k}"].mask
        inner, outer = zs[f"under_eye_inner_{k}"].mask, zs[f"under_eye_outer_{k}"].mask
        assert inner.any() and outer.any() and parent.any()
        assert not (inner & outer).any()
        assert ((inner | outer) == parent).all()


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_containment_and_clean_status(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    labels = _labels(p)
    zs = _derive(p, labels)
    clip = zs["face_skin"].mask & (labels == int(Region.SKIN))
    forbidden = np.isin(labels, FORBIDDEN)
    for zid in NEW:
        assert zs[zid].status == "ok" and zs[zid].mask.any(), (zid, zs[zid].reason)
    for zid in HALVES:
        assert (zs[zid].mask <= clip).all(), zid
        assert not (zs[zid].mask & forbidden).any(), zid
    for side, k in SIDES:
        eye_area = mk.dilate(mk.rasterize_polygon(
            p[f.indices(dz.EYE_UPPER + dz.EYE_LOWER[::-1][1:-1], side)] - np.array(ORIGIN, dtype=np.float64),
            SHAPE), 1)
        for name in ("inner", "outer"):
            assert (zs[f"eye_corner_{name}_{k}"].mask <= eye_area).all()
    for zid in LASHES:
        assert zs[zid].confidence is None and zs[zid].polygon is not None, zid


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_sides_and_inner_outer_order_hold_under_roll(pts, degrees):
    p = _rolled(pts, degrees) if degrees else pts
    f = FaceFrame.from_landmarks(p)
    zs = _derive(p, _labels(p))
    for base in ("under_eye_inner", "under_eye_outer", "eye_corner_inner", "eye_corner_outer",
                 "lash_line_upper", "lash_line_lower"):
        assert _proj(_centroid(zs[f"{base}_l"]), p, f) > 0 > _proj(_centroid(zs[f"{base}_r"]), p, f), base
    for base in ("under_eye", "eye_corner"):
        for k in "lr":
            inner = abs(_proj(_centroid(zs[f"{base}_inner_{k}"]), p, f))
            outer = abs(_proj(_centroid(zs[f"{base}_outer_{k}"]), p, f))
            assert inner < outer, (base, k)


def _glasses_over(labels, regions):
    out = labels.copy()
    out[np.isin(out, [int(r) for r in regions])] = int(Region.GLASSES)
    return out


def test_glasses_over_both_eyes_demote_every_new_zone(pts):
    labels = _labels(pts)
    assert all(_derive(pts, labels)[z].status == "ok" for z in NEW)
    zs = _derive(pts, _glasses_over(labels, (Region.EYE_L, Region.EYE_R)))
    for zid in NEW:
        assert zs[zid].status in ("low_confidence", "unavailable"), zid
        assert zs[zid].reason, zid


def test_glasses_over_one_eye_affects_that_side_only(pts):
    labels = _labels(pts)
    clean = _derive(pts, labels)
    assert all(clean[z].status == "ok" and clean[z].mask.any() for z in NEW)
    zs = _derive(pts, _glasses_over(labels, (Region.EYE_L,)))
    for zid in NEW:
        if zid.endswith("_l"):
            assert zs[zid].status in ("low_confidence", "unavailable") and zs[zid].reason, zid
        else:
            assert zs[zid].status == "ok", zid
            assert zs[zid].mask is not None and (zs[zid].mask == clean[zid].mask).all(), zid


def test_translated_parser_eye_makes_corners_not_ok(pts):
    labels = _labels(pts)
    clean = _derive(pts, labels)
    assert all(clean[z].status == "ok" and clean[z].mask.any() for z in CORNERS)
    ipd = FaceFrame.from_landmarks(pts).ipd
    eyes = np.isin(labels, [int(Region.EYE_L), int(Region.EYE_R)])
    ys, xs = np.nonzero(eyes)
    moved = labels.copy()
    moved[ys, xs] = int(Region.SKIN)
    moved[np.clip(ys + int(round(0.3 * ipd)), 0, SHAPE[0] - 1), xs] = labels[ys, xs]
    zs = _derive(pts, moved)
    for zid in CORNERS:
        assert zs[zid].status in ("low_confidence", "unavailable"), zid
        if zs[zid].status == "low_confidence":
            assert zs[zid].reason
            assert zs[zid].confidence is not None and zs[zid].confidence < dz.MIN_EYE_CORNER_COVERAGE, zid


def test_background_only_halves_are_unavailable_without_mask(pts):
    clean = _derive(pts, _labels(pts))
    assert all(clean[z].status == "ok" and clean[z].mask.any() for z in HALVES)
    zs = _derive(pts, np.zeros(SHAPE, dtype=np.uint8))
    for zid in HALVES:
        assert zs[zid].status == "unavailable" and zs[zid].mask is None and zs[zid].reason, zid


def test_eye_group_zones_follow_the_iris_demotion_predicate():
    """The iris demotion in face_regions selects zones by group and side suffix."""
    src = inspect.getsource(face_regions)
    assert 'z.spec.group == "eye" and z.spec.id.endswith(f"_{side}")' in src

    def selected(side):
        return {z.id for z in ZONE_SPECS if z.group == "eye" and z.id.endswith(f"_{side}")}

    for side, other in (("l", "r"), ("r", "l")):
        sel = selected(side)
        expected = {f"eye_{side}", f"under_eye_{side}", f"eyelid_upper_{side}", f"lower_eyelid_{side}",
                    *(z for z in NEW if z.endswith(f"_{side}"))}
        assert expected == sel
        assert not any(z.endswith(f"_{other}") for z in sel)


def test_new_zones_are_staged_in_the_expected_layers():
    layers = {(st.key, layer): ids for st in STAGES for layer, ids in st.layers.items()}
    for zid in HALVES:
        assert zid in layers[("concealer", "concealer")] and zid in layers[("color_corrector", "color_corrector")]
    for zid in CORNERS + LASHES:
        assert zid in layers[("eyebrow_eye", "eye")]
    assert {"brow_bone_r", "brow_bone_l"} <= set(layers[("highlighter", "highlighter")])
