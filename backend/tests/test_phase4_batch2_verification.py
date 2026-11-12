"""Phase 4 batch 2 verification. Reuses helpers from test_application_zones."""

import numpy as np
import pytest

from app.services.application_zones import derive as dz
from app.services.application_zones.catalog import STAGES
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.synthetic import synthetic_labels
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401

BATCH2 = ("forehead_center", "forehead_perimeter", "temple_l", "temple_r", "jawline_outer_l", "jawline_outer_r")
SIDED = ("temple", "jawline_outer")
FORBIDDEN = [int(r) for r in (Region.EYE_L, Region.EYE_R, Region.LIP_UPPER, Region.LIP_LOWER, Region.NOSE,
                              Region.HAIR, Region.BACKGROUND, Region.EAR_L, Region.EAR_R)]


@pytest.fixture(scope="module")
def zones(pts):
    return _derive(pts, synthetic_labels(pts, SHAPE, ORIGIN))


def test_batch2_zones_are_ok_on_synthetic_labels(zones):
    for zid in BATCH2:
        assert zones[zid].status == "ok", (zid, zones[zid].reason)
        assert zones[zid].mask.any(), zid


def test_batch2_zones_are_skin_subsets_disjoint_from_non_skin_regions(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[: int(pts[10][1]) + 10] = int(Region.HAIR)
    labels[:, : max(0, int(pts[234][0] - ORIGIN[0]))] = int(Region.BACKGROUND)
    labels[:, int(pts[454][0] - ORIGIN[0]) :] = int(Region.EAR_L)
    zs = _derive(pts, labels)
    clip = zs["face_skin"].mask & (labels == int(Region.SKIN))
    forbidden = np.isin(labels, FORBIDDEN)
    for zid in BATCH2:
        m = zs[zid].mask
        if m is None:
            continue
        assert (m <= clip).all(), zid
        assert not (m & forbidden).any(), zid


def test_forehead_center_is_inside_the_forehead_polygon_and_overlaps_its_mask(pts, zones):
    fh = zones["forehead"]
    inside = mk.rasterize_polygon(fh.polygon - np.array(ORIGIN, dtype=np.float64), SHAPE)
    fc = zones["forehead_center"].mask
    assert (fc <= inside).all()
    assert (fc & fh.mask).any()


def test_temples_are_disjoint_from_forehead_center(zones):
    for k in ("l", "r"):
        assert not (zones[f"temple_{k}"].mask & zones["forehead_center"].mask).any()


@pytest.mark.parametrize("degrees", [0, -20, 20])
def test_sides_follow_the_subject_including_roll(pts, degrees):
    p = _rolled(pts, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _derive(p, synthetic_labels(p, SHAPE, ORIGIN))
    for name in SIDED:
        zl, zr = zs[f"{name}_l"], zs[f"{name}_r"]
        assert zl.mask is not None and zr.mask is not None, (name, degrees)
        assert np.dot(_centroid(zl) - p[1], f.u) > 0 > np.dot(_centroid(zr) - p[1], f.u), (name, degrees)
    if degrees == 0:
        for name in SIDED:
            assert _centroid(zs[f"{name}_l"])[0] > p[1][0] > _centroid(zs[f"{name}_r"])[0], name


def test_fringe_makes_forehead_center_and_perimeter_not_ok(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[: int(pts[105][1]) + 5] = int(Region.HAIR)
    zs = _derive(pts, labels)
    for zid in ("forehead_center", "forehead_perimeter"):
        z = zs[zid]
        assert z.status in ("low_confidence", "unavailable"), zid
        assert z.reason, zid
        assert z.extra["hair_fraction"] > 0.5, zid
        if z.status == "unavailable":
            assert z.mask is None, zid


def test_hair_over_one_temple_affects_that_side_only(pts, zones):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    poly = zones["temple_r"].polygon - np.array(ORIGIN, dtype=np.float64)
    labels[mk.rasterize_polygon(poly, SHAPE)] = int(Region.HAIR)
    zs = _derive(pts, labels)
    assert zs["temple_r"].status in ("low_confidence", "unavailable") and zs["temple_r"].reason
    assert zs["temple_l"].status == "ok"
    assert (zs["temple_l"].mask == zones["temple_l"].mask).all()


def test_jawline_outer_unavailable_without_skin_on_the_jaw(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    ipd = FaceFrame.from_landmarks(pts).ipd
    labels[int(min(pts[132][1], pts[361][1]) - 0.2 * ipd) :] = int(Region.BACKGROUND)
    zs = _derive(pts, labels)
    for zid in ("jawline_outer_l", "jawline_outer_r"):
        assert zs[zid].status == "unavailable" and zs[zid].mask is None and zs[zid].reason, zid
    bg = _derive(pts, np.full(SHAPE, int(Region.BACKGROUND), dtype=np.uint8))
    for zid in BATCH2:
        assert bg[zid].status == "unavailable" and bg[zid].mask is None and bg[zid].reason, zid


def test_zones_with_low_skin_coverage_are_never_ok(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    ipd = FaceFrame.from_landmarks(pts).ipd
    brow_mid_y = (pts[105][1] + pts[334][1]) / 2.0
    top = pts[10][1] - 0.35 * abs(pts[10][1] - brow_mid_y)  # top of the forehead_center quad
    bottom = pts[9][1] - 0.15 * ipd  # bottom of the forehead_center quad
    skin_from = int(top + 0.7 * (bottom - top))  # skin starts 70% down the central forehead
    skin_to = int(min(pts[132][1], pts[361][1]) + 0.15 * ipd)  # skin ends just below the jaw start
    labels[:skin_from] = int(Region.BACKGROUND)
    labels[skin_to:] = int(Region.BACKGROUND)
    checked = 0
    for zid, z in _derive(pts, labels).items():
        if z.coverage is not None and z.coverage < dz.MIN_COVERAGE:
            checked += 1
            assert z.status != "ok", zid
    assert checked >= 2


@pytest.mark.parametrize("k", ["l", "r"])
def test_jawline_outer_is_in_the_outer_part_of_the_jaw_band(pts, zones, k):
    f = FaceFrame.from_landmarks(pts)
    oval_mean = pts[dz.OVAL].mean(axis=0)
    jaw = zones["jawline"].mask
    ys, xs = np.nonzero(jaw)
    sign = 1.0 if k == "l" else -1.0
    pix = np.stack([xs + ORIGIN[0], ys], axis=1).astype(np.float64)
    side = sign * ((pix - pts[1]) @ f.u) > 0
    jaw_centroid = pix[side].mean(axis=0)
    outer_centroid = _centroid(zones[f"jawline_outer_{k}"])
    assert np.linalg.norm(outer_centroid - oval_mean) > np.linalg.norm(jaw_centroid - oval_mean)

    side_name = "left" if k == "l" else "right"
    path = pts[f.indices(dz.JAW_OUTER_PATH_R, side_name)] - np.array(ORIGIN, dtype=np.float64)
    oys, oxs = np.nonzero(zones[f"jawline_outer_{k}"].mask)
    p = np.stack([oxs, oys], axis=1).astype(np.float64)
    best = np.full(len(p), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    assert best.max() <= dz.JAW_OUTER_BAND * f.ipd + 1.5
    outside = zones[f"jawline_outer_{k}"].mask & ~jaw
    assert int(outside.sum()) <= 0.01 * int(zones[f"jawline_outer_{k}"].mask.sum())


def _bg(pts):
    return np.zeros(SHAPE, dtype=np.uint8)


def _fringe(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[: int(pts[10][1]) + 25] = int(Region.HAIR)
    return labels


def _to_brows(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[: int((pts[105][1] + pts[334][1]) / 2)] = int(Region.HAIR)
    return labels


@pytest.mark.parametrize("degrees,build,status,pixels", [
    (0, lambda p: synthetic_labels(p, SHAPE, ORIGIN), "ok", 8548),
    (-20, lambda p: synthetic_labels(p, SHAPE, ORIGIN), "ok", 8539),
    (20, lambda p: synthetic_labels(p, SHAPE, ORIGIN), "ok", 8578),
    (0, _fringe, "low_confidence", 2692),
    (0, _to_brows, "unavailable", None),
    (0, _bg, "unavailable", None),
])
def test_forehead_zone_matches_pre_refactor_golden(pts, degrees, build, status, pixels):
    p = _rolled(pts, degrees) if degrees else pts
    z = _derive(p, build(p))["forehead"]
    assert z.status == status
    if pixels is None:
        assert z.mask is None
    else:
        assert int(z.mask.sum()) == pixels


def test_batch2_ids_appear_only_in_their_expected_stages():
    where = {zid: set() for zid in BATCH2}
    for st in STAGES:
        for ids in st.layers.values():
            for zid in ids:
                if zid in where:
                    where[zid].add(st.key)
    bronze = {"primer", "foundation", "bronzer_contour", "setting_spray"}
    expected = {
        "forehead_center": {"primer", "foundation", "highlighter", "setting_spray"},
        "forehead_perimeter": bronze,
        "temple_r": bronze,
        "temple_l": bronze,
        "jawline_outer_r": bronze,
        "jawline_outer_l": bronze,
    }
    assert set(expected) == set(BATCH2)
    for zid, keys in where.items():
        assert keys == expected[zid], zid
    contour = next(st for st in STAGES if st.key == "bronzer_contour")
    assert contour.layers["contour"][0] == "jawline"
