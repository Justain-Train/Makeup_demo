"""Phase 4 batch 1 verification (criteria 3-5). Reuses helpers from test_application_zones."""

import numpy as np
import pytest

from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.synthetic import synthetic_labels
from tests.test_application_zones import ORIGIN, SHAPE, _centroid, _derive, _rolled, pts  # noqa: F401

SKIN_CLIPPED = ("chin_center", "under_lower_lip", "mouth_surround")
CORNERS = ("lip_corner_l", "lip_corner_r")
FORBIDDEN = [int(r) for r in (Region.EYE_L, Region.EYE_R, Region.LIP_UPPER, Region.LIP_LOWER, Region.NOSE)]


@pytest.fixture(scope="module")
def zones(pts):
    return _derive(pts, synthetic_labels(pts, SHAPE, ORIGIN))


def test_skin_clipped_zones_are_skin_subsets_disjoint_from_eye_lip_nose(pts, zones):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    skin = zones["face_skin"].mask & (labels == int(Region.SKIN))
    forbidden = np.isin(labels, FORBIDDEN)
    for zid in SKIN_CLIPPED:
        z = zones[zid]
        assert z.status == "ok", (zid, z.reason)
        assert z.mask.any(), zid
        assert (z.mask <= skin).all(), zid
        assert not (z.mask & forbidden).any(), zid


def test_cupid_bow_and_lip_corners_stay_in_dilated_lip_area(pts, zones):
    area = mk.dilate(zones["lip_full"].mask, 0.15 * FaceFrame.from_landmarks(pts).ipd)
    for zid in ("cupid_bow", *CORNERS):
        assert zones[zid].mask.any(), zid
        assert (zones[zid].mask <= area).all(), zid


def test_mouth_surround_is_disjoint_from_lip_full(zones):
    assert not (zones["mouth_surround"].mask & zones["lip_full"].mask).any()


@pytest.mark.parametrize("degrees", [-20, 20])
def test_lip_corner_sides_hold_under_20_degree_roll(pts, degrees):
    p = _rolled(pts, degrees)
    f = FaceFrame.from_landmarks(p)
    zs = _derive(p, synthetic_labels(p, SHAPE, ORIGIN))
    assert zs["lip_corner_l"].status == "ok" and zs["lip_corner_r"].status == "ok"
    cl, cr = _centroid(zs["lip_corner_l"]), _centroid(zs["lip_corner_r"])
    assert np.dot(cl - cr, f.u) > 0, degrees
    assert np.linalg.norm(cl - p[291]) < 0.3 * f.ipd and np.linalg.norm(cr - p[61]) < 0.3 * f.ipd


def test_skin_clipped_zones_unavailable_on_background_only_labels(pts):
    zs = _derive(pts, np.full(SHAPE, int(Region.BACKGROUND), dtype=np.uint8))
    for zid in SKIN_CLIPPED:
        z = zs[zid]
        assert z.status == "unavailable" and z.mask is None and z.reason, zid
