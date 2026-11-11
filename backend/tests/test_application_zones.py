"""Zone engine: face frame / MIRROR, zone properties, catalog and stages."""

import numpy as np
import pytest
from mediapipe.tasks.python import vision

from app.services.application_zones import derive as dz
from app.services.application_zones.catalog import STAGES, ZONE_SPEC_BY_ID, ZONE_SPECS
from app.services.application_zones.frame import MIDLINE, MIRROR, MIRROR_PAIRS, FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region
from tests.synthetic import fixture_landmarks, synthetic_labels

ORIGIN = (140, 0)
SHAPE = (480, 500)


@pytest.fixture(scope="module")
def pts():
    return fixture_landmarks()[0]


@pytest.fixture(scope="module")
def frame(pts):
    return FaceFrame.from_landmarks(pts)


def _derive(pts, labels):
    conf = np.full(labels.shape, 0.9, dtype=np.float32)
    return {z.spec.id: z for z in dz.derive_zones(FaceFrame.from_landmarks(pts), labels, conf, ORIGIN)}


# --- frame / MIRROR ----------------------------------------------------------------

def test_mirror_is_an_involution_and_midline_is_fixed():
    assert all(MIRROR[MIRROR[i]] == i for i in MIRROR)
    assert len({r for r, _ in MIRROR_PAIRS} | {l for _, l in MIRROR_PAIRS}) == 2 * len(MIRROR_PAIRS)
    assert not set(MIDLINE) & set(MIRROR)


def test_mirror_pairs_are_symmetric_about_the_midline_on_a_real_face(pts, frame):
    a, b = pts[10], pts[152]
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    for r, l in MIRROR_PAIRS:
        mid_offset = abs(np.dot((pts[r] + pts[l]) / 2 - a, n)) / frame.ipd
        along = abs(np.dot(pts[l] - pts[r], t)) / frame.ipd
        assert mid_offset < 0.1 and along < 0.1, (r, l)  # a wrong pair is off by > 0.3
    assert all(abs(np.dot(pts[i] - a, n)) / frame.ipd < 0.1 for i in MIDLINE)


def test_mirror_matches_mediapipe_eye_brow_lip_connections():
    C = vision.FaceLandmarksConnections

    def idx(conns):
        return {c.start for c in conns} | {c.end for c in conns}

    right_eye = set(dz.EYE_UPPER + dz.EYE_LOWER)
    assert {MIRROR[i] for i in right_eye} == idx(C.FACE_LANDMARKS_LEFT_EYE)
    assert right_eye == idx(C.FACE_LANDMARKS_RIGHT_EYE)
    right_brow = set(dz.BROW_UPPER + dz.BROW_LOWER)
    assert {MIRROR[i] for i in right_brow} == idx(C.FACE_LANDMARKS_LEFT_EYEBROW)
    assert right_brow == idx(C.FACE_LANDMARKS_RIGHT_EYEBROW)
    lips = set(dz.LIP_OUTER_UPPER + dz.LIP_OUTER_LOWER + dz.LIP_INNER_UPPER + dz.LIP_INNER_LOWER)
    assert lips == idx(C.FACE_LANDMARKS_LIPS)
    assert all(MIRROR[i] in lips for i in lips if i in MIRROR)


def test_frame_axes_follow_head_roll(pts):
    f0 = FaceFrame.from_landmarks(pts)
    assert abs(f0.roll_degrees) < 5 and f0.v[1] > 0.9  # v points down the face
    c = pts.mean(axis=0)
    ang = np.radians(30)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    f1 = FaceFrame.from_landmarks((pts - c) @ rot.T + c)
    assert f1.roll_degrees == pytest.approx(f0.roll_degrees + 30, abs=0.01)
    assert f1.ipd == pytest.approx(f0.ipd)
    # the right-side point() and the mirrored left-side point() are symmetric about the face axis
    r, l = f0.point(33, du=-0.5), f0.point(33, du=-0.5, side="left")
    assert np.dot(r - f0.pts[33], f0.u) < 0 < np.dot(l - f0.pts[263], f0.u)


# --- zone properties ---------------------------------------------------------------

BATCH1_SKIN_ZONES = {"chin_center", "under_lower_lip", "mouth_surround"}
BATCH2_SKIN_ZONES = {"forehead_center", "forehead_perimeter", "temple_l", "temple_r",
                     "jawline_outer_l", "jawline_outer_r"}
BATCH3A_HALVES = {"under_eye_inner_l", "under_eye_inner_r", "under_eye_outer_l", "under_eye_outer_r"}
EYE_CORNERS = {"eye_corner_inner_l", "eye_corner_inner_r", "eye_corner_outer_l", "eye_corner_outer_r"}
CLIPPED_TO_SKIN = ({"forehead", "cheek_l", "cheek_r", "chin", "jawline", "under_eye_l", "under_eye_r"}
                   | BATCH1_SKIN_ZONES | BATCH2_SKIN_ZONES | BATCH3A_HALVES
                   | {"eyelid_upper_l", "eyelid_upper_r", "brow_bone_l", "brow_bone_r"})
LASH_LINES = {"lash_line_upper_l", "lash_line_upper_r", "lash_line_lower_l", "lash_line_lower_r"}
GEOMETRIC_ONLY = {"lip_outer_boundary", "lip_inner_boundary", "cupid_bow"} | LASH_LINES


def test_phase1_zones_are_valid_and_subsets_of_their_clip_regions(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    zones = _derive(pts, labels)
    assert set(zones) == set(ZONE_SPEC_BY_ID)
    skin = labels == int(Region.SKIN)
    face_skin = zones["face_skin"].mask
    assert face_skin is not None and (face_skin <= (skin | (labels == int(Region.NOSE)))).all()
    h, w = labels.shape
    for zid, z in zones.items():
        assert z.status == "ok", (zid, z.reason)
        assert z.mask is not None and z.mask.dtype == bool and z.mask.shape == labels.shape
        assert z.pixel_count == int(z.mask.sum()) >= dz.MIN_ZONE_PIXELS
        x, y, bw, bh = z.bbox  # image coordinates, inside the label map
        assert ORIGIN[0] <= x and x + bw <= ORIGIN[0] + w and 0 <= y and y + bh <= h
        assert mk.bbox_of(z.mask) == (x - ORIGIN[0], y - ORIGIN[1], bw, bh)
        if zid in CLIPPED_TO_SKIN:
            assert (z.mask <= skin).all() and (z.mask <= face_skin).all(), zid
    for side, region in (("l", Region.EYE_L), ("r", Region.EYE_R)):
        assert (zones[f"eye_{side}"].mask == (labels == int(region))).all()
    assert (zones["nose"].mask == (labels == int(Region.NOSE))).all()
    # face zones never cover the eyes, lips or nose class
    forbidden = np.isin(labels, [int(r) for r in (Region.EYE_L, Region.EYE_R, Region.LIP_UPPER, Region.LIP_LOWER, Region.NOSE)])
    for zid in CLIPPED_TO_SKIN:
        assert not (zones[zid].mask & forbidden).any(), zid


def test_left_and_right_zones_are_on_the_subjects_sides(pts, frame):
    zones = _derive(pts, synthetic_labels(pts, SHAPE, ORIGIN))
    for name in ("cheek", "under_eye", "eye", "brow"):
        cl = np.mean(np.nonzero(zones[f"{name}_l"].mask)[1]) + ORIGIN[0]
        cr = np.mean(np.nonzero(zones[f"{name}_r"].mask)[1]) + ORIGIN[0]
        # subject's left is on the image's RIGHT (larger x) in an unmirrored photo
        assert cl > pts[1][0] > cr, name


def test_low_overlap_is_flagged_not_hidden(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[: int(pts[105][1]) + 5] = int(Region.HAIR)  # bangs: hair down to the brows
    z = _derive(pts, labels)["forehead"]
    assert z.status in ("low_confidence", "unavailable")
    assert z.status == "unavailable" or z.coverage < dz.MIN_COVERAGE
    assert z.reason
    assert z.extra["hair_fraction"] > 0.5


def test_zone_without_usable_pixels_is_unavailable_with_no_mask(pts):
    labels = np.full(SHAPE, int(Region.BACKGROUND), dtype=np.uint8)
    zones = _derive(pts, labels)
    for zid, z in zones.items():
        if zid.startswith("lip_") and zid not in GEOMETRIC_ONLY:
            continue  # landmark polygons are validated against the parser, still have pixels
        if zid in LASH_LINES or zid in EYE_CORNERS or zid.startswith(("brow_head", "brow_arch", "brow_tail")):
            # geometric / tolerance-checked, but demoted because the parent eye is not parsed
            assert z.status == "low_confidence" and z.reason, zid
            continue
        if zid in GEOMETRIC_ONLY:
            assert z.status == "ok"  # purely geometric
            continue
        assert z.status == "unavailable" and z.mask is None and z.pixel_count == 0 and z.reason, zid
    assert all(zones[z].status == "low_confidence"
               for z in ("lip_upper", "lip_lower", "lip_full", "lip_corner_l", "lip_corner_r"))
    assert all(zones[z].status == "unavailable" and zones[z].mask is None and zones[z].reason
               for z in BATCH1_SKIN_ZONES)
    assert zones["lip_upper"].confidence == 0.0  # IoU with an empty parser mask


def test_parser_landmark_disagreement_lowers_confidence(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    labels[labels == int(Region.EYE_L)] = int(Region.SKIN)
    z = _derive(pts, labels)["eye_l"]
    assert z.status == "unavailable"
    # eye mask shifted far from the landmark polygon -> low_confidence, not ok
    shifted = synthetic_labels(pts, SHAPE, ORIGIN)
    ys, xs = np.nonzero(shifted == int(Region.EYE_L))
    shifted[ys, xs] = int(Region.SKIN)
    shifted[ys + 40, xs] = int(Region.EYE_L)
    assert _derive(pts, shifted)["eye_l"].status == "low_confidence"


def test_boundary_zones_are_geometric_only(pts):
    zones = _derive(pts, synthetic_labels(pts, SHAPE, ORIGIN))
    for zid in ("lip_outer_boundary", "lip_inner_boundary"):
        assert zones[zid].confidence is None and zones[zid].polygon is not None


# --- catalog -----------------------------------------------------------------------

def test_stages_reference_catalog_zones_and_every_zone_is_staged():
    ids = {z.id for z in ZONE_SPECS}
    used = {zid for st in STAGES for zs in st.layers.values() for zid in zs}
    assert used <= ids, used - ids  # no dangling references
    assert used == ids - {"nostril_r", "nostril_l"}, ids - used  # nostrils are unstaged exclusion areas
    assert [st.number for st in STAGES] == list(range(1, 12))
    assert len({st.key for st in STAGES}) == 11
    by_key = {st.key: st for st in STAGES}
    assert list(by_key["bronzer_contour"].layers) == ["contour", "bronzer"]  # separate layers
    assert by_key["bronzer_contour"].layers["bronzer"] == []
    assert list(by_key["lips"].layers) == ["lipstick", "lip_liner", "gloss"]
    assert by_key["blush"].layers["blush"] == ["cheek_apple_r", "cheek_apple_l", "cheek_upper_r", "cheek_upper_l",
                                               "cheek_outer_r", "cheek_outer_l"]
    assert by_key["highlighter"].layers["highlighter"] == ["brow_bone_r", "brow_bone_l", "cheekbone_r", "cheekbone_l",
                                                           "nose_bridge", "nose_tip"]
    assert "eyelid_crease" not in ids


# --- Phase 4 batch 1 -----------------------------------------------------------------

def _rolled(pts, degrees):
    c = pts.mean(axis=0)
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts - c) @ rot.T + c


def _centroid(z):
    ys, xs = np.nonzero(z.mask)
    return np.array([xs.mean() + ORIGIN[0], ys.mean()])


@pytest.mark.parametrize("degrees", [0, 20, -20])
def test_lip_corners_are_on_the_subjects_sides_including_roll(pts, degrees):
    p = _rolled(pts, degrees)
    f = FaceFrame.from_landmarks(p)
    zones = _derive(p, synthetic_labels(p, SHAPE, ORIGIN))
    cl, cr = _centroid(zones["lip_corner_l"]), _centroid(zones["lip_corner_r"])
    assert np.dot(cl - cr, f.u) > 0  # subject's left lies further along u (right eye -> left eye)
    assert np.linalg.norm(cl - p[291]) < 0.3 * f.ipd and np.linalg.norm(cr - p[61]) < 0.3 * f.ipd


def test_lip_corner_uses_tolerance_coverage_not_iou(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    ipd = FaceFrame.from_landmarks(pts).ipd
    lips = np.isin(labels, [int(Region.LIP_UPPER), int(Region.LIP_LOWER)])
    ys, xs = np.nonzero(lips)
    shift = int(round(0.03 * ipd))
    near = labels.copy()
    near[ys, xs] = int(Region.SKIN)
    near[np.clip(ys + shift, 0, labels.shape[0] - 1), xs] = labels[ys, xs]
    zones = _derive(pts, near)
    assert zones["lip_corner_l"].status == "ok" and zones["lip_corner_r"].status == "ok"

    absent = labels.copy()
    absent[lips] = int(Region.SKIN)
    zones = _derive(pts, absent)
    for zid in ("lip_corner_l", "lip_corner_r"):
        assert zones[zid].status in ("low_confidence", "unavailable"), zid
        assert zones[zid].reason


def test_lip_corner_is_not_ok_when_parser_lip_is_translated_by_0_3_ipd(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    ipd = FaceFrame.from_landmarks(pts).ipd
    lips = np.isin(labels, [int(Region.LIP_UPPER), int(Region.LIP_LOWER)])
    ys, xs = np.nonzero(lips)
    moved = labels.copy()
    moved[ys, xs] = int(Region.SKIN)
    moved[np.clip(ys + int(round(0.3 * ipd)), 0, labels.shape[0] - 1), xs] = labels[ys, xs]
    zones = _derive(pts, moved)
    for zid in ("lip_corner_l", "lip_corner_r"):
        z = zones[zid]
        assert z.status in ("low_confidence", "unavailable"), zid
        assert z.reason, zid
        if z.status == "low_confidence":
            assert z.confidence is not None and z.confidence < dz.MIN_LIP_CORNER_COVERAGE, zid
            assert "coverage within a tolerance" in z.reason, zid


def test_batch1_zones_are_valid_and_cupid_bow_is_geometric_only(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    zones = _derive(pts, labels)
    for zid in ("chin_center", "under_lower_lip", "mouth_surround", "cupid_bow", "lip_corner_l", "lip_corner_r"):
        assert zones[zid].status == "ok", (zid, zones[zid].reason)
    assert zones["cupid_bow"].confidence is None and zones["cupid_bow"].polygon is not None


def test_batch1_skin_zones_are_clipped_and_disjoint_from_eye_lip_nose(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    zones = _derive(pts, labels)
    clip = zones["face_skin"].mask & (labels == int(Region.SKIN))
    forbidden = np.isin(labels, [int(r) for r in (Region.EYE_L, Region.EYE_R, Region.LIP_UPPER,
                                                   Region.LIP_LOWER, Region.NOSE)])
    for zid in BATCH1_SKIN_ZONES:
        assert (zones[zid].mask <= clip).all(), zid
        assert not (zones[zid].mask & forbidden).any(), zid


def test_mouth_surround_is_disjoint_from_lip_full(pts):
    zones = _derive(pts, synthetic_labels(pts, SHAPE, ORIGIN))
    assert not (zones["mouth_surround"].mask & zones["lip_full"].mask).any()


def test_batch1_skin_zones_unavailable_on_background_only_labels(pts):
    zones = _derive(pts, np.full(SHAPE, int(Region.BACKGROUND), dtype=np.uint8))
    for zid in BATCH1_SKIN_ZONES:
        z = zones[zid]
        assert z.status == "unavailable" and z.mask is None and z.reason, zid


def test_catalog_has_no_model_specific_class_names():
    for z in ZONE_SPECS:
        assert not any(tok in " ".join(z.clipped_to) for tok in ("l_eye", "r_eye", "u_lip", "l_lip", "l_brow"))


# --- side assignment (landmark convention) and the wrong-side backstop ----------------

def test_assign_sides_follows_landmark_geometry_including_roll(pts):
    from app.services.segmentation.regions import Unsided
    from app.services.segmentation.sides import assign_sides

    for ang in (0, 30, -40):
        c = pts.mean(axis=0)
        r = np.radians(ang)
        rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        p = (pts - c) @ rot.T + c
        origin = (int(p[:, 0].min()) - 20, int(p[:, 1].min()) - 20)
        labels = np.zeros((int(p[:, 1].max()) - origin[1] + 20, int(p[:, 0].max()) - origin[0] + 20), np.uint8)
        for idx in (468, 473):  # one blob around each iris
            x, y = np.round(p[idx]).astype(int) - origin
            labels[y - 3 : y + 4, x - 3 : x + 4] = int(Unsided.EYE)
        out = assign_sides(labels, p, origin)
        x, y = np.round(p[468]).astype(int) - origin
        assert out[y, x] == int(Region.EYE_R)
        x, y = np.round(p[473]).astype(int) - origin
        assert out[y, x] == int(Region.EYE_L)
        assert not np.isin(out, [int(u) for u in Unsided]).any()


def test_wrong_side_mask_is_never_ok(pts):
    labels = synthetic_labels(pts, SHAPE, ORIGIN)
    both = (labels == int(Region.EYE_L)) | (labels == int(Region.EYE_R))
    labels[both] = int(Region.EYE_L)  # one mask covering both eyes
    zones = _derive(pts, labels)
    assert zones["eye_l"].status == "low_confidence" and "side mismatch" in zones["eye_l"].reason
