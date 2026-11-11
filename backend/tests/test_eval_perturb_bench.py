"""Perturbations (image and GT together), consistency scores and benchmark helpers. No weights needed."""

import json

import numpy as np
import pytest

from app.services.segmentation.regions import NOT_PARSED, Region
from evaluation import bench
from evaluation import labels as lb
from evaluation import metrics as mt
from evaluation import perturb as pt
from evaluation.dataset import SampleEntry, split_of


def _scene(h=60, w=90):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[10:20, 10:25] = lb.EP_LEFT_EYE   # image-left eye
    gt[10:20, 60:75] = lb.EP_RIGHT_EYE  # image-right eye
    gt[3:8, 10:25] = lb.EP_LEFT_BROW
    gt[3:8, 60:75] = lb.EP_RIGHT_BROW
    gt[40:50, 30:60] = lb.EP_LIPS
    return rgb, gt


def test_mirror_transforms_image_and_gt_together_and_swaps_sides():
    rgb, gt = _scene()
    r2, g2, valid = pt.apply_perturbation("mirror", rgb, gt)
    assert valid is None and np.array_equal(r2, rgb[:, ::-1])
    # the formerly image-left eye is now on the right and is called RIGHT_EYE (EP LEFT_* = image left)
    assert (g2[10:20, 90 - 25 : 90 - 10] == lb.EP_RIGHT_EYE).all() and (g2[10:20, 90 - 75 : 90 - 60] == lb.EP_LEFT_EYE).all()
    assert (g2[3:8, 90 - 25 : 90 - 10] == lb.EP_RIGHT_BROW).all()
    assert (g2 == lb.EP_LIPS).sum() == (gt == lb.EP_LIPS).sum()
    # mirroring twice is the identity
    r3, g3, _ = pt.apply_perturbation("mirror", r2, g2)
    assert np.array_equal(r3, rgb) and np.array_equal(g3, gt)


def test_prediction_mirror_swaps_sided_regions_like_the_gt():
    pred = np.full((10, 20), 255, dtype=np.uint8)
    pred[2:4, 2:5] = int(Region.EYE_R)
    pred[2:4, 14:17] = int(Region.EYE_L)
    pred[6:8, 8:12] = int(Region.LIP_UPPER)
    t = pt.transform_prediction("mirror", pred)
    assert (t[2:4, 20 - 5 : 20 - 2] == int(Region.EYE_L)).all() and (t[2:4, 20 - 17 : 20 - 14] == int(Region.EYE_R)).all()
    assert (t == int(Region.LIP_UPPER)).sum() == 8


def test_vflip_does_not_swap_sides():
    rgb, gt = _scene()
    r2, g2, _ = pt.apply_perturbation("vflip", rgb, gt)
    assert np.array_equal(g2, gt[::-1]) and np.array_equal(r2, rgb[::-1])
    assert (g2[60 - 20 : 60 - 10, 10:25] == lb.EP_LEFT_EYE).all()


def test_roll_fill_is_not_scored_and_labels_stay_valid():
    rgb, gt = _scene()
    r2, g2, valid = pt.apply_perturbation("roll+30", rgb, gt)
    assert r2.shape[:2] == g2.shape == valid.shape and r2.shape[0] > rgb.shape[0]  # expanded canvas
    assert valid.any() and not valid.all()
    assert not g2[~valid].any()  # fill carries background label 0 but is excluded by `valid`
    assert set(np.unique(g2)) <= set(np.unique(gt))  # nearest interpolation invents no classes
    assert abs(int(valid.sum()) - gt.size) < 0.05 * gt.size  # the valid area is about the original area
    # scoring: a perfect prediction on the valid area has no errors; the fill does not count
    pred = np.where(g2 == lb.EP_LIPS, int(Region.LIP_UPPER), int(Region.BACKGROUND)).astype(np.uint8)
    conf = mt.confusion_matrix(g2, pred, valid=valid)
    assert conf.sum() == valid.sum()
    # a prediction that is garbage only in the fill region changes nothing
    pred2 = pred.copy()
    pred2[~valid] = int(Region.SKIN)
    assert np.array_equal(mt.confusion_matrix(g2, pred2, valid=valid), conf)
    assert mt.confusion_matrix(g2, pred2).sum() == g2.size > conf.sum()


@pytest.mark.parametrize("angle", [15, -15, 45, -45])
def test_roll_rotates_geometry_in_the_stated_direction(angle):
    gt = np.zeros((41, 41), dtype=np.uint8)
    gt[20, 35:39] = lb.EP_SKIN  # a mark to the right of the centre
    g2 = pt.rotate_expand(gt, angle, 0, 0)  # cv2.INTER_NEAREST == 0
    ys, xs = np.nonzero(g2 == lb.EP_SKIN)
    cy, cx = (g2.shape[0] - 1) / 2, (g2.shape[1] - 1) / 2
    # counter-clockwise for positive angles: a mark on the right moves up (smaller y) in image coordinates
    assert (ys.mean() - cy < 0) == (angle > 0)


def test_other_perturbations_change_only_the_image_or_both_consistently():
    rgb, gt = _scene(80, 120)
    r, g, v = pt.apply_perturbation("scale0.5", rgb, gt)
    assert r.shape[:2] == g.shape == (40, 60) and v is None and set(np.unique(g)) <= set(np.unique(gt))
    for name in ("jpeg30", "blur", "gamma0.5", "gamma2.0", "gray"):
        r, g, v = pt.apply_perturbation(name, rgb, gt)
        assert g is gt and v is None and r.shape == rgb.shape and r.dtype == np.uint8
        assert not np.array_equal(r, rgb)
    r, _, _ = pt.apply_perturbation("gray", rgb, gt)
    assert (r[..., 0] == r[..., 1]).all() and (r[..., 1] == r[..., 2]).all()
    dark, bright = pt.apply_perturbation("gamma2.0", rgb, gt)[0], pt.apply_perturbation("gamma0.5", rgb, gt)[0]
    assert dark.mean() < rgb.mean() < bright.mean()
    with pytest.raises(ValueError):
        pt.apply_perturbation("nope", rgb, gt)


def test_equivariance_is_one_for_a_perfectly_equivariant_prediction_and_ignores_fill():
    pred = np.full((50, 80), int(Region.BACKGROUND), dtype=np.uint8)
    pred[10:20, 10:30] = int(Region.EYE_R)
    pred[10:20, 50:70] = int(Region.EYE_L)
    pred[30:40, 30:50] = int(Region.LIP_UPPER)
    mirrored_pred = pt.transform_prediction("mirror", pred)
    eq = pt.equivariance_counts(mirrored_pred, mirrored_pred)
    assert mt.scores(*eq["eye"])["iou"] == 1.0 and eq["eye_R"][0] > 0
    # a predicted region that disagrees lowers the score
    bad = mirrored_pred.copy()
    bad[10:20, 50:70] = int(Region.BACKGROUND)
    assert mt.scores(*pt.equivariance_counts(bad, mirrored_pred)["eye"])["iou"] < 1.0
    # NOT_PARSED pixels (and rotation fill) are excluded rather than counted as disagreement
    unparsed = mirrored_pred.copy()
    unparsed[:, :40] = NOT_PARSED
    assert mt.scores(*pt.equivariance_counts(unparsed, mirrored_pred)["eye"])["iou"] == 1.0


def test_subset_is_seeded_dev_only_and_exact():
    ids = [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(1, 400)]
    entries = [SampleEntry(i, split_of(i), 0, 0) for i in ids]
    a, b = pt.pick_subset(entries, 30), pt.pick_subset(list(reversed(entries)), 30)
    assert a == b and len(a) == 30 and all(e.split == "dev" for e in a)
    assert a != pt.pick_subset(entries, 30, seed=1)
    with pytest.raises(ValueError):
        pt.pick_subset(entries, 10_000)


def test_paired_delta_and_aggregate_on_synthetic_rows():
    def conf(tp):
        c = np.zeros((9, 20), dtype=np.int64)
        c[lb.EP_SKIN, int(Region.SKIN)] = tp
        c[lb.EP_SKIN, lb.NOT_PARSED_COL] = 100 - tp
        for ep, reg in ((lb.EP_LEFT_EYE, Region.EYE_R), (lb.EP_RIGHT_EYE, Region.EYE_L), (lb.EP_LEFT_BROW, Region.BROW_R), (lb.EP_RIGHT_BROW, Region.BROW_L)):
            c[ep, int(reg)] = tp
        c[lb.EP_LIPS, int(Region.LIP_UPPER)] = tp
        return c

    rows = []
    for i in range(20):
        ok_p = i % 10 != 0
        rows.append({"id": str(i), "rows": {
            "baseline": {"outcome": "ok", "conf": conf(90), "shift": {"eye": [0, 2, 0.8], "lips": None, "brow": None}},
            "mirror": {"outcome": "ok" if ok_p else "no_face", "conf": conf(90) if ok_p else conf(0),
                       "shift": {"eye": [0, 2, 0.8], "lips": None, "brow": None},
                       "equiv": {g: [9, 1, 1] for g in pt.EQUIV_GROUPS}},
        }})
    agg = pt.aggregate(rows, ("mirror",))
    m = agg["perturbations"]["mirror"]
    assert m["n_ok"] == 18 and m["failure_rate"] == pytest.approx(0.1)
    assert m["skin_B"] < agg["perturbations"]["baseline"]["skin_B"] and m["skin_A"] == pytest.approx(agg["perturbations"]["baseline"]["skin_A"])
    assert m["skin_B_delta"] < 0 and m["skin_B_delta_ci95"][1] <= 0.0 + 1e-9
    assert m["equivariance"]["eye"]["iou"] == pytest.approx(9 / 11)
    assert agg["perturbations"]["baseline"]["shift"]["eye"] == {"n": 20, "median_dx": 0.0, "median_dy": 2.0}
    json.dumps(agg)


# --- benchmark helpers ---------------------------------------------------------------------------------

def test_ru_maxrss_units():
    assert bench.normalise_maxrss(1_470_382_080, "Darwin") == pytest.approx(1470.38208)  # bytes on macOS
    assert bench.normalise_maxrss(1_435_920, "Linux") == pytest.approx(1470.38208, rel=1e-3)  # kilobytes on Linux
    assert bench.normalise_maxrss(1000, "Darwin") * 1024 == pytest.approx(bench.normalise_maxrss(1000, "Linux"))


def test_worker_output_parsing_ignores_log_noise():
    out = 'W0000 mediapipe noise\nINFO: Created delegate\n{"concurrency": 2, "peak_rss_mb": 900.5}\n'
    assert bench.parse_worker_output(out) == {"concurrency": 2, "peak_rss_mb": 900.5}
    assert bench.parse_worker_output('{"a": 1}\n{"b": 2}\n') == {"b": 2}
    with pytest.raises(ValueError):
        bench.parse_worker_output("no json here\n{broken")


def test_percentiles_and_upscaling_respect_the_pixel_cap():
    p = bench.percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p95"] == pytest.approx(95.05) and p["p99"] == pytest.approx(99.01)
    assert bench.percentiles([]) == {"p50": None, "p95": None, "p99": None}
    img = bench.upscaled_fixture(25.0)
    assert img.shape[0] * img.shape[1] <= bench.MAX_PIXELS
    assert 1.9e6 < (lambda a: a.shape[0] * a.shape[1])(bench.upscaled_fixture(2.0)) < 2.1e6


def test_label_map_cap_is_identity_below_the_cap_and_lossy_above():
    pred = np.full((300, 400), 255, dtype=np.uint8)
    pred[50:250, 100:300] = int(Region.SKIN)
    pred[100:103, 150:153] = int(Region.EYE_R)  # a tiny structure
    assert bench.cap_label_map(pred, None) is pred
    assert np.array_equal(bench.cap_label_map(pred, 2048), pred)
    capped = bench.cap_label_map(pred, 50)
    assert capped.shape == pred.shape and (capped == 255).sum() == (pred == 255).sum()  # parsed rectangle preserved
    assert not np.array_equal(capped, pred) and (capped == int(Region.EYE_R)).sum() != (pred == int(Region.EYE_R)).sum()
