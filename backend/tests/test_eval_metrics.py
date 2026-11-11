"""Evaluation metrics: pure numpy, no network, no dataset, no model weights."""

import numpy as np
import pytest

from app.services.segmentation.regions import NOT_PARSED, Region
from evaluation import labels as lb
from evaluation import metrics as mt


def _gt_pred(shape=(40, 60)):
    gt = np.zeros(shape, dtype=np.uint8)
    pred = np.full(shape, int(Region.BACKGROUND), dtype=np.uint8)
    return gt, pred


# --- counts / scores ---------------------------------------------------------------------------

def test_perfect_disjoint_partial_and_empty_cases():
    gt, pred = _gt_pred()
    gt[10:20, 10:30] = lb.EP_SKIN
    pred[10:20, 10:30] = int(Region.SKIN)
    conf = mt.confusion_matrix(gt, pred)
    c = mt.counts_from_confusion(conf, (lb.EP_SKIN,), [int(Region.SKIN)])
    assert (c.tp, c.fp, c.fn) == (200, 0, 0)
    assert mt.scores(c.tp, c.fp, c.fn) == {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}

    pred2 = np.full(gt.shape, int(Region.BACKGROUND), dtype=np.uint8)
    pred2[30:35, 40:50] = int(Region.SKIN)  # disjoint
    c = mt.counts_from_confusion(mt.confusion_matrix(gt, pred2), (lb.EP_SKIN,), [int(Region.SKIN)])
    assert (c.tp, c.fp, c.fn) == (0, 50, 200) and mt.scores(*[float(v) for v in (c.tp, c.fp, c.fn)])["iou"] == 0.0

    pred3 = np.full(gt.shape, int(Region.BACKGROUND), dtype=np.uint8)
    pred3[10:20, 10:20] = int(Region.SKIN)  # half of the GT
    c = mt.counts_from_confusion(mt.confusion_matrix(gt, pred3), (lb.EP_SKIN,), [int(Region.SKIN)])
    assert (c.tp, c.fp, c.fn) == (100, 0, 100)
    s = mt.scores(100.0, 0.0, 100.0)
    assert s["iou"] == pytest.approx(0.5) and s["dice"] == pytest.approx(2 / 3) and s["precision"] == 1.0 and s["recall"] == 0.5


def test_empty_gt_and_empty_prediction_scores():
    assert mt.scores(0, 0, 0) == {"iou": None, "dice": None, "precision": None, "recall": None}
    assert mt.scores(0, 5, 0)["precision"] == 0.0 and mt.scores(0, 5, 0)["recall"] is None  # empty GT, false positives
    assert mt.scores(0, 0, 5)["recall"] == 0.0 and mt.scores(0, 0, 5)["precision"] is None  # empty prediction


def test_confusion_derived_counts_equal_direct_mask_computation():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 9, (50, 70)).astype(np.uint8)
    pred = rng.integers(0, 19, (50, 70)).astype(np.uint8)
    pred[rng.random(pred.shape) < 0.2] = NOT_PARSED
    conf = mt.confusion_matrix(gt, pred)
    assert conf.shape == (9, 20) and conf.sum() == gt.size
    for comp in lb.COMPARISONS:
        via_conf = mt.counts_from_confusion(conf, comp.ep_ids, lb.column_ids(comp.canonical))
        direct = mt.counts_from_masks(lb.ep_mask(gt, comp.ep_ids), lb.pred_mask(pred, comp.canonical))
        assert via_conf == direct, comp.name


def test_not_parsed_is_its_own_column_and_never_a_region():
    gt, pred = _gt_pred()
    gt[:] = lb.EP_SKIN
    pred[:, :30] = int(Region.SKIN)
    pred[:, 30:] = NOT_PARSED
    conf = mt.confusion_matrix(gt, pred)
    assert conf[lb.EP_SKIN, lb.NOT_PARSED_COL] == 40 * 30 and conf[lb.EP_SKIN, int(Region.SKIN)] == 40 * 30
    c = mt.counts_from_confusion(conf, (lb.EP_SKIN,), lb.column_ids((Region.SKIN, Region.NOSE)))
    assert (c.tp, c.fp, c.fn) == (1200, 0, 1200)  # not-predicted pixels are misses, not false alarms
    # unsided internal ids (200+) are not canonical either
    pred[:] = 200
    assert mt.confusion_matrix(gt, pred)[lb.EP_SKIN, lb.NOT_PARSED_COL] == gt.size


def test_image_without_prediction_is_all_missed():
    gt, _ = _gt_pred()
    gt[5:15, 5:15] = lb.EP_LIPS
    conf = mt.not_predicted_confusion(gt)
    c = mt.counts_from_confusion(conf, (lb.EP_LIPS,), lb.column_ids((Region.LIP_UPPER, Region.LIP_LOWER)))
    assert (c.tp, c.fp, c.fn) == (0, 0, 100)


def test_full_image_and_window_scoring():
    gt, pred = _gt_pred((40, 60))
    gt[:, :] = lb.EP_SKIN
    pred[:, :] = NOT_PARSED
    pred[10:30, 20:50] = int(Region.SKIN)  # the parse window, 20 x 30
    rect = (20, 10, 50, 30)
    ids, cols = (lb.EP_SKIN,), [int(Region.SKIN)]
    full = mt.counts_from_confusion(mt.confusion_matrix(gt, pred), ids, cols)
    win = mt.counts_from_confusion(mt.confusion_matrix(gt, pred, rect), ids, cols)
    assert (full.tp, full.fp, full.fn) == (600, 0, 2400 - 600)
    assert (win.tp, win.fp, win.fn) == (600, 0, 0)  # outside pixels excluded from GT and prediction


def test_gt_values_out_of_range_rejected():
    gt, pred = _gt_pred()
    gt[0, 0] = 9
    with pytest.raises(ValueError):
        mt.confusion_matrix(gt, pred)
    with pytest.raises(ValueError):
        mt.confusion_matrix(gt[:, :10], pred)


# --- bootstrap and per-image statistics ---------------------------------------------------------------

def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    arr = np.stack([rng.integers(50, 100, 80), rng.integers(0, 20, 80), rng.integers(0, 20, 80)], axis=1)
    a = mt.bootstrap_scores(arr, 300, seed=7)
    b = mt.bootstrap_scores(arr, 300, seed=7)
    c = mt.bootstrap_scores(arr, 300, seed=8)
    assert a == b and a != c
    point = mt.scores(*arr.sum(axis=0).astype(float))["iou"]
    assert a["iou"][0] < point < a["iou"][1]
    assert mt.bootstrap_scores(np.zeros((0, 3)), 10, 0)["iou"] == (None, None)


def test_per_image_stats_use_only_images_with_enough_gt_and_false_alarms():
    counts = [mt.Counts(50, 0, 50), mt.Counts(100, 0, 0), mt.Counts(0, 40, 10), mt.Counts(0, 30, 0), mt.Counts(0, 3, 0)]
    st = mt.per_image_iou_stats(counts)
    assert st["n"] == 2 and st["mean"] == pytest.approx(0.75) and st["median"] == pytest.approx(0.75)
    fa = mt.false_alarm_rate(counts)
    assert (fa["n_absent"], fa["n_false_alarm"]) == (3, 2)  # GT < 25 px: [10 fn], [0], [0]; pred >= 25 in two of them
    assert mt.per_image_iou_stats([])["mean"] is None


# --- boundary F and best shift ------------------------------------------------------------------------------

def _disk(shape, cx, cy, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def test_boundary_f_perfect_empty_and_one_sided():
    g = _disk((100, 100), 50, 50, 20)
    assert mt.boundary_f(g, g, 1.0) == pytest.approx(1.0)
    assert mt.boundary_f(np.zeros_like(g), np.zeros_like(g), 1.0) is None
    assert mt.boundary_f(g, np.zeros_like(g), 1.0) == 0.0
    assert mt.boundary_f(np.zeros_like(g), g, 1.0) == 0.0


def test_boundary_f_drops_as_the_shift_crosses_tau():
    g = _disk((120, 120), 60, 60, 25)
    tau = 3.0
    f = {s: mt.boundary_f(g, _disk((120, 120), 60 + s, 60, 25), tau) for s in (0, 2, 3, 4, 5, 8)}
    assert f[0] == f[2] == f[3] == pytest.approx(1.0)  # every boundary pixel is within tau
    assert f[4] < 0.7 and f[5] < f[4] and f[8] < f[5]  # beyond tau it falls off
    assert mt.boundary_f(g, _disk((120, 120), 65, 60, 25), 6.0) == pytest.approx(1.0)  # larger tau forgives


def test_best_shift_finds_the_offset_and_sign():
    g = _disk((80, 80), 40, 40, 10)
    p = _disk((80, 80), 38, 41, 10)  # prediction is 2 px left and 1 px down of the GT
    dx, dy, iou = mt.best_shift(g, p, 3)
    assert (dx, dy) == (2, -1) and iou > 0.99  # moving the prediction right by 2 and up by 1 fixes it
    assert mt.best_shift(g, g, 3)[:2] == (0, 0)
    assert mt.best_shift(g, np.zeros_like(g)) is None


# --- GT geometry ---------------------------------------------------------------------------------------------------

def test_iod_luma_and_side_order():
    a = np.zeros((50, 100), dtype=bool)
    b = np.zeros_like(a)
    a[20:25, 20:30] = True
    b[20:25, 70:80] = True
    assert mt.inter_ocular_distance(a, b) == pytest.approx(50.0)
    assert mt.side_order_ok(a, b) is True and mt.side_order_ok(b, a) is False
    assert mt.side_order_ok(a, np.zeros_like(a)) is None
    rgb = np.zeros((50, 100, 3), dtype=np.uint8)
    rgb[a] = (255, 255, 255)
    assert mt.mean_luma(rgb, a) == pytest.approx(255.0)
    assert mt.mean_luma(rgb, np.zeros_like(a)) is None


def test_ep_side_mapping_is_image_left_equals_our_right():
    # EP LEFT_* -> our *_R (MediaPipe convention: subject-right is image-left in an unmirrored photo)
    assert lb.COMPARISON_BY_NAME["eye_R"].ep_ids == (lb.EP_LEFT_EYE,) and lb.COMPARISON_BY_NAME["eye_R"].canonical == (Region.EYE_R,)
    assert lb.COMPARISON_BY_NAME["eye_L"].ep_ids == (lb.EP_RIGHT_EYE,) and lb.COMPARISON_BY_NAME["eye_L"].canonical == (Region.EYE_L,)
    assert lb.COMPARISON_BY_NAME["brow_R"].ep_ids == (lb.EP_LEFT_BROW,)
    assert lb.COMPARISON_BY_NAME["brow_L"].canonical == (Region.BROW_L,)
    assert (lb.EP_BACKGROUND, lb.EP_PERSON, lb.EP_SKIN, lb.EP_LEFT_BROW, lb.EP_RIGHT_BROW, lb.EP_LEFT_EYE, lb.EP_RIGHT_EYE, lb.EP_LIPS, lb.EP_TEETH) == tuple(range(9))


def test_side_swap_detection_from_the_confusion_matrix():
    conf = np.zeros((9, 20), dtype=np.int64)
    conf[lb.EP_LEFT_EYE, int(Region.EYE_R)] = 100
    conf[lb.EP_RIGHT_EYE, int(Region.EYE_L)] = 90
    args = (lb.EP_LEFT_EYE, lb.EP_RIGHT_EYE, int(Region.EYE_R), int(Region.EYE_L))
    assert mt.side_swap(conf, *args) is False
    swapped = np.zeros_like(conf)
    swapped[lb.EP_LEFT_EYE, int(Region.EYE_L)] = 100
    swapped[lb.EP_RIGHT_EYE, int(Region.EYE_R)] = 90
    assert mt.side_swap(swapped, *args) is True
    assert mt.side_swap(np.zeros_like(conf), *args) is None
