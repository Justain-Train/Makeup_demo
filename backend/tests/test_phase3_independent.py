"""Independent Phase 3 completion checks (testing agent, 2026-11-10).

Written separately from the coder's tests so that they can fail independently:
  * registration: a noise-free float32 edge, an ideal-resampling least-squares shift estimator, and an
    injected-shift sensitivity check (the estimator must see a 0.2 px error that the 0.1 px limit rejects);
  * truncated landmarks through both endpoints, with log-content checks;
  * evaluation harness: EP side mapping, confusion-vs-mask identity, NOT_PARSED, both scorings, denominators,
    bootstrap, boundary F across tau, best-shift sign, side swap;
  * fetch and cache guards with in-memory archives and fake HTTP openers (no network);
  * committed report regeneration and sample determinism (skipped when the local cache is absent).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import urllib.error
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.config import get_settings
from app.main import create_app
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.segmentation.face_parser import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OnnxFaceParser,
    ParserInfo,
    compute_parse_window,
)
from app.services.segmentation.regions import MODEL_ID_TO_CANONICAL, NOT_PARSED, Region
from evaluation import dataset as ds
from evaluation import labels as lb
from evaluation import metrics as mt
from evaluation.remote_zip import (
    BudgetedFetcher,
    FetchError,
    Head,
    HttpsRangeFetcher,
    _NoRedirect,
    fetch_member,
    read_central_directory,
)
from tests.synthetic import FakeParser, FakeLandmarkService

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "docs" / "evaluation"
SETTINGS = get_settings()


# =====================================================================================
# 1. Registration (WP4a) - independent sub-pixel measurement
# =====================================================================================

MID = (127.5 - float(IMAGENET_MEAN[0])) / float(IMAGENET_STD[0])
EDGE_SIGMA = 10.0  # px, logistic edge width in the original image
REG_LIMIT_PX = 0.1

# (image width, image height, face bbox). Window = 2 * bbox side, clipped by the image.
REG_GEOMETRIES = {
    "square 600 == image": (600, 600, (150, 150, 450, 450)),
    "non-square 700x500, window past right/bottom": (700, 500, (450, 300, 650, 450)),
    "non-square 900x700, window past left/top": (900, 700, (40, 60, 340, 420)),
    "odd 1001x777, window past left/top": (1001, 777, (120, 90, 421, 437)),
    "pre-shrink 1300x1200, window past right/bottom": (1300, 1200, (500, 400, 1100, 1000)),
    "pre-shrink 1777x1500 odd, window past left/top": (1777, 1500, (300, 300, 1000, 1000)),
    "pre-shrink 2650 == image": (2650, 2650, (662, 662, 1988, 1988)),
    "pre-shrink 3001x2400, window past top-left": (3001, 2400, (20, 30, 1220, 1230)),
    "window 1026 (> MAX_WORK_SIZE), pre-shrink": (1200, 1200, (300, 300, 813, 813)),
}
# Edge phases avoid exact ties (an edge at x.5 gives pixels of exactly 127.5, where the label is ambiguous).
REG_FRACS = (0.1, 0.3, 0.6, 0.9)


def _logistic(c: np.ndarray, centre: float) -> np.ndarray:
    return 255.0 / (1.0 + np.exp(-(c - centre) / EDGE_SIGMA))


class _EchoParser(OnnxFaceParser):
    """Network stand-in: class 1 (skin) is the input's channel 0 vs. a constant mid-grey logit.

    `inject_net_px` (network pixels) resamples the echoed input along the measured axis, to prove the
    shift estimator can see a known offset.
    """

    def __init__(self, axis: int = 1, inject_net_px: float = 0.0):
        self.info = ParserInfo("stub", "stub", "0" * 64, 512)
        self._init_pipeline(2.0, 1, 1.0)
        self.axis, self.inject_net_px = axis, inject_net_px
        self.last_input: np.ndarray | None = None

    def _infer(self, x: np.ndarray) -> np.ndarray:
        ch = x[0, 0].astype(np.float64)
        if self.inject_net_px:
            pos = np.arange(512, dtype=np.float64) - self.inject_net_px  # content moves right/down
            i0 = np.clip(np.floor(pos).astype(int), 0, 511)
            i1 = np.clip(i0 + 1, 0, 511)
            f = np.clip(pos - np.floor(pos), 0.0, 1.0)
            if self.axis == 1:
                ch = ch[:, i0] * (1 - f) + ch[:, i1] * f
            else:
                ch = ch[i0, :] * (1 - f)[:, None] + ch[i1, :] * f[:, None]
        self.last_input = ch.astype(np.float32)
        out = np.zeros((19, 512, 512), dtype=np.float32)
        out[0] = MID
        out[1] = ch
        return out


def _edge_image(w: int, h: int, centre: float, axis: int) -> np.ndarray:
    """float32 RGB, pixel p (continuous centre p + 0.5) = logistic(p + 0.5 - centre); no quantisation."""
    n = w if axis == 1 else h
    prof = _logistic(np.arange(n) + 0.5, centre).astype(np.float32)
    plane = np.broadcast_to(prof[None, :], (h, w)) if axis == 1 else np.broadcast_to(prof[:, None], (h, w))
    return np.repeat(plane[:, :, None], 3, axis=2).copy()


def _estimate_shift(v: np.ndarray, X: np.ndarray, centre: float, valid: np.ndarray) -> float:
    """Least-squares shift (original px, + = right/down) of the echoed profile against the ideal edge.

    Model: the network pixel at continuous image coordinate X sees logistic(X - shift - centre).
    A grid search over shift, refined by a parabola through the minimum.
    """
    deltas = np.linspace(-1.5, 1.5, 3001)
    vv, XX = v[valid], X[valid]
    pred = _logistic(XX[None, :] - deltas[:, None], centre)
    err = ((vv[None, :] - pred) ** 2).mean(axis=1)
    i = int(np.argmin(err))
    if 0 < i < deltas.size - 1:
        e0, e1, e2 = err[i - 1], err[i], err[i + 1]
        denom = e0 - 2 * e1 + e2
        if denom > 0:
            return float(deltas[i] + 0.5 * (e0 - e2) / denom * (deltas[1] - deltas[0]))
    return float(deltas[i])


def _measure(w: int, h: int, bbox, axis: int, centre: float, inject_orig_px: float = 0.0):
    """(estimated network-input shift in original px, label-map first-index error in px, window size)."""
    parser = _EchoParser(axis=axis, inject_net_px=0.0)
    window = compute_parse_window(bbox, 2.0)
    k_in = 512 / window.size
    parser.inject_net_px = inject_orig_px * k_in
    res = parser.parse(_edge_image(w, h, centre, axis), bbox)
    n = w if axis == 1 else h
    origin = window.x if axis == 1 else window.y
    v = parser.last_input[256] if axis == 1 else parser.last_input[:, 256]
    v = v.astype(np.float64) * float(IMAGENET_STD[0]) + float(IMAGENET_MEAN[0])  # denormalise channel 0
    d = np.arange(512)
    X = origin + (d + 0.5) / k_in  # continuous image coordinate of each network pixel centre
    s = X - 0.5  # sample index in the image
    valid = (s >= 2) & (s <= n - 3) & (v > 5) & (v < 250)
    assert valid.sum() >= 10, "not enough usable transition samples"
    shift = _estimate_shift(v, X, centre, valid)

    lab = res.labels
    row = lab[lab.shape[0] // 2] if axis == 1 else lab[:, lab.shape[1] // 2]
    first = int(np.nonzero(row == int(Region.SKIN))[0][0]) + (res.origin[0] if axis == 1 else res.origin[1])
    expected = int(np.argmax(_logistic(np.arange(n) + 0.5, centre) > 127.5))
    return shift, first - expected, window.size


@pytest.mark.parametrize("name", list(REG_GEOMETRIES))
def test_network_input_shift_is_within_0p1_px(name):
    w, h, bbox = REG_GEOMETRIES[name]
    worst = 0.0
    for axis in (1, 0):
        n = w if axis == 1 else h
        for frac in REG_FRACS:
            centre = n / 2 + 3.0 * (1 if axis else -1) + frac
            shift, _, _ = _measure(w, h, bbox, axis, centre)
            worst = max(worst, abs(shift))
            assert abs(shift) <= REG_LIMIT_PX, f"{name} axis={'x' if axis else 'y'} frac={frac}: shift {shift:+.3f}"
    assert worst < 0.02  # the fix is much tighter than the limit; a regression to ~0.3 px is caught


def test_label_map_edge_is_within_1px_with_no_systematic_sign():
    errors, exact_small = [], []
    for name, (w, h, bbox) in REG_GEOMETRIES.items():
        for axis in (1, 0):
            n = w if axis == 1 else h
            for frac in REG_FRACS:
                centre = n / 2 + 3.0 * (1 if axis else -1) + frac
                _, err, win = _measure(w, h, bbox, axis, centre)
                assert abs(err) <= 1, f"{name} axis={axis} frac={frac}: label edge error {err:+d}px"
                errors.append(err)
                if win <= 1024:  # work resolution == window: the label map is an exact integer shift
                    exact_small.append((name, err))
    assert abs(float(np.mean(errors))) < 0.25, f"systematic label offset: mean {np.mean(errors):+.3f}"
    bad = [e for e in exact_small if e[1] != 0]
    assert not bad, bad


@pytest.mark.parametrize("axis", [1, 0])
def test_estimator_sees_an_injected_0p2_px_shift(axis):
    """Sensitivity: an injected 0.2 px error (above the 0.1 px limit) must be measured as ~0.2 px, with sign."""
    w, h, bbox = REG_GEOMETRIES["pre-shrink 2650 == image"]
    n = w if axis == 1 else h
    centre = n / 2 + 3.0 + 0.3
    plus, _, _ = _measure(w, h, bbox, axis, centre, inject_orig_px=+0.2)
    minus, _, _ = _measure(w, h, bbox, axis, centre, inject_orig_px=-0.2)
    assert abs(plus - 0.2) < 0.03, plus
    assert abs(minus + 0.2) < 0.03, minus


# =====================================================================================
# 2. Truncated landmarks (WP4a) - both endpoints, exact message, count-only logging
# =====================================================================================

INCOMPLETE = "Face landmarks were incomplete; please retake the photo with your whole face visible."


def _service_with_points(n: int) -> FaceLandmarkService:
    svc = FaceLandmarkService.__new__(FaceLandmarkService)  # real service logic, stubbed native landmarker
    from app.schemas.facial_landmarks import DetectionThresholds
    import threading as _t

    svc._secondary_ratio = 0.20
    svc._thresholds = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    svc._lock = _t.Lock()
    face = [SimpleNamespace(x=0.3 + (i % 17) * 0.011, y=0.3 + (i % 13) * 0.013, z=0.0) for i in range(n)]
    # one detected face with n points (n may be 0: the truncated case, not "no face")
    svc._landmarker = SimpleNamespace(detect=lambda _img: SimpleNamespace(face_landmarks=[face]))
    return svc


def _client(svc, parser) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: svc
    app.dependency_overrides[get_face_parser] = lambda: parser
    return TestClient(app)


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 240), (120, 110, 100)).save(buf, format="PNG")
    return buf.getvalue()


class _RaisingParser:
    """Must never be called on a truncated result."""

    info = FakeParser.info

    def parse(self, image, face_bbox):  # pragma: no cover - failure path
        raise AssertionError("parser must not run on an incomplete landmark set")


@pytest.mark.parametrize("n", [0, 1, 477])
def test_truncated_landmarks_status_on_both_endpoints(n, caplog):
    svc = _service_with_points(n)
    caplog.set_level(logging.DEBUG)
    client = _client(svc, _RaisingParser())
    png = _png_bytes()

    lm = client.post("/api/facial-analysis/landmarks", files={"file": ("a.png", png, "image/png")})
    assert lm.status_code == 200
    body = lm.json()
    assert body["status"] == "no_face_detected"
    assert body["face_detected"] is False and body["face_count"] == 0 and body["landmark_count"] == 0
    assert body["landmarks"] == [] and body["message"] == INCOMPLETE

    seg = client.post("/api/facial-analysis/segmentation", files={"file": ("a.png", png, "image/png")})
    assert seg.status_code == 200, seg.text
    sb = seg.json()
    assert sb["status"] == "no_face_detected" and sb["message"] == INCOMPLETE
    assert sb["anatomical"] is None and sb["zones"] == []
    assert sb["landmarks"]["landmark_count"] == 0 and sb["landmarks"]["landmarks"] == []

    # Logs: the warning may exist, but it carries only a count (no coordinates, no floats, no image data).
    for rec in caplog.records:
        if "incomplete" in rec.getMessage().lower():
            assert all(isinstance(a, (int, str)) for a in rec.args), rec.args
            assert "0.3" not in rec.getMessage() and "." not in rec.getMessage().split("(")[-1]


def test_478_landmarks_are_unchanged_on_both_endpoints():
    svc = _service_with_points(478)
    client = _client(svc, FakeParser())
    png = _png_bytes()
    lm = client.post("/api/facial-analysis/landmarks", files={"file": ("a.png", png, "image/png")}).json()
    assert lm["status"] == "ok" and lm["face_detected"] is True and lm["landmark_count"] == 478
    assert lm["message"] is None and len(lm["landmarks"]) == 478
    seg = client.post("/api/facial-analysis/segmentation", files={"file": ("a.png", png, "image/png")})
    assert seg.status_code == 200 and seg.json()["status"] == "ok"


def test_479_points_do_not_crash_and_are_documented_as_ok():
    """MediaPipe never returns more than 478 points; the code accepts >= 478 and keeps the first 478 indices.

    This is a documentation check: the schema says 478, the code reports the actual count.
    """
    svc = _service_with_points(479)
    r = svc.detect_landmarks(np.zeros((240, 200, 3), dtype=np.uint8))
    assert r.status == "ok" and r.landmark_count == 479 and len(r.landmarks) == 479


def test_parser_unavailable_is_503_even_when_landmarks_are_truncated():
    svc = _service_with_points(0)
    client = _client(svc, None)
    seg = client.post("/api/facial-analysis/segmentation", files={"file": ("a.png", _png_bytes(), "image/png")})
    # the route checks parser availability first (documented 503), so truncation cannot surface as a 500
    assert seg.status_code == 503


# =====================================================================================
# 3. Evaluation harness: labels, metrics, both scorings, denominators, bootstrap, boundary F
# =====================================================================================

def test_ep_side_mapping_is_image_left_equals_our_right():
    cmp = lb.COMPARISON_BY_NAME
    assert cmp["eye_R"].ep_ids == (lb.EP_LEFT_EYE,) and cmp["eye_R"].canonical == (Region.EYE_R,)
    assert cmp["eye_L"].ep_ids == (lb.EP_RIGHT_EYE,) and cmp["eye_L"].canonical == (Region.EYE_L,)
    assert cmp["brow_R"].ep_ids == (lb.EP_LEFT_BROW,) and cmp["brow_R"].canonical == (Region.BROW_R,)
    assert cmp["brow_L"].ep_ids == (lb.EP_RIGHT_BROW,) and cmp["brow_L"].canonical == (Region.BROW_L,)
    assert cmp["eye"].ep_ids == (lb.EP_LEFT_EYE, lb.EP_RIGHT_EYE)
    assert cmp["brow"].ep_ids == (lb.EP_LEFT_BROW, lb.EP_RIGHT_BROW)
    assert cmp["skin"].ep_ids == (lb.EP_SKIN,) and set(cmp["skin"].canonical) == {Region.SKIN, Region.NOSE}
    assert cmp["lips"].ep_ids == (lb.EP_LIPS,) and set(cmp["lips"].canonical) == {Region.LIP_UPPER, Region.LIP_LOWER}


def test_labels_module_does_not_import_the_model_mapping():
    import ast

    tree = ast.parse((REPO / "backend" / "evaluation" / "labels.py").read_text())
    imported = {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for a in node.names}
    assert "MODEL_ID_TO_CANONICAL" not in imported and "build_label_lut" not in imported
    assert "MODEL_ID_TO_CANONICAL" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_sided_eval_detects_a_true_swap_and_not_a_match():
    """Image-left EP LEFT_EYE should be scored against our EYE_R (image-left) and not against EYE_L."""
    h, w = 40, 80
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[10:20, 5:15] = lb.EP_LEFT_EYE
    gt[10:20, 60:70] = lb.EP_RIGHT_EYE
    aligned = np.full((h, w), NOT_PARSED, dtype=np.uint8)
    aligned[10:20, 5:15] = int(Region.EYE_R)
    aligned[10:20, 60:70] = int(Region.EYE_L)
    swapped = aligned.copy()
    swapped[10:20, 5:15], swapped[10:20, 60:70] = int(Region.EYE_L), int(Region.EYE_R)
    for pred, expect_swap in ((aligned, False), (swapped, True)):
        conf = mt.confusion_matrix(gt, pred)
        assert mt.side_swap(conf, lb.EP_LEFT_EYE, lb.EP_RIGHT_EYE, int(Region.EYE_R), int(Region.EYE_L)) is expect_swap
        r = mt.counts_from_confusion(conf, (lb.EP_LEFT_EYE,), [int(Region.EYE_R)])
        assert (r.tp == 100) is (not expect_swap)


def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    h, w = 60, 75
    gt = rng.integers(0, 9, size=(h, w)).astype(np.uint8)
    # canonical predictions include NOT_PARSED and the internal Unsided ids (200-202, which must count as not predicted)
    choices = np.array(list(range(19)) + [NOT_PARSED, 200, 201, 202], dtype=np.uint8)
    pred = choices[rng.integers(0, choices.size, size=(h, w))]
    rect = (int(rng.integers(0, 20)), int(rng.integers(0, 15)), int(rng.integers(45, w + 1)), int(rng.integers(35, h + 1)))
    valid = rng.random((h, w)) > 0.1
    return gt, pred, rect, valid


@pytest.mark.parametrize("seed", range(6))
def test_confusion_derived_counts_equal_direct_mask_counts(seed):
    gt, pred, rect, valid = _random_case(seed)
    x0, y0, x1, y1 = rect
    scopes = {
        "full": (None, None, np.ones(gt.shape, dtype=bool)),
        "window": (rect, None, np.ones(gt.shape, dtype=bool)),
        "valid": (None, valid, valid),
    }
    for scope, (rr, vv, keep) in scopes.items():
        conf = mt.confusion_matrix(gt, pred, rr, vv)
        keep_mask = np.zeros(gt.shape, dtype=bool)
        if scope == "window":
            keep_mask[y0:y1, x0:x1] = True
        else:
            keep_mask[:] = True
        if scope == "valid":
            keep_mask &= valid
        for comp in lb.COMPARISONS:
            cols = lb.column_ids(comp.canonical)
            got = mt.counts_from_confusion(conf, comp.ep_ids, cols)
            gm = np.isin(gt, comp.ep_ids) & keep_mask
            pm = np.isin(pred, [int(r) for r in comp.canonical]) & keep_mask  # Unsided ids are never in `canonical`
            want = mt.counts_from_masks(gm, pm)
            assert (got.tp, got.fp, got.fn) == (want.tp, want.fp, want.fn), (scope, comp.name)
            # the not-predicted column is a true "not predicted": those pixels are never in any region's count
            assert conf[:, lb.NOT_PARSED_COL].sum() + conf[:, : lb.NUM_CANONICAL].sum() == conf.sum()


def test_not_parsed_and_unsided_predictions_count_as_not_predicted_only():
    gt = np.full((10, 10), lb.EP_SKIN, dtype=np.uint8)
    pred = np.full((10, 10), NOT_PARSED, dtype=np.uint8)
    pred[:, :5] = int(Region.SKIN)
    pred[:, 5:] = 200  # an internal Unsided id must never reach the metric; it counts as not predicted
    conf = mt.confusion_matrix(gt, pred)
    c = mt.counts_from_confusion(conf, (lb.EP_SKIN,), [int(Region.SKIN), int(Region.NOSE)])
    assert (c.tp, c.fp, c.fn) == (50, 0, 50)
    assert conf[lb.EP_SKIN, lb.NOT_PARSED_COL] == 50


def test_full_and_window_scoring_differ_only_by_pixels_outside_the_window():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[:, :] = lb.EP_SKIN  # GT everywhere
    pred = np.full((20, 20), int(Region.SKIN), dtype=np.uint8)
    pred[:, 10:] = NOT_PARSED  # outside the parse window on the right half
    rect = (0, 0, 10, 20)  # the window covers the left half only
    full = mt.counts_from_confusion(mt.confusion_matrix(gt, pred), (lb.EP_SKIN,), [int(Region.SKIN)])
    win = mt.counts_from_confusion(mt.confusion_matrix(gt, pred, rect), (lb.EP_SKIN,), [int(Region.SKIN)])
    assert (full.tp, full.fn) == (200, 200) and (win.tp, win.fn) == (200, 0)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    arr = np.stack([rng.integers(50, 200, 80), rng.integers(0, 40, 80), rng.integers(0, 40, 80)], axis=1)
    a = mt.bootstrap_scores(arr, 1000, seed=7)
    b = mt.bootstrap_scores(arr, 1000, seed=7)
    c = mt.bootstrap_scores(arr, 1000, seed=8)
    assert a == b and a != c
    tot = arr.sum(axis=0)
    point = mt.scores(*[float(v) for v in tot])
    for k in ("iou", "dice", "precision", "recall"):
        lo, hi = a[k]
        assert lo <= point[k] <= hi, k
    assert mt.bootstrap_scores(np.zeros((0, 3)), 10, 0)["iou"] == (None, None)


def test_scores_edge_cases():
    assert mt.scores(0, 0, 0) == {"iou": None, "dice": None, "precision": None, "recall": None}
    s = mt.scores(10, 0, 0)
    assert s["iou"] == s["dice"] == s["precision"] == s["recall"] == 1.0
    s = mt.scores(0, 5, 5)
    assert s["iou"] == 0.0 and s["precision"] == 0.0 and s["recall"] == 0.0


def test_per_image_iou_uses_only_images_with_enough_gt_and_false_alarms_use_absent_gt():
    counts = [
        mt.Counts(30, 0, 0),  # GT 30 px: included
        mt.Counts(20, 0, 10),  # GT 30 px: included, IoU 2/3
        mt.Counts(5, 0, 5),  # GT 10 px: excluded from IoU, but "absent" for false alarms, pred 5 px: no alarm
        mt.Counts(0, 40, 0),  # GT absent, pred 40 px: a false alarm
    ]
    stats = mt.per_image_iou_stats(counts)
    assert stats["n"] == 2 and stats["median"] == pytest.approx((1.0 + 2 / 3) / 2)
    fa = mt.false_alarm_rate(counts)
    assert fa["n_absent"] == 2 and fa["n_false_alarm"] == 1 and fa["rate"] == 0.5


def _square(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_boundary_f_is_one_within_tau_and_drops_beyond_it():
    tau = 2.0
    gt = _square(80, 80, 20, 60, 20, 60)
    values = {}
    for s in (0, 1, 2, 3, 4, 6):
        pred = _square(80, 80, 20, 60, 20 + s, 60 + s)  # horizontal shift: vertical edges move, horizontal do not
        values[s] = mt.boundary_f(gt, pred, tau)
    assert values[0] == pytest.approx(1.0)
    assert values[1] > 0.95 and values[2] > 0.95  # within tau
    assert values[3] < 0.8 and values[4] < 0.75 and values[6] < 0.7  # beyond tau the vertical edges fail
    assert values[6] <= values[4] <= values[3] + 1e-9


def test_boundary_f_edge_cases():
    empty = np.zeros((10, 10), dtype=bool)
    box = _square(10, 10, 2, 8, 2, 8)
    assert mt.boundary_f(empty, empty, 1.0) is None
    assert mt.boundary_f(box, empty, 1.0) == 0.0 and mt.boundary_f(empty, box, 1.0) == 0.0
    assert mt.boundary_f(box, box, 0.5) == pytest.approx(1.0)


@pytest.mark.parametrize("dx,dy", [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (1, -3)])
def test_best_shift_returns_the_move_that_aligns_pred_to_gt(dx, dy):
    """pred = gt moved by (dx, dy) (+ = right/down): best_shift must return the move that undoes it, (-dx, -dy)."""
    gt = _square(60, 60, 20, 40, 20, 40)
    pred = _square(60, 60, 20 + dy, 40 + dy, 20 + dx, 40 + dx)
    best = mt.best_shift(gt, pred, max_shift=3)
    assert best is not None
    assert (best[0], best[1]) == (-dx, -dy)
    assert best[2] == pytest.approx(1.0)


def test_best_shift_none_for_empty_and_tiny_masks():
    gt = _square(20, 20, 2, 10, 2, 10)
    assert mt.best_shift(gt, np.zeros((20, 20), dtype=bool)) is None


def test_side_swap_is_undefined_without_overlap():
    conf = np.zeros((9, lb.CONF_COLS), dtype=np.int64)
    assert mt.side_swap(conf, 5, 6, 6, 5) is None


def test_denominators_and_failure_rows_count_as_empty_predictions():
    from evaluation import report as rp

    rows = [
        {"outcome": "ok", "conf_full": np.zeros((9, 20), int).tolist(), "conf_win": None},
        {"outcome": "no_face", "conf_full": np.zeros((9, 20), int).tolist(), "conf_win": None},
        {"outcome": "invalid_sample"},
    ]
    assert len(rp.valid_rows(rows, "A")) == 1
    assert len(rp.valid_rows(rows, "B")) == 2  # failures are kept as empty predictions; invalid samples are not


def test_report_regeneration_needs_no_network_and_matches_committed_json(tmp_path):
    run_dir = Path.home() / ".cache" / "makeup_demo" / "easyportrait_runs" / "2026-11-10"
    committed = EVAL_DIR / "easyportrait_test_2026-11-10.json"
    if not (run_dir / "per_image.jsonl").is_file() or not committed.is_file():
        pytest.skip("pilot run outputs are not present on this machine")
    from evaluation import report as rp

    rows = rp.load_rows(run_dir / "per_image.jsonl")
    entries = ds.parse_sample((EVAL_DIR / "easyportrait_test_sample_seed20261110_n600.txt").read_text())
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    rep = rp.build_report(rows, entries, run_meta, "docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt", "2026-11-10")
    md, js = rp.write_report(rep, tmp_path / "regen")
    assert js.read_text() == committed.read_text()

    # The Markdown carries one line that is not derived from the run data: the fingerprint of the source tree
    # at report time. It changes with any edit under backend/, so compare everything except that line.
    def _without_current_fingerprint(text: str) -> str:
        return "\n".join(l for l in text.splitlines() if "fingerprint of the **current tree at report time**" not in l)

    assert _without_current_fingerprint(md.read_text()) == _without_current_fingerprint(
        (EVAL_DIR / "easyportrait_test_2026-11-10.md").read_text()
    )


# =====================================================================================
# 4. Guards: cache location, UUIDs, HTTP fetcher, budgets, members (fake fetchers, in memory)
# =====================================================================================

GOOD_ID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"


@pytest.mark.parametrize(
    "bad",
    [
        "0A1B2C3D-4E5F-4A6B-8C7D-9E0F1A2B3C4D",  # upper case
        " " + GOOD_ID,
        GOOD_ID + "\n",
        GOOD_ID + "\x00",
        "../" + GOOD_ID,
        "..\\" + GOOD_ID,
        GOOD_ID[:-1] + "%2F",
        "0a1b2c3d4e5f4a6b8c7d9e0f1a2b3c4d",  # no hyphens
        "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4",  # one digit short
        "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d0",  # one digit long
        "١٢" + GOOD_ID[2:],  # Arabic-Indic digits
        "images/test/" + GOOD_ID,
    ],
)
def test_uuid_guard_rejects_hostile_ids_before_any_path_is_built(bad, tmp_path):
    assert not ds.is_valid_id(bad)
    with pytest.raises(ValueError):
        ds.require_id(bad)
    with pytest.raises(ValueError):
        ds.image_path(tmp_path, bad)
    with pytest.raises(ValueError):
        ds.annotation_path(tmp_path, bad)
    assert ds.is_valid_id(GOOD_ID)


def test_cache_location_guard_refuses_repo_parents_and_relative_paths(tmp_path, monkeypatch):
    with pytest.raises(ds.CacheError):
        ds.resolve_cache_dir(REPO)
    with pytest.raises(ds.CacheError):
        ds.resolve_cache_dir(REPO / "docs" / "evaluation")
    with pytest.raises(ds.CacheError):
        ds.resolve_cache_dir(REPO.parent)  # a parent of the repo
    monkeypatch.chdir(REPO)
    with pytest.raises(ds.CacheError):
        ds.resolve_cache_dir("backend/../docs")  # relative path that resolves inside the repo
    # siblings whose name starts like the repo are not "inside" it (component-wise check)
    sibling = REPO.parent / (REPO.name + "_sibling_check")
    assert ds.refuse_inside_repo(sibling) == sibling.resolve()
    assert ds.refuse_inside_repo(tmp_path) == tmp_path.resolve()


def test_cache_marker_guard_on_clean_and_on_foreign_directories(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not ours")
    with pytest.raises(ds.CacheError):
        ds.ensure_cache(foreign)
    with pytest.raises(ds.CacheError):
        ds.clean_cache(foreign)
    assert (foreign / "keep.txt").read_text() == "not ours"


def test_cache_directories_and_files_are_private(tmp_path):
    cache = ds.ensure_cache(tmp_path / "c")
    ds.write_private(cache / "x.bin", b"abc")
    assert oct((cache).stat().st_mode & 0o777) == "0o700"
    assert oct((cache / "images").stat().st_mode & 0o777) == "0o700"
    assert oct((cache / "x.bin").stat().st_mode & 0o777) == "0o600"
    assert oct((cache / ds.MARKER_NAME).stat().st_mode & 0o777) == "0o600"


# --- HTTPS fetcher with a fake opener (no network) ----------------------------------------

class _FakeResp:
    def __init__(self, status: int, headers: dict[str, str], body: bytes = b""):
        self.status, self.headers, self._body = status, headers, body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


class _FakeOpener:
    def __init__(self, resp=None, exc: Exception | None = None):
        self.resp, self.exc, self.calls = resp, exc, 0

    def open(self, req, timeout=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.resp


URL = "https://example.invalid/archive.zip"


def _fetcher_with(opener) -> HttpsRangeFetcher:
    f = HttpsRangeFetcher.__new__(HttpsRangeFetcher)
    f._opener = opener
    return f


def test_range_fetch_accepts_only_a_matching_206_with_exact_length():
    body = b"x" * 10
    ok = _fetcher_with(_FakeOpener(_FakeResp(206, {"Content-Range": "bytes 0-9/100"}, body)))
    assert ok.get_range(URL, 0, 9) == body


@pytest.mark.parametrize(
    "resp,label",
    [
        (_FakeResp(200, {"Content-Range": "bytes 0-9/100"}, b"x" * 100), "200 (server ignored the range)"),
        (_FakeResp(206, {"Content-Range": "bytes 5-14/100"}, b"x" * 10), "wrong start"),
        (_FakeResp(206, {}, b"x" * 10), "missing Content-Range"),
        (_FakeResp(206, {"Content-Range": "bytes 0-9/100"}, b"x" * 9), "short body"),
        (_FakeResp(206, {"Content-Range": "bytes 0-9/100"}, b"x" * 11), "long body"),
        (_FakeResp(206, {"Content-Range": "garbage"}, b"x" * 10), "malformed Content-Range"),
    ],
)
def test_range_fetch_aborts_on_any_mismatch(resp, label):
    with pytest.raises(FetchError):
        _fetcher_with(_FakeOpener(resp)).get_range(URL, 0, 9)


def test_range_fetch_http_errors_do_not_leak_the_url():
    err = urllib.error.HTTPError(URL, 416, "Range Not Satisfiable", {}, None)
    with pytest.raises(FetchError) as info:
        _fetcher_with(_FakeOpener(exc=err)).get_range(URL, 0, 9)
    assert "example.invalid" not in str(info.value) and "416" in str(info.value)
    with pytest.raises(FetchError) as info:
        _fetcher_with(_FakeOpener(exc=urllib.error.URLError("no route to example.invalid"))).get_range(URL, 0, 9)
    assert "example.invalid" not in str(info.value)


def test_plain_http_and_redirects_are_refused_without_contacting_the_server():
    opener = _FakeOpener(_FakeResp(206, {"Content-Range": "bytes 0-9/100"}, b"x" * 10))
    f = _fetcher_with(opener)
    with pytest.raises(FetchError):
        f.get_range("http://example.invalid/a.zip", 0, 9)
    with pytest.raises(FetchError):
        f.head("http://example.invalid/a.zip")
    assert opener.calls == 0
    with pytest.raises(FetchError):
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://other.invalid/")


def test_head_requires_200_and_a_length():
    with pytest.raises(FetchError):
        _fetcher_with(_FakeOpener(_FakeResp(200, {}))).head(URL)
    with pytest.raises(FetchError):
        _fetcher_with(_FakeOpener(_FakeResp(404, {"Content-Length": "5"}))).head(URL)
    assert _fetcher_with(_FakeOpener(_FakeResp(200, {"Content-Length": "5", "ETag": "e"}))).head(URL) == Head(5, "e")


# --- budgets, pins and members with an in-memory archive ---------------------------------

class _MemFetcher:
    """Serves byte ranges of in-memory blobs. Honours ranges; counts requests."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs, self.requests = blobs, []

    def head(self, url: str) -> Head:
        return Head(len(self.blobs[url]), None)

    def get_range(self, url: str, start: int, end: int) -> bytes:
        self.requests.append((url, start, end))
        data = self.blobs[url]
        if start < 0 or end < start or end >= len(data):
            raise FetchError("range out of bounds")
        return data[start : end + 1]


def _zip(members: dict[str, bytes], method=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_budget_is_reserved_before_the_request_and_never_overshoots_under_threads():
    url = "https://example.invalid/big.zip"
    blob = bytes(range(256)) * 400
    inner = _MemFetcher({url: blob})
    bf = BudgetedFetcher(inner, budget_bytes=1500, pins={url: len(blob)})  # room for exactly 15 x 100 bytes
    errors, ok = [], []
    lock = threading.Lock()

    def worker(i):
        try:
            bf.get_range(url, i * 100, i * 100 + 99)
            with lock:
                ok.append(i)
        except FetchError as exc:
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 15 and len(errors) == 25
    assert bf.bytes_received == 1500 == len(ok) * 100  # refused requests are not counted
    assert len(inner.requests) == len(ok)  # and never reach the network layer
    assert all("budget" in e for e in errors)


def test_pins_refuse_unpinned_urls_and_ranges_past_the_archive_end():
    url = "https://example.invalid/a.zip"
    bf = BudgetedFetcher(_MemFetcher({url: b"0123456789"}), budget_bytes=100, pins={url: 10})
    with pytest.raises(FetchError):
        bf.get_range("https://example.invalid/other.zip", 0, 1)
    with pytest.raises(FetchError):
        bf.verified_length("https://example.invalid/other.zip")
    with pytest.raises(FetchError):
        bf.get_range(url, 5, 10)  # byte 10 is past the end
    assert bf.verified_length(url) == 10 and bf.requests == 0


def test_central_directory_and_member_roundtrip_with_fetch_accounting():
    url = "https://example.invalid/test.zip"
    members = {"images/test/" + GOOD_ID + ".jpg": b"\xff\xd8" + b"a" * 5000}
    blob = _zip(members)
    bf = BudgetedFetcher(_MemFetcher({url: blob}), budget_bytes=10**6, pins={url: len(blob)})
    table = read_central_directory(bf, url)
    info = table["images/test/" + GOOD_ID + ".jpg"]
    assert fetch_member(bf, url, info, max_size=10_000) == members["images/test/" + GOOD_ID + ".jpg"]


def test_member_checks_run_before_any_download_and_inside_decompression():
    url = "https://example.invalid/test.zip"
    name = "images/test/" + GOOD_ID + ".jpg"
    payload = b"\x00" * (5 * 1024 * 1024)  # 5 MB of zeros compresses to a few KB: a decompression bomb
    blob = _zip({name: payload})
    mem = _MemFetcher({url: blob})
    bf = BudgetedFetcher(mem, budget_bytes=10**8, pins={url: len(blob)})
    info = read_central_directory(bf, url)[name]
    mem.requests.clear()
    # declared size above the cap: refused before any member request
    with pytest.raises(FetchError, match="size cap"):
        fetch_member(bf, url, info, max_size=1024 * 1024)
    assert mem.requests == []
    # declared sizes that pass the cap but a lying central directory: the bounded decompressor refuses
    lying = type(info)(info.name, info.header_offset, info.compress_size, 1024, info.crc, info.method)
    with pytest.raises(FetchError):
        fetch_member(bf, url, lying, max_size=1024 * 1024)


def test_member_crc_and_name_and_method_checks():
    url = "https://example.invalid/test.zip"
    name = "annotations/test/" + GOOD_ID + ".png"
    blob = _zip({name: b"hello world" * 10})
    mem = _MemFetcher({url: blob})
    bf = BudgetedFetcher(mem, budget_bytes=10**6, pins={url: len(blob)})
    info = read_central_directory(bf, url)[name]
    bad_crc = type(info)(info.name, info.header_offset, info.compress_size, info.file_size, info.crc ^ 1, info.method)
    with pytest.raises(FetchError, match="CRC"):
        fetch_member(bf, url, bad_crc, max_size=10_000)
    wrong_name = type(info)("annotations/test/" + GOOD_ID.replace("a", "b", 1) + ".png", info.header_offset, info.compress_size, info.file_size, info.crc, info.method)
    with pytest.raises(FetchError):
        fetch_member(bf, url, wrong_name, max_size=10_000)
    bzip = type(info)(info.name, info.header_offset, info.compress_size, info.file_size, info.crc, 12)
    with pytest.raises(FetchError, match="unsupported"):
        fetch_member(bf, url, bzip, max_size=10_000)


# =====================================================================================
# 5. Sample determinism and stratification, from the local cache (skipped when absent)
# =====================================================================================

CACHE_META = Path(ds.DEFAULT_CACHE_DIR).expanduser() / "meta_test.csv"
COMMITTED_SAMPLE = EVAL_DIR / "easyportrait_test_sample_seed20261110_n600.txt"


@pytest.mark.skipif(not CACHE_META.is_file(), reason="local EasyPortrait meta cache not present")
def test_sample_is_exact_stratified_and_reproducible_from_the_meta_table():
    text = CACHE_META.read_text()
    assert "user_id" not in text.splitlines()[0]
    rows = ds.load_meta_csv(text)
    assert len(rows) == 6000
    first = ds.draw_sample(rows, 600, 20261110)
    again = ds.draw_sample(rows, 600, 20261110)
    assert [e.image_id for e in first] == [e.image_id for e in again]
    assert ds.format_sample(first, 20261110, 600) == COMMITTED_SAMPLE.read_text()

    # independent stratum check: every non-empty (brightness quartile x resolution) cell is covered
    b = np.array([r.brightness for r in rows])
    edges = np.quantile(b, [0.25, 0.5, 0.75])
    cells: dict[tuple[int, int], int] = {}
    for r in rows:
        q = int(np.searchsorted(edges, r.brightness, side="right"))
        mp = r.height * r.width / 1e6
        g = 0 if mp < 1.5 else 1 if mp < 2.5 else 2 if mp < 3.0 else 3
        cells[(q, g)] = cells.get((q, g), 0) + 1
    picked: dict[tuple[int, int], int] = {}
    by_id = {r.image_id: r for r in rows}
    for e in first:
        r = by_id[e.image_id]
        assert e.brightness_quartile == int(np.searchsorted(edges, r.brightness, side="right"))
        picked[(e.brightness_quartile, e.resolution_group)] = picked.get((e.brightness_quartile, e.resolution_group), 0) + 1
    assert set(cells) == set(picked)
    for key, size in cells.items():
        assert picked[key] >= min(size, 5), key
    assert sum(picked.values()) == 600


def test_committed_sample_split_follows_the_sha256_rule_independently():
    lines = [ln for ln in COMMITTED_SAMPLE.read_text().splitlines() if ln and not ln.startswith("#")]
    dev = 0
    for ln in lines:
        image_id, split, _, _ = ln.split("\t")
        expect = "dev" if int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % 10 < 4 else "test"
        assert split == expect
        dev += split == "dev"
    assert len(lines) == 600 and dev == 239


# =====================================================================================
# 6. Privacy of committed artefacts
# =====================================================================================

def test_committed_evaluation_artefacts_hold_ids_and_aggregates_only():
    forbidden = ("user_id", "/Users/", "Makeup_demo", "data:image", "png_base64")
    for path in EVAL_DIR.glob("easyportrait_*"):
        text = path.read_text()
        for word in forbidden:
            assert word not in text, (path.name, word)
    report = json.loads((EVAL_DIR / "easyportrait_test_2026-11-10.json").read_text())
    assert '"per_image"' not in json.dumps(report) and "per_image_rows" not in json.dumps(report)

    def longest_list(node) -> int:
        if isinstance(node, dict):
            return max((longest_list(v) for v in node.values()), default=0)
        if isinstance(node, list):
            return max([len(node)] + [longest_list(v) for v in node])
        return 0

    assert longest_list(report) < 100  # a per-image dump would be ~600 entries long
