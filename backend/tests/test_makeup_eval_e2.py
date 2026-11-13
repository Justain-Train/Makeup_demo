"""Offline tests for evaluation step E-2 (raw signals, labels, stats, aggregate/report, run store).

No network: an autouse fixture makes any socket connect fail. Synthetic images and profiles only.
"""

from __future__ import annotations

import io
import json
import re
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_blueprint.cv_analysis import TARGETS, CvReferenceAnalyzer
from evaluation import makeup_dataset as md
from evaluation import makeup_eval as me
from evaluation.dataset import REPO_ROOT, CacheError
from evaluation.makeup_labels import (
    LABEL_MARGIN_FRACTION,
    SENSITIVITY_FRACTIONS,
    classify_stage,
    label_zone,
)
from evaluation.makeup_pipeline import PairResult
from evaluation.makeup_signals import prepare, raw_signals
from evaluation.makeup_stats import (
    bootstrap_ci,
    gap_closed,
    paired_auc,
    prf,
    rank_auc,
    threshold_at_fp,
    wilson,
)
from tests.test_makeup_cv_analysis import SKIN, _image, _profile

PRODUCTS = tuple(r.product_type for r in TARGETS)
CANARY_IDS = ("918273", "918274", "918275")
LIP = TARGETS[0]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("network access attempted in offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# --- replica equivalence (criterion 2) -----------------------------------------------------------

PAINT = {
    "lip_full": (200, 20, 40),
    "cheek_apple_l": (230, 120, 120),
    "cheek_apple_r": (230, 120, 120),
    "eyelid_upper_l": (90, 60, 120),
    "eyelid_upper_r": (90, 60, 120),
    "cheekbone_l": (250, 225, 205),
    "cheekbone_r": (250, 225, 205),
    "contour_hollow_l": (120, 90, 70),
    "contour_hollow_r": (120, 90, 70),
}


@pytest.mark.parametrize("paint", [PAINT, {"lip_full": (215, 150, 150)}, {}])
def test_raw_signals_match_analyze(paint):
    img, prof = _image(paint), _profile()
    raw = raw_signals(img, prof)
    assert raw is not None
    analysis = {c.component.product_type: c.component for c in CvReferenceAnalyzer().analyze(img, prof).components}
    for row in TARGETS:
        sig = raw["products"][row.product_type]
        assert sig["detected"] == (row.product_type in analysis), row.product_type
        if row.product_type in analysis:
            span = row.full_threshold - row.present_threshold
            expect = round(min(1.0, max(0.0, 0.2 + 0.8 * (sig["value"] - row.present_threshold) / span)), 3)
            assert analysis[row.product_type].intensity == pytest.approx(expect, abs=1e-9)


def test_raw_keeps_sub_threshold_value():
    raw = raw_signals(_image({"lip_full": (215, 150, 150)}), _profile())
    lip = raw["products"]["lipstick"]
    assert lip["usable"] and lip["value"] is not None and lip["value"] < LIP.present_threshold
    assert lip["detected"] is False


@pytest.mark.parametrize("bad", ["gray", "float", "empty"])
def test_prepare_rejects_malformed_image(bad):
    img = {"gray": np.zeros((300, 300), np.uint8), "float": np.zeros((300, 300, 3), np.float32),
           "empty": np.zeros((0, 0, 3), np.uint8)}[bad]
    assert prepare(img, _profile()) is None


def test_missing_face_profile_returns_none():
    assert raw_signals(_image(), _profile(status="no_face_detected")) is None


# --- labels (criterion 3) ------------------------------------------------------------------------

def _ctx_and_labels(bare, makeup):
    ctx = prepare(bare, _profile())
    out = {}
    for row in TARGETS:
        mask = ctx.product_masks.get(row.product_type)
        out[row.product_type] = None if mask is None else label_zone(bare, makeup, mask, ctx.skin_ref, row)
    return ctx, out


def test_global_shift_gives_no_positive_labels():
    bare = _image()
    offset = np.array([-25, -15, 10], dtype=np.int16)
    makeup = np.clip(bare.astype(np.int16) + offset, 0, 255).astype(np.uint8)
    ctx, labels = _ctx_and_labels(bare, makeup)
    assert ctx.product_masks["lipstick"] is not None
    for product, lab in labels.items():
        if lab is not None:
            assert lab[0] is False, product
            assert abs(lab[1]) < LIP.present_threshold * LABEL_MARGIN_FRACTION, product


def test_zone_only_shift_gives_positive_label():
    bare = _image()
    makeup = _image({"lip_full": (200, 20, 40)})
    _, labels = _ctx_and_labels(bare, makeup)
    positive, residual = labels["lipstick"]
    assert positive is True
    assert residual >= LABEL_MARGIN_FRACTION * LIP.present_threshold


def test_label_margin_constants():
    assert LABEL_MARGIN_FRACTION == 0.5
    assert SENSITIVITY_FRACTIONS == (0.35, 0.75)


# --- classify_stage (criterion 5, branches) ------------------------------------------------------

def _bp(category):
    return SimpleNamespace(components=[SimpleNamespace(component=SimpleNamespace(category=category))])


def _plan(category, mask="m", notes=()):
    comp = SimpleNamespace(index=0, component=SimpleNamespace(category=category), mask=mask, notes=list(notes))
    return SimpleNamespace(components=[comp])


def _render(status="rendered", pixels=10, reason=None):
    rep = SimpleNamespace(index=0, status=status, pixels_touched=pixels, reason=reason)
    return SimpleNamespace(reports=(rep,))


def test_classify_stage_none_when_not_positive():
    assert classify_stage(False, _bp("lip"), _plan("lip"), _render(), "lip") is None


def test_classify_stage_four_branches():
    assert classify_stage(True, None, None, None, "lip") == ("not_detected", None)
    assert classify_stage(True, _bp("eye"), _plan("lip"), _render(), "lip") == ("not_detected", None)
    assert classify_stage(True, _bp("lip"), _plan("eye"), _render(), "lip") == ("dropped_in_adapt", None)
    assert classify_stage(True, _bp("lip"), _plan("lip", mask=None, notes=["x"]), _render(), "lip") == \
        ("mask_missing_or_empty", "x")
    assert classify_stage(True, _bp("lip"), _plan("lip"), _render(status="skipped", reason="clip"), "lip") == \
        ("mask_missing_or_empty", "clip")
    assert classify_stage(True, _bp("lip"), _plan("lip"), _render(pixels=0), "lip") == \
        ("mask_missing_or_empty", "empty_mask")
    assert classify_stage(True, _bp("lip"), _plan("lip"), _render(), "lip") == ("rendered", None)


# --- stats (criterion 4, 6) ----------------------------------------------------------------------

def test_auc_perfect_inverted_ties_and_hand_case():
    assert rank_auc([3, 4], [0, 1]) == 1.0
    assert rank_auc([0, 1], [3, 4]) == 0.0
    assert rank_auc([2, 2], [2, 2]) == 0.5
    assert rank_auc([2, 5], [1, 3, 4]) == pytest.approx(4 / 6)  # hand: 4 of 6 pairs won
    assert rank_auc([2, 5], [2, 1]) == pytest.approx(0.875)  # hand: (0.5 + 1 + 1 + 1) / 4
    assert np.isnan(rank_auc([], [1.0]))


def test_paired_auc():
    assert paired_auc([3, 1], [1, 2]) == 0.5
    assert paired_auc([2, 2], [2, 1]) == 0.75


def test_threshold_at_fp_toy_set():
    neg = np.arange(10.0)
    t = threshold_at_fp(neg, 0.2)
    assert 7.0 < t <= 8.0
    assert (neg >= t).mean() == pytest.approx(0.2)
    assert (neg >= threshold_at_fp(neg, 0.05)).sum() == 0


def test_prf_counts():
    out = prf(2, 1, 1)
    assert out["precision"] == pytest.approx(2 / 3) and out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)
    assert prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}


def test_wilson_brackets_and_bounds():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson(10, 10)
    assert lo < 1.0 and hi == 1.0
    assert wilson(0, 0) is None


def test_bootstrap_ci_brackets_estimate():
    x = np.array([1.0] * 30 + [0.0] * 70)
    lo, hi = bootstrap_ci(lambda i: float(x[i].mean()), (x.size,))
    assert lo <= 0.3 <= hi
    assert bootstrap_ci(lambda i: 0.0, (0,)) is None
    assert bootstrap_ci(lambda i: None, (10,)) is None


def test_gap_closed_exclusion_and_values():
    truth = np.array([60.0, 20.0, 20.0])
    bare = np.array([50.0, 5.0, 5.0])
    assert gap_closed(truth, truth, bare) == pytest.approx(1.0)
    assert gap_closed(bare, truth, bare) == pytest.approx(0.0)
    assert gap_closed(truth, truth, truth) is None  # baseline 0 dE00 < 1


# --- aggregate / bands / report (criteria 5-8) ---------------------------------------------------

def _sig(value=10.0, detected=False):
    return {"products": {p: {"value": value, "usable": True, "gate_ok": True, "detected": detected} for p in PRODUCTS}}


def _kd(residual=None, stage=None, gaps=None, signals=None):
    labels = {p: ({"residual": residual} if residual is not None else None) for p in PRODUCTS}
    return {"signals": signals or _sig(), "labels": labels, "stages": {p: stage for p in PRODUCTS},
            "gaps": gaps or {}, "ms": 5.0, "status": None, "rendered": 1, "skipped": 0, "outside_changed": 0}


def _unit(depth="light", bare=None, k=()):
    return {"depth": depth, "bare": bare, "k": list(k)}


def test_aggregate_shares_sum_to_one_and_gap_exclusion():
    stages = [["rendered", None], ["dropped_in_adapt", None], ["mask_missing_or_empty", "empty_mask"],
              ["not_detected", None]]
    ks = [_kd(residual=30.0, stage=st, gaps={"rendered": None if i < 2 else 0.6}) for i, st in enumerate(stages)]
    agg = me.aggregate([_unit("light", bare=_sig(), k=ks)], [], [], 0.5)
    for p in PRODUCTS:
        shares = agg["e1_detection"][p]["loss_stage"]["shares"]
        assert sum(shares.values()) == pytest.approx(1.0), p
    gaps = agg["e2_gap_closed"]["gap_closed_per_stage"]
    assert gaps["rendered"]["excluded_baseline_lt_1"] == 2  # two kds had baseline < 1 dE00 (None)
    assert gaps["rendered"]["n"] == 2 and gaps["rendered"]["mean"] == pytest.approx(0.6)


def test_depth_bands_suppressed_below_five():
    small = [_unit("light", k=[_kd(residual=30.0, gaps={"rendered": 0.5})]) for _ in range(4)]
    agg = me.aggregate(small, [], [], 0.5)
    band = agg["fairness_depth_bands"]["bands"]["light"]
    assert band == {"n": 4, "suppressed": True}
    five = [_unit("dark", k=[_kd(residual=30.0)]) for _ in range(5)]
    band5 = me.aggregate(five, [], [], 0.5)["fairness_depth_bands"]["bands"]["dark"]
    assert band5["n"] == 5 and "products" in band5
    recall = band5["products"]["lipstick"]["recall"]  # 5 labelled positives, none detected
    assert recall.get("suppressed") is None and recall["n"] == 5 and recall["k"] == 0


def test_report_has_sections_and_no_ids_or_paths(tmp_path):
    agg = me.aggregate([_unit("light", bare=_sig(), k=[_kd(residual=30.0, stage=["rendered", None])])], [], [], 0.5)
    rep = me.build_report(agg, seed=1, margin=0.5, today="2026-11-13", easyportrait_images=0)
    md_text = me.render_markdown(rep)
    blob = json.dumps(rep) + md_text
    for cid in CANARY_IDS:
        assert cid not in blob
    assert str(tmp_path) not in blob and "/Users/" not in blob
    assert "CC BY-NC-SA 4.0" in rep["provenance"]["licence"]
    assert rep["caveat"] and "## Caveat" in md_text
    assert rep["recommendation_inputs"]["decision_rule"] == me.DECISION_RULE
    assert "AUC >= 0.9" in md_text and "## Recommendation inputs" in md_text and "Decision rule:" in md_text


# --- RunStore and run() (criterion 9, plus canary end-to-end) ------------------------------------

class CountingRunner:
    def __init__(self):
        self.runs = 0

    def profile(self, rgb):
        return None  # no face: deterministic, offline

    def run(self, bare, truth):
        self.runs += 1
        return PairResult(False, None, None, None, "no_profile", 1.0)

    def close(self):
        pass


def _write_cache(cache, ids):
    (cache / "images").mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 60)).save(buf, format="JPEG", quality=90)
    for i in ids:
        for kind in md.KINDS:
            (cache / "images" / f"{i}_{kind}.jpg").write_bytes(buf.getvalue())


def test_run_resume_skips_finished_units_and_canary_clean(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache, CANARY_IDS[:2])
    samples = md.MakeupSamples(pairs=list(CANARY_IDS[:2]), cross=[], sheet=[])
    runs, out = tmp_path / "runs", tmp_path / "out"
    first = CountingRunner()
    rep1 = me.run(cache, samples, runs, out, runner=first, today="2026-11-13")
    assert rep1["_run"] == {"computed": 2, "skipped": 0} and first.runs == 10
    second = CountingRunner()
    rep2 = me.run(cache, samples, runs, out, runner=second, today="2026-11-13")
    assert rep2["_run"] == {"computed": 0, "skipped": 2} and second.runs == 0
    text = (out / "makeup_eval_2026-11-13.json").read_text() + (out / "makeup_eval_2026-11-13.md").read_text()
    for cid in CANARY_IDS:
        assert not re.search(rf"\b{cid}\b", text)
    assert not re.search(r"[0-9a-f]{32}", text)
    assert str(tmp_path) not in text and "/Users/" not in text


def test_run_recomputes_corrupt_unit(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache, CANARY_IDS[:2])
    samples = md.MakeupSamples(pairs=list(CANARY_IDS[:2]), cross=[], sheet=[])
    runs = tmp_path / "runs"
    me.run(cache, samples, runs, tmp_path / "out", runner=CountingRunner(), today="2026-11-13")
    for f in (runs / "units").glob("*.json"):
        f.write_text("{not json")
        break
    rerun = CountingRunner()
    rep = me.run(cache, samples, runs, tmp_path / "out", runner=rerun, today="2026-11-13")
    assert rep["_run"] == {"computed": 1, "skipped": 1}
    assert rerun.runs == 5


def test_runs_dir_inside_repo_refused(tmp_path):
    with pytest.raises(CacheError):
        me.RunStore(REPO_ROOT / "runs_should_not_exist")
    assert not (REPO_ROOT / "runs_should_not_exist").exists()
    with pytest.raises(CacheError):
        me.run(tmp_path, md.MakeupSamples([], [], []), REPO_ROOT / "runs_should_not_exist", tmp_path,
               runner=CountingRunner())
