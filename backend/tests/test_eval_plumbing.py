"""Evaluation plumbing against the real pipeline (weights-gated) and the report generator."""

import copy
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from app.config import get_settings
from app.services.face_regions import segment_face
from app.services.image_loading import load_rgb_image
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import NOT_PARSED, Region
from evaluation import dataset as ds
from evaluation import labels as lb
from evaluation import metrics as mt
from evaluation import report as rp
from evaluation import runner as rn
from tests.synthetic import FIXTURES

SETTINGS = get_settings()
HAVE_WEIGHTS = SETTINGS.resolved_face_parser_path.is_file() and SETTINGS.resolved_model_path.is_file()
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="model weights not downloaded")

TO_EP = {
    Region.BACKGROUND: lb.EP_BACKGROUND, Region.SKIN: lb.EP_SKIN, Region.NOSE: lb.EP_SKIN,
    Region.BROW_R: lb.EP_LEFT_BROW, Region.BROW_L: lb.EP_RIGHT_BROW,
    Region.EYE_R: lb.EP_LEFT_EYE, Region.EYE_L: lb.EP_RIGHT_EYE,
    Region.LIP_UPPER: lb.EP_LIPS, Region.LIP_LOWER: lb.EP_LIPS, Region.MOUTH_INTERIOR: lb.EP_TEETH,
}


@pytest.fixture(scope="module")
def services():
    landmarks, parser = rn.build_services(2.0)
    yield landmarks, parser
    landmarks.close()


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _synthetic_gt(rgb: np.ndarray, services, swap_sides: bool = False) -> np.ndarray:
    """GT built from the pipeline's own prediction (EP ids), so agreement must be ~1."""
    resp = segment_face(rgb, *services)
    a = resp.anatomical
    labels = mk.decode_png(__import__("base64").b64decode(a.label_map_png_base64))
    lut = np.full(256, lb.EP_PERSON, dtype=np.uint8)
    for region, ep in TO_EP.items():
        lut[int(region)] = ep
    lut[NOT_PARSED] = lb.EP_BACKGROUND
    gt = np.zeros(rgb.shape[:2], dtype=np.uint8)
    gt[a.origin.y : a.origin.y + a.height, a.origin.x : a.origin.x + a.width] = lut[labels]
    if swap_sides:
        swapped = gt.copy()
        for x, y in ((lb.EP_LEFT_EYE, lb.EP_RIGHT_EYE), (lb.EP_LEFT_BROW, lb.EP_RIGHT_BROW)):
            swapped[gt == x], swapped[gt == y] = y, x
        gt = swapped
    return gt


@needs_weights
def test_plumbing_self_consistency_gives_agreement_of_one(services):
    data = (FIXTURES / "portrait.jpg").read_bytes()
    rgb = load_rgb_image(data)
    row = rn.evaluate_image(data, _png(_synthetic_gt(rgb, services)), *services)
    assert row["outcome"] == "ok"
    conf = np.array(row["conf_full"])
    for name in ("skin", "lips", "eye", "brow", "eye_R", "eye_L", "brow_R", "brow_L"):
        comp = lb.COMPARISON_BY_NAME[name]
        c = mt.counts_from_confusion(conf, comp.ep_ids, lb.column_ids(comp.canonical))
        assert c.gt > 0 and c.tp / (c.tp + c.fp + c.fn) > 0.999, name
    assert row["small"]["eye"]["bf"]["0.05"] == pytest.approx(1.0) and row["small"]["lips"]["shift"][:2] == [0, 0]
    assert row["order_ok"] == {"eye": True, "brow": True}
    assert row["landmark_polygons"]["eye"][0] > 0  # the landmark-polygon comparison ran
    win = np.array(row["conf_win"])
    assert win.sum() == (row["rect"][2] - row["rect"][0]) * (row["rect"][3] - row["rect"][1]) < conf.sum()
    assert {"inference", "total"} <= set(row["timings_ms"]) and row["decode_ms"] > 0 and row["wall_ms"] > 0


@needs_weights
def test_plumbing_detects_swapped_left_right(services):
    data = (FIXTURES / "portrait.jpg").read_bytes()
    rgb = load_rgb_image(data)
    row = rn.evaluate_image(data, _png(_synthetic_gt(rgb, services, swap_sides=True)), *services)
    conf = np.array(row["conf_full"])
    for kind, (a, b, ra, rb) in lb.SIDE_PAIRS.items():
        assert mt.side_swap(conf, a, b, int(ra), int(rb)) is True, kind
    eye_r = lb.COMPARISON_BY_NAME["eye_R"]
    c = mt.counts_from_confusion(conf, eye_r.ep_ids, lb.column_ids(eye_r.canonical))
    assert c.tp == 0  # the sided row collapses; the merged row is unaffected
    merged = lb.COMPARISON_BY_NAME["eye"]
    m = mt.counts_from_confusion(conf, merged.ep_ids, lb.column_ids(merged.canonical))
    assert m.tp / (m.tp + m.fp + m.fn) > 0.999


@needs_weights
def test_invalid_samples_are_classified_not_crashed(services):
    data = (FIXTURES / "portrait.jpg").read_bytes()
    rgb = load_rgb_image(data)
    gt = _synthetic_gt(rgb, services)
    with pytest.raises(ds.InvalidSample) as e:
        rn.evaluate_image(data, _png(gt.T.copy()), *services)  # transposed annotation: never rotated to fit
    assert e.value.reason == "annotation_shape"
    with pytest.raises(ds.InvalidSample) as e:
        rn.evaluate_image(b"not an image", _png(gt), *services)
    assert e.value.reason == "image_undecodable"


def test_no_face_and_pipeline_error_rows_count_as_empty_predictions():
    from tests.synthetic import FakeLandmarkService, FakeParser

    data = (FIXTURES / "portrait.jpg").read_bytes()
    gt = np.zeros(load_rgb_image(data).shape[:2], dtype=np.uint8)
    gt[100:200, 100:200] = lb.EP_SKIN
    row = rn.evaluate_image(data, _png(gt), FakeLandmarkService("no_face_detected"), FakeParser())
    assert row["outcome"] == "no_face" and row["conf_win"] is None
    assert np.array(row["conf_full"])[lb.EP_SKIN, lb.NOT_PARSED_COL] == 100 * 100
    row = rn.evaluate_image(data, _png(gt), FakeLandmarkService("multiple_faces_detected", 2), FakeParser())
    assert row["outcome"] == "multiple_faces"
    row = rn.evaluate_image(data, _png(gt), FakeLandmarkService(), FakeParser(error=RuntimeError("boom with pixels")))
    assert row["outcome"] == "pipeline_error" and row["error_type"] == "RuntimeError" and "boom" not in json.dumps(row)


# --- report ------------------------------------------------------------------------------------------------------------

def _fake_run_meta():
    return {
        "environment": {
            "python": "3.13", "packages": {"numpy": "x"}, "os": "test-os", "cpu": "test-cpu", "cpu_count": 1,
            "parser_id": "stub", "parser_weights_sha256": "0" * 64, "landmarker_sha256": "1" * 64, "crop_scale": 2.0,
            "parser_threads_setting": 0, "opencv_threads": 1, "source_fingerprint_sha256": "2" * 64, "git": "no git",
        },
        "summary": {"n": 0, "outcomes": {}, "elapsed_s": 1.0, "peak_rss_mb": 1.0, "stopped_early": False},
        "n_sample": 0, "warmups": 3,
    }


def _synthetic_rows(n=40, seed=0):
    """Rows with random-but-plausible confusion matrices, a few failures, no pipeline needed."""
    rng = np.random.default_rng(seed)
    ids = [f"{i + 1:08x}-2222-4333-8444-{i + 1:012x}" for i in range(n)]
    entries = [ds.SampleEntry(i, ds.split_of(i), k % 4, k % 3) for k, i in enumerate(ids)]
    rows = []
    for k, e in enumerate(entries):
        conf = np.zeros((9, 20), dtype=np.int64)
        conf[lb.EP_SKIN, int(Region.SKIN)] = rng.integers(8000, 9000)
        conf[lb.EP_SKIN, int(Region.NECK)] = rng.integers(500, 900)
        conf[lb.EP_SKIN, lb.NOT_PARSED_COL] = rng.integers(100, 300)
        conf[lb.EP_LIPS, int(Region.LIP_UPPER)] = rng.integers(200, 260)
        conf[lb.EP_LIPS, int(Region.LIP_LOWER)] = rng.integers(200, 260)
        conf[lb.EP_LIPS, int(Region.SKIN)] = 40
        conf[lb.EP_LEFT_EYE, int(Region.EYE_R)] = rng.integers(100, 140)
        conf[lb.EP_RIGHT_EYE, int(Region.EYE_L)] = rng.integers(100, 140)
        conf[lb.EP_LEFT_BROW, int(Region.BROW_R)] = rng.integers(150, 190)
        conf[lb.EP_RIGHT_BROW, int(Region.BROW_L)] = rng.integers(150, 190)
        conf[lb.EP_TEETH, int(Region.MOUTH_INTERIOR)] = rng.integers(0, 50)
        conf[lb.EP_BACKGROUND, int(Region.BACKGROUND)] = 20000
        outcome = "no_face" if k == 3 else "multiple_faces" if k == 7 else "ok"
        row = {
            "id": e.image_id, "outcome": outcome, "height": 1920, "width": 1440, "decode_ms": 20.0, "wall_ms": 200.0,
            "gt_counts": conf.sum(axis=1).tolist(), "gt_iod": float(rng.uniform(80, 300)), "gt_skin_luma": float(rng.uniform(60, 200)),
            "order_ok": {"eye": True, "brow": k != 5},
        }
        if outcome == "ok":
            row.update(conf_full=conf.tolist(), conf_win=(conf - np.where(np.arange(20) == lb.NOT_PARSED_COL, conf, 0)).tolist(),
                       timings_ms={s: float(rng.uniform(5, 50)) for s in rp.STAGES}, warnings=["glasses_detected"] if k % 9 == 0 else [],
                       alignment_passed=bool(k % 11), rect=[0, 0, 10, 10], window=[0, 0, 10],
                       small={n: {"bf": {"0.02": float(rng.uniform(0.5, 1)), "0.05": float(rng.uniform(0.8, 1))}, "shift": [int(rng.integers(-1, 2)), 0, 0.8]} for n in ("lips", "eye", "brow", "whole_mouth")},
                       landmark_polygons={"eye": [100, 20, 30], "mouth": [400, 80, 60]})
        else:
            nc = np.zeros((9, 20), dtype=np.int64)
            nc[:, lb.NOT_PARSED_COL] = conf.sum(axis=1)
            row.update(conf_full=nc.tolist(), conf_win=None)
        rows.append(row)
    rows.append({"id": "ffffffff-2222-4333-8444-ffffffffffff", "outcome": "invalid_sample", "reason": "annotation_shape"})
    entries.append(ds.SampleEntry("ffffffff-2222-4333-8444-ffffffffffff", ds.split_of("ffffffff-2222-4333-8444-ffffffffffff"), 0, 0))
    return rows, entries


def _build(rows, entries):
    rows = copy.deepcopy(rows)
    for r in rows:
        for k in ("conf_full", "conf_win"):
            if r.get(k) is not None:
                r[k] = np.array(r[k], dtype=np.int64)
    return rp.build_report(rows, entries, _fake_run_meta(), "docs/evaluation/sample.txt", "2026-01-01")


def test_report_has_every_section_and_json_matches_markdown():
    rows, entries = _synthetic_rows()
    rep = _build(rows, entries)
    md = rp.render_markdown(rep)
    for heading in ("Run header", "Outcomes", "Headline tables", "Denominator A, scoring: full image", "Denominator B, scoring: inside the parse window",
                    "Boundary agreement", "Side handling", "Confusion view", "Stratified agreement", "Worst-stratum gaps", "Latency",
                    "Landmark polygons vs parser", "Worst 10 images", "Weak regions", "Dev and test splits", "Provisional targets",
                    "Decision rules applied", "Limitations", "Licence and attribution"):
        assert heading in md, heading
    assert rep["outcomes"] == {"ok": 38, "no_face": 1, "multiple_faces": 1, "pipeline_error": 0, "invalid_sample": 1}
    assert rep["invalid_reasons"] == {"annotation_shape": 1}
    # JSON and Markdown agree on the numbers
    skin = rep["headline"]["A_full"]["skin"]
    assert f"{skin['iou']:.3f}" in md and f"[{skin['ci95']['iou'][0]:.3f}, {skin['ci95']['iou'][1]:.3f}]" in md
    assert rep["headline"]["B_full"]["skin"]["iou"] < skin["iou"]  # failures count as empty predictions
    assert rep["headline"]["B_full"]["skin"]["n_images"] == 40 and skin["n_images"] == 38
    json.dumps(rep)  # serialisable
    assert "provisional" in md.lower() and "easyportrait" in md.lower() and "arXiv:2304.13509" in md and "Kirillova" in md
    assert "low-sample" in md  # n < 30 cells are flagged (the synthetic strata are small)
    assert rep["side_handling"]["brow"]["gt_order_exceptions"] == 1


def test_accuracy_is_never_unqualified():
    rows, entries = _synthetic_rows()
    md = rp.render_markdown(_build(rows, entries))
    for m in re.finditer(r"accuracy", md, flags=re.I):
        before = md[max(0, m.start() - 12) : m.start()].lower()
        assert before.endswith(("not an ", "no ")), md[max(0, m.start() - 60) : m.end() + 30]


def test_report_contains_no_per_image_pixels_or_user_ids():
    rows, entries = _synthetic_rows()
    rep = _build(rows, entries)
    text = json.dumps(rep) + rp.render_markdown(rep)
    assert "user_id" not in text and "SECRET" not in text
    allowed_ids = {e.image_id for e in entries}
    for found in re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text):
        assert found in allowed_ids


def test_bootstrap_in_report_is_deterministic():
    rows, entries = _synthetic_rows()
    assert json.dumps(_build(rows, entries), sort_keys=True) == json.dumps(_build(rows, entries), sort_keys=True)


def test_run_outputs_must_be_outside_the_repo(tmp_path):
    with pytest.raises(ds.CacheError):
        rn.run_evaluation([], tmp_path, ds.REPO_ROOT / "docs" / "evaluation" / "run", services=(None, None))  # type: ignore[arg-type]
