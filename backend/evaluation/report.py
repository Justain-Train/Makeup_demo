"""Aggregate per-image rows into the report (Markdown + JSON, generated from the same dict).

Only aggregates and opaque dataset ids are written. The wording is "agreement with EasyPortrait annotations";
the report makes no accuracy claim.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import metrics
from evaluation.dataset import (
    DEFAULT_N,
    DEFAULT_SEED,
    PINNED_LENGTHS,
    RESOLUTION_GROUPS,
    SampleEntry,
)
from evaluation.labels import (
    COL_NAMES,
    COMPARISON_BY_NAME,
    COMPARISONS,
    EP_NAMES,
    HEADLINE,
    MIN_PIXELS,
    SIDE_PAIRS,
    SIDED,
    column_ids,
)

BOOT_SEED = 20261110
BOOT_N = 1000
LOW_SAMPLE_N = 30
SCORINGS = ("full", "window")
DENOMS = ("A", "B")
MP_BUCKETS = (("<1 MP", 0.0, 1.0), ("1-2 MP", 1.0, 2.0), ("2-4 MP", 2.0, 4.0), (">=4 MP", 4.0, 1e9))
STAGES = ("landmarks", "preprocess", "inference", "postprocess", "zones", "encode", "total")

# Provisional targets (plan, decision D2). IoU = dataset-level, full-image scoring, denominator A.
TARGETS = {
    "skin": {"iou": 0.90, "bf": None},
    "lips": {"iou": 0.75, "bf": 0.80},
    "eye": {"iou": 0.65, "bf": 0.80},
    "brow": {"iou": 0.60, "bf": 0.80},
}
SWAP_TARGET = 0.01
GAP_TARGET = {"skin": 0.05, "small": 0.08}
LATENCY_TARGET_MS = 300.0
MISS_TOLERANCE = 0.03

LICENCE_TEXT = (
    "Ground truth: EasyPortrait (Kapitanov, Kvanchiani, Kirillova, 2023), https://github.com/hukenovs/easyportrait . "
    "Licence: \"a variant of Creative Commons Attribution-ShareAlike 4.0 International\" (a reworked, non-Creative-Commons "
    "public licence with attribution and conditions reserved; no non-commercial clause; the privacy and portrait rights of "
    "the people depicted are not granted): https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf . "
    "Use here: evaluation only, aggregate results only. No images or annotations are redistributed, and none are used "
    "for training (that needs the separate legal review listed as a pre-deployment blocker)."
)
BIBTEX = (
    "@article{EasyPortrait,\n"
    "    title={EasyPortrait - Face Parsing and Portrait Segmentation Dataset},\n"
    "    author={Kapitanov, Alexander and Kvanchiani, Karina and Kirillova Sofia},\n"
    "    journal={arXiv preprint arXiv:2304.13509},\n"
    "    year={2023}\n"
    "}"
)


# --- loading -------------------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for r in rows:
        for key in ("conf_full", "conf_win"):
            if r.get(key) is not None:
                r[key] = np.array(r[key], dtype=np.int64)
    return rows


def annotate_rows(rows: list[dict[str, Any]], entries: list[SampleEntry]) -> list[dict[str, Any]]:
    """Attach split and sampling strata to each row (rows without a sample entry are dropped)."""
    by_id = {e.image_id: e for e in entries}
    out = []
    for r in rows:
        e = by_id.get(r["id"])
        if e is None:
            continue
        r["split"], r["bq"], r["rg"] = e.split, e.brightness_quartile, e.resolution_group
        out.append(r)
    return out


def _round(x: Any, nd: int = 6) -> Any:
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _round(float(x), nd)
    return x


# --- aggregation ---------------------------------------------------------------------------------

def valid_rows(rows: list[dict[str, Any]], denom: str) -> list[dict[str, Any]]:
    if denom == "A":
        return [r for r in rows if r["outcome"] == "ok"]
    return [r for r in rows if r["outcome"] != "invalid_sample"]


def _conf(r: dict[str, Any], scoring: str) -> np.ndarray:
    # In-window scoring: a failed image has no window, so its whole GT counts as missed (documented).
    if scoring == "window" and r.get("conf_win") is not None:
        return r["conf_win"]
    return r["conf_full"]


def _row_counts(r: dict[str, Any], comp_name: str, scoring: str) -> metrics.Counts:
    comp = COMPARISON_BY_NAME[comp_name]
    return metrics.counts_from_confusion(_conf(r, scoring), comp.ep_ids, column_ids(comp.canonical))


def aggregate_class(rows: list[dict[str, Any]], comp_name: str, scoring: str) -> dict[str, Any]:
    counts = [_row_counts(r, comp_name, scoring) for r in rows]
    arr = np.array([[c.tp, c.fp, c.fn] for c in counts], dtype=np.int64).reshape(-1, 3)
    tot = arr.sum(axis=0) if len(arr) else np.zeros(3, dtype=np.int64)
    sc = metrics.scores(*[float(v) for v in tot])
    ci = metrics.bootstrap_scores(arr, BOOT_N, BOOT_SEED) if len(arr) else {}
    gt_px, pred_px = int(tot[0] + tot[2]), int(tot[0] + tot[1])
    return {
        "n_images": len(rows),
        "gt_pixels": gt_px,
        "pred_pixels": pred_px,
        **sc,
        "ci95": {k: list(v) for k, v in ci.items()},
        "per_image_iou": metrics.per_image_iou_stats(counts),
        "false_alarm": metrics.false_alarm_rate(counts),
        "area_ratio": None if gt_px == 0 else pred_px / gt_px,
    }


def gt_outside_window_share(rows_ok: list[dict[str, Any]], comp_name: str) -> float | None:
    comp = COMPARISON_BY_NAME[comp_name]
    full = sum(int(r["conf_full"][list(comp.ep_ids)].sum()) for r in rows_ok)
    win = sum(int(_conf(r, "window")[list(comp.ep_ids)].sum()) for r in rows_ok)
    return None if full == 0 else (full - win) / full


def boundary_summary(rows_ok: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Per-image boundary F (mean/median) and best-shift summary for a small comparison class."""
    out: dict[str, Any] = {}
    for tau in ("0.02", "0.05"):
        vals = []
        for r in rows_ok:
            entry = r.get("small", {}).get(name)
            comp = COMPARISON_BY_NAME[name]
            gt_px = int(r["conf_full"][list(comp.ep_ids)].sum())
            if entry and "bf" in entry and entry["bf"].get(tau) is not None and gt_px >= MIN_PIXELS:
                vals.append(entry["bf"][tau])
        out[f"bf_{tau}"] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
        }
    shifts = [r["small"][name]["shift"] for r in rows_ok if r.get("small", {}).get(name, {}).get("shift")]
    if shifts:
        s = np.array(shifts, dtype=float)
        out["best_shift"] = {
            "n": len(shifts),
            "mean_dx": float(s[:, 0].mean()), "mean_dy": float(s[:, 1].mean()),
            "median_dx": float(np.median(s[:, 0])), "median_dy": float(np.median(s[:, 1])),
            "share_nonzero": float(np.mean((s[:, 0] != 0) | (s[:, 1] != 0))),
            "iou_at_best_shift_mean": float(s[:, 2].mean()),
        }
    else:
        out["best_shift"] = None
    return out


def headline_tables(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for denom in DENOMS:
        for scoring in SCORINGS:
            sub = valid_rows(rows, denom)
            out[f"{denom}_{scoring}"] = {n: aggregate_class(sub, n, scoring) for n in names}
    return out


def side_handling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = valid_rows(rows, "A")
    out: dict[str, Any] = {}
    for kind, (ep_first, ep_second, reg_first, reg_second) in SIDE_PAIRS.items():
        order = [r["order_ok"][kind] for r in valid_rows(rows, "B") if r.get("order_ok", {}).get(kind) is not None]
        exceptions = sum(1 for o in order if not o)
        swaps = [
            metrics.side_swap(r["conf_full"], ep_first, ep_second, int(reg_first), int(reg_second)) for r in ok
        ]
        defined = [s for s in swaps if s is not None]
        out[kind] = {
            "gt_order_defined": len(order),
            "gt_order_exceptions": exceptions,
            "gt_order_exception_rate": exceptions / len(order) if order else None,
            "swap_defined": len(defined),
            "swapped": sum(1 for s in defined if s),
            "swap_rate": (sum(1 for s in defined if s) / len(defined)) if defined else None,
        }
    rate = [v["gt_order_exception_rate"] for v in out.values() if v["gt_order_exception_rate"] is not None]
    out["sided_rows_flagged"] = bool(rate and max(rate) > 0.02)
    return out


def confusion_view(rows_ok: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum((r["conf_full"] for r in rows_ok), np.zeros((9, len(COL_NAMES)), dtype=np.int64))
    return {"columns": list(COL_NAMES), "rows": list(EP_NAMES), "counts": total.tolist()}


def prediction_view(rows_ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Where our predicted pixels fall: for each predicted group, the share lying in each EasyPortrait class."""
    total = sum((r["conf_full"] for r in rows_ok), np.zeros((9, len(COL_NAMES)), dtype=np.int64))
    out: dict[str, Any] = {"ep_classes": list(EP_NAMES)}
    for name, comp in (("skin + nose", COMPARISON_BY_NAME["skin"]), ("lips", COMPARISON_BY_NAME["lips"]),
                       ("eye", COMPARISON_BY_NAME["eye"]), ("brow", COMPARISON_BY_NAME["brow"])):
        col = total[:, column_ids(comp.canonical)].sum(axis=1)
        out[name] = {"pixels": int(col.sum()), "share_by_ep_class": (col / col.sum()).tolist() if col.sum() else None}
    return out


def latency(rows_ok: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, lo, hi in (("all", 0.0, 1e9), *MP_BUCKETS):
        sub = [r for r in rows_ok if lo <= r["height"] * r["width"] / 1e6 < hi and "timings_ms" in r]
        entry: dict[str, Any] = {"n": len(sub)}
        for st in STAGES:
            vals = [r["timings_ms"][st] for r in sub if st in r["timings_ms"]]
            entry[st] = {"p50": metrics.percentile(vals, 50), "p95": metrics.percentile(vals, 95), "max": max(vals) if vals else None}
        for key in ("decode_ms", "wall_ms"):
            vals = [r[key] for r in sub if key in r]
            entry[key] = {"p50": metrics.percentile(vals, 50), "p95": metrics.percentile(vals, 95), "max": max(vals) if vals else None}
        out[label] = entry
    return out


def landmark_vs_parser(rows_ok: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows_ok if r.get("landmark_polygons")]
    out: dict[str, Any] = {"n_images": len(rows)}
    for lm_name, comp_name in (("eye", "eye"), ("mouth", "whole_mouth")):
        lm = np.array([r["landmark_polygons"][lm_name] for r in rows], dtype=np.int64).reshape(-1, 3)
        pa = np.array([[c.tp, c.fp, c.fn] for c in (_row_counts(r, comp_name, "full") for r in rows)], dtype=np.int64).reshape(-1, 3)
        out[lm_name] = {
            "landmark_polygon": {**metrics.scores(*lm.sum(axis=0).astype(float)), "ci95": {k: list(v) for k, v in metrics.bootstrap_scores(lm, BOOT_N, BOOT_SEED).items()}} if len(lm) else None,
            "parser": {**metrics.scores(*pa.sum(axis=0).astype(float)), "ci95": {k: list(v) for k, v in metrics.bootstrap_scores(pa, BOOT_N, BOOT_SEED).items()}} if len(pa) else None,
            "gt": "EP eyes (LEFT+RIGHT)" if lm_name == "eye" else "EP lips + teeth (whole mouth)",
        }
    return out


def worst_ids(rows_ok: list[dict[str, Any]], k: int = 10) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name in HEADLINE + SIDED:
        scored = []
        for r in rows_ok:
            c = _row_counts(r, name, "full")
            if c.gt >= MIN_PIXELS:
                scored.append((c.tp / (c.tp + c.fp + c.fn), r["id"]))
        scored.sort()
        out[name] = [{"id": i, "iou": v} for v, i in scored[:k]]
    return out


# --- strata ----------------------------------------------------------------------------------------

def tercile_labels(values: list[float | None]) -> list[str | None]:
    arr = np.array([v for v in values if v is not None])
    if arr.size < 3:
        return [None] * len(values)
    e1, e2 = np.quantile(arr, [1 / 3, 2 / 3])
    return [None if v is None else ("low" if v <= e1 else "mid" if v <= e2 else "high") for v in values]


def assign_strata(rows: list[dict[str, Any]]) -> None:
    valid = [r for r in rows if r["outcome"] != "invalid_sample"]
    for key, field in (("luma", "gt_skin_luma"), ("face_size", "gt_iod")):
        labels = tercile_labels([r.get(field) for r in valid])
        for r, lab in zip(valid, labels):
            r.setdefault("strata", {})[key] = lab
    for r in valid:
        s = r.setdefault("strata", {})
        s["brightness_quartile"] = f"Q{r['bq'] + 1}"
        s["resolution"] = RESOLUTION_GROUPS[r["rg"]]
        ok = r["outcome"] == "ok"
        s["glasses_detected"] = None if not ok else ("yes" if "glasses_detected" in r.get("warnings", []) else "no")
        s["alignment_passed"] = None if not ok or r.get("alignment_passed") is None else ("yes" if r["alignment_passed"] else "no")


def stratified_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = valid_rows(rows, "A")
    out: dict[str, Any] = {}
    for strat in ("brightness_quartile", "resolution", "luma", "face_size", "glasses_detected", "alignment_passed"):
        cells = sorted({r["strata"][strat] for r in ok if r["strata"].get(strat) is not None})
        out[strat] = {}
        for cell in cells:
            sub = [r for r in ok if r["strata"].get(strat) == cell]
            out[strat][cell] = {
                "n": len(sub),
                "low_sample": len(sub) < LOW_SAMPLE_N,
                **{n: {k: v for k, v in aggregate_class(sub, n, "full").items() if k in ("iou", "ci95", "gt_pixels")} for n in HEADLINE},
            }
    return out


def worst_stratum_gaps(strata: dict[str, Any]) -> dict[str, Any]:
    """Gap between the best and worst stratum (cells with n >= 30) for the brightness and luma stratifiers.

    point = best IoU - worst IoU. lower_bound = max(0, best CI low - worst CI high): the smallest gap
    the bootstrap intervals allow. The target is judged on the lower bound.
    """
    out: dict[str, Any] = {}
    for strat in ("brightness_quartile", "luma"):
        for name in HEADLINE:
            cells = [(c, v[name]) for c, v in strata.get(strat, {}).items() if not v["low_sample"] and v[name]["iou"] is not None]
            if len(cells) < 2:
                out[f"{strat}/{name}"] = None
                continue
            best = max(cells, key=lambda t: t[1]["iou"])
            worst = min(cells, key=lambda t: t[1]["iou"])
            out[f"{strat}/{name}"] = {
                "best_cell": best[0], "worst_cell": worst[0],
                "point": best[1]["iou"] - worst[1]["iou"],
                "lower_bound": max(0.0, best[1]["ci95"]["iou"][0] - worst[1]["ci95"]["iou"][1]),
            }
    return out


# --- diagnosis, targets and decision rules -------------------------------------------------------------

def dominant_modes(conf_total: np.ndarray, name: str) -> dict[str, Any]:
    """Where the GT pixels of a class go when they are missed, and where the predicted pixels come from."""
    comp = COMPARISON_BY_NAME[name]
    cols = column_ids(comp.canonical)
    ep = list(comp.ep_ids)
    gt_total = conf_total[ep].sum()
    missed = conf_total[ep].sum(axis=0).astype(float)
    missed[cols] = 0
    pred_total = conf_total[:, cols].sum()
    extra = conf_total[:, cols].sum(axis=1).astype(float)
    extra[ep] = 0
    top_m = int(missed.argmax()) if missed.sum() else None
    top_f = int(extra.argmax()) if extra.sum() else None
    return {
        "missed_share_of_gt": float(missed.sum() / gt_total) if gt_total else None,
        "missed_to": None if top_m is None else {"class": COL_NAMES[top_m], "share_of_gt": float(missed[top_m] / gt_total)},
        "false_share_of_pred": float(extra.sum() / pred_total) if pred_total else None,
        "false_from": None if top_f is None else {"class": EP_NAMES[top_f], "share_of_pred": float(extra[top_f] / pred_total)},
    }


def evaluate_targets(
    rows: list[dict[str, Any]], head: dict[str, Any], bsum: dict[str, Any], sides: dict[str, Any], gaps: dict[str, Any], lat: dict[str, Any]
) -> dict[str, Any]:
    a_full = head["A_full"]
    items: list[dict[str, Any]] = []

    def judge(value: float | None, target: float, ci_hi: float | None, higher_better: bool = True) -> str:
        if value is None:
            return "not assessable"
        if (value >= target) == higher_better:
            return "met"
        miss = abs(target - value)
        overlap = ci_hi is not None and ci_hi >= target
        return "met within tolerance (miss <= 0.03 and CI overlaps target)" if miss <= MISS_TOLERANCE and overlap else "missed"

    for name, tgt in TARGETS.items():
        agg = a_full[name]
        items.append({
            "item": f"{name} IoU", "target": tgt["iou"], "value": agg["iou"],
            "ci95": agg["ci95"].get("iou"), "result": judge(agg["iou"], tgt["iou"], (agg["ci95"].get("iou") or [None, None])[1]),
        })
        if tgt["bf"] is not None:
            bf = bsum[name]["bf_0.02"]["mean"]
            items.append({"item": f"{name} boundary F @ 0.02*IOD (mean per image)", "target": tgt["bf"], "value": bf, "ci95": None, "result": judge(bf, tgt["bf"], None)})
    for kind in ("eye", "brow"):
        sr = sides[kind]["swap_rate"]
        items.append({"item": f"side-swap rate ({kind})", "target": SWAP_TARGET, "value": sr, "ci95": None,
                      "result": "not assessable" if sr is None else ("met" if sr <= SWAP_TARGET else "missed")})
    for key, g in gaps.items():
        if g is None:
            items.append({"item": f"worst-stratum gap {key}", "target": None, "value": None, "ci95": None, "result": "not assessable (fewer than 2 cells with n >= 30)"})
            continue
        tgt = GAP_TARGET["skin"] if key.endswith("/skin") else GAP_TARGET["small"]
        items.append({"item": f"worst-stratum gap {key} ({g['best_cell']} vs {g['worst_cell']})", "target": tgt, "value": g["point"],
                      "ci95": None, "lower_bound": g["lower_bound"], "result": "met" if g["lower_bound"] <= tgt else "missed"})
    l12 = lat.get("1-2 MP", {})
    p95 = (l12.get("total") or {}).get("p95")
    items.append({"item": "latency p95, 1-2 MP images, this machine (segment_face total)", "target": LATENCY_TARGET_MS,
                  "value": p95 if l12.get("n", 0) >= 10 else None, "ci95": None,
                  "result": "not assessable (fewer than 10 images in the 1-2 MP bucket)" if l12.get("n", 0) < 10 else ("met" if p95 <= LATENCY_TARGET_MS else "missed"),
                  "note": f"n = {l12.get('n', 0)}; PROVISIONAL, not measured on AC power: the pilot ran on battery with system sleep events in its window, so these timings are unreliable. "
                          "The re-run benchmark (docs/evaluation/benchmark_2026-11-10.md; 2 MP p95 176 ms, 0 sleep events, still on battery) is the latency reference; the 2-4 MP bucket is the nearest large one"})
    return {"provisional": True, "items": items}


def apply_decision_rules(
    rows: list[dict[str, Any]], head: dict[str, Any], bsum: dict[str, Any], targets: dict[str, Any], modes: dict[str, Any], gaps: dict[str, Any],
    outcomes: dict[str, int], rows_ok: list[dict[str, Any]],
) -> dict[str, Any]:
    n_valid = len(valid_rows(rows, "B"))
    fail_rate = (n_valid - outcomes.get("ok", 0)) / n_valid if n_valid else None
    results = [i["result"] for i in targets["items"]]
    unmet = [i for i in targets["items"] if i["result"] == "missed"]
    rule1 = {
        "all_targets_met_or_within_tolerance": bool(not unmet and all(r != "not assessable" for r in results)),
        "unmet": [i["item"] for i in unmet],
        "not_assessable": [i["item"] for i in targets["items"] if i["result"].startswith("not assessable")],
        "detection_failure_rate": fail_rate,
        "detection_failure_rate_ok": None if fail_rate is None else fail_rate <= 0.02,
        "note": "failure rate is over ALL valid images; 'fully visible face' is not annotated, so it cannot be restricted",
    }
    rule2: dict[str, Any] = {
        "triggers": [],
        "not_assessed": [],
        "roll_from_wp3": (
            "Update from the robustness suite (WP3, 100 dev images): roll of +/-30 deg costs 0.12-0.16 IoU and +/-45 deg costs 0.35-0.54 "
            "while detection stays 97-99%, so the roll item of rule 2 IS met; a roll-aligned crop is a WP6 candidate and needs user approval."
        ),
    }
    rule3: dict[str, Any] = {
        "triggers": [],
        "caveat": (
            "Caveat: no trigger fires on the headline numbers, but the glasses stratum (eye IoU 0.00, n=48) and the alignment-failure stratum "
            "(eye IoU 0.168, n=93) are very poor; all 48 glasses images are among the 93 alignment failures. "
            "The fine-tuning / replacement decision is NOT made here; it is owed to the user."
        ),
    }
    a_full = head["A_full"]
    for name in HEADLINE:
        iou_t = TARGETS[name]["iou"]
        agg, bs = a_full[name], bsum.get(name)
        iou = agg["iou"]
        low = iou is not None and iou < iou_t
        if bs:
            bf05 = bs["bf_0.05"]["mean"]
            if low and bf05 is not None and bf05 >= 0.85:
                rule2["triggers"].append(f"{name}: boundary F @ 0.05*IOD = {bf05:.3f} >= 0.85 while IoU {iou:.3f} < {iou_t}")
            if bf05 is not None and bf05 < 0.70:
                rule3["triggers"].append(f"{name}: boundary F @ 0.05*IOD = {bf05:.3f} < 0.70 (wrong shape or place)")
            sh = bs.get("best_shift")
            if sh and (abs(sh["median_dx"]) >= 1 or abs(sh["median_dy"]) >= 1):
                rule2["triggers"].append(f"{name}: consistent best shift (median dx {sh['median_dx']:+.0f}, dy {sh['median_dy']:+.0f} px)")
        ar = agg["area_ratio"]
        if ar is not None and not (0.85 <= ar <= 1.15):
            rule2["triggers"].append(f"{name}: area ratio {ar:.2f} outside 0.85-1.15")
        fa = agg["false_alarm"]
        if fa["rate"] is not None and fa["rate"] > 0.02 and fa["n_absent"] >= 10:
            rule2["triggers"].append(f"{name}: stray false positives in {fa['n_false_alarm']} of {fa['n_absent']} images with no GT region")
        m = modes.get(name)
        out_share = m.get("gt_outside_window_share") if m else None
        if low and out_share is not None and out_share >= 0.5 * (1 - (agg["recall"] or 0)):
            rule2["triggers"].append(f"{name}: GT outside the parse window ({out_share:.1%}) explains most missed pixels")
        if name in ("lips", "eye", "brow") and iou is not None and iou < 0.50:
            rule3["triggers"].append(f"{name}: IoU {iou:.3f} < 0.50 (before any fixes)")
    for key, g in gaps.items():
        if g and g["lower_bound"] > 0.10:
            rule3["triggers"].append(f"stratum gap {key}: lower bound {g['lower_bound']:.3f} > 0.10")
    skin_a, skin_b = head["A_full"]["skin"], head["B_full"]["skin"]
    tp_b = (skin_b["recall"] or 0) * skin_b["gt_pixels"]
    lost_fail = (skin_b["gt_pixels"] - skin_a["gt_pixels"])  # GT pixels of images that failed detection (all missed)
    lost_total_b = skin_b["gt_pixels"] - tp_b
    rule4 = {
        "skin_gt_pixels_lost_to_detection_failures": lost_fail,
        "skin_gt_pixels_missed_total_B": lost_total_b,
        "share_of_B_loss_from_detection": (lost_fail / lost_total_b) if lost_total_b else None,
        "detector_lever": bool(lost_total_b and lost_fail / lost_total_b > 0.5),
    }
    return {
        "provisional": True,
        "evaluated_on": "pilot: full sample (this is not the final test-split decision, which follows the protocol)",
        "rule1_acceptable_for_prototype": rule1,
        "rule2_post_processing_or_geometry_fix": rule2,
        "rule3_model_replacement_or_fine_tuning": rule3,
        "rule4_detector_lever": rule4,
        "rule5_licence": "Regardless of the outcome, the CelebAMask-HQ model is replaced before any commercial release; this harness is the comparison tool for that replacement.",
    }


# --- assembly ---------------------------------------------------------------------------------------------

def build_report(
    rows: list[dict[str, Any]], entries: list[SampleEntry], run_meta: dict[str, Any], sample_file: str, report_date: str | None = None
) -> dict[str, Any]:
    rows = annotate_rows(rows, entries)
    assign_strata(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    ok = valid_rows(rows, "A")
    names_all = tuple(c.name for c in COMPARISONS)
    head = headline_tables(rows, names_all)
    bsum = {c.name: boundary_summary(ok, c.name) for c in COMPARISONS if c.small and c.name not in SIDED}
    sides = side_handling(rows)
    strata = stratified_tables(rows)
    gaps = worst_stratum_gaps(strata)
    lat = latency(ok)
    conf_total = np.array(confusion_view(ok)["counts"], dtype=np.int64)
    modes = {}
    for name in names_all:
        m = dominant_modes(conf_total, name)
        m["gt_outside_window_share"] = gt_outside_window_share(ok, name)
        modes[name] = m
    targets = evaluate_targets(rows, head, bsum, sides, gaps, lat)
    decisions = apply_decision_rules(rows, head, bsum, targets, modes, gaps, dict(outcomes), ok)
    subsets = {}
    for split in ("dev", "test"):
        sub = [r for r in rows if r["split"] == split]
        subsets[split] = {"n_images": len(sub), "outcomes": dict(Counter(r["outcome"] for r in sub)),
                          "headline": headline_tables(sub, HEADLINE + SIDED)}
    ranking = []
    for name in HEADLINE:
        iou = head["A_full"][name]["iou"]
        gap = None if iou is None else TARGETS[name]["iou"] - iou
        ranking.append({"class": name, "iou": iou, "target": TARGETS[name]["iou"], "shortfall": gap, **modes[name]})
    ranking.sort(key=lambda r: -(r["shortfall"] if r["shortfall"] is not None else -9))
    report = {
        "title": "Agreement of the face-parsing pipeline with EasyPortrait v2 test annotations (pilot)",
        "date": report_date or date.today().isoformat(),
        "sample": {"file": sample_file, "n_listed": len(entries), "seed": DEFAULT_SEED, "n_target": DEFAULT_N,
                   "dev": sum(1 for e in entries if e.split == "dev"), "test": sum(1 for e in entries if e.split == "test")},
        "pinned_archive_lengths": PINNED_LENGTHS,
        "run": run_meta,
        "outcomes": {k: outcomes.get(k, 0) for k in ("ok", "no_face", "multiple_faces", "pipeline_error", "invalid_sample")},
        "invalid_reasons": dict(Counter(r.get("reason", "?") for r in rows if r["outcome"] == "invalid_sample")),
        "pipeline_error_types": dict(Counter(r.get("error_type", "?") for r in rows if r["outcome"] == "pipeline_error")),
        "headline": head,
        "boundary_and_shift": bsum,
        "side_handling": sides,
        "confusion": confusion_view(ok),
        "prediction_view": prediction_view(ok),
        "strata": strata,
        "worst_stratum_gaps": gaps,
        "latency_ms": lat,
        "landmark_polygon_vs_parser": landmark_vs_parser(ok),
        "worst_ids": worst_ids(ok),
        "weak_regions": ranking,
        "error_modes": modes,
        "dev_test": subsets,
        "targets": targets,
        "decision_rules": decisions,
        "licence": LICENCE_TEXT,
        "bibtex": BIBTEX,
    }
    return _round(report)


# --- Markdown ----------------------------------------------------------------------------------------------

def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(ci: Any) -> str:
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _headline_md(block: dict[str, Any], names: tuple[str, ...]) -> str:
    body = []
    for n in names:
        a = block[n]
        pi = a["per_image_iou"]
        body.append([n, str(a["n_images"]), _f(a["iou"]), _ci(a["ci95"].get("iou")), _f(a["dice"]), _f(a["precision"]), _f(a["recall"]),
                     f"{_f(pi['mean'])} / {_f(pi['median'])} / {_f(pi['p10'])} / {_f(pi['p90'])} (n={pi['n']})",
                     _f(a["area_ratio"], 2), "n/a" if a["false_alarm"]["rate"] is None else f"{a['false_alarm']['rate']:.1%} of {a['false_alarm']['n_absent']}"])
    return _table(["class", "images", "IoU", "IoU 95% CI", "Dice", "precision", "recall", "per-image IoU mean / median / p10 / p90", "area ratio", "false alarms"], body)


def _current_fingerprint() -> str:
    """Fingerprint of the source tree as it is when the report is generated (not stored in the JSON numbers)."""
    from evaluation.runner import source_fingerprint

    return source_fingerprint()


def render_markdown(rep: dict[str, Any]) -> str:
    L: list[str] = []
    run, env = rep["run"], rep["run"]["environment"]
    summ = rep["run"]["summary"]
    L += [f"# {rep['title']}", "", f"Date: {rep['date']}. Status: **pilot, provisional targets**. All numbers are *agreement with EasyPortrait annotations* "
          "on a stratified sample of the EasyPortrait v2 test split. They are not an accuracy figure for the product: the annotation conventions differ "
          "from the CelebAMask-HQ conventions the model was trained on, and skin-tone bias is not quantified (see Limitations).", ""]
    L += ["## Run header", "",
          f"- Sample: `{rep['sample']['file']}`: {rep['sample']['n_listed']} ids (target {rep['sample']['n_target']}, seed {rep['sample']['seed']}); dev {rep['sample']['dev']}, test {rep['sample']['test']}.",
          f"- Parser: `{env['parser_id']}`, weights SHA-256 `{env['parser_weights_sha256']}`.",
          f"- Landmarker SHA-256 `{env['landmarker_sha256']}`.",
          f"- Crop scale {env['crop_scale']}, parser threads setting {env['parser_threads_setting']} (0 = onnxruntime default), OpenCV threads {env['opencv_threads']}; sequential run, {run['warmups']} warm-ups.",
          f"- Python {env['python']}; packages {', '.join(f'{k} {v}' for k, v in env['packages'].items())}.",
          f"- OS {env['os']}; CPU {env['cpu']} ({env['cpu_count']} logical cores).",
          f"- Source-tree fingerprint of the **pilot-run tree** (recorded in run_meta.json when the run was made; SHA-256 over backend/app and backend/evaluation `*.py`): `{env['source_fingerprint_sha256']}`. {env['git']}.",
          f"- Source-tree fingerprint of the **current tree at report time** (the evaluation code has changed since the run, e.g. this report generator): `{_current_fingerprint()}`.",
          f"- Elapsed {summ['elapsed_s']:.1f} s for {summ['n']} images; peak RSS {summ['peak_rss_mb']:.0f} MB (process, includes the MediaPipe landmarker).", ""]
    oc = rep["outcomes"]
    L += ["## Outcomes", "", _table(["outcome", "images"], [[k, str(v)] for k, v in oc.items()]), "",
          f"Invalid-sample reasons: {rep['invalid_reasons'] or 'none'}. Pipeline-error types: {rep['pipeline_error_types'] or 'none'}.",
          "Denominator A = images with outcome `ok` only. Denominator B = all valid images (everything except `invalid_sample`); a failed image counts as an empty prediction. "
          "`multiple_faces` is a deliberate rejection of the pipeline; it is still a miss in B.", ""]
    L += ["## Headline tables", "",
          "Dataset-level (pixel-weighted) scores with 95% bootstrap CIs (1000 resamples over images, seeded). Per-image statistics use images whose GT region has at least 25 px. "
          "The false-alarm column is the share of images with no GT region (< 25 px) where the prediction has at least 25 px. "
          "Comparison classes: skin = EP SKIN vs our skin + nose; lips = EP LIPS vs our upper + lower lip; eye / brow merged; sided rows map EP LEFT_* to our `_R` (see Side handling). "
          "PERSON and BACKGROUND are not scored.", ""]
    for denom in DENOMS:
        for scoring in SCORINGS:
            desc = {"full": "full image (pixels outside the parse window count as 'not predicted')",
                    "window": "inside the parse window only (outside pixels excluded from GT and prediction; for failed images in B the whole GT counts as missed)"}[scoring]
            L += [f"### Denominator {denom}, scoring: {desc}", "", _headline_md(rep["headline"][f"{denom}_{scoring}"], HEADLINE + SIDED), ""]
    L += ["### Share of GT pixels outside the parse window (denominator A)", "",
          _table(["class", "GT pixels outside window"], [[n, "n/a" if rep["error_modes"][n]["gt_outside_window_share"] is None else f"{rep['error_modes'][n]['gt_outside_window_share']:.2%}"] for n in HEADLINE + SIDED]), ""]
    L += ["### Secondary classes (not comparable; denominator A, full image)", "",
          _headline_md(rep["headline"]["A_full"], ("teeth_vs_mouth_interior", "whole_mouth")), "",
          "EP TEETH vs our `mouth_interior` and EP lips + teeth vs our lips + mouth interior are not the same definitions; read recall of teeth only.", ""]
    L += ["## Boundary agreement and registration diagnostics (denominator A, per-image means)", ""]
    body = []
    for n, b in rep["boundary_and_shift"].items():
        sh = b["best_shift"]
        body.append([n, _f(b["bf_0.02"]["mean"]), _f(b["bf_0.02"]["median"]), _f(b["bf_0.05"]["mean"]), _f(b["bf_0.05"]["median"]), str(b["bf_0.02"]["n"]),
                     "n/a" if not sh else f"median ({sh['median_dx']:+.0f}, {sh['median_dy']:+.0f}); mean ({sh['mean_dx']:+.2f}, {sh['mean_dy']:+.2f}); non-zero {sh['share_nonzero']:.0%}"])
    L += [_table(["class", "F @0.02 IOD mean", "median", "F @0.05 IOD mean", "median", "images", "best (dx, dy) within +-3 px (+ = prediction moves right/down)"], body), ""]
    sh = rep["side_handling"]
    L += ["## Side handling", "",
          "EP LEFT_* is assumed to be the image-left structure (our `_R`, MediaPipe convention). Checked per image by centroid x ordering of the GT; the swap rate is judged by overlap.", "",
          _table(["pair", "GT order defined", "GT order exceptions", "exception rate", "swap defined", "swapped", "swap rate"],
                 [[k, str(sh[k]["gt_order_defined"]), str(sh[k]["gt_order_exceptions"]), _f(sh[k]["gt_order_exception_rate"]), str(sh[k]["swap_defined"]), str(sh[k]["swapped"]), _f(sh[k]["swap_rate"])] for k in ("eye", "brow")]), "",
          f"Sided rows flagged (exceptions > 2%): **{sh['sided_rows_flagged']}**. " + ("Rely on the merged rows." if sh["sided_rows_flagged"] else "Sided rows are usable."), ""]
    cv = rep["confusion"]
    L += ["## Confusion view (denominator A, full image): where each EP class's pixels go", "",
          "Row = EasyPortrait class; entries are the share of that class's pixels assigned to each canonical class (only shares >= 0.5% shown). `NOT_PARSED` = outside the parse window.", ""]
    for i, rname in enumerate(cv["rows"]):
        tot = sum(cv["counts"][i])
        if tot == 0:
            continue
        parts = [f"{cv['columns'][j]} {cv['counts'][i][j] / tot:.1%}" for j in np.argsort(cv["counts"][i])[::-1] if cv["counts"][i][j] / tot >= 0.005]
        L.append(f"- **{rname}** ({tot:,} px): " + ", ".join(parts))
    L.append("")
    pv = rep["prediction_view"]
    L += ["### Where our predictions fall (denominator A, full image): share of each predicted group lying in each EasyPortrait class", "",
          "Complement of the view above (precision side). EP PERSON is the catch-all for the person that is not skin, brows, eyes, lips or teeth (hair, neck, clothing and the like).", ""]
    for gname in ("skin + nose", "lips", "eye", "brow"):
        shares = pv[gname]["share_by_ep_class"]
        if shares:
            parts = [f"{pv['ep_classes'][j]} {shares[j]:.1%}" for j in np.argsort(shares)[::-1] if shares[j] >= 0.003]
            L.append(f"- **our {gname}** ({pv[gname]['pixels']:,} px): " + ", ".join(parts))
    L.append("")
    L += ["## Stratified agreement (denominator A, full image; IoU with 95% CI)", "",
          "Cells with n < 30 are marked `low-sample` and excluded from the gap computation. Luma is the mean Rec.601 luma of GT skin pixels, a **lighting** indicator and **not a skin-tone measure**; "
          "skin-tone bias remains unquantified.", ""]
    for strat, cells in rep["strata"].items():
        body = []
        for cell, v in cells.items():
            body.append([cell, str(v["n"]) + (" (low-sample)" if v["low_sample"] else "")] + [f"{_f(v[n]['iou'])} {_ci(v[n]['ci95'].get('iou'))}" for n in HEADLINE])
        L += [f"### {strat}", "", _table(["cell", "images"] + list(HEADLINE), body), ""]
    gp = rep["worst_stratum_gaps"]
    L += ["### Worst-stratum gaps (brightness quartile and luma tercile)", "",
          _table(["stratifier / class", "best cell", "worst cell", "gap", "lower bound"],
                 [[k, v["best_cell"], v["worst_cell"], _f(v["point"]), _f(v["lower_bound"])] if v else [k, "n/a", "n/a", "n/a", "n/a"] for k, v in gp.items()]), ""]
    lat = rep["latency_ms"]
    L += ["## Latency (denominator A; ms; this machine only; `total` = `segment_face` end to end excluding HTTP and decode)", ""]
    body = []
    for label, v in lat.items():
        body.append([label, str(v["n"])] + [f"{_f(v[s]['p50'], 1)} / {_f(v[s]['p95'], 1)}" for s in ("landmarks", "preprocess", "inference", "postprocess", "zones", "encode", "total", "decode_ms", "wall_ms")])
    L += [_table(["megapixels", "n", "landmarks", "preprocess", "inference", "postprocess", "zones", "encode", "total", "decode", "wall"], body), "", "Cells are p50 / p95.", ""]
    lv = rep["landmark_polygon_vs_parser"]
    L += ["## Landmark polygons vs parser (same images, same GT, full image)", "", f"Images: {lv['n_images']}.", "",
          _table(["region", "GT", "landmark polygon IoU [CI]", "parser IoU [CI]"],
                 [[k, lv[k]["gt"], f"{_f(lv[k]['landmark_polygon']['iou'])} {_ci(lv[k]['landmark_polygon']['ci95'].get('iou'))}" if lv[k]["landmark_polygon"] else "n/a",
                   f"{_f(lv[k]['parser']['iou'])} {_ci(lv[k]['parser']['ci95'].get('iou'))}" if lv[k]["parser"] else "n/a"] for k in ("eye", "mouth")]), ""]
    L += ["## Worst 10 images per class (opaque dataset ids; per-image IoU, GT >= 25 px)", ""]
    for n, items in rep["worst_ids"].items():
        L.append(f"- **{n}**: " + ", ".join(f"`{i['id']}` ({i['iou']:.3f})" for i in items))
    L.append("")
    L += ["## Weak regions and dominant error mode (denominator A, full image)", "",
          _table(["class", "IoU", "target", "shortfall", "missed GT pixels go to", "false pixels come from"],
                 [[w["class"], _f(w["iou"]), _f(w["target"], 2), _f(w["shortfall"]),
                   "n/a" if not w["missed_to"] else f"{w['missed_to']['class']} ({w['missed_to']['share_of_gt']:.1%} of GT; total missed {w['missed_share_of_gt']:.1%})",
                   "n/a" if not w["false_from"] else f"{w['false_from']['class']} ({w['false_from']['share_of_pred']:.1%} of prediction; total false {w['false_share_of_pred']:.1%})"] for w in rep["weak_regions"]]), ""]
    L += ["## Dev and test splits (IoU, denominator A and B, full image)", "",
          "Future fixes are tuned on dev only; the test split is for the single final decision.", ""]
    for split in ("dev", "test"):
        d = rep["dev_test"][split]
        L += [f"### {split} ({d['n_images']} images; outcomes {d['outcomes']})", ""]
        body = [[n, f"{_f(d['headline']['A_full'][n]['iou'])} {_ci(d['headline']['A_full'][n]['ci95'].get('iou'))}", f"{_f(d['headline']['B_full'][n]['iou'])} {_ci(d['headline']['B_full'][n]['ci95'].get('iou'))}",
                 f"{_f(d['headline']['A_window'][n]['iou'])}", f"{_f(d['headline']['B_window'][n]['iou'])}"] for n in HEADLINE + SIDED]
        L += [_table(["class", "A full", "B full", "A window", "B window"], body), ""]
    L += ["## Provisional targets (decision D2: to be confirmed by the user after this pilot)", "",
          _table(["item", "target", "value", "95% CI / lower bound", "result"],
                 [[i["item"], _f(i["target"], 2), _f(i["value"]), (_ci(i["ci95"]) if i.get("ci95") else (_f(i.get("lower_bound")) if "lower_bound" in i else "n/a")), i["result"] + (f" ({i['note']})" if i.get("note") else "")] for i in rep["targets"]["items"]]), "",
          "EasyPortrait boundary conventions differ from CelebAMask-HQ's, so the IoU targets may be revised after the pilot; boundary F and area ratio are reported alongside for that reason.", ""]
    dr = rep["decision_rules"]
    r1, r2, r3, r4 = dr["rule1_acceptable_for_prototype"], dr["rule2_post_processing_or_geometry_fix"], dr["rule3_model_replacement_or_fine_tuning"], dr["rule4_detector_lever"]
    L += ["## Decision rules applied (provisional)", "", f"Evaluated on: {dr['evaluated_on']}.", "",
          f"1. **Acceptable for the prototype:** all targets met or within tolerance: **{r1['all_targets_met_or_within_tolerance']}**. Unmet: {r1['unmet'] or 'none'}. Not assessable: {r1['not_assessable'] or 'none'}. "
          f"Detection failure rate {_f(r1['detection_failure_rate'])} (limit 0.02; ok: {r1['detection_failure_rate_ok']}). {r1['note']}.",
          f"2. **Post-processing or geometry fix is enough:** triggers: {r2['triggers'] or 'none'}."
          + (f" Not assessed: {r2['not_assessed']}. " if r2['not_assessed'] else " ") + f"{r2['roll_from_wp3']}",
          f"3. **Model replacement or fine-tuning is needed (commercial path only):** triggers: {r3['triggers'] or 'none'}. {r3['caveat']}",
          f"4. **Detector lever:** share of the B loss (skin GT pixels) from detection failures {_f(r4['share_of_B_loss_from_detection'])}; detector lever: **{r4['detector_lever']}**.",
          f"5. **Licence caveat:** {dr['rule5_licence']}", ""]
    L += ["## Limitations", "",
          "- Agreement with another dataset's annotation conventions, not hand-checked ground truth for this product. The skin shortfall is a precision issue (precision 0.888, recall 0.973, area ratio 1.10): about 10.1% of our skin + nose pixels fall in EP PERSON (see 'Where our predictions fall'), not in EP SKIN.",
          "- All aggregates are over the stratified 600-image sample (brightness quartile x resolution group, floor-based allocation); they are not re-weighted to the 6000-image test split.",
          "- The y-offset cause is unexplained (label convention vs pipeline). Landmark polygons show a best shift of about (-1, -1) px while the parser shows about (0, +2) px: opposite signs. Not fixed. A vertical-flip check was inconclusive (see the robustness report).",
          "- Sample of 600 of 6000 test images, stratified by brightness and resolution only. EasyPortrait is a webcam/portrait set with many near-frontal faces: large pose, heavy occlusion and extreme lighting are under-represented.",
          "- **Skin-tone bias is not quantified.** The dataset has no skin-tone labels; luma strata measure lighting, not tone.",
          "- The perturbation (robustness) suite and the benchmark are not part of this report. Roll failures (about +/-35 deg) are known from earlier tests and not measured here.",
          "- One machine; latency applies to this CPU only. EasyPortrait images are mostly 2.8 MP, so the 1-2 MP latency target is barely sampled.",
          "- Boundary F and best-shift are computed on full-image masks; best shift searches integer shifts within +-3 px.",
          "- The weights remain CelebAMask-HQ (non-commercial); this evaluation does not change that restriction.", ""]
    L += ["## Licence and attribution", "", rep["licence"], "", "```", rep["bibtex"], "```", ""]
    return "\n".join(L)


def write_report(rep: dict[str, Any], out_prefix: Path) -> tuple[Path, Path]:
    md, js = out_prefix.with_suffix(".md"), out_prefix.with_suffix(".json")
    md.write_text(render_markdown(rep))
    js.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return md, js
