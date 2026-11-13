"""E-2 evaluation of Phase 7 detection (diagnosis) and Phase 9 preview (gap closed) on FFHQ-Makeup.

Offline. Writes aggregates only to the report (no ids, hashes or paths). Private per-unit results live in the
runs directory (outside the repo, 0700/0600) under salted hashes, so a second run skips finished units.
Nothing here logs ids, pixels or paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import secrets
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.services.facial_features.skin_tone import classify_skin_tone
from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_preview.color import srgb_to_lab
from app.services.makeup_preview.masks import decode_mask_full
from app.services.makeup_preview.metrics import mean_delta_e_inside, outside_changed_count
from evaluation import dataset as ds
from evaluation import makeup_dataset as md
from evaluation.makeup_labels import (
    LABEL_MARGIN_FRACTION,
    SENSITIVITY_FRACTIONS,
    STAGES,
    classify_stage,
    label_zone,
)
from evaluation.makeup_signals import FaceContext, prepare, raw_signals, signals_from_context
from evaluation.makeup_stats import (
    bootstrap_ci,
    gap_closed,
    paired_auc,
    prf,
    rank_auc,
    threshold_at_fp,
    wilson,
)
from app.services.image_loading import load_rgb_image

logger = logging.getLogger("evaluation")

PRODUCTS = tuple(r.product_type for r in cv.TARGETS)
ROW = {r.product_type: r for r in cv.TARGETS}
LOWEST = -1e9
VISIBLE_DE = 2.0
MIN_BAND_N = 5
MIN_REGION_PX = 20
FP_TARGET = 0.05
AUC_RECALIBRATE = 0.9
AUC_VISION = 0.75
UNIT_VERSION = 1

LICENCE = (
    "FFHQ-Makeup (cyberagent) is licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). Evaluation only; no images, "
    "identities or per-image values are included here, and nothing derived from it is used for training."
)
CAVEAT = (
    "The makeup in FFHQ-Makeup is synthetic (generated on retouched FFHQ faces), and the images were retouched. Labels are "
    "pixel-residual labels (makeup minus bare, relative to the skin reference), not human annotation. Real reference "
    "photos are harder than this set, so recall and gap-closed figures are likely optimistic. EasyPortrait may contain "
    "real makeup, so its false-positive rate is an upper bound."
)
DECISION_RULE = (
    "Per product: AUC >= 0.9 -> recalibrate the thresholds; AUC < 0.75 -> a vision model; in between -> "
    "recalibrate plus review."
)


# --- private run store ---------------------------------------------------------------------------

class RunStore:
    def __init__(self, runs_dir: Path) -> None:
        self.root = ds.refuse_inside_repo(Path(runs_dir), "runs directory")
        for d in (self.root, self.root / "profiles", self.root / "units"):
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o700)
        salt_path = self.root / "salt"
        if not salt_path.is_file():
            ds.write_private(salt_path, secrets.token_hex(16).encode())
        self._salt = salt_path.read_bytes()
        self._manifest_path = self.root / "manifest.json"
        self._done: set[str] = set()
        if self._manifest_path.is_file():
            try:
                self._done = set(json.loads(self._manifest_path.read_text()).get("done", []))
            except (ValueError, OSError):
                self._done = set()

    def key(self, *parts: str) -> str:
        return hashlib.sha256(self._salt + b"|" + "|".join(parts).encode()).hexdigest()[:32]

    def _read(self, path: Path) -> Any | None:
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            return None
        return data if isinstance(data, dict) and data.get("v") == UNIT_VERSION else None

    def load_profile(self, key: str) -> dict[str, Any] | None:
        return self._read(self.root / "profiles" / f"{key}.json")

    def save_profile(self, key: str, data: dict[str, Any]) -> None:
        ds.write_private(self.root / "profiles" / f"{key}.json", json.dumps({"v": UNIT_VERSION, **data}).encode())

    def load_unit(self, key: str) -> dict[str, Any] | None:
        return self._read(self.root / "units" / f"{key}.json") if key in self._done else None

    def save_unit(self, key: str, data: dict[str, Any]) -> None:
        ds.write_private(self.root / "units" / f"{key}.json", json.dumps({"v": UNIT_VERSION, **data}).encode())
        self._done.add(key)
        ds.write_private(self._manifest_path, json.dumps({"done": sorted(self._done)}).encode())


class _Runner:
    """Lazy holder so a fully resumed run never loads the models."""

    def __init__(self, runner: Any | None) -> None:
        self._runner = runner
        self._own = runner is None

    def get(self) -> Any:
        if self._runner is None:
            from evaluation.makeup_pipeline import PairRunner

            self._runner = PairRunner()
        return self._runner

    def close(self) -> None:
        if self._own and self._runner is not None:
            self._runner.close()


# --- per-unit computation ------------------------------------------------------------------------

def _safe_profile(runner: Any, rgb: np.ndarray) -> Any | None:
    try:
        return runner.profile(rgb)
    except Exception:  # noqa: BLE001 - outcome, not a crash
        return None


def _summarize(rgb: np.ndarray, profile: Any | None) -> dict[str, Any]:
    sig = None
    depth = None
    if profile is not None:
        try:
            sig = raw_signals(rgb, profile)
            comps = profile.components if profile.status == "ok" else None
            if comps is not None and comps.zones.data is not None:
                glasses = any(w.code == "glasses_detected" for w in profile.quality.warnings)
                depth = classify_skin_tone(rgb, comps.zones.data.zones, glasses).depth.label
        except Exception:  # noqa: BLE001
            pass
    return {"ok": sig is not None, "signals": sig, "depth": depth}


def _region_lab(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.median(srgb_to_lab(img[mask].astype(np.float64) / 255.0), axis=0)


def _stage_gaps(bare: np.ndarray, truth: np.ndarray, res: Any, ctx: FaceContext) -> dict[str, float | None]:
    """Gap closed per stage on skin-relative Lab (region medians), using the bare face's skin reference."""
    if res.plan is None or res.render is None:
        return {}
    h, w = bare.shape[:2]
    by_index = {c.index: c for c in res.plan.components}
    stage_masks: dict[str, np.ndarray] = {}
    for rep in res.render.reports:
        comp = by_index.get(rep.index)
        if rep.status != "rendered" or comp is None:
            continue
        mask, _ = decode_mask_full(comp, (h, w))
        if mask is None:
            continue
        key = comp.stage_key or "none"
        m = mask > 0
        stage_masks[key] = (stage_masks[key] | m) if key in stage_masks else m
    preview = res.render.image
    skin = {name: _region_lab(img, ctx.skin_ref) for name, img in (("p", preview), ("t", truth), ("b", bare))}
    out: dict[str, float | None] = {}
    for stage, m in sorted(stage_masks.items()):
        if int(m.sum()) < MIN_REGION_PX:
            continue
        rel = {n: _region_lab(img, m) - skin[n] + skin["b"] for n, img in (("p", preview), ("t", truth), ("b", bare))}
        out[stage] = gap_closed(rel["p"], rel["t"], rel["b"])
    return out


def _pair_unit(store: RunStore, runner: Any, cache: Path, pid: str) -> dict[str, Any]:
    bare = md.load_image(cache, pid, "bare")
    bprof = _safe_profile(runner, bare)
    pkey = store.key("profile", pid, "bare")
    bsum = store.load_profile(pkey)
    if bsum is None:
        bsum = _summarize(bare, bprof)
        store.save_profile(pkey, bsum)
    ctx = prepare(bare, bprof) if bprof is not None else None
    ks: list[dict[str, Any]] = []
    for k in range(1, 6):
        truth = md.load_image(cache, pid, md.makeup_kind(k))
        skey = store.key("profile", pid, md.makeup_kind(k))
        msum = store.load_profile(skey)
        if msum is None:
            msum = _summarize(truth, _safe_profile(runner, truth))
            store.save_profile(skey, msum)
        res = runner.run(bare, truth)
        labels: dict[str, Any] = {}
        stages: dict[str, Any] = {}
        for row in cv.TARGETS:
            zmask = ctx.product_masks.get(row.product_type) if ctx is not None else None
            if ctx is None or zmask is None:
                labels[row.product_type] = None
            else:
                _, residual = label_zone(bare, truth, zmask, ctx.skin_ref, row, LABEL_MARGIN_FRACTION)
                labels[row.product_type] = {"residual": residual}
            st = classify_stage(True, res.blueprint, res.plan, res.render, row.category)
            stages[row.product_type] = list(st) if st else None
        gaps = _stage_gaps(bare, truth, res, ctx) if ctx is not None else {}
        reports = res.render.reports if res.render is not None else ()
        ks.append({
            "signals": msum["signals"],
            "labels": labels,
            "stages": stages,
            "gaps": {s: v for s, v in gaps.items()},
            "ms": res.ms,
            "status": res.skip_reason,
            "rendered": sum(r.status == "rendered" for r in reports),
            "skipped": sum(r.status == "skipped" for r in reports),
            "outside_changed": outside_changed_count(bare, res.render.image, res.render.alpha_union)
            if res.render is not None else None,
        })
    return {"depth": bsum["depth"], "bare": bsum["signals"], "k": ks}


def _cross_unit(runner: Any, cache: Path, a: str, b: str, k: int) -> dict[str, Any]:
    bare = md.load_image(cache, a, "bare")
    ref = md.load_image(cache, b, md.makeup_kind(k))
    res = runner.run(bare, ref)
    out: dict[str, Any] = {
        "ms": res.ms, "status": res.skip_reason, "plan_ok": res.plan is not None,
        "rendered": 0, "skipped": 0, "skip_reasons": {}, "visibility": [], "outside_changed": None,
    }
    if res.plan is None or res.render is None:
        return out
    h, w = bare.shape[:2]
    by_index = {c.index: c for c in res.plan.components}
    reasons: Counter[str] = Counter()
    for rep in res.render.reports:
        if rep.status != "rendered":
            out["skipped"] += 1
            reasons[rep.reason or "skipped"] += 1
            continue
        out["rendered"] += 1
        comp = by_index.get(rep.index)
        mask = decode_mask_full(comp, (h, w))[0] if comp is not None else None
        if mask is not None:
            out["visibility"].append(round(mean_delta_e_inside(bare, res.render.image, mask), 3))
    out["skip_reasons"] = dict(reasons)
    out["outside_changed"] = outside_changed_count(bare, res.render.image, res.render.alpha_union)
    return out


def _ep_unit(runner: Any, ep_cache: Path, image_id: str) -> dict[str, Any] | None:
    path = ds.image_path(ep_cache, image_id)
    if not path.exists():
        return None
    try:
        rgb = load_rgb_image(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None
    return _summarize(rgb, _safe_profile(runner, rgb))


# --- aggregation ---------------------------------------------------------------------------------

def _score(sig: dict[str, Any] | None, product: str) -> float | None:
    if sig is None:
        return None
    p = sig["products"][product]
    if not p["usable"] or p["value"] is None:
        return LOWEST
    return float(p["value"]) if p["gate_ok"] else 0.0


def _detected(sig: dict[str, Any] | None, product: str) -> bool:
    return bool(sig and sig["products"][product]["detected"])


def _ci_pair(ci: tuple[float, float] | None) -> list[float] | None:
    return None if ci is None else [ci[0], ci[1]]


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": _ci_pair(ci)}


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "ci95": None}
    a = np.asarray(values, dtype=np.float64)
    ci = bootstrap_ci(lambda i: float(a[i].mean()), (a.size,))
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "ci95": _ci_pair(ci)}


def _pctl(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def _product_block(
    p: str, pairs: list[dict[str, Any]], ep: list[dict[str, Any]], margin: float
) -> dict[str, Any]:
    thr = ROW[p].present_threshold
    bare_sc = [s for s in (_score(u["bare"], p) for u in pairs) if s is not None]
    bare_det = [_detected(u["bare"], p) for u in pairs if u["bare"] is not None]
    ep_det = [_detected(u["signals"], p) for u in ep if u["signals"] is not None]

    resid: list[float] = []
    score: list[float] = []
    det: list[bool] = []
    own: list[float] = []
    stage: list[Any] = []
    for u in pairs:
        own_sc = _score(u["bare"], p)
        for kd in u["k"]:
            lab = kd["labels"].get(p)
            if lab is None:
                continue
            sc = _score(kd["signals"], p)
            resid.append(lab["residual"])
            score.append(LOWEST if sc is None else sc)
            det.append(_detected(kd["signals"], p))
            own.append(float("nan") if own_sc is None else own_sc)
            stage.append(kd["stages"].get(p))
    r, s, d, o = (np.asarray(x, dtype=np.float64) for x in (resid, score, det, own))
    d = d.astype(bool)

    def prf_at(fraction: float) -> dict[str, Any]:
        pos = r >= fraction * thr
        tp, fp, fn = int((d & pos).sum()), int((d & ~pos).sum()), int((~d & pos).sum())
        out: dict[str, Any] = {"fraction": fraction, "n_labelled": int(r.size), "n_positive": int(pos.sum()),
                               "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
        out["recall_ci95"] = _ci_pair(wilson(tp, tp + fn))
        out["precision_ci95"] = _ci_pair(wilson(tp, tp + fp))

        def f1(i: np.ndarray) -> float | None:
            pp, dd = pos[i], d[i]
            t = int((dd & pp).sum())
            return prf(t, int((dd & ~pp).sum()), int((~dd & pp).sum()))["f1"]

        out["f1_ci95"] = _ci_pair(bootstrap_ci(f1, (r.size,)))
        return out

    pos = r >= margin * thr
    pos_sc = s[pos]
    neg_pool = np.concatenate([np.asarray(bare_sc, dtype=np.float64), s[~pos]])
    auc = rank_auc(pos_sc, neg_pool)
    auc_ci = bootstrap_ci(lambda i, j: rank_auc(pos_sc[i], neg_pool[j]), (pos_sc.size, neg_pool.size))
    own_pos = o[pos]
    keep = ~np.isnan(own_pos)
    paired = paired_auc(pos_sc[keep], own_pos[keep])
    paired_ci = bootstrap_ci(lambda i: paired_auc(pos_sc[keep][i], own_pos[keep][i]), (int(keep.sum()),))
    t5 = threshold_at_fp(neg_pool, FP_TARGET)
    recall5 = float((pos_sc >= t5).mean()) if pos_sc.size and not math.isnan(t5) else None

    def recall_at_fp(i: np.ndarray, j: np.ndarray) -> float | None:
        t = threshold_at_fp(neg_pool[j], FP_TARGET)
        return float((pos_sc[i] >= t).mean())

    recall5_ci = bootstrap_ci(recall_at_fp, (pos_sc.size, neg_pool.size))
    fp_real = float((neg_pool >= t5).mean()) if neg_pool.size and not math.isnan(t5) else None

    shares: dict[str, Any] = {}
    skip_reasons: Counter[str] = Counter()
    pos_stages = [st for st, is_pos in zip(stage, pos) if is_pos and st]
    for st in pos_stages:
        if st[1]:
            skip_reasons[f"{st[0]}:{st[1]}"] += 1
    if pos_stages:
        counts = Counter(st[0] for st in pos_stages)
        shares = {name: counts.get(name, 0) / len(pos_stages) for name in STAGES}

    fractions = sorted({*SENSITIVITY_FRACTIONS, margin})
    return {
        "product": p,
        "production_threshold": thr,
        "false_positive_bare": _rate(sum(bare_det), len(bare_det)),
        "false_positive_easyportrait_upper_bound": _rate(sum(ep_det), len(ep_det)),
        "production": prf_at(margin),
        "label_sensitivity": [prf_at(f) for f in fractions],
        "auc": {"pooled": auc, "pooled_ci95": _ci_pair(auc_ci), "same_pair": paired,
                "same_pair_ci95": _ci_pair(paired_ci), "n_positive": int(pos_sc.size), "n_negative": int(neg_pool.size)},
        "at_5pct_fp": {"threshold": None if math.isnan(t5) else t5, "recall": recall5,
                       "recall_ci95": _ci_pair(recall5_ci), "realised_fp": fp_real},
        "loss_stage": {"n_positive": len(pos_stages), "shares": shares, "detail": dict(sorted(skip_reasons.items()))},
        "verdict": _verdict(auc),
    }


def _verdict(auc: float) -> str:
    if math.isnan(auc):
        return "insufficient_data"
    if auc >= AUC_RECALIBRATE:
        return "recalibrate_thresholds"
    if auc < AUC_VISION:
        return "vision_model"
    return "recalibrate_plus_review"


def _bands(pairs: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for u in pairs:
        groups.setdefault(u["depth"] or "unknown", []).append(u)
    out: dict[str, Any] = {}
    for label, us in sorted(groups.items()):
        if len(us) < MIN_BAND_N:
            out[label] = {"n": len(us), "suppressed": True}
            continue
        per: dict[str, Any] = {}
        for p in PRODUCTS:
            fp = [_detected(u["bare"], p) for u in us if u["bare"] is not None]
            tp = fn = 0
            for u in us:
                for kd in u["k"]:
                    lab = kd["labels"].get(p)
                    if lab is None or lab["residual"] < margin * ROW[p].present_threshold:
                        continue
                    if _detected(kd["signals"], p):
                        tp += 1
                    else:
                        fn += 1
            per[p] = {
                "false_positive_bare": _rate(sum(fp), len(fp)),
                "recall": {"n": tp + fn, "suppressed": True} if tp + fn < MIN_BAND_N else _rate(tp, tp + fn),
            }
        gaps = [v for u in us for kd in u["k"] for v in kd["gaps"].values() if v is not None]
        out[label] = {"n": len(us), "products": per,
                      "gap_closed": {"n": len(gaps), "suppressed": True} if len(gaps) < MIN_BAND_N else _mean_ci(gaps)}
    return out


def aggregate(
    pairs: list[dict[str, Any]], cross: list[dict[str, Any]], ep: list[dict[str, Any]], margin: float
) -> dict[str, Any]:
    e1 = {p: _product_block(p, pairs, ep, margin) for p in PRODUCTS}

    gap_by_stage: dict[str, list[float]] = {}
    excluded: Counter[str] = Counter()
    outside_total = outside_pairs = outside_n = 0
    pair_ms: list[float] = []
    statuses: Counter[str] = Counter()
    for u in pairs:
        for kd in u["k"]:
            pair_ms.append(kd["ms"])
            if kd["status"]:
                statuses[kd["status"]] += 1
            for st, v in kd["gaps"].items():
                if v is None:
                    excluded[st] += 1
                else:
                    gap_by_stage.setdefault(st, []).append(v)
            if kd["outside_changed"] is not None:
                outside_n += 1
                outside_total += kd["outside_changed"]
                outside_pairs += kd["outside_changed"] > 0
    stages = sorted(set(gap_by_stage) | set(excluded))
    e2 = {
        "gap_closed_per_stage": {st: {**_mean_ci(gap_by_stage.get(st, [])), "excluded_baseline_lt_1": excluded.get(st, 0)}
                                 for st in stages},
        "outside_changed": {"pairs_rendered": outside_n, "pairs_with_outside_change": outside_pairs,
                            "total_pixels": outside_total},
        "runtime_ms": _pctl(pair_ms),
        "pipeline_outcomes": dict(sorted(statuses.items())),
    }

    vis = [v for c in cross for v in c["visibility"]]
    rendered = sum(c["rendered"] for c in cross)
    skipped = sum(c["skipped"] for c in cross)
    reasons: Counter[str] = Counter()
    for c in cross:
        reasons.update(c["skip_reasons"])
    e3 = {
        "n_cross": len(cross),
        "plan_ok": _rate(sum(c["plan_ok"] for c in cross), len(cross)),
        "components_rendered": rendered,
        "components_skipped": skipped,
        "render_rate": _rate(rendered, rendered + skipped),
        "skip_reasons": dict(sorted(reasons.items())),
        "visibility_de2000_inside_mask": _mean_ci(vis),
        "visible_fraction": {**_rate(sum(v >= VISIBLE_DE for v in vis), len(vis)), "visible_de": VISIBLE_DE},
        "outside_changed_total": sum(c["outside_changed"] or 0 for c in cross),
        "runtime_ms": _pctl([c["ms"] for c in cross]),
    }
    return {
        "e1_detection": e1,
        "e2_gap_closed": e2,
        "e3_cross_identity": e3,
        "fairness_depth_bands": {"min_band_n": MIN_BAND_N, "bands": _bands(pairs, margin)},
        "counts": {
            "pair_identities": len(pairs),
            "pair_identities_bare_profile_ok": sum(u["bare"] is not None for u in pairs),
            "pair_images": sum(len(u["k"]) for u in pairs),
            "cross_pairs": len(cross),
            "easyportrait_images_profile_ok": sum(u["signals"] is not None for u in ep),
        },
    }


def build_report(
    agg: dict[str, Any], *, seed: int | None, margin: float, today: str, easyportrait_images: int
) -> dict[str, Any]:
    inputs = {
        p: {
            "auc": b["auc"]["pooled"], "auc_ci95": b["auc"]["pooled_ci95"],
            "recall_at_5pct_fp": b["at_5pct_fp"]["recall"], "recall_at_5pct_fp_ci95": b["at_5pct_fp"]["recall_ci95"],
            "loss_stage_shares": b["loss_stage"]["shares"], "verdict": b["verdict"],
        }
        for p, b in agg["e1_detection"].items()
    }
    return {
        "title": "Makeup evaluation E-2: Phase 7 detection diagnosis and Phase 9 gap closed",
        "date": today,
        "provenance": {
            "dataset": "FFHQ-Makeup (cyberagent), committed sample of opaque identity ids",
            "seed": seed,
            "licence": LICENCE,
            "label_margin_fraction": margin,
            "easyportrait_images_requested": easyportrait_images,
            "counts": agg["counts"],
        },
        "caveat": CAVEAT,
        **{k: v for k, v in agg.items() if k != "counts"},
        "recommendation_inputs": {"decision_rule": DECISION_RULE, "per_product": inputs},
    }


# --- report writing ------------------------------------------------------------------------------

def _clean(x: Any, nd: int = 4) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(ci: Any) -> str:
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(rep: dict[str, Any]) -> str:
    prov = rep["provenance"]
    out = [f"# {rep['title']}", "", f"Date: {rep['date']}", "", "## Provenance", "",
           f"- Dataset: {prov['dataset']}; seed {prov['seed']}; label margin fraction {prov['label_margin_fraction']}",
           f"- Counts: {json.dumps(prov['counts'])}", f"- Licence: {prov['licence']}", "", "## Caveat", "", rep["caveat"], ""]
    e1 = rep["e1_detection"]
    out += ["## E1 Detection", ""]
    out.append(_table(
        ["product", "FP bare", "FP EasyPortrait (upper bound)", "precision", "recall", "F1", "AUC pooled", "AUC same-pair",
         "threshold@5%FP", "recall@5%FP"],
        [[p, f"{_f(b['false_positive_bare']['rate'])} {_ci(b['false_positive_bare']['ci95'])}",
          f"{_f(b['false_positive_easyportrait_upper_bound']['rate'])} {_ci(b['false_positive_easyportrait_upper_bound']['ci95'])}",
          _f(b["production"]["precision"]), f"{_f(b['production']['recall'])} {_ci(b['production']['recall_ci95'])}",
          f"{_f(b['production']['f1'])} {_ci(b['production']['f1_ci95'])}",
          f"{_f(b['auc']['pooled'])} {_ci(b['auc']['pooled_ci95'])}", f"{_f(b['auc']['same_pair'])} {_ci(b['auc']['same_pair_ci95'])}",
          _f(b["at_5pct_fp"]["threshold"]), f"{_f(b['at_5pct_fp']['recall'])} {_ci(b['at_5pct_fp']['recall_ci95'])}"]
         for p, b in e1.items()]))
    out += ["", "Label sensitivity (precision / recall / F1 by margin fraction):", ""]
    out.append(_table(
        ["product", "fraction", "positives", "precision", "recall", "F1"],
        [[p, _f(s["fraction"], 2), str(s["n_positive"]), _f(s["precision"]), _f(s["recall"]), _f(s["f1"])]
         for p, b in e1.items() for s in b["label_sensitivity"]]))
    out += ["", "Loss stage shares among labelled positives:", ""]
    out.append(_table(
        ["product", "n"] + list(STAGES),
        [[p, str(b["loss_stage"]["n_positive"])] + [_f(b["loss_stage"]["shares"].get(s)) for s in STAGES]
         for p, b in e1.items()]))
    e2 = rep["e2_gap_closed"]
    out += ["", "## E2 Gap closed", ""]
    out.append(_table(
        ["stage", "n", "mean", "CI95", "median", "excluded (baseline < 1 dE00)"],
        [[s, str(v["n"]), _f(v["mean"]), _ci(v["ci95"]), _f(v["median"]), str(v["excluded_baseline_lt_1"])]
         for s, v in e2["gap_closed_per_stage"].items()]))
    oc = e2["outside_changed"]
    out += ["", f"- outside_changed_count: total {oc['total_pixels']} px, {oc['pairs_with_outside_change']} of {oc['pairs_rendered']} pairs",
            f"- Runtime ms: p50 {_f(e2['runtime_ms']['p50'], 1)}, p95 {_f(e2['runtime_ms']['p95'], 1)}",
            f"- Pipeline outcomes: {json.dumps(e2['pipeline_outcomes'])}", ""]
    e3 = rep["e3_cross_identity"]
    out += ["## E3 Cross-identity", "",
            f"- Pairs: {e3['n_cross']}; plan ok {_f(e3['plan_ok']['rate'])} {_ci(e3['plan_ok']['ci95'])}",
            f"- Components rendered {e3['components_rendered']}, skipped {e3['components_skipped']}; "
            f"render rate {_f(e3['render_rate']['rate'])} {_ci(e3['render_rate']['ci95'])}; skip reasons {json.dumps(e3['skip_reasons'])}",
            f"- Visibility (mean dE2000 inside mask): {_f(e3['visibility_de2000_inside_mask']['mean'])} "
            f"{_ci(e3['visibility_de2000_inside_mask']['ci95'])}; fraction >= {e3['visible_fraction']['visible_de']} dE: "
            f"{_f(e3['visible_fraction']['rate'])} {_ci(e3['visible_fraction']['ci95'])}",
            f"- outside_changed total: {e3['outside_changed_total']} px", ""]
    out += ["## Fairness by skin depth band", "", f"Bands with n < {rep['fairness_depth_bands']['min_band_n']} are suppressed.", ""]
    rows = []
    for label, b in rep["fairness_depth_bands"]["bands"].items():
        if b.get("suppressed"):
            rows.append([label, str(b["n"]), "suppressed", "", ""])
            continue
        rec = "; ".join(f"{p}: {'suppressed' if v['recall'].get('suppressed') else _f(v['recall']['rate'])}"
                        for p, v in b["products"].items())
        g = b["gap_closed"]
        rows.append([label, str(b["n"]), rec, "suppressed" if g.get("suppressed") else _f(g["mean"]),
                     "" if g.get("suppressed") else _ci(g["ci95"])])
    out.append(_table(["band", "n", "recall at production", "gap closed mean", "CI95"], rows))
    ri = rep["recommendation_inputs"]
    out += ["", "## Recommendation inputs", "", f"Decision rule: {ri['decision_rule']}", ""]
    out.append(_table(
        ["product", "AUC", "recall@5%FP", "verdict"] + [f"lost: {s}" for s in STAGES],
        [[p, f"{_f(v['auc'])} {_ci(v['auc_ci95'])}", f"{_f(v['recall_at_5pct_fp'])} {_ci(v['recall_at_5pct_fp_ci95'])}", v["verdict"]]
         + [_f(v["loss_stage_shares"].get(s)) for s in STAGES] for p, v in ri["per_product"].items()]))
    return "\n".join(out) + "\n"


# --- audit sheet ---------------------------------------------------------------------------------

def _audit(store: RunStore, runner: Any, cache: Path, samples: md.MakeupSamples, margin: float) -> Path:
    tile = 256
    rows: list[list[Image.Image]] = []
    for idx, pid in enumerate(samples.sheet):
        bare = md.load_image(cache, pid, "bare")
        truth = md.load_image(cache, pid, md.makeup_kind((idx % 5) + 1))
        prof = _safe_profile(runner, bare)
        ctx = prepare(bare, prof) if prof is not None else None
        overlay = bare.astype(np.float32)
        if ctx is not None:
            for row in cv.TARGETS:
                zmask = ctx.product_masks.get(row.product_type)
                if zmask is None:
                    continue
                positive, _ = label_zone(bare, truth, zmask, ctx.skin_ref, row, margin)
                colour = np.array((0, 255, 0) if positive else (255, 0, 0), dtype=np.float32)
                overlay[zmask] = 0.6 * overlay[zmask] + 0.4 * colour
        rows.append([Image.fromarray(overlay.astype(np.uint8)).resize((tile, tile)),
                     Image.fromarray(truth).resize((tile, tile))])
    header = 18
    sheet = Image.new("RGB", (tile * 2, header + tile * max(1, len(rows))), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((6, 3), "bare + labels (green=positive, red=negative)", fill=(0, 0, 0))
    d.text((tile + 6, 3), "makeup", fill=(0, 0, 0))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * tile, header + r * tile))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    path = store.root / "label_audit.png"
    ds.write_private(path, buf.getvalue())
    return path


# --- orchestration -------------------------------------------------------------------------------

def parse_seed(text: str) -> int | None:
    m = re.search(r"seed=(\d+)", text)
    return int(m.group(1)) if m else None


def run(
    cache: Path,
    samples: md.MakeupSamples,
    runs_dir: Path,
    out_dir: Path,
    *,
    ep_cache: Path | None = None,
    ep_ids: list[str] | None = None,
    seed: int | None = None,
    limit: int | None = None,
    label_margin: float = LABEL_MARGIN_FRACTION,
    audit_sheet: bool = False,
    runner: Any | None = None,
    today: str | None = None,
) -> dict[str, Any]:
    store = RunStore(runs_dir)
    lazy = _Runner(runner)
    cut: Callable[[list[Any]], list[Any]] = (lambda xs: xs[:limit]) if limit else (lambda xs: xs)
    pair_units: list[dict[str, Any]] = []
    cross_units: list[dict[str, Any]] = []
    ep_units: list[dict[str, Any]] = []
    skipped = computed = 0
    try:
        for pid in cut(samples.pairs):
            key = store.key("pair", pid)
            unit = store.load_unit(key)
            if unit is None:
                unit = _pair_unit(store, lazy.get(), cache, pid)
                store.save_unit(key, unit)
                computed += 1
            else:
                skipped += 1
            pair_units.append(unit)
        for a, b, k in cut(samples.cross):
            key = store.key("cross", a, b, str(k))
            unit = store.load_unit(key)
            if unit is None:
                unit = _cross_unit(lazy.get(), cache, a, b, k)
                store.save_unit(key, unit)
                computed += 1
            else:
                skipped += 1
            cross_units.append(unit)
        if ep_cache is not None:
            for iid in cut(list(ep_ids or [])):
                key = store.key("ep", iid)
                unit = store.load_unit(key)
                if unit is None:
                    fresh = _ep_unit(lazy.get(), ep_cache, iid)
                    if fresh is None:
                        continue
                    unit = fresh
                    store.save_unit(key, unit)
                    computed += 1
                else:
                    skipped += 1
                ep_units.append(unit)
        if audit_sheet:
            _audit(store, lazy.get(), cache, samples, label_margin)
    finally:
        lazy.close()
    logger.info("units computed %d, skipped (already finished) %d", computed, skipped)

    agg = aggregate(pair_units, cross_units, ep_units, label_margin)
    rep = _clean(build_report(agg, seed=seed, margin=label_margin, today=today or date.today().isoformat(),
                              easyportrait_images=len(ep_units)))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"makeup_eval_{rep['date']}"
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    stem.with_suffix(".md").write_text(render_markdown(rep))
    rep["_run"] = {"computed": computed, "skipped": skipped}
    return rep
